"""Restarted right-preconditioned GMRES with device-side Arnoldi early exit.

The legacy Warp solver builds every column of a restart before checking the
residual. This implementation applies Givens rotations after each column, stops
when the projected residual is small, and *then recomputes the true residual*.
No acceptance is based only on the projected estimate. All buffers are allocated
before capture. CUDA loops are conditional graph nodes, never Python polling.

Two-pass classical Gram--Schmidt uses Warp's preallocated tiled reductions.
Independent grid entries run in parallel; only the restart-sized QR/backsolve
is serial. That small recurrence is not a global dense matrix factorization.
This is a single-system solver. It does not claim multi-robot sparse batching.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from warp.optim.linear import LinearOperator

    from .host_types import Shape

from functools import lru_cache
from types import SimpleNamespace

import warp as wp
from warp._src.optim.linear import LinearSolverState, TiledDot, _scalar_dof_count
from warp._src.types import type_length, type_scalar_type

from ._solver_limits import ADAPTIVE_GMRES_MAX_RESTART


@lru_cache(maxsize=8)
def _kernels(dtype: "type") -> "SimpleNamespace":
    """Return the cached adaptive Arnoldi, Givens rotation and backsolve namespace.

    Types and kernel identities depend on the requested physics/dimension and
    precision. Warp DSL annotations must be evaluated in this factory's
    scope; postponed annotations and relocation change kernel construction."""
    real = type_scalar_type(dtype)
    tag = f"{real.__name__}_{type_length(dtype)}"
    module = f"envs.soft_walker.engine.warp_native.adaptive_gmres_{tag}_v1"

    @wp.kernel(module=module, enable_backward=False)
    def initialize(
        bnorm: wp.array(dtype=real),
        rnorm: wp.array(dtype=real),
        rtol: real,
        atol: real,
        maxiter: int,
        state: wp.array(dtype=int),
        tolerance: wp.array(dtype=real),
        condition: wp.array(dtype=int),
    ):
        for i in range(state.shape[0]):
            state[i] = 0
        tolerance[0] = wp.max(atol * atol, rtol * rtol * bnorm[0])
        valid = wp.isfinite(bnorm[0]) and wp.isfinite(rnorm[0])
        if not valid:
            state[2] = 2
        condition[0] = int(valid and rnorm[0] > tolerance[0] and maxiter > 0)

    @wp.kernel(module=module, enable_backward=False)
    def begin_cycle(
        rnorm: wp.array(dtype=real),
        state: wp.array(dtype=int),
        rhs: wp.array(dtype=real),
        inner: wp.array(dtype=int),
    ):
        state[1] = 0  # current Arnoldi column / number of accepted columns
        state[3] = state[3] + 1  # restart cycles
        for i in range(rhs.shape[0]):
            rhs[i] = real(0)
        rhs[0] = wp.sqrt(rnorm[0])
        inner[0] = 1

    @wp.kernel(module=module, enable_backward=False)
    def seed_basis(
        residual: wp.array(dtype=dtype),
        rhs: wp.array(dtype=real),
        basis: wp.array2d(dtype=dtype),
        current: wp.array(dtype=dtype),
    ):
        i = wp.tid()
        value = residual[i] / rhs[0]
        basis[0, i] = value
        current[i] = value

    @wp.kernel(module=module, enable_backward=False)
    def orthogonalize(
        projections: wp.array2d(dtype=real),
        basis: wp.array2d(dtype=dtype),
        state: wp.array(dtype=int),
        second_pass: int,
        vector: wp.array(dtype=dtype),
        hessenberg: wp.array2d(dtype=real),
    ):
        i = wp.tid()
        column = state[1]
        if i < vector.shape[0]:
            value = vector[i]
            for k in range(column + 1):
                value = value - projections[k, 0] * basis[k, i]
            vector[i] = value
        if i <= column:
            coefficient = projections[i, 0]
            if second_pass != 0:
                coefficient = coefficient + hessenberg[i, column]
            hessenberg[i, column] = coefficient

    @wp.kernel(module=module, enable_backward=False)
    def rotate(
        rnorm: wp.array(dtype=real),
        tolerance: wp.array(dtype=real),
        maxiter: int,
        state: wp.array(dtype=int),
        hessenberg: wp.array2d(dtype=real),
        cosine: wp.array(dtype=real),
        sine: wp.array(dtype=real),
        rhs: wp.array(dtype=real),
        inner: wp.array(dtype=int),
    ):
        column = state[1]
        lower = wp.sqrt(rnorm[0])
        hessenberg[column + 1, column] = lower
        for k in range(column):
            a = hessenberg[k, column]
            b = hessenberg[k + 1, column]
            hessenberg[k, column] = cosine[k] * a + sine[k] * b
            hessenberg[k + 1, column] = -sine[k] * a + cosine[k] * b
        a = hessenberg[column, column]
        b = hessenberg[column + 1, column]
        scale = wp.max(wp.abs(a), wp.abs(b))
        denominator = real(0)
        if scale > real(0):
            denominator = scale * wp.sqrt((a / scale) ** real(2) + (b / scale) ** real(2))
        state[0] = state[0] + 1
        if not wp.isfinite(denominator) or denominator == real(0):
            state[2] = 1  # singular/nonfinite column: no fabricated regularization
            inner[0] = 0
        else:
            c = a / denominator
            s = b / denominator
            cosine[column] = c
            sine[column] = s
            hessenberg[column, column] = denominator
            hessenberg[column + 1, column] = real(0)
            g = rhs[column]
            rhs[column] = c * g
            rhs[column + 1] = -s * g
            state[1] = column + 1
            estimate = rhs[column + 1] * rhs[column + 1]
            inner[0] = int(
                estimate > tolerance[0]
                and lower > real(0)
                and column + 1 < hessenberg.shape[1]
                and state[0] < maxiter
            )

    @wp.kernel(module=module, enable_backward=False)
    def next_basis(
        vector: wp.array(dtype=dtype),
        rnorm: wp.array(dtype=real),
        state: wp.array(dtype=int),
        inner: wp.array(dtype=int),
        basis: wp.array2d(dtype=dtype),
        current: wp.array(dtype=dtype),
    ):
        i = wp.tid()
        if inner[0] != 0:
            value = vector[i] / wp.sqrt(rnorm[0])
            basis[state[1], i] = value
            current[i] = value

    @wp.kernel(module=module, enable_backward=False)
    def backsolve(
        hessenberg: wp.array2d(dtype=real),
        rhs: wp.array(dtype=real),
        state: wp.array(dtype=int),
        coefficients: wp.array(dtype=real),
    ):
        count = state[1]
        for ii in range(count):
            row = count - 1 - ii
            value = rhs[row]
            for column in range(row + 1, count):
                value = value - hessenberg[row, column] * coefficients[column]
            coefficients[row] = value / hessenberg[row, row]

    @wp.kernel(module=module, enable_backward=False)
    def combine(
        basis: wp.array2d(dtype=dtype),
        coefficients: wp.array(dtype=real),
        state: wp.array(dtype=int),
        out: wp.array(dtype=dtype),
    ):
        i = wp.tid()
        value = dtype(0.0)
        for column in range(state[1]):
            value = value + coefficients[column] * basis[column, i]
        out[i] = value

    @wp.kernel(module=module, enable_backward=False)
    def add(update: wp.array(dtype=dtype), x: wp.array(dtype=dtype)):
        i = wp.tid()
        x[i] = x[i] + update[i]

    @wp.kernel(module=module, enable_backward=False)
    def assess(
        rnorm: wp.array(dtype=real),
        tolerance: wp.array(dtype=real),
        maxiter: int,
        max_cycles: int,
        state: wp.array(dtype=int),
        condition: wp.array(dtype=int),
    ):
        valid = wp.isfinite(rnorm[0])
        if not valid:
            state[2] = 2
        # Always use b-Ax, not the Hessenberg residual estimate, to accept.
        condition[0] = int(
            valid
            and rnorm[0] > tolerance[0]
            and state[0] < maxiter
            and state[3] < max_cycles
            and state[2] == 0
        )

    return SimpleNamespace(**locals())


class AdaptiveGMRES(LinearSolverState):
    """Reusable GMRES(k), preserving the equation, tolerance, and total budget."""

    def __init__(
        self,
        A: "LinearOperator",
        b: "wp.array",
        x: "wp.array",
        *,
        restart: "int" = 20,
        tol: "float" = 1e-10,
        atol: "float" = 0.0,
        maxiter: "int" = 240,
        M: "LinearOperator | None" = None,
    ) -> "None":
        """Bind a right-preconditioned linear system and allocate fixed solver capacities."""
        if A.batch_offsets is not None:
            raise ValueError("AdaptiveGMRES currently supports one sparse system per solver")
        if not 1 <= restart <= ADAPTIVE_GMRES_MAX_RESTART or maxiter < 1 or tol < 0 or atol < 0:
            raise ValueError("Invalid GMRES capacity or tolerance")
        self.restart = min(int(restart), int(maxiter), _scalar_dof_count(b))
        super().__init__(
            A,
            b,
            x,
            tol=tol,
            atol=atol,
            maxiter=maxiter,
            M=M,
            check_every=0,
            use_cuda_graph=A.device.is_cuda,
        )

    def _allocate(self) -> "None":
        """Reserve Krylov vectors, reductions and device loop controls before capture."""
        if self._device.is_capturing:
            raise RuntimeError("Allocate AdaptiveGMRES before graph capture")
        n, m = self._b.shape[0], self.restart
        self.max_cycles = (self._maxiter + m - 1) // m
        self.k = _kernels(self._b.dtype)
        self.r = wp.zeros_like(self._b)
        self.w = wp.zeros_like(self._b)
        self.current = wp.zeros_like(self._b)
        self.temporary = wp.zeros_like(self._b)
        self.basis = wp.zeros((m, n), dtype=self._b.dtype, device=self._device)
        self.repeated_w = wp.array(
            ptr=self.w.ptr,
            shape=(m, n),
            strides=(0, self.w.strides[0]),
            dtype=self.w.dtype,
            device=self._device,
        )
        real = self._scalar_type
        self.hessenberg = wp.zeros((m + 1, m), dtype=real, device=self._device)
        for name, length in [
            ("cosine", m),
            ("sine", m),
            ("g", m + 1),
            ("coefficients", m),
            ("tolerance", 1),
        ]:
            setattr(self, name, wp.zeros(length, dtype=real, device=self._device))
        self.state = wp.zeros(4, dtype=int, device=self._device)
        self.condition = wp.zeros(1, dtype=int, device=self._device)
        self.inner = wp.zeros(1, dtype=int, device=self._device)
        self.dot = TiledDot(
            _scalar_dof_count(self._b), real, device=self._device, max_column_count=max(2, m)
        )

    def _launch(self, kernel: "wp.Kernel", count: "Shape", *arguments: "object") -> "None":
        """Launch a solver-control kernel on this linear system's device."""
        wp.launch(kernel, dim=count, inputs=list(arguments), device=self._device, record_tape=False)

    def _run(
        self, A: "LinearOperator", b: "wp.array", x: "wp.array", M: "LinearOperator | None"
    ) -> "tuple[wp.array, wp.array, wp.array]":
        """Solve Ax = b in place using retained Krylov workspace and device stopping rules."""
        if self._device.is_cuda and not self._device.is_capturing:
            raise RuntimeError("Capture the complete CUDA program before executing AdaptiveGMRES")
        n, m = b.shape[0], self.restart
        k = self.k
        self.dot.compute(b, b, col_offset=1)
        if getattr(self, "zero_initial_guess", False):
            # Explicitly enforce the contract, not an unverified claim by caller.
            x.zero_()
            wp.copy(self.r, b)
        else:
            A.matvec(x, b, self.r, alpha=-1.0, beta=1.0)
        self.dot.compute(self.r, self.r)
        self._launch(
            k.initialize,
            1,
            self.dot.col(1),
            self.dot.col(0),
            self._scalar_type(self._tol),
            self._scalar_type(self._atol),
            self._maxiter,
            self.state,
            self.tolerance,
            self.condition,
        )

        def arnoldi() -> "None":
            """Build and assess one Arnoldi column using device convergence controls."""
            if M is None:
                wp.copy(self.temporary, self.current)
            else:
                M.matvec(self.current, self.temporary, self.temporary, alpha=1.0, beta=0.0)
            A.matvec(self.temporary, self.w, self.w, alpha=1.0, beta=0.0)
            # Fixed-capacity projections allow a dynamic device-side column.
            # Entries beyond that column are ignored, even if a previous restart
            # left data there. Both passes use the same preallocated reduction.
            for second_pass in (0, 1):
                self.dot.compute(self.repeated_w, self.basis)
                self._launch(
                    k.orthogonalize,
                    max(n, m),
                    self.dot.cols(m),
                    self.basis,
                    self.state,
                    second_pass,
                    self.w,
                    self.hessenberg,
                )
            self.dot.compute(self.w, self.w)
            self._launch(
                k.rotate,
                1,
                self.dot.col(0),
                self.tolerance,
                self._maxiter,
                self.state,
                self.hessenberg,
                self.cosine,
                self.sine,
                self.g,
                self.inner,
            )
            self._launch(
                k.next_basis,
                n,
                self.w,
                self.dot.col(0),
                self.state,
                self.inner,
                self.basis,
                self.current,
            )

        def restart_cycle() -> "None":
            """Build a restart basis, update x and recompute the true residual."""
            self._launch(k.begin_cycle, 1, self.dot.col(0), self.state, self.g, self.inner)
            self._launch(k.seed_basis, n, self.r, self.g, self.basis, self.current)
            wp.capture_while(self.inner, arnoldi)
            self._launch(k.backsolve, 1, self.hessenberg, self.g, self.state, self.coefficients)
            self._launch(k.combine, n, self.basis, self.coefficients, self.state, self.w)
            if M is None:
                wp.copy(self.temporary, self.w)
            else:
                M.matvec(self.w, self.temporary, self.temporary, alpha=1.0, beta=0.0)
            self._launch(k.add, n, self.temporary, x)
            A.matvec(x, b, self.r, alpha=-1.0, beta=1.0)
            self.dot.compute(self.r, self.r)
            self._launch(
                k.assess,
                1,
                self.dot.col(0),
                self.tolerance,
                self._maxiter,
                self.max_cycles,
                self.state,
                self.condition,
            )

        wp.capture_while(self.condition, restart_cycle)
        return self.state, self.dot.col(0), self.tolerance
