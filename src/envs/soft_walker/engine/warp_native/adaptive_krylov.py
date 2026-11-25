# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Warp 1.16 GMRES with device-side adaptive Arnoldi.

The adaptive path supports unbatched CUDA, check_every=0, use_cuda_graph=True,
and right/no preconditioning. Other modes use CapturableGMRES unchanged. All
workspace and loop controls are allocated before capture; convergence decisions
stay on the device, including inside nested conditional graphs.

The Arnoldi, solution update and restart control derive from Warp 1.16's
``warp._src.optim.linear.GMRES._run`` (Apache-2.0; see third_party/WARP_LICENSE.md).
Single-pass classical Gram-Schmidt and upstream least-squares arithmetic are
preserved; this module does not add reorthogonalization. Incremental Givens
rotations estimate when to stop generating columns. The explicit true residual
b-A*x remains authoritative after each solution update. If estimated convergence
fails that check, all later cycles of the same solve use full restart length.
The original maximum restart-cycle budget and solver tolerances are retained.

``adaptive_counters`` is a preallocated int32 device array accumulating
[Arnoldi columns, entered cycles, early estimated exits, rejected estimates].
Call ``reset_adaptive_counters()`` before a measurement and read only after
completion, outside the timed/certified graph. Solves never read counters on the
host. The inherited iteration return counts restart slots, not executed columns;
use adaptive_counters[0] to measure the latter. Fallback paths do not update these
adaptive counters.

Private Warp kernels are used; review this module when upgrading Warp.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from warp.optim.linear import LinearOperator

from typing import Any

import warp as wp
from warp._src.optim import linear as _linear

from .krylov import CapturableGMRES


@wp.kernel(enable_backward=False)
def _begin_cycle(
    beta: wp.array(dtype=Any),
    remaining: wp.array(dtype=Any),
    condition: wp.array(dtype=int),
    estimated: wp.array(dtype=int),
    counters: wp.array(dtype=int),
):
    remaining[0] = beta[0]
    condition[0] = 1
    estimated[0] = 0
    counters[1] += 1


@wp.kernel(enable_backward=False)
def _update_qr(
    j: int,
    restart: int,
    pivot_tolerance: float,
    H: wp.array3d(dtype=Any),
    column: wp.array(dtype=Any),
    cosine: wp.array(dtype=Any),
    sine: wp.array(dtype=Any),
    remaining: wp.array(dtype=Any),
    atol_sq: wp.array(dtype=Any),
    force_full: wp.array(dtype=int),
    condition: wp.array(dtype=int),
    estimated: wp.array(dtype=int),
    counters: wp.array(dtype=int),
):
    # Keep H unchanged: upstream LS will apply its own Givens rotations.
    counters[0] += 1
    for i in range(j + 2):
        column[i] = H[0, i, j]
    for i in range(j):
        a = column[i]
        b = column[i + 1]
        column[i] = cosine[i] * a + sine[i] * b
        column[i + 1] = cosine[i] * b - sine[i] * a
    a = column[j]
    b = column[j + 1]
    norm_sq = a * a + b * b
    if not wp.isfinite(norm_sq):
        # No convergence claim. The unchanged LS/true residual propagate failure.
        condition[0] = 0
    elif norm_sq < H.dtype(pivot_tolerance):
        # Same rank-breakdown threshold as the upstream least-squares solve.
        condition[0] = 0
    else:
        norm = wp.sqrt(norm_sq)
        cosine[j] = a / norm
        sine[j] = b / norm
        residual = -sine[j] * remaining[0]
        remaining[0] = residual
        converged = wp.isfinite(residual) and residual * residual <= atol_sq[0]
        if converged and force_full[0] == 0:
            estimated[0] = 1
            condition[0] = 0
            if j + 1 < restart:
                counters[2] += 1


@wp.kernel(enable_backward=False)
def _check_estimate(
    residual_sq: wp.array(dtype=Any),
    atol_sq: wp.array(dtype=Any),
    estimated: wp.array(dtype=int),
    force_full: wp.array(dtype=int),
    counters: wp.array(dtype=int),
):
    if estimated[0] != 0:
        accepted = wp.isfinite(residual_sq[0]) and residual_sq[0] <= atol_sq[0]
        if not accepted:
            force_full[0] = 1
            counters[3] += 1


class AdaptiveCapturableGMRES(CapturableGMRES):
    """Reusable GMRES with adaptive CUDA Arnoldi and compatible operands."""

    def _allocate(self) -> "None":
        """Reserve Krylov vectors, reductions and device loop controls before capture."""
        super()._allocate()
        dtype, device = self._scalar_type, self._device
        self._qr_column = wp.empty(self._restart + 1, dtype=dtype, device=device)
        self._qr_cosine = wp.empty(self._restart, dtype=dtype, device=device)
        self._qr_sine = wp.empty(self._restart, dtype=dtype, device=device)
        self._qr_remaining = wp.empty(1, dtype=dtype, device=device)
        self._inner_condition = wp.zeros(1, dtype=int, device=device)
        self._estimated_convergence = wp.zeros(1, dtype=int, device=device)
        self._force_full = wp.zeros(1, dtype=int, device=device)
        self.adaptive_counters = wp.zeros(4, dtype=int, device=device)

    def reset_adaptive_counters(self) -> "None":
        """Clear resident early-exit counters before reusing the solver."""
        self.adaptive_counters.zero_()

    def _run(
        self, A: "LinearOperator", b: "wp.array", x: "wp.array", M: "LinearOperator | None"
    ) -> "tuple[wp.array, wp.array, wp.array]":
        """Solve Ax = b in place using retained Krylov workspace and device stopping rules."""
        if (
            not self._device.is_cuda
            or not self._use_cuda_graph
            or self._check_every > 0
            or self._batch_count != 1
            or self._is_left_preconditioner
        ):
            return super()._run(A, b, x, M)

        device, scalar_dtype = self._device, self._scalar_type
        restart = self._restart
        batch_offsets, dofs_per_entry = A.batch_offsets, self._dofs_per_entry
        tiled_dot = self._tiled_dot
        r, w, H, y, V = self._r, self._w, self._H, self._y, self._V
        beta, atol_sq = self._beta, self._atol_sq
        r_norm_sq = tiled_dot.col(0)
        normalize = self._normalize_arnoldi_vec
        axpy = self._arnoldi_axpy
        copy_column = self._copy_hessenberg_col

        _linear._initialize_absolute_tolerance(b, self._tol, self._atol, tiled_dot, atol_sq)
        if getattr(self, "zero_initial_guess", False):
            x.zero_()
            wp.copy(r, b)
        else:
            A.matvec(x, b, r, alpha=-1.0, beta=1.0)
        tiled_dot.compute(r, r, col_offset=0)
        w.zero_()
        self._force_full.zero_()

        def do_arnoldi(j: "int") -> "None":
            """Orthogonalize one restart column and update the residual estimate."""
            if M is not None:
                tmp = V[j + 1]
                M.matvec(V[j], tmp, tmp, alpha=1, beta=0)
                A.matvec(tmp, w, w, alpha=1, beta=0)
            else:
                A.matvec(V[j], w, w, alpha=1, beta=0)
            tiled_dot.compute(self._w_repeated[: j + 1], V[: j + 1])
            copy_column.set_param_at_index(0, j)
            copy_column.launch()
            axpy.set_param_at_index(0, j)
            axpy.launch()
            tiled_dot.compute(w, w)
            normalize.set_params(
                [w, V[j + 1], tiled_dot.col(0), H[:, j + 1, j], batch_offsets, dofs_per_entry]
            )
            normalize.launch()
            wp.launch(
                _update_qr,
                dim=1,
                device=device,
                inputs=[
                    j,
                    restart,
                    self._pivot_tolerance,
                    H,
                    self._qr_column,
                    self._qr_cosine,
                    self._qr_sine,
                    self._qr_remaining,
                    atol_sq,
                    self._force_full,
                    self._inner_condition,
                    self._estimated_convergence,
                    self.adaptive_counters,
                ],
            )

        def do_restart_cycle() -> "None":
            # A skipped column must be exactly zero for the existing LS kernel;
            # skipped basis vectors must not retain data from an earlier solve.
            """Complete one GMRES restart and recompute its true residual."""
            H.zero_()
            V.zero_()
            tiled_dot.compute(r, r)
            normalize.set_params([r, V[0], tiled_dot.col(0), beta, batch_offsets, dofs_per_entry])
            normalize.launch()
            wp.launch(
                _begin_cycle,
                dim=1,
                device=device,
                inputs=[
                    beta,
                    self._qr_remaining,
                    self._inner_condition,
                    self._estimated_convergence,
                    self.adaptive_counters,
                ],
            )
            for j in range(restart):
                wp.capture_if(self._inner_condition, lambda j=j: do_arnoldi(j))
            self._least_squares_solve.launch()

            if M is None:
                wp.launch(
                    _linear._gmres_update_x_kernel,
                    dim=x.shape,
                    device=device,
                    inputs=[restart, scalar_dtype(1.0), y, V, x, batch_offsets, dofs_per_entry],
                )
            else:
                wp.launch(
                    _linear._gmres_update_x_kernel,
                    dim=x.shape,
                    device=device,
                    inputs=[restart, scalar_dtype(0.0), y, V, w, batch_offsets, dofs_per_entry],
                )
                M.matvec(w, r, r, alpha=1, beta=0)
                wp.launch(
                    _linear._gmres_add_active_entries_kernel,
                    dim=x.shape,
                    device=device,
                    inputs=[r, x, batch_offsets, dofs_per_entry],
                )
            wp.copy(src=b, dest=r)
            A.matvec(x, b, r, alpha=-1.0, beta=1.0)
            tiled_dot.compute(r, r)
            wp.launch(
                _check_estimate,
                dim=1,
                device=device,
                inputs=[
                    r_norm_sq,
                    atol_sq,
                    self._estimated_convergence,
                    self._force_full,
                    self.adaptive_counters,
                ],
            )

        # Retain the original restart-budget accounting, including initial -1.
        self._loop_control.fill_(-1)
        self._update_loop.launch()
        if self._loop_callback is not None:
            self._loop_callback.launch()

        def do_cycle_with_condition() -> "None":
            """Run a restart and update the resident loop/callback controls."""
            do_restart_cycle()
            self._update_loop.launch()
            if self._loop_callback is not None:
                self._loop_callback.launch()

        if device.is_capturing:
            wp.capture_while(self._loop_condition, do_cycle_with_condition)
        else:
            with wp.ScopedCapture(device=device) as capture:
                wp.capture_while(self._loop_condition, do_cycle_with_condition)
            wp.capture_launch(capture.graph)
        return self._loop_iteration, r_norm_sq, atol_sq


def gmres(
    A: "LinearOperator",
    b: "wp.array",
    x: "wp.array",
    tol: "float | None" = None,
    atol: "float | None" = None,
    restart: "int" = 31,
    maxiter: "int" = 0,
    M: "LinearOperator | None" = None,
    callback: "Callable[..., None] | None" = None,
    check_every: "int" = 31,
    use_cuda_graph: "bool" = True,
    is_left_preconditioner: "bool" = False,
    run: "bool" = True,
) -> "AdaptiveCapturableGMRES | tuple[int, float, float]":
    """Warp-compatible factory with preallocated adaptive CUDA loop controls."""
    state = AdaptiveCapturableGMRES(
        A,
        b,
        x,
        tol=tol,
        atol=atol,
        restart=restart,
        maxiter=maxiter,
        M=M,
        callback=callback,
        check_every=check_every,
        use_cuda_graph=use_cuda_graph,
        is_left_preconditioner=is_left_preconditioner,
    )
    return state() if run else state
