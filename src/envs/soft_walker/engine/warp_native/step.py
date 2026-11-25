"""Implicit backward-Euler MPM with analytic matrix-free Jacobian actions.

CUDA execution must be captured by the outer caller. CPU execution is a
correctness/debugging path and may use host convergence checks. All allocations
are explicit and reusable. Newton iterates are never differentiated.
"""

from typing import TYPE_CHECKING, cast

from ...contracts.metrics import (
    COUNT_BACKTRACKS,
    COUNT_LINEAR_SOLVES,
    COUNT_NEWTON_ITERATIONS,
    MPM_WORK_COUNT_COLUMNS,
    NEWTON_COUNT_WIDTH,
    WORK_COUNT_WIDTH,
    WORK_INCREMENT_WIDTH,
)
from ._solver_limits import mpm_support_capacity

if TYPE_CHECKING:
    from collections.abc import Iterator

    from warp._src.optim.linear import LinearSolverState

    from .host_types import MPMState, Report, RootLoader, Shape, SolidState
    from .linearization import Linearization
    from .model import DeviceModel
    from .solid_model import SolidDeviceModel

import math

import numpy as np
import warp as wp
from warp.optim.linear import LinearOperator

from .adaptive_krylov import gmres
from .iteration import iteration_kernels
from .krylov import gmres as fixed_gmres
from .scan import InclusiveIntScan
from .support import CapturedSort


class ImplicitStep:
    """Reusable implicit MPM solve and converged-root adjoint workspace.

    All Newton/Krylov arrays belong to this instance. CUDA callers capture
    forward/backward programs into an outer graph; CPU may inspect stopping
    conditions. Solver iterates are not part of the differentiated function."""

    # Allocated by the fixed-shape buffer loops in __init__.
    q: wp.array
    guess: wp.array
    residual: wp.array
    trial_q: wp.array
    trial_r: wp.array
    delta: wp.array
    rhs: wp.array
    linear_ax: wp.array
    tangent_force: wp.array
    force_buffer: wp.array
    norm: wp.array
    minimum: wp.array
    residual_norm: wp.array
    max_linear: wp.array
    bad: wp.array
    code: wp.array
    condition: wp.array
    line_condition: wp.array
    accepted: wp.array

    def __init__(
        self,
        device_model: "DeviceModel",
        state: "MPMState | None" = None,
        command: "wp.array | None" = None,
        output: "MPMState | None" = None,
    ) -> "None":
        """Allocate fixed-capacity solver, support, tangent and adjoint workspaces."""
        self.owner: DeviceModel | SolidDeviceModel = device_model
        self.k = k = device_model.k
        self.it = iteration_kernels(device_model.config.dim, device_model.precision)
        self.device = device_model.device
        self.model = device_model.model
        self.cfg = device_model.parameters
        self.config = device_model.config
        self.state: MPMState | SolidState = device_model.initial if state is None else state
        self.output: MPMState | SolidState = (
            device_model.empty_state() if output is None else output
        )
        self.command = (
            wp.zeros(device_model.channels, dtype=k.real, device=self.device, requires_grad=True)
            if command is None
            else command
        )
        self.incidence_count, self.capacity = mpm_support_capacity(
            device_model.n, self.config.grid_shape, self.config.compact_nodes
        )
        n, s, cap = device_model.n, k.S, self.capacity
        self.st = k.Support()
        for name, shape, dtype in (
            ("ids", (n, s), int),
            ("weight", (n, s), k.real),
            ("gradient", (n, s), k.vec),
            ("offset", (n, s), k.vec),
            ("valid", n, int),
            ("mass", cap, k.real),
            ("momentum", cap, k.vec),
            ("predictor", cap, k.vec),
            ("scale", cap, k.real),
            ("diagonal", cap, k.real),
            ("totals", 2, k.real),
            ("live", 1, int),
            ("work_counts", WORK_COUNT_WIDTH, int),
        ):
            setattr(
                self.st,
                name,
                wp.zeros(
                    shape,
                    dtype=dtype,
                    device=self.device,
                    requires_grad=(dtype is not int and name != "diagonal"),
                ),
            )
        # Warp radix_sort_pairs requires 2 * count storage for its scratch half.
        self.keys = self.array(2 * self.incidence_count, int)
        self.order = self.array(2 * self.incidence_count, int)
        self.flags = self.array(self.incidence_count, int)
        self.prefix = self.array(self.incidence_count, int)
        self.sort = CapturedSort(self.keys, self.order, self.incidence_count)
        self.scan = InclusiveIntScan(self.incidence_count, self.device)
        for name in (
            "q",
            "guess",
            "residual",
            "trial_q",
            "trial_r",
            "delta",
            "rhs",
            "linear_ax",
            "tangent_force",
            "force_buffer",
        ):
            setattr(self, name, self.array(cap, k.vec, grad=True))
        self.next_a = self.array(device_model.channels, k.real, grad=True)
        self.work = self.array(WORK_INCREMENT_WIDTH, k.real, grad=True)
        for name in ("norm", "minimum", "residual_norm", "max_linear"):
            setattr(self, name, self.array(1, k.real))
        self.sums = self.array(2, k.real)
        for name in ("bad", "code", "condition", "line_condition", "accepted"):
            setattr(self, name, self.array(1, int))
        self.counts = self.array(NEWTON_COUNT_WIDTH, int)
        shape = (cap, cap)
        self.operator = LinearOperator(shape, k.vec, self.device, self.matvec)
        self.transpose_operator = LinearOperator(shape, k.vec, self.device, self.transpose_matvec)
        self.preconditioner = LinearOperator(shape, k.vec, self.device, self.precondition)
        rt = (
            self.config.linear_rtol64
            if self.owner.precision == "float64"
            else self.config.linear_rtol32
        )
        self.linear: LinearSolverState
        if self.config.krylov_method == "adaptive_dynamic":
            from .adaptive_gmres import AdaptiveGMRES

            self.linear = AdaptiveGMRES(
                self.operator,
                self.rhs,
                self.delta,
                tol=rt,
                atol=0.0,
                restart=min(self.config.restart, cap * self.config.dim),
                maxiter=self.config.restart * self.config.max_restarts,
                M=self.preconditioner,
            )
        else:
            factory = fixed_gmres if self.config.krylov_method == "legacy" else gmres
            self.linear = factory(
                self.operator,
                self.rhs,
                self.delta,
                tol=rt,
                atol=0.0,
                restart=min(self.config.restart, cap * self.config.dim),
                maxiter=self.config.restart * self.config.max_restarts,
                M=self.preconditioner,
                check_every=0 if self.device.is_cuda or self.config.linear_work_reuse else 1,
                use_cuda_graph=self.device.is_cuda,
                is_left_preconditioner=False,
                run=False,
            )
        self.linear.zero_initial_guess = self.config.linear_work_reuse
        self.correction_rhs = self.array(cap, k.vec)
        self.correction = self.array(cap, k.vec)
        self.refined = self.array(cap, k.vec)
        self.refine_count = self.array(1, int)
        self.refine_condition = self.array(1, int)
        self.refine_accepted = self.array(1, int)
        self.refine_error = self.array(1, k.real)
        # Small mass/volume sensitivities cancel much larger contributions.
        # A float32 adjoint accepted at the primal linear failure threshold can
        # therefore lose useful gradient digits. Use the configured refinement
        # budget to approach arithmetic precision, accepting only improvements.
        # This target changes neither the main GMRES tolerance nor acceptance,
        # and is deliberately specific to MPM (SolidStep has its own setup).
        self.adjoint_refinement_target = self.cfg.linear_limit
        if self.owner.precision == "float32":
            self.adjoint_refinement_target = min(
                self.cfg.linear_limit, 4 * np.finfo(np.float32).eps
            )
        self._linearization_ready = False
        self.linearization = None
        if self.config.cache_tangent:
            from .linearization import Linearization

            self.linearization = Linearization(device_model)
        self.block_inverse = self.array(cap, k.mat) if self.config.block_preconditioner else None
        self.block_matrix = self.array(cap, k.mat) if self.config.block_preconditioner else None
        self.block_fallbacks = self.array(1, int)
        self._transpose_solve = False
        self._tapes: tuple[wp.Tape, wp.Tape] | None = None

    def array(self, shape: "Shape", dtype: "type", grad: "bool" = False) -> "wp.array":
        """Allocate a zeroed Warp array on this step's device."""
        return wp.zeros(shape, dtype=dtype, device=self.device, requires_grad=grad)

    def launch(self, kernel: "wp.Kernel", dim: "Shape", *args: "object") -> "None":
        """Submit a kernel, respecting its explicit enable_backward setting."""
        wp.launch(
            kernel,
            dim=dim,
            inputs=list(args),
            device=self.device,
            record_tape=kernel.options.get("enable_backward") is not False,
        )

    def prepare(self) -> "None":
        """Build compact particle support, mass transfer, actuator state and predictor."""
        self._linearization_ready = False
        k, st = self.k, self.st
        self.launch(
            k.build_stencil,
            self.owner.n,
            self.model,
            self.state,
            self.cfg,
            st,
            self.keys,
            self.order,
        )
        self.sort()
        self.launch(k.mark_unique, self.incidence_count, self.keys, self.flags)
        self.scan(self.flags, self.prefix)
        self.launch(
            k.assign_rows,
            self.incidence_count,
            self.keys,
            self.order,
            self.prefix,
            self.incidence_count,
            st,
        )
        st.mass.zero_()
        st.momentum.zero_()
        st.totals.zero_()
        self.launch(k.p2g, self.owner.n, self.model, self.state, st)
        self.launch(k.mass_totals, self.capacity, st)
        self.launch(k.prepare_grid, self.capacity, st, self.cfg, self.guess)
        self.launch(
            k.actuator, self.owner.channels, self.state, self.command, self.cfg, self.next_a
        )
        st.diagonal.zero_()
        self.launch(
            k.bound_diagonal, self.owner.n, self.model, self.state, st, self.cfg, self.next_a
        )
        self.launch(k.finalize_diagonal, self.capacity, st, self.cfg)

    def evaluate(self, q: "wp.array", out: "wp.array") -> "None":
        """Write the nonlinear residual for dimensionless grid velocities q into out."""
        self.force_buffer.zero_()
        self.launch(
            self.k.force,
            self.owner.n,
            self.model,
            self.state,
            self.st,
            self.cfg,
            q,
            self.next_a,
            self.force_buffer,
        )
        self.launch(
            self.k.residual_grid, self.capacity, self.st, self.cfg, q, self.force_buffer, out
        )

    def validity(self, q: "wp.array") -> "None":
        """Update device admissibility flags for the proposed grid velocities."""
        self.bad.zero_()
        self.minimum.fill_(1e30)
        self.launch(
            self.k.valid_state,
            self.owner.n,
            self.model,
            self.state,
            self.st,
            self.cfg,
            q,
            self.minimum,
            self.bad,
        )

    def inspect_residual(self, q: "wp.array", r: "wp.array") -> "None":
        """Reduce residual norms and geometric failure flags into device controls."""
        self.validity(q)
        self.norm.zero_()
        self.launch(self.it.maximum, self.capacity, r, self.norm, self.bad)

    def matvec(
        self, x: "wp.array", y: "wp.array", z: "wp.array", alpha: "float" = 1.0, beta: "float" = 0.0
    ) -> "None":
        """Write z = alpha * Jx + beta * y for the current implicit residual."""
        self.apply_tangent(x, y, z, alpha, beta, transpose=False)

    def transpose_matvec(
        self, x: "wp.array", y: "wp.array", z: "wp.array", alpha: "float" = 1.0, beta: "float" = 0.0
    ) -> "None":
        """Write z = alpha * Jáµ€x + beta * y without assembling a global matrix."""
        self.apply_tangent(x, y, z, alpha, beta, transpose=True)

    def apply_tangent(
        self,
        x: "wp.array",
        y: "wp.array",
        z: "wp.array",
        alpha: "float",
        beta: "float",
        transpose: "bool",
    ) -> "None":
        """Apply the analytic residual Jacobian or transpose to device vectors."""
        self.tangent_force.zero_()
        if self.linearization is None or not self._linearization_ready:
            self.launch(
                self.k.force_tangent,
                self.owner.n,
                self.model,
                self.state,
                self.st,
                self.cfg,
                self.q,
                x,
                self.next_a,
                int(transpose),
                self.tangent_force,
            )
        else:
            self.launch(
                self.linearization.kernels.apply,
                self.owner.n,
                self.model,
                self.state,
                self.st,
                self.cfg,
                x,
                int(transpose),
                self.linearization.data,
                self.tangent_force,
            )
        self.launch(
            self.k.tangent_grid,
            self.capacity,
            self.st,
            self.cfg,
            x,
            self.tangent_force,
            y,
            self.k.real(alpha),
            self.k.real(beta),
            int(transpose),
            z,
        )

    def precondition(
        self, x: "wp.array", y: "wp.array", z: "wp.array", alpha: "float" = 1.0, beta: "float" = 0.0
    ) -> "None":
        """Apply the selected diagonal/block preconditioner into caller-owned z."""
        if self.block_inverse is not None:
            self.launch(
                cast("Linearization", self.linearization).kernels.apply_preconditioner,
                self.capacity,
                self.block_inverse,
                int(self._transpose_solve),
                x,
                y,
                self.k.real(alpha),
                self.k.real(beta),
                z,
            )
        else:
            self.launch(
                self.k.precondition,
                self.capacity,
                self.st,
                x,
                y,
                self.k.real(alpha),
                self.k.real(beta),
                z,
            )

    def linear_check(self, value: "wp.array | None" = None, transpose: "bool" = False) -> "None":
        """Measure the true linear residual on device after a Krylov solve."""
        self.sums.zero_()
        value = self.delta if value is None else value
        apply = self.transpose_matvec if transpose else self.matvec
        apply(value, self.linear_ax, self.linear_ax)
        self.launch(self.it.linear_error, self.capacity, self.linear_ax, self.rhs, self.sums)

    def refresh_linearization(self) -> "None":
        """Rebuild at this q; cached data never crosses a changed Newton root."""
        if self.linearization is not None:
            self.launch(
                self.linearization.kernels.build,
                self.owner.n,
                self.model,
                self.state,
                self.st,
                self.cfg,
                self.q,
                self.next_a,
                self.linearization.data,
            )
            self._linearization_ready = True
            if self.block_inverse is not None:
                cast("wp.array", self.block_matrix).zero_()
                self.block_fallbacks.zero_()
                self.launch(
                    self.linearization.kernels.diagonal_blocks,
                    (self.owner.n, self.k.S),
                    self.model,
                    self.state,
                    self.st,
                    self.cfg,
                    self.linearization.data,
                    self.block_matrix,
                )
                threshold = self.k.real(64 * np.finfo(self.owner.precision).eps)
                self.launch(
                    self.linearization.kernels.invert_blocks,
                    self.capacity,
                    self.st,
                    self.cfg,
                    self.block_matrix,
                    self.block_inverse,
                    threshold,
                    self.block_fallbacks,
                )

    def linear_solve(self, transpose: "bool" = False) -> "None":
        """Solve and optionally refine against the independently evaluated residual."""
        self._transpose_solve = bool(transpose)
        self.refresh_linearization()
        operator = self.transpose_operator if transpose else self.operator
        self.delta.zero_()
        self.linear(A=operator)
        self.linear_check(transpose=transpose)
        if self.config.linear_refinements:
            target = self.cfg.linear_limit
            if transpose:
                target = getattr(self, "adjoint_refinement_target", target)
            target = self.k.real(target)
            self.launch(
                self.it.refinement_begin,
                1,
                self.sums,
                target,
                self.refine_error,
                self.refine_count,
                self.refine_condition,
            )

            def refine() -> "None":
                """Attempt one correction and accept it only if the true residual improves."""
                self.linear_check(transpose=transpose)
                self.launch(
                    self.it.correction_rhs,
                    self.capacity,
                    self.rhs,
                    self.linear_ax,
                    self.correction_rhs,
                )
                self.correction.zero_()
                self.linear(A=operator, b=self.correction_rhs, x=self.correction)
                self.launch(
                    self.it.add_correction, self.capacity, self.delta, self.correction, self.refined
                )
                self.linear_check(value=self.refined, transpose=transpose)
                self.launch(
                    self.it.refinement_assess,
                    1,
                    self.sums,
                    target,
                    self.refine_error,
                    self.refine_count,
                    self.config.linear_refinements,
                    self.refine_accepted,
                    self.refine_condition,
                )
                self.launch(
                    self.it.refinement_update,
                    self.capacity,
                    self.refined,
                    self.refine_accepted,
                    self.delta,
                )

            wp.capture_while(self.refine_condition, refine)
            if self.config.linear_work_reuse:
                # The independent residual remains current when no correction
                # was attempted. A rejected candidate may overwrite linear_ax;
                # any attempted refinement therefore triggers a fresh check.
                wp.capture_if(self.refine_count, lambda: self.linear_check(transpose=transpose))
            else:
                self.linear_check(transpose=transpose)

        # Captured solver operators above use the fixed-q cache. Direct probes
        # after this solve use the original operator until explicitly refreshed.
        self._linearization_ready = False

    def line_body(self) -> "None":
        """Evaluate one safeguarded backtracking candidate and update device controls."""
        self.launch(self.it.candidate, self.capacity, self.q, self.delta, self.counts, self.trial_q)
        self.evaluate(self.trial_q, self.trial_r)
        self.inspect_residual(self.trial_q, self.trial_r)
        self.launch(
            self.it.assess,
            1,
            self.norm,
            self.bad,
            self.cfg,
            self.residual_norm,
            self.code,
            self.counts,
            self.line_condition,
            self.accepted,
        )
        self.launch(
            self.it.accept,
            self.capacity,
            self.trial_q,
            self.trial_r,
            self.accepted,
            self.q,
            self.residual,
        )

    def newton_body(self) -> "None":
        """Perform one Newton iteration with its linear solve and line search."""
        self.launch(self.it.negate, self.capacity, self.residual, self.rhs)
        self.linear_solve()
        self.launch(
            self.it.start_linesearch,
            1,
            self.sums,
            self.cfg,
            self.code,
            self.counts,
            self.condition,
            self.line_condition,
            self.accepted,
            self.max_linear,
        )
        wp.capture_while(self.line_condition, self.line_body)
        self.launch(
            self.it.finish_iteration,
            1,
            self.residual_norm,
            self.cfg,
            self.code,
            self.counts,
            self.condition,
        )

    def solve(self) -> "None":
        """Run Newton until device convergence/failure controls end the loop."""
        wp.copy(self.q, self.guess)
        self.validity(self.q)
        self.launch(self.it.fallback, self.capacity, self.q, self.bad)
        self.evaluate(self.q, self.residual)
        self.inspect_residual(self.q, self.residual)
        self.max_linear.zero_()
        self.launch(
            self.it.initialize,
            1,
            self.norm,
            self.bad,
            self.state.status,
            self.cfg,
            self.code,
            self.counts,
            self.condition,
            self.residual_norm,
        )
        wp.capture_while(self.condition, self.newton_body)

    def finish(self) -> "None":
        """Commit an admissible state and update work/energy/time ledger entries."""
        self.work.zero_()
        self.launch(
            self.k.commit,
            self.owner.n,
            self.model,
            self.state,
            self.st,
            self.cfg,
            self.q,
            self.next_a,
            self.code,
            self.output,
            self.work,
        )
        self.launch(
            self.k.finish_state,
            1,
            self.state,
            self.next_a,
            self.cfg,
            self.code,
            self.work,
            self.output,
        )

    def forward(
        self, record_derivatives: "bool" = False, solved_root_loader: "RootLoader | None" = None
    ) -> "MPMState | SolidState":
        """Advance or reconstruct output from a root saved in the same replay block.

        A saved root is used only for reverse reconstruction, never as an
        unchecked new forward solution. Its source solve and status are retained.
        The implicit VJP still differentiates prepare/output/residual kernels.
        """
        if solved_root_loader is not None and not record_derivatives:
            raise ValueError("A saved root is only supported for reverse reconstruction")
        if self.device.is_cuda and not self.device.is_capturing:
            raise RuntimeError(
                "CUDA stepping must be inside an outer graph capture; no host-stepped fallback"
            )
        if record_derivatives:
            prepare_tape = wp.Tape()
            with prepare_tape:
                self.prepare()
        else:
            self.prepare()
        if solved_root_loader is None:
            self.solve()
        else:
            solved_root_loader()
            # Rebuilt support/mass reductions can round differently on CUDA.
            # Recheck the saved root against this prepared system on device.
            self.evaluate(self.q, self.residual)
            self.inspect_residual(self.q, self.residual)
            self.max_linear.zero_()
            self.launch(
                self.it.check_replayed_root,
                1,
                self.norm,
                self.bad,
                self.state.status,
                self.cfg,
                self.code,
                self.counts,
                self.condition,
                self.residual_norm,
                self.st.work_counts,
            )
            wp.capture_if(self.condition, self.solve)
        if record_derivatives:
            output_tape = wp.Tape()
            with output_tape:
                self.finish()
            self._tapes = (prepare_tape, output_tape)
        else:
            self.finish()
        return self.output

    def diagnostics(self) -> "Report":
        """Explicit host reporting boundary; never called during graph capture."""
        if self.device.is_capturing:
            raise RuntimeError("Diagnostics cannot be downloaded during capture")
        return dict(
            code=int(self.code.numpy()[0]),
            residual=float(self.residual_norm.numpy()[0]),
            max_linear_residual=float(self.max_linear.numpy()[0]),
            iterations=int(self.counts.numpy()[COUNT_NEWTON_ITERATIONS]),
            backtracks=int(self.counts.numpy()[COUNT_BACKTRACKS]),
            linear_solves=int(self.counts.numpy()[COUNT_LINEAR_SOLVES]),
            live_nodes=int(self.st.live.numpy()[0]),
            capacity_nodes=self.capacity,
            capacity_exceeded=bool(self.st.live.numpy()[0] > self.capacity),
            capacity_overflow=int(self.st.live.numpy()[0]) > self.capacity,
            linear_work_reuse=self.config.linear_work_reuse,
            block_preconditioner=self.config.block_preconditioner,
            block_fallbacks=int(self.block_fallbacks.numpy()[0]),
            dense_grid_nodes=math.prod(self.config.grid_shape),
            incidence_count=self.incidence_count,
            krylov_method=self.config.krylov_method,
            tangent_cache_bytes=0 if self.linearization is None else self.linearization.bytes,
            work_counts=dict(
                zip(
                    MPM_WORK_COUNT_COLUMNS,
                    map(int, self.st.work_counts.numpy()),
                )
            ),
        )

    def gradient_arrays(self) -> "Iterator[wp.array]":
        """Unique differentiable storage, including shared model/state leaves."""
        seen = set()
        objects = [self.model, self.state, self.output, self.st]
        candidates = [v for obj in objects for v in vars(obj).values()]
        candidates += [v for v in vars(self).values() if isinstance(v, wp.array)]
        for array in candidates:
            if isinstance(array, wp.array) and array.grad is not None and array.ptr not in seen:
                seen.add(array.ptr)
                yield array

    def clear_gradients(self) -> "None":
        """Zero all uniquely owned or shared differentiable array cotangents."""
        for array in self.gradient_arrays():
            array.grad.zero_()

    def backward(self, seeds: "dict[wp.array, wp.array]", *, clear: "bool" = True) -> "wp.array":
        """Implicit VJP of the accepted step, not of Newton's iteration history.

        ``seeds`` maps output Warp arrays to matching device cotangents. Call
        forward(record_derivatives=True) first, without changing its inputs.
        Solver failure produces NaN cotangents, never fabricated zero gradients.
        """
        if self.device.is_cuda and not self.device.is_capturing:
            raise RuntimeError("CUDA reverse evaluation requires an outer graph capture")
        if self._tapes is None:
            raise RuntimeError("Call forward(record_derivatives=True) before backward")
        if clear:
            self.clear_gradients()
        prepare_tape, output_tape = self._tapes
        output_tape.backward(grads=seeds)
        wp.copy(self.rhs, self.q.grad)
        self.linear_solve(transpose=True)
        self.launch(
            self.it.check_adjoint, self.capacity, self.sums, self.cfg, self.delta, self.code
        )
        if self.config.parameter_adjoint == "mass":
            self.launch(
                self.k.mass_adjoint_seed, self.capacity, self.st, self.cfg, self.delta, self.rhs
            )
        else:
            self.launch(self.it.negate, self.capacity, self.delta, self.rhs)
        # R(q, inputs) is replayed at the converged q. Its q-cotangent is ignored;
        # the input contribution is -R_inputs^T * solve(R_q^T, output_q).
        with wp.Tape() as residual_tape:
            if self.config.parameter_adjoint == "mass":
                self.force_buffer.zero_()
                self.launch(
                    self.k.force,
                    self.owner.n,
                    self.model,
                    self.state,
                    self.st,
                    self.cfg,
                    self.q,
                    self.next_a,
                    self.force_buffer,
                )
                self.launch(
                    self.k.mass_residual_grid,
                    self.capacity,
                    self.st,
                    self.cfg,
                    self.q,
                    self.force_buffer,
                    self.residual,
                )
            else:
                self.evaluate(self.q, self.residual)
        residual_tape.backward(grads={self.residual: self.rhs})
        prepare_tape.backward()
        return self.command.grad
