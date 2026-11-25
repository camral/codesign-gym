"""Implicit native spring, beam and Hex8 steps with sparse interaction blocks.

Warp AD assembles independent rows of 4x4 / 12x12 / 24x24 interaction Jacobians.
No global dense matrix or numerical force differences are used. The original
residual is differentiated at the converged root; Newton iterates are not taped.
"""

from typing import TYPE_CHECKING

from ...contracts.metrics import (
    COUNT_BACKTRACKS,
    COUNT_LINEAR_SOLVES,
    COUNT_NEWTON_ITERATIONS,
    NEWTON_COUNT_WIDTH,
    SOLID_WORK_COUNT_COLUMNS,
    WORK_COUNT_WIDTH,
    WORK_INCREMENT_WIDTH,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from .host_types import Report, SolidState
    from .solid_model import SolidConfig, SolidDeviceModel

from types import SimpleNamespace

import warp as wp
from warp.optim.linear import LinearOperator

from .adaptive_gmres import AdaptiveGMRES
from .iteration import iteration_kernels
from .solid_model import ARRAY_FIELDS, STATE_FIELDS
from .step import ImplicitStep


def constant_view(value: "wp.array") -> "wp.array":
    """Alias device storage without cotangent storage; never download it."""
    if isinstance(value, wp.array):
        return wp.array(
            ptr=value.ptr,
            shape=value.shape,
            strides=value.strides,
            dtype=value.dtype,
            device=value.device,
            requires_grad=False,
        )
    if hasattr(value, "_cls"):
        out = value._cls()
        for name in value._cls.vars:
            if not name.startswith("_"):
                setattr(out, name, constant_view(getattr(value, name)))
        return out
    return value


class SolidStep(ImplicitStep):
    """Nodal spring/beam/FEM implicit step using cached local tangent blocks.

    Inherits Newton, line-search and implicit-adjoint orchestration from
    ImplicitStep while retaining solid-specific force and state kernels."""

    owner: "SolidDeviceModel"
    state: "SolidState"
    output: "SolidState"
    config: "SolidConfig"

    def __init__(
        self,
        owner: "SolidDeviceModel",
        state: "SolidState | None" = None,
        command: "wp.array | None" = None,
        output: "SolidState | None" = None,
    ) -> "None":
        """Allocate nodal, element-tangent and adjoint workspaces for one solid owner."""
        if owner.config.integrator != "backward_euler":
            raise ValueError(
                "SolidStep requires backward_euler; use SpringKvaernoStep for kvaerno3_native"
            )
        self.owner = owner
        self.k = k = owner.k
        self.it = iteration_kernels(owner.config.dim, owner.precision, k.Q)
        self.device, self.model, self.cfg, self.config = (
            owner.device,
            owner.model,
            owner.parameters,
            owner.config,
        )
        self.state = owner.initial if state is None else state
        self.output = owner.empty_state() if output is None else output
        self.command = self.array(owner.channels, k.real, True) if command is None else command
        self.capacity = n = owner.n
        self.incidence_count = owner.ne * k.N
        ne = owner.ne
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
            "correction_rhs",
            "correction",
            "refined",
        ):
            setattr(
                self,
                name,
                self.array(n, k.qvec, name not in ("correction_rhs", "correction", "refined")),
            )
        self.next_a = self.array(owner.channels, k.real, True)
        self.work = self.array(WORK_INCREMENT_WIDTH, k.real, True)
        for name in ("norm", "minimum", "residual_norm", "max_linear", "refine_error"):
            setattr(self, name, self.array(1, k.real))
        self.sums = self.array(2, k.real)
        for name in (
            "bad",
            "code",
            "condition",
            "line_condition",
            "accepted",
            "refine_count",
            "refine_condition",
            "refine_accepted",
        ):
            setattr(self, name, self.array(1, int))
        self.counts = self.array(NEWTON_COUNT_WIDTH, int)
        self.st = SimpleNamespace(
            diagonal=self.array(n, k.qvec),
            work_counts=self.array(WORK_COUNT_WIDTH, int),
            live=wp.array([n], dtype=int, device=self.device),
        )
        self.rows = self.array((ne, k.L), k.lvec, True)
        self.row_seeds = self.array((ne, k.L), k.lvec)
        self.row_output = self.array((ne, k.L), k.lvec)
        self.node_rows = self.array((n, k.Q), k.qvec, True)
        self.node_seeds = self.array((n, k.Q), k.qvec)
        self.node_output = self.array((n, k.Q), k.qvec)
        self.linear_model, self.linear_state, self.linear_a = (
            constant_view(self.model),
            constant_view(self.state),
            constant_view(self.next_a),
        )
        tape = wp.Tape()
        self.zero_model, self.zero_state, self.zero_cfg = (
            tape.get_adjoint(self.linear_model),
            tape.get_adjoint(self.linear_state),
            tape.get_adjoint(self.cfg),
        )
        shape = (n, n)
        self.operator = LinearOperator(shape, k.qvec, self.device, self.matvec)
        self.transpose_operator = LinearOperator(shape, k.qvec, self.device, self.transpose_matvec)
        self.preconditioner = LinearOperator(shape, k.qvec, self.device, self.precondition)
        rt = (
            self.config.linear_rtol64 if owner.precision == "float64" else self.config.linear_rtol32
        )
        self.linear = AdaptiveGMRES(
            self.operator,
            self.rhs,
            self.delta,
            restart=min(self.config.restart, n * k.Q),
            tol=rt,
            atol=0.0,
            maxiter=self.config.restart * self.config.max_restarts,
            M=self.preconditioner,
        )
        self.linear.zero_initial_guess = True
        self._linearization_ready = False
        self.linearization = None
        self._tapes = None
        self.analytic = None
        if self.config.tangent_assembly != "autodiff" and self.config.physics in ("spring", "fem"):
            from .solid_tangents import analytic_tangent_kernels

            self.analytic = analytic_tangent_kernels(self.config.physics, owner.precision)

    def linear_solve(self, transpose: "bool" = False) -> "None":
        """Capture a more accurate adjoint solve in the same solver workspace.

        Only the Krylov stopping target changes; forward solves, iteration and
        refinement budgets, and independent residual acceptance stay unchanged.
        These Python assignments happen during graph construction, not replay.
        """
        forward_target = self.linear._tol
        if transpose:
            target = (
                self.config.adjoint_rtol64
                if self.owner.precision == "float64"
                else self.config.adjoint_rtol32
            )
            self.linear._tol = min(forward_target, target)
        try:
            return super().linear_solve(transpose)
        finally:
            self.linear._tol = forward_target

    def prepare(self) -> "None":
        """Compute actuator state and dimensionless nodal predictor before Newton."""
        self._linearization_ready = False
        self.launch(
            self.k.prepare_guess, self.capacity, self.model, self.state, self.cfg, self.guess
        )
        self.launch(
            self.k.actuator, self.owner.channels, self.state, self.command, self.cfg, self.next_a
        )

    def evaluate(self, q: "wp.array", out: "wp.array") -> "None":
        """Write the nodal residual for candidate dimensionless velocities into out."""
        self.force_buffer.zero_()
        self.launch(
            self.k.force_kernel,
            self.owner.ne,
            self.model,
            self.state,
            self.cfg,
            q,
            self.next_a,
            self.force_buffer,
        )
        self.launch(
            self.k.node_residual,
            self.capacity,
            self.model,
            self.state,
            self.cfg,
            q,
            self.force_buffer,
            out,
        )

    def validity(self, q: "wp.array") -> "None":
        """Update nodal/entity/cell admissibility flags for the proposed solution."""
        self.bad.zero_()
        self.minimum.fill_(1.0)
        self.launch(
            self.k.valid_nodes, self.capacity, self.model, self.state, self.cfg, q, self.bad
        )
        self.launch(
            self.k.valid_entities,
            self.owner.ne,
            self.model,
            self.state,
            self.cfg,
            q,
            self.minimum,
            self.bad,
        )
        if self.config.physics == "spring":
            self.launch(
                self.k.valid_cells,
                self.model.cells.shape[0],
                self.model,
                self.state,
                self.cfg,
                q,
                self.bad,
            )

    def refresh_linearization(self) -> "None":
        """Build local force/contact tangent blocks and the nodal preconditioner."""
        k = self.k
        if self.analytic is not None:
            self.launch(
                self.analytic.build,
                (self.owner.ne, k.L),
                self.model,
                self.state,
                self.cfg,
                self.q,
                self.next_a,
                self.rows.grad,
                self.st.work_counts,
            )
        else:
            self.launch(
                k.prepare_rows,
                (self.owner.ne, k.L),
                self.model,
                self.q,
                self.rows,
                self.row_seeds,
                self.st.work_counts,
            )
            self.rows.grad.zero_()
            # AD re-evaluates the local primal. Its output is write-only; no
            # duplicated forward force launch is necessary here.
            wp.launch(
                k.row_force,
                dim=(self.owner.ne, k.L),
                inputs=[self.linear_model, self.linear_state, self.cfg, self.rows, self.linear_a],
                outputs=[self.row_output],
                adj_inputs=[self.zero_model, self.zero_state, self.zero_cfg, self.rows.grad, None],
                adj_outputs=[self.row_seeds],
                adjoint=True,
                record_tape=False,
                device=self.device,
            )
        self.launch(k.prepare_nodes, (self.capacity, k.Q), self.q, self.node_rows, self.node_seeds)
        self.node_rows.grad.zero_()
        wp.launch(
            k.row_contact,
            dim=(self.capacity, k.Q),
            inputs=[self.linear_model, self.linear_state, self.cfg, self.node_rows],
            outputs=[self.node_output],
            adj_inputs=[self.zero_model, self.zero_state, self.zero_cfg, self.node_rows.grad],
            adj_outputs=[self.node_seeds],
            adjoint=True,
            record_tape=False,
            device=self.device,
        )
        self.st.diagonal.zero_()
        self.launch(
            k.accumulate_diagonal,
            (self.owner.ne, k.L),
            self.model,
            self.cfg,
            self.rows.grad,
            self.st.diagonal,
        )
        self.launch(
            k.finalize_diagonal,
            self.capacity,
            self.model,
            self.cfg,
            self.node_rows.grad,
            self.st.diagonal,
        )
        self._linearization_ready = True

    def apply_tangent(
        self,
        x: "wp.array",
        y: "wp.array",
        z: "wp.array",
        alpha: "float",
        beta: "float",
        transpose: "bool",
    ) -> "None":
        """Accumulate element/contact blocks into Jx or Jáµ€x without a global matrix."""
        if not self._linearization_ready:
            # Direct probes use this root and current inputs. Only an
            # explicit refresh/linear solve establishes a reusable cache.
            self.refresh_linearization()
            self._linearization_ready = False
        self.tangent_force.zero_()
        self.launch(
            self.k.apply_blocks,
            self.owner.ne,
            self.model,
            self.cfg,
            self.rows.grad,
            x,
            int(transpose),
            self.tangent_force,
        )
        self.launch(
            self.k.apply_nodes,
            self.capacity,
            self.model,
            self.cfg,
            self.node_rows.grad,
            x,
            self.tangent_force,
            y,
            self.k.real(alpha),
            self.k.real(beta),
            int(transpose),
            z,
            self.st.work_counts,
        )

    def precondition(
        self, x: "wp.array", y: "wp.array", z: "wp.array", alpha: "float" = 1.0, beta: "float" = 0.0
    ) -> "None":
        """Apply nodal diagonal blocks into the caller-owned result array."""
        self.launch(
            self.k.precondition,
            self.capacity,
            self.st.diagonal,
            x,
            y,
            self.k.real(alpha),
            self.k.real(beta),
            z,
        )

    def finish(self) -> "None":
        """Commit admissible nodal state and accumulate physical work/time ledgers."""
        self.work.zero_()
        self.launch(
            self.k.commit_nodes,
            self.capacity,
            self.model,
            self.state,
            self.cfg,
            self.q,
            self.code,
            self.output,
            self.work,
        )
        self.launch(
            self.k.work_kernel,
            self.owner.ne,
            self.model,
            self.state,
            self.cfg,
            self.q,
            self.next_a,
            self.code,
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

    def diagnostics(self) -> "Report":
        """Download solver counts and workspace sizes; prohibited during capture."""
        if self.device.is_capturing:
            raise RuntimeError("Host diagnostics are prohibited during graph capture")
        arrays = (
            self.rows,
            self.rows.grad,
            self.row_output,
            self.row_seeds,
            self.node_rows,
            self.node_rows.grad,
            self.node_output,
            self.node_seeds,
        )
        return dict(
            physics=self.config.physics,
            code=int(self.code.numpy()[0]),
            residual=float(self.residual_norm.numpy()[0]),
            max_linear_residual=float(self.max_linear.numpy()[0]),
            iterations=int(self.counts.numpy()[COUNT_NEWTON_ITERATIONS]),
            linear_solves=int(self.counts.numpy()[COUNT_LINEAR_SOLVES]),
            backtracks=int(self.counts.numpy()[COUNT_BACKTRACKS]),
            dofs=self.capacity * self.k.Q,
            interactions=self.owner.ne,
            local_block_width=self.k.L,
            tangent_assembly="analytic" if self.analytic is not None else "autodiff",
            work_counts=dict(
                zip(
                    SOLID_WORK_COUNT_COLUMNS,
                    map(int, self.st.work_counts.numpy()),
                )
            ),
            local_jacobian_bytes=self.rows.grad.capacity + self.node_rows.grad.capacity,
            local_linearization_workspace_bytes=sum(a.capacity for a in arrays),
            krylov_columns=int(self.linear.state.numpy()[0]),
        )

    def gradient_arrays(self) -> "Iterator[wp.array]":
        """Yield unique differentiable model, state and workspace arrays by address."""
        seen = set()
        candidates = [getattr(self.model, n) for n in ARRAY_FIELDS]
        for state in (self.state, self.output):
            candidates.extend(getattr(state, n) for n in STATE_FIELDS)
        candidates.extend(v for v in vars(self).values() if isinstance(v, wp.array))
        for a in candidates:
            if a.grad is not None and a.ptr not in seen:
                seen.add(a.ptr)
                yield a
