"""Native fixed-step spring Kvaerno3 with three converged-root adjoints.

The tableau matches the JAX ``kvaerno3_native`` implementation. Position stage
equations are eliminated analytically, leaving existing sparse velocity solves
at gamma*dt. CUDA execution remains inside the caller's conditional graph.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from copy import copy
from dataclasses import replace
from types import SimpleNamespace
from typing import TYPE_CHECKING

import warp as wp

from ...contracts.metrics import (
    COUNT_BACKTRACKS,
    COUNT_LINEAR_SOLVES,
    COUNT_NEWTON_ITERATIONS,
    LEDGER_WIDTH,
    NEWTON_COUNT_WIDTH,
    WORK_COUNT_WIDTH,
    WORK_INCREMENT_WIDTH,
)
from ..numerics.integration_tableaux import (
    KVAERNO3_A as A,
    KVAERNO3_C as C,
    KVAERNO3_GAMMA as GAMMA,
)
from .solid_model import STATE_FIELDS
from .solid_step import SolidStep
from .spring_kvaerno_kernels import spring_kvaerno_kernels

if TYPE_CHECKING:
    from .host_types import Report, RootLoader, SolidState
    from .solid_model import SolidDeviceModel

STAGES = (
    (A[1][0], 0.0, 0.0, C[1]),
    (A[2][0], A[2][1], 0.0, C[2]),
    (A[3][0], A[3][1], A[3][2], C[3]),
)


def _parameters(value: wp.codegen.StructInstance) -> wp.codegen.StructInstance:
    """Copy runtime Warp scalar parameters without aliasing their struct storage."""
    result = value._cls()
    for name in value._cls.vars:
        if not name.startswith("_"):
            setattr(result, name, getattr(value, name))
    return result


class SpringKvaernoStep(SolidStep):
    """Fixed-topology spring stages with an explicit device displacement output.

    Shared model arrays are borrowed from the owner. Stage state, solver and
    cotangent storage belong to this step. Backward Euler remains the default
    elsewhere; this implementation is selected only by ``kvaerno3_native``.
    """

    def __init__(
        self,
        owner: SolidDeviceModel,
        state: SolidState | None = None,
        command: wp.array | None = None,
        output: SolidState | None = None,
    ) -> None:
        """Allocate three native solves and differentiable inter-stage buffers."""
        if owner.config.physics != "spring" or owner.config.integrator != "kvaerno3_native":
            raise ValueError("SpringKvaernoStep requires a spring kvaerno3_native bundle")
        self.owner, self.device = owner, owner.device
        self.k, self.model, self.cfg, self.config = (
            owner.k,
            owner.model,
            owner.parameters,
            owner.config,
        )
        self.algebra = spring_kvaerno_kernels(owner.precision)
        self.state = owner.initial if state is None else state
        self.output = owner.empty_state() if output is None else output
        self.command = self.array(owner.channels, self.k.real, True) if command is None else command
        self.capacity = 3 * owner.n  # Three packed roots for checkpoint replay.
        self.incidence_count = owner.ne * self.k.N
        self.q = self.array(self.capacity, self.k.qvec)
        self.next_a = self.array(owner.channels, self.k.real, True)
        self.displacement = self.array(owner.n, self.k.vec, True)
        self.displacement_seed = self.array(owner.n, self.k.vec)
        self.initial_q = self.array(owner.n, self.k.qvec, True)
        self.initial_force = self.array(owner.n, self.k.qvec, True)
        self.final_q = self.array(owner.n, self.k.qvec, True)
        self.work = self.array(WORK_INCREMENT_WIDTH, self.k.real, True)
        self.work_state = owner.empty_state()
        self.increment_u = [self.array(owner.n, self.k.vec, True) for _ in range(4)]
        self.increment_v = [self.array(owner.n, self.k.vec, True) for _ in range(4)]
        self.stage_states = [owner.empty_state() for _ in range(3)]
        self.st = SimpleNamespace(
            live=wp.array([owner.n], dtype=int, device=self.device),
            work_counts=self.array(WORK_COUNT_WIDTH, int),
        )
        self.code, self.saved_code, self.counts = (
            self.array(1, int),
            self.array(1, int),
            self.array(NEWTON_COUNT_WIDTH, int),
        )
        self.sums = self.array(2, self.k.real)
        self.residual_norm, self.max_linear = self.array(1, self.k.real), self.array(1, self.k.real)
        self.zero_dt = _parameters(self.cfg)
        self.zero_dt.dt = 0.0
        self.stages: list[SolidStep] = []
        for stage_state in self.stage_states:
            stage_owner = copy(owner)
            stage_owner.config = replace(
                owner.config,
                integrator="backward_euler",
                dt=GAMMA * owner.config.dt,
                atol64=GAMMA * owner.config.atol64,
                atol32=GAMMA * owner.config.atol32,
            )
            stage_owner.parameters = _parameters(self.cfg)
            stage_owner.parameters.dt *= GAMMA
            # R_velocity = gamma * R_Kvaerno; keep the original stage
            # infinity-norm criterion rather than relaxing it by 1/gamma.
            stage_owner.parameters.atol *= GAMMA
            stage = SolidStep(stage_owner, state=stage_state, command=self.command)
            stage.st.work_counts = self.st.work_counts
            self.stages.append(stage)
        self.it, self.analytic = self.stages[0].it, self.stages[0].analytic
        self._initial_tape: wp.Tape | None = None
        self._final_tape: wp.Tape | None = None
        self._setup_tapes: list[wp.Tape] = []
        self._increment_tapes: list[wp.Tape] = []

    def _tape(self, operation: Callable[[], None], record: bool) -> wp.Tape | None:
        """Record only stage algebra; Newton search history stays outside tapes."""
        if not record:
            operation()
            return None
        tape = wp.Tape()
        with tape:
            operation()
        return tape

    def _explicit_stage(self) -> None:
        """Evaluate dt*RHS at the original state with its original actuator."""
        self.launch(
            self.algebra.initial_velocity,
            self.owner.n,
            self.model,
            self.state,
            self.cfg,
            self.initial_q,
        )
        self.initial_force.zero_()
        self.launch(
            self.k.force_kernel,
            self.owner.ne,
            self.model,
            self.state,
            self.zero_dt,
            self.initial_q,
            self.state.actuator,
            self.initial_force,
        )
        self.launch(
            self.algebra.explicit_increment,
            self.owner.n,
            self.model,
            self.state,
            self.cfg,
            self.zero_dt,
            self.initial_q,
            self.initial_force,
            self.increment_u[0],
            self.increment_v[0],
        )

    def _setup_stage(self, index: int) -> None:
        """Build the known stage base and exact actuator state before its gamma step."""
        a0, a1, a2, time = STAGES[index]
        status = self.state.status if index == 0 else self.stages[index - 1].output.status
        self.launch(
            self.algebra.stage_input,
            max(self.owner.n, self.owner.channels, LEDGER_WIDTH),
            self.state,
            self.increment_u[0],
            self.increment_v[0],
            self.increment_u[1],
            self.increment_v[1],
            self.increment_u[2],
            self.increment_v[2],
            self.k.real(a0),
            self.k.real(a1),
            self.k.real(a2),
            self.k.real(time - GAMMA),
            self.command,
            self.cfg,
            status,
            self.stage_states[index],
        )

    def _record_increment(self, index: int) -> None:
        """Recover one Kvaerno velocity increment from the converged reduced root."""
        self.launch(
            self.algebra.implicit_increment,
            self.owner.n,
            self.stage_states[index],
            self.stages[index].output,
            self.cfg,
            self.increment_u[index + 1],
            self.increment_v[index + 1],
        )

    def finish(self) -> None:
        """Commit the tableau displacement and full-step work, or roll back the whole step."""
        self.launch(
            self.algebra.final_inputs,
            max(self.owner.n, self.owner.channels, LEDGER_WIDTH),
            self.state,
            self.stages[-1].output,
            *self.increment_u,
            self.cfg,
            self.displacement,
            self.work_state,
            self.final_q,
        )
        self.launch(
            self.k.actuator, self.owner.channels, self.state, self.command, self.cfg, self.next_a
        )
        self.work.zero_()
        self.launch(
            self.k.commit_nodes,
            self.owner.n,
            self.model,
            self.work_state,
            self.cfg,
            self.final_q,
            self.code,
            self.output,
            self.work,
        )
        self.launch(
            self.k.work_kernel,
            self.owner.ne,
            self.model,
            self.work_state,
            self.cfg,
            self.final_q,
            self.next_a,
            self.code,
            self.work,
        )
        self.launch(self.algebra.restore_failed, self.owner.n, self.state, self.code, self.output)
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
        self, record_derivatives: bool = False, solved_root_loader: RootLoader | None = None
    ) -> SolidState:
        """Run all native stages; optional saved roots are rechecked by each solve."""
        if solved_root_loader is not None and not record_derivatives:
            raise ValueError("A saved root is only supported for reverse reconstruction")
        if self.device.is_cuda and not self.device.is_capturing:
            raise RuntimeError("CUDA stepping requires an outer graph capture")
        if solved_root_loader is not None:
            solved_root_loader()
            wp.copy(self.saved_code, self.code)
        wp.copy(self.code, self.state.status)
        self.counts.zero_()
        self.residual_norm.zero_()
        self.max_linear.zero_()
        self._setup_tapes, self._increment_tapes = [], []
        self._initial_tape = self._tape(self._explicit_stage, record_derivatives)
        for index, stage in enumerate(self.stages):
            setup = self._tape(lambda: self._setup_stage(index), record_derivatives)

            def restore_root() -> None:
                """Load this stage's root while preserving the other packed roots."""
                wp.copy(stage.q, self.q, src_offset=index * self.owner.n, count=self.owner.n)
                wp.copy(stage.code, self.saved_code)

            stage.forward(
                record_derivatives=record_derivatives,
                solved_root_loader=restore_root if solved_root_loader is not None else None,
            )
            wp.copy(self.q, stage.q, dest_offset=index * self.owner.n, count=self.owner.n)
            self.launch(
                self.algebra.merge_diagnostics,
                1,
                stage.code,
                stage.counts,
                stage.residual_norm,
                stage.max_linear,
                self.code,
                self.counts,
                self.residual_norm,
                self.max_linear,
            )
            increment = self._tape(lambda: self._record_increment(index), record_derivatives)
            if setup is not None and increment is not None:
                self._setup_tapes.append(setup)
                self._increment_tapes.append(increment)
        self._final_tape = self._tape(self.finish, record_derivatives)
        return self.output

    def backward(self, seeds: dict[wp.array, wp.array], *, clear: bool = True) -> wp.array:
        """Propagate the tableau chain rule through three implicit native VJPs.

        Seeds may include ``displacement`` in addition to output-state leaves.
        This preserves gradients of travel without replacing it by dt*final_v.
        """
        if self.device.is_cuda and not self.device.is_capturing:
            raise RuntimeError("CUDA reverse evaluation requires an outer graph capture")
        if self._initial_tape is None or self._final_tape is None:
            raise RuntimeError("Call forward(record_derivatives=True) before backward")
        if clear:
            self.clear_gradients()
        self.sums.zero_()
        self._final_tape.backward(grads=seeds)
        for index in reversed(range(3)):
            self._increment_tapes[index].backward()
            stage = self.stages[index]
            stage.backward(
                {
                    getattr(stage.output, name): getattr(stage.output, name).grad
                    for name in STATE_FIELDS
                    if name != "status"
                },
                clear=False,
            )
            self.launch(self.algebra.merge_adjoint, 1, stage.sums, self.sums)
            self._setup_tapes[index].backward()
        self._initial_tape.backward()
        return self.command.grad

    def gradient_arrays(self) -> Iterator[wp.array]:
        """Yield unique model, input, stage, output and algebra cotangent storage."""
        seen: set[int] = set()
        candidates = list(value for stage in self.stages for value in stage.gradient_arrays())
        candidates.extend(value for value in vars(self).values() if isinstance(value, wp.array))
        candidates.extend(self.increment_u + self.increment_v)
        for state in (self.state, self.output, self.work_state):
            candidates.extend(getattr(state, name) for name in STATE_FIELDS)
        for value in candidates:
            if value.grad is not None and value.ptr not in seen:
                seen.add(value.ptr)
                yield value

    def diagnostics(self) -> Report:
        """Download aggregate stage diagnostics outside graph capture."""
        if self.device.is_capturing:
            raise RuntimeError("Host diagnostics are prohibited during graph capture")
        return dict(
            physics="spring",
            integrator="kvaerno3_native",
            stages=3,
            code=int(self.code.numpy()[0]),
            residual=float(self.residual_norm.numpy()[0]),
            max_linear_residual=float(self.max_linear.numpy()[0]),
            iterations=int(self.counts.numpy()[COUNT_NEWTON_ITERATIONS]),
            linear_solves=int(self.counts.numpy()[COUNT_LINEAR_SOLVES]),
            backtracks=int(self.counts.numpy()[COUNT_BACKTRACKS]),
            dofs=self.owner.n * 2,
            stage_diagnostics=[stage.diagnostics() for stage in self.stages],
        )
