"""Device-resident episode bookkeeping and block-checkpoint storage."""

from functools import lru_cache
from types import SimpleNamespace

import warp as wp

from ...contracts.metrics import (
    COUNT_BACKTRACKS,
    COUNT_NEWTON_ITERATIONS,
    DIAGNOSTIC_BACKTRACKS,
    DIAGNOSTIC_LINEAR_RESIDUAL,
    DIAGNOSTIC_LIVE_NODES,
    DIAGNOSTIC_NEWTON_ITERATIONS,
    DIAGNOSTIC_RESIDUAL,
    DIAGNOSTIC_STATUS,
    LEDGER_WIDTH,
    LEDGER_WORK_POSITIVE,
)
from ...contracts.status_codes import ADJOINT_REPLAY_FAILED, ADJOINT_REVERSE_FAILED
from .kernels import kernels


@lru_cache(maxsize=4)
def episode_kernels(dim: "int", precision: "str") -> "SimpleNamespace":
    """Return the cached MPM checkpoint, command, objective and reverse-clock namespace.

    Types and kernel identities depend on the requested physics/dimension and
    precision. Warp DSL annotations must be evaluated in this factory's
    scope; postponed annotations and relocation change kernel construction."""
    k = kernels(dim, precision)
    real, vec, mat, State, Model = k.real, k.vec, k.mat, k.State, k.Model
    module = f"envs.soft_walker.engine.warp_native.episode_d{dim}_{precision}_v1"
    nan = wp.constant(real(float("nan")))
    D = dim

    def struct(cls: "type") -> "wp.codegen.Struct":
        """Bind a struct to this precision-specific kernel module and disable AD defaults."""
        return wp.struct(cls, module=module)

    @struct
    class History:
        x: wp.array2d(dtype=vec)
        v: wp.array2d(dtype=vec)
        F: wp.array2d(dtype=mat)
        C: wp.array2d(dtype=mat)
        actuator: wp.array2d(dtype=real)
        ledger: wp.array2d(dtype=real)
        status: wp.array(dtype=int)

    @wp.kernel(module=module, enable_backward=False)
    def copy_state(source: State, destination: State):
        p = wp.tid()
        if p < source.x.shape[0]:
            destination.x[p] = source.x[p]
            destination.v[p] = source.v[p]
            destination.F[p] = source.F[p]
            destination.C[p] = source.C[p]
        if p < source.actuator.shape[0]:
            destination.actuator[p] = source.actuator[p]
        if p < LEDGER_WIDTH:
            destination.ledger[p] = source.ledger[p]
        if p == 0:
            destination.status[0] = source.status[0]

    @wp.kernel(module=module, enable_backward=False)
    def add_state(source: State, destination: State):
        p = wp.tid()
        if p < source.x.shape[0]:
            destination.x[p] = destination.x[p] + source.x[p]
            destination.v[p] = destination.v[p] + source.v[p]
            destination.F[p] = destination.F[p] + source.F[p]
            destination.C[p] = destination.C[p] + source.C[p]
        if p < source.actuator.shape[0]:
            destination.actuator[p] = destination.actuator[p] + source.actuator[p]
        if p < LEDGER_WIDTH:
            destination.ledger[p] = destination.ledger[p] + source.ledger[p]

    @wp.kernel(module=module, enable_backward=False)
    def store(
        history: History,
        state: State,
        index: wp.array(dtype=int),
        stride: int,
        modulo: int,
        conditional: int,
    ):
        p = wp.tid()
        time = index[0]
        if conditional == 0 or time % stride == 0:
            row = time // stride
            if modulo > 0:
                row = time % modulo
            if p < state.x.shape[0]:
                history.x[row, p] = state.x[p]
                history.v[row, p] = state.v[p]
                history.F[row, p] = state.F[p]
                history.C[row, p] = state.C[p]
            if p < state.actuator.shape[0]:
                history.actuator[row, p] = state.actuator[p]
            if p < LEDGER_WIDTH:
                history.ledger[row, p] = state.ledger[p]
            if p == 0:
                history.status[row] = state.status[0]

    @wp.kernel(module=module, enable_backward=False)
    def restore(history: History, state: State, index: wp.array(dtype=int), modulo: int):
        p = wp.tid()
        row = index[0]
        if modulo > 0:
            row = row % modulo
        if p < state.x.shape[0]:
            state.x[p] = history.x[row, p]
            state.v[p] = history.v[row, p]
            state.F[p] = history.F[row, p]
            state.C[p] = history.C[row, p]
        if p < state.actuator.shape[0]:
            state.actuator[p] = history.actuator[row, p]
        if p < LEDGER_WIDTH:
            state.ledger[p] = history.ledger[row, p]
        if p == 0:
            state.status[0] = history.status[row]

    @wp.kernel(module=module, enable_backward=False)
    def load_command(
        table: wp.array2d(dtype=real),
        time: wp.array(dtype=int),
        settling: int,
        substeps: int,
        command: wp.array(dtype=real),
    ):
        c = wp.tid()
        value = real(0)
        if time[0] >= settling:
            value = table[(time[0] - settling) // substeps, c]
        command[c] = value

    @wp.kernel(module=module, enable_backward=False)
    def add_command_gradient(
        command: wp.array(dtype=real),
        time: wp.array(dtype=int),
        settling: int,
        substeps: int,
        gradient: wp.array2d(dtype=real),
    ):
        c = wp.tid()
        if time[0] >= settling:
            row = (time[0] - settling) // substeps
            gradient[row, c] = gradient[row, c] + command[c]

    @wp.kernel(module=module, enable_backward=False)
    def advance_clock(
        time: wp.array(dtype=int), limit: int, state: State, condition: wp.array(dtype=int)
    ):
        time[0] = time[0] + 1
        condition[0] = int(time[0] < limit and state.status[0] == 0)

    @wp.kernel(module=module, enable_backward=False)
    def record_diag(
        time: wp.array(dtype=int),
        code: wp.array(dtype=int),
        counts: wp.array(dtype=int),
        residual: wp.array(dtype=real),
        linear: wp.array(dtype=real),
        live: wp.array(dtype=int),
        output: wp.array2d(dtype=real),
    ):
        t = time[0]
        output[t, DIAGNOSTIC_STATUS] = real(code[0])
        output[t, DIAGNOSTIC_NEWTON_ITERATIONS] = real(counts[COUNT_NEWTON_ITERATIONS])
        output[t, DIAGNOSTIC_BACKTRACKS] = real(counts[COUNT_BACKTRACKS])
        output[t, DIAGNOSTIC_RESIDUAL] = residual[0]
        output[t, DIAGNOSTIC_LINEAR_RESIDUAL] = linear[0]
        output[t, DIAGNOSTIC_LIVE_NODES] = real(live[0])

    @wp.kernel(module=module, enable_backward=False)
    def begin_reverse(
        block: wp.array(dtype=int),
        total: int,
        stride: int,
        condition: wp.array(dtype=int),
        state: State,
        adjoint_status: wp.array(dtype=int),
    ):
        adjoint_status[0] = state.status[0]
        block[0] = (total - 1) // stride
        condition[0] = int(state.status[0] == 0)

    @wp.kernel(module=module, enable_backward=False)
    def begin_replay(
        time: wp.array(dtype=int),
        block: wp.array(dtype=int),
        stride: int,
        condition: wp.array(dtype=int),
    ):
        time[0] = block[0] * stride
        condition[0] = 1

    @wp.kernel(module=module, enable_backward=False)
    def advance_replay(
        time: wp.array(dtype=int),
        block: wp.array(dtype=int),
        stride: int,
        total: int,
        condition: wp.array(dtype=int),
        state: State,
    ):
        time[0] = time[0] + 1
        condition[0] = int(
            time[0] < wp.min(total, (block[0] + 1) * stride) and state.status[0] == 0
        )

    @wp.kernel(module=module, enable_backward=False)
    def begin_block_reverse(
        time: wp.array(dtype=int),
        block: wp.array(dtype=int),
        stride: int,
        total: int,
        condition: wp.array(dtype=int),
        replay_state: State,
        adjoint_status: wp.array(dtype=int),
    ):
        # A stopped replay has not populated the intended scratch block.
        if replay_state.status[0] != 0:
            adjoint_status[0] = ADJOINT_REPLAY_FAILED
        time[0] = wp.min(total, (block[0] + 1) * stride) - 1
        condition[0] = int(adjoint_status[0] == 0)

    @wp.kernel(module=module, enable_backward=False)
    def reverse_clock(
        time: wp.array(dtype=int),
        block: wp.array(dtype=int),
        stride: int,
        condition: wp.array(dtype=int),
        adjoint_status: wp.array(dtype=int),
    ):
        time[0] = time[0] - 1
        condition[0] = int(time[0] >= block[0] * stride and adjoint_status[0] == 0)

    @wp.kernel(module=module, enable_backward=False)
    def reverse_block(
        block: wp.array(dtype=int),
        condition: wp.array(dtype=int),
        adjoint_status: wp.array(dtype=int),
    ):
        block[0] = block[0] - 1
        condition[0] = int(block[0] >= 0 and adjoint_status[0] == 0)

    @wp.kernel(module=module, enable_backward=False)
    def check_reverse_step(
        sums: wp.array(dtype=real),
        code: wp.array(dtype=int),
        limit: real,
        adjoint_status: wp.array(dtype=int),
    ):
        denominator = sums[1] if sums[1] > real(0) else real(1)
        error = wp.sqrt(sums[0] / denominator)
        if code[0] != 0 or not wp.isfinite(error) or error > limit:
            adjoint_status[0] = ADJOINT_REVERSE_FAILED

    @wp.kernel(module=module, enable_backward=False)
    def accumulate_travel(model: Model, accepted: State, dt: real, travel: wp.array(dtype=vec)):
        p = wp.tid()
        if accepted.status[0] == 0 and model.alpha[p] > real(0):
            # Use the actual integration increment, not subtraction of nearby
            # rounded absolute coordinates. Physics/state updates are unchanged.
            travel[p] = travel[p] + dt * accepted.v[p]

    @wp.kernel(module=module, enable_backward=False)
    def seed_travel_velocity(
        model: Model, travel_gradient: wp.array(dtype=vec), dt: real, rolling: State
    ):
        p = wp.tid()
        if model.alpha[p] > real(0):
            rolling.v[p] = rolling.v[p] + dt * travel_gradient[p]

    @wp.kernel(module=module)
    def objective_particles(
        model: Model,
        initial: State,
        final: State,
        travel: wp.array(dtype=vec),
        direction: vec,
        length_ref: real,
        velocity_weight: real,
        sums: wp.array(dtype=real),
    ):
        p = wp.tid()
        mass = model.alpha[p] * model.density[p] * model.volume[p]
        value = -wp.dot(direction, travel[p]) / length_ref + velocity_weight * wp.dot(
            final.v[p], final.v[p]
        )
        wp.atomic_add(sums, 0, mass * value)
        wp.atomic_add(sums, 1, mass)

    @wp.kernel(module=module)
    def objective_finish(
        sums: wp.array(dtype=real), final: State, work_weight: real, loss: wp.array(dtype=real)
    ):
        if final.status[0] == 0 and sums[1] > real(0):
            loss[0] = sums[0] / sums[1] + work_weight * final.ledger[LEDGER_WORK_POSITIVE]
        else:
            loss[0] = nan

    @wp.kernel(module=module, enable_backward=False)
    def add_model(source: Model, target: Model):
        p = wp.tid()
        if p < source.alpha.shape[0]:
            target.reference[p] = target.reference[p] + source.reference[p]
            target.volume[p] = target.volume[p] + source.volume[p]
            target.alpha[p] = target.alpha[p] + source.alpha[p]
            target.young[p] = target.young[p] + source.young[p]
            target.poisson[p] = target.poisson[p] + source.poisson[p]
            target.density[p] = target.density[p] + source.density[p]
            target.strength[p] = target.strength[p] + source.strength[p]
            target.direction[p] = target.direction[p] + source.direction[p]
            for face in range(2 * D):
                target.contact_area[p, face] = (
                    target.contact_area[p, face] + source.contact_area[p, face]
                )
        if p < 2 * D:
            target.face_offset[p] = target.face_offset[p] + source.face_offset[p]

    @wp.kernel(module=module, enable_backward=False)
    def poison_gradient(
        state: State, success: wp.array(dtype=int), command: wp.array2d(dtype=real)
    ):
        p = wp.tid()
        if success[0] != 0:
            if p < state.x.shape[0]:
                state.x[p] = vec(nan)
                state.v[p] = vec(nan)
                state.F[p] = mat(nan)
                state.C[p] = mat(nan)
            if p < command.shape[0] * command.shape[1]:
                command[p // command.shape[1], p % command.shape[1]] = nan

    @wp.kernel(module=module, enable_backward=False)
    def save_root(
        q: wp.array(dtype=vec),
        code: wp.array(dtype=int),
        clock: wp.array(dtype=int),
        roots: wp.array2d(dtype=vec),
        codes: wp.array(dtype=int),
    ):
        i = wp.tid()
        slot = clock[0] % roots.shape[0]
        roots[slot, i] = q[i]
        if i == 0:
            codes[slot] = code[0]

    @wp.kernel(module=module, enable_backward=False)
    def load_root(
        roots: wp.array2d(dtype=vec),
        codes: wp.array(dtype=int),
        clock: wp.array(dtype=int),
        q: wp.array(dtype=vec),
        code: wp.array(dtype=int),
    ):
        i = wp.tid()
        slot = clock[0] % roots.shape[0]
        q[i] = roots[slot, i]
        if i == 0:
            code[0] = codes[slot]

    return SimpleNamespace(**locals())
