"""Warp stage algebra for fixed-step spring Kvaerno3, independent of JAX.

The implicit stages reuse the native spring force and root adjoint. These
kernels connect those stages and retain the actual integration displacement.
"""

from functools import lru_cache
from types import SimpleNamespace

import warp as wp

from ...contracts.metrics import LEDGER_WIDTH, NEWTON_COUNT_WIDTH
from ...contracts.status_codes import ROOT_OK
from ..numerics.integration_tableaux import KVAERNO3_A, KVAERNO3_GAMMA
from .solid_kernels import solid_kernels


@lru_cache(maxsize=2)
def spring_kvaerno_kernels(precision: "str") -> "SimpleNamespace":
    """Return precision-specific differentiable stage algebra and diagnostics."""
    k = solid_kernels("spring", precision)
    real, vec = k.real, k.vec
    State, Model, Parameters = k.State, k.Model, k.Parameters
    gamma = wp.constant(real(KVAERNO3_GAMMA))
    b0 = wp.constant(real(KVAERNO3_A[-1][0]))
    b1 = wp.constant(real(KVAERNO3_A[-1][1]))
    b2 = wp.constant(real(KVAERNO3_A[-1][2]))
    infinity = wp.constant(real(float("inf")))
    module = f"envs.soft_walker.engine.warp_native.spring_kvaerno_{precision}_v1"

    @wp.kernel(module=module)
    def initial_velocity(m: Model, s: State, p: Parameters, q: wp.array(dtype=vec)):
        i = wp.tid()
        value = vec()
        if m.mass[i] > real(0):
            value = wp.cw_mul(s.v[i], m.free[i]) / p.velocity_ref
        q[i] = value

    @wp.kernel(module=module)
    def explicit_increment(
        m: Model,
        s: State,
        p: Parameters,
        zero_dt: Parameters,
        q: wp.array(dtype=vec),
        force: wp.array(dtype=vec),
        du: wp.array(dtype=vec),
        dv: wp.array(dtype=vec),
    ):
        i = wp.tid()
        x, v = vec(), vec()
        if m.mass[i] > real(0):
            contact, energy, dissipation = k.contact_value(i, q[i], m, s, zero_dt)
            x = p.dt * p.velocity_ref * q[i]
            v = p.dt * wp.cw_mul(force[i] + contact, m.free[i]) / m.mass[i]
        du[i] = x
        dv[i] = v

    @wp.kernel(module=module)
    def stage_input(
        original: State,
        u0: wp.array(dtype=vec),
        v0: wp.array(dtype=vec),
        u1: wp.array(dtype=vec),
        v1: wp.array(dtype=vec),
        u2: wp.array(dtype=vec),
        v2: wp.array(dtype=vec),
        a0: real,
        a1: real,
        a2: real,
        before: real,
        command: wp.array(dtype=real),
        p: Parameters,
        status: wp.array(dtype=int),
        out: State,
    ):
        i = wp.tid()
        if i < original.u.shape[0]:
            u, v = original.u[i] + a0 * u0[i], original.v[i] + a0 * v0[i]
            if a1 != real(0):
                u, v = u + a1 * u1[i], v + a1 * v1[i]
            if a2 != real(0):
                u, v = u + a2 * u2[i], v + a2 * v2[i]
            out.u[i] = u
            out.v[i] = v
            out.rotation[i] = original.rotation[i]
            out.omega[i] = original.omega[i]
        if i < original.actuator.shape[0]:
            out.actuator[i] = command[i] + (original.actuator[i] - command[i]) * wp.exp(
                -before * p.dt / p.tau
            )
        if i < LEDGER_WIDTH:
            out.ledger[i] = original.ledger[i]
        if i == 0:
            out.status[0] = status[0]

    @wp.kernel(module=module)
    def implicit_increment(
        base: State,
        accepted: State,
        p: Parameters,
        du: wp.array(dtype=vec),
        dv: wp.array(dtype=vec),
    ):
        i = wp.tid()
        du[i] = p.dt * accepted.v[i]
        dv[i] = (accepted.v[i] - base.v[i]) / gamma

    @wp.kernel(module=module)
    def final_inputs(
        original: State,
        last: State,
        u0: wp.array(dtype=vec),
        u1: wp.array(dtype=vec),
        u2: wp.array(dtype=vec),
        u3: wp.array(dtype=vec),
        p: Parameters,
        displacement: wp.array(dtype=vec),
        work_state: State,
        velocity: wp.array(dtype=vec),
    ):
        i = wp.tid()
        if i < original.u.shape[0]:
            dx = b0 * u0[i]
            dx = dx + b1 * u1[i]
            dx = dx + b2 * u2[i]
            dx = dx + gamma * u3[i]
            displacement[i] = dx
            # Reuse the exact native force/work law at the accepted final
            # position while its node_value helper expects u + dt * v.
            work_state.u[i] = original.u[i] + dx - p.dt * last.v[i]
            work_state.v[i] = original.v[i]
            work_state.rotation[i] = original.rotation[i]
            work_state.omega[i] = original.omega[i]
            velocity[i] = last.v[i] / p.velocity_ref
        if i < original.actuator.shape[0]:
            work_state.actuator[i] = original.actuator[i]
        if i < LEDGER_WIDTH:
            work_state.ledger[i] = original.ledger[i]
        if i == 0:
            work_state.status[0] = original.status[0]

    @wp.kernel(module=module)
    def restore_failed(original: State, code: wp.array(dtype=int), out: State):
        i = wp.tid()
        if code[0] != ROOT_OK:
            out.u[i] = original.u[i]
            out.v[i] = original.v[i]
            out.rotation[i] = original.rotation[i]
            out.omega[i] = original.omega[i]

    @wp.kernel(module=module, enable_backward=False)
    def merge_diagnostics(
        stage_code: wp.array(dtype=int),
        stage_counts: wp.array(dtype=int),
        stage_residual: wp.array(dtype=real),
        stage_linear: wp.array(dtype=real),
        code: wp.array(dtype=int),
        counts: wp.array(dtype=int),
        residual: wp.array(dtype=real),
        linear: wp.array(dtype=real),
    ):
        if code[0] == ROOT_OK:
            code[0] = stage_code[0]
        for i in range(NEWTON_COUNT_WIDTH):
            counts[i] = counts[i] + stage_counts[i]
        residual[0] = wp.max(residual[0], stage_residual[0] / gamma)
        linear[0] = wp.max(linear[0], stage_linear[0])

    @wp.kernel(module=module, enable_backward=False)
    def merge_adjoint(stage: wp.array(dtype=real), out: wp.array(dtype=real)):
        denominator = real(1)
        if stage[1] > real(0):
            denominator = stage[1]
        ratio = stage[0] / denominator
        if not wp.isfinite(ratio):
            out[0] = infinity
        else:
            out[0] = wp.max(out[0], ratio)
        out[1] = real(1)

    @wp.kernel(module=module, enable_backward=False)
    def accumulate_displacement(
        model: Model,
        accepted: State,
        displacement: wp.array(dtype=vec),
        travel: wp.array(dtype=vec),
        clock: wp.array(dtype=int),
        settling: int,
    ):
        i = wp.tid()
        if accepted.status[0] == 0 and model.mass[i] > real(0) and clock[0] >= settling:
            travel[i] = travel[i] + displacement[i]

    @wp.kernel(module=module, enable_backward=False)
    def seed_displacement(
        model: Model,
        travel_gradient: wp.array(dtype=vec),
        seed: wp.array(dtype=vec),
        clock: wp.array(dtype=int),
        settling: int,
    ):
        i = wp.tid()
        value = vec()
        if model.mass[i] > real(0) and clock[0] >= settling:
            value = travel_gradient[i]
        seed[i] = value

    return SimpleNamespace(**locals())
