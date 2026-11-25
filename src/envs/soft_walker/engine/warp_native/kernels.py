"""Fused particle operators and compact-grid operations for implicit MPM.

Each particle thread gathers its quadratic support, computes deformation,
material/contact response, and scatters the resulting contribution. Global
accumulation and a completed grid solve are true dependency boundaries. No
Python/NumPy material evaluation takes place in these kernels.
"""

from functools import lru_cache
from types import SimpleNamespace

import warp as wp

from ...contracts.metrics import (
    LEDGER_DISSIPATED,
    LEDGER_TIME,
    LEDGER_WIDTH,
    LEDGER_WORK_POSITIVE,
    LEDGER_WORK_SIGNED,
    WORK_FORWARD_TANGENT,
    WORK_INCREMENT_WIDTH,
    WORK_PRIMAL_RESIDUAL,
    WORK_SUPPORT_BUILD,
)
from ...contracts.status_codes import BINARY_STATE_FAILED, BINARY_STATE_OK, ROOT_OK
from .local import functions
from .work import positive_work_function


@lru_cache(maxsize=4)
def kernels(dim: "int", precision: "str") -> "SimpleNamespace":
    """Return the cached MPM material, support-transfer, contact and residual namespace.

    Types and kernel identities depend on the requested physics/dimension and
    precision. Warp DSL annotations must be evaluated in this factory's
    scope; postponed annotations and relocation change kernel construction."""
    local = functions(dim, precision)
    real, vec, mat = local.scalar, local.vector, local.matrix
    positive_work = positive_work_function(precision)
    ivec = wp.types.vector(length=dim, dtype=wp.int32)
    D = dim
    S = 3**dim
    FACES = 2 * dim
    eye = wp.constant(mat(*[float(i == j) for i in range(dim) for j in range(dim)]))
    module = f"envs.soft_walker.engine.warp_native.mpm_d{dim}_{precision}_v1"

    def struct(cls: "type") -> "wp.codegen.Struct":
        """Bind a struct to this precision-specific kernel module and disable AD defaults."""
        return wp.struct(cls, module=module)

    @struct
    class Model:
        reference: wp.array(dtype=vec)
        volume: wp.array(dtype=real)
        alpha: wp.array(dtype=real)
        young: wp.array(dtype=real)
        poisson: wp.array(dtype=real)
        density: wp.array(dtype=real)
        strength: wp.array(dtype=real)
        direction: wp.array(dtype=vec)
        kind: wp.array(dtype=int)
        channel: wp.array(dtype=int)
        face_offset: wp.array(dtype=vec)
        contact_area: wp.array2d(dtype=real)
        plane_normal: vec
        plane_offset: real
        box_centers: wp.array(dtype=vec)
        box_half: wp.array(dtype=vec)
        box_rounding: wp.array(dtype=real)
        box_active: wp.array(dtype=int)

    @struct
    class State:
        x: wp.array(dtype=vec)
        v: wp.array(dtype=vec)
        F: wp.array(dtype=mat)
        C: wp.array(dtype=mat)
        actuator: wp.array(dtype=real)
        ledger: wp.array(dtype=real)  # signed work, positive work, dissipated, time
        status: wp.array(dtype=int)

    @struct
    class Parameters:
        shape: ivec
        origin: vec
        h: real
        dt: real
        gravity: real
        axis: int
        velocity_ref: real
        tau: real
        log_stretch: real
        stress_ratio: real
        stress_mode: int
        weighted: int
        min_J: real
        contact: int
        stiffness: real
        width: real
        friction: real
        velocity_width: real
        atol: real
        linear_limit: real
        max_newton: int
        max_backtracks: int
        armijo: real

    @struct
    class Support:
        ids: wp.array2d(dtype=int)
        weight: wp.array2d(dtype=real)
        gradient: wp.array2d(dtype=vec)
        offset: wp.array2d(dtype=vec)
        valid: wp.array(dtype=int)
        mass: wp.array(dtype=real)
        momentum: wp.array(dtype=vec)
        predictor: wp.array(dtype=vec)
        scale: wp.array(dtype=real)
        diagonal: wp.array(dtype=real)
        totals: wp.array(dtype=real)  # mass sum, number of positive-mass nodes
        live: wp.array(dtype=int)
        work_counts: wp.array(
            dtype=int
        )  # primal/transpose Jv, cache builds, residuals, support builds

    @wp.func(module=module)
    def choice(slot: int):
        result = ivec()
        cursor = slot
        for reverse in range(D):
            axis = D - reverse - 1
            result[axis] = cursor % 3
            cursor = cursor // 3
        return result

    @wp.func(module=module)
    def basis(f: real, k: int):
        value = real(0)
        derivative = real(0)
        if k == 0:
            value = real(0.5) * (real(1.5) - f) ** real(2)
            derivative = f - real(1.5)
        elif k == 1:
            value = real(0.75) - (f - real(1)) ** real(2)
            derivative = real(2) * (real(1) - f)
        else:
            value = real(0.5) * (f - real(0.5)) ** real(2)
            derivative = f - real(0.5)
        return value, derivative

    @wp.func(module=module)
    def support_valid(x: vec, cfg: Parameters):
        good = int(1)
        for axis in range(D):
            base = int(wp.floor((x[axis] - cfg.origin[axis]) / cfg.h - real(0.5)))
            if base < 0 or base + 2 >= cfg.shape[axis]:
                good = 0
        return good

    @wp.kernel(module=module)
    def build_stencil(
        model: Model,
        state: State,
        cfg: Parameters,
        st: Support,
        keys: wp.array(dtype=int),
        order: wp.array(dtype=int),
    ):
        p = wp.tid()
        if p == 0:
            wp.atomic_add(st.work_counts, WORK_SUPPORT_BUILD, 1)
        active = model.alpha[p] * model.density[p] * model.volume[p] > real(0)
        x = model.reference[p]
        if active:
            x = state.x[p]
        u = (x - cfg.origin) / cfg.h
        base = ivec()
        f = vec()
        valid = int(1)
        for axis in range(D):
            base[axis] = int(wp.floor(u[axis] - real(0.5)))
            f[axis] = u[axis] - real(base[axis])
            if base[axis] < 0 or base[axis] + 2 >= cfg.shape[axis]:
                valid = 0
        st.valid[p] = valid if active else 1
        for slot in range(S):
            ch = choice(slot)
            weights = vec()
            deriv = vec()
            offset = vec()
            key = int(0)
            for axis in range(D):
                w, dw = basis(f[axis], ch[axis])
                weights[axis] = w
                deriv[axis] = dw / cfg.h
                node = base[axis] + ch[axis]
                key = key * cfg.shape[axis] + wp.clamp(node, 0, cfg.shape[axis] - 1)
                offset[axis] = (real(node) - u[axis]) * cfg.h
            weight = real(1)
            for axis in range(D):
                weight = weight * weights[axis]
            gradient = vec()
            for axis in range(D):
                value = deriv[axis]
                for other in range(D):
                    if other != axis:
                        value = value * weights[other]
                gradient[axis] = value
            st.weight[p, slot] = weight
            st.gradient[p, slot] = gradient
            st.offset[p, slot] = offset
            index = p * S + slot
            keys[index] = key if active else 2147483647
            order[index] = index

    @wp.kernel(module=module, enable_backward=False)
    def mark_unique(keys: wp.array(dtype=int), flags: wp.array(dtype=int)):
        i = wp.tid()
        is_new = keys[i] < 2147483647
        if i > 0:
            is_new = is_new and keys[i] != keys[i - 1]
        flags[i] = int(is_new)

    @wp.kernel(module=module, enable_backward=False)
    def assign_rows(
        keys: wp.array(dtype=int),
        order: wp.array(dtype=int),
        prefix: wp.array(dtype=int),
        count: int,
        st: Support,
    ):
        i = wp.tid()
        row = int(0)
        if keys[i] < 2147483647:
            row = prefix[i] - 1
        original = order[i]
        if row >= st.mass.shape[0]:
            # Keep every subsequent gather/scatter memory-safe. Marking the
            # particle invalid makes the whole step reject transactionally;
            # support is never truncated into an apparently valid solution.
            wp.atomic_min(st.valid, original // S, 0)
            row = 0
        st.ids[original // S, original % S] = row
        if i == count - 1:
            st.live[0] = prefix[i]

    @wp.kernel(module=module)
    def p2g(model: Model, state: State, st: Support):
        p = wp.tid()
        mass = model.alpha[p] * model.volume[p] * model.density[p]
        if mass > real(0):
            for slot in range(S):
                row = st.ids[p, slot]
                weighted_mass = mass * st.weight[p, slot]
                velocity = state.v[p] + state.C[p] * st.offset[p, slot]
                wp.atomic_add(st.mass, row, weighted_mass)
                wp.atomic_add(st.momentum, row, weighted_mass * velocity)

    @wp.kernel(module=module)
    def mass_totals(st: Support):
        i = wp.tid()
        if st.mass[i] > real(0):
            wp.atomic_add(st.totals, 0, st.mass[i])
            wp.atomic_add(st.totals, 1, real(1))

    @wp.kernel(module=module)
    def prepare_grid(st: Support, cfg: Parameters, guess: wp.array(dtype=vec)):
        i = wp.tid()
        predictor = vec()
        scale = real(1)
        if st.mass[i] > real(0):
            predictor = st.momentum[i] / st.mass[i]
            if cfg.weighted != 0:
                average = st.totals[0] / wp.max(st.totals[1], real(1))
                scale = wp.sqrt(st.mass[i] / wp.max(average, real(1.0e-30)))
        st.predictor[i] = predictor
        st.scale[i] = scale
        guess[i] = predictor * scale / cfg.velocity_ref

    @wp.kernel(module=module)
    def actuator(
        state: State, command: wp.array(dtype=real), cfg: Parameters, next_a: wp.array(dtype=real)
    ):
        c = wp.tid()
        value = command[c]
        if cfg.tau > real(0):
            value = command[c] + (state.actuator[c] - command[c]) * wp.exp(-cfg.dt / cfg.tau)
        next_a[c] = value

    @wp.func(module=module)
    def gather(p: int, st: Support, cfg: Parameters, q: wp.array(dtype=vec), transpose: int):
        velocity = vec()
        gradient = mat()
        affine = mat()
        for slot in range(S):
            row = st.ids[p, slot]
            factor = cfg.velocity_ref / st.scale[row]
            if transpose != 0:
                mass = st.mass[row] if st.mass[row] > real(0) else real(1)
                factor = st.scale[row] / (mass * cfg.velocity_ref)
            gv = q[row] * factor
            velocity = velocity + st.weight[p, slot] * gv
            gradient = gradient + wp.outer(gv, st.gradient[p, slot])
            affine = affine + st.weight[p, slot] * wp.outer(gv, st.offset[p, slot])
        return velocity, gradient, real(4) / (cfg.h * cfg.h) * affine

    @wp.func(module=module)
    def terrain(x: vec, model: Model):
        gap = wp.dot(x, model.plane_normal) - model.plane_offset
        normal = model.plane_normal
        hessian = mat()
        for box in range(model.box_rounding.shape[0]):
            if model.box_active[box] != 0:
                displacement = x - model.box_centers[box]
                q = vec()
                outside = vec()
                sign = vec()
                active = vec()
                face = int(0)
                for axis in range(D):
                    q[axis] = (
                        wp.abs(displacement[axis])
                        - model.box_half[box][axis]
                        + model.box_rounding[box]
                    )
                    outside[axis] = wp.max(q[axis], real(0))
                    sign[axis] = real(1) if displacement[axis] >= real(0) else real(-1)
                    active[axis] = real(1) if q[axis] > real(0) else real(0)
                    if q[axis] > q[face]:
                        face = axis
                length = wp.sqrt(wp.max(wp.dot(outside, outside), real(1.0e-24)))
                distance = q[face] - model.box_rounding[box]
                candidate_n = vec()
                candidate_n[face] = sign[face]
                candidate_h = mat()
                if q[face] > real(0):
                    distance = length - model.box_rounding[box]
                    candidate_n = wp.cw_mul(outside, sign) / length
                    candidate_h = (wp.diag(active) - wp.outer(candidate_n, candidate_n)) / length
                if distance < gap:
                    gap = distance
                    normal = candidate_n
                    hessian = candidate_h
        return gap, normal, hessian

    @wp.func(module=module)
    def contact(x: vec, velocity: vec, area: real, model: Model, cfg: Parameters):
        gap, normal, hessian = terrain(x, model)
        depth = cfg.width * local.softplus(-gap / cfg.width)
        gate = local.sigmoid(-gap / cfg.width)
        magnitude = cfg.stiffness * area * depth * gate
        vt = velocity - wp.dot(velocity, normal) * normal
        denominator = wp.sqrt(wp.dot(vt, vt) + cfg.velocity_width * cfg.velocity_width)
        friction = -cfg.friction * magnitude * vt / denominator
        return magnitude * normal + friction, -wp.dot(friction, vt)

    @wp.func(module=module)
    def contact_tangent(
        x: vec, velocity: vec, dx: vec, dv: vec, area: real, model: Model, cfg: Parameters
    ):
        gap, normal, hessian = terrain(x, model)
        depth = cfg.width * local.softplus(-gap / cfg.width)
        gate = local.sigmoid(-gap / cfg.width)
        magnitude = cfg.stiffness * area * depth * gate
        dnormal = hessian * dx
        dmagnitude = (
            -cfg.stiffness
            * area
            * (gate * gate + depth * gate * (real(1) - gate) / cfg.width)
            * wp.dot(normal, dx)
        )
        vn = wp.dot(velocity, normal)
        vt = velocity - vn * normal
        dvt = dv - (wp.dot(dv, normal) + wp.dot(velocity, dnormal)) * normal - vn * dnormal
        denominator = wp.sqrt(wp.dot(vt, vt) + cfg.velocity_width * cfg.velocity_width)
        force_change = dmagnitude * (normal - cfg.friction * vt / denominator)
        direction_change = (
            dnormal
            - cfg.friction * dvt / denominator
            + cfg.friction * vt * wp.dot(vt, dvt) / (denominator ** real(3))
        )
        return force_change + magnitude * direction_change

    @wp.func(module=module)
    def material_amount(p: int, model: Model, next_a: wp.array(dtype=real), cfg: Parameters):
        command = model.strength[p] * next_a[model.channel[p]]
        strain = cfg.log_stretch * command
        active_stress = real(0)
        if cfg.stress_mode != 0:
            strain = real(0)
            active_stress = (
                cfg.stress_ratio
                * model.young[p]
                / (real(2) * (real(1) + model.poisson[p]))
                * command
            )
        return strain, active_stress

    @wp.kernel(module=module)
    def force(
        model: Model,
        state: State,
        st: Support,
        cfg: Parameters,
        q: wp.array(dtype=vec),
        next_a: wp.array(dtype=real),
        output: wp.array(dtype=vec),
    ):
        p = wp.tid()
        if p == 0:
            wp.atomic_add(st.work_counts, WORK_PRIMAL_RESIDUAL, 1)
        if model.alpha[p] > real(0):
            velocity, gradient, affine = gather(p, st, cfg, q, 0)
            increment = cfg.dt * gradient * state.F[p]
            delta = state.F[p] - eye + increment
            amount, sigma = material_amount(p, model, next_a, cfg)
            P = local.stress(
                delta, model.young[p], model.poisson[p], amount, model.direction[p], model.kind[p]
            )
            active = sigma * local.projector(model.direction[p], model.kind[p])
            P = P + active + delta * active
            weighted_stress = -model.alpha[p] * model.volume[p] * P * wp.transpose(state.F[p])
            contact_force = vec()
            contact_moment = mat()
            if cfg.contact != 0:
                for face in range(FACES):
                    r = state.F[p] * model.face_offset[face]
                    marker_v = velocity + gradient * r
                    marker_x = state.x[p] + r + cfg.dt * marker_v
                    cf, diss = contact(marker_x, marker_v, model.contact_area[p, face], model, cfg)
                    contact_force = contact_force + cf
                    contact_moment = contact_moment + wp.outer(cf, r)
            for slot in range(S):
                contribution = (weighted_stress + contact_moment) * st.gradient[
                    p, slot
                ] + st.weight[p, slot] * contact_force
                wp.atomic_add(output, st.ids[p, slot], contribution)

    @wp.kernel(module=module, enable_backward=False)
    def force_tangent(
        model: Model,
        state: State,
        st: Support,
        cfg: Parameters,
        q: wp.array(dtype=vec),
        direction: wp.array(dtype=vec),
        next_a: wp.array(dtype=real),
        transpose: int,
        output: wp.array(dtype=vec),
    ):
        p = wp.tid()
        if p == 0:
            wp.atomic_add(st.work_counts, WORK_FORWARD_TANGENT + int(transpose != 0), 1)
        if model.alpha[p] > real(0):
            velocity, gradient, affine = gather(p, st, cfg, q, 0)
            dv, dgradient, daffine = gather(p, st, cfg, direction, transpose)
            delta = state.F[p] - eye + cfg.dt * gradient * state.F[p]
            increment = cfg.dt * dgradient * state.F[p]
            amount, sigma = material_amount(p, model, next_a, cfg)
            dP = local.tangent(
                delta,
                increment,
                model.young[p],
                model.poisson[p],
                amount,
                model.direction[p],
                model.kind[p],
            )
            dP = dP + increment * (sigma * local.projector(model.direction[p], model.kind[p]))
            weighted = -model.alpha[p] * model.volume[p] * dP * wp.transpose(state.F[p])
            contact_force = vec()
            contact_moment = mat()
            if cfg.contact != 0:
                for face in range(FACES):
                    r = state.F[p] * model.face_offset[face]
                    marker_v = velocity + gradient * r
                    marker_x = state.x[p] + r + cfg.dt * marker_v
                    marker_dv = dv + dgradient * r
                    cf = contact_tangent(
                        marker_x,
                        marker_v,
                        cfg.dt * marker_dv,
                        marker_dv,
                        model.contact_area[p, face],
                        model,
                        cfg,
                    )
                    if transpose != 0:
                        transposed_force = vec()
                        for axis in range(D):
                            basis_direction = vec()
                            basis_direction[axis] = real(1)
                            column = contact_tangent(
                                marker_x,
                                marker_v,
                                cfg.dt * basis_direction,
                                basis_direction,
                                model.contact_area[p, face],
                                model,
                                cfg,
                            )
                            transposed_force[axis] = wp.dot(column, marker_dv)
                        cf = transposed_force
                    contact_force = contact_force + cf
                    contact_moment = contact_moment + wp.outer(cf, r)
            for slot in range(S):
                contribution = (weighted + contact_moment) * st.gradient[p, slot] + st.weight[
                    p, slot
                ] * contact_force
                wp.atomic_add(output, st.ids[p, slot], contribution)

    @wp.kernel(module=module)
    def residual_grid(
        st: Support,
        cfg: Parameters,
        q: wp.array(dtype=vec),
        forces: wp.array(dtype=vec),
        residual: wp.array(dtype=vec),
    ):
        i = wp.tid()
        result = q[i]
        if st.mass[i] > real(0):
            gravity = vec()
            gravity[cfg.axis] = -cfg.gravity
            velocity = q[i] * cfg.velocity_ref / st.scale[i]
            result = (
                (velocity - st.predictor[i] - cfg.dt * (forces[i] / st.mass[i] + gravity))
                * st.scale[i]
                / cfg.velocity_ref
            )
        residual[i] = result

    @wp.kernel(module=module)
    def mass_residual_grid(
        st: Support,
        cfg: Parameters,
        q: wp.array(dtype=vec),
        forces: wp.array(dtype=vec),
        residual: wp.array(dtype=vec),
    ):
        """Unnormalized momentum equation at the accepted velocity-coordinate root.

        R_mass = mass * velocity_ref / scale * R_velocity. Cotangents are
        transformed separately, outside the tape; the multiplier is not an
        extra differentiable argument. No force or momentum term is clamped.
        """
        i = wp.tid()
        result = q[i]
        mass = st.mass[i]
        if mass > real(0):
            gravity = vec()
            gravity[cfg.axis] = -cfg.gravity
            velocity = q[i] * cfg.velocity_ref / st.scale[i]
            result = mass * (velocity - cfg.dt * gravity) - st.momentum[i] - cfg.dt * forces[i]
        residual[i] = result

    @wp.kernel(module=module, enable_backward=False)
    def mass_adjoint_seed(
        st: Support, cfg: Parameters, multiplier: wp.array(dtype=vec), seed: wp.array(dtype=vec)
    ):
        """Minus inverse-transpose residual scaling, held fixed during the VJP."""
        i = wp.tid()
        factor = real(1)
        if st.mass[i] > real(0):
            factor = st.scale[i] / (st.mass[i] * cfg.velocity_ref)
        seed[i] = -factor * multiplier[i]

    @wp.kernel(module=module, enable_backward=False)
    def tangent_grid(
        st: Support,
        cfg: Parameters,
        x: wp.array(dtype=vec),
        force_derivative: wp.array(dtype=vec),
        y: wp.array(dtype=vec),
        alpha: real,
        beta: real,
        transpose: int,
        out: wp.array(dtype=vec),
    ):
        i = wp.tid()
        value = x[i]
        if st.mass[i] > real(0):
            factor = st.scale[i] / (st.mass[i] * cfg.velocity_ref)
            if transpose != 0:
                factor = cfg.velocity_ref / st.scale[i]
            value = value - cfg.dt * force_derivative[i] * factor
        old = vec()
        if beta != real(0):
            old = beta * y[i]
        out[i] = alpha * value + old

    @wp.kernel(module=module, enable_backward=False)
    def bound_diagonal(
        model: Model, state: State, st: Support, cfg: Parameters, next_a: wp.array(dtype=real)
    ):
        p = wp.tid()
        if model.alpha[p] > real(0):
            young, nu = model.young[p], model.poisson[p]
            mu = young / (real(2) * (real(1) + nu))
            lam = young * nu / ((real(1) + nu) * (real(1) - real(2) * nu))
            amount, sigma = material_amount(p, model, next_a, cfg)
            modulus = lam + real(2) * mu + wp.abs(sigma)
            for slot in range(S):
                g = st.gradient[p, slot]
                value = model.alpha[p] * model.volume[p] * modulus * wp.dot(g, g)
                if cfg.contact != 0:
                    for face in range(FACES):
                        r = state.F[p] * model.face_offset[face]
                        h = st.weight[p, slot] + wp.dot(g, r)
                        value = value + cfg.stiffness * model.contact_area[p, face] * h * h
                wp.atomic_add(st.diagonal, st.ids[p, slot], value)

    @wp.kernel(module=module, enable_backward=False)
    def finalize_diagonal(st: Support, cfg: Parameters):
        i = wp.tid()
        value = real(1)
        if st.mass[i] > real(0):
            value = real(1) + cfg.dt * cfg.dt * st.diagonal[i] / st.mass[i]
        st.diagonal[i] = value

    @wp.kernel(module=module, enable_backward=False)
    def precondition(
        st: Support,
        x: wp.array(dtype=vec),
        y: wp.array(dtype=vec),
        alpha: real,
        beta: real,
        out: wp.array(dtype=vec),
    ):
        i = wp.tid()
        value = alpha * x[i] / st.diagonal[i]
        if beta != real(0):
            value = value + beta * y[i]
        out[i] = value

    @wp.kernel(module=module, enable_backward=False)
    def valid_state(
        model: Model,
        state: State,
        st: Support,
        cfg: Parameters,
        q: wp.array(dtype=vec),
        minimum: wp.array(dtype=real),
        bad: wp.array(dtype=int),
    ):
        p = wp.tid()
        if model.alpha[p] > real(0):
            velocity, gradient, affine = gather(p, st, cfg, q, 0)
            determinant = wp.determinant(eye + (state.F[p] - eye) + cfg.dt * gradient * state.F[p])
            wp.atomic_min(minimum, 0, determinant)
            valid = (
                wp.isfinite(determinant)
                and determinant > cfg.min_J
                and st.valid[p] != 0
                and support_valid(state.x[p] + cfg.dt * velocity, cfg) != 0
            )
            if not valid:
                wp.atomic_max(bad, 0, 1)

    @wp.kernel(module=module)
    def commit(
        model: Model,
        state: State,
        st: Support,
        cfg: Parameters,
        q: wp.array(dtype=vec),
        next_a: wp.array(dtype=real),
        code: wp.array(dtype=int),
        out: State,
        work: wp.array(dtype=real),
    ):
        p = wp.tid()
        if code[0] == ROOT_OK and state.status[0] == BINARY_STATE_OK and model.alpha[p] > real(0):
            velocity, gradient, affine = gather(p, st, cfg, q, 0)
            increment = cfg.dt * gradient * state.F[p]
            delta = state.F[p] - eye + increment
            out.x[p] = state.x[p] + cfg.dt * velocity
            out.v[p] = velocity
            out.F[p] = eye + delta
            out.C[p] = affine
            amount, sigma = material_amount(p, model, next_a, cfg)
            w = real(0)
            if cfg.stress_mode != 0:
                active = sigma * local.projector(model.direction[p], model.kind[p])
                w = -wp.ddot(active + delta * active, increment)
            else:
                previous = cfg.log_stretch * model.strength[p] * state.actuator[model.channel[p]]
                w = local.strain_work(
                    delta,
                    model.young[p],
                    model.poisson[p],
                    previous,
                    amount,
                    model.direction[p],
                    model.kind[p],
                )
            w = w * model.alpha[p] * model.volume[p]
            dissipation = real(0)
            if cfg.contact != 0:
                for face in range(FACES):
                    r = state.F[p] * model.face_offset[face]
                    marker_v = velocity + gradient * r
                    cf, power = contact(
                        state.x[p] + r + cfg.dt * marker_v,
                        marker_v,
                        model.contact_area[p, face],
                        model,
                        cfg,
                    )
                    dissipation = dissipation + power
            wp.atomic_add(work, LEDGER_WORK_SIGNED, w)
            wp.atomic_add(work, LEDGER_WORK_POSITIVE, positive_work(w))
            wp.atomic_add(work, LEDGER_DISSIPATED, cfg.dt * dissipation)
        else:
            out.x[p] = state.x[p]
            out.v[p] = state.v[p]
            out.F[p] = state.F[p]
            out.C[p] = state.C[p]

    @wp.kernel(module=module)
    def finish_state(
        state: State,
        next_a: wp.array(dtype=real),
        cfg: Parameters,
        code: wp.array(dtype=int),
        work: wp.array(dtype=real),
        out: State,
    ):
        if code[0] == ROOT_OK and state.status[0] == BINARY_STATE_OK:
            out.status[0] = BINARY_STATE_OK
            for i in range(WORK_INCREMENT_WIDTH):
                out.ledger[i] = state.ledger[i] + work[i]
            out.ledger[LEDGER_TIME] = state.ledger[LEDGER_TIME] + cfg.dt
            for channel in range(next_a.shape[0]):
                out.actuator[channel] = next_a[channel]
        else:
            out.status[0] = BINARY_STATE_FAILED
            for i in range(LEDGER_WIDTH):
                out.ledger[i] = state.ledger[i]
            for channel in range(next_a.shape[0]):
                out.actuator[channel] = state.actuator[channel]

    return SimpleNamespace(**locals())
