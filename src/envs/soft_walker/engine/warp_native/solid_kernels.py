"""Native spring, midpoint Cosserat beam, and fully integrated Hex8 operators.

One force thread owns a complete interaction. Shared-node scatters finish before
the residual is formed. Local sparse tangent blocks are built once per Newton
linearization using Warp's generated reverse kernels, not finite differences.
Beam forces are the spatial derivative of the same objective midpoint energy
as the JAX model. Nothing in this module imports or invokes JAX.
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
    WORK_LINEARIZATION_BUILD,
)
from ...contracts.status_codes import ROOT_OK
from ..physics.physics_constants import (
    BEAM_MAX_RELATIVE_ROTATION,
    BEAM_MAX_STEP_ROTATION,
    BEAM_MIN_LENGTH_RATIO,
    SO3_SERIES_SQUARED_THRESHOLD,
    SPRING_SQUARED_LENGTH_FLOOR,
)
from .kernels import kernels as mpm_kernels
from .local import functions
from .work import positive_work_function


@lru_cache(maxsize=6)
def solid_kernels(physics: "str", precision: "str") -> "SimpleNamespace":
    """Return the cached spring/beam/FEM force, state and local tangent namespace.

    Types and kernel identities depend on the requested physics/dimension and
    precision. Warp DSL annotations must be evaluated in this factory's
    scope; postponed annotations and relocation change kernel construction."""
    if physics not in ("spring", "beam", "fem"):
        raise ValueError("Unknown Lagrangian model")
    D = 2 if physics == "spring" else 3
    Q = 6 if physics == "beam" else D
    N = 8 if physics == "fem" else 2
    L = N * Q
    local, base = functions(D, precision), mpm_kernels(D, precision)
    positive_work = positive_work_function(precision)
    real, vec, mat = local.scalar, local.vector, local.matrix
    spring_length_floor = wp.constant(real(SPRING_SQUARED_LENGTH_FLOOR))
    so3_series_squared_threshold = wp.constant(real(SO3_SERIES_SQUARED_THRESHOLD))
    beam_min_length_ratio = wp.constant(real(BEAM_MIN_LENGTH_RATIO))
    beam_max_relative_rotation = wp.constant(real(BEAM_MAX_RELATIVE_ROTATION))
    beam_max_step_rotation_squared = wp.constant(real(BEAM_MAX_STEP_ROTATION**2))
    qvec = wp.types.vector(length=Q, dtype=real)
    lvec = wp.types.vector(length=L, dtype=real)
    Terrain, Parameters = base.Model, base.Parameters
    eye = wp.constant(mat(*[float(i == j) for i in range(D) for j in range(D)]))
    module = f"envs.soft_walker.engine.warp_native.solid_{physics}_{precision}_v1"
    if physics == "fem":
        # Level-0 FEM force adjoints exceeded 512 KiB of local memory per thread.
        # Keep optimization to eliminate AD temporaries; level 2 retains the CPU
        # default. Use Warp's normal CUDA output selection and budget for the
        # first build. Equations, AD, unrolling and math policy are unchanged.
        wp.set_module_options({"optimization_level": 2}, module=SimpleNamespace(__name__=module))

    def struct(cls: "type") -> "wp.codegen.Struct":
        """Bind a struct to this precision-specific kernel module and disable AD defaults."""
        return wp.struct(cls, module=module)

    @struct
    class Model:
        reference: wp.array(dtype=vec)
        mass: wp.array(dtype=real)
        inertia: wp.array(dtype=real)
        free: wp.array(dtype=vec)
        entities: wp.array2d(dtype=int)
        cells: wp.array2d(dtype=int)
        cell_alpha: wp.array(dtype=real)
        stiffness: wp.array(dtype=real)
        damping: wp.array(dtype=real)
        length: wp.array(dtype=real)
        routing: wp.array2d(dtype=real)
        extension: wp.array(dtype=mat)
        bending: wp.array(dtype=mat)
        axis: wp.array(dtype=vec)
        young: wp.array(dtype=real)
        poisson: wp.array(dtype=real)
        strength: wp.array(dtype=real)
        direction: wp.array(dtype=vec)
        kind: wp.array(dtype=int)
        channel: wp.array(dtype=int)
        alpha: wp.array(dtype=real)
        gradients: wp.array2d(dtype=vec)
        weights: wp.array2d(dtype=real)
        contact_area: wp.array2d(dtype=real)
        offsets: wp.array(dtype=vec)
        terrain: Terrain
        h: real
        area_ref: real
        min_length_ratio: real
        min_area_ratio: real
        legacy_contact: int

    @struct
    class State:
        u: wp.array(dtype=vec)
        v: wp.array(dtype=vec)
        rotation: wp.array(dtype=mat)
        omega: wp.array(dtype=vec)
        actuator: wp.array(dtype=real)
        ledger: wp.array(dtype=real)
        status: wp.array(dtype=int)

    @wp.func(module=module)
    def skew(v: vec):
        out = mat()
        if wp.static(D == 3):
            out = mat(real(0), -v[2], v[1], v[2], real(0), -v[0], -v[1], v[0], real(0))
        return out

    @wp.func(module=module)
    def exp_so3(v: vec):
        t2 = wp.dot(v, v)
        a, b = real(0), real(0)
        if t2 < so3_series_squared_threshold:
            a = real(1) - t2 / real(6) + t2 * t2 / real(120)
            b = real(0.5) - t2 / real(24) + t2 * t2 / real(720)
        else:
            t = wp.sqrt(t2)
            a, b = wp.sin(t) / t, (real(1) - wp.cos(t)) / t2
        K = skew(v)
        return eye + a * K + b * K * K

    @wp.func(module=module)
    def log_so3(R: mat):
        v = vec()
        if wp.static(D == 3):
            v = real(0.5) * vec(R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1])
        s2 = wp.dot(v, v)
        factor = real(0)
        if s2 < so3_series_squared_threshold:
            factor = real(1) + s2 / real(6) + real(3) * s2 * s2 / real(40)
        else:
            s = wp.sqrt(s2)
            factor = wp.atan2(s, (wp.trace(R) - real(1)) / real(2)) / s
        return factor * v

    @wp.func(module=module)
    def left_jacobian(v: vec):
        t2 = wp.dot(v, v)
        b, c = real(0), real(0)
        if t2 < so3_series_squared_threshold:
            b = real(0.5) - t2 / real(24) + t2 * t2 / real(720)
            c = real(1.0 / 6.0) - t2 / real(120) + t2 * t2 / real(5040)
        else:
            t = wp.sqrt(t2)
            b = (real(1) - wp.cos(t)) / t2
            c = (t - wp.sin(t)) / (t2 * t)
        K = skew(v)
        return eye + b * K + c * K * K

    @wp.func(module=module)
    def left_inverse(v: vec):
        t2 = wp.dot(v, v)
        c = real(0)
        if t2 < so3_series_squared_threshold:
            c = real(1.0 / 12.0) + t2 / real(720) + t2 * t2 / real(30240)
        else:
            t = wp.sqrt(t2)
            c = (real(1) - real(0.5) * t / wp.tan(real(0.5) * t)) / t2
        K = skew(v)
        return eye - real(0.5) * K + c * K * K

    @wp.func(module=module)
    def part(z: lvec, a: int):
        out = qvec()
        for d in range(Q):
            out[d] = z[a * Q + d]
        return out

    @wp.func(module=module)
    def node_value(z: qvec, node: int, m: Model, s: State, p: Parameters):
        v, w = vec(), vec()
        for d in range(D):
            v[d] = p.velocity_ref * z[d] * m.free[node][d]
            if wp.static(physics == "beam"):
                w[d] = p.velocity_ref * z[d + 3] / m.h
        u = s.u[node] + vec()
        R = s.rotation[node] + mat()
        if m.mass[node] > real(0):
            u = u + p.dt * v
            if wp.static(physics == "beam"):
                R = exp_so3(p.dt * w) * R
        else:
            u, v, w, R = vec(), vec(), vec(), eye
        return u, v, R, w

    @wp.func(module=module)
    def amount(e: int, a: wp.array(dtype=real), m: Model):
        out = real(0)
        for c in range(a.shape[0]):
            out = out + m.routing[e, c] * a[c]
        return out

    @wp.func(module=module)
    def amount_change(e: int, old: wp.array(dtype=real), new: wp.array(dtype=real), m: Model):
        # Keep the dynamic linear reduction behind a function call. Warp 1.16
        # does not replay a kernel-local dynamic accumulation before a later
        # nonlinear operation in its generated reverse pass.
        value = real(0)
        for c in range(new.shape[0]):
            value = value + m.routing[e, c] * (new[c] - old[c])
        return value

    @wp.func(module=module)
    def beam_data(
        e: int, ui: vec, uj: vec, Ri: mat, Rj: mat, a: wp.array(dtype=real), m: Model, p: Parameters
    ):
        r = log_so3(wp.transpose(Ri) * Rj)
        mid = Ri * exp_so3(real(0.5) * r)
        local_delta = wp.transpose(mid - eye) * m.axis[e] + wp.transpose(mid) * (uj - ui) / m.h
        gamma = local_delta - local.expm1(p.log_stretch * amount(e, a, m)) * m.axis[e]
        return r, mid, gamma

    @wp.func(module=module)
    def interaction_force(
        e: int, z: lvec, m: Model, s: State, p: Parameters, a: wp.array(dtype=real)
    ):
        result = lvec()
        if wp.static(physics == "spring"):
            i, j = int(m.entities[e, 0]), int(m.entities[e, 1])
            ui, vi, Ri, wi = node_value(part(z, 0), i, m, s, p)
            uj, vj, Rj, wj = node_value(part(z, 1), j, m, s, p)
            delta = m.reference[j] - m.reference[i] + uj - ui
            length = wp.sqrt(wp.dot(delta, delta) + spring_length_floor)
            n = delta / length
            preferred = m.length[e] * wp.exp(amount(e, a, m))
            speed = wp.dot(vj - vi, n)
            f = (m.stiffness[e] * (length - preferred) + m.damping[e] * speed) * n
            for d in range(D):
                result[d] = f[d]
                result[Q + d] = -f[d]
        elif wp.static(physics == "beam"):
            i, j = int(m.entities[e, 0]), int(m.entities[e, 1])
            ui, vi, Ri, wi = node_value(part(z, 0), i, m, s, p)
            uj, vj, Rj, wj = node_value(part(z, 1), j, m, s, p)
            r, mid, gamma = beam_data(e, ui, uj, Ri, Rj, a, m, p)
            f = mid * m.extension[e] * gamma
            delta = m.h * m.axis[e] + uj - ui
            inv = left_inverse(r)
            H = real(0.5) * Ri * left_jacobian(real(0.5) * r) * inv * wp.transpose(Ri)
            bend_moment = Ri * wp.transpose(inv) * m.bending[e] * r / m.h
            moment = wp.cross(delta, f)
            ti = wp.transpose(eye - H) * moment + bend_moment
            tj = wp.transpose(H) * moment - bend_moment
            for d in range(3):
                result[d] = f[d]
                result[d + 3] = ti[d]
                result[d + 6] = -f[d]
                result[d + 9] = tj[d]
        else:
            if m.alpha[e] > real(0):
                strain = p.log_stretch * m.strength[e] * a[m.channel[e]]
                for qp in range(8):
                    delta = mat()
                    for b in range(8):
                        u, v, R, w = node_value(part(z, b), m.entities[e, b], m, s, p)
                        delta = delta + wp.outer(u, m.gradients[qp, b])
                    stress = local.stress(
                        delta, m.young[e], m.poisson[e], strain, m.direction[e], m.kind[e]
                    )
                    for b in range(8):
                        f = -m.alpha[e] * m.weights[e, qp] * stress * m.gradients[qp, b]
                        increment = lvec()
                        for d in range(3):
                            increment[b * 3 + d] = f[d]
                        result = result + increment
        return result

    @wp.func(module=module)
    def contact_value(node: int, z: qvec, m: Model, s: State, p: Parameters):
        u, v, R, omega = node_value(z, node, m, s, p)
        force, torque = vec(), vec()
        energy, dissipation = real(0), real(0)
        if p.contact != 0 and m.mass[node] > real(0):
            for face in range(m.contact_area.shape[1]):
                r, velocity = vec(), v
                if wp.static(physics == "beam"):
                    r = R * m.offsets[face]
                    velocity = v + wp.cross(omega, r)
                x = m.reference[node] + u + r
                area = m.contact_area[node, face]
                cf, ce, cd = vec(), real(0), real(0)
                if wp.static(physics == "spring"):
                    if m.legacy_contact != 0:
                        depth = p.width * local.softplus((m.terrain.plane_offset - x[1]) / p.width)
                        gate = local.sigmoid((m.terrain.plane_offset - x[1]) / p.width)
                        normal = area * depth * gate
                        cf = vec(
                            -p.friction * normal * wp.tanh(velocity[0] / p.velocity_width), normal
                        )
                        ce, cd = real(0.5) * area * depth * depth, -cf[0] * velocity[0]
                    else:
                        cf, cd = base.contact(x, velocity, area, m.terrain, p)
                else:
                    cf, cd = base.contact(x, velocity, area, m.terrain, p)
                force = force + cf
                if wp.static(physics == "beam"):
                    torque = torque + wp.cross(r, cf)
                energy, dissipation = energy + ce, dissipation + cd
        gravity = vec()
        gravity[p.axis] = -m.mass[node] * p.gravity
        force = force + gravity
        result = qvec()
        for d in range(D):
            result[d] = force[d]
            if wp.static(physics == "beam"):
                result[d + 3] = torque[d]
        return result, energy, dissipation

    @wp.func(module=module)
    def gather(e: int, q: wp.array(dtype=qvec), m: Model):
        out = lvec()
        for a in range(N):
            value = q[m.entities[e, a]]
            for d in range(Q):
                out[a * Q + d] = value[d]
        return out

    @wp.func(module=module)
    def row_scale(node: int, d: int, m: Model, p: Parameters):
        out = real(0)
        if m.mass[node] > real(0):
            if d < D:
                out = p.dt * m.free[node][d] / (m.mass[node] * p.velocity_ref)
            elif wp.static(physics == "beam"):
                out = p.dt * m.h / (m.inertia[node] * p.velocity_ref)
        return out

    @wp.kernel(module=module)
    def actuator(s: State, command: wp.array(dtype=real), p: Parameters, out: wp.array(dtype=real)):
        i = wp.tid()
        out[i] = command[i] + (s.actuator[i] - command[i]) * wp.exp(-p.dt / p.tau)

    @wp.kernel(module=module)
    def force_kernel(
        m: Model,
        s: State,
        p: Parameters,
        q: wp.array(dtype=qvec),
        a: wp.array(dtype=real),
        out: wp.array(dtype=qvec),
    ):
        e = wp.tid()
        f = interaction_force(e, gather(e, q, m), m, s, p, a)
        for b in range(N):
            wp.atomic_add(out, m.entities[e, b], part(f, b))

    @wp.kernel(module=module)
    def node_residual(
        m: Model,
        s: State,
        p: Parameters,
        q: wp.array(dtype=qvec),
        elastic: wp.array(dtype=qvec),
        out: wp.array(dtype=qvec),
    ):
        i = wp.tid()
        cf, energy, dissipation = contact_value(i, q[i], m, s, p)
        f = elastic[i] + cf
        result = qvec()
        for d in range(Q):
            previous = real(0)
            if m.mass[i] > real(0):
                if d < D:
                    previous = s.v[i][d] * m.free[i][d] / p.velocity_ref
                elif wp.static(physics == "beam"):
                    previous = m.h * s.omega[i][d - 3] / p.velocity_ref
            result[d] = q[i][d] - previous - row_scale(i, d, m, p) * f[d]
        out[i] = result

    @wp.kernel(module=module, enable_backward=False)
    def prepare_guess(m: Model, s: State, p: Parameters, out: wp.array(dtype=qvec)):
        i = wp.tid()
        value = qvec()
        if m.mass[i] > real(0):
            for d in range(D):
                value[d] = s.v[i][d] * m.free[i][d] / p.velocity_ref
                if wp.static(physics == "beam"):
                    value[d + 3] = m.h * s.omega[i][d] / p.velocity_ref
        out[i] = value

    @wp.kernel(module=module, enable_backward=False)
    def prepare_rows(
        m: Model,
        q: wp.array(dtype=qvec),
        rows: wp.array2d(dtype=lvec),
        seeds: wp.array2d(dtype=lvec),
        work_counts: wp.array(dtype=int),
    ):
        e, row = wp.tid()
        if e == 0 and row == 0:
            wp.atomic_add(work_counts, WORK_LINEARIZATION_BUILD, 1)
        rows[e, row] = gather(e, q, m)
        one = lvec()
        one[row] = real(1)
        seeds[e, row] = one

    @wp.kernel(module=module)
    def row_force(
        m: Model,
        s: State,
        p: Parameters,
        q: wp.array2d(dtype=lvec),
        a: wp.array(dtype=real),
        out: wp.array2d(dtype=lvec),
    ):
        e, row = wp.tid()
        out[e, row] = interaction_force(e, q[e, row], m, s, p, a)

    @wp.kernel(module=module, enable_backward=False)
    def prepare_nodes(
        q: wp.array(dtype=qvec), rows: wp.array2d(dtype=qvec), seeds: wp.array2d(dtype=qvec)
    ):
        i, row = wp.tid()
        rows[i, row] = q[i]
        one = qvec()
        one[row] = real(1)
        seeds[i, row] = one

    @wp.kernel(module=module)
    def row_contact(
        m: Model, s: State, p: Parameters, q: wp.array2d(dtype=qvec), out: wp.array2d(dtype=qvec)
    ):
        i, row = wp.tid()
        f, energy, dissipation = contact_value(i, q[i, row], m, s, p)
        out[i, row] = f

    @wp.kernel(module=module, enable_backward=False)
    def apply_blocks(
        m: Model,
        p: Parameters,
        blocks: wp.array2d(dtype=lvec),
        x: wp.array(dtype=qvec),
        transpose: int,
        out: wp.array(dtype=qvec),
    ):
        e = wp.tid()
        v, value = gather(e, x, m), lvec()
        for r in range(L):
            scale = row_scale(m.entities[e, r // Q], r % Q, m, p)
            if transpose == 0:
                value[r] = -scale * wp.dot(blocks[e, r], v)
            else:
                value = value - scale * v[r] * blocks[e, r]
        for b in range(N):
            wp.atomic_add(out, m.entities[e, b], part(value, b))

    @wp.kernel(module=module, enable_backward=False)
    def apply_nodes(
        m: Model,
        p: Parameters,
        blocks: wp.array2d(dtype=qvec),
        x: wp.array(dtype=qvec),
        elastic: wp.array(dtype=qvec),
        y: wp.array(dtype=qvec),
        alpha: real,
        beta: real,
        transpose: int,
        out: wp.array(dtype=qvec),
        work_counts: wp.array(dtype=int),
    ):
        i = wp.tid()
        if i == 0:
            wp.atomic_add(work_counts, WORK_FORWARD_TANGENT + transpose, 1)
        value = x[i] + elastic[i]
        for r in range(Q):
            scale = row_scale(i, r, m, p)
            if transpose == 0:
                value[r] = value[r] - scale * wp.dot(blocks[i, r], x[i])
            else:
                value = value - scale * x[i][r] * blocks[i, r]
        result = alpha * value
        if beta != real(0):
            result = result + beta * y[i]
        out[i] = result

    @wp.kernel(module=module, enable_backward=False)
    def accumulate_diagonal(
        m: Model, p: Parameters, blocks: wp.array2d(dtype=lvec), diagonal: wp.array(dtype=qvec)
    ):
        e, r = wp.tid()
        node, d = m.entities[e, r // Q], r % Q
        add = qvec()
        add[d] = -row_scale(node, d, m, p) * blocks[e, r][r]
        wp.atomic_add(diagonal, node, add)

    @wp.kernel(module=module, enable_backward=False)
    def finalize_diagonal(
        m: Model, p: Parameters, blocks: wp.array2d(dtype=qvec), diagonal: wp.array(dtype=qvec)
    ):
        i = wp.tid()
        value = diagonal[i] + qvec()
        for d in range(Q):
            value[d] = wp.max(
                wp.abs(real(1) + value[d] - row_scale(i, d, m, p) * blocks[i, d][d]), real(1)
            )
        diagonal[i] = value

    @wp.kernel(module=module, enable_backward=False)
    def precondition(
        diagonal: wp.array(dtype=qvec),
        x: wp.array(dtype=qvec),
        y: wp.array(dtype=qvec),
        alpha: real,
        beta: real,
        out: wp.array(dtype=qvec),
    ):
        i = wp.tid()
        value = alpha * wp.cw_div(x[i], diagonal[i])
        if beta != real(0):
            value = value + beta * y[i]
        out[i] = value

    @wp.kernel(module=module, enable_backward=False)
    def valid_nodes(
        m: Model, s: State, p: Parameters, q: wp.array(dtype=qvec), bad: wp.array(dtype=int)
    ):
        i = wp.tid()
        for d in range(Q):
            if not wp.isfinite(q[i][d]):
                wp.atomic_max(bad, 0, 1)
        if wp.static(physics == "beam"):
            u, v, R, w = node_value(q[i], i, m, s, p)
            if p.dt * p.dt * wp.dot(w, w) >= beam_max_step_rotation_squared:
                wp.atomic_max(bad, 0, 1)

    @wp.kernel(module=module, enable_backward=False)
    def valid_entities(
        m: Model,
        s: State,
        p: Parameters,
        q: wp.array(dtype=qvec),
        minimum: wp.array(dtype=real),
        bad: wp.array(dtype=int),
    ):
        e = wp.tid()
        z = gather(e, q, m)
        metric = real(1)
        if wp.static(physics == "fem"):
            if m.alpha[e] > real(0):
                for qp in range(8):
                    F = eye
                    for a in range(8):
                        u, v, R, w = node_value(part(z, a), m.entities[e, a], m, s, p)
                        F = F + wp.outer(u, m.gradients[qp, a])
                    metric = wp.min(metric, wp.determinant(F))
                if metric <= p.min_J:
                    wp.atomic_max(bad, 0, 1)
        else:
            i, j = int(m.entities[e, 0]), int(m.entities[e, 1])
            ui, vi, Ri, wi = node_value(part(z, 0), i, m, s, p)
            uj, vj, Rj, wj = node_value(part(z, 1), j, m, s, p)
            delta = m.reference[j] - m.reference[i] + uj - ui
            if wp.static(physics == "beam"):
                if m.mass[i] > real(0) and m.mass[j] > real(0):
                    metric = wp.length(delta) / m.h
                    cosine = (wp.trace(wp.transpose(Ri) * Rj) - real(1)) / real(2)
                    if metric <= beam_min_length_ratio or cosine <= wp.cos(
                        beam_max_relative_rotation
                    ):
                        wp.atomic_max(bad, 0, 1)
            else:
                if m.stiffness[e] > real(0):
                    metric = wp.length(delta) / m.length[e]
                    if metric < m.min_length_ratio:
                        wp.atomic_max(bad, 0, 1)
        wp.atomic_min(minimum, 0, metric)

    @wp.kernel(module=module, enable_backward=False)
    def valid_cells(
        m: Model, s: State, p: Parameters, q: wp.array(dtype=qvec), bad: wp.array(dtype=int)
    ):
        c = wp.tid()
        if wp.static(physics == "spring"):
            if m.cell_alpha[c] > real(0):
                for a in range(4):
                    i, j, k = (
                        int(m.cells[c, a]),
                        int(m.cells[c, (a + 1) % 4]),
                        int(m.cells[c, (a + 2) % 4]),
                    )
                    ui, vi, Ri, wi = node_value(q[i], i, m, s, p)
                    uj, vj, Rj, wj = node_value(q[j], j, m, s, p)
                    uk, vk, Rk, wk = node_value(q[k], k, m, s, p)
                    one, two = (
                        m.reference[j] - m.reference[i] + uj - ui,
                        m.reference[k] - m.reference[j] + uk - uj,
                    )
                    if (one[0] * two[1] - one[1] * two[0]) / m.area_ref < m.min_area_ratio:
                        wp.atomic_max(bad, 0, 1)

    @wp.kernel(module=module)
    def commit_nodes(
        m: Model,
        s: State,
        p: Parameters,
        q: wp.array(dtype=qvec),
        code: wp.array(dtype=int),
        out: State,
        work: wp.array(dtype=real),
    ):
        i = wp.tid()
        u = s.u[i] + vec()
        v = s.v[i] + vec()
        R = s.rotation[i] + mat()
        w = s.omega[i] + vec()
        if code[0] == ROOT_OK:
            u, v, R, w = node_value(q[i], i, m, s, p)
            cf, energy, diss = contact_value(i, q[i], m, s, p)
            wp.atomic_add(work, LEDGER_DISSIPATED, p.dt * diss)
        out.u[i] = u
        out.v[i] = v
        out.rotation[i] = R
        out.omega[i] = w

    @wp.kernel(module=module)
    def work_kernel(
        m: Model,
        s: State,
        p: Parameters,
        q: wp.array(dtype=qvec),
        a: wp.array(dtype=real),
        code: wp.array(dtype=int),
        work: wp.array(dtype=real),
    ):
        e = wp.tid()
        if code[0] == ROOT_OK:
            z, change, diss = gather(e, q, m), real(0), real(0)
            if wp.static(physics == "spring"):
                i, j = int(m.entities[e, 0]), int(m.entities[e, 1])
                ui, vi, Ri, wi = node_value(part(z, 0), i, m, s, p)
                uj, vj, Rj, wj = node_value(part(z, 1), j, m, s, p)
                d = m.reference[j] - m.reference[i] + uj - ui
                length = wp.sqrt(wp.dot(d, d) + spring_length_floor)
                old = m.length[e] * wp.exp(amount(e, s.actuator, m))
                da = amount_change(e, s.actuator, a, m)
                dp = old * local.expm1(da)
                change = m.stiffness[e] * dp * (real(0.5) * dp - (length - old))
                speed = wp.dot(vj - vi, d / length)
                diss = p.dt * m.damping[e] * speed * speed
                wp.atomic_add(work, LEDGER_WORK_SIGNED, change)
                wp.atomic_add(work, LEDGER_WORK_POSITIVE, positive_work(change))
            elif wp.static(physics == "beam"):
                i, j = int(m.entities[e, 0]), int(m.entities[e, 1])
                ui, vi, Ri, wi = node_value(part(z, 0), i, m, s, p)
                uj, vj, Rj, wj = node_value(part(z, 1), j, m, s, p)
                r, mid, gamma = beam_data(e, ui, uj, Ri, Rj, s.actuator, m, p)
                da = amount_change(e, s.actuator, a, m)
                dp = wp.exp(p.log_stretch * amount(e, s.actuator, m)) * local.expm1(
                    p.log_stretch * da
                )
                dg = dp * m.axis[e]
                change = m.h * (
                    real(0.5) * wp.dot(dg, m.extension[e] * dg) - wp.dot(gamma, m.extension[e] * dg)
                )
                wp.atomic_add(work, LEDGER_WORK_SIGNED, change)
                wp.atomic_add(work, LEDGER_WORK_POSITIVE, positive_work(change))
            else:
                if m.alpha[e] > real(0):
                    old = p.log_stretch * m.strength[e] * s.actuator[m.channel[e]]
                    new = p.log_stretch * m.strength[e] * a[m.channel[e]]
                    for qp in range(8):
                        delta = mat()
                        for b in range(8):
                            u, v, R, w = node_value(part(z, b), m.entities[e, b], m, s, p)
                            delta = delta + wp.outer(u, m.gradients[qp, b])
                        value = (
                            m.alpha[e]
                            * m.weights[e, qp]
                            * local.strain_work(
                                delta, m.young[e], m.poisson[e], old, new, m.direction[e], m.kind[e]
                            )
                        )
                        wp.atomic_add(work, LEDGER_WORK_SIGNED, value)
                        wp.atomic_add(work, LEDGER_WORK_POSITIVE, positive_work(value))
            wp.atomic_add(work, LEDGER_DISSIPATED, diss)

    @wp.kernel(module=module)
    def finish_state(
        s: State,
        a: wp.array(dtype=real),
        p: Parameters,
        code: wp.array(dtype=int),
        work: wp.array(dtype=real),
        out: State,
    ):
        # Explicit branches preserve array-load adjoints in Warp 1.16. Do not
        # rewrite this as a ternary selecting references to different arrays.
        if code[0] == ROOT_OK:
            for c in range(a.shape[0]):
                out.actuator[c] = a[c]
            for i in range(WORK_INCREMENT_WIDTH):
                out.ledger[i] = s.ledger[i] + work[i]
            out.ledger[LEDGER_TIME] = s.ledger[LEDGER_TIME] + p.dt
        else:
            for c in range(a.shape[0]):
                out.actuator[c] = s.actuator[c]
            for i in range(LEDGER_WIDTH):
                out.ledger[i] = s.ledger[i]
        out.status[0] = code[0]

    return SimpleNamespace(**locals())
