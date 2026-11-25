"""Analytic sparse interaction rows; no global dense Jacobian or finite differences.

Spring rows include the changing unit direction in the viscous term. Hex8 rows
use hyperelastic major symmetry in physical displacement coordinates; input free
masks and velocity-to-displacement scales are then applied columnwise. Node
contact is intentionally kept in its separate, generally nonsymmetric AD block.
"""

from functools import lru_cache
from types import SimpleNamespace

import warp as wp

from ...contracts.metrics import WORK_LINEARIZATION_BUILD
from ..physics.physics_constants import SPRING_SQUARED_LENGTH_FLOOR
from .solid_kernels import solid_kernels


@lru_cache(maxsize=4)
def analytic_tangent_kernels(physics: "str", precision: "str") -> "SimpleNamespace":
    """Return the cached analytic spring/Hex8 local tangent namespace.

    Types and kernel identities depend on the requested physics/dimension and
    precision. Warp DSL annotations must be evaluated in this factory's
    scope; postponed annotations and relocation change kernel construction."""
    if physics not in ("spring", "fem"):
        raise ValueError("Analytic rows are implemented for spring and Hex8 only")
    k = solid_kernels(physics, precision)
    real, vec, mat, lvec = k.real, k.vec, k.mat, k.lvec
    spring_length_floor = wp.constant(real(SPRING_SQUARED_LENGTH_FLOOR))
    Model, State, Parameters = k.Model, k.State, k.Parameters
    local, node_value, part = k.local, k.node_value, k.part
    gather, amount = k.gather, k.amount
    D, Q, L = k.D, k.Q, k.L
    eye = wp.constant(mat(*[float(i == j) for i in range(D) for j in range(D)]))
    module = f"envs.soft_walker.engine.warp_native.analytic_{physics}_{precision}_v1"

    @wp.kernel(module=module, enable_backward=False)
    def build(
        m: Model,
        s: State,
        p: Parameters,
        q: wp.array(dtype=k.qvec),
        actuator: wp.array(dtype=real),
        rows: wp.array2d(dtype=lvec),
        work_counts: wp.array(dtype=int),
    ):
        e, row = wp.tid()
        if e == 0 and row == 0:
            wp.atomic_add(work_counts, WORK_LINEARIZATION_BUILD, 1)
        z = gather(e, q, m)
        result = lvec()
        if wp.static(physics == "spring"):
            i, j = m.entities[e, 0], m.entities[e, 1]
            ui, vi, Ri, wi = node_value(part(z, 0), i, m, s, p)
            uj, vj, Rj, wj = node_value(part(z, 1), j, m, s, p)
            delta = m.reference[j] - m.reference[i] + uj - ui
            length = wp.sqrt(wp.dot(delta, delta) + spring_length_floor)
            normal = delta / length
            relative = vj - vi
            speed = wp.dot(relative, normal)
            preferred = m.length[e] * wp.exp(amount(e, actuator, m))
            tension = m.stiffness[e] * (length - preferred) + m.damping[e] * speed
            nn = wp.outer(normal, normal)
            normal_derivative = (eye - nn) / length
            position_derivative = (
                m.stiffness[e] * nn
                + m.damping[e] * wp.outer(normal, normal_derivative * relative)
                + tension * normal_derivative
            )
            derivative = p.velocity_ref * (p.dt * position_derivative + m.damping[e] * nn)
            sign_row = real(1)
            if row >= Q:
                sign_row = real(-1)
            for column in range(L):
                node = m.entities[e, column // Q]
                sign_column = real(-1)
                if column >= Q:
                    sign_column = real(1)
                if m.mass[node] > real(0):
                    result[column] = (
                        sign_row
                        * sign_column
                        * derivative[row % Q, column % Q]
                        * m.free[node][column % Q]
                    )
        else:
            if m.alpha[e] > real(0):
                row_node, row_axis = row // D, row % D
                unit = vec()
                unit[row_axis] = real(1)
                strain = p.log_stretch * m.strength[e] * actuator[m.channel[e]]
                for qp in range(8):
                    delta = mat()
                    for b in range(8):
                        u, v, R, w = node_value(part(z, b), m.entities[e, b], m, s, p)
                        delta = delta + wp.outer(u, m.gradients[qp, b])
                    increment = wp.outer(unit, m.gradients[qp, row_node])
                    stress_increment = local.tangent(
                        delta,
                        increment,
                        m.young[e],
                        m.poisson[e],
                        strain,
                        m.direction[e],
                        m.kind[e],
                    )
                    contribution = lvec()
                    for b in range(8):
                        node = m.entities[e, b]
                        if m.mass[node] > real(0):
                            value = (
                                -m.alpha[e]
                                * m.weights[e, qp]
                                * p.dt
                                * p.velocity_ref
                                * (stress_increment * m.gradients[qp, b])
                            )
                            for axis in range(D):
                                contribution[b * D + axis] = value[axis] * m.free[node][axis]
                    result = result + contribution
        rows[e, row] = result

    return SimpleNamespace(build=build)
