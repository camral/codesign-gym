"""Reuse a fixed Newton linearization during Krylov iterations.

Only Jacobian actions use this cache. The differentiable residual is unchanged,
so implicit VJPs still differentiate the original force, including a contact-area
derivative at zero area. Rebuild before every forward/transpose linear solve;
never reuse this cache across a changed q, state, material, or command.

Storage is O(particles), not a dense grid Jacobian. Cached material factors avoid
repeating inverse/log/exp operations and the current-q gather at every matvec.
Each exposed contact marker stores its velocity-to-force Jacobian, including
position dependence through x_next = x + dt*v. Transpose products use its actual
transpose; frictional/contact tangents are not assumed symmetric.
"""

from typing import TYPE_CHECKING

from ...contracts.metrics import WORK_FORWARD_TANGENT, WORK_LINEARIZATION_BUILD

if TYPE_CHECKING:
    from .model import DeviceModel

from functools import lru_cache
from types import SimpleNamespace

import warp as wp

from .kernels import kernels
from .local import functions


@lru_cache(maxsize=4)
def cached_kernels(dim: "int", precision: "str") -> "SimpleNamespace":
    """Return the cached cached MPM tangent and preconditioner namespace.

    Types and kernel identities depend on the requested physics/dimension and
    precision. Warp DSL annotations must be evaluated in this factory's
    scope; postponed annotations and relocation change kernel construction."""
    k = kernels(dim, precision)
    local = functions(dim, precision)
    real, vec, mat = k.real, k.vec, k.mat
    Model, State, Support, Parameters = k.Model, k.State, k.Support, k.Parameters
    D, S, FACES = dim, k.S, 2 * dim
    eye = wp.constant(mat(*[float(i == j) for i in range(dim) for j in range(dim)]))
    module = f"envs.soft_walker.engine.warp_native.linearization_d{dim}_{precision}_v1"

    @wp.struct
    class Cache:
        inverse_active: wp.array(dtype=mat)
        inverse_transpose: wp.array(dtype=mat)
        active_stress: wp.array(dtype=mat)
        mu: wp.array(dtype=real)
        lam: wp.array(dtype=real)
        coefficient: wp.array(dtype=real)
        marker_jacobian: wp.array2d(dtype=mat)

    @wp.kernel(module=module, enable_backward=False)
    def build(
        model: Model,
        state: State,
        st: Support,
        cfg: Parameters,
        q: wp.array(dtype=vec),
        next_a: wp.array(dtype=real),
        cache: Cache,
    ):
        p = wp.tid()
        if p == 0:
            wp.atomic_add(st.work_counts, WORK_LINEARIZATION_BUILD, 1)
        if model.alpha[p] > real(0):
            velocity, gradient, affine = k.gather(p, st, cfg, q, 0)
            delta = state.F[p] - eye + cfg.dt * gradient * state.F[p]
            amount, sigma = k.material_amount(p, model, next_a, cfg)
            A = local.inverse_active(amount, model.direction[p], model.kind[p])
            elastic = local.elastic_delta(delta, amount, model.direction[p], model.kind[p])
            dv, nonlinear = local.volume_delta(elastic)
            mu = model.young[p] / (real(2) * (real(1) + model.poisson[p]))
            lam = (
                model.young[p]
                * model.poisson[p]
                / ((real(1) + model.poisson[p]) * (real(1) - real(2) * model.poisson[p]))
            )
            cache.inverse_active[p] = A
            cache.inverse_transpose[p] = wp.transpose(wp.inverse(eye + elastic))
            cache.active_stress[p] = sigma * local.projector(model.direction[p], model.kind[p])
            cache.mu[p] = mu
            cache.lam[p] = lam
            cache.coefficient[p] = lam * local.log1p(dv) - mu
            for face in range(FACES):
                jacobian = mat()
                # Safe here: this is d(force)/d(q), not the residual VJP w.r.t.
                # geometry/contact_area. That VJP still uses the original force.
                if cfg.contact != 0 and model.contact_area[p, face] > real(0):
                    r = state.F[p] * model.face_offset[face]
                    marker_v = velocity + gradient * r
                    marker_x = state.x[p] + r + cfg.dt * marker_v
                    for axis in range(D):
                        direction = vec()
                        direction[axis] = real(1)
                        column = k.contact_tangent(
                            marker_x,
                            marker_v,
                            cfg.dt * direction,
                            direction,
                            model.contact_area[p, face],
                            model,
                            cfg,
                        )
                        for row in range(D):
                            jacobian[row, axis] = column[row]
                cache.marker_jacobian[p, face] = jacobian

    @wp.kernel(module=module, enable_backward=False)
    def apply(
        model: Model,
        state: State,
        st: Support,
        cfg: Parameters,
        direction: wp.array(dtype=vec),
        transpose: int,
        cache: Cache,
        output: wp.array(dtype=vec),
    ):
        p = wp.tid()
        if p == 0:
            wp.atomic_add(st.work_counts, WORK_FORWARD_TANGENT + int(transpose != 0), 1)
        if model.alpha[p] > real(0):
            dv, gradient, affine = k.gather(p, st, cfg, direction, transpose)
            increment = cfg.dt * gradient * state.F[p]
            A = cache.inverse_active[p]
            G = cache.inverse_transpose[p]
            H = increment * A
            dP = (
                cache.mu[p] * H
                + cache.lam[p] * wp.ddot(G, H) * G
                - cache.coefficient[p] * G * wp.transpose(H) * G
            ) * wp.transpose(A)
            dP = dP + increment * cache.active_stress[p]
            weighted = -model.alpha[p] * model.volume[p] * dP * wp.transpose(state.F[p])
            contact_force = vec()
            contact_moment = mat()
            if cfg.contact != 0:
                for face in range(FACES):
                    if model.contact_area[p, face] > real(0):
                        r = state.F[p] * model.face_offset[face]
                        marker_dv = dv + gradient * r
                        J = cache.marker_jacobian[p, face]
                        change = J * marker_dv
                        if transpose != 0:
                            change = wp.transpose(J) * marker_dv
                        contact_force = contact_force + change
                        contact_moment = contact_moment + wp.outer(change, r)
            for slot in range(S):
                contribution = (weighted + contact_moment) * st.gradient[p, slot] + st.weight[
                    p, slot
                ] * contact_force
                wp.atomic_add(output, st.ids[p, slot], contribution)

    @wp.kernel(module=module, enable_backward=False)
    def diagonal_blocks(
        model: Model,
        state: State,
        st: Support,
        cfg: Parameters,
        cache: Cache,
        blocks: wp.array(dtype=mat),
    ):
        particle, slot = wp.tid()
        if model.alpha[particle] > real(0) and st.valid[particle] != 0:
            row = st.ids[particle, slot]
            gradient = st.gradient[particle, slot]
            F = state.F[particle]
            A = cache.inverse_active[particle]
            G = cache.inverse_transpose[particle]
            block = mat()
            for axis in range(D):
                basis = vec()
                basis[axis] = real(1)
                increment = cfg.dt * wp.outer(basis, gradient) * F
                H = increment * A
                dP = (
                    cache.mu[particle] * H
                    + cache.lam[particle] * wp.ddot(G, H) * G
                    - cache.coefficient[particle] * G * wp.transpose(H) * G
                ) * wp.transpose(A)
                dP = dP + increment * cache.active_stress[particle]
                column = (
                    -model.alpha[particle]
                    * model.volume[particle]
                    * dP
                    * wp.transpose(F)
                    * gradient
                )
                if cfg.contact != 0:
                    for face in range(FACES):
                        if model.contact_area[particle, face] > real(0):
                            r = F * model.face_offset[face]
                            w = st.weight[particle, slot] + wp.dot(gradient, r)
                            column = column + w * w * cache.marker_jacobian[particle, face] * basis
                for component in range(D):
                    block[component, axis] = column[component]
            wp.atomic_add(blocks, row, block)

    @wp.kernel(module=module, enable_backward=False)
    def invert_blocks(
        st: Support,
        cfg: Parameters,
        blocks: wp.array(dtype=mat),
        inverse: wp.array(dtype=mat),
        threshold: real,
        fallback_count: wp.array(dtype=int),
    ):
        row = wp.tid()
        B = eye
        if st.mass[row] > real(0):
            B = eye - cfg.dt / st.mass[row] * blocks[row]
        scale = wp.sqrt(wp.ddot(B, B))
        safe = wp.isfinite(scale) and scale > real(0)
        normal = eye
        if safe:
            normal = B / scale
            determinant = wp.determinant(normal)
            safe = wp.isfinite(determinant) and wp.abs(determinant) > threshold
        result = eye / st.diagonal[row]
        if safe:
            result = wp.inverse(normal) / scale
        else:
            # Only the preconditioner falls back. The equation/tolerance and
            # independently checked true residual never change.
            wp.atomic_add(fallback_count, 0, 1)
        inverse[row] = result

    @wp.kernel(module=module, enable_backward=False)
    def apply_preconditioner(
        blocks: wp.array(dtype=mat),
        transpose: int,
        x: wp.array(dtype=vec),
        y: wp.array(dtype=vec),
        alpha: real,
        beta: real,
        out: wp.array(dtype=vec),
    ):
        row = wp.tid()
        B = blocks[row] + mat()
        if transpose != 0:
            B = wp.transpose(B)
        result = alpha * B * x[row]
        if beta != real(0):
            result = result + beta * y[row]
        out[row] = result

    return SimpleNamespace(
        Cache=Cache,
        build=build,
        apply=apply,
        diagonal_blocks=diagonal_blocks,
        invert_blocks=invert_blocks,
        apply_preconditioner=apply_preconditioner,
    )


class Linearization:
    """Preallocated nondifferentiable factors for an ImplicitStep."""

    def __init__(self, owner: "DeviceModel") -> "None":
        """Allocate local MPM material/contact factors with the owner's fixed shapes."""
        k = owner.k
        self.kernels = cached_kernels(owner.config.dim, owner.precision)
        self.data = self.kernels.Cache()
        for name, shape, dtype in (
            ("inverse_active", owner.n, k.mat),
            ("inverse_transpose", owner.n, k.mat),
            ("active_stress", owner.n, k.mat),
            ("mu", owner.n, k.real),
            ("lam", owner.n, k.real),
            ("coefficient", owner.n, k.real),
            ("marker_jacobian", (owner.n, 2 * owner.config.dim), k.mat),
        ):
            setattr(self.data, name, wp.zeros(shape, dtype=dtype, device=owner.device))

    @property
    def bytes(self) -> "int":
        """Total allocated device bytes in this tangent cache."""
        return sum(a.capacity for a in vars(self.data).values() if isinstance(a, wp.array))
