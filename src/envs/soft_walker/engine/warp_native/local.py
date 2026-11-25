"""Small tensor functions shared by native MPM kernels.

Kernel modules are specialized by dimension and precision. Intermediate tensors
are fresh values: mutation of differentiable matrix components is intentionally
avoided. These functions implement the same reference-volume material and smooth
contact laws as ``continuum.material`` and ``continuum.contact``.
"""

from functools import lru_cache
from types import SimpleNamespace

import warp as wp


@lru_cache(maxsize=4)
def functions(dim: int, precision: str) -> "SimpleNamespace":
    """Return the cached constitutive energy, stress, tangent and work namespace.

    Types and kernel identities depend on the requested physics/dimension and
    precision. Warp DSL annotations must be evaluated in this factory's
    scope; postponed annotations and relocation change kernel construction."""
    if dim not in (2, 3) or precision not in ("float32", "float64"):
        raise ValueError("Expected dimension 2/3 and float32/float64")
    scalar = wp.float32 if precision == "float32" else wp.float64
    vector = wp.types.vector(length=dim, dtype=scalar)
    matrix = wp.types.matrix(shape=(dim, dim), dtype=scalar)
    module = f"envs.soft_walker.engine.warp_native.local_d{dim}_{precision}_v1"
    D = dim
    identity_value = wp.constant(matrix(*[float(i == j) for i in range(dim) for j in range(dim)]))

    @wp.func(module=module)
    def expm1(x: scalar):
        # Warp 1.16 has no expm1 builtin. Preserve small actuator increments.
        value = wp.exp(x) - scalar(1)
        if wp.abs(x) < scalar(0.01):
            p = scalar(1.0 / 40320.0)
            p = scalar(1.0 / 5040.0) + x * p
            p = scalar(1.0 / 720.0) + x * p
            p = scalar(1.0 / 120.0) + x * p
            p = scalar(1.0 / 24.0) + x * p
            p = scalar(1.0 / 6.0) + x * p
            p = scalar(0.5) + x * p
            value = x + x * x * p
        return value

    @wp.func(module=module)
    def log1p(x: scalar):
        value = wp.log(scalar(1) + x)
        if wp.abs(x) < scalar(0.01):
            p = scalar(-1.0 / 8.0)
            p = scalar(1.0 / 7.0) + x * p
            p = scalar(-1.0 / 6.0) + x * p
            p = scalar(1.0 / 5.0) + x * p
            p = scalar(-1.0 / 4.0) + x * p
            p = scalar(1.0 / 3.0) + x * p
            p = scalar(-0.5) + x * p
            value = x + x * x * p
        return value

    @wp.func(module=module)
    def volume_delta(delta: matrix):
        trace = wp.trace(delta)
        nonlinear = wp.determinant(delta)
        if wp.static(D == 3):
            nonlinear = nonlinear + (
                delta[0, 0] * delta[1, 1]
                - delta[0, 1] * delta[1, 0]
                + delta[0, 0] * delta[2, 2]
                - delta[0, 2] * delta[2, 0]
                + delta[1, 1] * delta[2, 2]
                - delta[1, 2] * delta[2, 1]
            )
        return trace + nonlinear, nonlinear

    @wp.func(module=module)
    def projector(direction: vector, kind: int):
        safe = direction
        if wp.dot(safe, safe) <= scalar(1.0e-12):
            safe = vector()
            safe[0] = scalar(1)
        normal = safe / wp.sqrt(wp.dot(safe, safe))
        result = matrix()
        if kind == 1:
            result = wp.outer(normal, normal)
        elif kind == 2:
            result = identity_value / scalar(D)
        return result

    @wp.func(module=module)
    def inverse_active(amount: scalar, direction: vector, kind: int):
        factor = expm1(-amount)
        if kind == 2:
            factor = scalar(D) * expm1(-amount / scalar(D))
        return identity_value + factor * projector(direction, kind)

    @wp.func(module=module)
    def elastic_delta(delta: matrix, amount: scalar, direction: vector, kind: int):
        factor = expm1(-amount)
        if kind == 2:
            factor = scalar(D) * expm1(-amount / scalar(D))
        change = factor * projector(direction, kind)
        return delta + change + delta * change

    @wp.func(module=module)
    def energy(
        delta: matrix, young: scalar, nu: scalar, amount: scalar, direction: vector, kind: int
    ):
        elastic = elastic_delta(delta, amount, direction, kind)
        dv, nonlinear = volume_delta(elastic)
        logarithm = log1p(dv)
        remainder = dv - logarithm
        if wp.abs(dv) < scalar(0.01):
            p = scalar(1.0 / 8.0)
            p = scalar(-1.0 / 7.0) + dv * p
            p = scalar(1.0 / 6.0) + dv * p
            p = scalar(-1.0 / 5.0) + dv * p
            p = scalar(1.0 / 4.0) + dv * p
            p = scalar(-1.0 / 3.0) + dv * p
            p = scalar(0.5) + dv * p
            remainder = dv * dv * p
        mu = young / (scalar(2) * (scalar(1) + nu))
        lam = young * nu / ((scalar(1) + nu) * (scalar(1) - scalar(2) * nu))
        return (
            scalar(0.5)
            * mu
            * (wp.ddot(elastic, elastic) - scalar(2) * nonlinear + scalar(2) * remainder)
            + scalar(0.5) * lam * logarithm * logarithm
        )

    @wp.func(module=module)
    def stress(
        delta: matrix, young: scalar, nu: scalar, amount: scalar, direction: vector, kind: int
    ):
        identity = identity_value
        active_inverse = inverse_active(amount, direction, kind)
        elastic = elastic_delta(delta, amount, direction, kind)
        fe = identity + elastic
        dv, nonlinear = volume_delta(elastic)
        mu = young / (scalar(2) * (scalar(1) + nu))
        lam = young * nu / ((scalar(1) + nu) * (scalar(1) - scalar(2) * nu))
        left_change = elastic + wp.transpose(elastic) + elastic * wp.transpose(elastic)
        pe = (mu * left_change + lam * log1p(dv) * identity) * wp.transpose(wp.inverse(fe))
        return pe * wp.transpose(active_inverse)

    @wp.func(module=module)
    def tangent(
        delta: matrix,
        increment: matrix,
        young: scalar,
        nu: scalar,
        amount: scalar,
        direction: vector,
        kind: int,
    ):
        active_inverse = inverse_active(amount, direction, kind)
        elastic = elastic_delta(delta, amount, direction, kind)
        fe = identity_value + elastic
        h = increment * active_inverse
        inverse_transpose = wp.transpose(wp.inverse(fe))
        dv, nonlinear = volume_delta(elastic)
        mu = young / (scalar(2) * (scalar(1) + nu))
        lam = young * nu / ((scalar(1) + nu) * (scalar(1) - scalar(2) * nu))
        coefficient = lam * log1p(dv) - mu
        trace = wp.ddot(inverse_transpose, h)
        dp = (
            mu * h
            + lam * trace * inverse_transpose
            - coefficient * inverse_transpose * wp.transpose(h) * inverse_transpose
        )
        return dp * wp.transpose(active_inverse)

    @wp.func(module=module)
    def strain_work(
        delta: matrix,
        young: scalar,
        nu: scalar,
        old_amount: scalar,
        new_amount: scalar,
        direction: vector,
        kind: int,
    ):
        q = projector(direction, kind)
        elastic = elastic_delta(delta, old_amount, direction, kind)
        change = new_amount - old_amount
        scale = scalar(1)
        if kind == 2:
            scale = scalar(D)
        x = -change / scale
        b = scale * expm1(x)
        dq = elastic * q
        projected = q + dq
        cross = scalar(2) * wp.ddot(elastic, q) + wp.ddot(elastic, dq)
        remainder = expm1(x) - x
        if wp.abs(x) < scalar(0.01):
            p = scalar(1.0 / 40320.0)
            p = scalar(1.0 / 5040.0) + x * p
            p = scalar(1.0 / 720.0) + x * p
            p = scalar(1.0 / 120.0) + x * p
            p = scalar(1.0 / 24.0) + x * p
            p = scalar(1.0 / 6.0) + x * p
            p = scalar(0.5) + x * p
            remainder = x * x * p
        active = scalar(0)
        if kind != 0:
            active = scalar(1)
        dlog = -change * active
        dv, nonlinear = volume_delta(elastic)
        mu = young / (scalar(2) * (scalar(1) + nu))
        lam = young * nu / ((scalar(1) + nu) * (scalar(1) - scalar(2) * nu))
        return (
            mu
            * (
                b * cross
                + scalar(0.5) * b * b * wp.ddot(projected, projected)
                + scale * remainder * active
            )
            + lam * log1p(dv) * dlog
            + scalar(0.5) * lam * dlog * dlog
        )

    @wp.func(module=module)
    def softplus(x: scalar):
        # Both stable branches differentiate to 1/2 at zero. Combining max
        # and abs gives the wrong derivative there under Warp's tie rules.
        value = scalar(0)
        if x >= scalar(0):
            value = x + log1p(wp.exp(-x))
        else:
            value = log1p(wp.exp(x))
        return value

    @wp.func(module=module)
    def sigmoid(x: scalar):
        result = scalar(1) / (scalar(1) + wp.exp(-wp.abs(x)))
        if x < scalar(0):
            result = scalar(1) - result
        return result

    @wp.kernel(module=module)
    def probe(
        delta: wp.array(dtype=matrix),
        increment: wp.array(dtype=matrix),
        young: wp.array(dtype=scalar),
        nu: wp.array(dtype=scalar),
        amount: wp.array(dtype=scalar),
        direction: wp.array(dtype=vector),
        kind: wp.array(dtype=int),
        output_energy: wp.array(dtype=scalar),
        output_stress: wp.array(dtype=matrix),
        output_tangent: wp.array(dtype=matrix),
    ):
        p = wp.tid()
        output_energy[p] = energy(delta[p], young[p], nu[p], amount[p], direction[p], kind[p])
        output_stress[p] = stress(delta[p], young[p], nu[p], amount[p], direction[p], kind[p])
        output_tangent[p] = tangent(
            delta[p], increment[p], young[p], nu[p], amount[p], direction[p], kind[p]
        )

    return SimpleNamespace(
        scalar=scalar,
        vector=vector,
        matrix=matrix,
        module=module,
        energy=energy,
        stress=stress,
        tangent=tangent,
        projector=projector,
        inverse_active=inverse_active,
        elastic_delta=elastic_delta,
        volume_delta=volume_delta,
        strain_work=strain_work,
        expm1=expm1,
        log1p=log1p,
        softplus=softplus,
        sigmoid=sigmoid,
        probe=probe,
    )
