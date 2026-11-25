"""Compressible neo-Hookean elasticity, reference-volume convention.

2D is plane strain: F_3=diag(F_2,1), Fa_3=diag(Fa_2,1), with 3D Lame
parameters. No determinant clamp. Invalid active elements must be rejected by
the nonlinear solver; masked entries are replaced BEFORE inverses/logarithms.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

# Shape convention: tensors end in (..., dim, dim), directions in (..., dim),
# and material/actuator scalar fields in (...). Leading dimensions broadcast.
# Keep the evaluation order of these cancellation-sensitive expressions intact.


def lame(young: jax.Array, poisson: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Return shear and first Lame moduli in Pa from Young modulus and Poisson ratio."""
    young, poisson = jnp.asarray(young), jnp.asarray(poisson)
    return (
        young / (2 * (1 + poisson)),
        young * poisson / ((1 + poisson) * (1 - 2 * poisson)),
    )


def determinant(F: jax.Array) -> jax.Array:
    """Local 2x2/3x3 determinant; never invokes a batched LU kernel."""
    if F.shape[-1] == 2:
        return F[..., 0, 0] * F[..., 1, 1] - F[..., 0, 1] * F[..., 1, 0]
    if F.shape[-1] == 3:
        return jnp.sum(
            F[..., 0, :] * jnp.cross(F[..., 1, :], F[..., 2, :]),
            axis=-1,
        )
    raise ValueError("Only 2D/3D continuum materials are supported")


def inverse_transpose(F: jax.Array) -> jax.Array:
    """Compute F^-T for 2D/3D tensors; singular determinants remain visible."""
    if F.shape[-1] == 2:
        cof = jnp.stack(
            (
                jnp.stack((F[..., 1, 1], -F[..., 1, 0]), axis=-1),
                jnp.stack((-F[..., 0, 1], F[..., 0, 0]), axis=-1),
            ),
            axis=-2,
        )
    elif F.shape[-1] == 3:
        cof = jnp.stack(
            (
                jnp.cross(F[..., 1, :], F[..., 2, :]),
                jnp.cross(F[..., 2, :], F[..., 0, :]),
                jnp.cross(F[..., 0, :], F[..., 1, :]),
            ),
            axis=-2,
        )
    else:
        raise ValueError("Only 2D/3D continuum materials are supported")
    return cof / determinant(F)[..., None, None]


def projector(direction: jax.Array, kind: jax.Array) -> jax.Array:
    """kind 0 passive, 1 fiber, 2 isotropic (trace Q=1 for either active mode)."""
    d = direction.shape[-1]
    norm2 = jnp.sum(direction**2, axis=-1, keepdims=True)
    fallback = jnp.broadcast_to(jnp.eye(d, dtype=direction.dtype)[0], direction.shape)
    safe = jnp.where(norm2 > 1e-12, direction, fallback)
    n = safe / jnp.sqrt(jnp.sum(safe**2, axis=-1, keepdims=True))
    fiber = n[..., :, None] * n[..., None, :]
    eye = jnp.eye(d, dtype=direction.dtype)
    return jnp.where(
        (kind == 1)[..., None, None],
        fiber,
        jnp.where((kind == 2)[..., None, None], eye / d, jnp.zeros_like(fiber)),
    )


def active_stress_from_delta(
    delta_F: jax.Array, stress: jax.Array, direction: jax.Array, kind: jax.Array
) -> jax.Array:
    """Directional active PK1 stress P=F S, S=stress*Q (stress in Pa).

    Positive material stress contracts the body; negative stress extends it.
    F S F^T is Kirchhoff stress, not Cauchy stress (which divides by J).
    This additive stress leaves the passive neo-Hookean material unchanged.
    """
    S = jnp.asarray(stress)[..., None, None] * projector(direction, kind)
    return S + delta_F @ S


def active_stress_potential_from_delta(
    delta_F: jax.Array, stress: jax.Array, direction: jax.Array, kind: jax.Array
) -> jax.Array:
    """Actuation potential .5*S:(F^T F-I), separate from stored elastic energy.

    Its F-gradient is active PK1 stress. This quantity can be negative and its
    change with command at fixed F is not mechanical actuator work.
    """
    S = jnp.asarray(stress)[..., None, None] * projector(direction, kind)
    metric_delta = delta_F + jnp.swapaxes(delta_F, -1, -2) + jnp.swapaxes(delta_F, -1, -2) @ delta_F
    return 0.5 * jnp.sum(S * metric_delta, axis=(-2, -1))


def active_stress_tangent_action(
    increment: jax.Array, stress: jax.Array, direction: jax.Array, kind: jax.Array
) -> jax.Array:
    """Exact local tangent at fixed command: dP=dF S."""
    return increment @ (jnp.asarray(stress)[..., None, None] * projector(direction, kind))


def active_stress_work_density(
    delta_F: jax.Array,
    deformation_increment: jax.Array,
    stress: jax.Array,
    direction: jax.Array,
    kind: jax.Array,
) -> jax.Array:
    """Mechanical work -P_active:dF using the integrator's stress evaluation F.

    delta_F is evaluation F-I, while deformation_increment is the actual step
    increment. Backward Euler uses its endpoint stress; the explicit reference
    uses its starting stress. No command-dependent potential offset is counted.
    """
    P = active_stress_from_delta(delta_F, stress, direction, kind)
    return -jnp.sum(P * deformation_increment, axis=(-2, -1))


def active_inverse(amount: jax.Array, direction: jax.Array, kind: jax.Array) -> jax.Array:
    """exp(-amount Q); rank-one and isotropic cases without matrix exponentials."""
    d = direction.shape[-1]
    q = projector(direction, kind)
    coeff = jnp.where(kind == 2, d * jnp.expm1(-amount / d), jnp.expm1(-amount))
    return jnp.eye(d, dtype=direction.dtype) + coeff[..., None, None] * q


def _volume_terms_delta(D: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """det(I+D)-1 and its nonlinear part, without subtracting two unit values."""
    trace = jnp.trace(D, axis1=-2, axis2=-1)
    if D.shape[-1] == 2:
        nonlinear = determinant(D)
    else:
        quadratic = (
            D[..., 0, 0] * D[..., 1, 1]
            - D[..., 0, 1] * D[..., 1, 0]
            + D[..., 0, 0] * D[..., 2, 2]
            - D[..., 0, 2] * D[..., 2, 0]
            + D[..., 1, 1] * D[..., 2, 2]
            - D[..., 1, 2] * D[..., 2, 1]
        )
        nonlinear = quadratic + determinant(D)
    return (D, trace + nonlinear, nonlinear)


def log_determinant(F: jax.Array) -> jax.Array:
    """Evaluate log(det(F)) accurately near the identity deformation."""
    return jnp.log1p(_volume_terms_delta(F - jnp.eye(F.shape[-1], dtype=F.dtype))[1])


def elastic_delta(
    delta_F: jax.Array, amount: jax.Array, direction: jax.Array, kind: jax.Array
) -> jax.Array:
    """Fe-I without first rounding I+delta_F or expm1(active strain) to unity."""
    d = direction.shape[-1]
    q = projector(direction, kind)
    coeff = jnp.where(kind == 2, d * jnp.expm1(-amount / d), jnp.expm1(-amount))
    delta_A = coeff[..., None, None] * q
    return delta_F + delta_A + delta_F @ delta_A


def energy_from_delta(
    delta_F: jax.Array,
    young: jax.Array,
    poisson: jax.Array,
    amount: jax.Array,
    direction: jax.Array,
    kind: jax.Array,
) -> jax.Array:
    """Compute passive energy density in J/m^3 from cancellation-safe F-I."""
    D = elastic_delta(delta_F, amount, direction, kind)
    mu, lam = lame(young, poisson)
    D, delta, nonlinear = _volume_terms_delta(D)
    logJ = jnp.log1p(delta)

    # Eighth-order stable evaluation of delta-log1p(delta) near the reference.
    series = jnp.asarray(1 / 8.0, D.dtype)
    for k in range(7, 1, -1):
        series = (-1.0) ** k / k + delta * series
    remainder = jnp.where(jnp.abs(delta) < 0.01, delta * delta * series, delta - logJ)
    return (
        0.5 * mu * (jnp.sum(D * D, axis=(-2, -1)) - 2 * nonlinear + 2 * remainder)
        + 0.5 * lam * logJ**2
    )


def energy_density(
    F: jax.Array,
    young: jax.Array,
    poisson: jax.Array,
    amount: jax.Array,
    direction: jax.Array,
    kind: jax.Array,
) -> jax.Array:
    """Compute passive reference-volume energy density for deformation F."""
    return energy_from_delta(
        F - jnp.eye(F.shape[-1], dtype=F.dtype),
        young,
        poisson,
        amount,
        direction,
        kind,
    )


def stress_from_delta(
    delta_F: jax.Array,
    young: jax.Array,
    poisson: jax.Array,
    amount: jax.Array,
    direction: jax.Array,
    kind: jax.Array,
) -> jax.Array:
    """Compute first Piola stress in Pa from cancellation-safe F-I."""
    A = active_inverse(amount, direction, kind)
    D = elastic_delta(delta_F, amount, direction, kind)
    eye = jnp.eye(D.shape[-1], dtype=D.dtype)
    Fe = eye + D
    mu, lam = lame(young, poisson)
    BminusI = D + jnp.swapaxes(D, -1, -2) + D @ jnp.swapaxes(D, -1, -2)
    logJ = jnp.log1p(_volume_terms_delta(D)[1])
    Pe = (mu[..., None, None] * BminusI + (lam * logJ)[..., None, None] * eye) @ inverse_transpose(
        Fe
    )
    return Pe @ jnp.swapaxes(A, -1, -2)


def stress(
    F: jax.Array,
    young: jax.Array,
    poisson: jax.Array,
    amount: jax.Array,
    direction: jax.Array,
    kind: jax.Array,
) -> jax.Array:
    """Return first Piola stress at fixed actuator amount."""
    return stress_from_delta(
        F - jnp.eye(F.shape[-1], dtype=F.dtype),
        young,
        poisson,
        amount,
        direction,
        kind,
    )


def tangent_action(
    F: jax.Array,
    dF: jax.Array,
    young: jax.Array,
    poisson: jax.Array,
    amount: jax.Array,
    direction: jax.Array,
    kind: jax.Array,
) -> jax.Array:
    """Exact local tangent-vector product, no global stiffness tensor."""
    A = active_inverse(amount, direction, kind)
    Fe = F @ A
    H = dF @ A
    G = inverse_transpose(Fe)
    mu, lam = lame(young, poisson)
    q = lam * log_determinant(Fe) - mu
    tr = jnp.sum(G * H, axis=(-2, -1))
    dP = (
        mu[..., None, None] * H
        + lam[..., None, None] * tr[..., None, None] * G
        - q[..., None, None] * (G @ jnp.swapaxes(H, -1, -2) @ G)
    )
    return dP @ jnp.swapaxes(A, -1, -2)


def work_density(
    F: jax.Array,
    young: jax.Array,
    poisson: jax.Array,
    old_amount: jax.Array,
    new_amount: jax.Array,
    direction: jax.Array,
    kind: jax.Array,
    *,
    delta_F: jax.Array | None = None,
) -> jax.Array:
    """Exact fixed-F energy change; unchanged actuation yields exact zero.

    Direction/type fixed within a step. Positive work is accumulated per material
    quadrature location, not after spatial cancellation. No battery model.
    """
    d = F.shape[-1]
    Q = projector(direction, kind)
    delta_F = F - jnp.eye(d, dtype=F.dtype) if delta_F is None else delta_F
    D = elastic_delta(delta_F, old_amount, direction, kind)
    da = new_amount - old_amount
    scale = jnp.where(kind == 2, d, 1.0)
    x = -da / scale
    b = scale * jnp.expm1(x)
    DQ = D @ Q
    B = Q + DQ
    cross_deviation = 2 * jnp.sum(D * Q, axis=(-2, -1)) + jnp.sum(D * DQ, axis=(-2, -1))

    # expm1(x)-x: remove the O(da) cancellation in mechanical work at rest.
    series = jnp.asarray(1 / 40320.0, F.dtype)
    for k in range(7, 1, -1):
        series = 1 / math.factorial(k) + x * series
    rem = jnp.where(jnp.abs(x) < 0.01, x * x * series, jnp.expm1(x) - x)
    active = (kind != 0).astype(F.dtype)
    dlog = -da * active
    logJ = jnp.log1p(_volume_terms_delta(D)[1])
    mu, lam = lame(young, poisson)
    return (
        mu
        * (b * cross_deviation + 0.5 * b * b * jnp.sum(B * B, axis=(-2, -1)) + scale * rem * active)
        + lam * logJ * dlog
        + 0.5 * lam * dlog * dlog
    )
