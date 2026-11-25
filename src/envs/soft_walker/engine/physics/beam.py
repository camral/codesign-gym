"""Corotational voxel-centroid beam lattice with translational/rotational DOFs.

One cube mass/inertia at each occupied voxel, nearest-neighbor beam links.
The link is a midpoint Cosserat/Timoshenko energy (axial, shear, bending,
torsion), not a port of Voxelyze and not continuum-equivalent by declaration.
Large common rigid rotations are objective; relative rotations are restricted
below one radian. Square-section torsion uses polar moment (a stated model
approximation). Contact uses exposed cube face-centroid markers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy.typing import DTypeLike

from ...contracts.fields import POISSON_DEFAULT
from ...contracts.status_codes import BINARY_STATE_FAILED, BINARY_STATE_OK
from ..embodiments.decoders import Fields
from .common import DynamicsConfig, StepInfo
from .physics_constants import (
    BEAM_MAX_RELATIVE_ROTATION,
    BEAM_MAX_STEP_ROTATION,
    BEAM_MIN_LENGTH_RATIO,
    SO3_DENOMINATOR_SQUARED_FLOOR,
    SO3_SERIES_SQUARED_THRESHOLD,
)

if TYPE_CHECKING:
    from .terrain import Terrain

from itertools import product
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from ..numerics.solvers import solve_root
from . import contact
from .common import info_from_root, surface_exposure


class Topology(NamedTuple):
    """Fixed reference lattice and oriented beam links."""

    reference: jax.Array
    points: jax.Array
    edges: jax.Array
    axis: jax.Array
    h: jax.Array
    neighbors: jax.Array
    has_neighbor: jax.Array
    offsets: jax.Array


class Model(NamedTuple):
    """Beam stiffness, inertia, actuation, and exposed-surface quadrature."""

    topology: Topology
    alpha: jax.Array
    young: jax.Array
    poisson: jax.Array
    density: jax.Array
    strength: jax.Array
    direction: jax.Array
    kind: jax.Array
    channel: jax.Array
    mass: jax.Array
    active: jax.Array
    inertia: jax.Array
    extension: jax.Array
    bend: jax.Array
    strain_map: jax.Array
    contact_area: jax.Array
    channels: jax.Array
    terrain: Terrain | None = None


class State(NamedTuple):
    """Translations, rotations, velocities, actuator state, and energy ledgers."""

    u: jax.Array
    v: jax.Array
    rotation: jax.Array
    omega: jax.Array
    actuator: jax.Array
    work_signed: jax.Array
    work_positive: jax.Array
    dissipated: jax.Array
    time: jax.Array
    status: jax.Array


def skew(v: jax.Array) -> jax.Array:
    """Construct the cross-product matrix for vectors ending in dimension three."""
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    o = jnp.zeros_like(x)
    return jnp.stack(
        (jnp.stack((o, -z, y), -1), jnp.stack((z, o, -x), -1), jnp.stack((-y, x, o), -1)), -2
    )


def exp_so3(v: jax.Array) -> jax.Array:
    """Map rotation vectors to SO(3), using stable small-angle coefficients."""
    t2 = jnp.sum(v * v, axis=-1)
    t = jnp.sqrt(jnp.maximum(t2, SO3_DENOMINATOR_SQUARED_FLOOR))
    a = jnp.where(t2 < SO3_SERIES_SQUARED_THRESHOLD, 1 - t2 / 6 + t2 * t2 / 120, jnp.sin(t) / t)
    b = jnp.where(
        t2 < SO3_SERIES_SQUARED_THRESHOLD, 0.5 - t2 / 24 + t2 * t2 / 720, (1 - jnp.cos(t)) / (t * t)
    )
    K = skew(v)
    return jnp.eye(3, dtype=v.dtype) + a[..., None, None] * K + b[..., None, None] * (K @ K)


def log_so3(R: jax.Array) -> jax.Array:
    """Map admissible rotation matrices to rotation vectors in radians."""
    v = 0.5 * jnp.stack(
        (R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]), -1
    )
    s2 = jnp.sum(v * v, axis=-1)
    s = jnp.sqrt(jnp.maximum(s2, SO3_DENOMINATOR_SQUARED_FLOOR))
    c = (jnp.trace(R, axis1=-2, axis2=-1) - 1) / 2
    angle = jnp.arctan2(s, c)
    # arcsin(s)/s is valid for the accepted (<1 rad) relative-angle domain.
    f = jnp.where(s2 < SO3_SERIES_SQUARED_THRESHOLD, 1 + s2 / 6 + 3 * s2 * s2 / 40, angle / s)
    return f[..., None] * v


def topology(
    shape: tuple[int, ...], size: tuple[float, ...], origin: tuple[float, ...], dtype: DTypeLike
) -> Topology:
    """Create a beam lattice with fixed reference nodes and oriented links."""
    shape_array = np.asarray(shape)
    h = np.asarray(size) / shape_array
    if shape_array.shape != (3,) or np.any(shape_array < 1) or not np.allclose(h, h[0]):
        raise ValueError("Beam voxels must be cubic in a positive 3D grid")
    ix = np.asarray(list(product(*[range(n) for n in shape_array])))
    ref = (ix + 0.5) * h + np.asarray(origin)
    edges = []
    axes = []
    neighbors = []
    offsets = []
    for k in range(3):
        for sign in (-1, 1):
            nxt = ix.copy()
            nxt[:, k] += sign
            valid = np.all((nxt >= 0) & (nxt < shape_array), axis=-1)
            nid = np.ravel_multi_index(tuple(np.clip(nxt, 0, shape_array - 1).T), shape_array)
            neighbors.append(np.where(valid, nid, -1))
            off = np.zeros(3)
            off[k] = sign * h[k] / 2
            offsets.append(off)
            if sign == 1:
                for i, j in zip(np.where(valid)[0], nid[valid]):
                    edges.append((i, j))
                    axes.append(np.eye(3)[k])
    nb = np.stack(neighbors, axis=1)
    return Topology(
        jnp.asarray(ref, dtype),
        jnp.asarray(2 * (ix + 0.5) / shape_array - 1, dtype),
        jnp.asarray(edges, np.int32).reshape(-1, 2),
        jnp.asarray(axes, dtype).reshape(-1, 3),
        jnp.asarray(h[0], dtype),
        jnp.asarray(np.maximum(nb, 0), jnp.int32),
        jnp.asarray(nb >= 0),
        jnp.asarray(offsets, dtype),
    )


def instantiate(
    t: Topology, fields: Fields, channels: int = 2, poisson: jax.Array | None = None
) -> Model:
    """Resolve voxel material and equal-half-link axial/shear compliances.

    The optional Poisson argument overrides the spatial field for legacy callers.
    A link's two half-length materials act in series, so both Young and shear
    moduli use harmonic endpoint averages. Poisson itself is not averaged.
    """
    a = fields.occupancy
    if poisson is None:
        poisson = getattr(fields, "poisson", None)
    poisson = jnp.broadcast_to(
        jnp.asarray(POISSON_DEFAULT if poisson is None else poisson, a.dtype), a.shape
    )
    i, j = t.edges.T
    mass = a * fields.density * t.h**3
    inertia = mass * t.h**2 / 6
    # Continuous material harmonics, explicitly masked before division.
    E = 2 / (1 / fields.young[i] + 1 / fields.young[j])
    local_G = fields.young / (2 * (1 + poisson))
    G = 2 / (1 / local_G[i] + 1 / local_G[j])
    weight = jnp.minimum(a[i], a[j])
    axis = t.axis
    axial = axis[..., :, None] * axis[..., None, :]
    perp = jnp.eye(3, dtype=a.dtype) - axial
    A = t.h**2
    second = t.h**4 / 12
    polar = 2 * second
    extension = weight[:, None, None] * (
        E[:, None, None] * A * axial + (5 / 6) * G[:, None, None] * A * perp
    )
    bend = weight[:, None, None] * (
        G[:, None, None] * polar * axial + E[:, None, None] * second * perp
    )

    def projection(index: jax.Array) -> jax.Array:
        """Project local actuator directions onto a beam axis according to actuator type."""
        p = jnp.sum(fields.direction[index] * axis, axis=-1) ** 2
        return jnp.where(
            fields.actuator_type[index] == 1,
            p,
            jnp.where(fields.actuator_type[index] == 2, 1 / 3, 0.0),
        )

    pi, pj = projection(i), projection(j)
    routing = jax.nn.one_hot(fields.channel, channels, dtype=a.dtype) * fields.strength[:, None]
    strain = 0.5 * (pi[:, None] * routing[i] + pj[:, None] * routing[j])
    area = surface_exposure(a, t.neighbors, t.has_neighbor) * t.h**2
    return Model(
        t,
        a,
        fields.young,
        poisson,
        fields.density,
        fields.strength,
        fields.direction,
        fields.actuator_type,
        fields.channel,
        mass,
        mass > 0,
        inertia,
        extension,
        bend,
        strain,
        area,
        jnp.zeros(channels, a.dtype),
    )


def initial_state(model: Model) -> State:
    """Create an at-rest device state with zero work/time ledgers."""
    p = model.topology.reference
    z = jnp.asarray(0.0, p.dtype)
    return State(
        jnp.zeros_like(p),
        jnp.zeros_like(p),
        jnp.broadcast_to(jnp.eye(3, dtype=p.dtype), (len(p), 3, 3)),
        jnp.zeros_like(p),
        model.channels,
        z,
        z,
        z,
        z,
        jnp.asarray(BINARY_STATE_OK, jnp.int32),
    )


def strains(
    model: Model, u: jax.Array, R: jax.Array, a: jax.Array, cfg: DynamicsConfig
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Return local beam extension/shear, curvature, and preferred active geometry."""
    t = model.topology
    i, j = t.edges.T
    u = jnp.where(model.active[:, None], u, 0.0)
    R = jnp.where(model.active[:, None, None], R, jnp.eye(3, dtype=u.dtype))
    relative = jnp.swapaxes(R[i], -1, -2) @ R[j]
    rv = log_so3(relative)
    mid = R[i] @ exp_so3(0.5 * rv)
    # Work with the small displacement/rotation strain rather than subtracting
    # two nearly unit stretches. Rest links are exactly h*axis by construction.
    # This is algebraically identical but retains float32 actuator/density
    # sensitivity near the undeformed state.
    du = u[j] - u[i]
    local_delta = (
        jnp.einsum("eji,ej->ei", mid - jnp.eye(3, dtype=u.dtype), t.axis)
        + jnp.einsum("eji,ej->ei", mid, du) / t.h
    )
    log_preferred = cfg.max_log_stretch * (model.strain_map @ a)
    preferred = jnp.exp(log_preferred)
    gamma = local_delta - jnp.expm1(log_preferred)[:, None] * t.axis
    return (gamma, rv / t.h, t.axis + local_delta, preferred)


def link_energy(
    model: Model, u: jax.Array, R: jax.Array, a: jax.Array, cfg: DynamicsConfig
) -> jax.Array:
    """Compute each beam link elastic energy in joules."""
    gamma, kappa, _, _ = strains(model, u, R, a, cfg)
    return (
        0.5
        * model.topology.h
        * (
            jnp.einsum("ei,eij,ej->e", gamma, model.extension, gamma)
            + jnp.einsum("ei,eij,ej->e", kappa, model.bend, kappa)
        )
    )


def elastic_energy(
    model: Model, u: jax.Array, R: jax.Array, a: jax.Array, cfg: DynamicsConfig
) -> jax.Array:
    """Sum passive reference-volume elastic energy in joules."""
    return jnp.sum(link_energy(model, u, R, a, cfg))


def elastic_force_torque(
    model: Model, u: jax.Array, R: jax.Array, a: jax.Array, cfg: DynamicsConfig
) -> tuple[jax.Array, jax.Array]:
    """Differentiate beam energy with respect to translation and rotation increments."""
    if cfg.beam_cache_tangent:
        from .beam_tangent import elastic_force_torque as cached_force_torque

        return cached_force_torque(model, u, R, a, cfg)

    # Spatial rotation variation, so the returned torque is in world coordinates.
    def energy(du: jax.Array, theta: jax.Array) -> jax.Array:
        """Evaluate elastic energy for the force differentiation boundary."""
        return elastic_energy(model, u + du, exp_so3(theta) @ R, a, cfg)

    f, t = jax.grad(energy, argnums=(0, 1))(jnp.zeros_like(u), jnp.zeros_like(u))
    return (-f, -t)


def surface(
    model: Model, u: jax.Array, R: jax.Array, v: jax.Array, omega: jax.Array, cfg: DynamicsConfig
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Return beam-surface contact forces, torques, energy, dissipation, and positions."""
    r = jnp.einsum("nij,fj->nfi", R, model.topology.offsets)
    x = model.topology.reference[:, None, :] + u[:, None, :] + r
    vel = v[:, None, :] + jnp.cross(omega[:, None, :], r)
    f, en, diss = contact.response(x, vel, model.contact_area, cfg.contact, model.terrain)
    return jnp.sum(f, axis=1), jnp.sum(jnp.cross(r, f), axis=1), jnp.sum(en), jnp.sum(diss), x


def forces(
    model: Model,
    u: jax.Array,
    R: jax.Array,
    v: jax.Array,
    omega: jax.Array,
    a: jax.Array,
    cfg: DynamicsConfig,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Combine beam elastic and contact loads with gravity."""
    f, t = elastic_force_torque(model, u, R, a, cfg)
    cf, ct, _, diss, _ = surface(model, u, R, v, omega, cfg)
    return (f + cf).at[:, cfg.vertical_axis].add(-model.mass * cfg.gravity), t + ct, diss


def energy_budget(model: Model, state: State, cfg: DynamicsConfig) -> jax.Array:
    """Return kinetic, elastic, gravity, and contact energies in joules."""
    _, _, uc, _, _ = surface(model, state.u, state.rotation, state.v, state.omega, cfg)
    return jnp.stack(
        (
            0.5
            * jnp.sum(model.mass[:, None] * state.v**2 + model.inertia[:, None] * state.omega**2),
            elastic_energy(model, state.u, state.rotation, state.actuator, cfg),
            jnp.sum(
                model.mass
                * cfg.gravity
                * (model.topology.reference + state.u)[:, cfg.vertical_axis]
            ),
            uc,
        )
    )


def trial(
    model: Model, state: State, command: jax.Array, dt: jax.Array, cfg: DynamicsConfig
) -> tuple[State, StepInfo]:
    """Attempt one backward-Euler step with the configured implicit adjoint.

    Returns the proposed state and solver diagnostics; retry/rollback decisions
    belong to the caller, so failed solves remain observable.
    """
    active = model.active[:, None]
    state = state._replace(
        u=jnp.where(active, state.u, 0.0),
        v=jnp.where(active, state.v, 0.0),
        omega=jnp.where(active, state.omega, 0.0),
        rotation=jnp.where(active[..., None], state.rotation, jnp.eye(3, dtype=state.v.dtype)),
    )
    n = model.mass.size
    h = model.topology.h
    m = jnp.where(model.active, model.mass, 1.0)
    inertia = jnp.where(model.active, model.inertia, 1.0)
    a = command + (state.actuator - command) * jnp.exp(-dt / cfg.actuator_tau)

    def unpack(z: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Convert scaled six-component unknowns into active linear and angular velocities."""
        zz = z.reshape(n, 6) * cfg.velocity_ref
        return jnp.where(active, zz[:, :3], 0.0), jnp.where(active, zz[:, 3:] / h, 0.0)

    def update(z: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """Reconstruct displacement, rotation, and velocities from scaled root unknowns."""
        v, w = unpack(z)
        u = state.u + dt * v
        R = exp_so3(dt * w) @ state.rotation
        return (u, R, v, w)

    def residual(z: jax.Array) -> jax.Array:
        """Return flattened translational and angular momentum residuals; inactive rows stay fixed."""
        u, R, v, w = update(z)
        f, t, _ = forces(model, u, R, v, w, a, cfg)
        rv = (v - state.v - dt * f / m[:, None]) / cfg.velocity_ref
        rw = h * (w - state.omega - dt * t / inertia[:, None]) / cfg.velocity_ref
        return jnp.where(active, jnp.concatenate((rv, rw), axis=-1), z.reshape(n, 6)).reshape(-1)

    def admissible(z: jax.Array) -> jax.Array:
        """Reject trial link collapse, excessive relative rotation, and oversized rotation
        increments.
        """
        u, R, _, w = update(z)
        i, j = model.topology.edges.T
        d = model.topology.reference[j] - model.topology.reference[i] + u[j] - u[i]
        lengths = jnp.sqrt(jnp.sum(d * d, axis=-1)) / h
        c = (jnp.trace(jnp.swapaxes(R[i], -1, -2) @ R[j], axis1=-2, axis2=-1) - 1) / 2
        links = model.active[i] & model.active[j]
        return (
            jnp.all(jnp.where(links, lengths > BEAM_MIN_LENGTH_RATIO, True))
            & jnp.all(jnp.where(links, c > jnp.cos(BEAM_MAX_RELATIVE_ROTATION), True))
            & jnp.all(jnp.sum((dt * w) ** 2, axis=-1) < BEAM_MAX_STEP_ROTATION**2)
        )

    guess = jnp.concatenate((state.v, h * state.omega), axis=-1).reshape(-1) / cfg.velocity_ref
    guess = jnp.where(admissible(guess), guess, jnp.zeros_like(guess))
    # Conservative scale diagonal; no global dense tangent.
    stiff = jnp.zeros(n, state.v.dtype)
    i, j = model.topology.edges.T
    k = jnp.trace(model.extension, axis1=-2, axis2=-1) / h
    stiff = stiff.at[i].add(k).at[j].add(k)
    diag = jnp.broadcast_to(
        (1 + dt * dt * (stiff + cfg.contact.stiffness * jnp.sum(model.contact_area, axis=-1)) / m)[
            :, None
        ],
        (n, 6),
    ).reshape(-1)
    z, root = solve_root(residual, guess, cfg.newton, diagonal=diag, admissible=admissible)
    u, R, v, w = update(z)
    ok = (
        (root.accepted > 0)
        & (state.status == BINARY_STATE_OK)
        & admissible(z)
        & jnp.all(jnp.isfinite(z))
    )
    gamma, _, _, oldpre = strains(model, u, R, state.actuator, cfg)
    dpre = oldpre * jnp.expm1(cfg.max_log_stretch * (model.strain_map @ (a - state.actuator)))
    dg = dpre[:, None] * model.topology.axis
    work = model.topology.h * (
        0.5 * jnp.einsum("ei,eij,ej->e", dg, model.extension, dg)
        - jnp.einsum("ei,eij,ej->e", gamma, model.extension, dg)
    )
    _, _, _, diss, points = surface(model, u, R, v, w, cfg)
    new = State(
        u,
        v,
        R,
        w,
        a,
        state.work_signed + jnp.sum(work),
        state.work_positive + jnp.sum(jnp.maximum(work, 0.0)),
        state.dissipated + dt * diss,
        state.time + dt,
        jnp.where(ok, 0, 1).astype(jnp.int32),
    )
    new = jax.tree.map(lambda x, y: jnp.where(ok, x, y), new, state)._replace(
        status=jnp.where(ok, BINARY_STATE_OK, BINARY_STATE_FAILED).astype(jnp.int32)
    )
    from .terrain import gap_normal

    gap = (
        points[..., cfg.vertical_axis] - cfg.contact.height
        if model.terrain is None
        else gap_normal(points, model.terrain)[0]
    )
    penetration = (
        jnp.max(jnp.where(model.contact_area > 0, jnp.maximum(-gap, 0.0), 0.0))
        if cfg.contact.enabled
        else jnp.asarray(0.0, u.dtype)
    )
    return new, info_from_root(root, jnp.asarray(1.0, u.dtype), jnp.asarray(True), penetration, ok)
