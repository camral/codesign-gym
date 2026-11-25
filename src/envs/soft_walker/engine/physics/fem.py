"""Total-Lagrangian structured Hex8, full 2x2x2 integration, lumped mass.

The M2 core is not yet a registered locomotion driver. No reduced integration,
hourglass stabilization, incompressibility fix, imported meshing, or self-contact.
Positions are stored as displacement from reference to reduce cancellation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy.typing import DTypeLike

from ...contracts.fields import POISSON_DEFAULT
from ...contracts.status_codes import BINARY_STATE_FAILED, BINARY_STATE_OK
from ..embodiments.decoders import Fields
from ..types import HostArray
from .common import StepInfo

if TYPE_CHECKING:
    from .terrain import Terrain
from itertools import product
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from ..numerics.solvers import solve_root
from . import contact, material
from .common import DynamicsConfig, info_from_root, surface_exposure


class Mesh(NamedTuple):
    """Reference hexahedra with volume and surface quadrature."""

    reference: jax.Array
    cells: jax.Array
    grad: jax.Array
    weights: jax.Array
    points: jax.Array
    faces: jax.Array
    neighbors: jax.Array
    has_neighbor: jax.Array
    face_area: jax.Array


class Model(NamedTuple):
    """Instantiated nodal mass and spatial element properties."""

    mesh: Mesh
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
    fixed: jax.Array
    contact_area: jax.Array
    channels: jax.Array
    terrain: Terrain | None = None


class State(NamedTuple):
    """Nodal displacement/velocity, actuator state, and accepted-step ledgers."""

    u: jax.Array
    v: jax.Array
    actuator: jax.Array
    work_signed: jax.Array
    work_positive: jax.Array
    dissipated: jax.Array
    time: jax.Array
    status: jax.Array


def structured_mesh(
    shape: tuple[int, ...] = (2, 1, 1),
    size: tuple[float, ...] = (0.1, 0.05, 0.05),
    origin: tuple[float, ...] = (0.0, 0.02, 0.0),
    dtype: DTypeLike = np.float64,
) -> Mesh:
    """Create a structured hex mesh and reference quadrature in metres."""
    if len(shape) != 3 or any(n < 1 for n in shape):
        raise ValueError("Positive 3D element shape required")
    size_array = np.asarray(size)
    origin_array = np.asarray(origin)
    h = size_array / shape
    if np.any(h <= 0):
        raise ValueError("Positive element dimensions required")
    node_shape = tuple(n + 1 for n in shape)
    bits = np.asarray(list(product((0, 1), repeat=3)))
    centers = np.asarray(list(product(*[range(n) for n in shape])))
    cell_indices = centers[:, None, :] + bits[None, :, :]
    cells: HostArray = np.ravel_multi_index(
        tuple(cell_indices[..., i] for i in range(3)), node_shape
    ).astype(np.int32)
    ref = np.asarray(list(product(*[range(n) for n in node_shape]))) * h + origin_array
    xi = np.asarray(list(product((-1 / np.sqrt(3), 1 / np.sqrt(3)), repeat=3)))
    signs = 2 * bits - 1
    grad = np.zeros((8, 8, 3))
    for q in range(8):
        for a in range(8):
            for i in range(3):
                grad[q, a, i] = (
                    signs[a, i]
                    / 4
                    / h[i]
                    * np.prod(1 + signs[a, np.arange(3) != i] * xi[q, np.arange(3) != i])
                )
    neigh = []
    faces = []
    areas = []
    for i in range(3):
        for s in (-1, 1):
            c = centers.copy()
            c[:, i] += s
            good = np.all((c >= 0) & (c < np.asarray(shape)), axis=-1)
            clipped = np.clip(c, 0, np.asarray(shape) - 1)
            index = np.ravel_multi_index(tuple(clipped[:, k] for k in range(3)), shape)
            neigh.append(np.where(good, index, -1))
            faces.append(cells[:, np.flatnonzero(bits[:, i] == (s + 1) // 2)])
            areas.append(np.prod(h[np.arange(3) != i]))
    nbr = np.stack(neigh, axis=1)
    return Mesh(
        jnp.asarray(ref, dtype),
        jnp.asarray(cells),
        jnp.asarray(grad, dtype),
        jnp.full((len(cells), 8), np.prod(h) / 8, dtype),
        jnp.asarray(2 * (centers + 0.5) / shape - 1, dtype),
        jnp.asarray(np.stack(faces, axis=1)),
        jnp.asarray(np.maximum(nbr, 0), jnp.int32),
        jnp.asarray(nbr >= 0),
        jnp.asarray(areas, dtype),
    )


def instantiate(
    mesh: Mesh,
    fields: Fields,
    channels: int = 2,
    poisson: jax.Array | None = None,
    fixed: jax.Array | None = None,
    surface_smoothing: float = 0.0,
) -> Model:
    """Resolve local material fields; an explicit Poisson argument overrides them."""
    a = fields.occupancy
    if poisson is None:
        poisson = getattr(fields, "poisson", None)
    poisson = jnp.broadcast_to(
        jnp.asarray(POISSON_DEFAULT if poisson is None else poisson, a.dtype), a.shape
    )
    n = mesh.reference.shape[0]
    vol = jnp.sum(mesh.weights, axis=-1)
    mass = (
        jnp.zeros(n, a.dtype)
        .at[mesh.cells.reshape(-1)]
        .add(jnp.repeat(a * fields.density * vol / 8, 8))
    )
    fixed = (
        jnp.zeros_like(mesh.reference, dtype=bool)
        if fixed is None
        else jnp.broadcast_to(fixed, mesh.reference.shape)
    )
    exposure = surface_exposure(a, mesh.neighbors, mesh.has_neighbor, surface_smoothing)
    face = exposure * mesh.face_area[None, :]
    area = jnp.zeros(n, a.dtype).at[mesh.faces.reshape(-1)].add(jnp.repeat(face.reshape(-1) / 4, 4))
    return Model(
        mesh,
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
        fixed,
        area,
        jnp.zeros(channels, a.dtype),
    )


def initial_state(model: Model) -> State:
    """Create an at-rest device state with zero work/time ledgers."""
    z = jnp.asarray(0.0, model.mass.dtype)
    return State(
        jnp.zeros_like(model.mesh.reference),
        jnp.zeros_like(model.mesh.reference),
        model.channels,
        z,
        z,
        z,
        z,
        jnp.asarray(BINARY_STATE_OK, jnp.int32),
    )


def deformation_delta(model: Model, u: jax.Array) -> jax.Array:
    """Compute cancellation-safe element deformation increments F-I."""
    safe = jnp.where(model.active[:, None], u, 0.0)
    delta = jnp.einsum("eai,qaj->eqij", safe[model.mesh.cells], model.mesh.grad)
    return jnp.where((model.alpha > 0)[:, None, None, None], delta, 0.0)


def deformation(model: Model, u: jax.Array) -> jax.Array:
    """Compute element deformation gradients at quadrature samples."""
    return jnp.eye(3, dtype=u.dtype) + deformation_delta(model, u)


def amounts(model: Model, a: jax.Array, cfg: DynamicsConfig) -> jax.Array:
    """Map channel actuator state to local bounded log-stretch amounts."""
    return cfg.max_log_stretch * model.strength * a[model.channel]


def density_energy(model: Model, F: jax.Array, a: jax.Array, cfg: DynamicsConfig) -> jax.Array:
    """Evaluate material energy density at element quadrature points."""
    amount = amounts(model, a, cfg)
    return material.energy_density(
        F,
        model.young[:, None],
        model.poisson[:, None],
        amount[:, None],
        model.direction[:, None, :],
        model.kind[:, None],
    )


def elastic_energy(model: Model, u: jax.Array, a: jax.Array, cfg: DynamicsConfig) -> jax.Array:
    """Sum passive reference-volume elastic energy in joules."""
    e = material.energy_from_delta(
        deformation_delta(model, u),
        model.young[:, None],
        model.poisson[:, None],
        amounts(model, a, cfg)[:, None],
        model.direction[:, None, :],
        model.kind[:, None],
    )
    return jnp.sum(model.alpha[:, None] * model.mesh.weights * e)


def internal_force(model: Model, u: jax.Array, a: jax.Array, cfg: DynamicsConfig) -> jax.Array:
    """Assemble element internal forces into nodal force vectors."""
    delta = deformation_delta(model, u)
    if cfg.fem_cache_tangent:
        from .mpm_tangent import cached_particle_stress

        samples = delta.shape[1]

        def expand(value: jax.Array) -> jax.Array:
            """Give each quadrature sample its own copy of the element parameter view."""
            return jnp.broadcast_to(
                value[:, None], (len(value), samples) + value.shape[1:]
            ).reshape((-1,) + value.shape[1:])

        P = cached_particle_stress(
            delta.reshape(-1, 3, 3),
            expand(model.young),
            expand(model.poisson),
            expand(amounts(model, a, cfg)),
            expand(model.direction),
            expand(model.kind),
            jnp.zeros(delta.shape[0] * samples, delta.dtype),
            "active_strain",
        ).reshape(delta.shape)
    else:
        P = material.stress_from_delta(
            delta,
            model.young[:, None],
            model.poisson[:, None],
            amounts(model, a, cfg)[:, None],
            model.direction[:, None, :],
            model.kind[:, None],
        )
    fe = -jnp.einsum(
        "eq,eqij,qaj->eai", model.alpha[:, None] * model.mesh.weights, P, model.mesh.grad
    )
    return jnp.zeros_like(u).at[model.mesh.cells.reshape(-1)].add(fe.reshape(-1, 3))


def force(
    model: Model, u: jax.Array, v: jax.Array, a: jax.Array, cfg: DynamicsConfig
) -> tuple[jax.Array, jax.Array]:
    """Combine internal, gravity, contact forces and contact dissipation."""
    f = internal_force(model, u, a, cfg)
    f = f.at[:, cfg.vertical_axis].add(-model.mass * cfg.gravity)
    cf, _, diss = contact.response(
        model.mesh.reference + u, v, model.contact_area, cfg.contact, model.terrain
    )
    return f + cf, jnp.sum(diss)


def energy_budget(model: Model, state: State, cfg: DynamicsConfig) -> jax.Array:
    """Return kinetic, elastic, gravity, and contact energies in joules."""
    _, uc, _ = contact.response(
        model.mesh.reference + state.u, state.v, model.contact_area, cfg.contact, model.terrain
    )
    return jnp.stack(
        (
            0.5 * jnp.sum(model.mass[:, None] * state.v**2),
            elastic_energy(model, state.u, state.actuator, cfg),
            jnp.sum(
                model.mass * cfg.gravity * (model.mesh.reference + state.u)[:, cfg.vertical_axis]
            ),
            jnp.sum(uc),
        )
    )


def trial(
    model: Model,
    state: State,
    command: jax.Array,
    dt: jax.Array,
    cfg: DynamicsConfig = DynamicsConfig(),
) -> tuple[State, StepInfo]:
    """Attempt one backward-Euler step with the configured implicit adjoint.

    Returns the proposed state and solver diagnostics; retry/rollback decisions
    belong to the caller, so failed solves remain observable.
    """
    free = model.active[:, None] & (~model.fixed)
    old_u = jnp.where(model.active[:, None], state.u, 0.0)
    old_v = jnp.where(free, state.v, 0.0)
    a = command + (state.actuator - command) * jnp.exp(-dt / cfg.actuator_tau)
    m = jnp.where(model.active, model.mass, 1.0)

    def unpack(z: jax.Array) -> jax.Array:
        """Unscale flattened velocities and preserve fixed-node constraints."""
        return jnp.where(free, z.reshape(old_v.shape) * cfg.velocity_ref, 0.0)

    def residual(z: jax.Array) -> jax.Array:
        """Return the flattened implicit nodal momentum residual, retaining inactive constraints."""
        v = unpack(z)
        f, _ = force(model, old_u + dt * v, v, a, cfg)
        r = (v - old_v - dt * f / m[:, None]) / cfg.velocity_ref
        return jnp.where(free, r, z.reshape(old_v.shape)).reshape(-1)

    def admissible(z: jax.Array) -> jax.Array:
        """Require each active element to retain an admissible deformation determinant."""
        return jnp.min(material.determinant(deformation(model, old_u + dt * unpack(z)))) > cfg.min_J

    # Conservative local stiffness bound, not an assembled global tangent.
    mu, lam = material.lame(model.young, model.poisson)
    ke = (
        model.alpha[:, None]
        * (lam + 2 * mu)[:, None]
        * jnp.einsum("eq,qaj,qaj->ea", model.mesh.weights, model.mesh.grad, model.mesh.grad)
    )
    kd = jnp.zeros_like(model.mass).at[model.mesh.cells.reshape(-1)].add(ke.reshape(-1))
    diag = jnp.broadcast_to(
        (
            1
            + dt
            * dt
            * (kd + (cfg.contact.stiffness if cfg.contact.enabled else 0.0) * model.contact_area)
            / m
        )[:, None],
        old_v.shape,
    )
    diag = jnp.where(free, diag, 1.0).reshape(-1)
    guess = (old_v / cfg.velocity_ref).reshape(-1)
    guess = jnp.where(admissible(guess), guess, jnp.zeros_like(guess))
    z, root = solve_root(residual, guess, cfg.newton, diagonal=diag, admissible=admissible)
    v = unpack(z)
    u = old_u + dt * v
    F = deformation(model, u)
    minJ = jnp.min(material.determinant(F))
    ok = (
        (root.accepted > 0)
        & (minJ > cfg.min_J)
        & (state.status == BINARY_STATE_OK)
        & jnp.all(jnp.isfinite(v))
    )
    work = (
        model.alpha[:, None]
        * model.mesh.weights
        * material.work_density(
            F,
            model.young[:, None],
            model.poisson[:, None],
            amounts(model, state.actuator, cfg)[:, None],
            amounts(model, a, cfg)[:, None],
            model.direction[:, None, :],
            model.kind[:, None],
            delta_F=deformation_delta(model, u),
        )
    )
    _, diss = force(model, u, v, a, cfg)
    new = State(
        u,
        v,
        a,
        state.work_signed + jnp.sum(work),
        state.work_positive + jnp.sum(jnp.maximum(work, 0.0)),
        state.dissipated + dt * diss,
        state.time + dt,
        jnp.where(ok, 0, 1).astype(jnp.int32),
    )
    new = jax.tree.map(lambda n, o: jnp.where(ok, n, o), new, state)._replace(
        status=jnp.where(ok, BINARY_STATE_OK, BINARY_STATE_FAILED).astype(jnp.int32)
    )
    pen = (
        jnp.max(
            jnp.where(
                model.active,
                jnp.maximum(-contact_gap(model.mesh.reference + u, model.terrain, cfg), 0.0),
                0.0,
            )
        )
        if cfg.contact.enabled
        else jnp.asarray(0.0, u.dtype)
    )
    return new, info_from_root(root, minJ, jnp.asarray(True), pen, ok)


def contact_gap(x: jax.Array, terrain: Terrain | None, cfg: DynamicsConfig) -> jax.Array:
    """Return signed terrain separation in metres; positive means above support."""
    if terrain is None:
        return x[..., cfg.vertical_axis] - cfg.contact.height
    from .terrain import gap_normal

    return gap_normal(x, terrain)[0]
