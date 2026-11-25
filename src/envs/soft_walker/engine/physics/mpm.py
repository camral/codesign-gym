"""Quadratic APIC transfers and implicit hyperelastic MPM, uniform sampling.

2D is plane strain with declared thickness. Dimension-generic transfer tests
also run in 3D; the complete 3D locomotion driver remains M3. Grid weights are
frozen WITHIN a substep, but their dependence on incoming particle positions is
fully differentiated across steps. No mass floor, clipped physical grid wall,
particle reseeding, MLS stress approximation, or global tangent assembly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy.typing import DTypeLike

from ...contracts.fields import POISSON_DEFAULT
from ...contracts.status_codes import (
    BINARY_STATE_FAILED,
    BINARY_STATE_OK,
    STEP_COMPACT_CAPACITY_EXCEEDED as COMPACT_CAPACITY_EXCEEDED,
)
from ..embodiments.decoders import Fields
from ..types import HostArray
from .common import StepInfo

if TYPE_CHECKING:
    from .terrain import Terrain
from dataclasses import dataclass
from itertools import product
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from ..numerics.solvers import solve_root
from . import contact, material
from .common import DynamicsConfig, info_from_root, surface_exposure
from .mpm_support import compact_support
from .mpm_tangent import cached_particle_stress, grid_tangent_blocks, inverse_node_blocks


@dataclass(frozen=True)
class Grid:
    """Immutable fixed background grid geometry."""

    shape: tuple[int, ...] = (12, 10)
    spacing: float = 0.025
    origin: tuple[float, ...] = (-0.05, -0.05)

    def __post_init__(self) -> None:
        """Validate fixed grid dimension, spacing, and extents."""
        if (
            len(self.shape) not in (2, 3)
            or len(self.origin) != len(self.shape)
            or min(self.shape) < 4
            or self.spacing <= 0
        ):
            raise ValueError(
                "Grid needs 2D/3D shape, >=4 nodes/axis, positive spacing, matching origin"
            )

    @property
    def count(self) -> int:
        """Number of nodes in the fixed background grid."""
        return int(np.prod(self.shape))


class Cloud(NamedTuple):
    """Reference particles and material face quadrature."""

    reference: jax.Array
    points: jax.Array
    volume: jax.Array
    face_offset: jax.Array
    face_area: jax.Array
    neighbors: jax.Array
    has_neighbor: jax.Array


class Model(NamedTuple):
    """Spatial particle materials, mass, actuation, and contact geometry."""

    cloud: Cloud
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
    contact_area: jax.Array
    channels: jax.Array
    terrain: Terrain | None = None


class State(NamedTuple):
    """Particle positions, velocities, deformation/affine tensors, and ledgers."""

    x: jax.Array
    v: jax.Array
    F: jax.Array
    C: jax.Array
    actuator: jax.Array
    work_signed: jax.Array
    work_positive: jax.Array
    dissipated: jax.Array
    time: jax.Array
    status: jax.Array


class Stencil(NamedTuple):
    """Particle-to-grid indices, weights, gradients, offsets, and valid support."""

    ids: jax.Array
    weight: jax.Array
    gradient: jax.Array
    offset: jax.Array
    valid: jax.Array


def uniform_cloud(
    cell_shape: tuple[int, ...] = (4, 2),
    spacing: float = 0.025,
    origin: tuple[float, ...] = (0.025, 0.025),
    particles_per_axis: int = 2,
    thickness: float = 0.01,
    dtype: DTypeLike = np.float64,
) -> Cloud:
    """Create material particles and exposed-face quadrature at fixed reference spacing."""
    dim = len(cell_shape)
    if (
        dim not in (2, 3)
        or len(origin) != dim
        or min(cell_shape) < 1
        or particles_per_axis < 1
        or spacing <= 0
        or thickness <= 0
    ):
        raise ValueError("Invalid uniform reference cloud")
    sh = tuple(n * particles_per_axis for n in cell_shape)
    hp = spacing / particles_per_axis
    index: HostArray = np.asarray(list(product(*[range(n) for n in sh])), np.int32)
    ref = (index + 0.5) * hp + np.asarray(origin)
    nb = []
    off = []
    for k in range(dim):
        for sign in (-1, 1):
            nn = index.copy()
            nn[:, k] += sign
            valid = np.all((nn >= 0) & (nn < np.asarray(sh)), axis=-1)
            clipped = np.clip(nn, 0, np.asarray(sh) - 1)
            nb.append(
                np.where(
                    valid, np.ravel_multi_index(tuple(clipped[:, i] for i in range(dim)), sh), -1
                )
            )
            rr = np.zeros(dim)
            rr[k] = sign * hp / 2
            off.append(rr)
    nb = np.stack(nb, axis=1)
    factor = thickness if dim == 2 else 1.0
    return Cloud(
        jnp.asarray(ref, dtype),
        jnp.asarray(2 * (index + 0.5) / sh - 1, dtype),
        jnp.full(len(ref), hp**dim * factor, dtype),
        jnp.asarray(off, dtype),
        jnp.full(2 * dim, hp ** (dim - 1) * factor, dtype),
        jnp.asarray(np.maximum(nb, 0), jnp.int32),
        jnp.asarray(nb >= 0),
    )


def instantiate(
    cloud: Cloud,
    fields: Fields,
    channels: int = 2,
    poisson: jax.Array | None = None,
    surface_smoothing: float = 0.0,
) -> Model:
    """Resolve local material fields; an explicit Poisson argument overrides them."""
    a = fields.occupancy
    if poisson is None:
        poisson = getattr(fields, "poisson", None)
    poisson = jnp.broadcast_to(
        jnp.asarray(POISSON_DEFAULT if poisson is None else poisson, a.dtype), a.shape
    )
    area = (
        surface_exposure(a, cloud.neighbors, cloud.has_neighbor, surface_smoothing)
        * cloud.face_area[None, :]
    )
    mass = a * fields.density * cloud.volume
    return Model(
        cloud,
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
        area,
        jnp.zeros(channels, a.dtype),
    )


def initial_state(model: Model) -> State:
    """Create an at-rest device state with zero work/time ledgers."""
    n, d = model.cloud.reference.shape
    z = jnp.asarray(0.0, model.mass.dtype)
    eye = jnp.broadcast_to(jnp.eye(d, dtype=model.mass.dtype), (n, d, d))
    return State(
        model.cloud.reference,
        jnp.zeros_like(model.cloud.reference),
        eye,
        jnp.zeros_like(eye),
        model.channels,
        z,
        z,
        z,
        z,
        jnp.asarray(BINARY_STATE_OK, jnp.int32),
    )


def stencil(x: jax.Array, grid: Grid) -> Stencil:
    """Build particle grid indices, weights, gradients, offsets, and support validity."""
    dim = x.shape[-1]
    u = (x - jnp.asarray(grid.origin, x.dtype)) / grid.spacing
    base = jnp.floor(u - 0.5).astype(jnp.int32)
    f = u - base
    choices = jnp.asarray(list(product(range(3), repeat=dim)), jnp.int32)
    indices = base[:, None, :] + choices[None, :, :]
    good = jnp.all((indices >= 0) & (indices < jnp.asarray(grid.shape)), axis=-1)
    # Safe memory gather only; full-support validity rejects these configurations
    # before commit. Clamping does not introduce a physical wall.
    safe = jnp.clip(indices, 0, jnp.asarray(grid.shape) - 1)
    strides = jnp.asarray([int(np.prod(grid.shape[k + 1 :])) for k in range(dim)], jnp.int32)
    ids = jnp.sum(safe * strides, axis=-1)
    w = jnp.stack((0.5 * (1.5 - f) ** 2, 0.75 - (f - 1.0) ** 2, 0.5 * (f - 0.5) ** 2), axis=-1)
    dw = jnp.stack((f - 1.5, 2 * (1 - f), f - 0.5), axis=-1) / grid.spacing
    factors = [w[:, k, choices[:, k]] for k in range(dim)]
    weight = jnp.prod(jnp.stack(factors, axis=-1), axis=-1)
    gradients = []
    for k in range(dim):
        g = dw[:, k, choices[:, k]]
        for j in range(dim):
            if j != k:
                g = g * factors[j]
        gradients.append(g)
    offset = (indices - u[:, None, :]) * grid.spacing
    return Stencil(ids, weight, jnp.stack(gradients, axis=-1), offset, jnp.all(good, axis=-1))


def p2g(
    mass: jax.Array,
    v: jax.Array,
    C: jax.Array,
    st: Stencil,
    grid: Grid,
    *,
    node_count: int | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Transfer mass and APIC momentum to full or explicitly remapped grid buffers.

    A supplied ``node_count`` is the static compact-buffer length; the caller
    must provide matching remapped stencil indices and masked missing weights.
    Omitting it preserves the full background-grid transfer.

    Float32 momentum is accumulated relative to an active particle velocity.
    This algebraically equivalent form preserves uniform translation despite
    independently rounded mass/momentum scatters. Empty grid nodes stay zero;
    float64 retains the original arithmetic.
    """
    node_count = grid.count if node_count is None else node_count
    reference = None
    if v.dtype == jnp.float32 and v.shape[0] > 0:
        first_active = jnp.argmax(mass > 0)
        reference = jnp.where(
            mass[first_active] > 0, v[first_active], jnp.zeros(v.shape[-1], v.dtype)
        )
    particle_velocity = v if reference is None else v - reference
    local = particle_velocity[:, None, :] + jnp.einsum("pij,pkj->pki", C, st.offset)
    gm = (
        jnp.zeros(node_count, mass.dtype)
        .at[st.ids.reshape(-1)]
        .add((mass[:, None] * st.weight).reshape(-1))
    )
    momentum = (
        jnp.zeros((node_count, v.shape[-1]), v.dtype)
        .at[st.ids.reshape(-1)]
        .add((mass[:, None, None] * st.weight[..., None] * local).reshape(-1, v.shape[-1]))
    )
    gv = momentum / jnp.where(gm > 0, gm, 1.0)[:, None]
    if reference is not None:
        # This is gm / safe_mass without CUDA reciprocal roundoff on gm / gm.
        # The inactive branch retains the original derivative with respect to
        # mass at an empty node, where the safe denominator is held at one.
        gv = gv + jnp.where(gm > 0, 1.0, gm)[:, None] * reference
    return (gm, gv)


def g2p(gv: jax.Array, st: Stencil, grid: Grid) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Interpolate grid velocity, affine velocity, and velocity gradient to particles."""
    local = gv[st.ids]
    v = jnp.sum(st.weight[..., None] * local, axis=1)
    C = 4 / grid.spacing**2 * jnp.einsum("pk,pki,pkj->pij", st.weight, local, st.offset)
    L = jnp.einsum("pki,pkj->pij", local, st.gradient)
    return (v, C, L)


def amounts(model: Model, a: jax.Array, cfg: DynamicsConfig) -> jax.Array:
    """Map channel actuator state to local bounded log-stretch amounts."""
    return cfg.max_log_stretch * model.strength * a[model.channel]


def stress_amounts(model: Model, a: jax.Array, cfg: DynamicsConfig) -> jax.Array:
    """Convert channels to active material stress in Pa for the MPM stress model."""
    mu, _ = material.lame(model.young, model.poisson)
    return cfg.active_stress_ratio * mu * model.strength * a[model.channel]


def actuator_next(
    previous: jax.Array, command: jax.Array, dt: jax.Array, cfg: DynamicsConfig
) -> jax.Array:
    """Advance actuator relaxation for one physical timestep."""
    # Static config branch avoids ever evaluating division by zero.
    if cfg.actuator_tau == 0:
        return command
    return command + (previous - command) * jnp.exp(-dt / cfg.actuator_tau)


def deformation_next(F: jax.Array, L: jax.Array, dt: jax.Array) -> jax.Array:
    """Update particle deformation gradients from the grid velocity gradient."""
    return (jnp.eye(F.shape[-1], dtype=F.dtype) + dt * L) @ F


def elastic_energy(model: Model, F: jax.Array, a: jax.Array, cfg: DynamicsConfig) -> jax.Array:
    """Sum passive reference-volume elastic energy in joules."""
    F = jnp.where(model.active[:, None, None], F, jnp.eye(F.shape[-1], dtype=F.dtype))
    amount = (
        jnp.zeros_like(model.alpha)
        if cfg.actuation_model == "active_stress"
        else amounts(model, a, cfg)
    )
    return jnp.sum(
        model.alpha
        * model.cloud.volume
        * material.energy_density(
            F, model.young, model.poisson, amount, model.direction, model.kind
        )
    )


def constitutive_potential(
    model: Model, F: jax.Array, a: jax.Array, cfg: DynamicsConfig
) -> jax.Array:
    """Force-generating potential; active stress term is not stored energy."""
    energy = elastic_energy(model, F, a, cfg)
    if cfg.actuation_model == "active_stress":
        F = jnp.where(model.active[:, None, None], F, jnp.eye(F.shape[-1], dtype=F.dtype))
        potential = material.active_stress_potential_from_delta(
            F - jnp.eye(F.shape[-1], dtype=F.dtype),
            stress_amounts(model, a, cfg),
            model.direction,
            model.kind,
        )
        energy = energy + jnp.sum(model.alpha * model.cloud.volume * potential)
    return energy


def internal_grid_force(
    model: Model,
    F: jax.Array,
    Fold: jax.Array,
    a: jax.Array,
    st: Stencil,
    grid: Grid,
    cfg: DynamicsConfig,
    *,
    delta_F: jax.Array | None = None,
    node_count: int | None = None,
) -> jax.Array:
    """Scatter particle forces onto full or explicitly remapped compact grid buffers."""
    node_count = grid.count if node_count is None else node_count
    delta = F - jnp.eye(F.shape[-1], dtype=F.dtype) if delta_F is None else delta_F
    if cfg.mpm_cache_tangent:
        P = cached_particle_stress(
            delta,
            model.young,
            model.poisson,
            amounts(model, a, cfg),
            model.direction,
            model.kind,
            stress_amounts(model, a, cfg),
            cfg.actuation_model,
        )
    elif cfg.actuation_model == "active_stress":
        P = material.stress_from_delta(
            delta,
            model.young,
            model.poisson,
            jnp.zeros_like(model.alpha),
            model.direction,
            model.kind,
        )
        P = P + material.active_stress_from_delta(
            delta, stress_amounts(model, a, cfg), model.direction, model.kind
        )
    else:
        P = material.stress_from_delta(
            delta, model.young, model.poisson, amounts(model, a, cfg), model.direction, model.kind
        )
    # The tangent is the derivative of THIS discrete update energy.
    stress = P @ jnp.swapaxes(Fold, -1, -2)
    local = -(model.alpha * model.cloud.volume)[:, None, None] * jnp.einsum(
        "pij,pkj->pki", stress, st.gradient
    )
    return (
        jnp.zeros((node_count, F.shape[-1]), F.dtype)
        .at[st.ids.reshape(-1)]
        .add(local.reshape(-1, F.shape[-1]))
    )


def surface_mapping(model: Model, state: State, st: Stencil) -> tuple[jax.Array, jax.Array]:
    """Reference face-centroid markers advected with each particle's affine F.

    q_next=q_old+dt*sum_i H_i v_i, H_i=w_i+grad(w_i) dot F_old*r_face.
    Thus contact force and its tangent are consistent with marker energy. Face
    weights use REFERENCE area, with no silently changing current-area factor.
    """
    r = jnp.einsum("pij,fj->pfi", state.F, model.cloud.face_offset)
    H = st.weight[:, None, :] + jnp.einsum("pkj,pfj->pfk", st.gradient, r)
    return state.x[:, None, :] + r, H


def grid_contact(
    model: Model,
    q0: jax.Array,
    H: jax.Array,
    gv: jax.Array,
    st: Stencil,
    grid: Grid,
    dt: jax.Array,
    cfg: DynamicsConfig,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Evaluate smooth terrain contact and scatter forces to grid nodes."""
    vel = jnp.einsum("pfk,pki->pfi", H, gv[st.ids])
    q = q0 + dt * vel
    cf, en, diss = contact.response(q, vel, model.contact_area, cfg.contact, model.terrain)
    local = jnp.einsum("pfk,pfi->pki", H, cf)
    force = jnp.zeros_like(gv).at[st.ids.reshape(-1)].add(local.reshape(-1, gv.shape[-1]))
    return force, jnp.sum(en), jnp.sum(diss), q


def sanitize(model: Model, state: State) -> State:
    """Replace inactive particle state with safe reference values before arithmetic."""
    active = model.active
    eye = jnp.eye(state.x.shape[-1], dtype=state.x.dtype)
    return state._replace(
        x=jnp.where(active[:, None], state.x, model.cloud.reference),
        v=jnp.where(active[:, None], state.v, 0.0),
        C=jnp.where(active[:, None, None], state.C, 0.0),
        F=jnp.where(active[:, None, None], state.F, eye),
    )


def trial(
    model: Model,
    state: State,
    command: jax.Array,
    dt: jax.Array,
    grid: Grid,
    cfg: DynamicsConfig = DynamicsConfig(),
) -> tuple[State, StepInfo]:
    """Attempt one backward-Euler step with the configured implicit adjoint.

    Returns the proposed state and solver diagnostics; retry/rollback decisions
    belong to the caller, so failed solves remain observable.
    """
    state = sanitize(model, state)
    st = stencil(state.x, grid)
    node_count = grid.count
    capacity_valid = jnp.asarray(True)
    if cfg.mpm_compact_nodes:
        node_count = min(cfg.mpm_compact_nodes, grid.count, st.ids.size)
        compact = compact_support(st.ids, model.active, grid_nodes=grid.count, capacity=node_count)
        st = st._replace(
            ids=compact.ids,
            weight=jnp.where(compact.covered, st.weight, 0.0),
            gradient=jnp.where(compact.covered[..., None], st.gradient, 0.0),
        )
        capacity_valid = compact.valid
    gm, gv0 = p2g(model.mass, state.v, state.C, st, grid, node_count=node_count)
    active = gm > 0
    mass = jnp.where(active, gm, 1.0)
    a = actuator_next(state.actuator, command, dt, cfg)
    q0, H = surface_mapping(model, state, st)
    support = jnp.all(st.valid | (~model.active)) & capacity_valid
    # Optional equivalent coordinates: y_i = sqrt(m_i / m_ref) * v_i / Vref.
    # This balances tiny-support grid nodes without a mass floor or deleting
    # particles. It changes nonlinear error *scaling*, not the converged root.
    # The original velocity-coordinate path remains the default and is tested.
    reference_mass = jnp.sum(gm) / jnp.maximum(jnp.sum(active), 1)
    root_scale = (
        jnp.sqrt(mass / jnp.maximum(reference_mass, 1e-30))
        if cfg.mpm_mass_weighted
        else jnp.ones_like(mass)
    )
    root_scale = jnp.where(active, root_scale, 1.0)

    def unpack(z: jax.Array) -> jax.Array:
        """Convert scaled active-grid unknowns into the physical grid velocity field."""
        return jnp.where(
            active[:, None], z.reshape(gv0.shape) * cfg.velocity_ref / root_scale[:, None], 0.0
        )

    def updated(gv: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        """Reconstruct particle deformation and velocity from one trial grid velocity field."""
        v, C, L = g2p(gv, st, grid)
        increment = dt * (L @ state.F)
        delta = (state.F - jnp.eye(state.x.shape[-1], dtype=state.x.dtype)) + increment
        delta = jnp.where(model.active[:, None, None], delta, 0.0)
        F = jnp.eye(state.x.shape[-1], dtype=state.x.dtype) + delta
        return (v, C, F, delta, increment)

    def residual(z: jax.Array) -> jax.Array:
        """Return the implicit grid momentum residual with inactive grid rows constrained."""
        gv = unpack(z)
        _, _, F, delta, _ = updated(gv)
        f = internal_grid_force(
            model, F, state.F, a, st, grid, cfg, delta_F=delta, node_count=node_count
        )
        cf, _, _, _ = grid_contact(model, q0, H, gv, st, grid, dt, cfg)
        f = f + cf
        f = f.at[:, cfg.vertical_axis].add(-gm * cfg.gravity)
        r = (gv - gv0 - dt * f / mass[:, None]) / cfg.velocity_ref
        return jnp.where(active[:, None], r * root_scale[:, None], z.reshape(gv0.shape)).reshape(-1)

    def admissible(z: jax.Array) -> jax.Array:
        """Require admissible deformation determinants and valid interpolation support."""
        gv = unpack(z)
        v, _, F, _, _ = updated(gv)
        valid = jnp.min(material.determinant(F)) > cfg.min_J
        # The next position must also have full interpolation support.
        nextst = stencil(state.x + dt * v, grid)
        return support & valid & jnp.all(nextst.valid | (~model.active))

    mu, lam = material.lame(model.young, model.poisson)
    modulus_bound = lam + 2 * mu
    if cfg.actuation_model == "active_stress":
        modulus_bound = modulus_bound + jnp.abs(stress_amounts(model, a, cfg))
    bound = (model.alpha * model.cloud.volume * modulus_bound)[:, None] * jnp.sum(
        st.gradient**2, axis=-1
    )
    kd = jnp.zeros(node_count, gm.dtype).at[st.ids.reshape(-1)].add(bound.reshape(-1))
    ca = jnp.sum(model.contact_area[:, :, None] * H**2, axis=1)
    ck = jnp.zeros(node_count, gm.dtype).at[st.ids.reshape(-1)].add(ca.reshape(-1))
    diag = jnp.broadcast_to(
        (1 + dt * dt * (kd + (cfg.contact.stiffness if cfg.contact.enabled else 0.0) * ck) / mass)[
            :, None
        ],
        gv0.shape,
    )
    diag = jnp.where(active[:, None], diag, 1.0).reshape(-1)
    guess = (gv0 * root_scale[:, None] / cfg.velocity_ref).reshape(-1)
    guess = jnp.where(admissible(guess), guess, jnp.zeros_like(guess))
    inverse_blocks = None
    inverse_blocks_factory = None
    if cfg.mpm_block_preconditioner:

        def build_inverse_blocks(point: jax.Array) -> jax.Array:
            """Build exact node-diagonal blocks at the selected physical predictor/iterate."""
            return inverse_node_blocks(
                grid_tangent_blocks(model, state, a, unpack(point), gm, st, dt, cfg)
            )

        if cfg.mpm_preconditioner_update == "newton":
            inverse_blocks_factory = build_inverse_blocks
        else:
            inverse_blocks = build_inverse_blocks(guess)
    z, root = solve_root(
        residual,
        guess,
        cfg.newton,
        diagonal=diag,
        admissible=admissible,
        inverse_blocks=inverse_blocks,
        inverse_blocks_factory=inverse_blocks_factory,
    )
    gv = unpack(z)
    v, C, F, delta, increment = updated(gv)
    x = state.x + dt * v
    minJ = jnp.min(material.determinant(F))
    support = support & jnp.all(stencil(x, grid).valid | (~model.active))
    ok = (
        (root.accepted > 0)
        & (minJ > cfg.min_J)
        & support
        & (state.status == BINARY_STATE_OK)
        & jnp.all(jnp.isfinite(v))
    )
    if cfg.actuation_model == "active_stress":
        w = (
            model.alpha
            * model.cloud.volume
            * material.active_stress_work_density(
                delta, increment, stress_amounts(model, a, cfg), model.direction, model.kind
            )
        )
    else:
        w = (
            model.alpha
            * model.cloud.volume
            * material.work_density(
                F,
                model.young,
                model.poisson,
                amounts(model, state.actuator, cfg),
                amounts(model, a, cfg),
                model.direction,
                model.kind,
                delta_F=delta,
            )
        )
    _, _, diss, q = grid_contact(model, q0, H, gv, st, grid, dt, cfg)
    new = State(
        x,
        v,
        F,
        C,
        a,
        state.work_signed + jnp.sum(w),
        state.work_positive + jnp.sum(jnp.maximum(w, 0.0)),
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
                model.contact_area > 0, jnp.maximum(-contact_gap(q, model.terrain, cfg), 0.0), 0.0
            )
        )
        if cfg.contact.enabled
        else jnp.asarray(0.0, x.dtype)
    )
    info = info_from_root(root, minJ, support, pen, ok)
    return new, info._replace(code=jnp.where(capacity_valid, info.code, COMPACT_CAPACITY_EXCEEDED))


def explicit_trial(
    model: Model,
    state: State,
    command: jax.Array,
    dt: jax.Array,
    grid: Grid,
    cfg: DynamicsConfig = DynamicsConfig(),
) -> State:
    """Symplectic explicit reference for tiny-dt comparisons, not production.

    Active-stress work uses the starting stress that generated this explicit
    impulse paired with the resulting deformation increment. Legacy active-
    strain reference work remains untracked, as before.
    """
    state = sanitize(model, state)
    st = stencil(state.x, grid)
    gm, gv = p2g(model.mass, state.v, state.C, st, grid)
    a = actuator_next(state.actuator, command, dt, cfg)
    f = internal_grid_force(model, state.F, state.F, a, st, grid, cfg)
    q0, H = surface_mapping(model, state, st)
    cf, _, _, _ = grid_contact(model, q0, H, gv, st, grid, 0.0, cfg)
    f = f + cf
    f = f.at[:, cfg.vertical_axis].add(-gm * cfg.gravity)
    gv = gv + dt * f / jnp.where(gm > 0, gm, 1.0)[:, None]
    v, C, L = g2p(gv, st, grid)
    F = deformation_next(state.F, L, dt)
    x = state.x + dt * v
    new = state._replace(x=x, v=v, C=C, F=F, actuator=a, time=state.time + dt)
    if cfg.actuation_model == "active_stress":
        w = (
            model.alpha
            * model.cloud.volume
            * material.active_stress_work_density(
                state.F - jnp.eye(state.F.shape[-1], dtype=state.F.dtype),
                dt * (L @ state.F),
                stress_amounts(model, a, cfg),
                model.direction,
                model.kind,
            )
        )
        new = new._replace(
            work_signed=state.work_signed + jnp.sum(w),
            work_positive=state.work_positive + jnp.sum(jnp.maximum(w, 0.0)),
        )
    return new


def contact_gap(x: jax.Array, terrain: Terrain | None, cfg: DynamicsConfig) -> jax.Array:
    """Return signed terrain separation in metres; positive means above support."""
    if terrain is None:
        return x[..., cfg.vertical_axis] - cfg.contact.height
    from .terrain import gap_normal

    return gap_normal(x, terrain)[0]
