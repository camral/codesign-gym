"""Small JAX spring lattice reference. All continuous model values are dynamic.

No topology repair, strain projection, global stiffness normalization, or learned
controller is hidden here. Dense primal/adjoint algebra is deliberately limited
by SpringConfig.max_dense_dofs. Large-system solvers are an M2 gate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..types import HostArray

if TYPE_CHECKING:
    from .terrain import Terrain

from collections.abc import Callable
from functools import lru_cache
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import DTypeLike

from ...contracts.status_codes import (
    SPRING_SOLVER_NONFINITE_SOLVE,
    SPRING_SOLVER_NONLINEAR_DIVERGENCE,
    SPRING_SOLVER_NONLINEAR_MAX_STEPS,
    SPRING_SOLVER_ODE_MAX_STEPS,
    SPRING_SOLVER_OK,
    SPRING_SOLVER_OTHER_DIFFRAX_ERROR,
    SPRING_SOLVER_ROOT_NOT_CONVERGED,
    SPRING_SOLVER_SINGULAR_LINEAR_SYSTEM,
    SPRING_STATE_INVALID_GEOMETRY,
    SPRING_STATE_NONFINITE,
    SPRING_STATE_OK,
    SPRING_STATE_SOLVE_FAILED,
)
from ..embodiments.decoders import Fields
from .physics_constants import SPRING_SQUARED_LENGTH_FLOOR
from .spring_config import SpringConfig


class Topology(NamedTuple):
    """Reference connectivity and quadrature; fixed while differentiating a rollout."""

    reference: jax.Array
    points: jax.Array
    cells: jax.Array
    edges: jax.Array
    owners: jax.Array
    owner_mask: jax.Array
    lengths: jax.Array
    directions: jax.Array
    side_nodes: jax.Array
    side_neighbors: jax.Array
    side_has_neighbor: jax.Array
    side_lengths: jax.Array


class Model(NamedTuple):
    """Instantiated nodal mass, edge material, actuation, and optional terrain."""

    reference: jax.Array
    cells: jax.Array
    edges: jax.Array
    mass: jax.Array
    active: jax.Array
    stiffness: jax.Array
    damping: jax.Array
    length: jax.Array
    logstrain_map: jax.Array  # edge x native channel; never padded channels
    contact_k: jax.Array
    occupancy: jax.Array
    young: jax.Array
    density: jax.Array
    strength: jax.Array
    channel_mask: jax.Array
    terrain: Terrain | None = None


class State(NamedTuple):
    """Device-resident spring state and accepted-step work/time ledgers."""

    x: jax.Array
    v: jax.Array
    actuator: jax.Array
    work_positive: jax.Array
    work_signed: jax.Array
    effort: jax.Array
    dissipated: jax.Array
    time: jax.Array
    initial_com: jax.Array
    com_displacement: jax.Array  # accumulated from integrator increments, avoids COM subtraction
    status: jax.Array  # 0 OK; 1 root/ODE failure; 2 geometric invalidity; 3 nonfinite


class Diagnostics(NamedTuple):
    """Aggregated physical-step solve, geometry, and rejection diagnostics."""

    accepted: jax.Array
    failed: jax.Array
    nonlinear_iterations: jax.Array
    max_residual: jax.Array
    min_length_ratio: jax.Array
    min_area_ratio: jax.Array
    max_penetration: jax.Array
    solver_code: jax.Array  # named mapping in SOLVER_CODES


SOLVER_CODES = {
    SPRING_SOLVER_OK: "OK",
    SPRING_SOLVER_ROOT_NOT_CONVERGED: "ROOT_NOT_CONVERGED",
    SPRING_SOLVER_ODE_MAX_STEPS: "ODE_MAX_STEPS",
    SPRING_SOLVER_NONLINEAR_MAX_STEPS: "NONLINEAR_MAX_STEPS",
    SPRING_SOLVER_NONLINEAR_DIVERGENCE: "NONLINEAR_DIVERGENCE",
    SPRING_SOLVER_SINGULAR_LINEAR_SYSTEM: "SINGULAR_LINEAR_SYSTEM",
    SPRING_SOLVER_NONFINITE_SOLVE: "NONFINITE_SOLVE",
    SPRING_SOLVER_OTHER_DIFFRAX_ERROR: "OTHER_DIFFRAX_ERROR",
}


def topology(config: SpringConfig, dtype: DTypeLike) -> Topology:
    """Create fixed spring cells, edges, ownership, and reference-area quadrature."""
    nx, ny = config.grid_shape
    sx, sy = config.size
    dx, dy = (sx / nx, sy / ny)
    reference = np.array(
        [
            [i * dx, j * dy + config.initial_clearance + config.ground_height]
            for i in range(nx + 1)
            for j in range(ny + 1)
        ],
        dtype=dtype,
    )
    node = lambda i, j: i * (ny + 1) + j
    cells = []
    pts = []
    neighbors = []
    for i in range(nx):
        for j in range(ny):
            cells.append([node(i, j), node(i + 1, j), node(i + 1, j + 1), node(i, j + 1)])
            pts.append([2 * (i + 0.5) / nx - 1, 2 * (j + 0.5) / ny - 1])
            neighbors.append(
                [
                    (i * ny + j - 1) if j > 0 else -1,
                    ((i + 1) * ny + j) if i < nx - 1 else -1,
                    (i * ny + j + 1) if j < ny - 1 else -1,
                    ((i - 1) * ny + j) if i > 0 else -1,
                ]
            )
    owners: dict[tuple[int, ...], list[int]] = {}
    for c, n in enumerate(cells):
        for a, b in ((0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3)):
            edge = tuple(sorted((n[a], n[b])))
            owners.setdefault(edge, []).append(c)
    edges = np.array(sorted(owners), np.int32)
    own: HostArray = np.zeros((len(edges), 2), np.int32)
    om = np.zeros((len(edges), 2), dtype=dtype)
    for k, e in enumerate(edges):
        members = owners[tuple(e)]
        own[k, : len(members)] = members
        om[k, : len(members)] = 1
    delta = reference[edges[:, 1]] - reference[edges[:, 0]]
    lengths = np.linalg.norm(delta, axis=-1)
    cells = np.asarray(cells, np.int32)
    sides = cells[:, np.array([[0, 1], [1, 2], [2, 3], [3, 0]])]
    nbr: HostArray = np.asarray(neighbors, np.int32)
    return Topology(
        *[
            jnp.asarray(a)
            for a in (
                reference,
                np.asarray(pts, dtype=dtype),
                cells,
                edges,
                own,
                om,
                lengths,
                delta / lengths[:, None],
                sides,
                np.maximum(nbr, 0),
                nbr >= 0,
                np.asarray([dx, dy, dx, dy], dtype=dtype),
            )
        ]
    )


def instantiate(fields: Fields, topo: Topology, cfg: SpringConfig, channels: int) -> Model:
    """Assemble mass, edge stiffness, damping, and actuator maps from decoded fields."""
    alpha = fields.occupancy
    # NumPy division produces a strongly typed float64 scalar. Cast explicitly
    # so a float32 model stays float32 when JAX's x64 mode is also enabled.
    cell_volume = jnp.asarray(
        cfg.size[0] * cfg.size[1] * cfg.thickness / np.prod(cfg.grid_shape),
        alpha.dtype,
    )
    mass = (
        jnp.zeros(topo.reference.shape[0], dtype=alpha.dtype)
        .at[topo.cells.reshape(-1)]
        .add(jnp.repeat(alpha * fields.density * cell_volume / 4, 4))
    )
    active = mass > 0
    w = alpha[topo.owners] * topo.owner_mask
    wsum = jnp.sum(w, axis=-1)
    compliance = jnp.sum(w / fields.young[topo.owners], axis=-1)
    # Harmonic material mean. Each adjacent cell contributes one half-area link.
    effective_e = wsum / jnp.where(compliance > 0, compliance, 1.0)
    cell_area = cell_volume / cfg.thickness
    stiffness = effective_e * wsum * cfg.thickness * cell_area / (2 * topo.lengths**2)
    mi, mj = mass[topo.edges[:, 0]], mass[topo.edges[:, 1]]
    reduced_mass = mi * mj / jnp.where(mi + mj > 0, mi + mj, 1.0)
    damp_arg = stiffness * reduced_mass
    damping = jnp.where(
        damp_arg > 0, 2 * cfg.damping_ratio * jnp.sqrt(jnp.where(damp_arg > 0, damp_arg, 1.0)), 0.0
    )
    fiber = fields.direction[topo.owners]
    projection = jnp.sum(fiber * topo.directions[:, None, :], axis=-1) ** 2
    kind = fields.actuator_type[topo.owners]
    projection = jnp.where(kind == 1, projection, jnp.where(kind == 2, 0.5, 0.0))
    onehot = jax.nn.one_hot(fields.channel[topo.owners], channels, dtype=alpha.dtype)
    strain_map = jnp.sum(
        (w * fields.strength[topo.owners] * projection)[..., None] * onehot, axis=1
    )
    strain_map = cfg.max_log_stretch * strain_map / jnp.where(wsum[:, None] > 0, wsum[:, None], 1.0)
    channel_mass = jnp.sum(
        alpha[:, None]
        * fields.strength[:, None]
        * jax.nn.one_hot(fields.channel, channels, dtype=alpha.dtype),
        axis=0,
    )
    channel_mask = channel_mass > 0
    neighbor_alpha = jnp.where(topo.side_has_neighbor, alpha[topo.side_neighbors], 0.0)
    surface = (
        jnp.maximum(alpha[:, None] - neighbor_alpha, 0.0)
        * topo.side_lengths[None, :]
        * cfg.thickness
    )
    surface_area = (
        jnp.zeros_like(mass)
        .at[topo.side_nodes.reshape(-1)]
        .add(jnp.repeat(surface.reshape(-1) / 2, 2))
    )
    contact_k = surface_area * cfg.contact_modulus / cfg.length_ref
    return Model(
        topo.reference,
        topo.cells,
        topo.edges,
        mass,
        active,
        stiffness,
        damping,
        topo.lengths,
        strain_map,
        contact_k,
        alpha,
        fields.young,
        fields.density,
        fields.strength,
        channel_mask,
    )


def initial_state(model: Model) -> State:
    """Create an at-rest spring state with zero objective ledgers."""
    z = jnp.asarray(0.0, model.reference.dtype)
    c = com(model, model.reference)
    return State(
        model.reference,
        jnp.zeros_like(model.reference),
        jnp.zeros_like(model.channel_mask, dtype=model.reference.dtype),
        z,
        z,
        z,
        z,
        z,
        c,
        jnp.zeros_like(c),
        jnp.asarray(SPRING_STATE_OK, jnp.int32),
    )


def com(model: Model, x: jax.Array) -> jax.Array:
    """Return the mass-weighted mean of node values."""
    x = jnp.where(model.active[:, None], x, 0.0)
    return jnp.sum(model.mass[:, None] * x, axis=0) / jnp.where(
        jnp.sum(model.mass) > 0, jnp.sum(model.mass), 1.0
    )


def safe_state(model: Model, x: jax.Array, v: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Replace inactive node positions/velocities before force evaluation."""
    return jnp.where(model.active[:, None], x, model.reference), jnp.where(
        model.active[:, None], v, 0.0
    )


def edge_geometry(
    model: Model, x: jax.Array, a: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return edge lengths, unit directions, and preferred actuator lengths."""
    x = jnp.where(model.active[:, None], x, model.reference)
    delta = x[model.edges[:, 1]] - x[model.edges[:, 0]]
    # Guard collapse, not ordinary lengths. Geometric validity still rejects it.
    length = jnp.sqrt(
        jnp.sum(delta * delta, axis=-1) + jnp.asarray(SPRING_SQUARED_LENGTH_FLOOR, x.dtype)
    )
    preferred = model.length * jnp.exp(model.logstrain_map @ a)
    return length, delta / length[:, None], preferred


def edge_energy(model: Model, x: jax.Array, a: jax.Array) -> jax.Array:
    """Return per-edge elastic energy in joules."""
    length, _, preferred = edge_geometry(model, x, a)
    return 0.5 * model.stiffness * (length - preferred) ** 2


def elastic_energy(model: Model, x: jax.Array, a: jax.Array) -> jax.Array:
    """Sum edge elastic energies without changing their evaluation order."""
    return jnp.sum(edge_energy(model, x, a))


def actuation_work(model: Model, x: jax.Array, old_a: jax.Array, new_a: jax.Array) -> jax.Array:
    """Cancellation-resistant exact fixed-position change of each edge energy.

    delta_P = P_old * expm1(delta_log_strain)
    delta_U = -k*(length-P_old)*delta_P + .5*k*delta_P**2.
    An unchanged actuator therefore supplies EXACTLY zero work, even when its
    stored elastic energy is large compared with the increment.
    """
    length, _, old_preferred = edge_geometry(model, x, old_a)
    delta_preferred = old_preferred * jnp.expm1(model.logstrain_map @ (new_a - old_a))
    return model.stiffness * delta_preferred * (0.5 * delta_preferred - (length - old_preferred))


def contact(
    model: Model, x: jax.Array, v: jax.Array, cfg: SpringConfig
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return nodal contact forces, potential energy, and dissipated power."""
    if not cfg.ground:
        return jnp.zeros_like(x), jnp.asarray(0.0, x.dtype), jnp.asarray(0.0, x.dtype)
    if model.terrain is not None:
        from .contact import PlaneContact, response

        law = PlaneContact(
            enabled=True,
            stiffness=1.0,
            width=cfg.contact_width,
            friction=cfg.friction,
            velocity_width=cfg.friction_velocity,
        )
        f, u, d = response(x, v, model.contact_k, law, model.terrain)
        return f, jnp.sum(u), jnp.sum(d)
    depth = cfg.contact_width * jax.nn.softplus((cfg.ground_height - x[:, 1]) / cfg.contact_width)
    gate = jax.nn.sigmoid((cfg.ground_height - x[:, 1]) / cfg.contact_width)
    normal = model.contact_k * depth * gate
    tangent = -cfg.friction * normal * jnp.tanh(v[:, 0] / cfg.friction_velocity)
    force = jnp.stack((tangent, normal), axis=-1)
    energy = jnp.sum(0.5 * model.contact_k * depth**2)
    dissipated = -jnp.sum(tangent * v[:, 0])
    return (force, energy, dissipated)


def forces(
    model: Model, x: jax.Array, v: jax.Array, a: jax.Array, cfg: SpringConfig
) -> tuple[jax.Array, jax.Array]:
    """Assemble active-node elastic, damping, gravity, and contact forces."""
    x, v = safe_state(model, x, v)
    length, normal, preferred = edge_geometry(model, x, a)
    rel = v[model.edges[:, 1]] - v[model.edges[:, 0]]
    speed = jnp.sum(rel * normal, axis=-1)
    if cfg.spring_cache_tangent:
        from .spring_tangent import edge_forces

        f = edge_forces(
            x[model.edges[:, 1]] - x[model.edges[:, 0]],
            rel,
            model.stiffness,
            model.damping,
            preferred,
        )
    else:
        f = (model.stiffness * (length - preferred) + model.damping * speed)[:, None] * normal
    force = jnp.zeros_like(x).at[model.edges[:, 0]].add(f).at[model.edges[:, 1]].add(-f)
    force = force.at[:, 1].add(-cfg.gravity * model.mass)
    cf, _, cd = contact(model, x, v, cfg)
    return jnp.where(model.active[:, None], force + cf, 0.0), jnp.sum(model.damping * speed**2) + cd


def energy_budget(model: Model, state: State, cfg: SpringConfig) -> jax.Array:
    """Return kinetic, elastic, gravity, and contact energies in joules."""
    _, uc, _ = contact(model, state.x, state.v, cfg)
    kinetic = 0.5 * jnp.sum(model.mass[:, None] * state.v**2)
    elastic = elastic_energy(model, state.x, state.actuator)
    potential = jnp.sum(model.mass * cfg.gravity * (state.x[:, 1] - cfg.ground_height))
    return jnp.stack((kinetic, elastic, potential, uc))


def admissibility(
    model: Model, x: jax.Array, cfg: SpringConfig
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Measure minimum edge length, cell area, and maximum contact penetration."""
    delta = x[model.edges[:, 1]] - x[model.edges[:, 0]]
    ratio = jnp.sqrt(jnp.sum(delta**2, axis=-1)) / model.length
    min_length = jnp.min(jnp.where(model.stiffness > 0, ratio, 1.0))
    q = x[model.cells]
    edges = jnp.roll(q, -1, axis=1) - q
    next_edges = jnp.roll(edges, -1, axis=1)
    cross = edges[:, :, 0] * next_edges[:, :, 1] - edges[:, :, 1] * next_edges[:, :, 0]
    area_ref = jnp.asarray(cfg.size[0] * cfg.size[1] / np.prod(cfg.grid_shape), x.dtype)
    min_area = jnp.min(jnp.where(model.occupancy[:, None] > 0, cross / area_ref, 1.0))
    if cfg.ground and model.terrain is not None:
        from .terrain import gap_normal

        gap, _ = gap_normal(x, model.terrain)
        penetration = jnp.max(jnp.where(model.active, jnp.maximum(-gap, 0.0), 0.0))
    else:
        penetration = (
            jnp.max(jnp.where(model.active, jnp.maximum(cfg.ground_height - x[:, 1], 0.0), 0.0))
            if cfg.ground
            else jnp.asarray(0.0, x.dtype)
        )
    return (min_length, min_area, penetration)


def spring_diagonal(model: Model, cfg: SpringConfig) -> jax.Array:
    """Positive nodal stiffness/damping bound for a right preconditioner.

    This changes only Krylov coordinates, not the residual or its acceptance
    tolerance. Shared-node contributions are scattered in parallel. The bound
    is fixed within a substep, and does not assemble an N-by-N Jacobian.
    """
    dt = jnp.asarray(cfg.dt, model.mass.dtype)
    links = dt * (dt * model.stiffness + model.damping)
    diagonal = jnp.zeros_like(model.mass)
    diagonal = diagonal.at[model.edges[:, 0]].add(links)
    diagonal = diagonal.at[model.edges[:, 1]].add(links)
    if cfg.ground:
        diagonal = diagonal + dt * dt * model.contact_k
    mass = jnp.where(model.active, model.mass, 1.0)
    diagonal = jnp.where(model.active, 1.0 + diagonal / mass, 1.0)
    return jnp.repeat(diagonal, model.reference.shape[1])


def solve_root(
    residual: Callable[[jax.Array], jax.Array],
    guess: jax.Array,
    cfg: SpringConfig,
    *,
    diagonal: jax.Array | None = None,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array]]:
    """Safeguarded dense Newton; implicit JVP/VJP of its converged residual.

    Primal stopping uses an infinity norm (padding-invariant). Line search only
    accepts finite residual reduction. Aux statistics are floating arrays to
    comply with lax.custom_root's zero-tangent auxiliary convention.
    """
    if cfg.linear_solver == "gmres":
        from ..numerics.solvers import NewtonConfig, solve_root as matrix_free

        nc = NewtonConfig(
            atol64=cfg.root_tol64,
            atol32=cfg.root_tol32,
            max_steps=cfg.newton_max_steps,
            backtracks=cfg.line_search_steps,
            gmres_method=cfg.gmres_method,
            preconditioner_side="right",
            linear_refinements=cfg.gmres_refinement_steps,
            restart=cfg.gmres_restart,
            max_restarts=cfg.gmres_max_restarts,
            linear_rtol64=cfg.linear_rtol64,
            linear_rtol32=cfg.linear_rtol32,
            armijo=cfg.armijo,
            linear_work_reuse=cfg.gmres_linear_work_reuse,
        )
        y, info = matrix_free(residual, guess, nc, diagonal=diagonal)
        return y, (info.residual, info.iterations, info.accepted)
    tol = cfg.root_tol64 if guess.dtype == jnp.float64 else cfg.root_tol32

    def solve(
        f: Callable[[jax.Array], jax.Array], y0: jax.Array
    ) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array]]:
        """Run the dense Newton reference solve and return residual, iterations, and acceptance."""
        r0 = jnp.max(jnp.abs(f(y0)))

        def cond(c: tuple[jax.Array, ...]) -> jax.Array:
            """Continue Newton iteration only while its residual and iteration budget require it."""
            _, r, i, live = c
            return (r > tol) & (i < cfg.newton_max_steps) & live

        def body(c: tuple[jax.Array, ...]) -> tuple[jax.Array, ...]:
            """Take one Newton update with a first-acceptable residual-decreasing line search."""
            y, r, i, _ = c
            fy = f(y)
            tangent = jax.jacfwd(f)(y)
            step = jnp.linalg.solve(tangent, -fy)

            def trial(
                k: jax.Array, best: tuple[jax.Array, jax.Array, jax.Array]
            ) -> tuple[jax.Array, jax.Array, jax.Array]:
                """Evaluate one backtracking candidate without committing a rejected state."""
                out, rbest, found = best
                candidate = y + (0.5**k) * step
                rc = jnp.max(jnp.abs(f(candidate)))
                accept = (~found) & jnp.isfinite(rc) & (rc < r)
                return (
                    jnp.where(accept, candidate, out),
                    jnp.where(accept, rc, rbest),
                    found | accept,
                )

            # M1 used a fixed fori_loop and evaluated ALL eight candidates,
            # even after the first success. Stop once the same first acceptable
            # trial is found; custom_root still differentiates the final residual.
            def more(c: tuple[jax.Array, ...]) -> jax.Array:
                """Continue backtracking until the first accepted candidate or the configured
                limit.
                """
                return (c[0] < cfg.line_search_steps) & (~c[3])

            def backtrack(c: tuple[jax.Array, ...]) -> tuple[jax.Array, ...]:
                """Try the current backtracking scale and increment the candidate counter."""
                k, out, rr, found = c
                out, rr, found = trial(k, (out, rr, found))
                return (k + 1, out, rr, found)

            _, ynew, rnew, found = jax.lax.while_loop(
                more, backtrack, (jnp.asarray(0), y, r, jnp.asarray(False))
            )
            return (ynew, rnew, i + 1, found)

        y, r, it, live = jax.lax.while_loop(
            cond, body, (y0, r0, jnp.asarray(0, jnp.int32), jnp.asarray(True))
        )
        return y, (r, jnp.asarray(it, y.dtype), jnp.asarray((r <= tol) & jnp.isfinite(r), y.dtype))

    def tangent_solve(g: Callable[[jax.Array], jax.Array], y: jax.Array) -> jax.Array:
        """Solve the linearized root system for the custom implicit derivative."""
        matrix = jax.jacfwd(g)(jnp.zeros_like(y))
        return jnp.linalg.solve(matrix, y)

    return jax.lax.custom_root(residual, guess, solve, tangent_solve, has_aux=True)


def be_trial(
    model: Model, state: State, command: jax.Array, cfg: SpringConfig
) -> tuple[jax.Array, ...]:
    """Propose one backward-Euler substep and report solve acceptance diagnostics."""
    dt = cfg.dt
    a = command + (state.actuator - command) * jnp.exp(-dt / cfg.actuator_tau)
    mass = jnp.where(model.active, model.mass, 1.0)

    def residual(flat_velocity: jax.Array) -> jax.Array:
        """Return the flattened implicit force-balance residual in the configured units."""
        v = flat_velocity.reshape(state.v.shape) * cfg.velocity_ref
        x = state.x + dt * v
        f, _ = forces(model, x, v, a, cfg)
        r = (v - state.v - dt * f / mass[:, None]) / cfg.velocity_ref
        return jnp.where(model.active[:, None], r, v / cfg.velocity_ref).reshape(-1)

    diagonal = spring_diagonal(model, cfg) if cfg.gmres_preconditioner == "diagonal" else None
    result, (residual_value, iters, ok) = solve_root(
        residual,
        state.v.reshape(-1) / cfg.velocity_ref,
        cfg,
        diagonal=diagonal,
    )
    v = result.reshape(state.v.shape) * cfg.velocity_ref
    x = state.x + dt * v
    return (
        x,
        v,
        a,
        residual_value,
        iters,
        ok > 0,
        jnp.where(ok > 0, float(SPRING_SOLVER_OK), float(SPRING_SOLVER_ROOT_NOT_CONVERGED)),
        dt * v,
    )


def kvaerno_trial(
    model: Model, state: State, command: jax.Array, cfg: SpringConfig
) -> tuple[jax.Array, ...]:
    """Propose one Diffrax Kvaerno interval using the legacy integrator path."""
    import diffrax as dfx
    import optimistix as optx

    dt = cfg.dt
    # Normalize this substep's independent variable to [0,1]. Diffrax may solve
    # in vector-field space; without this time scaling an atol intended for a
    # velocity increment was instead applied to acceleration, below float32's
    # cancellation floor. Equations/tableau are unchanged by time rescaling.
    tol = cfg.root_tol64 if state.x.dtype == jnp.float64 else cfg.root_tol32
    a_at = lambda unit_time: (
        command + (state.actuator - command) * jnp.exp(-dt * unit_time / cfg.actuator_tau)
    )
    mass = jnp.where(model.active, model.mass, 1.0)

    def rhs(
        t: jax.Array, y: tuple[jax.Array, jax.Array], args: None
    ) -> tuple[jax.Array, jax.Array]:
        """Evaluate displacement and velocity rates for the existing ODE integration path."""
        # All model/state data are explicit args or traced closure arrays, never
        # Python mutable state. Actions are held over precisely this interval.
        x, v = state.x + y[0] * cfg.length_ref, y[1] * cfg.velocity_ref
        f, _ = forces(model, x, v, a_at(t), cfg)
        return (
            dt * jnp.where(model.active[:, None], v / cfg.length_ref, 0.0),
            dt * f / mass[:, None] / cfg.velocity_ref,
        )

    solver = dfx.Kvaerno3(
        root_finder=optx.Newton(rtol=tol, atol=tol, norm=optx.max_norm),
        root_find_max_steps=cfg.newton_max_steps,
    )
    sol = dfx.diffeqsolve(
        dfx.ODETerm(rhs),
        solver,
        t0=jnp.asarray(0.0, state.x.dtype),
        t1=jnp.asarray(1.0, state.x.dtype),
        dt0=jnp.asarray(1.0, state.x.dtype),
        y0=(jnp.zeros_like(state.x), state.v / cfg.velocity_ref),
        stepsize_controller=dfx.ConstantStepSize(),
        saveat=dfx.SaveAt(t1=True),
        max_steps=2,
        throw=False,
        adjoint=dfx.RecursiveCheckpointAdjoint(),
    )
    ok = sol.result == dfx.RESULTS.successful
    code = jnp.where(ok, float(SPRING_SOLVER_OK), float(SPRING_SOLVER_OTHER_DIFFRAX_ERROR))
    for name, c in (
        ("max_steps_reached", SPRING_SOLVER_ODE_MAX_STEPS),
        ("nonlinear_max_steps_reached", SPRING_SOLVER_NONLINEAR_MAX_STEPS),
        ("nonlinear_divergence", SPRING_SOLVER_NONLINEAR_DIVERGENCE),
        ("singular", SPRING_SOLVER_SINGULAR_LINEAR_SYSTEM),
        ("nonfinite", SPRING_SOLVER_NONFINITE_SOLVE),
        ("nonfinite_input", SPRING_SOLVER_NONFINITE_SOLVE),
    ):
        code = jnp.where(sol.result == getattr(dfx.RESULTS, name), float(c), code)
    # Diffrax does not expose final per-stage root residual/iteration counts.
    # -1 denotes unavailable; never fabricate a zero residual or Newton count.
    missing = jnp.asarray(-1.0, state.x.dtype)
    dx = sol.ys[0][0] * cfg.length_ref
    return (
        state.x + dx,
        sol.ys[1][0] * cfg.velocity_ref,
        a_at(jnp.asarray(1.0, state.x.dtype)),
        missing,
        missing,
        ok,
        code,
        dx,
    )


def native_kvaerno_trial(
    model: Model, state: State, command: jax.Array, cfg: SpringConfig
) -> tuple[jax.Array, ...]:
    """Propose one native Kvaerno interval with matrix-free stage adjoints."""
    from ..numerics.integrators import kvaerno3_step
    from ..numerics.solvers import NewtonConfig

    dt = cfg.dt
    n = state.x.size
    mass = jnp.where(model.active, model.mass, 1.0)

    def a_at(t: jax.Array) -> jax.Array:
        """Evaluate the exact exponential actuator response at one substep time."""
        return command + (state.actuator - command) * jnp.exp(-dt * t / cfg.actuator_tau)

    def rhs(t: jax.Array, y: jax.Array, args: None) -> jax.Array:
        """Evaluate displacement and velocity rates for the existing ODE integration path."""
        dx = y[:n].reshape(state.x.shape) * cfg.length_ref
        v = y[n:].reshape(state.v.shape) * cfg.velocity_ref
        f, _ = forces(model, state.x + dx, v, a_at(t), cfg)
        return jnp.concatenate(
            (
                (dt * v / cfg.length_ref).reshape(-1),
                (dt * f / mass[:, None] / cfg.velocity_ref).reshape(-1),
            )
        )

    y0 = jnp.concatenate((jnp.zeros(n, state.x.dtype), state.v.reshape(-1) / cfg.velocity_ref))
    nc = NewtonConfig(
        atol64=cfg.root_tol64,
        atol32=cfg.root_tol32,
        max_steps=cfg.newton_max_steps,
        backtracks=cfg.line_search_steps,
        gmres_method=cfg.gmres_method,
        preconditioner_side="right",
        linear_refinements=cfg.gmres_refinement_steps,
        restart=cfg.gmres_restart,
        max_restarts=cfg.gmres_max_restarts,
        linear_rtol64=cfg.linear_rtol64,
        linear_rtol32=cfg.linear_rtol32,
        armijo=cfg.armijo,
        linear_work_reuse=cfg.gmres_linear_work_reuse,
    )
    y, infos = kvaerno3_step(
        rhs, jnp.asarray(0.0, state.x.dtype), y0, jnp.asarray(1.0, state.x.dtype), None, nc
    )
    dx = y[:n].reshape(state.x.shape) * cfg.length_ref
    v = y[n:].reshape(state.v.shape) * cfg.velocity_ref
    ok = jnp.all(infos.accepted > 0)
    return (
        state.x + dx,
        v,
        a_at(1.0),
        jnp.max(infos.residual),
        jnp.sum(infos.iterations),
        ok,
        jnp.where(ok, float(SPRING_SOLVER_OK), float(SPRING_SOLVER_ROOT_NOT_CONVERGED)),
        dx,
    )


def zero_diagnostics(dtype: DTypeLike) -> Diagnostics:
    """Construct neutral per-step diagnostics in the state precision."""
    z = jnp.asarray(0.0, dtype)
    one = jnp.asarray(1.0, dtype)
    return Diagnostics(z, z, z, z, one, one, z, z)


def substep(
    model: Model, state: State, command: jax.Array, cfg: SpringConfig
) -> tuple[State, Diagnostics]:
    """Advance a valid state or preserve a previously failed transaction."""

    def do_step(state: State) -> tuple[State, Diagnostics]:
        """Attempt one substep, then commit valid state and diagnostics or retain failure status."""
        x, v, a, res, it, root_ok, solver_code, dx = (
            {
                "backward_euler": be_trial,
                "kvaerno3": kvaerno_trial,
                "kvaerno3_native": native_kvaerno_trial,
            }[cfg.integrator]
        )(model, state, command, cfg)
        lr, ar, pen = admissibility(model, x, cfg)
        finite = jnp.all(jnp.isfinite(x)) & jnp.all(jnp.isfinite(v)) & jnp.all(jnp.isfinite(a))
        status = jnp.where(
            ~finite,
            SPRING_STATE_NONFINITE,
            jnp.where(
                ~root_ok,
                SPRING_STATE_SOLVE_FAILED,
                jnp.where(
                    (lr < cfg.min_length_ratio) | (ar < cfg.min_area_ratio),
                    SPRING_STATE_INVALID_GEOMETRY,
                    SPRING_STATE_OK,
                ),
            ),
        ).astype(jnp.int32)
        ok = status == SPRING_STATE_OK
        work = actuation_work(model, x, state.actuator, a)
        _, diss = forces(model, x, v, a, cfg)
        new = State(
            x,
            v,
            a,
            state.work_positive + jnp.sum(jnp.maximum(work, 0.0)),
            state.work_signed + jnp.sum(work),
            state.effort + cfg.dt * jnp.sum(command**2),
            state.dissipated + cfg.dt * diss,
            state.time + cfg.dt,
            state.initial_com,
            state.com_displacement + com(model, dx),
            status,
        )
        # Failed roots never commit partial state. Status remains visible.
        new = jax.tree.map(lambda n, o: jnp.where(ok, n, o), new, state)._replace(status=status)
        diag = Diagnostics(
            ok.astype(x.dtype),
            (~ok).astype(x.dtype),
            it,
            res,
            lr,
            ar,
            pen,
            solver_code.astype(x.dtype),
        )
        return (new, diag)

    return jax.lax.cond(
        state.status == SPRING_STATE_OK, do_step, lambda s: (s, zero_diagnostics(s.x.dtype)), state
    )


def advance(
    model: Model, state: State, command: jax.Array, cfg: SpringConfig
) -> tuple[State, Diagnostics]:
    """Scan one held-action interval and aggregate physical-step diagnostics."""
    x, v = safe_state(model, state.x, state.v)
    state = state._replace(x=x, v=v)
    command = jnp.where(model.channel_mask, command, 0.0)

    def body(s: State, _: None) -> tuple[State, Diagnostics]:
        """Take one Newton update with a first-acceptable residual-decreasing line search."""
        return substep(model, s, command, cfg)

    new, diags = jax.lax.scan(body, state, None, length=cfg.substeps)
    diag = Diagnostics(
        jnp.sum(diags.accepted),
        jnp.sum(diags.failed),
        jnp.sum(diags.nonlinear_iterations),
        jnp.max(diags.max_residual),
        jnp.min(diags.min_length_ratio),
        jnp.min(diags.min_area_ratio),
        jnp.max(diags.max_penetration),
        jnp.max(diags.solver_code),
    )
    if cfg.integrator == "kvaerno3":
        diag = diag._replace(
            max_residual=jnp.asarray(-1.0, state.x.dtype),
            nonlinear_iterations=jnp.asarray(-1.0, state.x.dtype),
        )
    return (new, diag)


def initialize(model: Model, cfg: SpringConfig) -> State:
    """Settle with resting actuation, then reset all task/time ledgers."""
    state = initial_state(model)

    def settle(s: State, _: None) -> tuple[State, Diagnostics]:
        """Advance a zero-command settling interval while discarding its reporting payload."""
        s, diag = advance(model, s, jnp.zeros_like(s.actuator), cfg)
        return (s, diag)

    if cfg.settle_steps:
        state, diags = jax.lax.scan(settle, state, None, length=cfg.settle_steps)
    # Set the evaluation origin AFTER the same deterministic settling protocol.
    z = jnp.asarray(0.0, state.x.dtype)
    return state._replace(
        work_positive=z,
        work_signed=z,
        effort=z,
        dissipated=z,
        time=z,
        initial_com=com(model, state.x),
        com_displacement=jnp.zeros_like(state.com_displacement),
    )


def metrics(model: Model, state: State, cfg: SpringConfig) -> dict[str, jax.Array]:
    """Return scalar task measures and the two-component optimization objective."""
    displacement = state.com_displacement[0]
    objective = jnp.stack((-displacement / cfg.length_ref, state.work_positive / cfg.work_ref))
    return {
        "distance_m": displacement,
        "work_positive_j": state.work_positive,
        "work_signed_j": state.work_signed,
        "effort_s": state.effort,
        "dissipated_j": state.dissipated,
        "time_s": state.time,
        "objective": objective,
        "valid": state.status == SPRING_STATE_OK,
        "status": state.status,
    }


def transition_reward(model: Model, before: State, after: State, cfg: SpringConfig) -> jax.Array:
    """Return normalized forward displacement minus positive actuator work."""
    d = after.com_displacement[0] - before.com_displacement[0]
    return (
        d / cfg.length_ref
        - cfg.energy_weight * (after.work_positive - before.work_positive) / cfg.work_ref
    )


@lru_cache(maxsize=32)
def kernels(
    cfg: SpringConfig,
) -> tuple[
    Callable[[Model, State, jax.Array], tuple[State, Diagnostics]],
    Callable[[Model, State, jax.Array], tuple[State, Diagnostics]],
    Callable[[Model], State],
]:
    """Share executables across env instances with identical static numerics."""
    f = lambda m, s, a: advance(m, s, a, cfg)
    return jax.jit(f), jax.jit(jax.vmap(f)), jax.jit(lambda m: initialize(m, cfg))
