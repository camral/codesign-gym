"""Reconstruct JAX physics from resolved host bundles without reassembling material.

This boundary copies host arrays onto the current JAX device. It does not allocate
Warp owners or execute either simulator. Resolved stiffness, routing, mass, contact,
and reference quadrature are authoritative. Builder-only metadata that a bundle
does not store is represented by zeros; the result must not be used to infer the
original voxel design or fed back through a material assembler.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from .numerics.solvers import NewtonConfig
from .physics import beam, fem, mpm, spring
from .physics.common import DynamicsConfig
from .physics.contact import PlaneContact
from .physics.spring_config import SpringConfig
from .physics.terrain import Terrain
from .types import PhysicsModel, PhysicsState

if TYPE_CHECKING:
    from .warp_native.model import Bundle
    from .warp_native.solid_model import SolidBundle, SolidConfig


@dataclass(frozen=True)
class ResolvedJaxBody:
    """Resolved JAX arrays plus static integration settings; coordinates remain SI.

    Spring state stores absolute positions; beam/FEM store reference displacement.
    MPM alone requires a background grid. Arrays are newly owned by JAX; changing
    source NumPy buffers afterward cannot change these installed model values.
    """

    model: PhysicsModel
    state: PhysicsState
    configuration: SpringConfig | DynamicsConfig
    grid: mpm.Grid | None = None


def _terrain(bundle: SolidBundle) -> Terrain | None:
    """Copy optional terrain, including an explicit plane for nonlegacy springs."""
    if bundle.terrain is not None:
        return Terrain(*(jnp.array(bundle.terrain[key], copy=True) for key in Terrain._fields))
    c = bundle.config
    if c.physics != "spring" or c.legacy_spring_contact:
        return None
    dtype = bundle.fields["reference"].dtype
    normal = np.zeros(c.dim, dtype)
    normal[c.vertical_axis] = 1
    return Terrain(
        jnp.asarray(normal),
        jnp.asarray(c.contact_height, dtype),
        jnp.zeros((0, c.dim), dtype),
        jnp.zeros((0, c.dim), dtype),
        jnp.zeros(0, dtype),
        jnp.zeros(0, dtype=bool),
    )


def _dynamics(c: SolidConfig) -> DynamicsConfig:
    """Preserve continuum physical scales and checked solver acceptance settings."""
    return DynamicsConfig(
        gravity=c.gravity,
        vertical_axis=c.vertical_axis,
        velocity_ref=c.velocity_ref,
        actuator_tau=c.actuator_tau,
        max_log_stretch=c.max_log_stretch,
        min_J=c.min_J,
        max_halvings=0,
        actuation_model=c.actuation_model,
        active_stress_ratio=c.active_stress_ratio,
        contact=PlaneContact(
            enabled=c.contact_enabled,
            height=c.contact_height,
            stiffness=c.contact_stiffness,
            width=c.contact_width,
            friction=c.friction,
            velocity_width=c.friction_width,
            vertical_axis=c.vertical_axis,
        ),
        newton=NewtonConfig(
            atol64=c.atol64,
            atol32=c.atol32,
            max_steps=c.max_newton,
            backtracks=c.max_backtracks,
            restart=c.restart,
            max_restarts=c.max_restarts,
            linear_rtol64=c.linear_rtol64,
            linear_rtol32=c.linear_rtol32,
            linear_refinements=c.linear_refinements,
            armijo=c.armijo,
            gmres_method="fused",
            preconditioner_side="right",
        ),
    )


def _spring(bundle: SolidBundle, arrays: dict[str, jax.Array]) -> ResolvedJaxBody:
    """Retain resolved springs and normalize area checks to the saved cell area."""
    c, a = bundle.config, arrays
    if c.vertical_axis != 1:
        raise ValueError("The JAX spring model requires vertical_axis=1")
    if not np.array_equal(
        bundle.fields["free"], (bundle.fields["mass"] > 0)[:, None] * np.ones((1, 2))
    ):
        raise ValueError("JAX springs cannot represent additional free-coordinate constraints")
    terrain = _terrain(bundle)
    contact = a["contact_area"][:, 0]
    if terrain is not None:
        contact = contact * c.contact_stiffness
    count = len(a["cells"])
    missing = jnp.zeros(count, a["mass"].dtype)
    model = spring.Model(
        reference=a["reference"],
        cells=a["cells"],
        edges=a["entities"],
        mass=a["mass"],
        active=a["mass"] > 0,
        stiffness=a["stiffness"],
        damping=a["damping"],
        length=a["length"],
        logstrain_map=a["routing"],
        contact_k=contact,
        occupancy=a["cell_alpha"],
        young=missing,
        density=missing,
        strength=missing,
        channel_mask=jnp.ones(bundle.channels, dtype=bool),
        terrain=terrain,
    )
    # Geometry is already resolved; this nominal single cell is used only by
    # geometric admissibility's area_ref = prod(size)/prod(grid_shape).
    side = float(np.sqrt(bundle.area_ref))
    cfg = SpringConfig(
        grid_shape=(1, 1),
        size=(side, side),
        control_dt=c.dt,
        substeps=1,
        settle_steps=0,
        integrator=c.integrator,
        linear_solver="gmres",
        gmres_method="fused",
        gravity=c.gravity,
        ground=c.contact_enabled,
        ground_height=c.contact_height,
        contact_width=c.contact_width,
        friction=c.friction,
        friction_velocity=c.friction_width,
        actuator_tau=c.actuator_tau,
        max_log_stretch=c.max_log_stretch,
        newton_max_steps=c.max_newton,
        line_search_steps=c.max_backtracks,
        root_tol64=c.atol64,
        root_tol32=c.atol32,
        min_length_ratio=c.min_length_ratio,
        min_area_ratio=c.min_area_ratio,
        velocity_ref=c.velocity_ref,
        gmres_restart=c.restart,
        gmres_max_restarts=c.max_restarts,
        gmres_refinement_steps=c.linear_refinements,
        linear_rtol64=c.linear_rtol64,
        linear_rtol32=c.linear_rtol32,
        armijo=c.armijo,
    )
    state = spring.initial_state(model)
    if bundle.initial is not None:
        values = {key: jnp.array(value, copy=True) for key, value in bundle.initial.items()}
        x = a["reference"] + values["u"]
        state = state._replace(
            x=x,
            v=values["v"],
            actuator=values["actuator"],
            work_signed=values["ledger"][0],
            work_positive=values["ledger"][1],
            dissipated=values["ledger"][2],
            time=values["ledger"][3],
            status=values["status"][0],
            initial_com=spring.com(model, x),
        )
    return ResolvedJaxBody(model, state, cfg)


def _beam(bundle: SolidBundle, a: dict[str, jax.Array]) -> ResolvedJaxBody:
    """Retain beam matrices/routing and node inertia without inferring voxel fields."""
    n, dim = a["reference"].shape
    if not np.array_equal(
        bundle.fields["free"], (bundle.fields["mass"] > 0)[:, None] * np.ones((1, dim))
    ):
        raise ValueError("JAX beams cannot represent additional free-coordinate constraints")
    dtype = a["reference"].dtype
    topology = beam.Topology(
        reference=a["reference"],
        points=jnp.zeros((n, dim), dtype),
        edges=a["entities"],
        axis=a["axis"],
        h=jnp.asarray(bundle.h, dtype),
        neighbors=jnp.zeros((n, 2 * dim), jnp.int32),
        has_neighbor=jnp.zeros((n, 2 * dim), bool),
        offsets=a["offsets"],
    )
    missing = jnp.zeros(n, dtype)
    model = beam.Model(
        topology=topology,
        alpha=(a["mass"] > 0).astype(dtype),
        young=missing,
        poisson=missing,
        density=missing,
        strength=missing,
        direction=jnp.zeros((n, dim), dtype),
        kind=jnp.zeros(n, jnp.int32),
        channel=jnp.zeros(n, jnp.int32),
        mass=a["mass"],
        active=a["mass"] > 0,
        inertia=a["inertia"],
        extension=a["extension"],
        bend=a["bending"],
        strain_map=a["routing"],
        contact_area=a["contact_area"],
        channels=jnp.zeros(bundle.channels, dtype),
        terrain=_terrain(bundle),
    )
    state = beam.initial_state(model)
    if bundle.initial is not None:
        values = {key: jnp.array(value, copy=True) for key, value in bundle.initial.items()}
        state = state._replace(
            u=values["u"],
            v=values["v"],
            rotation=values["rotation"],
            omega=values["omega"],
            actuator=values["actuator"],
            work_signed=values["ledger"][0],
            work_positive=values["ledger"][1],
            dissipated=values["ledger"][2],
            time=values["ledger"][3],
            status=values["status"][0],
        )
    return ResolvedJaxBody(model, state, _dynamics(bundle.config))


def _fem(bundle: SolidBundle, a: dict[str, jax.Array]) -> ResolvedJaxBody:
    """Retain Hex8 quadrature, constitutive fields, nodal masses and component locks."""
    n, dim = a["reference"].shape
    count, dtype = len(a["entities"]), a["reference"].dtype
    # Faces and design-sampling metadata are not used after contact_area is resolved.
    mesh = fem.Mesh(
        reference=a["reference"],
        cells=a["entities"],
        grad=a["gradients"],
        weights=a["weights"],
        points=jnp.zeros((count, dim), dtype),
        faces=jnp.zeros((count, 6, 4), jnp.int32),
        neighbors=jnp.zeros((count, 6), jnp.int32),
        has_neighbor=jnp.zeros((count, 6), bool),
        face_area=jnp.zeros(6, dtype),
    )
    model = fem.Model(
        mesh=mesh,
        alpha=a["alpha"],
        young=a["young"],
        poisson=a["poisson"],
        density=jnp.zeros(count, dtype),
        strength=a["strength"],
        direction=a["direction"],
        kind=a["kind"],
        channel=a["channel"],
        mass=a["mass"],
        active=a["mass"] > 0,
        fixed=a["free"] == 0,
        contact_area=a["contact_area"][:, 0],
        channels=jnp.zeros(bundle.channels, dtype),
        terrain=_terrain(bundle),
    )
    state = fem.initial_state(model)
    if bundle.initial is not None:
        values = {key: jnp.array(value, copy=True) for key, value in bundle.initial.items()}
        state = state._replace(
            u=values["u"],
            v=values["v"],
            actuator=values["actuator"],
            work_signed=values["ledger"][0],
            work_positive=values["ledger"][1],
            dissipated=values["ledger"][2],
            time=values["ledger"][3],
            status=values["status"][0],
        )
    return ResolvedJaxBody(model, state, _dynamics(bundle.config))


def to_jax(bundle: Bundle | SolidBundle) -> ResolvedJaxBody:
    """Copy a validated resolved bundle to JAX on the caller's current device.

    Enable JAX x64 before converting float64 storage; precision is never silently
    truncated. Spring/beam coordinate locks beyond zero-mass nodes are rejected
    because their JAX state models lack that constraint contract. Spring auxiliary
    effort/COM-displacement ledgers are absent from the native schema and start at
    zero at this conversion boundary; physical work/time/status are preserved.
    """
    from .warp_native.model import Bundle
    from .warp_native.solid_model import SolidBundle

    bundle.validate()
    if bundle.fields["reference"].dtype == np.float64 and not jax.config.x64_enabled:
        raise ValueError("Enable JAX x64 before converting a float64 resolved bundle")
    if isinstance(bundle, Bundle):
        from .warp_native.mpm_conversion import to_jax as mpm_to_jax

        model, state, grid, cfg = mpm_to_jax(bundle)
        model, state = jax.tree.map(lambda value: jnp.array(value, copy=True), (model, state))
        return ResolvedJaxBody(model, state, cfg, grid)
    if not isinstance(bundle, SolidBundle):
        raise TypeError("Expected a resolved Bundle or SolidBundle")
    arrays = {key: jnp.array(value, copy=True) for key, value in bundle.fields.items()}
    convert = {"spring": _spring, "beam": _beam, "fem": _fem}[bundle.config.physics]
    return convert(bundle, arrays)
