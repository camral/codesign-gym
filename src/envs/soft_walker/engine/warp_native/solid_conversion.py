"""Export resolved JAX spring, beam, and FEM data into native host bundles.

The converter preserves topology, materials, and initial state. Unsupported
integration or constitutive choices fail explicitly before native execution.
"""

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from ..physics import beam, fem, spring
    from ..physics.common import DynamicsConfig
    from ..physics.spring import SpringConfig
    from .host_types import SolidExportOptions

import numpy as np

from .solid_model import SolidBundle, SolidConfig


def from_jax(
    physics: "str",
    model: "spring.Model | beam.Model | fem.Model",
    state: "spring.State | beam.State | fem.State",
    config: "SpringConfig | DynamicsConfig",
    *,
    dt: "float | None" = None,
) -> "SolidBundle":
    """Export a JAX spring/beam/FEM model into a validated resolved native bundle.

    Unsupported integrators or constitutive settings raise explicitly.
    Topology/material/state values are preserved; no native fallback is used."""
    if physics not in ("spring", "beam", "fem"):
        raise ValueError("physics must be spring, beam, or fem")
    if physics != "spring":
        dynamics = cast("DynamicsConfig", config)
        if dynamics.max_halvings != 0:
            raise ValueError("Export requires max_halvings=0; native bisection is not implemented")
        if dynamics.actuation_model != "active_strain":
            raise ValueError("Native solids require active_strain; actuation is never substituted")
    d = 2 if physics == "spring" else 3
    reference = (
        cast("spring.Model", model).reference
        if physics == "spring"
        else cast("fem.Model", model).mesh.reference
        if physics == "fem"
        else cast("beam.Model", model).topology.reference
    )
    dtype = np.asarray(reference).dtype
    n = len(reference)
    entities = (
        cast("spring.Model", model).edges
        if physics == "spring"
        else cast("fem.Model", model).mesh.cells
        if physics == "fem"
        else cast("beam.Model", model).topology.edges
    )
    ne = len(entities)
    channels = len(state.actuator)
    f = dict(
        reference=np.asarray(reference),
        mass=np.asarray(model.mass),
        inertia=np.ones(n, dtype),
        free=np.broadcast_to(np.asarray(model.active)[:, None], (n, d)).astype(dtype),
        entities=np.asarray(entities, np.int32),
        cells=np.empty((0, 0), np.int32),
        cell_alpha=np.empty(0, dtype),
        stiffness=np.zeros(ne, dtype),
        damping=np.zeros(ne, dtype),
        length=np.ones(ne, dtype),
        routing=np.zeros((ne, channels), dtype),
        extension=np.zeros((ne, d, d), dtype),
        bending=np.zeros((ne, d, d), dtype),
        axis=np.zeros((ne, d), dtype),
        young=np.ones(ne, dtype),
        poisson=np.full(ne, 0.35, dtype),
        strength=np.zeros(ne, dtype),
        direction=np.zeros((ne, d), dtype),
        kind=np.zeros(ne, np.int32),
        channel=np.zeros(ne, np.int32),
        alpha=np.ones(ne, dtype),
        gradients=np.empty((0, 0, d), dtype),
        weights=np.empty((ne, 0), dtype),
        contact_area=np.zeros((n, 1), dtype),
        offsets=np.empty((0, d), dtype),
    )
    terrain = (
        None
        if model.terrain is None
        else {
            k: np.asarray(getattr(model.terrain, k))
            for k in ("normal", "offset", "centers", "half_sizes", "rounding", "active")
        }
    )
    cfg: SolidExportOptions = dict(
        physics=physics,
        grid_shape=(4,) * d,
        origin=(0.0,) * d,
        gravity=config.gravity,
        velocity_ref=config.velocity_ref,
        actuator_tau=config.actuator_tau,
        max_log_stretch=config.max_log_stretch,
        cache_tangent=True,
        krylov_method="adaptive_dynamic",
    )
    if physics == "spring":
        integrator = cast("SpringConfig", config).integrator
        if integrator not in ("backward_euler", "kvaerno3_native"):
            raise ValueError("Export requires explicit backward_euler or kvaerno3_native selection")
        cfg.update(
            dict(
                integrator=integrator,
                dt=cast("SpringConfig", config).dt,
                vertical_axis=1,
                contact_enabled=cast("SpringConfig", config).ground,
                contact_height=cast("SpringConfig", config).ground_height,
                contact_stiffness=1.0,
                contact_width=cast("SpringConfig", config).contact_width,
                friction=cast("SpringConfig", config).friction,
                friction_width=cast("SpringConfig", config).friction_velocity,
                max_newton=cast("SpringConfig", config).newton_max_steps,
                max_backtracks=cast("SpringConfig", config).line_search_steps,
                atol64=cast("SpringConfig", config).root_tol64,
                atol32=cast("SpringConfig", config).root_tol32,
                min_length_ratio=cast("SpringConfig", config).min_length_ratio,
                min_area_ratio=cast("SpringConfig", config).min_area_ratio,
                restart=cast("SpringConfig", config).gmres_restart,
                max_restarts=cast("SpringConfig", config).gmres_max_restarts,
                linear_rtol64=cast("SpringConfig", config).linear_rtol64,
                linear_rtol32=cast("SpringConfig", config).linear_rtol32,
                linear_refinements=cast("SpringConfig", config).gmres_refinement_steps,
                armijo=cast("SpringConfig", config).armijo,
            )
        )
        for dest, src in [
            ("stiffness", "stiffness"),
            ("damping", "damping"),
            ("length", "length"),
            ("routing", "logstrain_map"),
            ("cells", "cells"),
            ("cell_alpha", "occupancy"),
        ]:
            f[dest] = np.asarray(getattr(model, src))
        f["contact_area"] = np.asarray(cast("spring.Model", model).contact_k)[:, None]
        h = min(
            np.array(cast("SpringConfig", config).size) / cast("SpringConfig", config).grid_shape
        )
        area_ref: float | np.float64 = np.prod(cast("SpringConfig", config).size) / np.prod(
            cast("SpringConfig", config).grid_shape
        )
        u = np.asarray(cast("spring.State", state).x) - np.asarray(reference)
    else:
        if dt is None:
            raise ValueError("An explicit physical timestep is required")
        nc = cast("DynamicsConfig", config).newton
        cfg.update(
            dict(
                dt=dt,
                vertical_axis=cast("DynamicsConfig", config).vertical_axis,
                contact_enabled=cast("DynamicsConfig", config).contact.enabled,
                contact_height=cast("DynamicsConfig", config).contact.height,
                contact_stiffness=cast("DynamicsConfig", config).contact.stiffness,
                contact_width=cast("DynamicsConfig", config).contact.width,
                friction=cast("DynamicsConfig", config).contact.friction,
                friction_width=cast("DynamicsConfig", config).contact.velocity_width,
                min_J=cast("DynamicsConfig", config).min_J,
                max_newton=nc.max_steps,
                max_backtracks=nc.backtracks,
                atol64=nc.atol64,
                atol32=nc.atol32,
                linear_rtol64=nc.linear_rtol64,
                linear_rtol32=nc.linear_rtol32,
                armijo=nc.armijo,
                restart=nc.restart,
                max_restarts=nc.max_restarts,
                linear_refinements=nc.linear_refinements,
            )
        )
        u = np.asarray(cast("beam.State | fem.State", state).u)
        if physics == "beam":
            h = float(cast("beam.Model", model).topology.h)
            for dest, src in [
                ("inertia", "inertia"),
                ("extension", "extension"),
                ("bending", "bend"),
                ("routing", "strain_map"),
                ("contact_area", "contact_area"),
            ]:
                f[dest] = np.asarray(getattr(model, src))
            f["axis"] = np.asarray(cast("beam.Model", model).topology.axis)
            f["offsets"] = np.asarray(cast("beam.Model", model).topology.offsets)
            # Beam bundles are resolved per edge, while the JAX design retains
            # voxel material. Preserve the effective equal-half-link moduli.
            i, j = np.asarray(cast("beam.Model", model).topology.edges).T
            young, poisson = (
                np.asarray(model.young),
                np.asarray(cast("beam.Model | fem.Model", model).poisson),
            )
            local_shear = young / (2 * (1 + poisson))
            f["young"] = 2 / (1 / young[i] + 1 / young[j])
            shear = 2 / (1 / local_shear[i] + 1 / local_shear[j])
            f["poisson"] = f["young"] / (2 * shear) - 1
        else:
            h = float(np.min(np.ptp(np.asarray(reference)[np.asarray(entities)[0]], axis=0)))
            for k in ("alpha", "young", "poisson", "strength", "direction", "kind", "channel"):
                f[k] = np.asarray(getattr(model, k))
            f["gradients"] = np.asarray(cast("fem.Model", model).mesh.grad)
            f["weights"] = np.asarray(cast("fem.Model", model).mesh.weights)
            f["free"] *= ~np.asarray(cast("fem.Model", model).fixed)
            f["contact_area"] = np.asarray(cast("beam.Model | fem.Model", model).contact_area)[
                :, None
            ]
        area_ref = h * h
    f = {
        k: np.asarray(v, dtype=np.int32 if k in ("entities", "cells", "kind", "channel") else dtype)
        for k, v in f.items()
    }
    R = (
        np.asarray(cast("beam.State", state).rotation)
        if physics == "beam"
        else np.tile(np.eye(d, dtype=dtype), (n, 1, 1))
    )
    omega = (
        np.asarray(cast("beam.State", state).omega)
        if physics == "beam"
        else np.zeros((n, d), dtype)
    )
    initial = dict(
        u=u.astype(dtype),
        v=np.asarray(state.v),
        rotation=R,
        omega=omega,
        actuator=np.asarray(state.actuator),
        ledger=np.asarray(
            [state.work_signed, state.work_positive, state.dissipated, state.time], dtype
        ),
        status=np.asarray([state.status], np.int32),
    )
    return SolidBundle(f, SolidConfig(**cfg), channels, h, area_ref, terrain, initial).validate()
