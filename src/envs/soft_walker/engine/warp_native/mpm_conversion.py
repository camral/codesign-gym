"""Convert resolved MPM models between JAX structures and native host bundles.

Export freezes host data; the reverse conversion imports JAX only when called.
Native simulation stepping does not import this module.
"""

from typing import TYPE_CHECKING, cast

from ...contracts.metrics import (
    LEDGER_DISSIPATED,
    LEDGER_TIME,
    LEDGER_WORK_POSITIVE,
    LEDGER_WORK_SIGNED,
)

if TYPE_CHECKING:
    from ..physics import mpm
    from ..physics.common import DynamicsConfig
    from ..physics.terrain import Terrain

import numpy as np

from .config import MPMConfig
from .model import Bundle


def from_jax(
    model: "mpm.Model",
    state: "mpm.State",
    grid: "mpm.Grid",
    dynamics: "DynamicsConfig",
    *,
    dt: "float",
) -> "Bundle":
    """Freeze resolved data. Unsupported bisection is rejected, not approximated."""
    if dynamics.max_halvings != 0:
        raise ValueError("Export requires max_halvings=0; native bisection is not implemented")
    c, n = dynamics.contact, dynamics.newton
    if n.preconditioner_side != "right":
        raise ValueError(
            "Native slice requires right preconditioning; export does not silently change this choice"
        )
    config = MPMConfig(
        grid_shape=tuple(grid.shape),
        origin=tuple(grid.origin),
        spacing=grid.spacing,
        dt=dt,
        gravity=dynamics.gravity,
        vertical_axis=dynamics.vertical_axis,
        velocity_ref=dynamics.velocity_ref,
        actuator_tau=dynamics.actuator_tau,
        max_log_stretch=dynamics.max_log_stretch,
        min_J=dynamics.min_J,
        actuation_model=dynamics.actuation_model,
        active_stress_ratio=dynamics.active_stress_ratio,
        mass_weighted=dynamics.mpm_mass_weighted,
        contact_enabled=c.enabled,
        contact_height=c.height,
        contact_stiffness=c.stiffness,
        contact_width=c.width,
        friction=c.friction,
        friction_width=c.velocity_width,
        max_newton=n.max_steps,
        max_backtracks=n.backtracks,
        restart=n.restart,
        max_restarts=n.max_restarts,
        linear_refinements=n.linear_refinements,
        armijo=n.armijo,
        atol64=n.atol64,
        atol32=n.atol32,
        linear_rtol64=n.linear_rtol64,
        linear_rtol32=n.linear_rtol32,
    )
    fields = {
        key: np.asarray(getattr(model, key)).copy()
        for key in (
            "alpha",
            "young",
            "poisson",
            "density",
            "strength",
            "direction",
            "kind",
            "channel",
            "contact_area",
        )
    }
    fields["kind"] = fields["kind"].astype(np.int32)
    fields["channel"] = fields["channel"].astype(np.int32)
    fields.update(
        {
            key: np.asarray(getattr(model.cloud, key)).copy()
            for key in ("reference", "volume", "face_offset")
        }
    )
    initial = {
        key: np.asarray(getattr(state, key)).copy() for key in ("x", "v", "F", "C", "actuator")
    }
    initial["ledger"] = np.array(
        [state.work_signed, state.work_positive, state.dissipated, state.time],
        dtype=fields["reference"].dtype,
    )
    initial["status"] = np.array([state.status], np.int32)
    terrain = (
        None
        if model.terrain is None
        else {k: np.asarray(v).copy() for k, v in cast("Terrain", model.terrain)._asdict().items()}
    )
    return Bundle(fields, config, len(model.channels), terrain, initial).validate()


def to_jax(bundle: "Bundle") -> "tuple[mpm.Model, mpm.State, mpm.Grid, DynamicsConfig]":
    """Independent oracle using the existing checked-out JAX implementation."""
    import jax.numpy as j

    from envs.soft_walker.engine.numerics.solvers import NewtonConfig
    from envs.soft_walker.engine.physics import mpm
    from envs.soft_walker.engine.physics.common import DynamicsConfig
    from envs.soft_walker.engine.physics.contact import PlaneContact
    from envs.soft_walker.engine.physics.terrain import Terrain

    bundle.validate()
    f, c = bundle.fields, bundle.config
    dtype = f["reference"].dtype
    n, dim = f["reference"].shape
    a = {key: j.asarray(value) for key, value in f.items()}
    cloud = mpm.Cloud(
        a["reference"],
        j.zeros((n, dim), dtype),
        a["volume"],
        a["face_offset"],
        j.ones(2 * dim, dtype),
        j.zeros((n, 2 * dim), j.int32),
        j.zeros((n, 2 * dim), bool),
    )
    mass = a["alpha"] * a["density"] * a["volume"]
    terrain = (
        None
        if bundle.terrain is None
        else Terrain(*(j.asarray(bundle.terrain[key]) for key in Terrain._fields))
    )
    model = mpm.Model(
        cloud,
        a["alpha"],
        a["young"],
        a["poisson"],
        a["density"],
        a["strength"],
        a["direction"],
        a["kind"],
        a["channel"],
        mass,
        mass > 0,
        a["contact_area"],
        j.zeros(bundle.channels, dtype),
        terrain,
    )
    state = mpm.initial_state(model)
    if bundle.initial is not None:
        v = bundle.initial
        state = mpm.State(
            j.asarray(v["x"]),
            j.asarray(v["v"]),
            j.asarray(v["F"]),
            j.asarray(v["C"]),
            j.asarray(v["actuator"]),
            j.asarray(v["ledger"][LEDGER_WORK_SIGNED]),
            j.asarray(v["ledger"][LEDGER_WORK_POSITIVE]),
            j.asarray(v["ledger"][LEDGER_DISSIPATED]),
            j.asarray(v["ledger"][LEDGER_TIME]),
            j.asarray(v["status"][0], j.int32),
        )
    cfg = DynamicsConfig(
        gravity=c.gravity,
        vertical_axis=c.vertical_axis,
        velocity_ref=c.velocity_ref,
        actuator_tau=c.actuator_tau,
        max_log_stretch=c.max_log_stretch,
        min_J=c.min_J,
        max_halvings=0,
        mpm_mass_weighted=c.mass_weighted,
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
            armijo=c.armijo,
            linear_refinements=c.linear_refinements,
            preconditioner_side="right",
            gmres_method="fused",
        ),
    )
    return model, state, mpm.Grid(c.grid_shape, c.spacing, c.origin), cfg
