"""Construct scenario drivers through the fixed co-design API; estimate cost first."""

from __future__ import annotations

import math

import numpy as np

from envs.soft_walker.contracts.observations import (
    body_observation_layout,
    spring_observation_layout,
)
from envs.soft_walker.engine import make
from envs.soft_walker.engine.drivers.spring_environment import Spring2DEnv
from envs.soft_walker.engine.drivers.task import BodyConfig
from envs.soft_walker.engine.numerics.solvers import NewtonConfig
from envs.soft_walker.engine.physics.common import DynamicsConfig
from envs.soft_walker.engine.physics.contact import PlaneContact
from envs.soft_walker.engine.physics.spring_config import SpringConfig
from envs.soft_walker.engine.types import HostEmbedding
from envs.soft_walker.presets.embodiments.chambers import ChamberDecoder
from envs.soft_walker.presets.locomotion import ENV_IDS
from envs.soft_walker.presets.scenarios.config import Scenario
from envs.soft_walker.presets.terrain.scenario_courses import case_terrain_config


def grid_padding(case: Scenario) -> tuple[tuple[int, int], ...]:
    """Reserve fixed MPM grid travel space while retaining the reference envelope."""
    factor = case.refinement
    forward = max(4 * factor, math.ceil(case.travel_body_lengths * 8 * factor) + 2)
    pads = ((4 * factor, forward), (4 * factor, 6 * factor))
    return pads + (((4 * factor, 4 * factor),) if case.dim == 3 else ())


def decoder_and_embodiment(
    case: Scenario, *, design_mode: str = "hard", material_mode: str = "continuous"
) -> tuple[ChamberDecoder, HostEmbedding]:
    """Create the selected chamber representation and its spatial material embedding."""
    decoder = ChamberDecoder(
        case.dim,
        case.representation,
        case.dtype,
        design_mode=design_mode,
        material_mode=material_mode,
    )
    embodiment = decoder.base()
    if material_mode == "continuous":
        embodiment["young"] = np.asarray([case.leg_young, case.torso_young], case.dtype)
    embodiment["density"] = np.asarray(case.density, case.dtype)
    embodiment["strength"] = np.asarray(case.strength, case.dtype)
    embodiment["poisson_values"].fill(case.poisson)
    return decoder, embodiment


def build(
    case: Scenario,
    *,
    device: str = "cpu",
    design_mode: str = "hard",
    material_mode: str = "continuous",
) -> tuple[Spring2DEnv, HostEmbedding]:
    """Return (environment, embodiment); caller still calls reset explicitly."""
    decoder, embodiment = decoder_and_embodiment(
        case, design_mode=design_mode, material_mode=material_mode
    )
    terrain = case_terrain_config(case)
    dt = case.effective_control_dt
    config: SpringConfig | BodyConfig
    common = dict(
        grid_shape=case.grid_shape,
        size=case.size,
        control_dt=dt,
        substeps=case.numerical_substeps,
        horizon=case.steps,
        settle_steps=0 if case.tier == "smoke" else round(case.settle_seconds / dt),
        initial_clearance=0.001,
        length_ref=case.size[0],
        work_ref=4.0 if case.dim == 3 else 0.4,
        energy_weight=0.0,
        terrain=terrain,
    )
    if case.driver == "spring":
        config = SpringConfig(
            # The selected config validates dimension-specific common options.
            **common,  # type: ignore[arg-type]
            linear_solver="gmres",
            gmres_method=case.gmres_method,
            friction=case.friction,
            friction_velocity=case.friction_velocity,
            contact_modulus=1.0e6 * case.size[0],
            contact_width=0.0005,
            actuator_tau=case.actuator_tau,
            max_log_stretch=case.max_log_stretch,
        )
        observation_size = spring_observation_layout(
            int(np.prod(np.asarray(case.grid_shape) + 1)),
            int(np.prod(case.grid_shape)),
            case.channels,
            terrain=True,
        ).width
    else:
        contact = PlaneContact(
            enabled=True,
            stiffness=1.0e6,
            width=0.0005,
            friction=case.friction,
            velocity_width=case.friction_velocity,
        )
        dynamics = DynamicsConfig(
            contact=contact,
            actuator_tau=case.actuator_tau,
            max_log_stretch=case.max_log_stretch,
            max_halvings=0,
            mpm_mass_weighted=True,
            newton=NewtonConfig(
                gmres_method=case.gmres_method,
                preconditioner_side="right",
                linear_refinements=2,
                restart=40,
                max_restarts=20,
            ),
        )
        config = BodyConfig(
            # BodyConfig validates the shared factory options.
            **common,  # type: ignore[arg-type]
            dynamics=dynamics,
            grid_padding=grid_padding(case),
            observation_profile="full",
        )
        sites = np.prod(case.grid_shape) * (2**case.dim if case.driver.startswith("mpm") else 1)
        nodes = sites if case.driver != "fem" else np.prod(np.asarray(case.grid_shape) + 1)
        observation_size = body_observation_layout(
            case.driver, case.dim, int(nodes), int(sites), case.channels, profile="full"
        ).width
    env, _ = make(
        ENV_IDS[case.driver],
        config=config,
        decoder=decoder,
        device=device,
        dtype=case.dtype,
        channels=case.channels,
        max_action_space_size=case.channels,
        max_observation_space_size=int(observation_size),
        is_eval_env=True,
    )
    return env, embodiment
