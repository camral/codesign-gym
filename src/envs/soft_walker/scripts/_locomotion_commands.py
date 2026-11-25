"""Configuration and replay artifact handling for the locomotion command.

The command selects precision/device before these helpers import an accelerator.
Recorder state owns only the cached executable and timing dictionary; simulation
model/state arrays are borrowed and remain unchanged by artifact generation.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from envs.soft_walker.contracts.metrics import LOCOMOTION_SOLVER_COLUMNS

if TYPE_CHECKING:
    import jax

    from envs.soft_walker.engine.drivers.spring_environment import Spring2DEnv
    from envs.soft_walker.engine.types import (
        Array,
        DriverState,
        ExperimentResult,
        HostArray,
        PhysicsModel,
    )
    from envs.soft_walker.experiments.locomotion.rollout import ObjectiveConfig
    from envs.soft_walker.presets.locomotion import DemoConfig


def build_config(args: argparse.Namespace) -> DemoConfig:
    """Translate the existing CLI options into an immutable physical task."""
    from envs.soft_walker.presets.locomotion import DemoConfig

    return DemoConfig(
        driver=args.driver,
        dtype=args.dtype,
        cycles=args.cycles,
        period=args.period,
        control_dt=args.control_dt,
        substeps=args.substeps,
        settle_seconds=args.settle_seconds,
        refinement=args.refinement,
        grid_margin=args.grid_margin,
        travel_body_lengths=args.travel_body_lengths,
        spring_solver=args.spring_solver,
        gmres_method=args.gmres_method,
        friction=args.friction,
        friction_velocity=args.friction_velocity,
        gravity=args.gravity,
        max_log_stretch=args.max_log_stretch,
        actuator_tau=args.actuator_tau,
        seed=args.seed,
        actuation_model=args.actuation_model,
        active_stress_ratio=args.active_stress_ratio,
    )


def controller_parameters(args: argparse.Namespace, config: DemoConfig) -> HostArray:
    """Load or seed coefficients, preserving dtype, channel validation and passivity."""
    import numpy as np

    from envs.soft_walker.presets.gaits import seed_controller

    parameters = seed_controller(
        config.dim,
        amplitude=args.amplitude,
        lift=args.lift,
        phase=args.phase,
        gait=args.gait,
        harmonics=args.harmonics,
        dtype=args.dtype,
    )
    if args.passive:
        parameters[:] = 0
    if args.resume or args.controller:
        with np.load(args.resume or args.controller, allow_pickle=False) as saved:
            parameters = saved["parameters"].astype(args.dtype)
    if parameters.shape[0] != config.channels or not np.isfinite(parameters).all():
        raise ValueError("Controller channels/values do not match this robot")
    return parameters


@dataclass
class ExperimentRecorder:
    """Render/save repeated controllers while reusing one compiled recording graph.

    Output directories are supplied per call. The original NPZ/JSON/GIF names and
    schemas are retained. Compilation is lazy and its time is recorded separately
    from synchronized replay time. Optional graphics imports stay in the call.
    """

    env: Spring2DEnv
    config: DemoConfig
    args: argparse.Namespace
    model: PhysicsModel
    initial: DriverState
    objective: ObjectiveConfig
    compile_times: dict[str, float]
    # JAX compiled calls return a heterogeneous rollout pytree, not one ndarray.
    compiled_recording: (
        Callable[[PhysicsModel, DriverState, jax.Array], ExperimentResult] | None
    ) = field(default=None, init=False)

    def __call__(
        self, p: Array, directory: Path, *, iteration: int | None = None
    ) -> tuple[ExperimentResult, HostArray, dict[str, object], float]:
        """Save one replay and return its record, metrics, assessment and wall time."""
        import jax
        import jax.numpy as jnp
        import numpy as np

        from envs.soft_walker.engine.drivers.model_access import task_positions as positions
        from envs.soft_walker.experiments.locomotion.rollout import (
            bind_experiment,
            flight_assessment,
            flight_columns,
            gait_assessment,
            metric_columns,
            objective_components,
            sample_flight_metrics,
            sample_metrics,
        )
        from envs.soft_walker.shared import region_tags, write_json, write_npz
        from envs.soft_walker.viewer.trajectory_animation import render_animation, topology_record

        directory.mkdir(parents=True, exist_ok=True)
        coefficients = jnp.asarray(p)
        if self.compiled_recording is None:
            recorded = jax.jit(
                bind_experiment(
                    self.env,
                    self.config,
                    record=True,
                    block_size=self.args.block_size,
                    objective=self.objective,
                )
            )
            started = time.perf_counter()
            self.compiled_recording = recorded.lower(
                self.model, self.initial, coefficients
            ).compile()
            self.compile_times["recording_compile_s"] = time.perf_counter() - started
        started = time.perf_counter()
        replay = jax.block_until_ready(
            self.compiled_recording(self.model, self.initial, coefficients)
        )
        replay_wall = time.perf_counter() - started
        records = np.asarray(replay["metrics"])
        xyz = np.asarray(replay["positions"])
        torso, feet = region_tags(self.env, self.model)
        initial_row = np.asarray(sample_metrics(self.env, self.model, self.initial, torso, feet))
        frames_xyz = np.concatenate(
            (np.asarray(positions(self.env, self.model, self.initial))[None], xyz)
        )
        frame_metrics = np.concatenate((initial_row[None], records))
        initial_flight = np.asarray(
            sample_flight_metrics(self.env, self.model, self.initial, torso)
        )
        flight = np.concatenate((initial_flight[None], np.asarray(replay["flight_metrics"])))
        topology = topology_record(self.env, self.model)
        write_npz(
            directory / "trajectory.npz",
            positions=frames_xyz,
            metrics=frame_metrics,
            actions=replay["actions"],
            parameters=p,
            solver_diagnostics=replay["solver_diagnostics"],
            flight_metrics=flight,
            **topology,
        )
        write_npz(directory / "best.npz", parameters=p, loss=replay["loss"])
        assessment = gait_assessment(records, self.config)
        write_json(directory / "assessment.json", assessment)
        airborne = flight_assessment(flight, self.config, valid_rollout=bool(replay["valid"]))
        write_json(directory / "flight_assessment.json", airborne)
        write_json(
            directory / "replay.json",
            {
                "columns": metric_columns(self.config.dim),
                "synchronized_replay_s": replay_wall,
                "compile": self.compile_times,
                "final_metrics": replay["final_metrics"],
                "loss": replay["loss"],
                "solver_columns": list(LOCOMOTION_SOLVER_COLUMNS),
                "flight_columns": flight_columns(),
                "objective": self.args.objective,
                "objective_components": objective_components(
                    frame_metrics,
                    flight,
                    self.config,
                    self.objective,
                    work_ref=self.env.config.work_ref,
                ),
            },
        )
        if not self.args.no_animation:
            title = f"{self.args.driver}: chamber-actuated {'biped' if self.config.dim == 2 else 'quadruped'}"
            if iteration is not None:
                title = f"{self.args.driver}: {self.args.optimizer.upper()} iteration {iteration} | loss={float(replay['loss']):+.6f}"
            if self.args.objective == "hop":
                verdict = (
                    "repeated flight observed"
                    if airborne["repeated_airborne_observed"]
                    else "pronk candidate — repeated flight not observed"
                )
            else:
                verdict = (
                    "gait gate passed"
                    if assessment["walking_gate_passed"]
                    else "candidate replay — walking gate not passed"
                )
            render_animation(
                frames_xyz,
                frame_metrics,
                topology,
                directory / "replay.gif",
                title=title,
                period=self.config.period,
                fps=self.args.fps,
                verdict=verdict,
            )
        return replay, records, assessment, replay_wall
