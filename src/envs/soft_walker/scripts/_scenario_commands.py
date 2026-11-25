"""Legacy scenario recording: explicit simulation inputs and saved outputs.

The resident route remains in _resident_commands. This diagnostic recorder keeps
the existing chunk size, tail truncation, failure-prefix behavior and filenames.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple, Protocol

if TYPE_CHECKING:
    from envs.soft_walker.engine.drivers.spring_environment import Spring2DEnv
    from envs.soft_walker.engine.types import Array, DriverState, HostArray, PhysicsModel
    from envs.soft_walker.presets.scenarios.config import Scenario


class StageReporter(Protocol):
    """Receive a phase name and JSON-serializable progress fields."""

    def __call__(self, name: str, **details: object) -> None: ...


class ReplayData(NamedTuple):
    """Host arrays in the existing trajectory schema, including the initial sample."""

    rows: HostArray
    positions: HostArray
    commands: HostArray
    solver: HostArray
    topology: dict[str, HostArray]


def replay_chunks(
    env: Spring2DEnv,
    case: Scenario,
    args: argparse.Namespace,
    model: PhysicsModel,
    initial: DriverState,
    parameters: Array,
    costs: dict[str, float],
    stage: StageReporter,
    output: Path,
) -> ReplayData:
    """Record the valid simulation prefix and write the unchanged trajectory artifacts.

    Model/state are borrowed. ``costs`` is updated with compilation time, and
    ``stage`` is called before each chunk. A failed solver stops recording;
    the returned samples retain its diagnostic row for the caller's assessment.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np

    from envs.soft_walker.engine.drivers.model_access import task_positions as positions
    from envs.soft_walker.experiments.scenarios.rollout import bind_chunk, columns, measure
    from envs.soft_walker.shared import write_json, write_npz
    from envs.soft_walker.viewer.trajectory_animation import topology_record

    stage("replay_chunk_compile")
    t0 = time.perf_counter()
    chunk = (
        jax.jit(bind_chunk(env, case, args.chunk_size))
        .lower(model, initial, jnp.asarray(parameters), jnp.asarray(0, jnp.int32))
        .compile()
    )
    costs["replay_chunk_compile_s"] = time.perf_counter() - t0
    topology = topology_record(env, model)
    state = initial
    initial_metrics = np.asarray(measure(env, model, initial))
    all_rows = [initial_metrics[None]]
    all_positions = [np.asarray(positions(env, model, initial))[None]]
    all_actions, all_solver = [], []
    (output / "chunks").mkdir()
    write_npz(output / "topology.npz", **topology)
    write_npz(output / "parameters.npz", parameters=parameters)
    chunk_times = []
    for start in range(0, case.steps, args.chunk_size):
        stage("replay_chunk", control_interval=start, simulated_time_s=float(all_rows[-1][-1, 0]))
        t0 = time.perf_counter()
        state, history = jax.block_until_ready(
            chunk(model, state, jnp.asarray(parameters), jnp.asarray(start, jnp.int32))
        )
        elapsed = time.perf_counter() - t0
        n = min(args.chunk_size, case.steps - start)
        rows, commands, xyz, solver = [np.asarray(a)[:n] for a in history]
        all_rows.append(rows)
        all_positions.append(xyz)
        all_actions.append(commands)
        all_solver.append(solver)
        write_npz(
            output / "chunks" / f"{start:06d}.npz",
            metrics=rows,
            actions=commands,
            positions=xyz,
            solver=solver,
        )
        chunk_times.append({"start_interval": start, "intervals": n, "synchronized_s": elapsed})
        write_json(output / "timings.json", {**costs, "chunks": chunk_times})
        if int(state.status) != 0:
            break
    rows = np.concatenate(all_rows)
    xyz = np.concatenate(all_positions)
    commands = np.concatenate(all_actions)
    solver = np.concatenate(all_solver)
    write_npz(
        output / "trajectory.npz",
        positions=xyz,
        metrics=rows,
        actions=commands,
        solver=solver,
        parameters=parameters,
        **topology,
    )
    write_json(output / "metric_columns.json", columns(case.dim))

    return ReplayData(rows, xyz, commands, solver, topology)
