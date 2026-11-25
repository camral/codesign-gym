"""Host command execution for whole-episode demos/scenarios.

Only the optimizer/evidence/renderer run on the host. A candidate is uploaded
once, then one certified executable owns initialization, settling, all timesteps,
controller feedback, objective, and (where selected) reverse differentiation.
"""

from __future__ import annotations

import time
from argparse import Namespace
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar, Unpack, cast

import jax
import numpy as np

from envs.soft_walker.engine.execution.compiled import CompiledProgram, compile_program
from envs.soft_walker.engine.execution.contracts import EpisodeOutput, IterationArguments
from envs.soft_walker.engine.execution.policy import ResidencyError, ResidentPolicy
from envs.soft_walker.engine.execution.program import stage_embedding
from envs.soft_walker.engine.types import Array, Embedding, HostArray, PathLike, PhysicsModel
from envs.soft_walker.experiments.locomotion.resident import bind_demo_episode
from envs.soft_walker.experiments.scenarios.resident import bind_scenario_episode
from envs.soft_walker.shared import append_event, write_json, write_npz

if TYPE_CHECKING:
    from envs.soft_walker.engine.drivers.spring_environment import Spring2DEnv
    from envs.soft_walker.experiments.locomotion.rollout import ObjectiveConfig
    from envs.soft_walker.presets.locomotion import DemoConfig
    from envs.soft_walker.presets.scenarios.config import Scenario

OutputT = TypeVar("OutputT")


def _device_output(result: OutputT) -> OutputT:
    # Explicit output boundary, after the executable and all device work finish.
    """Download a completed output pytree at the optimizer/evidence boundary only."""
    with jax.transfer_guard("allow"):
        return jax.device_get(result)


def _assess_metrics(
    metrics: HostArray, protocol: Scenario | DemoConfig, family: str, design_mode: str = "hard"
) -> dict[str, object]:
    """Adapt initialized recording rows to the historical scenario/demo assessment API."""
    from envs.soft_walker.experiments.locomotion.rollout import gait_assessment
    from envs.soft_walker.experiments.scenarios.rollout import assess

    # Both new recorders INCLUDE the initialized sample. The historical demo
    # assessor takes only POST-step rows; the scenario assessor includes reset.
    return (
        assess(metrics, cast("Scenario", protocol), design_mode=design_mode)
        if family == "scenario"
        else gait_assessment(metrics[1:], cast("DemoConfig", protocol))
    )


def _save_replay(
    result: EpisodeOutput,
    *,
    env: Spring2DEnv,
    model: PhysicsModel,
    protocol: Scenario | DemoConfig,
    family: str,
    output: PathLike,
    parameters: Array,
    no_animation: bool,
    fps: int,
    design_mode: str = "hard",
) -> dict[str, object]:
    """Download one device recording and write the existing replay artifact schema.

    Writes trajectory/topology/parameter archives, assessment and terrain metadata,
    and an optional GIF. Positions are metres and recorded times are seconds; no
    additional physics steps run during this host-side reporting operation."""
    from envs.soft_walker.engine.physics.terrain import Terrain, serialize
    from envs.soft_walker.experiments.locomotion.rollout import flight_columns, metric_columns
    from envs.soft_walker.experiments.scenarios.rollout import columns
    from envs.soft_walker.viewer.trajectory_animation import render_animation, topology_record

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    host = _device_output(result)
    history = host["history"]
    metrics = np.asarray(history["metrics"])
    frame_indices = np.asarray(history["frame_indices"])
    samples = history["samples"]
    topology = topology_record(env, model)
    write_npz(
        output / "trajectory.npz",
        positions=samples["positions"],
        metrics=metrics,
        actions=history["actions"],
        solver=history["solver"],
        frame_indices=frame_indices,
        parameters=parameters,
        **({"flight_samples": samples["flight"]} if "flight" in samples else {}),
    )
    write_npz(output / "topology.npz", **topology)
    write_npz(output / "best.npz", parameters=parameters, loss=host["loss"])
    assessment = _assess_metrics(metrics, protocol, family, design_mode)
    assessment["execution_valid"] = bool(host["valid"])
    assessment["recording"] = {
        "single_whole_episode_call": True,
        "host_timestep_loop": False,
        "metrics_include_initial_sample": True,
        "sample_indices": frame_indices.tolist(),
        "physical_time_s": metrics[frame_indices, 0].tolist(),
        "sampled_flight_only": "flight" in samples,
        "no_new_walking_claim": True,
    }
    write_json(output / "assessment.json", assessment)
    write_json(
        output / "metrics_columns.json",
        columns(protocol.dim) if family == "scenario" else metric_columns(protocol.dim),
    )
    write_json(output / "terrain.json", serialize(cast(Terrain, env._terrain)))
    if "flight" in samples:
        write_json(output / "flight_columns.json", flight_columns())
    if not no_animation:
        render_animation(
            samples["positions"],
            metrics[frame_indices],
            topology,
            output / "replay.gif",
            title=f"{protocol.driver}: whole-episode replay",
            period=protocol.period,
            fps=fps,
            terrain=env._terrain,
            verdict="candidate; see assessment.json, not gait-certified",
        )
    return assessment


def execute(
    env: Spring2DEnv,
    embedding: Embedding,
    protocol: Scenario | DemoConfig,
    parameters: Array,
    args: Namespace,
    *,
    family: str,
    objective: ObjectiveConfig | None = None,
) -> dict[str, object]:
    """Runs inside the existing CLI's exception/log/archive boundary."""
    output = Path(args.out)
    if getattr(args, "api_check", False):
        raise ValueError(
            "--api-check intentionally executes host-stepped reference simulation. "
            "Use the focused regression suite for parity; do not enable it in a resident run."
        )
    if getattr(args, "repeats", 1) < 1:
        raise ValueError("repeats must be positive")
    policy = ResidentPolicy(
        args.device,
        int(args.recording_mib * 1024**2),
        deterministic_reductions=getattr(args, "deterministic_reductions", False),
    )
    block = getattr(args, "block_size", getattr(args, "chunk_size", 10))
    program = (
        bind_scenario_episode(
            env,
            cast("Scenario", protocol),
            record_stride=args.record_stride,
            checkpoint_block=block,
        )
        if family == "scenario"
        else bind_demo_episode(
            env,
            cast("DemoConfig", protocol),
            objective=objective,
            record_stride=args.record_stride,
            checkpoint_block=block,
        )
    )
    values, model = stage_embedding(env, embedding, seed=protocol.seed)
    values = jax.device_put(values, env.device)
    parameters = np.asarray(parameters, dtype=protocol.dtype)
    write_json(output / "resident_policy.json", policy.descriptor())
    write_json(output / "episode.json", program.description)
    write_json(output / "embodiment.json", embedding)
    write_npz(output / "initial_parameters.npz", parameters=parameters)
    completed: dict[str, object] = {"compiled_boundaries": [], "mode": args.mode, "finished": False}
    start = time.perf_counter()

    def event(name: str, **details: object) -> None:
        """Append a timestamped execution event and report the active boundary to the CLI."""
        append_event(
            output / "resident_events.jsonl",
            name,
            seconds_since_start=time.perf_counter() - start,
            **details,
        )
        print(f"[whole-episode/{name}]", flush=True)

    def compile_boundary(
        raw: Callable[..., OutputT], name: str, p: Array
    ) -> tuple[CompiledProgram[OutputT], tuple[object, ...], OutputT]:
        """Compile, stage, and certify one complete candidate function before first use."""
        event("compile", executable=name)
        compiled = compile_program(
            raw, (values, p), policy=policy, out=output / "executables" / name, name=name
        )
        inputs = compiled.stage(values, p)
        event("certify", executable=name)
        alternative = compiled.stage(
            values, np.asarray(p) * np.asarray(0.999, dtype=np.asarray(p).dtype)
        )
        result = compiled.certify(*inputs, alternate_arguments=alternative)
        cast(list[str], completed["compiled_boundaries"]).append(name)
        write_json(output / "resident_status.json", completed)
        return compiled, inputs, result

    recorder: CompiledProgram[EpisodeOutput] | None = None

    def record_candidate(p: Array, location: PathLike, name: str = "record") -> dict[str, object]:
        """Reuse the certified recorder and persist one candidate at the requested location."""
        nonlocal recorder
        if recorder is None:
            recorder, inputs, result = compile_boundary(program.record, name, p)
        else:
            inputs = recorder.stage(values, p)
            result = recorder(*inputs)
        if getattr(args, "repeats", 1) > 1:
            result = recorder.benchmark(*inputs, repeats=args.repeats)
        event(
            "record_complete",
            simulation_host_transfers=0,
            gpu_residency_certified=policy.device == "cuda",
        )
        return _save_replay(
            result,
            env=env,
            model=model,
            protocol=protocol,
            family=family,
            output=location,
            parameters=p,
            no_animation=args.no_animation,
            fps=args.fps,
            design_mode=getattr(args, "design_mode", "hard"),
        )

    try:
        if args.mode == "replay":
            assessment = record_candidate(parameters, output)
        elif args.mode == "scout":
            rng = np.random.default_rng(protocol.seed)
            candidates = np.repeat(parameters[None], args.population, axis=0)
            candidates[1:] += rng.normal(scale=0.1, size=candidates[1:].shape)
            population, inputs, result = compile_boundary(
                program.population(), "population", candidates
            )
            losses = np.asarray(_device_output(result))
            write_npz(output / "population.npz", parameters=candidates, loss=losses)
            if not np.isfinite(losses).all():
                raise ResidencyError(
                    "Population contains invalid simulations; inspect candidate results"
                )
            parameters = candidates[int(np.argmin(losses))]
            assessment = record_candidate(parameters, output)
        else:
            forward, f_inputs, _ = compile_boundary(program.loss, "loss", parameters)
            derivative, g_inputs, _ = compile_boundary(
                program.value_and_grad(), "value_and_gradient", parameters
            )
            forward.benchmark(*f_inputs, repeats=getattr(args, "repeats", 1))
            derivative.benchmark(*g_inputs, repeats=getattr(args, "repeats", 1))

            def evaluate(p: HostArray) -> tuple[Array, Array]:
                # Upload and output reads are optimizer-boundary operations.
                # There is exactly one compiled forward+reverse call between.
                """Evaluate one host optimizer candidate through a single certified gradient call."""
                return cast(
                    tuple[Array, Array], _device_output(derivative(*derivative.stage(values, p)))
                )

            if args.mode == "gradcheck":
                from envs.soft_walker.shared import difference_ladder

                ladder = difference_ladder(
                    lambda p: _device_output(forward(*forward.stage(values, p))),
                    evaluate,
                    parameters,
                    seed=protocol.seed,
                )
                write_json(output / "whole_episode_gradient_ladder.json", ladder)
                if not ladder["agreement_passed"]:
                    raise AssertionError("Whole-episode derivative ladder failed")
                assessment = {"gradient_agreement": True, "walking_gate_passed": False}
            else:
                from envs.soft_walker.shared import optimize

                def on_iteration(**state: Unpack[IterationArguments]) -> None:
                    """Optionally record the optimizer checkpoint without changing accepted parameters."""
                    if getattr(args, "animate_every_iteration", False):
                        record_candidate(
                            state["parameters"], output / "iterations" / f"{state['iteration']:05d}"
                        )

                import shutil

                optimization_dir = output / "optimization"
                optimization_dir.mkdir(parents=True, exist_ok=True)
                for metadata in ("protocol.json", "embodiment.json", "objective.json", "case.json"):
                    if (output / metadata).exists():
                        shutil.copy2(output / metadata, optimization_dir / metadata)
                result = optimize(
                    evaluate,
                    parameters,
                    output / "optimization",
                    method=args.optimizer,
                    iterations=args.iterations,
                    learning_rate=args.learning_rate,
                    resume=getattr(args, "resume", None),
                    on_iteration=on_iteration if args.optimizer in ("adam", "gd") else None,
                )
                parameters = result.parameters
                assessment = record_candidate(parameters, output)
                write_json(output / "optimization_result.json", result)
        completed.update(
            finished=True, assessment=assessment, gpu_residency_demonstrated=policy.device == "cuda"
        )
        return completed
    except Exception as error:
        completed["exception"] = repr(error)
        raise
    finally:
        completed["wall_seconds_including_compilation_and_certification"] = (
            time.perf_counter() - start
        )
        write_json(output / "resident_status.json", completed)
