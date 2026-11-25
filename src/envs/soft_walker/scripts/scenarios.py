#!/usr/bin/env python3
"""Plan, preview, replay, or optimize a local locomotion scenario.

All large runs are explicit. Planning writes small JSON recipes without importing
JAX. Each numerical worker saves a crash-visible stage and an evidence ZIP.
"""

from __future__ import annotations

import argparse

# Resolve integrated imports independently of the caller's working directory.
import sys as _script_sys
from pathlib import Path, Path as _ScriptPath
from typing import TYPE_CHECKING, Any, cast

_source_root = str(_ScriptPath(__file__).resolve().parents[3])
if _source_root not in _script_sys.path:
    _script_sys.path.insert(0, _source_root)
_script_dir = str(_ScriptPath(__file__).resolve().parent)
if _script_dir not in _script_sys.path:
    _script_sys.path.insert(0, _script_dir)
import json
import os
import sys
import time
import traceback
import zipfile
from dataclasses import replace

from envs.soft_walker.shared import (
    ENGINE_ROOT as ENGINE_ROOT,
    REPO_ROOT as REPO_ROOT,
    SCRIPTS_ROOT as SCRIPTS_ROOT,
    SOFT_ROOT as SOFT_ROOT,
    SRC_ROOT as SRC_ROOT,
    TESTS_ROOT as TESTS_ROOT,
)

ROOT = REPO_ROOT
sys.path.insert(0, str(ROOT / "src"))


from _scenario_commands import replay_chunks

if TYPE_CHECKING:
    from envs.soft_walker.engine.types import (
        PathLike,
    )

# Serialized reports can contain backend-specific auxiliary diagnostic trees.
Report = dict[str, Any]


def parser() -> argparse.ArgumentParser:
    """Build the command-line parser without constructing a simulator or running physics."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("plan", "preview", "replay", "train", "gradcheck", "scout"))
    p.add_argument(
        "--suite",
        choices=("drivers", "terrains", "representations", "controllers", "airborne"),
        default="drivers",
    )
    p.add_argument("--case", type=Path)
    p.add_argument(
        "--terrain-file",
        type=Path,
        help="Custom terrain recipe JSON; embedded in the saved case for replay",
    )
    for field, values in (
        ("driver", ("spring", "mpm2", "mpm3", "beam", "fem")),
        ("representation", ("primitives", "direct", "bspline", "cppn")),
        ("controller", ("fourier", "pulse", "feedback", "neural")),
        ("gait", ("walk", "crawl", "trot", "bound", "pronk")),
        (
            "terrain",
            (
                "flat",
                "uphill",
                "downhill",
                "cross_slope",
                "low_blocks",
                "rough_blocks",
                "stairs",
                "hurdles",
            ),
        ),
        ("tier", ("smoke", "coarse", "resolved")),
        ("dtype", ("float64", "float32")),
    ):
        p.add_argument("--" + field, choices=values)
    p.add_argument("--seed", type=int)
    p.add_argument("--cycles", type=int)
    p.add_argument("--set", action="append", default=[], metavar="FIELD=JSON")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--design-mode", choices=("hard", "relaxed"), default="hard")
    p.add_argument("--material-mode", choices=("continuous", "categorical"), default="continuous")
    p.add_argument(
        "--parameters",
        type=Path,
        help="NPZ containing flat 'parameters'; validate against this controller",
    )
    p.add_argument(
        "--execution",
        choices=("resident", "legacy"),
        default="resident",
        help="resident requires whole-episode CUDA trace certification; legacy is diagnostic only",
    )
    p.add_argument("--record-stride", type=int, default=1)
    p.add_argument("--recording-mib", type=int, default=512)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument(
        "--deterministic-reductions",
        action="store_true",
        help="Use deterministic resident CUDA reductions for strict gradient repeats; may cost runtime",
    )
    p.add_argument("--chunk-size", type=int, default=10)
    p.add_argument("--no-animation", action="store_true")
    p.add_argument("--fps", type=float, default=25.0)
    p.add_argument("--optimizer", choices=("adam", "gd", "lbfgs"), default="adam")
    p.add_argument("--iterations", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=0.03)
    p.add_argument("--population", type=int, default=4)
    p.add_argument("--api-check", action="store_true")
    p.add_argument(
        "--offline-spaces", action="store_true", help="Explicit offline development only"
    )
    return p


def plain_json(path: PathLike, data: Any) -> None:
    """Atomically replace a JSON report using a sibling temporary file."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str) + "\n")
    os.replace(tmp, path)


def overrides(args: argparse.Namespace) -> Report:
    """Translate explicit CLI overrides and an optional terrain recipe into scenario fields."""
    values = {
        k: getattr(args, k)
        for k in (
            "driver",
            "representation",
            "controller",
            "gait",
            "terrain",
            "tier",
            "dtype",
            "seed",
            "cycles",
        )
        if getattr(args, k) is not None
    }
    for item in args.set:
        key, separator, value = item.partition("=")
        if not separator:
            raise ValueError("--set requires FIELD=JSON")
        try:
            values[key] = json.loads(value)
        except json.JSONDecodeError:
            values[key] = value
    if args.terrain_file:
        if "terrain" in values or "terrain_recipe" in values:
            raise ValueError(
                "--terrain-file cannot be combined with --terrain or a terrain --set override"
            )
        values["terrain_recipe"] = json.loads(args.terrain_file.read_text(encoding="utf-8"))
        values["terrain"] = "flat"
    return values


def main() -> int:
    """Plan, preview or execute a scenario, preserving failure evidence and its exit status."""
    args = parser().parse_args()
    if args.chunk_size < 1 or args.fps <= 0 or args.population < 1 or args.iterations < 0:
        raise SystemExit("Invalid chunk/frame/population/iteration limit")
    output = args.out.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit("Output directory must be fresh")
    output.mkdir(parents=True, exist_ok=True)
    from envs.soft_walker.presets.scenarios.config import Scenario
    from envs.soft_walker.presets.scenarios.suites import suite

    edits = overrides(args)
    if args.mode == "plan":
        cases = [replace(c, **edits) for c in suite(args.suite)]
        files = []
        for i, case in enumerate(cases):
            path = output / f"{i:02d}_{case.key}.json"
            plain_json(path, case.to_dict())
            files.append(path.name)
        plain_json(
            output / "plan.json",
            {
                "schema": "codesign_scenario_plan_v1",
                "suite": args.suite,
                "cases": files,
                "source_root_hint": str(ROOT),
                "status": "recipes only; no dynamics run",
                "design_mode": args.design_mode,
                "material_mode": args.material_mode,
            },
        )
        print(f"Wrote {len(files)} cases: {output / 'plan.json'}")
        return 0

    case = Scenario.from_dict(json.loads(args.case.read_text())) if args.case else Scenario()
    if case.terrain_recipe is not None and "terrain" in edits and not args.terrain_file:
        raise ValueError(
            "The saved case contains custom terrain; remove terrain_recipe before selecting --terrain"
        )
    case = replace(case, **edits)
    os.environ["JAX_PLATFORMS"] = args.device
    os.environ["JAX_ENABLE_X64"] = "1" if case.dtype == "float64" else "0"
    os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", str(Path.home() / ".cache/codesign-jax"))
    if args.offline_spaces:
        os.environ["CODESIGN_OFFLINE_SPACES"] = "1"
    else:
        os.environ.pop("CODESIGN_OFFLINE_SPACES", None)
    status: Report = {
        "completed": False,
        "stage": "imports",
        "mode": args.mode,
        "valid_rollout": None,
        "walking_gate_passed": False,
        "exitcode": 1,
    }
    plain_json(output / "command.json", vars(args))
    plain_json(output / "case.json", case.to_dict())
    plain_json(output / "status.json", status)
    env = None
    started = time.perf_counter()
    try:
        import jax
        import jax.numpy as jnp
        import numpy as np

        from envs.soft_walker.engine.physics.terrain import serialize
        from envs.soft_walker.experiments.scenarios.controllers import seed_parameters
        from envs.soft_walker.experiments.scenarios.rollout import assess, bind_loss
        from envs.soft_walker.presets.scenarios.build import build
        from envs.soft_walker.presets.terrain.scenario_courses import case_terrain
        from envs.soft_walker.shared import (
            append_event,
            environment_record,
            preflight,
            write_json,
            write_npz,
        )
        from envs.soft_walker.viewer.scenario_pictures import scene_picture
        from envs.soft_walker.viewer.trajectory_animation import render_animation

        jax.config.update("jax_default_matmul_precision", "highest")
        write_json(output / "manifest.json", environment_record(ROOT))
        write_json(output / "terrain.json", serialize(case_terrain(case)))

        def stage(name: str, **details: object) -> None:
            status.update(stage=name, **details)
            write_json(output / "status.json", status)
            append_event(
                output / "events.jsonl", name, elapsed_s=time.perf_counter() - started, **details
            )
            print(f"[{name}] {case.key}", flush=True)

        stage("geometry_preflight")
        check = preflight(case, design_mode=args.design_mode, material_mode=args.material_mode)
        write_json(output / "preflight.json", check)
        write_json(
            output / "initial_scene.json",
            scene_picture(
                case,
                output / "initial_scene.png",
                design_mode=args.design_mode,
                material_mode=args.material_mode,
            ),
        )
        if check["geometry_errors"]:
            raise ValueError(f"Geometry preflight: {check['geometry_errors']}")
        if args.mode == "preview":
            status.update(
                completed=True,
                stage="preview_finished",
                exitcode=0,
                scope="sampled initial geometry and mathematical screening; no dynamics",
            )
            return 0

        stage("factory_reset_settling")
        t0 = time.perf_counter()
        env, embodiment = build(
            case, device=args.device, design_mode=args.design_mode, material_mode=args.material_mode
        )
        write_json(output / "embodiment.json", embodiment)
        write_json(output / "decoder.json", env.decoder.descriptor())
        if args.execution == "resident":
            from envs.soft_walker.scripts._resident_commands import execute

            parameters = seed_parameters(case)
            if args.parameters:
                with np.load(args.parameters, allow_pickle=False) as saved:
                    loaded = saved["parameters"]
                if loaded.shape != parameters.shape or not np.isfinite(loaded).all():
                    raise ValueError("Controller parameter shape/finite check failed")
                parameters = loaded.astype(case.dtype)
            resident_result = execute(env, embodiment, case, parameters, args, family="scenario")
            status.update(
                completed=resident_result["finished"],
                stage="resident_finished",
                exitcode=0,
                valid_rollout=cast(Report, resident_result["assessment"]).get("execution_valid"),
                walking_gate_passed=cast(Report, resident_result["assessment"]).get(
                    "walking_gate_passed", False
                ),
                gpu_residency_demonstrated=resident_result["gpu_residency_demonstrated"],
            )
            return 0
        print("LEGACY DIAGNOSTIC EXECUTION: GPU residency is NOT enforced", flush=True)
        observation, rejected, information = env.reset(seed=case.seed, embodiment=embodiment)
        write_json(output / "reset.json", information)
        reset_seconds = time.perf_counter() - t0
        if bool(rejected[0]):
            raise RuntimeError(f"Reset rejected: {information.get('embodiment_errors')}")
        from envs.soft_walker.engine.physics.terrain import Terrain
        from envs.soft_walker.engine.types import DriverState, PhysicsModel

        model, initial = cast(PhysicsModel, env._model), cast(DriverState, env._state)
        # This is the exact contact geometry used by reset and subsequent steps.
        write_json(output / "terrain.json", serialize(cast(Terrain, model.terrain)))
        parameters = seed_parameters(case)
        if args.parameters:
            with np.load(args.parameters, allow_pickle=False) as stored:
                loaded = stored["parameters"]
            if loaded.shape != parameters.shape or not np.isfinite(loaded).all():
                raise ValueError("Controller parameter shape/finite check failed")
            parameters = loaded.astype(case.dtype)
        write_npz(output / "initial_parameters.npz", parameters=parameters)
        write_json(
            output / "controller.json",
            {
                "type": case.controller,
                "parameter_count": parameters.size,
                "feedback_observation_slice": [case.dim, 2 * case.dim]
                if case.controller in ("feedback", "neural")
                else None,
                "period_s": case.period,
                "gait_seed": case.gait,
                "qualification": "initial guesses; not optimized locomotion baselines",
            },
        )
        costs = {
            "factory_reset_settling_s": reset_seconds,
            "physical_steps": case.steps * case.numerical_substeps,
        }
        if args.mode in ("train", "gradcheck", "scout"):
            stage("loss_compile")
            loss = bind_loss(env, case, args.chunk_size)
            t0 = time.perf_counter()
            forward = jax.jit(loss).lower(model, initial, jnp.asarray(parameters)).compile()
            costs["loss_compile_s"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            initial_loss = float(
                jax.block_until_ready(forward(model, initial, jnp.asarray(parameters)))
            )
            costs["synchronized_forward_s"] = time.perf_counter() - t0
            costs["initial_loss"] = initial_loss
            costs["estimate_100_forward_s"] = 100 * costs["synchronized_forward_s"]
            write_json(output / "cost_probe.json", costs)
            if not np.isfinite(initial_loss):
                raise RuntimeError("Seed rollout failed; no optimizer will run on invalid physics")
            if args.mode == "scout":
                stage("vmapped_candidate_scout")
                rng = np.random.default_rng(case.seed)
                candidates = np.repeat(parameters[None], args.population, axis=0)
                candidates[1:] += rng.normal(scale=0.1, size=candidates[1:].shape)
                population_loss = jax.jit(jax.vmap(loss, in_axes=(None, None, 0)))
                t0 = time.perf_counter()
                losses = np.asarray(
                    jax.block_until_ready(population_loss(model, initial, jnp.asarray(candidates)))
                )
                write_npz(output / "population.npz", parameters=candidates, losses=losses)
                costs["population_compile_and_execution_s"] = time.perf_counter() - t0
                if not np.isfinite(losses).any():
                    raise RuntimeError("All independent candidates failed")
                parameters = candidates[np.argmin(np.where(np.isfinite(losses), losses, np.inf))]
            else:
                stage("value_and_gradient_compile")
                t0 = time.perf_counter()
                derivative = (
                    jax.jit(jax.value_and_grad(loss, argnums=2))
                    .lower(model, initial, jnp.asarray(parameters))
                    .compile()
                )
                costs["value_gradient_compile_s"] = time.perf_counter() - t0
                evaluate = lambda p: jax.block_until_ready(
                    derivative(model, initial, jnp.asarray(p))
                )
                if args.mode == "train":
                    from envs.soft_walker.shared import optimize

                    stage("optimization")
                    result = optimize(
                        evaluate,
                        parameters,
                        output / "optimization",
                        method=args.optimizer,
                        iterations=args.iterations,
                        learning_rate=args.learning_rate,
                    )
                    parameters = result.parameters
                    write_json(output / "optimization_result.json", result)
                else:
                    from envs.soft_walker.shared import difference_ladder

                    stage("task_gradient_ladder")
                    ladder = difference_ladder(
                        lambda p: forward(model, initial, jnp.asarray(p)),
                        evaluate,
                        parameters,
                        seed=case.seed,
                    )
                    write_json(output / "gradient_ladder.json", ladder)
                    status["gradient_agreement_passed"] = ladder["agreement_passed"]
            write_json(output / "cost_probe.json", costs)

        rows, xyz, commands, solver, topology = replay_chunks(
            env, case, args, model, initial, parameters, costs, stage, output
        )
        assessment = assess(rows, case, design_mode=args.design_mode)
        write_json(output / "assessment.json", assessment)
        if args.api_check:
            stage("public_api_equivalence")
            checks = []
            for i, command in enumerate(commands[:2]):
                _, _, _, _, information = env.step(command)
                actual = float(information["distance_m"])
                expected = float(rows[i + 1, 1])
                checks.append(
                    {
                        "interval": i,
                        "api": actual,
                        "functional": expected,
                        "passed": bool(np.isclose(actual, expected, rtol=1e-4, atol=1e-7)),
                    }
                )
            write_json(output / "api_equivalence.json", checks)
            if not all(c["passed"] for c in checks):
                raise RuntimeError("Public API / functional replay mismatch")
        if not args.no_animation:
            stage("animation_export")
            render_animation(
                xyz,
                rows,
                topology,
                output / "replay.gif",
                title=case.key,
                period=case.period,
                fps=args.fps,
                terrain=model.terrain,
                verdict="Candidate: inspect assessment.json; not gait-certified",
            )
        status.update(
            completed=True,
            stage="finished",
            valid_rollout=assessment["valid_rollout"],
            walking_gate_passed=assessment["walking_gate_passed"],
            repeated_airborne_observed=assessment["repeated_airborne_observed"],
            exitcode=0 if assessment["valid_rollout"] else 2,
        )
        if args.mode == "gradcheck" and not status.get("gradient_agreement_passed", False):
            status["exitcode"] = 3
    except BaseException as error:
        status.update(
            error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc()
        )
        print(status["traceback"], file=sys.stderr, flush=True)
    finally:
        if env is not None:
            env.close()
        status["total_wall_s"] = time.perf_counter() - started
        plain_json(output / "status.json", status)
        try:
            from envs.soft_walker.shared import archive_run

            target = archive_run(output)
        except Exception:
            target = output.with_name(output.name + ".zip")
            with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
                for file in output.rglob("*"):
                    if file.is_file() and not file.name.endswith(".tmp"):
                        archive.write(file, output.name + "/" + str(file.relative_to(output)))
        print(f"Evidence: {target}", flush=True)
    return status["exitcode"]


if __name__ == "__main__":
    raise SystemExit(main())
