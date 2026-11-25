#!/usr/bin/env python3
"""Run, optimize, audit, and animate actual legged-robot trajectories.

Each invocation uses a fresh output folder and automatically creates an evidence
ZIP beside it, including on failure. No historical validation suite is rerun.
Device and precision are selected before importing JAX. CUDA never falls back.
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
import shutil
import sys
import time
import traceback

from _locomotion_commands import ExperimentRecorder, build_config, controller_parameters

from envs.soft_walker.shared import (
    ENGINE_ROOT as ENGINE_ROOT,
    REPO_ROOT as REPO_ROOT,
    SCRIPTS_ROOT as SCRIPTS_ROOT,
    SOFT_ROOT as SOFT_ROOT,
    SRC_ROOT as SRC_ROOT,
    TESTS_ROOT as TESTS_ROOT,
)

if TYPE_CHECKING:
    from pathlib import Path

    from envs.soft_walker.engine.types import (
        Array,
    )

# Serialized reports can contain backend-specific auxiliary diagnostic trees.
Report = dict[str, Any]


def parse_args() -> argparse.Namespace:
    """Parse the existing command-line options and reject incompatible combinations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("replay", "train", "gradcheck", "scout"))
    parser.add_argument(
        "--driver", choices=("spring", "mpm2", "mpm3", "beam", "fem"), default="mpm2"
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--dtype", choices=("float64", "float32"), default="float64")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cycles", type=int, default=6)
    parser.add_argument("--period", type=float, default=0.5)
    parser.add_argument("--control-dt", type=float, default=0.02)
    parser.add_argument("--substeps", type=int, default=40)
    parser.add_argument("--settle-seconds", type=float, default=0.2)
    parser.add_argument("--refinement", type=int, default=1)
    parser.add_argument("--grid-margin", type=int, default=5)
    parser.add_argument("--travel-body-lengths", type=float, default=3.0)
    parser.add_argument(
        "--gmres-method",
        choices=("incremental", "batched", "fused"),
        default="incremental",
        help="Krylov implementation; fused removes nested dynamic loops",
    )
    parser.add_argument("--spring-solver", choices=("dense", "gmres"), default="dense")
    parser.add_argument("--friction", type=float, default=0.8)
    parser.add_argument("--friction-velocity", type=float, default=0.003)
    parser.add_argument("--gravity", type=float, default=9.81)
    parser.add_argument("--max-log-stretch", type=float, default=0.3)
    parser.add_argument("--actuator-tau", type=float, default=0.03)
    parser.add_argument(
        "--actuation-model", choices=("active_strain", "active_stress"), default="active_strain"
    )
    parser.add_argument("--active-stress-ratio", type=float, default=0.4)
    parser.add_argument("--amplitude", type=float, default=0.8)
    parser.add_argument("--lift", type=float, default=0.5)
    parser.add_argument("--phase", type=float, default=0.0)
    parser.add_argument("--gait", choices=("alternating", "crawl", "pronk"), default="alternating")
    parser.add_argument(
        "--objective",
        choices=("travel", "hop"),
        default="travel",
        help="Hop rewards smooth COM lift/uprightness and penalizes torso contact; it does not certify flight",
    )
    parser.add_argument("--passive", action="store_true")
    parser.add_argument("--harmonics", type=int, default=1)
    parser.add_argument("--leg-young", type=float, default=12000.0)
    parser.add_argument("--torso-young", type=float, default=60000.0)
    parser.add_argument("--strength", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--controller", type=Path, help="NPZ containing coefficients under 'parameters'"
    )
    parser.add_argument("--optimizer", choices=("adam", "gd", "lbfgs"), default="adam")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument(
        "--resume", type=Path, help="Resume Adam/GD state from last.npz; same task required"
    )
    parser.add_argument("--population", type=int, default=4)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--no-animation", action="store_true")
    parser.add_argument(
        "--animate-every-iteration",
        action="store_true",
        help="Save a checkpoint, physical replay, and GIF after each Adam/GD iteration",
    )
    parser.add_argument(
        "--execution",
        choices=("resident", "legacy"),
        default="resident",
        help="resident: whole-episode graph with mandatory CUDA certificate; legacy: diagnostic host path",
    )
    parser.add_argument(
        "--record-stride",
        type=int,
        default=1,
        help="Store every Nth position on device; no CPU recording chunks",
    )
    parser.add_argument(
        "--recording-mib",
        type=int,
        default=512,
        help="Maximum output buffer; reject rather than host-offload",
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--deterministic-reductions",
        action="store_true",
        help="Use deterministic resident CUDA reductions for strict gradient repeats; may cost runtime",
    )
    parser.add_argument("--block-size", type=int, default=5)
    parser.add_argument(
        "--api-check",
        action="store_true",
        help="Compare first two recorded intervals with ordinary env.step",
    )
    parser.add_argument(
        "--offline-spaces",
        action="store_true",
        help="Explicit offline development only; not Gymnasium validation",
    )
    args = parser.parse_args()
    if args.animate_every_iteration and (
        args.mode != "train" or args.optimizer not in ("adam", "gd") or args.no_animation
    ):
        parser.error("--animate-every-iteration requires Adam/GD training with animation enabled")
    return args


def validate_resume_objective(previous_file: Path, current: Report) -> None:
    """Older checkpoints have the unchanged travel loss, never an implicit hop loss."""
    if previous_file.exists():
        if json.loads(previous_file.read_text()) != current:
            raise ValueError(
                "Resume objective changed; start a fresh optimizer for a different loss"
            )
    elif current["name"] != "travel":
        raise ValueError("Resume lacks objective.json; historical checkpoints are travel-only")


def main() -> int:
    """Run the selected locomotion workflow and always save status plus an evidence archive."""
    args = parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit("Output directory must be fresh; resume into a new directory")
    args.out.mkdir(parents=True, exist_ok=True)
    os.environ["JAX_PLATFORMS"] = args.device
    os.environ["JAX_ENABLE_X64"] = "1" if args.dtype == "float64" else "0"
    os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", str(Path.home() / ".cache/codesign-jax"))
    if args.offline_spaces:
        os.environ["CODESIGN_OFFLINE_SPACES"] = "1"
    # The bootstrap file exists even if JAX or Gymnasium import fails.
    (args.out / "command.json").write_text(json.dumps(vars(args), default=str, indent=2))
    status: Report = {"completed": False, "stage": "imports", "walking_gate_passed": False}
    env = None
    exitcode = 1
    try:
        import jax
        import jax.numpy as jnp
        import numpy as np

        from envs.soft_walker.experiments.locomotion.rollout import (
            ObjectiveConfig,
            bind_experiment,
            objective_record,
        )
        from envs.soft_walker.presets.locomotion import DemoConfig, build_demo
        from envs.soft_walker.shared import append_event, environment_record, write_json, write_npz

        root = REPO_ROOT
        manifest = environment_record(root)
        write_json(args.out / "manifest.json", manifest)
        jax.config.update("jax_default_matmul_precision", "highest")
        config = build_config(args)
        write_json(args.out / "protocol.json", config)
        objective = ObjectiveConfig(name=args.objective)
        objective_metadata = objective_record(objective, config)
        write_json(args.out / "objective.json", objective_metadata)
        status["stage"] = "reset_and_settling"
        t = time.perf_counter()
        env, embodiment = build_demo(config, device=args.device)
        embodiment["young"] = np.asarray([args.leg_young, args.torso_young], args.dtype)
        embodiment["strength"] = np.asarray(args.strength, args.dtype)
        write_json(args.out / "embodiment.json", embodiment)
        if args.resume:
            previous_protocol = args.resume.parent / "protocol.json"
            previous_design = args.resume.parent / "embodiment.json"
            if not previous_protocol.exists() or not previous_design.exists():
                raise ValueError(
                    "Resume requires protocol.json and embodiment.json beside checkpoint"
                )
            if DemoConfig(**json.loads(previous_protocol.read_text())) != config or json.loads(
                previous_design.read_text()
            ) != json.loads((args.out / "embodiment.json").read_text()):
                raise ValueError(
                    "Resume task/design changed; use --controller to warm-start instead"
                )
            validate_resume_objective(args.resume.parent / "objective.json", objective_metadata)
        if args.execution == "resident":
            # No reset call here: numerical initialization and settling belong
            # inside the same executable as the complete forward/adjoint run.
            from envs.soft_walker.scripts._resident_commands import execute

            parameters = controller_parameters(args, config)
            resident_result = execute(
                env, embodiment, config, parameters, args, family="demo", objective=objective
            )
            status.update(
                completed=resident_result["finished"],
                stage="resident_finished",
                valid_rollout=cast(Report, resident_result["assessment"]).get("execution_valid"),
                walking_gate_passed=cast(Report, resident_result["assessment"]).get(
                    "walking_gate_passed", False
                ),
                gpu_residency_demonstrated=resident_result["gpu_residency_demonstrated"],
            )
            exitcode = 0
            return exitcode
        print("LEGACY DIAGNOSTIC EXECUTION: GPU residency is NOT enforced", flush=True)
        obs, rejected, info = env.reset(seed=args.seed, embodiment=embodiment)
        reset_time = time.perf_counter() - t
        write_json(args.out / "reset.json", info)
        if rejected[0]:
            raise RuntimeError(f"Reset rejected: {info.get('embodiment_errors')}")
        print(f"{args.driver}: reset/settling {reset_time:.3f}s", flush=True)
        model, initial = env._model, env._state
        parameters = controller_parameters(args, config)
        write_npz(args.out / "initial_controller.npz", parameters=parameters)
        t = time.perf_counter()
        forward = bind_experiment(env, config, block_size=args.block_size, objective=objective)
        loss_fn = lambda m, s, p: forward(m, s, p)["loss"]
        loss_jit = jax.jit(loss_fn)
        value_grad = jax.jit(jax.value_and_grad(loss_fn, argnums=2))
        compile_times: dict[str, float] = {}

        export_recording = ExperimentRecorder(
            env, config, args, model, initial, objective, compile_times
        )

        def record_iteration(
            *, iteration: int, parameters: Array, loss: float, accepted: bool, backtracks: int
        ) -> None:
            directory = args.out / "iterations" / f"iteration_{iteration:04d}"
            directory.mkdir(parents=True, exist_ok=False)
            # optimize writes last.npz before invoking this observer. Preserve
            # its momentum and absolute iteration, not only the coefficients.
            shutil.copy2(args.out / "last.npz", directory / "last.npz")
            shutil.copy2(args.out / "protocol.json", directory / "protocol.json")
            shutil.copy2(args.out / "embodiment.json", directory / "embodiment.json")
            shutil.copy2(args.out / "objective.json", directory / "objective.json")
            replay, _, _, wall = export_recording(parameters, directory, iteration=iteration)
            matches = bool(
                np.isclose(
                    float(replay["loss"]),
                    loss,
                    rtol=1e-8 if args.dtype == "float64" else 1e-4,
                    atol=1e-10 if args.dtype == "float64" else 1e-6,
                )
            )
            row: Report = dict(
                iteration=iteration,
                accepted=accepted,
                loss=loss,
                backtracks=backtracks,
                replay_valid=bool(replay["valid"]),
                replay_loss_matches=matches,
                synchronized_replay_s=wall,
                gif=str(directory / "replay.gif"),
            )
            write_json(directory / "iteration.json", row)
            append_event(args.out / "iteration_replays.jsonl", "recording", **row)
            print(
                f"iteration {iteration}: loss={loss:+.7f}; accepted={accepted}; "
                f"backtracks={backtracks}; replay={wall:.2f}s; GIF={directory / 'replay.gif'}",
                flush=True,
            )
            if not row["replay_valid"] or not matches:
                raise RuntimeError(
                    "Iteration recording disagrees with the accepted optimizer state"
                )

        initial_loss = None
        # A plain replay compiles/runs the recording path only. Repeating the
        # same long trajectory for timing would needlessly multiply demo cost.
        if args.mode != "replay":
            compiled_loss = loss_jit.lower(model, initial, jnp.asarray(parameters)).compile()
            compile_times["loss_compile_s"] = time.perf_counter() - t
            t = time.perf_counter()
            initial_loss = jax.block_until_ready(
                compiled_loss(model, initial, jnp.asarray(parameters))
            )
            forward_wall = time.perf_counter() - t
            write_json(
                args.out / "cost_probe.json",
                {
                    "compile": compile_times,
                    "synchronized_forward_s": forward_wall,
                    "initial_loss": initial_loss,
                    "estimate_100_forward_evaluations_s": 100 * forward_wall,
                    "physical_duration_s": config.cycles * config.period,
                    "num_physics_steps": config.steps * config.substeps,
                },
            )
            print(
                f"{args.driver}: loss={float(initial_loss):+.6g}; forward={forward_wall:.3f}s",
                flush=True,
            )
        if initial_loss is not None and not np.isfinite(float(initial_loss)):
            status["stage"] = "failed_seed_replay"
        elif args.mode in ("train", "gradcheck"):
            status["stage"] = "gradient_compile"
            t = time.perf_counter()
            compiled_grad = value_grad.lower(model, initial, jnp.asarray(parameters)).compile()
            compile_times["value_gradient_compile_s"] = time.perf_counter() - t

            def vg(p: Array) -> tuple[Array, Array]:
                return jax.block_until_ready(compiled_grad(model, initial, jnp.asarray(p)))

            if args.mode == "train":
                from envs.soft_walker.shared import optimize

                status["stage"] = "optimization"
                result = optimize(
                    vg,
                    parameters,
                    args.out,
                    method=args.optimizer,
                    iterations=args.iterations,
                    learning_rate=args.learning_rate,
                    resume=args.resume,
                    on_iteration=record_iteration if args.animate_every_iteration else None,
                )
                parameters = result.parameters
                status["optimization_completed"] = result.completed
                print(
                    f"optimizer: {result.initial_loss:+.6g} -> {result.best_loss:+.6g}", flush=True
                )
            else:
                from envs.soft_walker.shared import difference_ladder

                status["stage"] = "gradient_ladder"
                ladder = difference_ladder(
                    lambda p: jax.block_until_ready(compiled_loss(model, initial, jnp.asarray(p))),
                    vg,
                    parameters,
                    seed=args.seed,
                )
                write_json(args.out / "gradient_ladder.json", ladder)
                status["gradient_agreement_passed"] = ladder["agreement_passed"]
        elif args.mode == "scout":
            status["stage"] = "vmapped_population"
            if args.population < 1:
                raise ValueError("Population must be positive")
            rng = np.random.default_rng(args.seed)
            candidates = np.repeat(parameters[None], args.population, axis=0)
            candidates[1:] += rng.normal(scale=0.2, size=candidates[1:].shape)
            batched = jax.jit(jax.vmap(loss_fn, in_axes=(None, None, 0)))
            t = time.perf_counter()
            losses = np.asarray(
                jax.block_until_ready(batched(model, initial, jnp.asarray(candidates)))
            )
            write_npz(args.out / "population.npz", parameters=candidates, losses=losses)
            write_json(
                args.out / "population.json", {"wall_s": time.perf_counter() - t, "losses": losses}
            )
            if np.isfinite(losses).any():
                parameters = candidates[np.argmin(np.where(np.isfinite(losses), losses, np.inf))]
        status["stage"] = "recorded_replay"
        replay, records, assessment, _ = export_recording(parameters, args.out)
        if args.api_check:
            status["stage"] = "api_equivalence"
            checks = []
            for k in range(min(2, len(records))):
                _, _, terminal, _, step_info = env.step(np.asarray(replay["actions"][k]))
                expected = records[k, 1]
                actual = float(step_info["distance_m"])
                checks.append(
                    {
                        "step": k,
                        "recorded": expected,
                        "api": actual,
                        "passed": bool(np.isclose(expected, actual, rtol=1e-4, atol=1e-7))
                        and not terminal,
                    }
                )
            write_json(args.out / "api_equivalence.json", checks)
            if not all(c["passed"] for c in checks):
                raise RuntimeError("API and functional replay disagree")
        status.update(
            completed=True,
            stage="finished",
            valid_rollout=bool(replay["valid"]),
            walking_gate_passed=assessment["walking_gate_passed"],
            repeated_airborne_observed=json.loads(
                (args.out / "flight_assessment.json").read_text()
            )["repeated_airborne_observed"],
        )
        exitcode = 0 if replay["valid"] else 2
        if args.mode == "gradcheck" and not status.get("gradient_agreement_passed", False):
            exitcode = 3
        print(json.dumps({"valid": bool(replay["valid"]), **assessment}, indent=2), flush=True)
    except BaseException as error:
        status.update(
            error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc()
        )
        print(status["traceback"], file=sys.stderr, flush=True)
    finally:
        if env is not None:
            env.close()
        # Stdlib fallback produces an honest failure archive even on import errors.
        status["exitcode"] = exitcode
        (args.out / "status.json").write_text(json.dumps(status, indent=2))
        try:
            from envs.soft_walker.shared import archive_run

            target = archive_run(args.out)
        except Exception:
            import zipfile

            target = args.out.with_name(args.out.name + ".zip")
            with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
                for p in args.out.rglob("*"):
                    if p.is_file():
                        archive.write(p, args.out.name + "/" + str(p.relative_to(args.out)))
        print(f"Evidence: {target}", flush=True)
    return exitcode


if __name__ == "__main__":
    raise SystemExit(main())
