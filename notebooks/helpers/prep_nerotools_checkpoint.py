"""Turn a NeroGrasp-Tools training run into a tutorial bundle that runs without a decomposition backend.

Steps:
  1. Pick the best ``best_checkpoint.pt`` under RUN_DIR (a ``nerotoolsworking_results_*`` folder).
  2. Decode its CBD-SA design into fingertip meshes, decompose them with a real backend, and roll
     the policy out in N evaluation worlds.
  3. Save ``<name>.pt`` (the checkpoint, unchanged), ``<name>_fingertips.npz`` (meshes and convex
     parts) and ``<name>_config.json`` (training config) to OUT_DIR.
  4. Build a new env that may only use the saved convex parts, run the same rollouts, and compare.

Examples (from the codesign-gym root):
  python notebooks/helpers/prep_nerotools_checkpoint.py nerotoolsworking_results_carabiner --num-runs 16
  python notebooks/helpers/prep_nerotools_checkpoint.py nerotoolsworking_results_wirecutters --num-runs 16
  # Check a saved bundle on another machine (no decomposition backend needed):
  python notebooks/helpers/prep_nerotools_checkpoint.py --load-only --name carabiner --num-runs 16
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shlex
import shutil
import time
from pathlib import Path

import numpy as np
import torch

import nerotools_bundle as nb

SEED_OFFSET = 3_000_003  # the training evaluator resets with config seed + this offset
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "nerotools_checkpoint_data"


def find_checkpoints(run_dir: Path) -> list[dict]:
    found = []
    for path in sorted(run_dir.rglob("artifacts/best_checkpoint.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        result = path.parent / "result.json"
        status = json.loads(result.read_text()).get("status", "?") if result.is_file() else "incomplete"
        found.append({"path": path, "run": path.parent.parent.name, "status": status,
                      "eval_reward": float(payload.get("eval_reward", float("nan"))),
                      "generation": payload.get("generation"), "payload": payload})
    return found


def training_command(config: dict, name: str) -> str | None:
    """Rebuild the ``fasttd3_cmaes.gym_train`` command from a run's config.json (non-default flags only)."""
    try:
        from fasttd3_cmaes.gym_train import build_parser
    except ImportError as error:
        print(f"  warning: cannot import fasttd3_cmaes.gym_train ({error}); not saving the training command")
        return None
    skip = {"help", "output_dir", "project", "entity", "group", "run_name"}  # paths and W&B identifiers
    lines = ["python -m fasttd3_cmaes.gym_train"]
    for action in build_parser()._actions:
        if action.dest in skip or action.dest not in config or not action.option_strings:
            continue
        value, flag = config[action.dest], action.option_strings[0]
        if isinstance(action, argparse.BooleanOptionalAction):
            if bool(value) != bool(action.default):
                lines.append(flag if value else "--no-" + flag[2:])
        elif value is not None and (action.required or value != action.default):
            lines.append(f"{flag} {shlex.quote(str(value))}")
    lines.append(f"--output-dir exp_results/{name}_seed_{config.get('seed', 0)}")
    return " \\\n  ".join(lines)


def default_name(env_id: str) -> str:
    return env_id.removeprefix("NeroGrasp-Tools-").lower()


def default_backend(device: str) -> str:
    # Training decomposed with cuacd; use it when available so the saved parts match training.
    if importlib.util.find_spec("cuacd") is not None and torch.cuda.is_available():
        return "cuacd"
    return "vhacd"


def print_results(title: str, res: dict, reference: float | None = None) -> None:
    print(f"\n{title}")
    print(f"  {'world':>5} {'noise':>6} {'return':>9} {'success':>8} {'articulation':>13} {'max lift cm':>12}")
    rows = zip(res["returns"], res["success"], res["articulation"], res["max_lift_cm"], res["noise"])
    for i, (ret, ok, art, lift, noise) in enumerate(rows):
        print(f"  {i:>5} {noise:>6.2f} {ret:>9.3f} {str(bool(ok)):>8} {art:>13.3f} {lift:>12.2f}")
    r = res["returns"]
    print(f"  mean return {r.mean():.3f} ± {r.std():.3f} | success rate {res['success'].mean():.0%} "
          f"| mean articulation {res['articulation'].mean():.3f} | mean max lift {res['max_lift_cm'].mean():.2f} cm")
    if reference is not None:
        print(f"  (training eval reward: {reference:.3f}, 64 worlds on the training GPU)")


def run_rollouts(env_id, policy_path, design, theta, bound, *, num_runs, device, seed, backend=None):
    started = time.perf_counter()
    env, _ = nb.make_tools_env(env_id, num_runs, device=device, seed=seed, backend=backend)
    print(f"  env built in {time.perf_counter() - started:.1f}s")
    try:
        _, policy = nb.load_tools_policy(policy_path, device)
        started = time.perf_counter()
        res = nb.rollout(env, policy, design, theta, bound, seed=seed)
        print(f"  rollouts finished in {time.perf_counter() - started:.1f}s")
        meshes = nb.harvest_decompositions(env) if backend is not None else None
    finally:
        env.close()
    return res, meshes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", nargs="?", type=Path, help="nerotoolsworking_results_* folder")
    parser.add_argument("--run", help="substring of the run folder to use (default: highest eval reward)")
    parser.add_argument("--name", help="bundle name (default: tool name from the env id, e.g. carabiner)")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--num-runs", type=int, default=16, help="evaluation worlds (one rollout each)")
    parser.add_argument("--device", default="cpu", help="simulation and policy device (default: cpu)")
    parser.add_argument("--seed", type=int, help="rollout seed (default: training seed + 3000003, as in training eval)")
    parser.add_argument("--backend", choices=("cuacd", "vhacd", "visacd"),
                        help="decomposition backend for the first pass (default: cuacd if available, else vhacd)")
    parser.add_argument("--atol", type=float, default=1e-6, help="max per-world |return difference| counted as consistent")
    parser.add_argument("--load-only", action="store_true", help="skip steps 1-3 and only check an existing bundle")
    args = parser.parse_args()

    fresh = None
    if not args.load_only:
        if args.run_dir is None:
            parser.error("run_dir is required unless --load-only is given")
        candidates = find_checkpoints(args.run_dir)
        if not candidates:
            parser.error(f"no artifacts/best_checkpoint.pt under {args.run_dir}")
        print("Checkpoints found:")
        for c in candidates:
            print(f"  {c['run']:<28} status={c['status']:<10} generation={c['generation']!s:<4} eval_reward={c['eval_reward']:.3f}")
        if args.run:
            candidates = [c for c in candidates if args.run in c["run"]]
            if len(candidates) != 1:
                parser.error(f"--run {args.run!r} must match exactly one run")
        chosen = max(candidates, key=lambda c: c["eval_reward"])
        payload = chosen["payload"]
        env_id, bound = payload["env_id"], float(payload["mesh_bound_m"])
        theta = np.asarray(payload["design"], dtype=np.float32)
        if not payload.get("use_cbd", False):
            raise SystemExit("this script expects a checkpoint trained with --use-cbd")
        sibling = chosen["path"].parent / "best_embodiment.npy"
        if sibling.is_file() and not np.allclose(np.load(sibling), theta):
            print("  warning: best_embodiment.npy differs from the checkpoint design; using the checkpoint design")
        name = args.name or default_name(env_id)
        seed = args.seed if args.seed is not None else int(payload["seed"]) + SEED_OFFSET
        backend = args.backend or default_backend(args.device)
        print(f"Using {chosen['run']} ({env_id}), bundle name '{name}', seed {seed}, device {args.device}")
        if backend != "cuacd":
            print(f"  note: decomposing with {backend}; training used cuacd, so results may differ from training eval")

        # 1-2. Decode the design and run with a real decomposition backend.
        from fasttd3_cmaes.cbdsa_codec import CBDSAMeshSpaceCodec

        probe, _ = nb.make_tools_env(env_id, 1, device=args.device, seed=seed, backend=backend)
        codec = CBDSAMeshSpaceCodec(probe.embodiment_space, bound_m=bound)
        design = codec.decode(theta)
        base = {tip: mesh.copy() for tip, mesh in probe.embodiment_space.base_meshes.items()}
        probe.close()
        print(f"\n[1/3] Fresh decomposition ({backend}) + {args.num_runs} rollouts")
        fresh, meshes = run_rollouts(env_id, chosen["path"], design, theta, bound,
                                     num_runs=args.num_runs, device=args.device, seed=seed, backend=backend)
        print_results("Fresh decomposition results:", fresh, chosen["eval_reward"])

        # 3. Save the bundle.
        args.out_dir.mkdir(parents=True, exist_ok=True)
        ckpt_out, npz_out = nb.bundle_paths(args.out_dir, name)
        shutil.copy2(chosen["path"], ckpt_out)
        config = chosen["path"].parent / "config.json"
        command = None
        if config.is_file():
            shutil.copy2(config, args.out_dir / f"{name}_config.json")
            command = training_command(json.loads(config.read_text()), name)
        nb.save_fingertips(npz_out, env_id=env_id, theta=theta, mesh_bound_m=bound, design=design, base=base,
                           operators=codec.operators, meshes=meshes, backend=backend, train_command=command)
        n_parts = sum(len(parts) for _, _, parts in meshes)
        print(f"\n[2/3] Saved {ckpt_out.name} and {npz_out.name} ({len(meshes)} meshes, {n_parts} convex parts, "
              f"{npz_out.stat().st_size / 1024:.0f} KB) to {args.out_dir}")
    else:
        if args.name is None:
            parser.error("--load-only needs --name")
        name = args.name

    # 4. Reload with precomputed decompositions only.
    bundle = nb.load_bundle(args.out_dir, name)
    payload = torch.load(bundle.checkpoint, map_location="cpu", weights_only=False)
    seed = args.seed if args.seed is not None else int(payload["seed"]) + SEED_OFFSET
    print(f"\n[3/3] Reload {name}: precomputed decompositions ({bundle.backend}) only + {args.num_runs} rollouts, seed {seed}")
    with nb.precomputed_decompositions(bundle):
        reloaded, _ = run_rollouts(bundle.env_id, bundle.checkpoint, bundle.design, bundle.theta, bundle.mesh_bound_m,
                                   num_runs=args.num_runs, device=args.device, seed=seed)
    print_results("Precomputed decomposition results:", reloaded, float(payload.get("eval_reward", float("nan"))))

    if fresh is not None:
        diff = np.abs(fresh["returns"] - reloaded["returns"])
        same_success = bool(np.array_equal(fresh["success"], reloaded["success"]))
        ok = bool(diff.max() <= args.atol) and same_success
        print(f"\nConsistency: max |return difference| = {diff.max():.3g} (tolerance {args.atol:g}), "
              f"success flags {'match' if same_success else 'differ'} -> {'PASS' if ok else 'FAIL'}")
        if not ok:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
