"""Audit every registered factory in fresh processes, retaining stdout/stderr.

Usage: python tests/audit_environment_output.py --output /tmp/codesign-audit
Optional backends must be installed to exercise their constructors. Failures are
recorded, never silently counted as successful summary checks.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import io
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def audit_case(args):
    import codesign_gym as cg

    result = {"env_id": args.env, "mode": args.mode}
    env = None
    started = time.monotonic()
    try:
        kwargs = {}
        num_envs = 2
        if args.env.startswith("NeroGrasp-"):
            kwargs.update(runtime_dir=str(Path(args.output).parent / "runtime"), shape_parallel_workers=1)
            if args.mode == "vector" and not args.env.startswith("NeroGrasp-Tools-"):
                from envs.nerograsp.mujoco_primitive_shapes_utils import resolve_object_names
                selector = cg.REGISTRY[args.env][1]["object_name"]
                num_envs = 2 * len(resolve_object_names(selector))
        if args.env.startswith("Pokenv-"):
            kwargs["shards"] = 1
        result["num_envs"] = num_envs if args.mode == "vector" else 1
        if args.mode == "single":
            env, _ = cg.make(args.env, **kwargs)
        else:
            env, _ = cg.make_vec(args.env, num_envs, **kwargs)
        space = env.embodiment_space
        representation = repr(space)
        printed = io.StringIO()
        print(space, file=printed)
        assert space is not None
        assert representation == str(space) == printed.getvalue().rstrip("\n")
        assert len(representation) < 1500, "Embodiment summary dumps large arrays"
        assert not re.search(r"object at 0x[0-9a-f]+", representation)
        result.update(status="ok", summary=representation, space_type=type(space).__name__)
    except Exception as error:
        result.update(status="error", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as error:
                result.update(status="error", close_error=repr(error))
        result["seconds"] = round(time.monotonic() - started, 2)
        Path(args.output).write_text(json.dumps(result, indent=2))


def run_process(command, env, timeout):
    # A timed-out constructor may own Showdown servers or mesh workers. Give its
    # subprocess group the same lifetime as the audit case instead of leaking it.
    with subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          text=True, env=env, start_new_session=True) as process:
        try:
            stdout, stderr = process.communicate(timeout=timeout)
            return process.returncode, stdout, stderr, False
        except subprocess.TimeoutExpired:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
            stdout, stderr = process.communicate()
            return process.returncode, stdout, stderr, True


def audit_registry(args):
    import codesign_gym as cg

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    cache = output / ".cache"
    cache.mkdir(exist_ok=True)
    child_env = dict(os.environ)
    child_env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(ROOT / "src"), os.environ.get("PYTHONPATH")]))
    child_env.setdefault("MPLCONFIGDIR", str(cache / "matplotlib"))
    child_env.setdefault("XDG_CACHE_HOME", str(cache))
    ids = [name for name in cg.list_envs() if re.search(args.select, name)]

    def run(case):
        name, mode = case
        folder = output / f"{name}_{mode}"
        folder.mkdir(exist_ok=True)
        destination = folder / "result.json"
        destination.unlink(missing_ok=True)
        command = [sys.executable, __file__, "--env", name, "--mode", mode, "--output", str(destination)]
        code, stdout, stderr, timed_out = run_process(command, child_env, args.timeout)
        (folder / "stdout.txt").write_text(stdout)
        (folder / "stderr.txt").write_text(stderr)
        if code == 0 and destination.exists():
            result = json.loads(destination.read_text())
        else:
            result = {"env_id": name, "mode": mode, "status": "timeout" if timed_out else "crashed", "exit_code": code}
        result.update(stdout=stdout, stderr=stderr)
        print(f"{name} {mode} {result['status']}", flush=True)
        return result

    cases = [(name, mode) for name in ids for mode in ("single", "vector")]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(run, cases))
    for name in ids:
        pair = [result for result in results if result["env_id"] == name and result["status"] == "ok"]
        if len(pair) == 2:
            matches = pair[0]["summary"] == pair[1]["summary"]
            for result in pair:
                result["summary_matches_other_factory"] = matches
                if not matches:
                    result["status"] = "summary_mismatch"
    (output / "results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", help=argparse.SUPPRESS)
    parser.add_argument("--mode", choices=("single", "vector"), help=argparse.SUPPRESS)
    parser.add_argument("--output", required=True)
    parser.add_argument("--select", default=".", help="Regular expression selecting registered IDs")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=180, help="Timeout in seconds per constructor")
    args = parser.parse_args()
    if args.workers < 1 or args.timeout < 1:
        parser.error("workers and timeout must be positive")
    if args.env:
        audit_case(args)
    else:
        audit_registry(args)
