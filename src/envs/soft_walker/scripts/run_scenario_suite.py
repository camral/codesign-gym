#!/usr/bin/env python3
"""Run explicit scenario plans sequentially, isolating compilations and failures.

Timeout is a wall-clock budget, not a solver tolerance. Skipped/failed cases are
listed, not misreported as a successful full-suite validation. No JAX import.
"""

from __future__ import annotations

import argparse

# Resolve integrated imports independently of the caller's working directory.
import sys as _script_sys
from pathlib import Path, Path as _ScriptPath
from typing import Any

_source_root = str(_ScriptPath(__file__).resolve().parents[3])
if _source_root not in _script_sys.path:
    _script_sys.path.insert(0, _source_root)
_script_dir = str(_ScriptPath(__file__).resolve().parent)
if _script_dir not in _script_sys.path:
    _script_sys.path.insert(0, _script_dir)
import hashlib
import json
import subprocess
import sys
import time
import zipfile

from envs.soft_walker.shared import (
    ENGINE_ROOT as ENGINE_ROOT,
    REPO_ROOT as REPO_ROOT,
    SCRIPTS_ROOT as SCRIPTS_ROOT,
    SOFT_ROOT as SOFT_ROOT,
    SRC_ROOT as SRC_ROOT,
    TESTS_ROOT as TESTS_ROOT,
)

# Serialized reports can contain backend-specific auxiliary diagnostic trees.
Report = dict[str, Any]


def main() -> int:
    """Command entry point. Run explicit scenario plans sequentially, isolating compilations and failures."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--mode", choices=("preview", "smoke", "replay"), default="preview")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument(
        "--execution",
        choices=("legacy", "resident"),
        default="legacy",
        help="Execution path forwarded to every worker (default: legacy)",
    )
    p.add_argument("--dtype", choices=("float64", "float32"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--timeout", type=float, default=1800.0)
    p.add_argument("--limit", type=int)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--no-animation", action="store_true")
    p.add_argument("--offline-spaces", action="store_true")
    a = p.parse_args()
    if a.timeout <= 0 or a.start < 0 or (a.limit is not None and a.limit < 1):
        p.error("Invalid timeout/start/limit")
    root = REPO_ROOT
    plan = json.loads(a.plan.read_text())
    out = a.out.expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        p.error("Output directory must be fresh")
    out.mkdir(parents=True, exist_ok=True)
    cases = plan["cases"][a.start :]
    if a.limit is not None:
        cases = cases[: a.limit]
    report: Report = {
        "mode": a.mode,
        "device": a.device,
        "execution": a.execution,
        "completed": False,
        "plan_sha256": hashlib.sha256(a.plan.read_bytes()).hexdigest(),
        "selected_cases": cases,
        "stages": [],
        "not_a_full_historical_suite": True,
    }
    status_path = out / "suite_status.json"

    def save() -> None:
        status_path.write_text(json.dumps(report, indent=2) + "\n")

    save()
    try:
        for i, relative in enumerate(cases, start=a.start):
            case_path = (a.plan.resolve().parent / relative).resolve()
            if case_path.parent != a.plan.resolve().parent:
                raise ValueError("Plan case paths must remain beside the plan")
            run = out / f"{i:02d}_{case_path.stem}"
            command = [
                sys.executable,
                str(root / "src/envs/soft_walker/scripts/scenarios.py"),
                "preview" if a.mode == "preview" else "replay",
                "--case",
                str(case_path),
                "--device",
                a.device,
                "--execution",
                a.execution,
                "--out",
                str(run),
                "--design-mode",
                plan.get("design_mode", "hard"),
                "--material-mode",
                plan.get("material_mode", "continuous"),
            ]
            if a.mode == "smoke":
                command += ["--tier", "smoke"]
                if a.execution == "legacy":
                    command.append("--api-check")
            if a.dtype:
                command += ["--dtype", a.dtype]
            if a.no_animation:
                command.append("--no-animation")
            if a.offline_spaces:
                command.append("--offline-spaces")
            entry: Report = {
                "case": relative,
                "command": command,
                "returncode": None,
                "case_sha256": hashlib.sha256(case_path.read_bytes()).hexdigest(),
            }
            report["stages"].append(entry)
            save()
            t0 = time.perf_counter()
            with (out / f"{i:02d}_stdout.log").open("w") as log:
                try:
                    completed = subprocess.run(
                        command, cwd=root, stdout=log, stderr=subprocess.STDOUT, timeout=a.timeout
                    )
                    entry["returncode"] = completed.returncode
                except subprocess.TimeoutExpired:
                    entry.update(returncode=124, timed_out=True)
            entry["wall_s"] = time.perf_counter() - t0
            if (run / "status.json").exists():
                entry["worker_status"] = json.loads((run / "status.json").read_text())
            save()
            print(
                json.dumps(
                    {"case": relative, "returncode": entry["returncode"], "wall_s": entry["wall_s"]}
                ),
                flush=True,
            )
        report["completed"] = True
        report["all_selected_passed"] = bool(report["stages"]) and all(
            s["returncode"] == 0 for s in report["stages"]
        )
    finally:
        save()
        hashes = {
            str(f.relative_to(out)): hashlib.sha256(f.read_bytes()).hexdigest()
            for f in sorted(out.rglob("*"))
            if f.is_file() and f.name != "SHA256SUMS" and f.suffix != ".zip"
        }
        (out / "SHA256SUMS").write_text("".join(f"{h}  {n}\n" for n, h in hashes.items()))
        target = out.with_name(out.name + ".zip")
        with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
            for f in sorted(out.rglob("*")):
                if f.is_file() and f.suffix != ".zip":
                    archive.write(f, out.name + "/" + str(f.relative_to(out)))
        print(f"Suite evidence: {target}", flush=True)
    return 0 if report.get("all_selected_passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
