#!/usr/bin/env python3
"""Reanalyze saved residency evidence without compiling or running a simulator.

Accept an extracted run directory or its ZIP. Historical certificates are never
rewritten; a separate report records current parser decisions, original errors,
input integrity, and analyzer hashes. Passing review is NOT a fresh GPU run.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import sys
import traceback
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

if __package__ in (None, ""):
    # Direct-file execution must select this checkout before package imports.
    _source_root = str(Path(__file__).resolve().parents[3])
    if _source_root not in sys.path:
        sys.path.insert(0, _source_root)

from envs.soft_walker.shared import (
    ENGINE_ROOT as ENGINE_ROOT,
    REPO_ROOT as REPO_ROOT,
    SCRIPTS_ROOT as SCRIPTS_ROOT,
    SOFT_ROOT as SOFT_ROOT,
    SRC_ROOT as SRC_ROOT,
    TESTS_ROOT as TESTS_ROOT,
)

ROOT = REPO_ROOT
sys.path[:0] = [str(ROOT / "src")]

# Serialized reports can contain backend-specific auxiliary diagnostic trees.
Report = dict[str, Any]


class Evidence:
    """Read a bounded archive or directory without extracting untrusted paths."""

    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        self.archive = None
        if path.is_dir():
            self.names = [p.relative_to(path).as_posix() for p in path.rglob("*") if p.is_file()]
        else:
            self.archive = zipfile.ZipFile(path)
            self.names = [i.filename for i in self.archive.infolist() if not i.is_dir()]
            if len(self.names) != len(set(self.names)):
                raise ValueError("Duplicate archive members")
            if not any(n.endswith("submission_audit.json") for n in self.names):
                nested = [n for n in self.names if n.endswith(".zip")]
                if len(nested) == 1:
                    data = self.read(nested[0])
                    self.archive.close()
                    self.archive = zipfile.ZipFile(io.BytesIO(data))
                    self.names = [i.filename for i in self.archive.infolist() if not i.is_dir()]
                    if len(self.names) != len(set(self.names)):
                        raise ValueError("Duplicate nested archive members")
        for name in self.names:
            p = PurePosixPath(name)
            if p.is_absolute() or ".." in p.parts or "\\" in name:
                raise ValueError("Unsafe evidence path: " + name)

    def read(self, name: str) -> bytes:
        """Read one relative member; reject oversized ZIP entries or escaping symlinks.

        Missing members propagate the underlying archive or filesystem exception.
        The caller owns the returned bytes; the source evidence is never changed.
        """
        if self.archive:
            if self.archive.getinfo(name).file_size > 512 * 1024**2:
                raise ValueError("Evidence member exceeds 512 MiB read limit")
            return self.archive.read(name)
        path = (self.path / name).resolve()
        if not path.is_relative_to(self.path):
            raise ValueError("Evidence symlink escapes run directory")
        return path.read_bytes()

    def close(self) -> None:
        """Close an owned ZIP handle; directory-backed evidence needs no cleanup."""
        if self.archive:
            self.archive.close()


def main() -> int:
    """Command entry point. Reanalyze saved residency evidence without compiling or running a simulator."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if out.suffix == ".zip" or out.exists() and any(out.iterdir()):
        parser.error("--out must be a fresh directory, not a ZIP filename")
    out.mkdir(parents=True, exist_ok=True)
    report: Report = {
        "passed": False,
        "new_gpu_execution": False,
        "historical_certificate_modified": False,
        "review_only": True,
        "cases": [],
        "integrity_errors": [],
        "checked_files": 0,
    }
    source = None
    try:
        source = Evidence(args.run)
        from envs.soft_walker.engine.execution.submissions import inspect_submissions
        from envs.soft_walker.engine.execution.trace import inspect_trace

        # Verify each archived checksum map independently. A folder without a
        # checksum map can be reviewed, but its incomplete integrity is explicit.
        maps = [n for n in source.names if n.endswith("SHA256SUMS.json")]
        report["checksum_maps_found"] = len(maps)
        for name in maps:
            base = name.rsplit("/", 1)[0] + "/" if "/" in name else ""
            sums = json.loads(source.read(name))
            for relative, expected in sums.items():
                member = base + relative
                if (
                    member not in source.names
                    or hashlib.sha256(source.read(member)).hexdigest() != expected
                ):
                    report["integrity_errors"].append(member)
                report["checked_files"] += 1
        for name in sorted(n for n in source.names if n.endswith("submission_audit.json")):
            base = name.rsplit("/", 1)[0] + "/" if "/" in name else ""
            old = json.loads(source.read(name))
            submission = inspect_submissions(
                old.get("driver_api_counts", {}),
                old.get("driver_api_records", []),
                records_complete=old.get("recording_complete", False),
            )
            traces = [
                n
                for n in source.names
                if n.startswith(base + "trace/") and n.endswith(".trace.json.gz")
            ]
            case: Report = {
                "submission": name,
                "submission_audit": submission,
                "original_submission_passed": old.get("passed"),
                "trace_count": len(traces),
                "passed": False,
            }
            if len(traces) == 1:
                data = source.read(traces[0])
                case["trace_sha256"] = hashlib.sha256(data).hexdigest()
                case["trace_audit"] = inspect_trace(
                    json.loads(gzip.decompress(data)), submission_audit=old
                )
                case["passed"] = submission["passed"] and case["trace_audit"]["passed"]
            else:
                case["error"] = (
                    "Missing or ambiguous detailed trace; no complete reanalysis claimed"
                )
            old_trace = base + "detailed_trace_audit.json"
            if old_trace in source.names:
                prior = json.loads(source.read(old_trace))
                case["original_trace_passed"] = prior.get("passed")
                case["original_trace_errors"] = prior.get("errors")
            report["cases"].append(case)
        report["analyzer_sha256"] = {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT / "src/envs/soft_walker/engine/execution").glob("*.py"))
        }
        report["passed"] = (
            bool(report["cases"])
            and not report["integrity_errors"]
            and all(c["passed"] for c in report["cases"])
        )
    except Exception as error:
        report.update(exception=repr(error), traceback=traceback.format_exc())
    finally:
        if source:
            source.close()
        (out / "review.json").write_text(json.dumps(report, indent=2) + "\n")
        with zipfile.ZipFile(out.with_suffix(".zip"), "w", zipfile.ZIP_DEFLATED) as archive:
            archive.write(out / "review.json", out.name + "/review.json")
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "cases": len(report["cases"]),
                "new_gpu_execution": False,
                "evidence": str(out.with_suffix(".zip")),
            }
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
