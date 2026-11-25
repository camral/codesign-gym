"""Compare matching benchmark records across versions, preserving caveats.

Unlike compare_evidence.py, different source hashes are EXPECTED here. This is
not a numerical-validation replacement. No speedup is asserted across hosts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
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

# Serialized reports can contain backend-specific auxiliary diagnostic trees.
Report = dict[str, Any]


def main() -> int:
    """Command entry point. Compare matching benchmark records across versions, preserving caveats."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("before")
    p.add_argument("after")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    old, new = Path(a.before), Path(a.after)
    rows = []
    warnings: list[str] = []
    manifests = [json.loads((r / "manifest.json").read_text()) for r in (old, new)]
    for dtype in sorted(set(manifests[0]["dtypes"]) & set(manifests[1]["dtypes"])):
        for stage in ("benchmark", "m2_benchmark"):
            dirs = [r / dtype / stage for r in (old, new)]
            if not all((d / "environment.json").exists() for d in dirs):
                continue
            env = [json.loads((d / "environment.json").read_text()) for d in dirs]
            keys = (
                "cpu_model",
                "devices",
                "requested_device",
                "requested_dtype",
                "cgroup_limits",
                "cpu_affinity",
            )
            same = all(env[0].get(k) == env[1].get(k) for k in keys)
            if not same:
                warnings.append(
                    dtype + "/" + stage + ":different_hardware_or_resource_configuration"
                )
            for f in sorted(dirs[0].glob("*.json")):
                peer = dirs[1] / f.name
                if not peer.exists():
                    continue
                x, y = json.loads(f.read_text()), json.loads(peer.read_text())
                if (
                    not isinstance(x, dict)
                    or not isinstance(y, dict)
                    or "median_s" not in x
                    or "median_s" not in y
                ):
                    continue
                config_keys = set(x.get("config", {})) & set(y.get("config", {}))
                differences = {
                    k: [x["config"][k], y["config"][k]]
                    for k in config_keys
                    if x["config"][k] != y["config"][k]
                }
                compatible = not differences
                row: Report = {
                    "dtype": dtype,
                    "stage": stage,
                    "case": f.stem,
                    "hardware_signature_match": same,
                    "shared_configuration_match": compatible,
                    "config_differences": differences,
                    "before_median_s": x["median_s"],
                    "after_median_s": y["median_s"],
                    "before_over_after": x["median_s"] / y["median_s"],
                    "before_p10_s": x.get("p10_s"),
                    "before_p90_s": x.get("p90_s"),
                    "after_p10_s": y.get("p10_s"),
                    "after_p90_s": y.get("p90_s"),
                }
                if x.get("api_episode_median_s") and y.get("api_episode_median_s"):
                    row.update(
                        before_api_s=x["api_episode_median_s"],
                        after_api_s=y["api_episode_median_s"],
                        api_before_over_after=x["api_episode_median_s"] / y["api_episode_median_s"],
                    )
                rows.append(row)
    report: Report = {
        "rows": rows,
        "warnings": warnings,
        "source_equal": manifests[0]["source_sha256"] == manifests[1]["source_sha256"],
        "note": "Ratios are descriptive, not causal. Check numerical results, physical configs, hardware, warm samples and load. Cross-host ratios are not speedup claims.",
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(out.read_text())
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
