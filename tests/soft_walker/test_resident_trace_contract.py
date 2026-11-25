"""Replayed CUDA evidence plus adversarial audits; these are NOT new GPU runs."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


import ctypes as ct
import importlib.util
import json
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from soft_paths import SOFT_ROOT, needs_scripts

from envs.soft_walker.engine.execution.compiled import _equivalent
from envs.soft_walker.engine.execution.cupti import _copy_parameters, _StreamDestroy
from envs.soft_walker.engine.execution.submissions import inspect_submissions
from envs.soft_walker.engine.execution.trace import MARKER, inspect_trace
from envs.soft_walker.engine.execution.trace_metadata import activity_kind, positive_id

FIXTURES = Path(__file__).parent / "fixtures" / "resident_trace"


def returned() -> tuple[dict[str, object], dict[str, object]]:
    trace = json.loads((FIXTURES / "control_activity.json").read_text())
    audit = json.loads((FIXTURES / "control_submissions.json").read_text())
    return trace, audit


def kernel(trace: dict[str, object], *, missing_graph: bool = False) -> dict[str, object]:
    return next(
        e
        for e in trace["traceEvents"]
        if e.get("pid") == 1
        and e.get("ph") == "X"
        and (not missing_graph or "cuda_graph_id" not in e["args"])
    )


def graph(trace: dict[str, object]) -> dict[str, object]:
    return next(e for e in trace["traceEvents"] if e.get("name", "").startswith("cuGraphLaunch"))


def test_exact_returned_activity_replays_without_name_or_graph_id_assumptions() -> None:
    trace, audit = returned()
    report = inspect_trace(trace, submission_audit=audit)
    assert report["passed"], report["errors"]
    assert report["gpu_compute_events"] == 17
    assert report["copy_named_kernels"] == 2
    assert report["graph_node_correlated_events"] == 12
    assert report["counts"]["host_graph_submissions"] == 1
    coverage = report["callback_coverage"]
    assert coverage["raw_api_events"] == 0
    assert coverage["xla_scheduling_scopes"] == 1
    assert coverage["logical_scheduling_sites"] == 1
    assert coverage["count_status"] == "corroborated_separate_invocations"
    assert not coverage["dependency_proven_in_this_trace"]
    assert not report["host_function_dependency_from_paired_cupti_audit"]


@pytest.mark.parametrize("name", ["memcpy32_post", "MemcpyD2H", "memset_kernel", "normal_kernel"])
def test_launch_metadata_not_kernel_name_determines_activity(name: str) -> None:
    trace, audit = returned()
    kernel(trace)["name"] = name
    assert activity_kind(kernel(trace))[0] == "kernel"
    assert inspect_trace(trace, submission_audit=audit)["passed"]


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_context",
        "wrong_context",
        "missing_correlation",
        "wrong_correlation",
        "missing_node_and_graph",
        "zero_node_and_graph",
        "negative_node",
        "float_node",
        "missing_kernel_details",
        "conflicting_copy_metadata",
        "wrong_copy_direction",
        "unknown_copy",
        "h2d",
        "d2h",
        "d2d_unrelated_graph",
        "memset_unrelated_graph",
        "host_generic_copy",
        "host_memset",
        "host_lifetime",
        "host_stream_write",
        "host_kernel",
        "second_graph",
        "launch_missing_context",
        "launch_missing_correlation",
        "launch_graph_conflict",
        "starts_before_launch",
        "overlaps_marker_start",
        "escapes_marker_end",
        "invalid_duration",
        "missing_marker",
        "duplicate_marker",
        "invalid_marker",
        "unknown_gpu_process",
        "unsupported_gpu_phase",
        "callback_before_graph",
        "callback_missing_all_coverage",
        "extra_callback",
        "forged_audit",
        "incomplete_audit",
        "tampered_record_status",
    ],
)
def test_nearby_true_violations_remain_failures(mutation: str) -> None:
    trace, audit = returned()
    k = kernel(trace, missing_graph=True)
    args = k["args"]
    g = graph(trace)
    marker = next(e for e in trace["traceEvents"] if e.get("name") == MARKER)
    callback = next(e for e in trace["traceEvents"] if e.get("name") == "ThenExecuteCallback")
    if mutation in ("missing_context", "missing_correlation", "missing_kernel_details"):
        args.pop(
            {
                "missing_context": "context_id",
                "missing_correlation": "correlation_id",
                "missing_kernel_details": "kernel_details",
            }[mutation]
        )
    elif mutation == "wrong_context":
        args["context_id"] = "$$9"
    elif mutation == "wrong_correlation":
        args["correlation_id"] = "999"
    elif mutation == "missing_node_and_graph":
        args.pop("cuda_graph_node_id")
    elif mutation == "zero_node_and_graph":
        args["cuda_graph_node_id"] = "0"
    elif mutation == "negative_node":
        args["cuda_graph_node_id"] = "-9"
    elif mutation == "float_node":
        args["cuda_graph_node_id"] = 1.5
    elif mutation == "conflicting_copy_metadata":
        args["memcpy_details"] = "kind_src:device kind_dst:pinned"
    elif mutation == "wrong_copy_direction":
        args.pop("kernel_details")
        k["name"] = "MemcpyD2D"
        args["memcpy_details"] = "kind_src:device kind_dst:pinned"
    elif mutation == "unknown_copy":
        args.pop("kernel_details")
        k["name"] = "memcpy32_post"
    elif mutation in ("h2d", "d2h"):
        args.pop("kernel_details")
        k["name"] = "opaque_DMA"
        args["memcpy_details"] = (
            "kind_src:pinned kind_dst:device"
            if mutation == "h2d"
            else "kind_src:device kind_dst:pageable"
        )
    elif mutation in ("d2d_unrelated_graph", "memset_unrelated_graph"):
        args.pop("kernel_details")
        args["correlation_id"] = "999"
        k["name"] = "MemcpyD2D" if mutation.startswith("d2d") else "Memset"
    elif mutation.startswith("host_"):
        names = {
            "host_generic_copy": "cuMemcpyAsync",
            "host_memset": "cuMemsetD8Async",
            "host_lifetime": "cuStreamDestroy_v2",
            "host_stream_write": "cuStreamWriteValue32",
            "host_kernel": "cuLaunchKernel",
        }
        e = deepcopy(g)
        e["name"] = names[mutation]
        e["ts"] += 100.0
        e["dur"] = 1.0
        trace["traceEvents"].append(e)
    elif mutation == "second_graph":
        trace["traceEvents"].append(deepcopy(g))
    elif mutation == "launch_missing_context":
        g["args"].pop("context_id")
    elif mutation == "launch_missing_correlation":
        g["args"].pop("correlation_id")
    elif mutation == "launch_graph_conflict":
        g["args"]["cuda_graph_id"] = "999"
    elif mutation == "starts_before_launch":
        k["ts"] = g["ts"] - 1.0
    elif mutation == "overlaps_marker_start":
        k.update(ts=marker["ts"] - 1.0, dur=10.0)
    elif mutation == "escapes_marker_end":
        k["dur"] = marker["dur"]
    elif mutation == "invalid_duration":
        k["dur"] = float("nan")
    elif mutation == "missing_marker":
        trace["traceEvents"].remove(marker)
    elif mutation == "duplicate_marker":
        trace["traceEvents"].append(deepcopy(marker))
    elif mutation == "invalid_marker":
        marker["dur"] = float("nan")
    elif mutation == "unknown_gpu_process":
        k["pid"] = 999
    elif mutation == "unsupported_gpu_phase":
        k["ph"] = "B"
    elif mutation == "callback_before_graph":
        callback["ts"] = g["ts"] - 1.0
    elif mutation == "callback_missing_all_coverage":
        trace["traceEvents"].remove(callback)
    elif mutation == "extra_callback":
        trace["traceEvents"].append(deepcopy(callback))
    elif mutation == "forged_audit":
        audit = {"passed": True, "recording_complete": True}
    elif mutation == "incomplete_audit":
        audit["recording_complete"] = False
    elif mutation == "tampered_record_status":
        audit["driver_api_records"][-1]["return_status"] = 1
    else:
        raise AssertionError(mutation)
    report = inspect_trace(trace, submission_audit=audit)
    assert not report["passed"], mutation
    assert report["errors"]


@pytest.mark.parametrize("kind", ["d2d", "memset", "child_graph_id", "no_node_but_matching_graph"])
def test_device_internal_nodes_keep_provenance_requirements(kind: str) -> None:
    trace, audit = returned()
    k = kernel(trace, missing_graph=True)
    if kind in ("d2d", "memset"):
        k["args"].pop("kernel_details")
        k["name"] = "MemcpyD2D" if kind == "d2d" else "Memset"
    elif kind == "child_graph_id":
        k["args"]["cuda_graph_id"] = "991"
    else:
        k["args"].pop("cuda_graph_node_id")
        k["args"]["cuda_graph_id"] = "4"
    assert inspect_trace(trace, submission_audit=audit)["passed"]


def test_callback_scope_and_raw_event_are_not_double_counted() -> None:
    trace, audit = returned()
    scope = next(e for e in trace["traceEvents"] if e.get("name") == "ThenExecuteCallback")
    raw = deepcopy(scope)
    raw.update(name="cuLaunchHostFunc", ts=scope["ts"] + 1.0, dur=1.0)
    trace["traceEvents"].append(raw)
    report = inspect_trace(trace, submission_audit=audit)
    assert report["passed"], report["errors"]
    assert report["callback_coverage"]["logical_scheduling_sites"] == 1
    assert report["callback_coverage"]["raw_events_nested_in_scopes"] == 1


def row(name: str, start: int, end: int, **parameters: object) -> dict[str, object]:
    return dict(
        name=name, enter_ns=start, exit_ns=end, return_status=0, context=1000, parameters=parameters
    )


def inspect(rows: list[dict[str, object]]) -> dict[str, object]:
    for i, r in enumerate(rows):
        r["sequence"] = i
    return inspect_submissions(Counter(r["name"] for r in rows), rows)


def test_returned_review_concurrent_rerecord_is_rejected() -> None:
    rows = [
        row("cuGraphLaunch", 0, 2, graph_exec=6000, stream=2000),
        row("cuEventRecord", 10, 12, event=4000, stream=2000),
        row("cuStreamWaitEvent", 20, 35, stream=3000, event=4000, flags=0),
        row("cuEventRecord", 25, 27, event=4000, stream=9000),
        row("cuLaunchHostFunc", 40, 42, stream=3000, function=5000, user_data=7000),
    ]
    report = inspect(rows)
    assert not report["passed"]
    assert report["ambiguous_event_accesses"][0]["entries"] == [2, 3]


@pytest.mark.parametrize(
    "names",
    [
        ("cuStreamDestroy_v2", "cuStreamCreate"),
        ("cuCtxDestroy_v2", "cuCtxCreate_v2"),
        ("cuEventCreate",),
        ("cuDevicePrimaryCtxReset_v2",),
    ],
)
def test_lifetime_changes_cannot_reuse_stale_stream_proof(
    names: tuple[str, str] | tuple[str],
) -> None:
    rows = [row("cuGraphLaunch", 0, 2, graph_exec=6000, stream=2000)]
    for i, name in enumerate(names):
        rows.append(row(name, 10 + 10 * i, 12 + 10 * i))
    rows.append(row("cuLaunchHostFunc", 50, 52, stream=2000, function=5000, user_data=7000))
    assert not inspect(rows)["passed"]


@pytest.mark.parametrize("mutation", ["record", "destroy"])
def test_event_changes_AFTER_a_completed_wait_do_not_rewrite_queued_dependency(
    mutation: str,
) -> None:
    rows = [
        row("cuGraphLaunch", 0, 2, graph_exec=6000, stream=2000),
        row("cuEventRecord", 10, 12, event=4000, stream=2000),
        row("cuStreamWaitEvent", 20, 22, stream=3000, event=4000, flags=0),
        row(
            "cuEventRecord" if mutation == "record" else "cuEventDestroy_v2",
            30,
            32,
            event=4000,
            stream=9000,
        ),
        row("cuLaunchHostFunc", 40, 42, stream=3000, function=5000, user_data=7000),
    ]
    report = inspect(rows)
    assert report["passed"], report["errors"]
    assert report["terminal_notifications"][0]["dependency_entries"] == [0, 1, 2, 4]


@pytest.mark.parametrize(
    "name", ["cuStreamWriteValue32", "cuStreamWaitValue32", "cuGraphExecUpdate", "cuMysteryWork"]
)
def test_unknown_or_host_controlled_work_before_callback_is_not_ignored(name: str) -> None:
    rows = [
        row("cuGraphLaunch", 0, 2, graph_exec=6000, stream=2000),
        row(name, 10, 12),
        row("cuLaunchHostFunc", 20, 22, stream=2000, function=5000, user_data=7000),
    ]
    assert not inspect(rows)["passed"]


@pytest.mark.parametrize(
    "field,value",
    [("exit_ns", None), ("return_status", None), ("return_status", False), ("enter_ns", True)],
)
def test_complete_records_required_even_without_host_function(
    field: str, value: None | bool
) -> None:
    rows = [
        row("cuGraphLaunch", 0, 2, graph_exec=6000, stream=2000),
        row("cuCtxSetCurrent", 10, 12),
    ]
    rows[-1][field] = value
    assert not inspect(rows)["passed"]


def test_ctypes_stream_destroy_reads_only_documented_handle() -> None:
    data = _StreamDestroy(2000)
    assert _copy_parameters("cuStreamDestroy_v2", ct.addressof(data)) == {"stream": 2000}


@pytest.mark.parametrize(
    "value,expected",
    [
        ("$$1", 1),
        ("0x10", 16),
        ("17179869198", 17179869198),
        ("0", None),
        (False, None),
        (1.1, None),
        ("nan", None),
    ],
)
def test_profiler_identifier_normalization_does_not_round_large_ids(
    value: bool | float | str, expected: None | int
) -> None:
    assert positive_id(value) == expected


def test_equivalence_does_not_broadcast_or_silently_change_dtype() -> None:
    assert not _equivalent(np.ones(1), np.ones(3))["passed"]
    assert not _equivalent(np.ones(3, np.float32), np.ones(3, np.float64))["passed"]
    assert _equivalent(np.ones(3), np.ones(3))["passed"]


@needs_scripts
def test_preflight_runner_collects_independent_failures_but_blocks_physics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = SOFT_ROOT / "diagnostics/run_resident_tests.py"
    spec = importlib.util.spec_from_file_location("resident_runner_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = []

    def fake(command: Sequence[str], **kwargs: object) -> dict[str, object]:
        calls.append(command)
        kwargs["log_path"].write_text("synthetic subprocess result; no CUDA execution\n")
        return {
            "returncode": 1 if len(calls) == 1 else 0,
            "completed": True,
            "timed_out": False,
            "seconds": 0.0,
        }

    monkeypatch.setattr(module, "run_child", fake)
    out = tmp_path / "independent"
    monkeypatch.setattr(
        sys, "argv", [str(path), "--device", "cuda", "--skip-unit", "--out", str(out)]
    )
    assert module.main() == 1
    manifest = json.loads((out / "manifest.json").read_text())
    assert len(calls) == 3
    assert [s["passed"] for s in manifest["stages"]] == [False, True, True, False]
    assert manifest["stages"][-1]["skipped"]
    assert out.with_suffix(".zip").is_file()


@needs_scripts
def test_timeout_stops_further_device_work_and_still_archives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = SOFT_ROOT / "diagnostics/run_resident_tests.py"
    spec = importlib.util.spec_from_file_location("resident_timeout_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = []

    def fake(command: Sequence[str], **kwargs: object) -> dict[str, object]:
        calls.append(command)
        kwargs["log_path"].write_text("synthetic timeout\n")
        return {"returncode": -15, "completed": False, "timed_out": True, "seconds": 0.1}

    monkeypatch.setattr(module, "run_child", fake)
    out = tmp_path / "timeout"
    monkeypatch.setattr(
        sys, "argv", [str(path), "--device", "cuda", "--preflight-only", "--out", str(out)]
    )
    assert module.main() == 1
    assert len(calls) == 1
    assert out.with_suffix(".zip").is_file()


@needs_scripts
@pytest.mark.parametrize("corrupt", [False, True])
def test_offline_review_archives_results_and_detects_checksum_corruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, corrupt: bool
) -> None:
    import gzip
    import hashlib
    import zipfile

    path = SOFT_ROOT / "diagnostics/review_resident_evidence.py"
    spec = importlib.util.spec_from_file_location("offline_review_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    trace, audit = returned()
    files = {
        "run/probe/executable/submission_audit.json": json.dumps(audit).encode(),
        "run/probe/executable/trace/test.trace.json.gz": gzip.compress(json.dumps(trace).encode()),
    }
    hashes = {n.removeprefix("run/"): hashlib.sha256(b).hexdigest() for n, b in files.items()}
    if corrupt:
        hashes[next(iter(hashes))] = "0" * 64
    files["run/SHA256SUMS.json"] = json.dumps(hashes).encode()
    archive = tmp_path / "old.zip"
    with zipfile.ZipFile(archive, "w") as z:
        for name, data in files.items():
            z.writestr(name, data)
    original_hash = hashlib.sha256(archive.read_bytes()).hexdigest()
    out = tmp_path / "review"
    monkeypatch.setattr(sys, "argv", [str(path), "--run", str(archive), "--out", str(out)])
    assert module.main() == int(corrupt)
    report = json.loads((out / "review.json").read_text())
    assert report["passed"] is (not corrupt)
    assert not report["new_gpu_execution"]
    assert report["checked_files"] == 2
    assert out.with_suffix(".zip").is_file()
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == original_hash


def test_overlapping_mutations_completed_before_wait_remain_ambiguous() -> None:
    rows = [
        row("cuGraphLaunch", 0, 2, graph_exec=6000, stream=2000),
        row("cuEventRecord", 10, 19, event=4000, stream=2000),
        row("cuEventRecord", 12, 15, event=4000, stream=9000),
        row("cuStreamWaitEvent", 20, 22, stream=3000, event=4000, flags=0),
        row("cuLaunchHostFunc", 30, 32, stream=3000, function=5000, user_data=7000),
    ]
    assert not inspect(rows)["passed"]
