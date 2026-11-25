"""Corroborate a whole CUDA invocation using typed XProf device activity.

The lightweight CUPTI submission audit and the detailed XProf trace observe
SEPARATE equivalent invocations: CUPTI cannot be subscribed twice here. This
module never borrows a same-invocation stream proof from the other execution.
It checks launch provenance, copies, host fallback and callback-site coverage;
compile_program separately requires the complete submission audit and matching
numerical results. Missing profiler events are not interpreted as zero calls.
"""

from __future__ import annotations

import gzip
import json
import math
import re
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from .audit_types import CallbackCoverage, SubmissionReport, TraceDocument, TraceEvent, TraceReport
from .policy import ResidencyError
from .trace_metadata import activity_kind, graph_identity, graph_provenance

MARKER = "codesign/whole_episode_certificate"


def load_trace(path: str | Path) -> TraceDocument:
    """Read a plain or gzip-compressed XProf JSON document.

    I/O and JSON parse failures propagate; inspect_trace validates the event
    structure and does not infer residency from successful parsing.
    """
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def _span(event: TraceEvent) -> tuple[float, float] | None:
    """Return finite start/end timestamps in microseconds, or None if invalid."""
    start, duration = event.get("ts"), event.get("dur")
    if any(
        isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
        for x in (start, duration)
    ):
        return None
    start, duration = cast(float, start), cast(float, duration)
    return (start, start + duration) if duration >= 0 else None


def _names(rows: Sequence[TraceEvent], pattern: str) -> list[TraceEvent]:
    """Select events whose names match a case-insensitive regular expression."""
    return [e for e in rows if re.search(pattern, str(e.get("name", "")), re.I)]


def _callback_coverage(
    host: Sequence[TraceEvent],
    graphs: Sequence[TraceEvent],
    submission_audit: SubmissionReport | None,
) -> CallbackCoverage:
    """Count observed scheduling sites without pretending to see callback bodies.

    This XProf export filters cuLaunchHostFunc DRIVER_API activity. Its XLA
    ThenExecuteCallback scheduling scope is still present. A scope may wrap a
    raw API event in other exporter versions; nesting is deduplicated, not
    counted twice. Absent raw events AND absent scopes do not match a nonzero
    audited count. No count agreement establishes callback stream ordering in
    the separately traced call; only the independent audit establishes that.
    """
    raw = _names(host, r"^(?:cu|cuda)LaunchHostFunc(?:_\w+)?(?:\W|$)")
    scopes = _names(host, r"^ThenExecuteCallback$")
    errors = []
    covered = set()
    for scope in scopes:
        a, b = cast(tuple[float, float], _span(scope))
        nested = [
            i
            for i, event in enumerate(raw)
            if event.get("pid") == scope.get("pid")
            and event.get("tid") == scope.get("tid")
            and a <= cast(tuple[float, float], _span(event))[0]
            and cast(tuple[float, float], _span(event))[1] <= b
        ]
        if len(nested) > 1 or any(i in covered for i in nested):
            errors.append("Ambiguous nested callback scheduling scopes")
        covered.update(nested)
    sites = len(scopes) + len(raw) - len(covered)
    evidence: CallbackCoverage = {
        "raw_api_events": len(raw),
        "xla_scheduling_scopes": len(scopes),
        "logical_scheduling_sites": sites,
        "raw_events_nested_in_scopes": len(covered),
        "expected_from_independent_audit": None,
        "count_status": "no_independent_audit",
        "dependency_proven_in_this_trace": False,
        "callback_body_identified": False,
        "invocation_relation": "separate_equivalent_calls_not_same_invocation",
    }
    independent_ok = False
    if submission_audit is not None:
        from .submissions import inspect_submissions

        # Recompute from the raw records: a stale/forged 'passed' field cannot
        # excuse an event race, missing EXIT, or incomplete callback metadata.
        checked = inspect_submissions(
            submission_audit.get("driver_api_counts", {}),
            submission_audit.get("driver_api_records", []),
            records_complete=submission_audit.get("recording_complete", False),
        )
        independent_ok = (
            submission_audit.get("passed") is True
            and not submission_audit.get("errors")
            and checked["passed"]
        )
        if not independent_ok:
            errors.append("Independent submission audit is absent, failed, or fails revalidation")
        expected = checked["counts"].get("terminal_host_notifications", 0)
        evidence["expected_from_independent_audit"] = expected
        if sites != expected:
            evidence["count_status"] = "coverage_mismatch"
            errors.append(f"Callback scheduling coverage mismatch: trace={sites}, audit={expected}")
        else:
            evidence["count_status"] = "corroborated_separate_invocations"
    elif sites:
        errors.append("Callback scheduling sites lack an independent complete submission audit")

    for event in raw + scopes:
        if (
            len(graphs) != 1
            or cast(tuple[float, float], _span(event))[0]
            < cast(tuple[float, float], _span(graphs[0]))[1]
        ):
            errors.append("Callback scheduling precedes the completed graph API submission")
    evidence["independent_submission_audit_revalidated"] = independent_ok
    evidence["errors"] = errors
    return evidence


def inspect_trace(
    trace: TraceDocument,
    *,
    marker: str = MARKER,
    require_marker: bool = True,
    submission_audit: SubmissionReport | None = None,
) -> TraceReport:
    """Validate host submissions and device activity for one marked invocation.

    trace is an exported XProf document with microsecond timestamps. The
    optional submission_audit describes a separate equivalent invocation; it
    cannot prove stream ordering in this trace. Return a report with passed,
    errors, counts, and representative evidence without changing the input.
    """
    events = trace.get("traceEvents", [])
    if not isinstance(events, list):
        return {
            "schema": 3,
            "passed": False,
            "errors": ["traceEvents must be a list"],
            "counts": {},
        }
    events = [e for e in events if isinstance(e, dict)]
    processes = {
        e["pid"]: str(e.get("args", {}).get("name", ""))
        for e in events
        if e.get("ph") == "M" and e.get("name") == "process_name"
    }
    gpu_pids = {pid for pid, name in processes.items() if "GPU" in name.upper()}
    host_pids = {
        pid for pid, name in processes.items() if "HOST" in name.upper() or "CPU" in name.upper()
    }
    markers = [e for e in events if e.get("ph") == "X" and e.get("name") == marker]
    errors = []
    if require_marker and len(markers) != 1:
        errors.append(f"Expected exactly one {marker!r} span, found {len(markers)}")
    interval = _span(markers[0]) if len(markers) == 1 else None
    if len(markers) == 1 and (interval is None or interval[0] >= interval[1]):
        errors.append("Invalid/empty invocation span")
        interval = None
    if len(markers) == 1 and markers[0].get("pid") not in host_pids:
        errors.append("Invocation marker is not on an identified host process")

    complete = [e for e in events if e.get("ph") == "X"]
    invalid = [e for e in complete if _span(e) is None]
    if invalid:
        errors.append(f"Malformed complete-event timestamps: {len(invalid)}")
    unsupported_gpu_phases = [
        e
        for e in events
        if e.get("pid") in gpu_pids
        and e.get("ph") not in ("M", "X", "s", "t", "f")
        and (
            interval is None
            or (isinstance(e.get("ts"), (int, float)) and interval[0] <= e["ts"] < interval[1])
        )
    ]
    if unsupported_gpu_phases:
        errors.append("Unsupported GPU activity phases; complete-event export required")
    selected = []
    overlapping_gpu = []
    for event in complete:
        span = _span(event)
        if span is None:
            continue
        a, b = span
        if interval is None or interval[0] <= a < interval[1]:
            selected.append(event)
        # A pre-existing kernel or transfer overlapping the marker is not
        # silently discarded merely because its START precedes the marker.
        if event.get("pid") in gpu_pids and (
            interval is None or (a < interval[1] and b > interval[0])
        ):
            overlapping_gpu.append(event)
    host = [e for e in selected if e.get("pid") in host_pids]
    gpu = overlapping_gpu
    unlabelled = [
        e
        for e in selected
        if e.get("pid") not in gpu_pids | host_pids
        and (
            "kernel_details" in e.get("args", {})
            or re.match(r"^(?:cu|cuda)(?:Graph|Launch|Memcpy|Memset)", str(e.get("name", "")))
        )
    ]
    if unlabelled:
        errors.append("CUDA activity on a process without host/device metadata")

    graphs = _names(host, r"^(?:cu|cuda)GraphLaunch(?:_\w+)?(?:\W|$)")
    kernels = _names(host, r"^(?:cu|cuda)Launch(?:Kernel|CooperativeKernel|KernelEx|Grid)")
    while_host = _names(host, r"^(?:While|WhileThunk)(?:::.*)?$")
    branch_host = _names(host, r"^(?:Conditional|ConditionalThunk)(?:::.*)?$")
    host_copies = _names(host, r"^(?:(?:cu|cuda))?(?:Memcpy|Memset)")
    callbacks = _names(
        host,
        r"(?:python.*callback|host_callback|xla_python_.*callback|outside_compilation|io_callback|GraphHostNode|cuGraphAddHostNode|StreamAddCallback)",
    )
    graph_build = _names(
        host, r"^(?:cu|cuda)Graph(?:Create|Instantiate|ExecUpdate|Add|Exec.*SetParams)"
    )
    from .submissions import allowed_driver_operation

    unknown_driver_ops = []
    for event in host:
        match = re.match(r"^(cu[A-Z]\w*)", str(event.get("name", "")))
        if match and not allowed_driver_operation(match[1]):
            unknown_driver_ops.append(event)
    if unknown_driver_ops:
        errors.append(
            "Unclassified driver work in detailed trace: "
            + ", ".join(str(e.get("name")) for e in unknown_driver_ops[:8])
        )
    coverage = _callback_coverage(host, graphs, submission_audit)
    errors.extend(coverage["errors"])

    typed = [(e, *activity_kind(e)) for e in gpu]
    compute = [e for e, kind, _, _ in typed if kind == "kernel"]
    d2h = [e for e, kind, direction, _ in typed if kind == "dma" and direction == "d2h"]
    h2d = [e for e, kind, direction, _ in typed if kind == "dma" and direction == "h2d"]
    d2d = [e for e, kind, direction, _ in typed if kind == "dma" and direction == "d2d"]
    fills = [e for e, kind, _, _ in typed if kind == "memset"]
    unknown_copies = [
        e
        for e, kind, direction, _ in typed
        if direction == "unknown" or (kind == "conflicting" and "memcpy" in str(e).lower())
    ]
    unknown_activity = [e for e, kind, _, _ in typed if kind in ("unknown", "conflicting")]

    provenance = (
        [(e, graph_provenance(e, graphs[0])) for e in compute + d2d + fills]
        if len(graphs) == 1
        else []
    )
    compute_ids = {id(e) for e in compute}
    nongraph_compute = [
        e for e, proof in provenance if id(e) in compute_ids and not proof["passed"]
    ]
    nongraph_memory = [
        e for e, proof in provenance if id(e) not in compute_ids and not proof["passed"]
    ]
    if len(graphs) != 1:
        nongraph_compute, nongraph_memory = compute, d2d + fills
    if len(graphs) == 1:
        identity = graph_identity(graphs[0])
        if identity["conflict"] or not identity["context"] or not identity["correlation"]:
            errors.append("Unique graph launch lacks consistent context/correlation metadata")

    escaped_before = [
        e for e in gpu if interval and cast(tuple[float, float], _span(e))[0] < interval[0]
    ]
    escaped_after = [
        e for e in gpu if interval and cast(tuple[float, float], _span(e))[1] > interval[1] + 1.0
    ]
    # Host wrapper scopes may enclose the annotation. Only actual CUDA API
    # calls overlapping its boundary are a capture-scope error.
    host_api_overlap = [
        e
        for e in complete
        if e.get("pid") in host_pids
        and _span(e)
        and re.match(r"^(?:cu|cuda)(?:GraphLaunch|Launch|Memcpy|Memset)", str(e.get("name", "")))
        and interval
        and cast(tuple[float, float], _span(e))[0]
        < interval[0]
        < cast(tuple[float, float], _span(e))[1]
    ]
    if host_api_overlap:
        errors.append("CUDA API overlaps invocation start boundary")
    if not gpu_pids or not host_pids or not gpu or not host:
        errors.append("Missing host/GPU instrumentation; cannot certify residency")
    if len(gpu_pids) != 1:
        errors.append("Certificate requires a single CUDA device")
    if not compute:
        errors.append("No typed GPU kernel activity recorded")

    checks = {
        "host_graph_submissions": (len(graphs), 1),
        "host_kernel_launches": (len(kernels), 0),
        "host_while_checks": (len(while_host), 0),
        "host_conditional_checks": (len(branch_host), 0),
        "gpu_d2h_copies": (len(d2h), 0),
        "gpu_h2d_copies": (len(h2d), 0),
        "host_transfer_calls": (len(host_copies), 0),
        "host_callbacks": (len(callbacks), 0),
        "unclassified_gpu_copies": (len(unknown_copies), 0),
        "unclassified_gpu_activity": (len(unknown_activity), 0),
        "gpu_kernels_without_graph_provenance": (len(nongraph_compute), 0),
        "gpu_memory_ops_without_graph_provenance": (len(nongraph_memory), 0),
        "graph_creation_during_invocation": (len(graph_build), 0),
        "gpu_events_before_invocation_boundary": (len(escaped_before), 0),
        "gpu_events_after_completion_boundary": (len(escaped_after), 0),
    }
    for label, (actual, expected) in checks.items():
        if actual != expected:
            errors.append(f"{label}={actual}, required {expected}")
    return {
        "schema": 3,
        "passed": not errors,
        "errors": list(dict.fromkeys(errors)),
        "interval_us": interval,
        "processes": processes,
        "counts": {label: value[0] for label, value in checks.items()},
        "gpu_compute_events": len(compute),
        "gpu_d2d_events": len(d2d),
        "gpu_memset_events": len(fills),
        "copy_named_kernels": sum("memcpy" in str(e.get("name", "")).lower() for e in compute),
        "graph_node_correlated_events": sum(
            p["passed"] and p["basis"] == "graph_node_and_launch_correlation" for _, p in provenance
        ),
        "post_graph_host_function_calls": coverage["raw_api_events"],
        "callback_coverage": coverage,
        "host_function_dependency_from_paired_cupti_audit": False,
        "standalone_host_function_identity_proven": False,
        "command_buffer_execute_events": len(_names(host, r"^command_buffer::execute$")),
        "examples": {
            "host_loops": (while_host + branch_host)[:8],
            "copies": (d2h + h2d + host_copies)[:8],
            "graph_submissions": graphs[:8],
            "unknown_activity": unknown_activity[:8],
            "graph_provenance": [
                {"name": e.get("name"), "ts": e.get("ts"), **p} for e, p in provenance[:32]
            ],
        },
        "scope": "Device-activity corroboration of a separate equivalent invocation; not a complete raw-host-API or single-kernel certificate",
    }


def require_resident_trace(
    path: str | Path, *, submission_audit: SubmissionReport | None = None
) -> TraceReport:
    """Read and validate a trace, raising ResidencyError if evidence fails.

    The returned report preserves the exported host/device observations.
    File and JSON errors propagate from load_trace.
    """
    report = inspect_trace(load_trace(path), submission_audit=submission_audit)
    if not report["passed"]:
        raise ResidencyError("GPU residency failed: " + "; ".join(report["errors"]))
    return report
