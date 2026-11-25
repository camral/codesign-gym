"""Classify CUDA submissions without mistaking a completion hook for physics.

A driver callback is not accepted merely because it follows cuGraphLaunch in
wall-clock order. Its stream must be ordered *after completion of the whole
graph*: either the same explicit stream, or a recorded-event/wait chain. No
simulation dispatch, copy, graph mutation, or callback-dependent GPU work may
follow it. Missing or ambiguous stream evidence remains a failure.

The callback body is not inspected. 'terminal notification' means an observed
post-graph boundary operation, not a claim about a particular PJRT C++ closure.
JAXPR effects/host-offload checks and the separate detailed trace remain needed.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping, Sequence

from .audit_types import DriverRecord, SubmissionReport

_GRAPH = re.compile(r"^cuGraphLaunch(?:_ptsz)?$")
_HOST = re.compile(r"^cuLaunchHostFunc(?:_v2)?(?:_ptsz)?$")
_RECORD = re.compile(r"^cuEventRecord(?:WithFlags)?(?:_ptsz)?$")
_WAIT = re.compile(r"^cuStreamWaitEvent(?:_ptsz)?$")
_EVENT_DESTROY = re.compile(r"^cuEventDestroy(?:_v2)?$")
_LIFETIME = re.compile(
    r"^cu(?:Stream(?:Create|Destroy)|EventCreate|Ctx(?:Create|Destroy)|"
    r"DevicePrimaryCtx(?:Reset|Release))"
)
# Warm certified calls have already allocated their streams and graph. A
# resource-lifetime change inside this boundary is not presumed harmless: a
# reused handle could inherit a stale dependency association.
_SAFE_QUERY = re.compile(
    r"^cu(?:Ctx(?:GetDevice|GetFlags|GetApiVersion)|Device(?:Get|GetAttribute|GetCount|GetName|GetUuid|TotalMem)|"
    r"PointerGetAttribute(?:s)?|MemGetInfo|StreamGet(?:Flags|Priority|Ctx)|StreamIsCapturing|EventElapsedTime)(?:_v2)?(?:_ptsz)?$"
)
# After a terminal callback only event/context/lifetime bookkeeping is allowed.
# Unknown driver work is not guessed to be harmless. Memory frees may release
# PJRT buffers after completion; allocations/transfers/graph updates are not here.
_EPILOGUE = re.compile(
    r"^cu(?:Ctx(?:SetCurrent|GetCurrent|PushCurrent_v2|PopCurrent_v2)|"
    r"Event(?:Record|RecordWithFlags|Query|Synchronize|Destroy(?:_v2)?)|"
    r"Stream(?:WaitEvent|Query|Synchronize)|LaunchHostFunc(?:_v2)?|"
    r"MemFree(?:_v2|Async)?)(?:_ptsz)?$"
)


def allowed_driver_operation(name: str) -> bool:
    """Known warmed-execution/notification operations, not an unknown-op bypass."""
    return bool(
        _GRAPH.fullmatch(name)
        or _EPILOGUE.fullmatch(name)
        or _SAFE_QUERY.fullmatch(name)
        or _EVENT_DESTROY.fullmatch(name)
    )


def inspect_api_counts(counts: Mapping[str, int], errors: Sequence[str] = ()) -> SubmissionReport:
    """Strict aggregate-only screen: counts alone never excuse host functions."""
    counts = {str(k): int(v) for k, v in counts.items()}
    failures = list(errors)

    def total(pattern: str) -> int:
        return sum(n for name, n in counts.items() if re.match(pattern, name))

    summary = {
        "host_graph_submissions": total(r"^cuGraphLaunch(?:_|$)"),
        "host_kernel_submissions": total(r"^cuLaunch(?:Kernel|Cooperative|Grid)|^cuLaunch$"),
        "host_memory_operations": total(r"^cu(?:Memcpy|Memset)"),
        "host_function_submissions": total(
            r"^cuLaunchHostFunc|^cuStreamAddCallback|"
            r"^cuGraph(?:AddHostNode|HostNodeSetParams|ExecHostNodeSetParams)"
        ),
        "api_entries": sum(counts.values()),
    }
    if summary["host_graph_submissions"] != 1:
        failures.append(
            f"Expected ONE graph submission, observed {summary['host_graph_submissions']}"
        )
    for name in ("host_kernel_submissions", "host_memory_operations", "host_function_submissions"):
        if summary[name]:
            failures.append(f"{name}={summary[name]}")
    if any(n < 0 for n in counts.values()):
        failures.append("Invalid negative API count")
    return {
        "passed": not failures,
        "counts": summary,
        "driver_api_counts": dict(sorted(counts.items())),
        "errors": failures,
        "scope": "one synchronized executable invocation; all process DRIVER_API entries",
        "kernel_activity_collected": False,
        "single_cuda_kernel_claimed": False,
    }


def _handle(record: DriverRecord, field: str) -> int | None:
    """Return an explicit CUDA handle, rejecting null and special streams."""
    value = record.get("parameters", {}).get(field)
    # No implicit/default-stream dependency assumptions. XLA uses explicit
    # streams. Integers 1 and 2 are CUDA special stream handles.
    if not isinstance(value, int) or isinstance(value, bool) or value <= 2:
        return None
    return value


def _context(record: DriverRecord) -> int | None:
    """Return a positive CUDA context identifier, or None when unproven."""
    value = record.get("context")
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def _succeeded(record: DriverRecord) -> bool:
    """Check successful return and a complete, ordered API interval."""
    return (
        record.get("return_status") == 0
        and isinstance(record.get("enter_ns"), int)
        and isinstance(record.get("exit_ns"), int)
        and record["exit_ns"] >= record["enter_ns"]
    )


def inspect_submissions(
    counts: Mapping[str, int],
    records: list[DriverRecord],
    errors: Sequence[str] = (),
    *,
    records_complete: bool = True,
) -> SubmissionReport:
    """Require a complete ordered trace before classifying boundary callbacks.

    Stream/event handles are copied from CUDA API parameter blocks during CUPTI
    callbacks. No GPU data is read. Event re-recording invalidates the previous
    association; contexts and explicit streams are never conflated.
    """
    report = inspect_api_counts(counts, errors)
    report["schema"] = 3
    report["driver_api_records"] = records
    report["recording_complete"] = bool(records_complete)
    report["terminal_notifications"] = []
    report["unclassified_host_functions"] = []
    report["counts"]["terminal_host_notifications"] = 0
    report["counts"]["intra_simulation_or_unclassified_host_functions"] = report["counts"][
        "host_function_submissions"
    ]
    failures = report["errors"]
    valid_records = (
        records_complete
        and Counter(r.get("name", "") for r in records) == Counter(counts)
        and all(r.get("sequence") == i for i, r in enumerate(records))
        and all(
            isinstance(r.get("enter_ns"), int)
            and not isinstance(r.get("enter_ns"), bool)
            and isinstance(r.get("exit_ns"), int)
            and not isinstance(r.get("exit_ns"), bool)
            and r["exit_ns"] >= r["enter_ns"]
            and isinstance(r.get("return_status"), int)
            and not isinstance(r.get("return_status"), bool)
            for r in records
        )
        and all(a["enter_ns"] <= b["enter_ns"] for a, b in zip(records, records[1:]))
    )
    if not valid_records:
        failures.append("Incomplete/mismatched ordered driver evidence")
        report["passed"] = False
        return report

    # Failed CUDA calls cannot establish stream dependencies. Only NOT_READY
    # from an event/stream query is an expected nonzero return code.
    for r in records:
        status = r.get("return_status")
        query = r.get("name", "") in ("cuEventQuery", "cuStreamQuery", "cuStreamQuery_ptsz")
        if status != 0 and not (query and status == 600):
            failures.append(f"CUDA driver failure at entry {r['sequence']}: {r['name']}={status}")

    graph_records = [r for r in records if _GRAPH.fullmatch(r.get("name", ""))]
    if len(graph_records) != 1:
        failures.append("No unique supported graph launch in ordered evidence")
        report["passed"] = False
        return report
    graph = graph_records[0]
    context, stream = _context(graph), _handle(graph, "stream")
    if (
        not _succeeded(graph)
        or context is None
        or stream is None
        or _handle(graph, "graph_exec") is None
    ):
        failures.append("Graph launch lacks successful return/context/explicit-stream evidence")
        report["passed"] = False
        return report

    # A concurrent record/destroy can change which event state a wait captures.
    # Check COMPLETE intervals, including mutations whose ENTER is later than
    # the wait's ENTER. Sequential post-wait re-records do not alter a previously
    # queued wait, per CUDA's event semantics, and remain allowed.
    accesses: dict[tuple[int | None, int | None], list[DriverRecord]] = {}
    for record in records:
        name = record["name"]
        if _RECORD.fullmatch(name) or _WAIT.fullmatch(name) or _EVENT_DESTROY.fullmatch(name):
            event_key = (_context(record), _handle(record, "event"))
            if None in event_key:
                failures.append(
                    f"Event API lacks explicit context/handle at entry {record['sequence']}"
                )
            else:
                accesses.setdefault(event_key, []).append(record)
    ambiguities: list[dict[str, object]] = []
    for event_key, group in accesses.items():
        for i, a in enumerate(group):
            for b in group[i + 1 :]:
                if b["enter_ns"] >= a["exit_ns"]:
                    break  # Group is in monotone ENTER order.
                if _WAIT.fullmatch(a["name"]) and _WAIT.fullmatch(b["name"]):
                    continue  # Two reads do not mutate the captured event state.
                if max(a["enter_ns"], b["enter_ns"]) < min(a["exit_ns"], b["exit_ns"]):
                    ambiguities.append(
                        {
                            "context": event_key[0],
                            "event": event_key[1],
                            "entries": [a["sequence"], b["sequence"]],
                        }
                    )
    report["ambiguous_event_accesses"] = ambiguities
    if ambiguities:
        failures.append("Overlapping event mutation/access has no established capture order")

    for record in records:
        name = record["name"]
        if _LIFETIME.match(name):
            failures.append(
                f"Resource lifetime change inside audited invocation: {name} at entry {record['sequence']}"
            )
        elif not allowed_driver_operation(name):
            # Aggregate counters flag known work; the explicit allowlist also
            # catches stream memory writes/waits, graph mutation and new driver
            # operations that would otherwise disappear into a zero counter.
            failures.append(
                f"Unclassified CUDA driver operation: {name} at entry {record['sequence']}"
            )

    # A proof is a sequence of graph/record/wait operations. Every CPU API call
    # in a proof completed before the next one began; this conservatively rejects
    # races between threads instead of deriving ordering from ENTER timestamps.
    streams: dict[tuple[int | None, int | None], list[int]] = {
        (context, stream): [graph["sequence"]]
    }
    events: dict[tuple[int | None, int | None], list[int] | None] = {}
    terminal_seen = False

    def append(proof: list[int] | None, record: DriverRecord) -> list[int] | None:
        """Extend a dependency proof only after the prior API call has exited."""
        if proof is None or not _succeeded(record):
            return None
        if records[proof[-1]]["exit_ns"] > record["enter_ns"]:
            return None
        return proof + [record["sequence"]]

    for r in records:
        i, name = r["sequence"], r["name"]
        if i <= graph["sequence"]:
            if _HOST.fullmatch(name):
                report["unclassified_host_functions"].append(
                    {"entry": i, "reason": "Host function precedes whole-graph submission"}
                )
            continue
        ctx, st = _context(r), _handle(r, "stream")
        key = (ctx, st)
        if terminal_seen and not _EPILOGUE.fullmatch(name):
            failures.append(f"Non-epilogue API after terminal callback: {name} at entry {i}")
        if _RECORD.fullmatch(name):
            event = _handle(r, "event")
            if event is not None:
                events[(ctx, event)] = append(streams.get(key), r)
        elif _WAIT.fullmatch(name):
            event = _handle(r, "event")
            proof = append(events.get((ctx, event)), r)
            if proof is not None:
                streams[key] = proof
        elif _EVENT_DESTROY.fullmatch(name):
            events.pop((ctx, _handle(r, "event")), None)
        elif _HOST.fullmatch(name):
            proof = append(streams.get(key), r)
            if proof is None or ctx is None or st is None or _handle(r, "function") is None:
                report["unclassified_host_functions"].append(
                    {
                        "entry": i,
                        "reason": "No successful same-stream or event-chain proof after graph",
                    }
                )
            else:
                terminal_seen = True
                report["terminal_notifications"].append(
                    {
                        "entry": i,
                        "stream": st,
                        "context": ctx,
                        "function_address": r["parameters"]["function"],
                        "dependency_entries": proof,
                        "classification": "post_graph_boundary_notification",
                        "callback_body_identified": False,
                        "basis": "CUDA same-stream ordering / recorded-event wait dependency",
                    }
                )

    accepted = len(report["terminal_notifications"])
    total = report["counts"]["host_function_submissions"]
    report["counts"]["terminal_host_notifications"] = accepted
    report["counts"]["intra_simulation_or_unclassified_host_functions"] = total - accepted
    if accepted == total and total:
        failures.remove(f"host_function_submissions={total}")
    if report["unclassified_host_functions"]:
        failures.append("At least one host function has no proven post-graph dependency")
    report["errors"] = list(dict.fromkeys(failures))
    report["passed"] = not failures
    report["scope"] += "; stream-ordered post-graph notifications reported separately"
    return report
