"""Typed interpretation of XProf CUDA activity, independent of simulation.

A CUDA kernel can be named ``memcpy32_post``. Its launch metadata, not a
substring in its name, identifies it as a kernel. A graph child can omit the
exported graph ID while retaining its graph-node and launch-correlation IDs.
Neither case justifies ignoring an actual DMA or accepting an unrelated launch.

This module reads Chrome complete-event JSON exported by XProf. Unknown or
contradictory activity is reported, never guessed to be harmless. CUPTI's kernel
record defines correlationId and graphNodeId; the corresponding XProf keys are
correlation_id and cuda_graph_node_id. See docs/resident_trace_audit.md.
"""

from __future__ import annotations

import re
from typing import cast

from .audit_types import GraphIdentity, GraphProvenance, TraceEvent

_COPY_NAME = re.compile(
    r"^(?:Memcpy|MemCopy)(?:D2H|DtoH|DeviceToHost|H2D|HtoD|HostToDevice|"
    r"D2D|DtoD|DeviceToDevice|P2P|H2H|HtoH)(?:\b|\s|\()",
    re.I,
)


def positive_id(value: object) -> int | None:
    """Normalize integral profiler IDs, including XProf's ``$$1`` contexts.

    Never use floating point to normalize a 64-bit graph-node identifier.
    Missing/zero/invalid IDs cannot establish provenance.
    """
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if not isinstance(value, str):
        return None
    text = value.strip().removeprefix("$$")
    try:
        number = int(text, 16 if text.lower().startswith("0x") else 10)
    except ValueError:
        return None
    return number if number > 0 else None


def activity_kind(event: TraceEvent) -> tuple[str, str | None, str]:
    """Return (kernel | dma | memset | unknown | conflicting, direction, basis)."""
    args = event.get("args", {})
    if not isinstance(args, dict):
        return "unknown", None, "invalid_args"
    name = str(event.get("name", ""))
    kernel = args.get("kernel_details")
    kernel_metadata = isinstance(kernel, str) and bool(kernel.strip())
    copy_metadata = any(key in args for key in ("memcpy_details", "copy_kind", "memcpy_kind"))
    fill_metadata = "memset_details" in args
    if sum((kernel_metadata, copy_metadata, fill_metadata)) > 1:
        return "conflicting", None, "contradictory_activity_metadata"
    if kernel_metadata:
        # Require actual launch dimensions, not a truthy arbitrary marker.
        if re.search(r"\bgrid:\s*\d", cast(str, kernel)) and re.search(
            r"\bblock:\s*\d", cast(str, kernel)
        ):
            return "kernel", None, "kernel_details"
        return "unknown", None, "incomplete_kernel_details"
    if fill_metadata or re.fullmatch(r"Memset(?:\s.*|\(.*\))?", name, re.I):
        return "memset", None, "memset_details_or_activity_label"

    # Copy labels are a supported compatibility format only in the absence of
    # kernel metadata. Direction markers in compiler-generated kernel *names*
    # are not interpreted as host transfers.
    is_dma = copy_metadata or bool(_COPY_NAME.match(name))
    if not is_dma:
        if "memcpy" in name.lower():
            return "unknown", "unknown", "untyped_copy_like_name"
        return "unknown", None, "missing_activity_metadata"

    directions = set()
    if re.match(r"^Memcpy(?:D2H|DtoH|DeviceToHost)(?:\b|\s|\()", name, re.I):
        directions.add("d2h")
    if re.match(r"^Memcpy(?:H2D|HtoD|HostToDevice)(?:\b|\s|\()", name, re.I):
        directions.add("h2d")
    if re.match(r"^Memcpy(?:D2D|DtoD|DeviceToDevice)(?:\b|\s|\()", name, re.I):
        directions.add("d2d")
    detail = str(args.get("memcpy_details", "")).lower()
    source = re.search(r"\bkind_src:\s*([\w]+)", detail)
    destination = re.search(r"\bkind_dst:\s*([\w]+)", detail)
    if source and destination:
        src, dst = source[1], destination[1]
        host = {"host", "pageable", "pinned"}
        if src == "device" and dst in host:
            directions.add("d2h")
        elif src in host and dst == "device":
            directions.add("h2d")
        elif src == dst == "device":
            directions.add("d2d")
        else:
            directions.add("unknown")
    for key in ("copy_kind", "memcpy_kind"):
        if key in args:
            token = str(args[key]).lower().replace("_", "").replace(" ", "")
            aliases = {
                "dtoh": "d2h",
                "d2h": "d2h",
                "devicetohost": "d2h",
                "htod": "h2d",
                "h2d": "h2d",
                "hosttodevice": "h2d",
                "dtod": "d2d",
                "d2d": "d2d",
                "devicetodevice": "d2d",
            }
            directions.add(aliases.get(token, "unknown"))
    if len(directions) > 1:
        return "conflicting", None, "contradictory_copy_direction"
    return "dma", next(iter(directions), "unknown"), "dma_metadata_or_activity_label"


def graph_identity(event: TraceEvent) -> GraphIdentity:
    """Return typed graph launch metadata and any contradictory graph label."""
    args = event.get("args", {})
    args = args if isinstance(args, dict) else {}
    graph = positive_id(args.get("cuda_graph_id"))
    label = re.search(r"\bCudaGraph:\s*(\d+)", str(event.get("name", "")))
    label_graph = positive_id(label[1]) if label else None
    return {
        "context": positive_id(args.get("context_id")),
        "correlation": positive_id(args.get("correlation_id")),
        "graph": graph or label_graph,
        "conflict": graph is not None and label_graph is not None and graph != label_graph,
    }


def graph_provenance(event: TraceEvent, launch: TraceEvent) -> GraphProvenance:
    """Link activity to this invocation, not merely to some graph execution.

    CUPTI graphNodeId is nonzero for graph-launched kernels. Matching the
    correlation and context to the unique host launch ties that evidence to
    this invocation. A missing graph ID is not equated with a missing node ID.
    A different positive graph ID is accepted only with the same launch and a
    positive node ID: graph hierarchy is not reconstructed from encoded bits.
    """
    args = event.get("args", {})
    args = args if isinstance(args, dict) else {}
    identity = graph_identity(launch)
    context = positive_id(args.get("context_id"))
    correlation = positive_id(args.get("correlation_id"))
    node = positive_id(args.get("cuda_graph_node_id"))
    graph = positive_id(args.get("cuda_graph_id"))
    reasons = []
    if identity["conflict"]:
        reasons.append("conflicting_launch_graph_id")
    if not context or not identity["context"] or context != identity["context"]:
        reasons.append("missing_or_mismatched_launch_context")
    if not correlation or not identity["correlation"] or correlation != identity["correlation"]:
        reasons.append("missing_or_mismatched_launch_correlation")
    direct = graph is not None and graph == identity["graph"]
    if not node and not direct:
        reasons.append("no_graph_node_or_matching_graph_id")
    if event.get("ts", float("-inf")) < launch.get("ts", float("inf")):
        reasons.append("activity_precedes_its_launch")
    return {
        "passed": not reasons,
        "basis": "matching_graph_id_and_launch" if direct else "graph_node_and_launch_correlation",
        "context": context,
        "correlation": correlation,
        "graph_id": graph,
        "graph_node_id": node,
        "errors": reasons,
    }
