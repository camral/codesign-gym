"""Host-only schemas for observed CUDA submissions and exported profiler events.

These annotations describe known fields; they do not certify imported evidence.
The audit functions still validate timestamps, identifiers, completeness, and
dependencies before returning ``passed=True``. Extra profiler metadata remains
opaque so new exporter fields cannot silently become residency evidence.
"""

from __future__ import annotations

from typing import NotRequired, TypedDict


class DriverRecord(TypedDict, total=False):
    """Copied API metadata; raw C pointers are represented only as integer handles."""

    sequence: int
    name: str
    thread: int
    callback_id: int
    enter_ns: int
    exit_ns: int
    context: int
    return_status: int
    parameters: dict[str, int]


class SubmissionReport(TypedDict, total=False):
    """Aggregate counts and ordered dependency proof for one observed invocation."""

    passed: bool
    counts: dict[str, int]
    driver_api_counts: dict[str, int]
    driver_api_records: list[DriverRecord]
    errors: list[str]
    scope: str
    kernel_activity_collected: bool
    single_cuda_kernel_claimed: bool
    schema: int
    recording_complete: bool
    terminal_notifications: list[dict[str, object]]
    unclassified_host_functions: list[dict[str, object]]
    ambiguous_event_accesses: list[dict[str, object]]
    library: str
    cupti_version: int
    first_api_names: list[str]
    record_limit: int
    abi_prefix_fields: list[str]
    abi_pointer_bytes: int


class TraceEvent(TypedDict, total=False):
    """Known Chrome complete-event fields; timestamp units are microseconds."""

    ph: str
    name: str
    pid: int | str
    tid: int | str
    ts: float
    dur: float
    args: dict[str, object]


class TraceDocument(TypedDict):
    """The complete-event list exported by XProf."""

    traceEvents: list[TraceEvent]


class GraphIdentity(TypedDict):
    """Integral launch identifiers; absent or invalid identifiers stay None."""

    context: int | None
    correlation: int | None
    graph: int | None
    conflict: bool


class GraphProvenance(TypedDict):
    """Evidence connecting a device activity event to one host graph launch."""

    passed: bool
    basis: str
    context: int | None
    correlation: int | None
    graph_id: int | None
    graph_node_id: int | None
    errors: list[str]


class CallbackCoverage(TypedDict):
    """Scheduling-site counts corroborate an independent ordered API audit."""

    raw_api_events: int
    xla_scheduling_scopes: int
    logical_scheduling_sites: int
    raw_events_nested_in_scopes: int
    expected_from_independent_audit: int | None
    count_status: str
    dependency_proven_in_this_trace: bool
    callback_body_identified: bool
    invocation_relation: str
    independent_submission_audit_revalidated: NotRequired[bool]
    errors: NotRequired[list[str]]


class TraceReport(TypedDict, total=False):
    """Trace corroboration result; it never replaces the ordered API certificate."""

    schema: int
    passed: bool
    errors: list[str]
    interval_us: tuple[float, float] | None
    processes: dict[int | str, str]
    counts: dict[str, int]
    gpu_compute_events: int
    gpu_d2d_events: int
    gpu_memset_events: int
    copy_named_kernels: int
    graph_node_correlated_events: int
    post_graph_host_function_calls: int
    callback_coverage: CallbackCoverage
    host_function_dependency_from_paired_cupti_audit: bool
    standalone_host_function_identity_proven: bool
    command_buffer_execute_events: int
    examples: dict[str, object]
    scope: str


class EligibilityReport(TypedDict):
    """Static capture obstacles; passing this screen is not a runtime certificate."""

    passed: bool
    errors: list[str]
    custom_call_targets: dict[str, int]
    primitive_triangular_solve_count: int
    runtime_residency_proven: bool
    source_scope: str
