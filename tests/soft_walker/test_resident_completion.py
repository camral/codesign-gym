"""Boundary-notification regression tests; synthetic records are NOT GPU runs."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Self


import ctypes as ct
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from envs.soft_walker.engine.execution.compiled import CompiledProgram, _jaxpr_audit
from envs.soft_walker.engine.execution.cupti import (
    DriverAudit,
    _CallbackPrefix,
    _copy_parameters,
    _EventRecord,
    _GraphLaunch,
    _HostFunction,
    _StreamWait,
)
from envs.soft_walker.engine.execution.policy import ResidencyError, ResidentPolicy
from envs.soft_walker.engine.execution.submissions import inspect_api_counts, inspect_submissions
from envs.soft_walker.engine.execution.trace import MARKER, inspect_trace

CTX, STREAM, NOTIFY, EVENT, FUNCTION = 0x1000, 0x2000, 0x3000, 0x4000, 0x5000


def records(*, same_stream: bool = False, suffix: str = "") -> list[dict[str, object]]:
    specs = [
        ("cuCtxSetCurrent", {}),
        ("cuGraphLaunch" + suffix, {"stream": STREAM, "graph_exec": 0x6000}),
        ("cuEventRecord" + suffix, {"stream": STREAM, "event": EVENT}),
    ]
    if not same_stream:
        specs += [("cuStreamWaitEvent" + suffix, {"stream": NOTIFY, "event": EVENT, "flags": 0})]
    specs += [
        (
            "cuLaunchHostFunc" + suffix,
            {
                "stream": STREAM if same_stream else NOTIFY,
                "function": FUNCTION,
                "user_data": 0x7000,
            },
        )
    ]
    return [
        dict(
            sequence=i,
            name=name,
            parameters=parameters,
            context=CTX,
            enter_ns=i * 10,
            exit_ns=i * 10 + 2,
            return_status=0,
        )
        for i, (name, parameters) in enumerate(specs)
    ]


def audit(rows: list[dict[str, object]], **kwargs: object) -> dict[str, object]:
    return inspect_submissions(Counter(r["name"] for r in rows), rows, **kwargs)


def renumber(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    for i, row in enumerate(rows):
        row.update(sequence=i, enter_ns=10 * i, exit_ns=10 * i + 2)
    return rows


@pytest.mark.parametrize("same_stream", [False, True])
@pytest.mark.parametrize("suffix", ["", "_ptsz"])
def test_terminal_notification_needs_real_dependency(same_stream: bool, suffix: str) -> None:
    rows = records(same_stream=same_stream, suffix=suffix)
    report = audit(rows)
    assert report["passed"], report["errors"]
    assert report["counts"]["host_graph_submissions"] == 1
    assert report["counts"]["host_function_submissions"] == 1  # Not erased.
    assert report["counts"]["terminal_host_notifications"] == 1
    assert report["counts"]["intra_simulation_or_unclassified_host_functions"] == 0
    proof = report["terminal_notifications"][0]["dependency_entries"]
    assert proof == ([1, 3] if same_stream else [1, 2, 3, 4])
    assert not report["terminal_notifications"][0]["callback_body_identified"]
    assert not inspect_api_counts(report["driver_api_counts"])["passed"]


@pytest.mark.parametrize(
    "change",
    [
        "unrelated_stream",
        "no_wait",
        "wrong_event",
        "other_context",
        "default_stream",
        "missing_parameters",
        "graph_failure",
        "wait_failure",
        "callback_failure",
        "missing_exit",
        "overlapping_calls",
        "callback_before_graph",
        "event_rerecorded",
        "event_destroyed",
        "unknown_host_variant",
        "in_graph_host_node",
        "work_after",
        "transfer",
        "second_graph",
        "standalone_kernel",
    ],
)
def test_real_cpu_fallback_and_ambiguous_notification_remain_failures(change: str) -> None:
    rows = records()
    if change == "unrelated_stream":
        rows[-1]["parameters"]["stream"] = 0x9990
    elif change == "no_wait":
        del rows[3]
        renumber(rows)
    elif change == "wrong_event":
        rows[3]["parameters"]["event"] += 16
    elif change == "other_context":
        rows[-1]["context"] += 16
    elif change == "default_stream":
        rows[-1]["parameters"]["stream"] = 0
    elif change == "missing_parameters":
        del rows[1]["parameters"]
    elif change == "graph_failure":
        rows[1]["return_status"] = 1
    elif change == "wait_failure":
        rows[3]["return_status"] = 1
    elif change == "callback_failure":
        rows[-1]["return_status"] = 1
    elif change == "missing_exit":
        del rows[3]["exit_ns"]
    elif change == "overlapping_calls":
        rows[2]["enter_ns"] = rows[1]["enter_ns"] + 1
    elif change == "callback_before_graph":
        row = rows.pop()
        rows.insert(0, row)
        renumber(rows)
    elif change in ("event_rerecorded", "event_destroyed"):
        row = deepcopy(rows[2])
        if change == "event_rerecorded":
            row["parameters"]["stream"] = 0x9990
        else:
            row["name"] = "cuEventDestroy_v2"
        rows.insert(3, row)
        renumber(rows)
    elif change == "unknown_host_variant":
        rows[-1]["name"] = "cuLaunchHostFunc_v99"
    else:
        row = deepcopy(rows[-1])
        row["name"] = {
            "in_graph_host_node": "cuGraphAddHostNode",
            "work_after": "cuStreamWriteValue32",
            "transfer": "cuMemcpyDtoHAsync_v2",
            "second_graph": "cuGraphLaunch",
            "standalone_kernel": "cuLaunchKernel",
        }[change]
        rows.append(row)
        renumber(rows)
    assert not audit(rows)["passed"], change


def test_counter_only_returned_evidence_cannot_be_retroactively_certified() -> None:
    counts = {
        "cuCtxSetCurrent": 4,
        "cuEventRecord": 2,
        "cuGraphLaunch": 1,
        "cuLaunchHostFunc": 1,
        "cuStreamWaitEvent": 1,
    }
    assert not inspect_api_counts(counts)["passed"]
    assert not inspect_submissions(counts, records())["passed"]
    assert not audit(records(), records_complete=False)["passed"]
    assert not audit(records(), errors=["collector failed"])["passed"]


def test_transitive_event_dependency_and_multiple_terminal_notifications() -> None:
    rows = records()
    record = deepcopy(rows[2])
    record["parameters"] = {"event": EVENT + 16, "stream": NOTIFY}
    wait = deepcopy(rows[3])
    wait["parameters"] = {"event": EVENT + 16, "stream": NOTIFY + 16, "flags": 0}
    rows[-1]["parameters"]["stream"] = NOTIFY + 16
    rows[-1:-1] = [record, wait]
    rows.append(deepcopy(rows[-1]))
    report = audit(renumber(rows))
    assert report["passed"], report["errors"]
    assert report["counts"]["terminal_host_notifications"] == 2


class Function:
    def __init__(self, fn: Callable[..., object]) -> None:
        self.fn = fn

    def __call__(self, *args: object) -> object:
        return self.fn(*args)


class Library:
    """No CUDA use; drives the real ctypes callback with owned host memory."""

    def __init__(self) -> None:
        self.callback = None
        self.enabled = False
        self.cuptiSubscribe = Function(self.subscribe)
        self.cuptiEnableDomain = Function(self.enable)
        self.cuptiUnsubscribe = Function(lambda h: 0)
        self.cuptiGetResultString = Function(lambda *a: 0)
        self.cuptiGetVersion = Function(self.version)

    def subscribe(self, handle: object, callback: Callable[..., None], user: object) -> int:
        ct.cast(handle, ct.POINTER(ct.c_void_p))[0] = ct.c_void_p(123)
        self.callback = callback
        return 0

    def enable(self, enabled: int, handle: object, domain: int) -> int:
        self.enabled = bool(enabled)
        return 0

    def version(self, value: object) -> int:
        ct.cast(value, ct.POINTER(ct.c_uint32))[0] = 28
        return 0

    def emit(self, name: str, parameters: object, status: int = 0, *, exit: bool = True) -> None:
        code = ct.c_int(status)
        prefix = _CallbackPrefix(
            0, name.encode(), ct.addressof(parameters), ct.addressof(code), None, CTX
        )
        self.callback(None, 1, 7, ct.byref(prefix))
        if exit:
            prefix.callbackSite = 1
            self.callback(None, 1, 7, ct.byref(prefix))


def test_live_ctypes_parameter_capture_and_return_codes() -> None:
    library = Library()
    observer = DriverAudit(library=library)
    with observer:
        library.emit("cuGraphLaunch", _GraphLaunch(0x6000, STREAM))
        library.emit("cuEventRecord", _EventRecord(EVENT, STREAM))
        library.emit("cuStreamWaitEvent", _StreamWait(NOTIFY, EVENT, 0))
        library.emit("cuLaunchHostFunc", _HostFunction(NOTIFY, FUNCTION, 0x7000))
    report = observer.report()
    assert report["passed"], report["errors"]
    assert report["counts"]["api_entries"] == 4  # EXIT is not double counted.
    assert report["driver_api_records"][2]["parameters"]["event"] == EVENT
    assert all(r["return_status"] == 0 for r in report["driver_api_records"])
    assert not library.enabled


@pytest.mark.parametrize("mode", ["truncated", "missing_exit", "failed_exit", "null_params"])
def test_collector_does_not_approve_incomplete_callback_evidence(mode: str) -> None:
    library = Library()
    observer = DriverAudit(library=library, max_records=1 if mode == "truncated" else 4096)
    with observer:
        library.emit("cuGraphLaunch", _GraphLaunch(0x6000, STREAM))
        if mode == "null_params":
            value = _CallbackPrefix(0, b"cuLaunchHostFunc", None, None, None, CTX)
            library.callback(None, 1, 7, ct.byref(value))
        else:
            library.emit(
                "cuLaunchHostFunc",
                _HostFunction(STREAM, FUNCTION, 0x7000),
                status=1 if mode == "failed_exit" else 0,
                exit=mode != "missing_exit",
            )
    assert not observer.report()["passed"]
    assert not library.enabled


def test_parameter_prefix_layout_and_unknown_names() -> None:
    assert _CallbackPrefix.functionParams.offset == 2 * ct.sizeof(ct.c_void_p)
    assert _CallbackPrefix.context.offset == 5 * ct.sizeof(ct.c_void_p)
    value = _HostFunction(STREAM, FUNCTION, 0x7000)
    assert _copy_parameters("cuLaunchHostFunc_v2_ptsz", ct.addressof(value))["function"] == FUNCTION
    assert _copy_parameters("unknown", None) == {}
    with pytest.raises(ValueError, match="Missing functionParams"):
        _copy_parameters("cuGraphLaunch", None)


def trace_fixture(host_name: str = "cuLaunchHostFunc") -> dict[str, object]:
    return {
        "traceEvents": [
            {"ph": "M", "pid": 1, "name": "process_name", "args": {"name": "/device:GPU:0"}},
            {"ph": "M", "pid": 7, "name": "process_name", "args": {"name": "/host:CPU"}},
            {"ph": "X", "pid": 7, "name": MARKER, "ts": 10.0, "dur": 100.0},
            {
                "ph": "X",
                "pid": 7,
                "name": "cuGraphLaunch (CudaGraph:123)",
                "ts": 11.0,
                "dur": 2.0,
                "args": {"context_id": "1", "correlation_id": "11"},
            },
            {
                "ph": "X",
                "pid": 1,
                "name": "kernel",
                "ts": 15.0,
                "dur": 90.0,
                "args": {
                    "cuda_graph_id": "123",
                    "cuda_graph_node_id": "41",
                    "context_id": "1",
                    "correlation_id": "11",
                    "kernel_details": "regs:16 grid:1,1,1 block:32,1,1",
                },
            },
            {"ph": "X", "pid": 7, "name": host_name, "ts": 14.0, "dur": 1.0},
        ]
    }


def test_trace_is_supplemental_and_not_a_callback_allowlist() -> None:
    trace = trace_fixture()
    assert not inspect_trace(trace)["passed"]
    report = audit(records())
    assert inspect_trace(trace, submission_audit=report)["passed"]
    assert not inspect_trace(trace_fixture("xla_python_cpu_callback"), submission_audit=report)[
        "passed"
    ]
    before = deepcopy(trace)
    before["traceEvents"][-1]["ts"] = 10.5
    assert not inspect_trace(before, submission_audit=report)["passed"]
    twice = deepcopy(trace)
    twice["traceEvents"].append(dict(twice["traceEvents"][-1]))
    assert not inspect_trace(twice, submission_audit=report)["passed"]


def test_rejected_submission_still_writes_requested_trace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import envs.soft_walker.engine.execution.compiled as module

    bad = audit(records())
    bad["passed"] = False
    bad["errors"] = ["injected collector failure"]

    class Observer:
        def __enter__(self) -> Self:
            return self

        def __exit__(self, *a: object) -> None:
            pass

        def report(self) -> dict[str, object]:
            return bad

    monkeypatch.setattr(module, "DriverAudit", Observer)

    def start(path: Path, **kwargs: object) -> None:
        import gzip

        dest = Path(path) / "probe.trace.json.gz"
        dest.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(dest, "wt") as f:
            json.dump(trace_fixture(), f)

    monkeypatch.setattr(jax.profiler, "start_trace", start)
    monkeypatch.setattr(jax.profiler, "stop_trace", lambda: None)
    # Fake policy here isolates host evidence code. It is NOT a GPU execution.
    program = CompiledProgram(None, jax.devices()[0], ResidentPolicy("cuda"), tmp_path, {})
    program._invoke = lambda args: jnp.asarray(0.25)
    with pytest.raises(ResidencyError, match="injected collector failure"):
        program.certify(
            jnp.asarray(0.2),
            detailed_trace=True,
            expected_host=np.asarray(0.25, dtype=np.asarray(jnp.asarray(0.25)).dtype),
        )
    assert not program.certified
    saved = json.loads((tmp_path / "certificate.json").read_text())
    assert "detailed_trace" in saved
    assert (tmp_path / "submission_audit.json").is_file()
    assert (tmp_path / "detailed_trace_audit.json").is_file()


def test_no_changes_to_gpu_compiler_policy_or_callback_free_function_requirement() -> None:
    options = ResidentPolicy("cuda").compiler_options()
    assert "WHILE" in options["xla_gpu_enable_command_buffer"]
    assert options["xla_gpu_command_buffer_unroll_loops"] is False
    description = ResidentPolicy("cuda").descriptor()
    assert description["cuda_acceptance"]["simulation_callbacks"] == 0

    def callback(x: jax.Array) -> jax.Array:
        return jax.pure_callback(lambda y: np.asarray(y), jax.ShapeDtypeStruct((), x.dtype), x)

    assert not _jaxpr_audit(jax.make_jaxpr(callback)(jnp.asarray(0.2)))["passed"]
