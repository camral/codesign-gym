"""Lightweight, fail-closed CUDA submission audit for a WHOLE executable.

Use CUPTI's synchronous DRIVER_API callbacks, not its kernel-activity buffers.
A long implicit trajectory can contain millions of child kernels; recording all
of them just to count host submissions would make validation impractical.

This observer counts host CUDA API calls. It does not execute simulation code,
copy simulation values, or decide timesteps. It is installed for certification
only, outside the timed production calls. A certificate is NOT a claim that the
CUDA graph contains one kernel. See trace.py for optional detailed XProf audits.
"""

from __future__ import annotations

import ctypes as ct
import ctypes.util
import os
import re
import sys
import threading
import time
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from types import TracebackType
from typing import Self, cast

from .audit_types import DriverRecord, SubmissionReport
from .policy import ResidencyError
from .submissions import inspect_api_counts as inspect_api_counts, inspect_submissions

DRIVER_API = 1
API_ENTER = 0


class _CallbackPrefix(ct.Structure):
    # Documented initial fields of CUpti_CallbackData (CUDA 12/13). Read no
    # trailing version-dependent fields. functionParams is valid only during
    # the CUPTI callback, so copy handles immediately; never retain C pointers.
    _fields_ = [
        ("callbackSite", ct.c_int),
        ("functionName", ct.c_char_p),
        ("functionParams", ct.c_void_p),
        ("functionReturnValue", ct.c_void_p),
        ("symbolName", ct.c_char_p),
        ("context", ct.c_void_p),
    ]


class _GraphLaunch(ct.Structure):
    _fields_ = [("graph_exec", ct.c_void_p), ("stream", ct.c_void_p)]


class _EventRecord(ct.Structure):
    _fields_ = [("event", ct.c_void_p), ("stream", ct.c_void_p)]


class _StreamWait(ct.Structure):
    _fields_ = [("stream", ct.c_void_p), ("event", ct.c_void_p), ("flags", ct.c_uint)]


class _HostFunction(ct.Structure):
    _fields_ = [("stream", ct.c_void_p), ("function", ct.c_void_p), ("user_data", ct.c_void_p)]


class _EventDestroy(ct.Structure):
    _fields_ = [("event", ct.c_void_p)]


class _StreamDestroy(ct.Structure):
    _fields_ = [("stream", ct.c_void_p)]


def _copy_parameters(name: str, pointer: int | ct.c_void_p | None) -> dict[str, int]:
    """Copy CUDA's documented parameter prefixes, not any device allocation.

    The versioned EventRecordWithFlags/LaunchHostFunc_v2 forms add trailing
    fields; the dependency handles remain in the prefixes read here. Unknown
    names are retained in the log but are never guessed to have a known layout.
    """
    declarations = (
        (r"^cuGraphLaunch(?:_ptsz)?$", _GraphLaunch),
        (r"^cuEventRecord(?:WithFlags)?(?:_ptsz)?$", _EventRecord),
        (r"^cuStreamWaitEvent(?:_ptsz)?$", _StreamWait),
        (r"^cuLaunchHostFunc(?:_v2)?(?:_ptsz)?$", _HostFunction),
        (r"^cuEventDestroy(?:_v2)?$", _EventDestroy),
        (r"^cuStreamDestroy(?:_v2)?$", _StreamDestroy),
    )
    for pattern, layout in declarations:
        if re.fullmatch(pattern, name):
            if not pointer:
                raise ValueError(f"Missing functionParams for {name}")
            values = ct.cast(pointer, ct.POINTER(layout)).contents
            return {
                field: int(getattr(values, field) or 0)
                for field, _ in cast(list[tuple[str, object]], layout._fields_)
            }
    return {}


_Callback = ct.CFUNCTYPE(None, ct.c_void_p, ct.c_int, ct.c_uint32, ct.c_void_p)
_subscription_lock = threading.Lock()


def _library_candidates() -> Iterator[str]:
    """Yield candidate CUPTI library paths in load-preference order.

    An explicit environment override is authoritative; discovery never changes
    the loaded CUDA toolkit or executes a candidate library.
    """
    override = os.environ.get("CODESIGN_CUPTI_LIBRARY")
    if override:
        yield override
        return  # An explicit invalid path is not silently ignored.
    # Prefer the toolkit already loaded by the running JAX CUDA plugin.
    maps = Path("/proc/self/maps")
    if maps.exists():
        for line in maps.read_text().splitlines():
            if "libcupti.so" in line and "/" in line:
                yield line[line.index("/") :].removesuffix(" (deleted)")
    for base in sys.path:
        if base:
            yield from map(str, Path(base).glob("nvidia/cuda_cupti/lib/libcupti.so*"))
    found = ctypes.util.find_library("cupti")
    if found:
        yield found
    for base in ("/usr/local/cuda/extras/CUPTI/lib64", "/usr/local/cuda/lib64"):
        yield from map(str, Path(base).glob("libcupti.so*"))


def _load_library() -> tuple[ct.CDLL, str]:
    """Load the first available CUPTI library or raise ResidencyError.

    The returned handle owns the host library, and the string records its path.
    This operation does not subscribe to callbacks.
    """
    failures = []
    for path in dict.fromkeys(_library_candidates()):
        try:
            return ct.CDLL(path), path
        except OSError as error:
            failures.append(f"{path}: {error}")
    raise ResidencyError(
        "CUPTI callback library unavailable. Use the CUPTI library matching your "
        "installed JAX CUDA toolkit, or set CODESIGN_CUPTI_LIBRARY to its absolute "
        "libcupti.so path. Residency is NOT approved without evidence. " + "; ".join(failures)
    )


class DriverAudit:
    """Process-wide CUPTI observer; no Nsight/XProf session may overlap it.

    Counts remain complete. Ordered metadata is bounded; reaching its limit
    fails certification instead of approving a truncated trace. Exceptions in
    a C callback remain failures and never escape the C ABI. No device values
    or host callback user-data objects are dereferenced by this observer.
    """

    def __init__(self, *, library: ct.CDLL | None = None, max_records: int = 4096) -> None:
        """Prepare an observer and bind the host CUPTI ABI.

        The optional library supports injected test doubles. max_records bounds
        metadata storage; exceeding it later fails certification. Subscription is
        owned by the context manager, never by construction.
        """
        self.library, self.library_path = (
            _load_library() if library is None else (library, "injected-test-library")
        )
        if type(max_records) is not int or max_records < 1:
            raise ValueError("max_records must be a positive integer")
        self.max_records = max_records
        self.records: list[DriverRecord] = []
        self.records_complete = True
        self.pending: dict[tuple[int, int, str], list[DriverRecord]] = {}
        self.record_lock = threading.Lock()
        self.counts: Counter[str] = Counter()
        self.errors: list[str] = []
        self.examples: list[str] = []
        self.handle = ct.c_void_p()
        self.enabled = False
        self.locked = False
        self.callback = _Callback(self._observe)
        self._bind()
        version = ct.c_uint32()
        self._check(self.library.cuptiGetVersion(ct.byref(version)), "cuptiGetVersion")
        self.version = int(version.value)

    def _bind(self) -> None:
        """Set C argument and return layouts before any CUPTI call."""
        signatures = {
            "cuptiSubscribe": [ct.POINTER(ct.c_void_p), _Callback, ct.c_void_p],
            "cuptiEnableDomain": [ct.c_uint32, ct.c_void_p, ct.c_int],
            "cuptiUnsubscribe": [ct.c_void_p],
            "cuptiGetVersion": [ct.POINTER(ct.c_uint32)],
            "cuptiGetResultString": [ct.c_int, ct.POINTER(ct.c_char_p)],
        }
        for name, args in signatures.items():
            function = getattr(self.library, name)
            function.argtypes, function.restype = args, ct.c_int

    def _check(self, result: int, operation: str) -> None:
        """Raise ResidencyError for a nonzero CUPTI result with its decoded message."""
        if result:
            text = ct.c_char_p()
            self.library.cuptiGetResultString(int(result), ct.byref(text))
            message = text.value.decode(errors="replace") if text.value else str(result)
            raise ResidencyError(
                f"{operation}: {message}. Do not run this certificate inside "
                "Nsight/XProf or another CUPTI subscriber; it cannot be skipped."
            )

    def _observe(self, userdata: int | None, domain: int, cbid: int, data: int | None) -> None:
        # The lock gives one consistent ENTER order across submitting threads.
        # EXIT timestamps additionally guard against overlapping cross-thread
        # API calls when constructing stream/event dependency proofs.
        """Copy one synchronous driver callback into host-owned metadata.

        CUPTI owns all incoming pointers only during this call. Exceptions are
        retained as audit failures instead of crossing the C callback boundary.
        """
        with self.record_lock:
            try:
                if domain != DRIVER_API:
                    raise ValueError(f"Unexpected callback domain {domain}")
                if not data:
                    raise ValueError("Missing CUpti_CallbackData")
                prefix = ct.cast(data, ct.POINTER(_CallbackPrefix)).contents
                if not prefix.functionName:
                    raise ValueError(f"Missing function name for callback {cbid}")
                name = prefix.functionName.decode("ascii", errors="strict")
                key = (threading.get_ident(), int(cbid), name)
                now = time.monotonic_ns()
                if prefix.callbackSite == API_ENTER:
                    self.counts[name] += 1
                    if len(self.examples) < 128:
                        self.examples.append(name)
                    if len(self.records) >= self.max_records:
                        self.records_complete = False
                        return
                    row: DriverRecord = {
                        "sequence": len(self.records),
                        "name": name,
                        "thread": threading.get_native_id(),
                        "callback_id": int(cbid),
                        "enter_ns": now,
                        "context": int(prefix.context or 0),
                    }
                    self.records.append(row)
                    self.pending.setdefault(key, []).append(row)
                    row["parameters"] = _copy_parameters(name, prefix.functionParams)
                elif prefix.callbackSite == 1:  # CUPTI_API_EXIT
                    stack = self.pending.get(key)
                    if not stack:
                        # A domain may be enabled after an unrelated API entry.
                        # Such an exit cannot prove anything and is not used.
                        return
                    row = stack.pop()
                    row["exit_ns"] = now
                    if not prefix.functionReturnValue:
                        raise ValueError(f"Missing functionReturnValue for {name}")
                    row["return_status"] = int(
                        ct.cast(prefix.functionReturnValue, ct.POINTER(ct.c_int)).contents.value
                    )
                else:
                    raise ValueError(f"Unknown CUPTI callback site {prefix.callbackSite}")
            except Exception as error:
                self.errors.append(repr(error))

    def __enter__(self) -> Self:
        """Acquire the process-wide observer lock and subscribe, or raise."""
        if not _subscription_lock.acquire(blocking=False):
            raise ResidencyError("A residency audit is already running in this process")
        self.locked = True
        try:
            self._check(
                self.library.cuptiSubscribe(ct.byref(self.handle), self.callback, None),
                "cuptiSubscribe",
            )
            self._check(
                self.library.cuptiEnableDomain(1, self.handle, DRIVER_API), "cuptiEnableDomain"
            )
            self.enabled = True
            return self
        except Exception:
            self._close()
            raise

    def _close(self) -> None:
        """Unsubscribe and release the process lock, retaining teardown failures."""
        try:
            if self.enabled:
                result = self.library.cuptiEnableDomain(0, self.handle, DRIVER_API)
                self.enabled = False
                if result:
                    self.errors.append(f"Disabling driver callbacks failed: {result}")
            if self.handle.value:
                result = self.library.cuptiUnsubscribe(self.handle)
                self.handle = ct.c_void_p()
                if result:
                    self.errors.append(f"Unsubscribing driver callbacks failed: {result}")
        finally:
            if self.locked:
                self.locked = False
                _subscription_lock.release()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Release the observer even when the measured invocation raises."""
        self._close()

    def report(self) -> SubmissionReport:
        """Validate copied submissions and return their host-side evidence.

        Outstanding callbacks, truncated records, or teardown errors remain
        failures. This never reads simulation arrays from the device.
        """
        pending = [row["sequence"] for stack in self.pending.values() for row in stack]
        errors = list(self.errors)
        if pending:
            errors.append(f"Missing API EXIT records: {pending[:16]}")
        report = inspect_submissions(
            self.counts, self.records, errors, records_complete=self.records_complete
        )
        cast(dict[str, object], report).update(
            library=self.library_path,
            cupti_version=self.version,
            first_api_names=self.examples,
            record_limit=self.max_records,
            abi_prefix_fields=[
                name for name, _ in cast(list[tuple[str, object]], _CallbackPrefix._fields_)
            ],
            abi_pointer_bytes=ct.sizeof(ct.c_void_p),
        )
        return report
