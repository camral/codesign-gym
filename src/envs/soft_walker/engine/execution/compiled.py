"""Compile complete programs, then certify the actual CUDA execution boundary.

Compilation, initial library/graph warmup, device input staging, and one final
output read are setup/boundary work. Simulation, settling, and differentiation
must not dispatch or copy to the host within the certified invocation. There is
no fallback from an uncertified CUDA executable to the old host-driven path.
"""

from __future__ import annotations

import json
import os
import platform
import time
from collections.abc import Callable
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Generic, Protocol, TypeVar, cast

import jax
import numpy as np

from ..types import PathLike
from .contracts import Certificate, EquivalenceReport, JaxprAudit, OutputSummary, ShapeRecord
from .cupti import DriverAudit
from .eligibility import capture_eligibility
from .policy import ResidencyError, ResidentPolicy

OutputT = TypeVar("OutputT")


class _Effectful(Protocol):
    """The effects attribute shared by open JAXPR variants."""

    effects: object


def _write(path: PathLike, value: object) -> None:
    """Atomically replace a JSON evidence file after creating its parent directory.

    Unsupported metadata values use their string representation, as in the existing
    evidence format. Filesystem errors propagate to the caller."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temporary.replace(path)


def _jaxpr_audit(closed: object) -> JaxprAudit:
    """Traverse nested JAXPRs once and inventory primitives and prohibited host effects.

    The visitor accepts JAX internal container variants through their structural
    attributes; it neither evaluates nor transforms the traced program."""
    counts: dict[str, int] = {}
    forbidden: list[str] = []
    seen: set[int] = set()

    def visit(value: object) -> None:
        """Visit nested graphs, dictionaries, and sequences without revisiting graph objects."""
        if id(value) in seen:
            return
        if hasattr(value, "jaxpr"):
            seen.add(id(value))
            visit(value.jaxpr)
        elif hasattr(value, "eqns"):
            seen.add(id(value))
            if cast(_Effectful, value).effects:
                forbidden.append("effects: " + str(cast(_Effectful, value).effects))
            for equation in value.eqns:
                name = equation.primitive.name
                counts[name] = counts.get(name, 0) + 1
                if any(word in name for word in ("callback", "outside_call", "debug_print")):
                    forbidden.append(name)
                visit(equation.params)
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)

    visit(closed)
    return {
        "primitive_counts": counts,
        "forbidden": sorted(set(forbidden)),
        "passed": not forbidden,
    }


def _check_host_workspace(memory: dict[str, int] | None, policy: ResidentPolicy) -> None:
    """Reject compiler-declared host arrays, including intermediate offload."""
    if policy.device != "cuda":
        return
    required = (
        "host_argument_size_in_bytes",
        "host_output_size_in_bytes",
        "host_temp_size_in_bytes",
        "host_alias_size_in_bytes",
    )
    if memory is None or any(name not in memory for name in required):
        raise ResidencyError("CUDA compiler did not expose required host-memory accounting")
    used = {name: memory[name] for name in required if memory[name] != 0}
    if used:
        raise ResidencyError("Host-memory offload inside compiled program: " + str(used))


def _runtime_inventory() -> dict[str, object]:
    """Record exact installed wheels; do not infer an XLA commit from a version."""
    packages = {}
    for distribution in metadata.distributions():
        name = distribution.metadata.get("Name", "")
        if name.lower().startswith(("jax", "nvidia-")):
            packages[name] = distribution.version
    return {
        "packages": dict(sorted(packages.items())),
        "XLA_FLAGS": os.environ.get("XLA_FLAGS"),
        "JAX_COMPILATION_CACHE_DIR": os.environ.get("JAX_COMPILATION_CACHE_DIR"),
        "xla_build_commit": "not established by package-version metadata",
    }


def _output_summary(tree: object) -> OutputSummary:
    """Host-side evidence only, after the complete invocation has returned."""
    leaves: list[dict[str, object]] = []
    for index, item in enumerate(jax.tree.leaves(tree)):
        value = np.asarray(item)
        info: dict[str, object] = {
            "leaf": index,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
        if value.dtype == jax.dtypes.float0:
            info.update(finite=True, discrete_zero_tangent=True)
        else:
            finite = np.isfinite(value)
            bad = np.flatnonzero(~finite.reshape(-1))
            info.update(
                finite=bool(finite.all()),
                nan_count=int(np.isnan(value).sum()),
                positive_inf_count=int(np.isposinf(value).sum()),
                negative_inf_count=int(np.isneginf(value).sum()),
                nonfinite_flat_indices=bad[:8].tolist(),
                sample_values=[repr(v) for v in value.reshape(-1)[:8]],
            )
        leaves.append(info)
    return {"leaves": leaves, "self_check": _equivalent(tree, tree)}


def _shapes(tree: object) -> list[ShapeRecord]:
    """Describe flattened array leaves and estimate their total device allocation bytes."""
    return [
        {
            "shape": list(x.shape),
            "dtype": str(x.dtype),
            "bytes": int(np.prod(x.shape, dtype=np.int64)) * np.dtype(x.dtype).itemsize,
        }
        for x in jax.tree.leaves(tree)
    ]


def _equivalent(a: object, b: object) -> EquivalenceReport:
    """Compare synchronized output pytrees with existing dtype-specific tolerances.

    Shape, dtype, finite-value, and physical-validity mismatches are failures; this
    check never broadcasts a mismatched output into apparent agreement."""
    pairs = list(zip(jax.tree.leaves(a), jax.tree.leaves(b)))
    if jax.tree.structure(a) != jax.tree.structure(b):
        return {"passed": False, "errors": ["Output trees differ"]}
    errors = []
    largest = 0.0
    for i, (x, y) in enumerate(pairs):
        x, y = np.asarray(x), np.asarray(y)
        # np.allclose broadcasts. A scalar output must not silently validate a
        # vector, and a precision-changing execution is not the same program.
        if x.shape != y.shape:
            errors.append(f"Output shape differs at leaf {i}: {x.shape} vs {y.shape}")
            continue
        if x.dtype != y.dtype:
            errors.append(f"Output dtype differs at leaf {i}: {x.dtype} vs {y.dtype}")
            continue
        if x.dtype == jax.dtypes.float0 and y.dtype == jax.dtypes.float0:
            continue
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            errors.append(f"Nonfinite output leaf {i}")
            continue
        difference = float(np.max(np.abs(x.astype(float) - y.astype(float)), initial=0.0))
        largest = max(largest, difference)
        tol = 5e-10 if x.dtype == np.float64 else 2e-5
        if not np.allclose(x, y, rtol=tol, atol=tol * 0.01):
            errors.append(f"Profiled/unprofiled leaf {i} disagrees (max abs {difference})")
    # Physical forward/record returns a validity bit; gradients propagate NaN
    # on invalid physics so the finite check handles scalar gradient programs.
    if isinstance(a, dict) and "valid" in a and not bool(np.all(a["valid"])):
        errors.append("Physical rollout invalid")
    return {"passed": not errors, "max_absolute_difference": largest, "errors": errors}


@dataclass
class CompiledProgram(Generic[OutputT]):
    """One executable and its device-owned inputs, policy, and persisted evidence.

    Production calls are rejected until certification succeeds. Compilation and
    ownership stay at this boundary; only explicit stage/evidence calls transfer.
    """

    executable: Callable[..., OutputT]
    device: jax.Device
    policy: ResidentPolicy
    output: Path
    manifest: dict[str, object]
    certified: bool = False

    def stage(self, *arguments: object) -> tuple[object, ...]:
        """Explicit boundary upload; never invoked from simulation."""
        with jax.transfer_guard("allow"):
            values = jax.device_put(arguments, self.device)
            jax.block_until_ready(values)
        return values

    def _check_inputs(self, arguments: object) -> None:
        """Reject non-device leaves, wrong-device buffers, and host-offloaded CUDA memory."""
        for leaf in jax.tree.leaves(arguments):
            if not isinstance(leaf, jax.Array):
                raise ResidencyError("Stage array inputs before invoking the executable")
            if leaf.devices() != {self.device}:
                raise ResidencyError("Input is not resident on the selected device")
            memory = getattr(leaf.sharding, "memory_kind", None)
            if self.policy.device == "cuda" and memory not in (None, "device"):
                raise ResidencyError(f"Host-offloaded input memory kind: {memory}")

    def _invoke(self, arguments: tuple[object, ...]) -> OutputT:
        """Synchronously execute one staged call while prohibiting explicit host transfers."""
        self._check_inputs(arguments)
        with jax.transfer_guard("disallow_explicit"):
            result = self.executable(*arguments)
            jax.block_until_ready(result)
        self._check_inputs(result)  # No output may be silently host-offloaded.
        return result

    def certify(
        self,
        *arguments: object,
        alternate_arguments: tuple[object, ...] | None = None,
        detailed_trace: bool = False,
        expected_host: object | None = None,
        alternate_expected_host: object | None = None,
    ) -> OutputT:
        """Warm, verify values, then audit one entire executable invocation.

        Expected values can come from an independent analytic/ordinary-compiler
        diagnostic. Such a diagnostic is not production fallback. Evidence is
        committed before every possibly blocking call; transfers for evidence
        occur only after complete invocations, never inside an audited span.
        """
        self.certified = False
        report: Certificate = {
            "passed": False,
            "device": self.policy.device,
            "phases": [],
            "external_reference_supplied": expected_host is not None,
        }
        started = time.perf_counter()

        def phase(label: str) -> None:
            """Persist phase-start evidence before entering a potentially blocking device call."""
            report["phase"] = label
            report["phases"].append(
                {"name": label, "state": "started", "elapsed_s": time.perf_counter() - started}
            )
            _write(self.output / "certificate.json", report)
            _write(self.output / "progress.json", report["phases"][-1])

        def finished(label: str, result: OutputT, expected: object | None = None) -> object:
            # Outside the measured/audited invocation. A failed first warm call
            # is inspected immediately rather than executing it a second time.
            """Download completed outputs, persist finite-value evidence, and check references."""
            with jax.transfer_guard("allow"):
                host = jax.device_get(result)
            summary = _output_summary(host)
            if expected is not None:
                summary["external_reference"] = _equivalent(expected, host)
            _write(self.output / (label + ".outputs.json"), summary)
            total = sum(np.asarray(x).nbytes for x in jax.tree.leaves(host))
            if total <= 8 * 1024**2:
                np.savez_compressed(
                    self.output / (label + ".outputs.npz"),
                    # Generated leaf keys never collide with NumPy 2.5 allow_pickle.
                    **{f"leaf_{i}": np.asarray(x) for i, x in enumerate(jax.tree.leaves(host))},  # type: ignore[arg-type]
                )
            report["phases"][-1].update(
                state="returned",
                output_summary=label + ".outputs.json",
                elapsed_s=time.perf_counter() - started,
            )
            _write(self.output / "certificate.json", report)
            _write(self.output / "progress.json", report["phases"][-1])
            check = summary["self_check"]
            if not check["passed"]:
                raise ResidencyError(label + ": " + "; ".join(check["errors"]))
            if expected is not None and not summary["external_reference"]["passed"]:
                raise ResidencyError(label + ": differs from independent reference")
            return host

        def checked(
            label: str, values: tuple[object, ...], expected: object | None = None
        ) -> tuple[OutputT, object]:
            """Run a named synchronized certification phase and return device and host outputs."""
            phase(label)
            result = self._invoke(values)
            host = finished(label, result, expected)
            return result, host

        try:
            _, reference_host = checked("warmup_first", arguments, expected_host)
            repeat, repeat_host = checked("warmup_repeat", arguments, reference_host)
            report["unprofiled_repeat"] = _equivalent(reference_host, repeat_host)
            if self.policy.device == "cpu":
                report.update(
                    dict(gpu_residency_demonstrated=False, status="CPU numerical reference only")
                )
            else:
                audit_arguments = arguments if alternate_arguments is None else alternate_arguments
                audit_expected = (
                    expected_host if alternate_arguments is None else alternate_expected_host
                )
                _, audit_reference_host = checked(
                    "alternate_warmup", audit_arguments, audit_expected
                )
                report["alternate_buffers_checked"] = alternate_arguments is not None
                checked("audit_warmup", audit_arguments, audit_reference_host)
                phase("submission_audit")
                observer = DriverAudit()
                with observer:
                    result = self._invoke(audit_arguments)
                report["submission_audit"] = observer.report()
                observed_host = finished("submission_audit", result, audit_reference_host)
                report["observed_comparison"] = _equivalent(audit_reference_host, observed_host)
                # Preserve optional detailed evidence even if metadata rejects
                # the invocation. Previously this early raise discarded the
                # trace requested precisely to diagnose a submission failure.
                _write(self.output / "submission_audit.json", report["submission_audit"])
                report["evidence_scope"] = {
                    "submission_audit": "complete raw DRIVER_API observation of one invocation",
                    "detailed_trace": "device-activity corroboration of a separate equivalent invocation"
                    if detailed_trace
                    else "not collected",
                    "callback_bodies_identified": False,
                    "same_invocation_stream_proof_transferred_to_trace": False,
                }
                if detailed_trace:
                    from .trace import MARKER, inspect_trace, load_trace

                    phase("detailed_trace")
                    trace_dir = self.output / "trace"
                    jax.profiler.start_trace(str(trace_dir), create_perfetto_trace=False)
                    try:
                        self._invoke(audit_arguments)
                        with jax.profiler.TraceAnnotation(MARKER):
                            traced = self._invoke(audit_arguments)
                    finally:
                        jax.profiler.stop_trace()
                    files = list(trace_dir.glob("**/*.trace.json.gz"))
                    if len(files) != 1:
                        raise ResidencyError("Detailed CUPTI trace is missing/ambiguous")
                    report["detailed_trace"] = inspect_trace(
                        load_trace(files[0]), submission_audit=report["submission_audit"]
                    )
                    traced_host = finished("detailed_trace", traced, audit_reference_host)
                    report["detailed_numerical_comparison"] = _equivalent(
                        audit_reference_host, traced_host
                    )
                    _write(self.output / "detailed_trace_audit.json", report["detailed_trace"])
                problems = []
                if not report["submission_audit"]["passed"]:
                    problems.extend(report["submission_audit"]["errors"])
                if detailed_trace and not report["detailed_trace"]["passed"]:
                    problems.extend(report["detailed_trace"]["errors"])
                if problems:
                    raise ResidencyError(
                        "Whole-program GPU residency failed: " + "; ".join(problems)
                    )
                checked_result, _ = checked("certified_result", arguments, reference_host)
                repeat = checked_result
                report.update(dict(gpu_residency_demonstrated=True))
            self.certified = True
            report.update(dict(passed=True, phase="complete"))
            return repeat
        except Exception as error:
            report["exception"] = repr(error)
            raise
        finally:
            report["seconds_including_warmup_and_audit"] = time.perf_counter() - started
            _write(self.output / "certificate.json", report)

    def __call__(self, *arguments: object) -> OutputT:
        """Invoke a previously certified executable; uncertified calls raise ResidencyError."""
        if not self.certified:
            raise ResidencyError("Call certify on this executable before production use")
        return self._invoke(arguments)

    def benchmark(self, *arguments: object, repeats: int = 5) -> OutputT:
        """Measure synchronized warm calls and persist seconds, median, and percentiles.

        Compilation and profiling are excluded. Inputs must already be staged; a
        positive repeat count guarantees the returned value is the final device result."""
        if type(repeats) is not int or repeats < 1:
            raise ValueError("repeats must be positive")
        times = []
        result = None
        for _ in range(repeats):
            start = time.perf_counter()
            result = self(*arguments)
            times.append(time.perf_counter() - start)
        _write(
            self.output / "timing.json",
            {
                "seconds": times,
                "median": float(np.median(times)),
                "p10": float(np.percentile(times, 10)),
                "p90": float(np.percentile(times, 90)),
                "synchronized": True,
                "includes_compilation": False,
                "includes_profiling": False,
            },
        )
        return cast(OutputT, result)


def compile_program(
    function: Callable[..., OutputT],
    arguments: tuple[object, ...],
    *,
    policy: ResidentPolicy | None = None,
    out: PathLike,
    name: str = "episode",
) -> CompiledProgram[OutputT]:
    """Transform the WHOLE raw function before calling this compiler boundary.

    For gradients pass jax.value_and_grad(raw_loss), not a compiled forward
    executable. For populations apply vmap first. Shape changes require a new
    compilation and a new certificate. No trace result is reused by filename.
    """
    policy = ResidentPolicy() if policy is None else policy
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "name": name,
        "policy": policy.descriptor(),
        "jax": jax.__version__,
        "python": platform.python_version(),
        "compiled": False,
        "runtime_inventory": _runtime_inventory(),
    }
    started = time.perf_counter()
    try:
        devices = jax.devices("gpu" if policy.device == "cuda" else "cpu")
        if not devices:
            raise ResidencyError("Requested device unavailable; no fallback")
        device = devices[0]
        if policy.device == "cuda":
            description = (str(device) + " " + str(getattr(device, "device_kind", ""))).upper()
            if not any(tag in description for tag in ("NVIDIA", "CUDA", "RTX", "TESLA")):
                raise ResidencyError("CUDA policy requires an identified NVIDIA device")
        manifest["device"] = str(device)
        manifest["runtime_platform_version"] = getattr(device.client, "platform_version", None)
        with jax.default_device(device):
            args = jax.device_put(arguments, device)
            jax.block_until_ready(args)
            abstract = jax.eval_shape(function, *args)
            output_shape = _shapes(abstract)
            output_bytes = sum(x["bytes"] for x in output_shape)
            manifest.update(
                input_shapes=_shapes(args), output_shapes=output_shape, output_bytes=output_bytes
            )
            if output_bytes > policy.max_recording_bytes:
                raise ResidencyError(
                    f"Output requires {output_bytes} bytes, budget is {policy.max_recording_bytes}; "
                    "increase sample stride/budget explicitly; no chunked/host-offload fallback"
                )
            audit = _jaxpr_audit(jax.make_jaxpr(function)(*args))
            manifest["jaxpr_audit"] = audit
            if not audit["passed"]:
                raise ResidencyError(
                    "Host callback/effect in simulation: " + str(audit["forbidden"])
                )
            eligibility = capture_eligibility(audit["primitive_counts"])
            manifest["capture_eligibility"] = eligibility
            if policy.device == "cuda" and not eligibility["passed"]:
                raise ResidencyError("Static capture obstacle: " + "; ".join(eligibility["errors"]))
            jit = jax.jit(function, compiler_options=policy.compiler_options())
            lower_started = time.perf_counter()
            lowered = jit.lower(*args)
            manifest["lowering_seconds"] = time.perf_counter() - lower_started
            (out / "program.stablehlo.mlir").write_text(str(lowered.compiler_ir()))
            compile_started = time.perf_counter()
            executable = lowered.compile()
            manifest["compile_or_cache_seconds"] = time.perf_counter() - compile_started
            optimized_hlo = executable.as_text()
            (out / "program.optimized.hlo.txt").write_text(optimized_hlo)
            eligibility = capture_eligibility(audit["primitive_counts"], optimized_hlo)
            manifest["capture_eligibility"] = eligibility
            if policy.device == "cuda" and not eligibility["passed"]:
                raise ResidencyError(
                    "Compiled capture obstacle: " + "; ".join(eligibility["errors"])
                )
            memory = executable.memory_analysis()
            manifest["compiler_memory"] = (
                {
                    key: getattr(memory, key)
                    for key in dir(memory)
                    if key.endswith("_size_in_bytes") and not key.startswith("_")
                }
                if memory is not None
                else None
            )
            _check_host_workspace(
                cast("dict[str, int] | None", manifest["compiler_memory"]), policy
            )
        manifest["compiled"] = True
        return CompiledProgram(executable, device, policy, out, manifest)
    except Exception as error:
        manifest["exception"] = repr(error)
        raise
    finally:
        manifest["setup_seconds"] = time.perf_counter() - started
        _write(out / "compilation.json", manifest)
