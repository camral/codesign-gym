"""Whole-episode semantics, gradients, recording, and fail-closed certificates."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from envs.soft_walker.engine.drivers.environment import BodyEnv
    from envs.soft_walker.engine.drivers.spring_environment import Spring2DEnv
    from envs.soft_walker.engine.execution.program import EpisodeProgram
    from envs.soft_walker.engine.types import Embedding


import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from soft_paths import needs_scripts

from envs.soft_walker.engine.execution import ResidencyError, ResidentPolicy, compile_program
from envs.soft_walker.engine.execution.compiled import _jaxpr_audit
from envs.soft_walker.engine.execution.program import bind_episode, stage_embedding
from envs.soft_walker.engine.execution.trace import MARKER, inspect_trace


def trace_fixture(
    extra: Sequence[dict[str, object]] = (), *, launch: str = "cuGraphLaunch (CudaGraph:123)"
) -> dict[str, object]:
    return {
        "traceEvents": [
            {"ph": "M", "pid": 1, "name": "process_name", "args": {"name": "/device:GPU:0"}},
            {"ph": "M", "pid": 7, "name": "process_name", "args": {"name": "/host:CPU"}},
            {"ph": "X", "pid": 7, "name": MARKER, "ts": 10.0, "dur": 100.0},
            {
                "ph": "X",
                "pid": 7,
                "name": launch,
                "ts": 11.0,
                "dur": 2.0,
                "args": {"context_id": "1", "correlation_id": "11"},
            },
            {
                "ph": "X",
                "pid": 1,
                "name": "a_device_kernel",
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
            *extra,
        ]
    }


def test_trace_certificate_recognizes_decorated_api_and_excludes_warmup() -> None:
    trace = trace_fixture([{"ph": "X", "pid": 7, "name": "While", "ts": 1.0, "dur": 2.0}])
    audit = inspect_trace(trace)
    assert audit["passed"]
    assert audit["counts"]["host_graph_submissions"] == 1
    assert audit["counts"]["host_while_checks"] == 0


@pytest.mark.parametrize(
    "pid,name",
    [
        (7, "While"),
        (7, "Conditional"),
        (7, "cuGraphLaunch (CudaGraph:456)"),
        (7, "cuLaunchKernel_v2"),
        (7, "cudaLaunchKernelExC"),
        (1, "MemcpyD2H"),
        (1, "MemcpyH2D"),
        (7, "MemcpyDtoH"),
        (7, "xla_python_cpu_callback"),
        (7, "cuGraphCreate (CudaGraph:456)"),
    ],
)
def test_trace_certificate_rejects_internal_host_work(pid: int, name: str) -> None:
    extra = {"ph": "X", "pid": pid, "name": name, "ts": 20.0, "dur": 1.0}
    assert not inspect_trace(trace_fixture([extra]))["passed"]


def test_missing_trace_is_not_a_pass() -> None:
    assert not inspect_trace({"traceEvents": []})["passed"]
    assert not inspect_trace(trace_fixture(launch="not_instrumented"))["passed"]
    trace = trace_fixture()
    trace["traceEvents"] = [e for e in trace["traceEvents"] if e.get("name") != MARKER]
    assert not inspect_trace(trace)["passed"]


def test_effect_and_pure_callback_audit() -> None:
    def debug(x: jax.Array) -> jax.Array:
        jax.debug.print("forbidden {}", x)
        return x * x

    def callback(x: jax.Array) -> jax.Array:
        return jax.pure_callback(lambda a: np.asarray(a), jax.ShapeDtypeStruct((), x.dtype), x)

    x = jnp.asarray(0.2)
    assert not _jaxpr_audit(jax.make_jaxpr(debug)(x))["passed"]
    assert not _jaxpr_audit(jax.make_jaxpr(callback)(x))["passed"]
    assert _jaxpr_audit(jax.make_jaxpr(lambda v: v**2)(x))["passed"]


def test_cpu_compilation_boundary_requires_cert_and_device_inputs(
    tmp_path: Path, dtype: np.dtype[np.floating]
) -> None:
    if jax.default_backend() != "cpu":
        pytest.skip("CPU-only certificate fixture; CUDA boundary tested by actual physical workers")
    x = jnp.asarray(0.3, dtype)
    raw = lambda v: jax.lax.fori_loop(0, 5, lambda i, y: y + jnp.sin(y), v)
    program = compile_program(
        jax.value_and_grad(raw), (x,), policy=ResidentPolicy("cpu"), out=tmp_path / "compiled"
    )
    with pytest.raises(ResidencyError):
        program(x)
    program.certify(x)
    for v in (0.3, -0.2, 0.7):
        inputs = program.stage(np.asarray(v, dtype))
        actual = program(*inputs)
        expected = jax.value_and_grad(raw)(jnp.asarray(v, dtype))
        for a, b in zip(actual, expected):
            np.testing.assert_allclose(a, b, rtol=2e-5, atol=2e-7)
    with pytest.raises(ResidencyError):
        program(np.asarray(0.3, dtype))
    certificate = json.loads((tmp_path / "compiled/certificate.json").read_text())
    assert certificate["passed"] and not certificate["gpu_residency_demonstrated"]


def test_recording_budget_rejects_instead_of_host_streaming(tmp_path: Path) -> None:
    if jax.default_backend() != "cpu":
        pytest.skip("CPU memory-budget fixture; rejection logic is backend-independent")
    with pytest.raises(ResidencyError, match="no chunked"):
        compile_program(
            lambda x: jnp.ones((100,), x.dtype),
            (jnp.asarray(0.2),),
            policy=ResidentPolicy("cpu", max_recording_bytes=8),
            out=tmp_path / "large",
        )
    assert (tmp_path / "large/compilation.json").exists()


def test_policy_has_whole_control_flow_and_no_host_dynamic_slice() -> None:
    flags = ResidentPolicy().compiler_options()
    assert "WHILE" in flags["xla_gpu_enable_command_buffer"]
    assert "CONDITIONAL" in flags["xla_gpu_enable_command_buffer"]
    assert "DYNAMIC_SLICE" not in flags["xla_gpu_enable_command_buffer"]
    assert not flags["xla_gpu_enable_dynamic_slice_fusion"]
    assert flags["xla_enable_command_buffers_during_profiling"]
    assert not flags["xla_gpu_command_buffer_unroll_loops"]


def test_deterministic_policy_preserves_residency_options_and_cpu_behavior() -> None:
    base = ResidentPolicy().compiler_options()
    policy = ResidentPolicy(deterministic_reductions=True)
    options = policy.compiler_options()
    assert all(options[key] == value for key, value in base.items())
    assert options["xla_gpu_deterministic_ops"] is True
    assert options["xla_gpu_enable_scatter_determinism_expander"] is True
    assert policy.descriptor()["deterministic_reductions"] is True
    assert ResidentPolicy("cpu", deterministic_reductions=True).compiler_options() == {}
    with pytest.raises(ValueError):
        ResidentPolicy(deterministic_reductions="yes")


def physical_case(
    driver: str,
    dtype: np.dtype[np.floating] | type[np.floating],
    steps: int = 3,
    *,
    gmres_method: str = "fused",
) -> tuple[Callable, Callable, np.ndarray, Spring2DEnv | BodyEnv, Embedding, EpisodeProgram]:
    from envs.soft_walker.diagnostics.m3_cases import diagnostic_value, driver_case, spring_case
    from envs.soft_walker.engine.drivers.model_access import task_positions as positions

    data = (
        spring_case(dtype, steps, gmres_method=gmres_method)
        if driver == "spring"
        else driver_case(driver, dtype, steps=steps, gmres_method=gmres_method)
    )
    old_loss, old_run, theta, metadata, env, embedding = data
    action_offset = 4 if driver == "spring" else 5
    program = bind_episode(
        env,
        steps=steps,
        action=lambda m, s, p, i: jnp.stack(
            (p[action_offset] * (1 - 0.1 * i), p[action_offset + 1] * (1 - 0.08 * i))
        ),
        measure=lambda m, s: jnp.stack(
            (s.time, diagnostic_value(s.body if hasattr(s, "body") else s, driver))
        ),
        sample=lambda m, s: {"position": positions(env, m, s)},
        terminal_loss=lambda m, initial, s, row, c: row[1],
        checkpoint_block=2,
        record_stride=2,
    )
    return old_loss, old_run, theta, env, embedding, program


@needs_scripts
@pytest.mark.parametrize("driver", ["spring", "mpm2", "mpm3", "fem", "beam"])
def test_whole_episode_matches_original_settling_and_gradient(
    driver: str, dtype: np.dtype[np.floating]
) -> None:
    old_loss, old_run, theta, env, embedding, program = physical_case(driver, dtype)
    try:
        raw = lambda p: program.loss(embedding(p), p)
        actual = jax.jit(jax.value_and_grad(raw))(theta)
        reference = jax.jit(jax.value_and_grad(old_loss))(theta)
        for a, b in zip(actual, reference):
            assert np.isfinite(a).all()
            np.testing.assert_allclose(
                a,
                b,
                rtol=1e-8 if dtype.name == "float64" else 5e-4,
                atol=1e-10 if dtype.name == "float64" else 1e-6,
            )
        recorded = jax.jit(program.record)(embedding(theta), theta)
        final = jax.jit(old_run)(theta)["state"]
        assert bool(recorded["valid"])
        for a, b in zip(jax.tree.leaves(recorded["state"]), jax.tree.leaves(final)):
            np.testing.assert_allclose(a, b, rtol=3e-5, atol=2e-7)
        np.testing.assert_array_equal(recorded["history"]["frame_indices"], [0, 2, 3])
        assert int(recorded["history"]["frames_written"]) == 3
        np.testing.assert_allclose(
            recorded["history"]["metrics"][:, 0], np.arange(4) * 0.001, atol=1e-8
        )
        np.testing.assert_allclose(recorded["loss"], actual[0], rtol=5e-5, atol=1e-7)
    finally:
        env.close()
        # Independent physical families are large; retaining their executables
        # is unrelated to production performance and can exhaust small hosts.
        jax.clear_caches()


@needs_scripts
def test_population_gradients_are_independent_and_invalid_candidate_visible(
    dtype: np.dtype[np.floating],
) -> None:
    old_loss, old_run, theta, env, embedding, program = physical_case("mpm2", dtype, 2)
    try:
        base = embedding(theta)
        population = jnp.stack((theta, theta.at[-1].add(0.02)))
        actual = jax.jit(program.population(gradients=True))(base, population)
        scalar = jax.jit(jax.value_and_grad(program.loss, argnums=1))
        for i in range(2):
            expected = scalar(base, population[i])
            for a, b in zip(actual, expected):
                np.testing.assert_allclose(
                    a[i], b, rtol=1e-7 if dtype.name == "float64" else 5e-4, atol=1e-7
                )
        invalid = jax.jit(program.forward)(base, theta.at[-1].set(jnp.nan))
        assert not bool(invalid["valid"]) and np.isnan(invalid["loss"])
    finally:
        env.close()
        jax.clear_caches()


@pytest.mark.parametrize("representation", ["primitives", "direct", "bspline", "cppn"])
def test_composite_decoder_pure_validation_only_checks_exposed_edges(
    representation: str, dtype: np.dtype[np.floating]
) -> None:
    from envs.soft_walker.engine.differentiation import compiled_design_valid
    from envs.soft_walker.presets.scenarios.build import build
    from envs.soft_walker.presets.scenarios.config import Scenario

    case = Scenario(tier="smoke", representation=representation, dtype=dtype.name)
    device = "cuda" if jax.default_backend() in ("gpu", "cuda") else "cpu"
    env, embodiment = build(case, device=device)
    try:
        e, m = stage_embedding(env, embodiment, seed=case.seed)
        # No time integration needed to check this schema regression.
        points = env._domain.points
        assert bool(
            jax.jit(
                lambda values: compiled_design_valid(
                    env, values, env.decoder.decode(values, points)
                )
            )(e)
        )
    finally:
        env.close()
        jax.clear_caches()


@pytest.mark.parametrize(
    "name",
    [
        "cuLaunchKernel",
        "cuLaunchKernelEx",
        "cuLaunchCooperativeKernel_ptsz",
        "cuMemcpyDtoHAsync_v2",
        "cuMemcpyHtoD_v2",
        "cuMemcpyDtoDAsync_v2",
        "cuMemsetD32Async",
        "cuGraphAddHostNode",
        "cuLaunchHostFunc",
    ],
)
def test_driver_callback_audit_rejects_host_simulation_operations(name: str) -> None:
    from envs.soft_walker.engine.execution.cupti import inspect_api_counts

    assert not inspect_api_counts({"cuGraphLaunch_ptsz": 1, name: 1})["passed"]


def test_driver_callback_audit_empty_multiple_and_observer_failures() -> None:
    from envs.soft_walker.engine.execution.cupti import inspect_api_counts

    assert not inspect_api_counts({})["passed"]
    assert not inspect_api_counts({"cuGraphLaunch": 2})["passed"]
    assert not inspect_api_counts({"cuGraphLaunch": 1}, ["collector exception"])["passed"]
    assert inspect_api_counts({"cuGraphLaunch_ptsz": 1, "cuEventSynchronize": 1})["passed"]


def test_callback_abi_lifetime_entry_exit_and_cleanup() -> None:
    """Exercise the ctypes callback bridge without claiming a CUDA execution."""
    import ctypes as ct

    from envs.soft_walker.engine.execution.cupti import DriverAudit, _CallbackPrefix, _GraphLaunch

    class Function:
        def __init__(self, fn: Callable[..., object]) -> None:
            self.fn = fn

        def __call__(self, *args: object) -> object:
            return self.fn(*args)

    class Library:
        def __init__(self) -> None:
            self.callback = None
            self.enabled = False
            self.unsubscribed = False
            self.cuptiSubscribe = Function(self.subscribe)
            self.cuptiEnableDomain = Function(self.enable)
            self.cuptiUnsubscribe = Function(self.unsubscribe)
            self.cuptiGetVersion = Function(self.version)
            self.cuptiGetResultString = Function(lambda result, text: 0)

        def subscribe(self, handle: object, callback: Callable[..., None], userdata: object) -> int:
            ct.cast(handle, ct.POINTER(ct.c_void_p))[0] = ct.c_void_p(123)
            self.callback = callback
            return 0

        def enable(self, value: int, handle: object, domain: int) -> int:
            assert domain == 1
            self.enabled = bool(value)
            return 0

        def unsubscribe(self, handle: object) -> int:
            self.unsubscribed = True
            self.callback = None
            return 0

        def version(self, value: object) -> int:
            ct.cast(value, ct.POINTER(ct.c_uint32))[0] = 99
            return 0

        def emit(self, name: str, site: int = 0) -> None:
            assert self.enabled
            parameters = _GraphLaunch(0x1230, 0x4560)
            status = ct.c_int(0)
            prefix = _CallbackPrefix(
                site,
                name.encode("ascii"),
                ct.addressof(parameters),
                ct.addressof(status),
                None,
                0x7890,
            )
            self.callback(None, 1, 7, ct.byref(prefix))

    library = Library()
    observer = DriverAudit(library=library)
    with observer:
        library.emit("cuGraphLaunch_ptsz")
        library.emit("cuGraphLaunch_ptsz", site=1)  # EXIT must not double count.
        for _ in range(200):
            library.emit("cuEventQuery")
            library.emit("cuEventQuery", site=1)
    report = observer.report()
    assert report["passed"] and report["counts"]["api_entries"] == 201
    assert len(report["first_api_names"]) == 128  # Counters remain complete.
    assert library.unsubscribed and not library.enabled

    library = Library()
    broken = DriverAudit(library=library)
    with pytest.raises(RuntimeError):
        with broken:
            library.callback(None, 1, 7, None)
            raise RuntimeError("simulation failed")
    assert library.unsubscribed and not broken.report()["passed"]


def test_host_intermediate_workspace_cannot_bypass_transfer_guards() -> None:
    from envs.soft_walker.engine.execution.compiled import _check_host_workspace

    empty = {
        name: 0
        for name in (
            "host_argument_size_in_bytes",
            "host_output_size_in_bytes",
            "host_temp_size_in_bytes",
            "host_alias_size_in_bytes",
        )
    }
    _check_host_workspace(empty, ResidentPolicy("cuda"))
    for name in empty:
        with pytest.raises(ResidencyError, match="offload"):
            _check_host_workspace(dict(empty, **{name: 1}), ResidentPolicy("cuda"))
    with pytest.raises(ResidencyError, match="accounting"):
        _check_host_workspace(None, ResidentPolicy("cuda"))
    _check_host_workspace(None, ResidentPolicy("cpu"))


@needs_scripts
def test_original_profiler_summary_counts_decorated_graph_launch(tmp_path: Path) -> None:
    import gzip

    from envs.soft_walker.diagnostics.audit_krylov_physics import summarize_trace

    path = tmp_path / "short.trace.json.gz"
    with gzip.open(path, "wt") as stream:
        json.dump(trace_fixture(), stream)
    assert summarize_trace(path)["graph_launches"] == 1


def test_demo_assessment_uses_poststep_rows_not_initial_sample() -> None:
    from types import SimpleNamespace

    from envs.soft_walker.experiments.locomotion.rollout import gait_assessment
    from envs.soft_walker.scripts._resident_commands import _assess_metrics

    demo = SimpleNamespace(steps=12, steps_per_cycle=2, dim=2, cell_size=0.025, size=(0.15, 0.15))
    rows = np.zeros((13, 13))
    rows[:, 0] = np.arange(13) * 0.02
    rows[:, 1] = np.arange(13) * 0.025
    rows[:, 7] = 0.1
    rows[:, 9:11] = (np.arange(13) % 2 * 0.005)[:, None]
    result = _assess_metrics(rows, demo, "demo")
    assert result == gait_assessment(rows[1:], demo)
    assert result["valid_rollout"] and result["walking_gate_passed"]
