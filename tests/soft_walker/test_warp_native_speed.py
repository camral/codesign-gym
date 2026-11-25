"""New algorithm checks; WARP_TEST_DEVICE=cuda exercises whole-graph replay."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

    from envs.soft_walker.engine.warp_native.episode import Episode


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


import os
from dataclasses import replace

import numpy as np
import pytest

wp = pytest.importorskip("warp")
from warp.optim.linear import LinearOperator

from envs.soft_walker.engine.warp_native.adaptive_gmres import AdaptiveGMRES

PRECISION = os.environ.get("WARP_TEST_DTYPE", "float64")
REAL = wp.float64 if PRECISION == "float64" else wp.float32
DEVICE = os.environ.get("WARP_TEST_DEVICE", "cpu")


@wp.kernel(enable_backward=False)
def multiply(
    matrix: wp.array2d(dtype=REAL),
    transpose: int,
    x: wp.array(dtype=REAL),
    y: wp.array(dtype=REAL),
    alpha: REAL,
    beta: REAL,
    out: wp.array(dtype=REAL),
):
    i = wp.tid()
    value = REAL(0)
    for j in range(x.shape[0]):
        coefficient = matrix[j, i] if transpose else matrix[i, j]
        value = value + coefficient * x[j]
    old = REAL(0)
    if beta != REAL(0):
        old = beta * y[i]
    out[i] = alpha * value + old


@wp.kernel(enable_backward=False)
def diagonal(
    matrix: wp.array2d(dtype=REAL),
    x: wp.array(dtype=REAL),
    y: wp.array(dtype=REAL),
    alpha: REAL,
    beta: REAL,
    out: wp.array(dtype=REAL),
):
    i = wp.tid()
    old = REAL(0)
    if beta != REAL(0):
        old = beta * y[i]
    out[i] = alpha * x[i] / matrix[i, i] + old


def setup(
    matrix: "np.ndarray", restart: "int" = 20, maxiter: "int" = 240, precondition: "bool" = False
) -> "Callable[..., tuple[np.ndarray, np.ndarray, float]]":
    device = wp.get_device(DEVICE)
    matrix = np.asarray(matrix, dtype=PRECISION)
    a = wp.array(matrix, dtype=REAL, device=device)
    n = len(matrix)
    b = wp.zeros(n, dtype=REAL, device=device)
    x = wp.zeros_like(b)

    def operator(transpose: "bool | int") -> "LinearOperator":
        def matvec(
            v: "wp.array", y: "wp.array", z: "wp.array", alpha: "float" = 1.0, beta: "float" = 0.0
        ) -> "None":
            wp.launch(
                multiply, n, inputs=[a, transpose, v, y, REAL(alpha), REAL(beta), z], device=device
            )

        return LinearOperator((n, n), REAL, device, matvec)

    def p(
        v: "wp.array", y: "wp.array", z: "wp.array", alpha: "float" = 1.0, beta: "float" = 0.0
    ) -> "None":
        wp.launch(diagonal, n, inputs=[a, v, y, REAL(alpha), REAL(beta), z], device=device)

    M = LinearOperator((n, n), REAL, device, p) if precondition else None
    tol = 1e-11 if PRECISION == "float64" else 2e-6
    solver = AdaptiveGMRES(operator(0), b, x, restart=restart, maxiter=maxiter, tol=tol, M=M)

    def execute(transpose: "bool | int" = False) -> "None":
        x.zero_()
        solver(A=operator(int(transpose)))

    graphs = {}
    if device.is_cuda:
        for transpose in (False, True):
            with wp.ScopedCapture(device=device, force_module_load=True) as capture:
                execute(transpose)
            graphs[transpose] = capture.graph

    def run(
        rhs: "np.ndarray", transpose: "bool | int" = False
    ) -> "tuple[np.ndarray, np.ndarray, float] | tuple[Episode, dict[str, object]]":
        b.assign(np.asarray(rhs, PRECISION))
        if device.is_cuda:
            wp.capture_launch(graphs[transpose])
        else:
            execute(transpose)
        wp.synchronize_device(device)
        return x.numpy(), solver.state.numpy(), float(solver.dot.col(0).numpy()[0])

    return run


@pytest.mark.parametrize("precondition", [False, True])
def test_adaptive_diagonal_stops_without_building_full_restart(precondition: "bool") -> None:
    matrix = np.eye(32) * 2.0
    run = setup(matrix, precondition=precondition)
    solution, state, error = run(np.ones(32))
    np.testing.assert_allclose(solution, 0.5, rtol=1e-6, atol=1e-7)
    assert state[0] == 1
    assert state[3] == 1
    assert error < 1e-12


@pytest.mark.parametrize("restart", [2, 8])
@pytest.mark.parametrize("precondition", [False, True])
def test_adaptive_nonsymmetric_transpose_and_reuse(restart: "int", precondition: "bool") -> None:
    rng = np.random.default_rng(120)
    matrix = np.diag(np.linspace(2.0, 4.0, 12)) + 0.1 * rng.normal(size=(12, 12))
    run = setup(matrix, restart=restart, precondition=precondition)
    tolerance = 5e-10 if PRECISION == "float64" else 2e-5
    for rhs in [rng.normal(size=12), np.zeros(12), rng.normal(size=12)]:
        for transpose in (False, True):
            solution, state, error = run(rhs, transpose)
            reference_matrix = matrix.T if transpose else matrix
            np.testing.assert_allclose(
                solution, np.linalg.solve(reference_matrix, rhs), rtol=tolerance, atol=tolerance
            )
            if np.all(rhs == 0):
                assert state[0] == 0
            assert np.linalg.norm(reference_matrix @ solution - rhs) <= tolerance * max(
                np.linalg.norm(rhs), 1.0
            )
            assert state[0] <= 240


def test_adaptive_budget_and_singular_failure_are_visible() -> None:
    rng = np.random.default_rng(2)
    matrix = rng.normal(size=(8, 8)) + np.diag(np.arange(1.0, 9.0))
    run = setup(matrix, restart=2, maxiter=1)
    solution, state, error = run(np.ones(8))
    assert state[0] == 1
    assert np.linalg.norm(matrix @ solution - 1) > 1e-3
    assert error > 1e-6
    run = setup(np.zeros((8, 8)), restart=2)
    solution, state, error = run(np.ones(8))
    assert state[2] != 0
    assert error > 1
    assert np.all(np.isfinite(solution))


def test_adaptive_invalid_method_rejected() -> None:
    from envs.soft_walker.engine.warp_native.config import MPMConfig

    with pytest.raises(ValueError, match="krylov_method"):
        MPMConfig(krylov_method="fast-but-inexact")


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("law", ["active_strain", "active_stress"])
@pytest.mark.parametrize("terrain", ["plane", "slope", "box"])
def test_cached_tangent_matches_original_and_refreshes(
    dim: "int", law: "str", terrain: "str"
) -> None:
    from envs.soft_walker.diagnostics.warp.checks import contact_fixture
    from envs.soft_walker.engine.warp_native.model import DeviceModel
    from envs.soft_walker.engine.warp_native.step import ImplicitStep

    bundle = contact_fixture(dim, PRECISION, law, terrain)
    bundle.config = replace(bundle.config, krylov_method="adaptive", cache_tangent=True)
    owner = DeviceModel(bundle, DEVICE)
    step = ImplicitStep(owner)
    step.command.assign(np.array([0.4], PRECISION))
    rng = np.random.default_rng(94 + dim)
    direction = wp.array(
        rng.normal(size=(step.capacity, dim)).astype(PRECISION),
        dtype=owner.k.vec,
        device=owner.device,
    )
    tangent = wp.zeros_like(direction)
    transpose_tangent = wp.zeros_like(direction)
    old_tangent = wp.zeros_like(direction)
    old_transpose = wp.zeros_like(direction)

    def program() -> "None":
        step.prepare()
        wp.copy(step.q, step.guess)
        step.refresh_linearization()
        step.apply_tangent(direction, tangent, tangent, 1.0, 0.0, False)
        step.apply_tangent(direction, transpose_tangent, transpose_tangent, 1.0, 0.0, True)
        saved = step.linearization
        step.linearization = None  # independent original kernels at capture time
        step.apply_tangent(direction, old_tangent, old_tangent, 1.0, 0.0, False)
        step.apply_tangent(direction, old_transpose, old_transpose, 1.0, 0.0, True)
        step.linearization = saved

    if owner.device.is_cuda:
        with wp.ScopedCapture(device=owner.device, force_module_load=True) as capture:
            program()
    tolerance = 5e-13 if PRECISION == "float64" else 5e-6
    for scale in (1.0, 1.3):
        owner.model.young.assign((scale * bundle.fields["young"]).astype(PRECISION))
        step.command.assign(np.array([0.4 * scale], PRECISION))
        if owner.device.is_cuda:
            wp.capture_launch(capture.graph)
        else:
            program()
        wp.synchronize_device(owner.device)
        for a, b in [(tangent, old_tangent), (transpose_tangent, old_transpose)]:
            np.testing.assert_allclose(a.numpy(), b.numpy(), rtol=tolerance, atol=tolerance)
        assert step.linearization.bytes > 0


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("law", ["active_strain", "active_stress"])
def test_whole_episode_values_and_gradients_match_legacy(dim: "int", law: "str") -> None:
    from envs.soft_walker.diagnostics.warp.checks import (
        _validate_execution,
        compare_array,
        contact_fixture,
    )
    from envs.soft_walker.engine.warp_native.episode import Episode
    from envs.soft_walker.engine.warp_native.model import DeviceModel

    bundle = contact_fixture(dim, PRECISION, law)
    commands = np.array([[0.4], [-0.2]], PRECISION)

    def run(
        method: "str", cache: "bool"
    ) -> "tuple[np.ndarray, np.ndarray, float] | tuple[Episode, dict[str, object]]":
        bundle.config = replace(bundle.config, krylov_method=method, cache_tangent=cache)
        episode = Episode(
            DeviceModel(bundle, DEVICE),
            commands,
            settling_steps=1,
            checkpoint_stride=2,
            velocity_weight=0.1,
            work_weight=0.05,
        )
        _validate_execution(episode.run(gradient=True), gradient=True, phase="speed parity")
        return episode, episode.gradients()

    old, old_g = run("legacy", False)
    new, new_g = run("adaptive", True)
    compare_array(new.loss.numpy(), old.loss.numpy(), name="loss", precision=PRECISION)
    for name in ("x", "v", "F", "C", "actuator", "ledger"):
        compare_array(
            getattr(new.final, name).numpy(),
            getattr(old.final, name).numpy(),
            name="state_" + name,
            precision=PRECISION,
        )
    compare_array(
        new_g["commands"], old_g["commands"], name="gradient_commands", precision=PRECISION
    )
    for group in ("initial", "model"):
        for name in old_g[group]:
            if name != "status":
                compare_array(
                    new_g[group][name],
                    old_g[group][name],
                    name="gradient_" + group + "_" + name,
                    precision=PRECISION,
                )


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("stride", [1, 2, 4])
def test_replay_root_cache_preserves_reused_episode_gradients(dim: "int", stride: "int") -> None:
    from envs.soft_walker.diagnostics.warp.checks import (
        _validate_execution,
        compare_array,
        contact_fixture,
    )
    from envs.soft_walker.engine.warp_native.episode import Episode
    from envs.soft_walker.engine.warp_native.model import DeviceModel

    bundle = contact_fixture(dim, PRECISION)
    bundle.config = replace(bundle.config, krylov_method="adaptive", cache_tangent=True)
    commands = np.array([[0.4], [-0.2]], PRECISION)
    episodes = [
        Episode(
            DeviceModel(bundle, DEVICE),
            commands,
            settling_steps=1,
            checkpoint_stride=stride,
            velocity_weight=0.1,
            work_weight=0.05,
            reuse_replay_roots=reuse,
        )
        for reuse in (False, True)
    ]
    assert episodes[1].history_bytes == episodes[0].history_bytes + episodes[1].root_cache_bytes
    for multiplier in (1.0, -0.7):
        results = []
        for episode in episodes:
            episode.set_commands(multiplier * commands)
            _validate_execution(
                episode.run(gradient=True), gradient=True, phase="root-cache replay"
            )
            results.append(episode.gradients())
        for group in ("initial", "model"):
            for name in results[0][group]:
                compare_array(
                    results[1][group][name],
                    results[0][group][name],
                    name="gradient_" + group + "_" + name,
                    precision=PRECISION,
                )
        compare_array(
            results[1]["commands"],
            results[0]["commands"],
            name="gradient_commands",
            precision=PRECISION,
        )
        a, b = [e.step.diagnostics()["work_counts"] for e in episodes]
        assert b["forward_tangent"] < a["forward_tangent"]
        assert b["transpose_tangent"] == a["transpose_tangent"]


def test_root_checkpoint_memory_budget_is_enforced() -> None:
    from envs.soft_walker.diagnostics.warp.checks import contact_fixture
    from envs.soft_walker.engine.warp_native.episode import Episode
    from envs.soft_walker.engine.warp_native.model import DeviceModel

    bundle = contact_fixture(2, PRECISION)
    owner = DeviceModel(bundle, DEVICE)
    reference = Episode(owner, np.zeros((3, 1), PRECISION), checkpoint_stride=3)
    budget = (reference.history_bytes + 1) / 1024**2
    with pytest.raises(MemoryError, match="root checkpoints"):
        Episode(
            owner,
            np.zeros((3, 1), PRECISION),
            checkpoint_stride=3,
            reuse_replay_roots=True,
            memory_mib=budget,
        )


def _worker_module() -> "ModuleType":
    """Import the argument-only helper without initializing a simulation."""
    import importlib.util

    from soft_paths import SOFT_ROOT

    path = SOFT_ROOT / "diagnostics" / "warp_speed_worker.py"
    spec = importlib.util.spec_from_file_location("warp_speed_worker_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_export_protocol_preserves_duration_and_settling(tmp_path: "Path") -> None:
    import json
    from types import SimpleNamespace

    np.save(tmp_path / "commands.npy", np.zeros((7, 4)))
    (tmp_path / "protocol.json").write_text(json.dumps({"settling_steps": 80, "substeps": 40}))
    args = SimpleNamespace(
        bundle=tmp_path / "model.npz", commands=None, steps=None, settling=None, substeps=None
    )
    result = _worker_module().resolve_protocol(args)
    assert (result.steps, result.settling, result.substeps) == (7, 80, 40)


def test_export_protocol_missing_information_or_truncation_rejected(tmp_path: "Path") -> None:
    from types import SimpleNamespace

    np.save(tmp_path / "commands.npy", np.zeros((7, 4)))
    args = SimpleNamespace(
        bundle=tmp_path / "model.npz", commands=None, steps=None, settling=None, substeps=None
    )
    module = _worker_module()
    with pytest.raises(ValueError, match="protocol.json"):
        module.resolve_protocol(args)
    args = SimpleNamespace(
        bundle=tmp_path / "model.npz", commands=None, steps=6, settling=0, substeps=1
    )
    with pytest.raises(ValueError, match="whole command table"):
        module.resolve_protocol(args)


def test_default_fixture_protocol_is_short() -> None:
    from types import SimpleNamespace

    args = SimpleNamespace(bundle=None, commands=None, steps=None, settling=None, substeps=None)
    result = _worker_module().resolve_protocol(args)
    assert (result.steps, result.settling, result.substeps) == (3, 1, 1)
