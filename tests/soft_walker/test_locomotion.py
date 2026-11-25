"""New locomotion infrastructure tests; not a substitute for frozen gait replays."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from envs.soft_walker.experiments.locomotion.rollout import gait_assessment
from envs.soft_walker.presets.gaits import seed_controller
from envs.soft_walker.presets.locomotion import DemoConfig, LeggedDecoder
from envs.soft_walker.shared import periodic_action


@pytest.mark.parametrize("dim,channels,count", [(2, 4, 28), (3, 16, 136)])
def test_legged_regions_and_uniform_volume(
    dim: int, channels: int, count: int, dtype: np.dtype[np.floating]
) -> None:
    shape = (6,) * dim
    indices = np.indices(shape).reshape(dim, -1).T
    points = jnp.asarray(2 * (indices + 0.5) / np.asarray(shape) - 1, dtype)
    decoder = LeggedDecoder(dim, dtype)
    emb = decoder.base()
    fields = decoder.decode(emb, points)
    assert int(fields.occupancy.sum()) == count
    assert decoder.space().contains(emb)
    active = np.asarray(fields.strength) > 0
    assert set(np.asarray(fields.channel)[active]) == set(range(channels))
    assert np.all(np.asarray(fields.actuator_type)[~active] == 0)
    np.testing.assert_allclose(np.asarray(fields.direction)[:, 1], 1)
    for factor in (2, 3):
        refined = np.indices(tuple(n * factor for n in shape)).reshape(dim, -1).T
        q = jnp.asarray(2 * (refined + 0.5) / (np.asarray(shape) * factor) - 1, dtype)
        f = decoder.decode(emb, q)
        assert int(f.occupancy.sum()) == count * factor**dim


@pytest.mark.parametrize("dim", [2, 3])
def test_controller_periodicity_bounding_and_population(
    dim: int, dtype: np.dtype[np.floating]
) -> None:
    p = jnp.asarray(seed_controller(dim, dtype=dtype))
    t = jnp.asarray(0.073, dtype)
    np.testing.assert_allclose(
        periodic_action(p, t, 0.5),
        periodic_action(p, t + 0.5, 0.5),
        atol=1e-6 if dtype == np.float32 else 1e-13,
    )
    assert np.max(np.abs(periodic_action(100 * p, t, 0.5))) <= 1
    batch = jnp.stack((p, p * 0.5, p * -0.75))
    got = jax.vmap(lambda a: periodic_action(a, t, 0.5))(batch)
    expected = jnp.stack([periodic_action(a, t, 0.5) for a in batch])
    np.testing.assert_array_equal(got, expected)


def test_compact_sine_equivalence(dtype: np.dtype[np.floating]) -> None:
    rng = np.random.default_rng(2)
    w = rng.normal(size=(4, 4)).astype(dtype)
    b = rng.normal(size=4).astype(dtype)
    p = np.stack((b, w[:, 0] - w[:, 2], w[:, 1] - w[:, 3]), axis=-1)
    for t in (0.0, 0.13, 0.4):
        expected = np.tanh(
            b + np.sum(w * np.sin(2 * np.pi * t / 0.5 + np.arange(4) * np.pi / 2), axis=-1)
        )
        np.testing.assert_allclose(
            periodic_action(jnp.asarray(p), t, 0.5),
            expected,
            atol=5e-7 if dtype == np.float32 else 2e-14,
        )


@pytest.mark.parametrize(
    "kw",
    [
        dict(period=0.51),
        dict(cycles=0),
        dict(friction=-1),
        dict(control_dt=0),
        dict(grid_margin=0),
        dict(driver="bogus"),
    ],
)
def test_invalid_protocol_rejected(kw: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        DemoConfig(**kw)


def test_gait_gate_does_not_call_small_motion_walking() -> None:
    cfg = DemoConfig(cycles=6)
    rows = np.zeros((cfg.steps, 13))
    rows[:, 0] = (np.arange(cfg.steps) + 1) * cfg.control_dt
    rows[:, 7] = 0.12
    rows[:, 1] = np.linspace(0, 0.001, cfg.steps)
    assert not gait_assessment(rows, cfg)["walking_gate_passed"]
    rows[:, 1] = np.linspace(0, 0.07, cfg.steps)
    assert not gait_assessment(rows, cfg)["walking_gate_passed"]  # feet never swing
    rows[::2, 9:11] = 0.004
    assert gait_assessment(rows, cfg)["walking_gate_passed"]
    rows[-1, 6] = 1
    assert not gait_assessment(rows, cfg)["walking_gate_passed"]


def test_optimizer_checkpoint_and_resume(tmp_path: Path, dtype: np.dtype[np.floating]) -> None:
    from envs.soft_walker.shared import optimize

    target = np.asarray([[0.2, -0.3, 0.5]], dtype)

    def quadratic(p: np.ndarray) -> tuple[np.floating | float, np.ndarray]:
        return np.sum((p - target) ** 2), 2 * (p - target)

    init = np.zeros_like(target)
    first = optimize(quadratic, init, tmp_path / "a", method="adam", iterations=4)
    assert first.best_loss < first.initial_loss
    resumed = optimize(
        quadratic, init, tmp_path / "b", method="adam", iterations=4, resume=tmp_path / "a/last.npz"
    )
    whole = optimize(quadratic, init, tmp_path / "c", method="adam", iterations=8)
    np.testing.assert_array_equal(resumed.parameters, whole.parameters)
    assert (tmp_path / "a/optimization.jsonl").exists()
    gd = optimize(quadratic, init, tmp_path / "gd", method="gd", iterations=3)
    assert gd.best_loss < gd.initial_loss
    lbfgs = optimize(quadratic, init, tmp_path / "lbfgs", method="lbfgs", iterations=5)
    assert lbfgs.best_loss < first.best_loss


@pytest.mark.parametrize("method", ["adam", "gd"])
def test_optimizer_iteration_callback_sees_saved_current_state(
    tmp_path: Path, dtype: np.dtype[np.floating], method: str
) -> None:
    from envs.soft_walker.shared import optimize

    target = np.asarray([[0.4, -0.6, 0.8]], dtype)
    initial = np.zeros_like(target)

    def quadratic(p: np.ndarray) -> tuple[np.floating | float, np.ndarray]:
        return np.sum((p - target) ** 2), 2 * (p - target)

    output = tmp_path / "callback"
    snapshots = []

    def observe(
        *, iteration: int, parameters: np.ndarray, loss: float, accepted: bool, backtracks: int
    ) -> None:
        with np.load(output / "last.npz", allow_pickle=False) as saved:
            assert int(saved["iteration"]) == iteration
            np.testing.assert_array_equal(saved["parameters"], parameters)
            assert float(saved["loss"]) == loss
        event = json.loads((output / "optimization.jsonl").read_text().splitlines()[-1])
        assert event["event"] == "iteration" and event["iteration"] == iteration
        assert event["accepted"] == accepted and event["backtracks"] == backtracks
        snapshots.append((iteration, parameters.copy(), loss, accepted))
        parameters[:] = 2.9  # Observers cannot mutate the optimizer's live state.

    observed = optimize(
        quadratic,
        initial,
        output,
        method=method,
        iterations=3,
        learning_rate=0.03,
        on_iteration=observe,
    )
    baseline = optimize(
        quadratic, initial, tmp_path / "baseline", method=method, iterations=3, learning_rate=0.03
    )
    assert [row[0] for row in snapshots] == [1, 2, 3]
    assert all(row[3] for row in snapshots)
    assert snapshots[-1][2] < snapshots[0][2]
    np.testing.assert_array_equal(observed.parameters, baseline.parameters)
    np.testing.assert_array_equal(observed.parameters, snapshots[-1][1])
    assert observed.best_loss == baseline.best_loss
    assert observed.evaluations == baseline.evaluations


@pytest.mark.parametrize("method", ["adam", "gd"])
def test_optimizer_iteration_callback_resume_preserves_state_and_absolute_index(
    tmp_path: Path, dtype: np.dtype[np.floating], method: str
) -> None:
    from envs.soft_walker.shared import optimize

    target = np.asarray([[0.4, -0.6, 0.8]], dtype)
    initial = np.zeros_like(target)

    def quadratic(p: np.ndarray) -> tuple[np.floating | float, np.ndarray]:
        return np.sum((p - target) ** 2), 2 * (p - target)

    optimize(
        quadratic, initial, tmp_path / "first", method=method, iterations=2, learning_rate=0.03
    )
    calls = []
    resumed = optimize(
        quadratic,
        initial,
        tmp_path / "resumed",
        method=method,
        iterations=2,
        learning_rate=0.03,
        resume=tmp_path / "first/last.npz",
        on_iteration=lambda **row: calls.append(row),
    )
    whole = optimize(
        quadratic, initial, tmp_path / "whole", method=method, iterations=4, learning_rate=0.03
    )
    assert [row["iteration"] for row in calls] == [3, 4]
    assert all(row["accepted"] for row in calls)
    assert resumed.iterations == 4
    np.testing.assert_array_equal(resumed.parameters, whole.parameters)
    with (
        np.load(tmp_path / "resumed/last.npz", allow_pickle=False) as a,
        np.load(tmp_path / "whole/last.npz", allow_pickle=False) as b,
    ):
        for key in ("parameters", "m", "v", "iteration", "loss"):
            np.testing.assert_array_equal(a[key], b[key])


@pytest.mark.parametrize("method", ["adam", "gd"])
def test_optimizer_iteration_callback_reports_rejection_without_parameter_change(
    tmp_path: Path, dtype: np.dtype[np.floating], method: str
) -> None:
    from envs.soft_walker.shared import optimize

    initial = np.asarray([[0.2, -0.3]], dtype)

    def seed_only(p: np.ndarray) -> tuple[np.floating | float, np.ndarray]:
        # An invalid physical/adjoint trial must leave the accepted state intact.
        valid = np.array_equal(p, initial)
        return (float(np.sum(p**2)) if valid else np.nan), 2 * p

    calls = []
    result = optimize(
        seed_only,
        initial,
        tmp_path,
        method=method,
        iterations=3,
        max_backtracks=3,
        on_iteration=lambda **row: calls.append(row),
    )
    assert len(calls) == 1
    assert calls[0]["iteration"] == 1 and not calls[0]["accepted"]
    assert calls[0]["backtracks"] == 2
    np.testing.assert_array_equal(calls[0]["parameters"], initial)
    np.testing.assert_array_equal(result.parameters, initial)
    assert result.evaluations == 4 and not result.completed
    with np.load(tmp_path / "last.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["parameters"], initial)
        np.testing.assert_array_equal(saved["m"], np.zeros_like(initial))
        np.testing.assert_array_equal(saved["v"], np.zeros_like(initial))
        assert int(saved["iteration"]) == calls[0]["iteration"]
        assert float(saved["loss"]) == calls[0]["loss"]


def test_finite_difference_ladder_keeps_full_curve(dtype: np.dtype[np.floating]) -> None:
    from envs.soft_walker.shared import difference_ladder

    # Independent analytical reference, no JAX derivative in this fixture.
    p = np.asarray([[0.42, -0.7, 1.1]], dtype)

    def loss(q: np.ndarray) -> np.floating:
        return np.sum(np.exp(q))

    result = difference_ladder(loss, lambda q: (loss(q), np.exp(q)), p)
    assert len(result["rows"]) >= 10
    assert result["agreement_passed"]
    assert result["second_order_observed"]


def test_real_state_animation_and_failed_tail(tmp_path: Path) -> None:
    from PIL import Image

    from envs.soft_walker.viewer.trajectory_animation import render_animation

    pos = np.array(
        [
            [[0.0, 0.02], [0.02, 0.02]],
            [[0.001, 0.021], [0.021, 0.02]],
            [[0.002, 0.02], [0.022, 0.02]],
            [[0.002, 0.02], [0.022, 0.02]],
        ]
    )
    rows = np.zeros((4, 13))
    rows[:, 0] = [0, 0.04, 0.08, 0.08]
    rows[-1, 6] = 1
    topology = {
        "active": np.array([True, True]),
        "torso": np.array([False, False]),
        "reference": pos[0],
        "feet": np.eye(2, dtype=bool),
    }
    report = render_animation(
        pos,
        rows,
        topology,
        tmp_path / "fixture.gif",
        title="Synthetic recorder unit fixture — not a robot",
        period=0.5,
    )
    assert report["status"] == "failure_preview"
    assert report["no_state_interpolation"]
    assert max(report["source_indices"]) == 2
    with Image.open(tmp_path / "fixture.gif") as image:
        assert image.n_frames == 3


def test_atomic_artifacts_and_checksums(tmp_path: Path) -> None:
    from envs.soft_walker.shared import archive_run, write_json, write_npz

    folder = tmp_path / "run"
    folder.mkdir()
    write_json(folder / "status.json", {"finite": 1.0, "nonfinite": np.nan})
    assert json.loads((folder / "status.json").read_text())["nonfinite"] is None
    write_npz(folder / "p.npz", parameters=np.ones((4, 3)))
    archive = archive_run(folder)
    with zipfile.ZipFile(archive) as zipped:
        assert zipped.testzip() is None
        assert "run/SHA256SUMS" in zipped.namelist()


def test_right_preconditioned_solve_and_gradient(dtype: np.dtype[np.floating]) -> None:
    from envs.soft_walker.engine.numerics.solvers import NewtonConfig, linear_solve

    matrix = jnp.asarray([[5.0, 2.0, 0.0], [0.01, 0.04, 0.002], [0.0, 1.0, 8.0]], dtype)
    rhs = jnp.asarray([0.2, -0.03, 0.7], dtype)
    cfg = NewtonConfig(preconditioner_side="right")
    f = lambda b: linear_solve(lambda x: matrix @ x, b, jnp.diag(matrix), cfg)[0]
    actual = f(rhs)
    expected = np.linalg.solve(np.asarray(matrix, dtype=np.float64), np.asarray(rhs))
    np.testing.assert_allclose(actual, expected, rtol=2e-4 if dtype == np.float32 else 1e-9)
    got = jax.grad(lambda b: jnp.sum(f(b) ** 2))(rhs)
    reference = 2 * np.linalg.solve(np.asarray(matrix).T, expected)
    np.testing.assert_allclose(got, reference, rtol=2e-4 if dtype == np.float32 else 1e-9)
