"""Adam retry, resume, best-state, and checkpoint persistence contracts."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


import json

import numpy as np
import pytest

from envs.soft_walker.shared import InvalidEvaluation, run_adam


def quadratic(p: np.ndarray) -> tuple[np.floating, np.ndarray]:
    return np.sum(p * p), 2 * p


def test_adam_accepts_finite_uphill_updates_and_saves_best(tmp_path: Path) -> None:
    callbacks = []
    result = run_adam(
        quadratic,
        np.array([0.1]),
        tmp_path,
        iterations=2,
        learning_rate=1.0,
        on_iteration=lambda **kw: callbacks.append(kw),
    )
    assert result["completed"] and result["accepted_updates"] == 2
    assert result["evaluations"] == 3
    assert callbacks[1]["loss"] > callbacks[0]["loss"]
    assert result["best_loss"] == pytest.approx(0.01)
    assert result["final_loss"] > result["best_loss"]
    np.testing.assert_array_equal(result["parameters"], [0.1])
    assert [c["iteration"] for c in callbacks] == [0, 1, 2]
    assert len(list((tmp_path / "checkpoints").glob("*.npz"))) == 3
    assert all(c["timings"]["value_and_gradient_s"] >= 0 for c in callbacks)
    assert all(
        "parameters" not in json.loads(line)
        for line in (tmp_path / "optimization.jsonl").read_text().splitlines()
    )


@pytest.mark.parametrize("failure", ["nan", "exception"])
def test_invalid_physics_retries_without_advancing_adam(tmp_path: Path, failure: str) -> None:
    calls = []

    def objective(p: np.ndarray) -> tuple[np.floating | float, np.ndarray]:
        calls.append(p.copy())
        if p[0] < 0.75:
            if failure == "exception":
                raise InvalidEvaluation("solver did not converge")
            return np.nan, np.full_like(p, np.nan)
        return quadratic(p)

    result = run_adam(objective, np.array([1.0]), tmp_path, iterations=1, learning_rate=1.0)
    assert result["accepted_updates"] == 1 and result["evaluations"] == 4
    assert result["rejected_trials"] == 2
    assert len(calls) == 4
    np.testing.assert_allclose(result["final_parameters"], [0.75], atol=1e-8)
    with np.load(tmp_path / "last.npz") as saved:
        assert int(saved["t"]) == 1
        np.testing.assert_allclose(saved["m"], [0.2])
        np.testing.assert_allclose(saved["v"], [0.004])


def test_failed_step_preserves_last_valid_state_and_reports_failure(tmp_path: Path) -> None:
    calls = 0

    def objective(p: np.ndarray) -> tuple[np.floating | float, np.ndarray]:
        nonlocal calls
        calls += 1
        if calls > 1:
            raise InvalidEvaluation("invalid adjoint")
        return quadratic(p)

    with pytest.raises(InvalidEvaluation, match="eight invalid"):
        run_adam(objective, np.array([1.0]), tmp_path, iterations=50)
    report = json.loads((tmp_path / "optimization_summary.json").read_text())
    assert not report["completed"] and report["accepted_updates"] == 0
    assert report["evaluations"] == 9 and report["rejected_trials"] == 8
    with np.load(tmp_path / "last.npz") as saved:
        np.testing.assert_array_equal(saved["parameters"], [1.0])
        np.testing.assert_array_equal(saved["m"], [0.0])
        assert int(saved["iteration"]) == 0 and int(saved["evaluations"]) == 9


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_resume_matches_uninterrupted_50_updates_exactly(
    tmp_path: Path, dtype: np.dtype[np.floating]
) -> None:
    initial = np.array([1.3, -0.6, 0.12], dtype=dtype)
    full = run_adam(quadratic, initial, tmp_path / "full", iterations=50)
    run_adam(quadratic, initial, tmp_path / "split", iterations=17)
    resumed = run_adam(
        quadratic,
        initial,
        tmp_path / "split",
        iterations=33,
        resume=tmp_path / "split" / "last.npz",
    )
    assert resumed["accepted_updates"] == 50
    assert resumed["session_accepted_updates"] == 33
    assert resumed["session_evaluations"] == 33
    assert resumed["evaluations"] == full["evaluations"] == 51
    assert resumed["best_loss"] == full["best_loss"]
    for key in ("parameters", "final_parameters"):
        np.testing.assert_array_equal(resumed[key], full[key])
    with (
        np.load(tmp_path / "full" / "last.npz") as a,
        np.load(tmp_path / "split" / "last.npz") as b,
    ):
        for key in ("parameters", "gradient", "m", "v", "t", "best_parameters", "best_iteration"):
            np.testing.assert_array_equal(a[key], b[key])


def test_bounds_callback_isolation_and_bad_resume(tmp_path: Path) -> None:
    def callback(**kw: object) -> None:
        kw["parameters"][:] = 99
        kw["gradient"][:] = np.nan

    result = run_adam(
        lambda p: (-p.sum(), -np.ones_like(p)),
        np.array([0.9]),
        tmp_path,
        iterations=2,
        learning_rate=1.0,
        bound=1.0,
        on_iteration=callback,
    )
    np.testing.assert_array_equal(result["final_parameters"], [1.0])
    with pytest.raises(ValueError, match="learning rate/bound mismatch"):
        run_adam(quadratic, np.array([0.9]), tmp_path, resume=tmp_path / "last.npz")
    with pytest.raises(FileExistsError):
        run_adam(quadratic, np.array([0.9]), tmp_path)


def test_invalid_initial_does_not_create_fake_checkpoint(tmp_path: Path) -> None:
    with pytest.raises(InvalidEvaluation, match="initial"):
        run_adam(lambda p: (0.0, np.full_like(p, np.nan)), np.array([1.0]), tmp_path)
    assert not (tmp_path / "last.npz").exists()
    assert not (tmp_path / "best.npz").exists()
    report = json.loads((tmp_path / "optimization_summary.json").read_text())
    assert not report["completed"] and report["best_loss"] is None


def test_programming_errors_are_not_retried(tmp_path: Path) -> None:
    calls = 0

    def wrong_shape(p: np.ndarray) -> tuple[np.floating, np.ndarray]:
        nonlocal calls
        calls += 1
        return 1.0, np.zeros((2, 2))

    with pytest.raises(ValueError, match="same-shaped"):
        run_adam(wrong_shape, np.array([1.0]), tmp_path)
    assert calls == 1


def test_moment_overflow_never_becomes_an_accepted_zero_step(tmp_path: Path) -> None:
    with pytest.raises(InvalidEvaluation, match="Adam moment"):
        run_adam(
            lambda p: (1.0, np.full_like(p, 1e30)), np.array([1.0], dtype=np.float32), tmp_path
        )
    report = json.loads((tmp_path / "optimization_summary.json").read_text())
    assert not report["completed"] and report["accepted_updates"] == 0
    with np.load(tmp_path / "last.npz") as saved:
        assert np.isfinite(saved["v"]).all()
