"""Invalid numerical settings fail before tracing or device allocation."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from envs.soft_walker.engine.numerics.solvers import NewtonConfig
from envs.soft_walker.engine.physics.common import DynamicsConfig
from envs.soft_walker.engine.warp_native.config import MPMConfig


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_dynamics_rejects_nonfinite_gravity(value: float) -> None:
    with pytest.raises(ValueError, match="gravity must be finite"):
        DynamicsConfig(gravity=value)


@pytest.mark.parametrize("name", ["velocity_ref", "max_log_stretch", "min_J"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), 0.0, -1.0])
def test_dynamics_rejects_nonfinite_or_nonpositive_scales(name: str, value: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        DynamicsConfig(**{name: value})


@pytest.mark.parametrize("value", [True, False, np.bool_(True), -1, 5, 1.5, 2.0, float("nan")])
def test_retry_capacity_requires_a_bounded_integer(value: object) -> None:
    with pytest.raises(ValueError, match="max_halvings must be an integer in 0..4"):
        DynamicsConfig(max_halvings=value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [0, 4, np.int32(0), np.int64(4)])
def test_retry_capacity_accepts_python_and_numpy_integer_boundaries(value: int) -> None:
    config = DynamicsConfig(gravity=-9.81, max_halvings=value)
    assert config.gravity == -9.81
    assert config.max_halvings == value


@pytest.mark.parametrize("name", ["max_steps", "backtracks", "restart", "max_restarts"])
@pytest.mark.parametrize(
    "value", [True, False, np.bool_(True), 0, -1, 1.5, 2.0, float("nan"), float("inf")]
)
def test_newton_capacities_require_positive_integers(name: str, value: object) -> None:
    with pytest.raises(ValueError, match="Iteration capacities must be positive integers"):
        NewtonConfig(**{name: value})


@pytest.mark.parametrize("value", [1, np.int32(1), np.int64(128)])
def test_newton_capacities_preserve_numpy_integral_inputs(value: int) -> None:
    config = NewtonConfig(max_steps=value, backtracks=value, restart=value, max_restarts=value)
    assert config.max_steps == config.backtracks == config.restart == config.max_restarts == value


@pytest.mark.parametrize("method", ["legacy", "adaptive"])
def test_other_native_gmres_methods_keep_their_capacity_contract(method: str) -> None:
    config = MPMConfig(krylov_method=method, restart=256)
    config.validate_solver_capacity(1024)


def test_dynamic_adaptive_capacity_uses_existing_dof_clamp() -> None:
    config = MPMConfig(krylov_method="adaptive_dynamic", restart=256)
    config.validate_solver_capacity(128)
    with pytest.raises(ValueError, match="effective restart exceeds 128"):
        config.validate_solver_capacity(129)


def test_mpm_bundle_rejects_unsupported_effective_restart_without_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wp = pytest.importorskip("warp")
    from envs.soft_walker.engine.warp_native.model import uniform_body

    def unexpected_allocation(*args: object, **kwargs: object) -> None:
        pytest.fail("Bundle validation attempted a device allocation")

    monkeypatch.setattr(wp, "array", unexpected_allocation)
    bundle = uniform_body(2, cells=(3, 3), particles_per_axis=1)
    bundle.config = replace(
        bundle.config, grid_shape=(12, 12), krylov_method="adaptive_dynamic", restart=256
    )
    with pytest.raises(ValueError, match="effective restart exceeds 128"):
        bundle.validate()
    bundle.config = replace(bundle.config, compact_nodes=64)
    assert bundle.validate() is bundle
    bundle.config = replace(bundle.config, compact_nodes=65)
    with pytest.raises(ValueError, match="effective restart exceeds 128"):
        bundle.validate()


@pytest.mark.parametrize(
    ("physics", "large_shape"),
    [("spring", (8, 8)), ("beam", (22, 1, 1)), ("fem", (3, 3, 3))],
)
def test_solid_bundle_preserves_small_system_clamps_and_rejects_large_restarts(
    physics: str, large_shape: tuple[int, ...], monkeypatch: pytest.MonkeyPatch
) -> None:
    wp = pytest.importorskip("warp")
    from envs.soft_walker.engine.warp_native.solid_builders import regular_solid

    def unexpected_allocation(*args: object, **kwargs: object) -> None:
        pytest.fail("Bundle validation attempted a device allocation")

    monkeypatch.setattr(wp, "array", unexpected_allocation)
    small = regular_solid(physics, shape=(1,) * len(large_shape))
    small.config = replace(small.config, restart=256)
    assert small.validate() is small
    large = regular_solid(physics, shape=large_shape)
    large.config = replace(large.config, restart=256)
    with pytest.raises(ValueError, match="effective restart exceeds 128"):
        large.validate()
