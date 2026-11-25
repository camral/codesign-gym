"""Shared selector contracts and equivalence to the pre-existing JAX configuration API."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import asdict, replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from envs.soft_walker.engine import available_integrators, make, make_gymnasium_vec, make_vec
from envs.soft_walker.engine.differentiation import bind_rollout
from envs.soft_walker.engine.embodiments.decoders import cast_embedding
from envs.soft_walker.engine.numerics.selection import resolve_integrator
from envs.soft_walker.engine.physics.spring_config import SpringConfig, with_integrator


@pytest.mark.parametrize("backend", ["jax", "warp"])
@pytest.mark.parametrize("physics", ["spring", "beam", "fem", "mpm2", "mpm3"])
@pytest.mark.parametrize("interface", ["engine", "registered"])
def test_integrator_catalog(backend: str, physics: str, interface: str) -> None:
    """The catalog distinguishes physics, runtime, and wrapper constraints."""
    expected = ["backward_euler"]
    if physics == "spring":
        expected.append("kvaerno3_native")
        if backend == "jax" and interface == "engine":
            expected.append("kvaerno3")
    choices = available_integrators(backend=backend, physics=physics, interface=interface)
    assert choices == tuple(expected)
    for method in choices:
        assert (
            resolve_integrator(method, backend=backend, physics=physics, interface=interface)
            == method
        )
        assert (
            resolve_integrator(
                None, backend=backend, physics=physics, interface=interface, configured=method
            )
            == method
        )


def test_integrator_catalog_is_host_only() -> None:
    """Capability discovery must work without initializing optional accelerator runtimes."""
    script = """
import json, sys
from envs.soft_walker.engine import available_integrators
assert available_integrators(backend='warp', physics='spring') == (
    'backward_euler', 'kvaerno3_native')
print(json.dumps({name: name in sys.modules for name in ('jax', 'warp', 'diffrax')}))
"""
    result = subprocess.run(
        [sys.executable, "-c", script], check=True, capture_output=True, text=True
    )
    assert json.loads(result.stdout) == {"jax": False, "warp": False, "diffrax": False}


@pytest.mark.parametrize(
    "options",
    [
        {"backend": "unknown", "physics": "spring"},
        {"backend": "jax", "physics": "unknown"},
        {"backend": "jax", "physics": "spring", "interface": "unknown"},
    ],
)
def test_unknown_catalog_selectors_raise(options: dict[str, str]) -> None:
    """A misspelling must not look like an empty or default capability set."""
    with pytest.raises(ValueError, match="Unknown"):
        available_integrators(**options)


def test_jax_config_precedence_and_solver_preservation() -> None:
    """An override cannot silently change user-selected solvers or mutate a config."""
    default = SpringConfig()
    assert with_integrator(default, None) is default
    assert with_integrator(default, "backward_euler") is default
    assert with_integrator(None, None) == default
    configured = replace(default, linear_solver="gmres", gmres_method="fused")
    selected = with_integrator(configured, "kvaerno3_native")
    assert asdict(selected) == dict(asdict(configured), integrator="kvaerno3_native")
    assert configured.integrator == "backward_euler"
    assert with_integrator(selected, None) is selected
    assert with_integrator(None, "kvaerno3_native").linear_solver == "gmres"
    assert with_integrator(None, "kvaerno3").linear_solver == "dense"
    with pytest.raises(ValueError, match="gmres"):
        with_integrator(default, "kvaerno3_native")
    with pytest.raises(ValueError, match="dense"):
        with_integrator(configured, "kvaerno3")


@pytest.mark.parametrize(
    "env_id", ["SoftBeam3D-v0", "SoftHexFEM3D-v0", "SoftMPM2D-v0", "SoftMPM3D-v0"]
)
def test_jax_body_rejects_unsupported_integrator_before_device(env_id: str) -> None:
    """Unsupported schemes fail before the deliberately invalid device is resolved."""
    with pytest.raises(ValueError, match="Integrator .* unsupported"):
        make(env_id, integrator="kvaerno3_native", device="unavailable")


@pytest.mark.parametrize("method", ["backward_euler", "kvaerno3_native", "kvaerno3"])
def test_selector_matches_configured_jax_trajectory_and_gradient(
    method: str, dtype: np.dtype[np.floating], tolerance: float
) -> None:
    """Selection preserves observations, rewards, states, and pure-rollout control gradients.

    The numerical configuration, precision and device are identical. The selector
    route overrides a backward-Euler config while the reference uses the legacy
    config field. A nonzero gradient rules out accidentally testing a dead path.
    """
    if method == "kvaerno3":
        pytest.importorskip("diffrax")
    config = SpringConfig(
        grid_shape=(1, 1),
        ground=False,
        settle_steps=0,
        substeps=2,
        control_dt=0.002,
        linear_solver="gmres" if method == "kvaerno3_native" else "dense",
    )
    options = dict(
        dtype=dtype,
        device=os.environ.get("CODESIGN_TEST_DEVICE", "cpu"),
        max_action_space_size=2,
        max_observation_space_size=128,
    )
    selected, base = make(config=config, integrator=method, **options)
    reference, _ = make(config=replace(config, integrator=method), **options)
    try:
        assert selected.integrator == reference.integrator == method
        assert config.integrator == "backward_euler"
        assert selected.config == reference.config
        action = np.asarray([0.2, -0.1], dtype=dtype)
        for env in (selected, reference):
            _, rejected, _ = env.reset(embodiment=base)
            assert not rejected.any()
        actual = selected.step(action)
        expected = reference.step(action)
        assert not actual[2] and not expected[2]
        assert actual[4]["integrator"] == method
        for got, want in zip(actual[:4], expected[:4]):
            np.testing.assert_array_equal(got, want)
        for got, want in zip(jax.tree.leaves(selected._state), jax.tree.leaves(reference._state)):
            np.testing.assert_array_equal(got, want)

        results = []
        with jax.default_device(selected.device):
            controls = jnp.asarray(action[None, :])
            for env in (selected, reference):
                rollout = bind_rollout(env)
                embedding = cast_embedding(env.decoder, base)

                def loss(actions: jax.Array) -> jax.Array:
                    return rollout(embedding, actions)["metrics"]["work_positive_j"]

                value, gradient = jax.jit(jax.value_and_grad(loss))(controls)
                assert gradient.devices() == {selected.device}
                assert np.isfinite(value) and np.isfinite(gradient).all()
                assert np.linalg.norm(np.asarray(gradient)) > 0
                results.append((value, gradient))
        for got, want in zip(results[0], results[1]):
            np.testing.assert_allclose(got, want, rtol=tolerance, atol=tolerance)
    finally:
        selected.close()
        reference.close()


def test_jax_vector_keeps_selection_after_reset(dtype: np.dtype[np.floating]) -> None:
    """Vector construction and repeated lane resets retain the chosen fixed method."""
    config = SpringConfig(
        grid_shape=(1, 1), ground=False, settle_steps=0, substeps=1, linear_solver="gmres"
    )
    env, base = make_vec(
        num_envs=2,
        config=config,
        integrator="kvaerno3_native",
        device=os.environ.get("CODESIGN_TEST_DEVICE", "cpu"),
        dtype=dtype,
    )
    try:
        assert env.integrator == "kvaerno3_native"
        for _ in range(2):
            _, rejected, _ = env.reset(embodiment=base)
            assert not rejected.any()
            assert all(lane.config.integrator == env.integrator for lane in env.envs)
    finally:
        env.close()


@pytest.mark.parametrize("backend", ["jax", "warp"])
def test_gymnasium_adapter_exposes_integrator(backend: str) -> None:
    """The standard vector adapter reports the backend choice before the first reset."""
    env = make_gymnasium_vec(
        backend=backend, integrator="kvaerno3_native", dtype="float32", device="cpu"
    )
    try:
        assert env.integrator == env.inner.integrator == "kvaerno3_native"
        assert env.get_attr("integrator") == ("kvaerno3_native",)
    finally:
        env.close()
