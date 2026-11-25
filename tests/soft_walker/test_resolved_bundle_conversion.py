"""Resolved bundle conversion preserves physics and explicit numerical choices."""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from envs.soft_walker.diagnostics.warp.solid_fixtures import fixture, reference_functions
from envs.soft_walker.engine.numerics.solvers import NewtonConfig
from envs.soft_walker.engine.physics.spring_config import SpringConfig
from envs.soft_walker.engine.resolved_bundle import to_jax
from envs.soft_walker.engine.warp_native.model import uniform_body
from envs.soft_walker.engine.warp_native.solid_builders import regular_solid
from envs.soft_walker.engine.warp_native.solid_conversion import from_jax


@pytest.mark.parametrize("kind", ("spring", "beam", "fem"))
@pytest.mark.parametrize("contact", (False, True))
def test_solid_resolved_residual_and_state(kind: str, contact: bool, dtype: np.dtype) -> None:
    """Compare force residuals/JVPs to original independent JAX assembled models."""
    bundle, old_model, old_state, old_config, _, _ = fixture(kind, dtype.name, contact)
    body = to_jax(bundle)
    qdim = 6 if kind == "beam" else bundle.config.dim
    q = jnp.asarray(np.linspace(-0.01, 0.02, len(old_model.mass) * qdim), dtype)
    command = jnp.asarray([0.3, -0.2], dtype)
    old_residual, old_update = reference_functions(
        kind, old_model, old_state, old_config, bundle.config.dt
    )
    residual, update = reference_functions(
        kind, body.model, body.state, body.configuration, bundle.config.dt
    )
    tolerance = (
        dict(rtol=2e-5, atol=2e-6) if dtype.name == "float32" else dict(rtol=2e-10, atol=2e-11)
    )
    np.testing.assert_allclose(residual(q, command), old_residual(q, command), **tolerance)
    tangent = jnp.full_like(q, 0.01)
    observed = jax.jvp(lambda v: residual(v, command), (q,), (tangent,))[1]
    expected = jax.jvp(lambda v: old_residual(v, command), (q,), (tangent,))[1]
    np.testing.assert_allclose(observed, expected, **tolerance)
    for key, value in old_update(q, command).items():
        np.testing.assert_allclose(update(q, command)[key], value, **tolerance)
    assert body.grid is None


@pytest.mark.parametrize("kind", ("spring", "beam", "fem", "mpm2", "mpm3"))
def test_bundle_conversion_owns_arrays_and_preserves_ledgers(kind: str, dtype: np.dtype) -> None:
    """Conversion retains nonzero initial motion/work without borrowing host storage."""
    bundle = (
        uniform_body(int(kind[-1]), precision=dtype.name)
        if kind.startswith("mpm")
        else regular_solid(kind, precision=dtype.name)
    )
    # Builders may omit the optional initial state; exercise explicit checkpoints
    # on solids by first asking the reference model for its schema-compatible state.
    body = to_jax(bundle)
    reference = (
        body.model.cloud.reference
        if kind.startswith("mpm")
        else body.model.reference
        if kind == "spring"
        else body.model.topology.reference
        if kind == "beam"
        else body.model.mesh.reference
    )
    expected = np.asarray(reference).copy()
    bundle.fields["reference"] += dtype.type(0.3)
    np.testing.assert_array_equal(reference, expected)
    if kind.startswith("mpm"):
        assert body.grid is not None
        return
    n, dim = expected.shape
    bundle.initial = {
        "u": np.full((n, dim), 0.001, dtype),
        "v": np.full((n, dim), 0.01, dtype),
        "rotation": np.broadcast_to(np.eye(dim, dtype=dtype), (n, dim, dim)).copy(),
        "omega": np.zeros((n, dim), dtype),
        "actuator": np.full(bundle.channels, 0.02, dtype),
        "ledger": np.asarray([0.1, 0.2, 0.3, 0.4], dtype),
        "status": np.asarray([0], np.int32),
    }
    converted = to_jax(bundle)
    state = converted.state
    np.testing.assert_array_equal(
        [state.work_signed, state.work_positive, state.dissipated, state.time],
        bundle.initial["ledger"],
    )
    np.testing.assert_array_equal(state.v, bundle.initial["v"])
    position = state.x - converted.model.reference if kind == "spring" else state.u
    np.testing.assert_allclose(position, bundle.initial["u"], rtol=3e-5, atol=1e-8)


def test_spring_bundle_solver_settings_are_explicit(dtype: np.dtype) -> None:
    """Nondefault restart/tolerance fields reach both exported and restored configs."""
    bundle = regular_solid("spring", precision=dtype.name)
    bundle.config = replace(
        bundle.config,
        restart=9,
        max_restarts=7,
        linear_rtol64=5e-11,
        linear_rtol32=7e-6,
        armijo=2e-4,
        linear_refinements=1,
        integrator="kvaerno3_native",
    )
    cfg = to_jax(bundle).configuration
    assert isinstance(cfg, SpringConfig)
    assert (cfg.gmres_restart, cfg.gmres_max_restarts) == (9, 7)
    assert (cfg.linear_rtol64, cfg.linear_rtol32, cfg.armijo) == (5e-11, 7e-6, 2e-4)
    assert cfg.gmres_refinement_steps == 1 and cfg.integrator == "kvaerno3_native"
    defaults, old = SpringConfig(), NewtonConfig()
    assert defaults.gmres_restart == old.restart and defaults.gmres_max_restarts == old.max_restarts
    assert (
        defaults.linear_rtol64 == old.linear_rtol64 and defaults.linear_rtol32 == old.linear_rtol32
    )
    assert defaults.armijo == old.armijo
    for bad in (
        {"gmres_restart": 0},
        {"gmres_max_restarts": True},
        {"armijo": 1.0},
        {"linear_rtol32": 0.0},
    ):
        with pytest.raises(ValueError):
            replace(defaults, **bad)


@pytest.mark.parametrize("kind", ("spring", "beam"))
def test_unrepresentable_coordinate_locks_are_explicit(kind: str, dtype: np.dtype) -> None:
    """Native coordinate constraints cannot silently disappear during conversion."""
    bundle = regular_solid(kind, precision=dtype.name)
    bundle.fields["free"][0, 0] = 0
    with pytest.raises(ValueError, match="constraints"):
        to_jax(bundle)


@pytest.mark.parametrize("kind", ("beam", "fem"))
def test_solid_export_rejects_changed_integrator_and_material_law(
    kind: str, dtype: np.dtype
) -> None:
    """Unsupported retry integration and actuation cannot become native defaults."""
    bundle, model, state, cfg, _, _ = fixture(kind, dtype.name)
    with pytest.raises(ValueError, match="max_halvings=0"):
        from_jax(kind, model, state, replace(cfg, max_halvings=1), dt=bundle.config.dt)
    with pytest.raises(ValueError, match="active_strain"):
        from_jax(
            kind, model, state, replace(cfg, actuation_model="active_stress"), dt=bundle.config.dt
        )
