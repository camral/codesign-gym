"""Compact JAX MPM support preserves transfers, implicit derivatives, and rollback."""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from envs.soft_walker.engine.embodiments.decoders import Fields
from envs.soft_walker.engine.numerics.solvers import NewtonConfig
from envs.soft_walker.engine.physics import mpm
from envs.soft_walker.engine.physics.common import DynamicsConfig
from envs.soft_walker.engine.physics.contact import PlaneContact
from envs.soft_walker.engine.physics.mpm_support import compact_support


def fixture(dim: int, dtype: np.dtype[np.floating]) -> tuple[mpm.Model, mpm.State, mpm.Grid]:
    """Create a small deformed body within a larger, mostly empty background grid."""
    origin = (0.025, 0.0002) + ((0.025,) if dim == 3 else ())
    cloud = mpm.uniform_cloud(cell_shape=(1,) * dim, origin=origin, dtype=dtype)
    n = len(cloud.reference)
    ones = jnp.ones(n, dtype)
    fields = Fields(
        occupancy=ones,
        young=20000.0 * ones,
        density=1000.0 * ones,
        strength=0.5 * ones,
        direction=jnp.broadcast_to(jnp.eye(dim, dtype=dtype)[1], (n, dim)),
        actuator_type=jnp.ones(n, jnp.int32),
        channel=jnp.arange(n, dtype=jnp.int32) % 2,
    )
    model = mpm.instantiate(cloud, fields)
    state = mpm.initial_state(model)
    state = state._replace(
        v=state.v.at[:, 0].set(0.015).at[:, 1].set(-0.01),
        F=state.F.at[:, 0, 0].add(0.002),
    )
    grid = mpm.Grid(shape=(8,) * dim, origin=(-0.05,) * dim)
    return model, state, grid


def remapped(st: mpm.Stencil, ids: jax.Array, covered: jax.Array) -> mpm.Stencil:
    """Apply support masks without altering surviving interpolation expressions."""
    return st._replace(
        ids=ids,
        weight=jnp.where(covered, st.weight, 0.0),
        gradient=jnp.where(covered[..., None], st.gradient, 0.0),
    )


@pytest.mark.parametrize("dim", [2, 3])
def test_compact_transfers_match_full_grid_with_inactive_particles(
    dim: int, dtype: np.dtype[np.floating]
) -> None:
    """Mass/momentum and all particle reconstructions agree, including inactive sites."""
    model, state, grid = fixture(dim, dtype)
    # One inactive site shares active support; another lies outside its support.
    extra = jnp.asarray([[0.031, 0.008] + ([0.031] if dim == 3 else [])], dtype)
    far = jnp.full((1, dim), 0.10, dtype)
    x = jnp.concatenate((state.x, extra, far))
    velocity = jnp.concatenate((state.v, jnp.zeros((2, dim), dtype)))
    affine = jnp.zeros((len(x), dim, dim), dtype)
    mass = jnp.concatenate((model.mass, jnp.zeros(2, dtype)))
    st = mpm.stencil(x, grid)
    expected_ids = np.unique(np.asarray(st.ids)[: len(state.x)])
    capacity = len(expected_ids) + 3
    mapping = jax.jit(
        lambda indices, active: compact_support(
            indices, active, grid_nodes=grid.count, capacity=capacity
        )
    )(st.ids, mass > 0)
    assert bool(mapping.valid)
    assert int(mapping.live_count) == len(expected_ids)
    np.testing.assert_array_equal(mapping.node_ids[: len(expected_ids)], expected_ids)
    compact_st = remapped(st, mapping.ids, mapping.covered)
    full_mass, full_velocity = mpm.p2g(mass, velocity, affine, st, grid)
    small_mass, small_velocity = mpm.p2g(
        mass, velocity, affine, compact_st, grid, node_count=capacity
    )
    tolerance = 3e-5 if dtype.name == "float32" else 2e-12
    np.testing.assert_allclose(
        small_mass[: len(expected_ids)], full_mass[expected_ids], rtol=tolerance
    )
    np.testing.assert_allclose(
        small_velocity[: len(expected_ids)], full_velocity[expected_ids], rtol=tolerance
    )
    np.testing.assert_array_equal(small_mass[len(expected_ids) :], 0.0)
    for actual, expected in zip(
        mpm.g2p(small_velocity, compact_st, grid), mpm.g2p(full_velocity, st, grid)
    ):
        np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance * 1e-3)


def test_compact_support_empty_and_overflow_are_explicit(dtype: np.dtype[np.floating]) -> None:
    """Capacity failure remains observable and every mapped index remains in bounds."""
    model, state, grid = fixture(2, dtype)
    st = mpm.stencil(state.x, grid)
    for active in (model.active, jnp.zeros_like(model.active)):
        result = jax.jit(
            lambda mask: compact_support(st.ids, mask, grid_nodes=grid.count, capacity=1)
        )(active)
        assert np.all(np.asarray(result.ids) == 0)
        expected_count = len(np.unique(np.asarray(st.ids))) if np.any(active) else 0
        assert int(result.live_count) == expected_count
        assert bool(result.valid) == (expected_count == 0)
        if expected_count == 0:
            assert not np.any(result.covered)


@pytest.mark.parametrize("dim", [2, 3])
def test_compact_implicit_state_and_design_control_gradients(
    dim: int, dtype: np.dtype[np.floating]
) -> None:
    """Full and compact roots preserve physical states and continuous-input adjoints."""
    model, state, grid = fixture(dim, dtype)
    config = DynamicsConfig(
        gravity=3.0,
        actuator_tau=0.0,
        actuation_model="active_stress",
        contact=PlaneContact(enabled=True, friction=0.15),
        newton=NewtonConfig(gmres_method="fused"),
        max_halvings=0,
    )
    capacity = len(np.unique(np.asarray(mpm.stencil(state.x, grid).ids))) + 4
    parameters = jnp.asarray([0.08, -0.04, 0.02, -0.01, 0.001], dtype)

    def run(p: jax.Array, cfg: DynamicsConfig) -> tuple[mpm.State, jax.Array]:
        """Vary controls, stiffness, density, and incoming positions in one pure trial."""
        density = model.density * jnp.exp(p[3])
        body = model._replace(
            young=model.young * jnp.exp(p[2]),
            density=density,
            mass=model.alpha * density * model.cloud.volume,
        )
        initial = state._replace(x=state.x.at[:, 0].add(p[4]))
        out, info = mpm.trial(body, initial, p[:2], jnp.asarray(0.00025, dtype), grid, cfg)
        return out, info.accepted

    def loss(p: jax.Array, cfg: DynamicsConfig) -> jax.Array:
        """Score motion, deformation, and actuation work to exercise multiple adjoint paths."""
        out, accepted = run(p, cfg)
        value = (
            100.0 * jnp.mean(out.x[:, 0])
            + 0.1 * jnp.sum(out.v**2)
            + 0.01 * jnp.sum(out.F**2)
            + out.work_positive
        )
        return jnp.where(accepted == 1, value, jnp.nan)

    compact_config = replace(config, mpm_compact_nodes=capacity)
    expected, expected_accepted = jax.jit(lambda p: run(p, config))(parameters)
    actual, actual_accepted = jax.jit(lambda p: run(p, compact_config))(parameters)
    assert float(expected_accepted) == float(actual_accepted) == 1.0
    tolerance = 3e-5 if dtype.name == "float32" else 2e-9
    # Match the existing resident/full-driver state comparison near zero;
    # compact reduction order may change the last float32 bits of APIC C.
    state_atol = 2e-7 if dtype.name == "float32" else tolerance * 1e-3
    for actual_leaf, expected_leaf in zip(actual, expected):
        np.testing.assert_allclose(actual_leaf, expected_leaf, rtol=tolerance, atol=state_atol)
    full_value, full_gradient = jax.jit(jax.value_and_grad(lambda p: loss(p, config)))(parameters)
    small_value, small_gradient = jax.jit(jax.value_and_grad(lambda p: loss(p, compact_config)))(
        parameters
    )
    assert np.isfinite(full_gradient).all() and np.isfinite(small_gradient).all()
    np.testing.assert_allclose(small_value, full_value, rtol=tolerance, atol=tolerance * 1e-3)
    np.testing.assert_allclose(small_gradient, full_gradient, rtol=tolerance, atol=tolerance * 1e-3)


def test_compact_capacity_failure_rolls_back_without_newton(dtype: np.dtype[np.floating]) -> None:
    """A valid physical stencil cannot commit a truncated compact approximation."""
    model, state, grid = fixture(2, dtype)
    config = DynamicsConfig(mpm_compact_nodes=1, max_halvings=0)
    out, info = jax.jit(
        lambda: mpm.trial(model, state, jnp.zeros(2, dtype), 0.00025, grid, config)
    )()
    assert int(out.status) == 1
    assert int(info.code) == mpm.COMPACT_CAPACITY_EXCEEDED
    assert float(info.accepted) == 0.0 and float(info.nonlinear_iterations) == 0.0
    assert not bool(info.support_valid)
    for actual, expected in zip(out[:-1], state[:-1]):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("capacity", [-1, 1.5, True])
def test_compact_capacity_configuration_rejects_invalid_values(capacity: object) -> None:
    """Static capacities require plain nonnegative integers, with zero preserving defaults."""
    with pytest.raises(ValueError, match="mpm_compact_nodes"):
        DynamicsConfig(mpm_compact_nodes=capacity)  # type: ignore[arg-type]
