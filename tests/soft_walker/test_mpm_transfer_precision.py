"""Float32 transfer preserves translation and the original empty-mass derivatives."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from envs.soft_walker.engine.physics import mpm


def _stencil(dim: int, dtype: np.dtype[np.floating]) -> tuple[mpm.Grid, mpm.Stencil]:
    """Place seven particles inside a grid with both shared and empty nodes."""
    grid = mpm.Grid(shape=(8,) * dim, spacing=0.05, origin=(0.0,) * dim)
    points = jnp.asarray(np.random.default_rng(31).uniform(0.10, 0.18, (7, dim)), dtype)
    return grid, mpm.stencil(points, grid)


@pytest.mark.parametrize("dim", [2, 3])
def test_uniform_translation_and_empty_support(dim: int, dtype: np.dtype[np.floating]) -> None:
    """Uniform velocity survives float32 scatters exactly; empty buffers remain zero."""
    grid, st = _stencil(dim, dtype)
    mass = jnp.linspace(0.5, 1.1, 7, dtype=dtype).at[0].set(0.0)
    constant = jnp.asarray([0.015, -0.01, 0.02][:dim], dtype)
    velocity = jnp.broadcast_to(constant, (7, dim)).at[0].set(3.0)
    affine = jnp.zeros((7, dim, dim), dtype)
    transfer = jax.jit(lambda m: mpm.p2g(m, velocity, affine, st, grid))
    grid_mass, grid_velocity = transfer(mass)
    occupied = np.asarray(grid_mass) > 0
    expected = np.broadcast_to(np.asarray(constant), (int(occupied.sum()), dim))
    if dtype.name == "float32":
        np.testing.assert_array_equal(np.asarray(grid_velocity)[occupied], expected)
    else:
        np.testing.assert_allclose(np.asarray(grid_velocity)[occupied], expected, rtol=2e-12)
    np.testing.assert_array_equal(np.asarray(grid_velocity)[~occupied], 0.0)
    for value in transfer(jnp.zeros_like(mass)):
        np.testing.assert_array_equal(value, 0.0)

    empty_stencil = mpm.stencil(jnp.empty((0, dim), dtype), grid)
    for value in mpm.p2g(
        jnp.empty(0, dtype),
        jnp.empty((0, dim), dtype),
        jnp.empty((0, dim, dim), dtype),
        empty_stencil,
        grid,
    ):
        np.testing.assert_array_equal(value, 0.0)


@pytest.mark.parametrize("dim", [2, 3])
def test_galilean_shift_and_mass_velocity_affine_derivatives(
    dim: int, dtype: np.dtype[np.floating]
) -> None:
    """A velocity boost and joint input derivatives agree with independent NumPy sums.

    The all-zero-mass case checks the original AD contract at the inactive branch:
    its safe denominator is one, and mass tangents still contribute momentum.
    """
    grid, st = _stencil(dim, dtype)
    random = np.random.default_rng(19)
    mass = jnp.asarray(random.uniform(0.5, 1.2, 7), dtype).at[0].set(0.0)
    velocity = jnp.asarray(random.normal(size=(7, dim)) * 0.04, dtype)
    affine = jnp.asarray(random.normal(size=(7, dim, dim)) * 0.1, dtype)
    tangents = (
        jnp.asarray(random.normal(size=7) * 0.1, dtype),
        jnp.asarray(random.normal(size=(7, dim)) * 0.03, dtype),
        jnp.asarray(random.normal(size=(7, dim, dim)) * 0.2, dtype),
    )
    boost = jnp.asarray([0.3, -0.2, 0.1][:dim], dtype)
    grid_mass, grid_velocity = mpm.p2g(mass, velocity, affine, st, grid)
    shifted_mass, shifted_velocity = mpm.p2g(mass, velocity + boost, affine, st, grid)
    occupied = np.asarray(grid_mass) > 0
    relative = 3e-5 if dtype.name == "float32" else 2e-12
    absolute = 2e-7 if dtype.name == "float32" else 2e-12
    expected_mass = np.zeros(grid.count)
    np.add.at(
        expected_mass,
        np.asarray(st.ids).reshape(-1),
        (np.asarray(mass, dtype=np.float64)[:, None] * np.asarray(st.weight)).reshape(-1),
    )
    for observed_mass in (grid_mass, shifted_mass):
        np.testing.assert_allclose(observed_mass, expected_mass, rtol=relative, atol=absolute)
    np.testing.assert_allclose(
        np.asarray(shifted_velocity)[occupied],
        np.asarray(grid_velocity)[occupied] + np.asarray(boost),
        rtol=relative,
        atol=absolute,
    )
    np.testing.assert_array_equal(np.asarray(shifted_velocity)[~occupied], 0.0)

    def transfer(m: jax.Array, v: jax.Array, c: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Keep the interpolation geometry fixed while differentiating physical inputs."""
        return mpm.p2g(m, v, c, st, grid)

    indices = np.asarray(st.ids).reshape(-1)
    weights = np.asarray(st.weight, dtype=np.float64)
    offsets = np.asarray(st.offset, dtype=np.float64)
    local = np.asarray(velocity, dtype=np.float64)[:, None, :] + np.einsum(
        "pij,pkj->pki", np.asarray(affine, dtype=np.float64), offsets
    )
    mass_tangent, velocity_tangent, affine_tangent = (
        np.asarray(value, dtype=np.float64) for value in tangents
    )
    local_tangent = velocity_tangent[:, None, :] + np.einsum(
        "pij,pkj->pki", affine_tangent, offsets
    )
    for current_mass in (mass, jnp.zeros_like(mass)):
        _, actual_tangent = jax.jvp(transfer, (current_mass, velocity, affine), tangents)
        host_mass = np.asarray(current_mass, dtype=np.float64)
        node_mass = np.zeros(grid.count)
        momentum = np.zeros((grid.count, dim))
        node_mass_tangent = np.zeros(grid.count)
        momentum_tangent = np.zeros((grid.count, dim))
        np.add.at(node_mass, indices, (host_mass[:, None] * weights).reshape(-1))
        np.add.at(
            momentum,
            indices,
            (host_mass[:, None, None] * weights[..., None] * local).reshape(-1, dim),
        )
        np.add.at(node_mass_tangent, indices, (mass_tangent[:, None] * weights).reshape(-1))
        np.add.at(
            momentum_tangent,
            indices,
            (
                weights[..., None]
                * (mass_tangent[:, None, None] * local + host_mass[:, None, None] * local_tangent)
            ).reshape(-1, dim),
        )
        safe_mass = np.where(node_mass > 0, node_mass, 1.0)
        safe_mass_tangent = np.where(node_mass > 0, node_mass_tangent, 0.0)
        expected_velocity_tangent = (
            momentum_tangent / safe_mass[:, None]
            - momentum * (safe_mass_tangent / safe_mass**2)[:, None]
        )
        np.testing.assert_allclose(
            actual_tangent[0], node_mass_tangent, rtol=relative, atol=absolute
        )
        np.testing.assert_allclose(
            actual_tangent[1], expected_velocity_tangent, rtol=relative, atol=absolute
        )
