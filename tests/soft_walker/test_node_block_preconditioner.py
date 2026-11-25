"""Local block assembly and nonsymmetric preconditioner transpose correctness."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from envs.soft_walker.engine.embodiments.decoders import Fields
from envs.soft_walker.engine.numerics.solvers import NewtonConfig, linear_solve
from envs.soft_walker.engine.physics import mpm
from envs.soft_walker.engine.physics.common import DynamicsConfig
from envs.soft_walker.engine.physics.contact import PlaneContact
from envs.soft_walker.engine.physics.mpm_support import compact_support
from envs.soft_walker.engine.physics.mpm_tangent import grid_tangent_blocks, inverse_node_blocks


@pytest.mark.parametrize("side", ["left", "right"])
def test_nonsymmetric_blocks_preserve_linear_primal_and_transpose(
    side: str, dtype: np.dtype[np.floating]
) -> None:
    """Both preconditioner sides solve the true operator and its implicit transpose."""
    matrix = jnp.asarray(
        [[4, 0.7, 0.1, 0], [-0.2, 3, 0, 0.2], [0.3, 0, 2, -0.5], [0, 0.1, 0.8, 4]], dtype
    )
    rhs = jnp.asarray([0.2, -0.3, 0.4, 0.1], dtype)
    cotangent = jnp.asarray([0.4, 0.1, -0.2, 0.3], dtype)
    blocks = jnp.stack((matrix[:2, :2], matrix[2:, 2:]))
    inverse = inverse_node_blocks(blocks)
    config = NewtonConfig(gmres_method="fused", preconditioner_side=side)

    def solve(value: jax.Array) -> jax.Array:
        """Keep the preconditioner fixed while differentiating the linear right-hand side."""
        return linear_solve(
            lambda x: matrix @ x, value, jnp.ones_like(value), config, inverse_blocks=inverse
        )[0]

    actual = jax.jit(solve)(rhs)
    derivative = jax.jit(jax.grad(lambda value: jnp.vdot(solve(value), cotangent)))(rhs)
    tolerance = 3e-5 if dtype.name == "float32" else 2e-9
    np.testing.assert_allclose(actual, np.linalg.solve(matrix, rhs), rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(
        derivative, np.linalg.solve(np.asarray(matrix).T, cotangent), rtol=tolerance, atol=tolerance
    )
    # Unequal off-diagonal entries make accidental use of D^-1 instead of D^-T visible.
    assert not np.allclose(np.asarray(inverse), np.asarray(inverse).swapaxes(-1, -2))


@pytest.mark.parametrize("dim", [2, 3])
def test_mpm_node_blocks_match_exact_contact_momentum_jacobian(
    dim: int, dtype: np.dtype[np.floating]
) -> None:
    """Each assembled block matches the corresponding diagonal of a tiny dense oracle."""
    cloud = mpm.uniform_cloud(
        cell_shape=(1,) * dim,
        particles_per_axis=1,
        origin=(0.025, -0.0001) + ((0.025,) if dim == 3 else ()),
        dtype=dtype,
    )
    one = jnp.ones(1, dtype)
    fields = Fields(
        one,
        one * 20000,
        one * 1000,
        one * 0.5,
        jnp.broadcast_to(jnp.eye(dim, dtype=dtype)[1], (1, dim)),
        jnp.ones(1, jnp.int32),
        jnp.zeros(1, jnp.int32),
    )
    model = mpm.instantiate(cloud, fields, channels=1)
    state = mpm.initial_state(model)
    state = state._replace(F=state.F.at[:, 0, 1].set(0.01))
    grid = mpm.Grid(shape=(8,) * dim, origin=(-0.05,) * dim)
    st = mpm.stencil(state.x, grid)
    count = 3**dim
    compact = compact_support(st.ids, model.active, grid_nodes=grid.count, capacity=count)
    st = st._replace(ids=compact.ids)
    mass, initial_velocity = mpm.p2g(model.mass, state.v, state.C, st, grid, node_count=count)
    velocity = jnp.asarray(np.random.default_rng(11).normal(scale=0.02, size=(count, dim)), dtype)
    config = DynamicsConfig(
        actuation_model="active_stress", contact=PlaneContact(enabled=True, friction=0.6)
    )
    actuator = jnp.asarray([0.05], dtype)
    dt = jnp.asarray(0.00025, dtype)
    q0, weights = mpm.surface_mapping(model, state, st)

    def residual(v: jax.Array) -> jax.Array:
        """Independent full momentum residual used only for this tiny Jacobian oracle."""
        _, _, gradient = mpm.g2p(v, st, grid)
        delta = state.F - jnp.eye(dim, dtype=dtype) + dt * (gradient @ state.F)
        force = mpm.internal_grid_force(
            model,
            jnp.eye(dim, dtype=dtype) + delta,
            state.F,
            actuator,
            st,
            grid,
            config,
            delta_F=delta,
            node_count=count,
        )
        force = force + mpm.grid_contact(model, q0, weights, v, st, grid, dt, config)[0]
        value = v - initial_velocity - dt * force / jnp.where(mass > 0, mass, 1)[:, None]
        return jnp.where((mass > 0)[:, None], value, v)

    jacobian = jax.jit(jax.jacfwd(residual))(velocity)
    expected = jacobian[jnp.arange(count), :, jnp.arange(count), :]
    actual = jax.jit(
        lambda v: grid_tangent_blocks(model, state, actuator, v, mass, st, dt, config)
    )(velocity)
    tolerance = 3e-5 if dtype.name == "float32" else 2e-9
    np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)


def test_singular_blocks_do_not_hide_linear_failure(dtype: np.dtype[np.floating]) -> None:
    """A singular preconditioner fails explicitly instead of substituting another solve."""
    inverse = inverse_node_blocks(jnp.zeros((1, 2, 2), dtype))
    result, information = jax.jit(
        lambda: linear_solve(
            lambda x: x,
            jnp.ones(2, dtype),
            jnp.ones(2, dtype),
            NewtonConfig(gmres_method="fused"),
            inverse_blocks=inverse,
        )
    )()
    assert not bool(information[1])
    assert np.isnan(result).all()
