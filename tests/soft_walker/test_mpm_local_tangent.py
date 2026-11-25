"""Exact local constitutive caching and its material/control transpose derivatives."""

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
from envs.soft_walker.engine.physics.mpm_tangent import cached_particle_stress, particle_stress


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("law", ["active_strain", "active_stress"])
def test_cached_stress_jvp_and_all_parameter_transpose(
    dim: int, law: str, dtype: np.dtype[np.floating]
) -> None:
    """Caching agrees with ordinary autodiff for deformation and every continuous field."""
    rng = np.random.default_rng(42)
    width = dim * dim + dim + 4
    parameters = jnp.asarray(rng.normal(scale=0.03, size=(3, width)), dtype)
    tangent = jnp.asarray(rng.normal(size=parameters.shape), dtype)
    cotangent = jnp.asarray(rng.normal(size=(3, dim, dim)), dtype)
    kinds = jnp.arange(3, dtype=jnp.int32)

    def stress(p: jax.Array, cached: bool) -> jax.Array:
        """Unpack nonuniform physical fields without discarding parameter derivatives."""
        delta = p[:, : dim * dim].reshape(3, dim, dim)
        offset = dim * dim
        evaluate = cached_particle_stress if cached else particle_stress
        return evaluate(
            delta,
            20000.0 * jnp.exp(p[:, offset]),
            0.35 + 0.05 * jnp.tanh(p[:, offset + 1]),
            p[:, offset + 2],
            p[:, offset + 4 :] + jnp.eye(dim, dtype=dtype)[0],
            kinds,
            1000.0 * p[:, offset + 3],
            law,
        )

    reference, reference_jvp = jax.jit(
        lambda p, t: jax.jvp(lambda x: stress(x, False), (p,), (t,))
    )(parameters, tangent)
    actual, actual_jvp = jax.jit(lambda p, t: jax.jvp(lambda x: stress(x, True), (p,), (t,)))(
        parameters, tangent
    )
    reference_vjp = jax.jit(jax.grad(lambda p: jnp.sum(stress(p, False) * cotangent)))(parameters)
    actual_vjp = jax.jit(jax.grad(lambda p: jnp.sum(stress(p, True) * cotangent)))(parameters)
    tolerance = 3e-5 if dtype.name == "float32" else 2e-10
    # CUDA can fuse the same primal expressions differently when the custom
    # tangent is present. Permit only rounding-scale differences (observed one
    # float64 ulp), separately from the material derivative acceptance below.
    np.testing.assert_allclose(actual, reference, rtol=4 * np.finfo(dtype).eps, atol=0.0)
    np.testing.assert_allclose(actual_jvp, reference_jvp, rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(actual_vjp, reference_vjp, rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(
        jnp.vdot(actual_jvp, cotangent),
        jnp.vdot(tangent, actual_vjp),
        rtol=tolerance,
        atol=tolerance,
    )


@pytest.mark.parametrize("law", ["active_strain", "active_stress"])
@pytest.mark.parametrize("blocks", ["off", "predictor", "newton"])
def test_cached_particle_blocks_preserve_implicit_step_and_gradient(
    law: str, blocks: str, dtype: np.dtype[np.floating]
) -> None:
    """The cached constitutive path retains the same converged root and implicit adjoint."""
    cloud = mpm.uniform_cloud(cell_shape=(1, 1), dtype=dtype)
    ones = jnp.ones(len(cloud.reference), dtype)
    fields = Fields(
        ones,
        20000 * ones,
        1000 * ones,
        0.5 * ones,
        jnp.broadcast_to(jnp.asarray([0.2, 1.0], dtype), (len(ones), 2)),
        jnp.ones(len(ones), jnp.int32),
        jnp.arange(len(ones), dtype=jnp.int32) % 2,
    )
    model = mpm.instantiate(cloud, fields)
    state = mpm.initial_state(model)
    state = state._replace(F=state.F.at[:, 0, 1].set(0.003))
    grid = mpm.Grid(shape=(8, 8))
    config = DynamicsConfig(
        gravity=0,
        actuation_model=law,
        newton=NewtonConfig(gmres_method="fused"),
        mpm_compact_nodes=24,
    )
    controls = jnp.asarray([0.1, -0.05], dtype)

    def evaluate(command: jax.Array, cached: bool) -> tuple[jax.Array, mpm.State]:
        """Use a deformation/velocity objective with visible primal failure."""
        out, info = mpm.trial(
            model,
            state,
            command,
            0.00025,
            grid,
            replace(
                config,
                mpm_cache_tangent=cached,
                mpm_block_preconditioner=cached and blocks != "off",
                mpm_preconditioner_update="newton" if blocks == "newton" else "predictor",
            ),
        )
        value = jnp.sum(out.v**2) + jnp.sum(out.F**2) + out.work_positive
        return jnp.where(info.accepted == 1, value, jnp.nan), out

    expected, expected_gradient = jax.jit(
        jax.value_and_grad(lambda p: evaluate(p, False), has_aux=True)
    )(controls)
    actual, actual_gradient = jax.jit(
        jax.value_and_grad(lambda p: evaluate(p, True), has_aux=True)
    )(controls)
    assert np.isfinite(actual_gradient).all() and np.isfinite(expected_gradient).all()
    tolerance = 3e-5 if dtype.name == "float32" else 2e-9
    absolute = 2e-7 if dtype.name == "float32" else 2e-12
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        np.testing.assert_allclose(a, b, rtol=tolerance, atol=absolute)
    np.testing.assert_allclose(actual_gradient, expected_gradient, rtol=tolerance, atol=absolute)
