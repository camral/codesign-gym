"""Small physical-step checks for the new solver; no gait search is launched."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from envs.soft_walker.engine.types import DriverState


from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from envs.soft_walker.engine import make
from envs.soft_walker.engine.drivers.task import BodyConfig


@pytest.mark.parametrize(
    "driver", ["SoftMPM2D-v0", "SoftMPM3D-v0", "SoftHexFEM3D-v0", "SoftBeam3D-v0"]
)
def test_krylov_physical_step_and_action_gradient(
    driver: str, dtype: np.dtype[np.floating]
) -> None:
    from envs.soft_walker.engine.drivers import environment as drivers

    dim = 2 if driver == "SoftMPM2D-v0" else 3
    shape = (2, 1) if dim == 2 else (2, 1, 1)
    common = BodyConfig(
        grid_shape=shape,
        size=tuple(0.05 * n for n in shape),
        control_dt=0.00025,
        substeps=1,
        settle_steps=0,
        horizon=3,
        grid_margin=1,
    )
    env, base = make(driver, config=common, dtype=dtype, max_observation_space_size=4096)
    try:
        _, rejected, _ = env.reset(seed=12, embodiment=base)
        assert not rejected.any()
        initial, model = env._state, env._model
        action = jnp.asarray([0.21, -0.13], dtype)
        results = {}
        for method in ("incremental", "fused"):
            config = replace(
                common,
                dynamics=replace(
                    common.dynamics, newton=replace(common.dynamics.newton, gmres_method=method)
                ),
            )

            def step(a: jax.Array) -> DriverState:
                return drivers.advance(env.kind, model, initial, a, config, env._grid)[0]

            def feature(a: jax.Array) -> jax.Array:
                state = step(a)
                return 20 * jnp.sum(state.v**2) + 1e5 * state.body.work_signed

            result = jax.jit(step)(action)
            gradient = jax.jit(jax.grad(feature))(action)
            assert int(result.status) == 0 and np.isfinite(gradient).all()
            results[method] = (result, gradient)
        rtol, atol = (5e-4, 3e-6) if dtype == np.float32 else (2e-8, 2e-10)
        for x, y in zip(jax.tree.leaves(results["incremental"]), jax.tree.leaves(results["fused"])):
            np.testing.assert_allclose(x, y, rtol=rtol, atol=atol)
    finally:
        env.close()
        drivers.physics_kernels.cache_clear()
        drivers.kernels.cache_clear()
        drivers.instantiation_kernel.cache_clear()
        jax.clear_caches()


@pytest.mark.parametrize("integrator", ["backward_euler", "kvaerno3_native"])
def test_krylov_spring_step_and_gradient(integrator: str, dtype: np.dtype[np.floating]) -> None:
    from envs.soft_walker.engine.physics import spring
    from envs.soft_walker.engine.physics.spring_config import SpringConfig

    config = SpringConfig(
        grid_shape=(2, 1),
        size=(0.1, 0.05),
        substeps=1,
        control_dt=0.00025,
        settle_steps=0,
        horizon=3,
        integrator=integrator,
        linear_solver="gmres",
    )
    env, base = make("SoftSpring2D-v0", config=config, dtype=dtype)
    try:
        _, rejected, _ = env.reset(embodiment=base)
        assert not rejected.any()
        action = jnp.asarray([0.21, -0.13], dtype)
        outputs = []
        for method in ("incremental", "fused"):
            cfg = replace(config, gmres_method=method)

            def step(a: jax.Array) -> DriverState:
                return spring.advance(env._model, env._state, a, cfg)[0]

            def feature(a: jax.Array) -> jax.Array:
                state = step(a)
                return 20 * jnp.sum(state.v**2) + 1e5 * state.work_signed

            state = jax.jit(step)(action)
            gradient = jax.jit(jax.grad(feature))(action)
            assert int(state.status) == 0 and np.isfinite(gradient).all()
            outputs.append((state, gradient))
        rtol, atol = (5e-4, 3e-6) if dtype == np.float32 else (2e-8, 2e-10)
        for a, b in zip(jax.tree.leaves(outputs[0]), jax.tree.leaves(outputs[1])):
            np.testing.assert_allclose(a, b, rtol=rtol, atol=atol)
    finally:
        env.close()
        jax.clear_caches()
