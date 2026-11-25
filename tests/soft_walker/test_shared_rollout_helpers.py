"""Extracted assembly and loss helpers retain physical and derivative contracts."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from envs.soft_walker.contracts.metrics import scenario_metric_layout
from envs.soft_walker.engine import make
from envs.soft_walker.engine.drivers.environment import spawn_shift
from envs.soft_walker.engine.execution.preparation import assemble_spring
from envs.soft_walker.engine.physics import spring
from envs.soft_walker.engine.physics.spring_config import SpringConfig
from envs.soft_walker.engine.physics.terrain import TerrainConfig
from envs.soft_walker.experiments.scenarios.objectives import running_cost, terminal_cost
from envs.soft_walker.presets.scenarios.config import Scenario


@pytest.mark.parametrize("driver", ["mpm2", "mpm3"])
@pytest.mark.parametrize("gait", ["walk", "bound", "pronk"])
def test_scenario_cost_values_and_gradients_keep_historical_formula(
    driver: str, gait: str, dtype: np.dtype[np.floating]
) -> None:
    """Both row sensitivity and accumulated-cost sensitivity survive extraction."""
    case = Scenario(driver=driver, gait=gait)
    row = jnp.asarray(
        np.linspace(-0.02, 0.04, len(scenario_metric_layout(case.dim).columns())), dtype
    )
    cost = jnp.asarray(0.13, dtype)
    work_reference = 0.73

    def original(row: jax.Array, cost: jax.Array) -> jax.Array:
        shortage = 0.05 * jax.nn.softplus((0.045 - row[7]) / 0.005)
        height = row[2] / case.size[1] if case.gait in ("bound", "pronk") else jnp.asarray(0, dtype)
        accumulated = cost + (0.25 * shortage**2 - 0.15 * height) / case.steps
        return -row[1] / case.size[0] + 0.001 * row[4] / work_reference + accumulated

    def extracted(row: jax.Array, cost: jax.Array) -> jax.Array:
        accumulated = cost + running_cost(row, case, jnp.asarray(0, dtype))
        return terminal_cost(row, case, work_reference, accumulated)

    before = jax.jit(jax.value_and_grad(original, argnums=(0, 1)))(row, cost)
    after = jax.jit(jax.value_and_grad(extracted, argnums=(0, 1)))(row, cost)
    for expected, actual in zip(jax.tree.leaves(before), jax.tree.leaves(after), strict=True):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("terrain_kind", [None, "slopes"])
def test_spring_assembly_preserves_fields_and_spawn_without_mutating_env(
    terrain_kind: str | None, dtype: np.dtype[np.floating]
) -> None:
    config = SpringConfig(
        grid_shape=(2, 1),
        settle_steps=0,
        substeps=1,
        horizon=2,
        terrain=None if terrain_kind is None else TerrainConfig(kind=terrain_kind, blocks=0),
    )
    env, embedding = make(config=config, dtype=dtype)
    try:
        assert not env.reset(seed=31, embodiment=embedding)[1][0]
        live_state = env._state
        values = jax.tree.map(jnp.asarray, embedding)
        fields = env.decoder.decode(values, env._topology.points)
        expected = spring.instantiate(fields, env._topology, config, env.channels)
        if env._terrain is not None:
            expected = expected._replace(
                reference=expected.reference.at[:, 1].add(
                    spawn_shift(env._terrain, config.size) - config.ground_height
                ),
                terrain=env._terrain,
            )
        actual, actual_fields = jax.jit(
            lambda emb: assemble_spring(
                env.decoder, emb, env._topology, config, env.channels, env._terrain
            )
        )(values)
        for before, after in zip(
            jax.tree.leaves((expected, fields)),
            jax.tree.leaves((actual, actual_fields)),
            strict=True,
        ):
            np.testing.assert_allclose(
                after, before, rtol=2e-6 if dtype == np.float32 else 2e-14, atol=0
            )
        assert env._state is live_state
    finally:
        env.close()
