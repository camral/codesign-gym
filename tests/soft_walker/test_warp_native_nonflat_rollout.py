"""Two accepted spring steps on a slope/rounded step, with implicit gradients.

JAX uses its small dense reference solve; native uses its normal implicit
GMRES path. CUDA selection affects Warp only; a CPU JAX oracle is sufficient.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from envs.soft_walker.engine.physics import mpm


import os
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

wp = pytest.importorskip("warp")

from envs.soft_walker.diagnostics.warp.solid_fixtures import fixture
from envs.soft_walker.engine.physics import spring, terrain
from envs.soft_walker.engine.warp_native.solid_conversion import from_jax
from envs.soft_walker.engine.warp_native.solid_episode import SolidEpisode
from envs.soft_walker.engine.warp_native.solid_model import SolidDeviceModel

PRECISION = os.environ.get("WARP_TEST_DTYPE", "float64")
DEVICE = os.environ.get("WARP_TEST_DEVICE", "cpu")


@pytest.mark.parametrize("kind", ["slope", "rounded_step"])
def test_nonflat_spring_rollout_forward_and_command_gradient(kind: str) -> None:
    original, model, state, cfg, _, _ = fixture("spring", PRECISION, contact=True)
    # Lift above the ground half-space so the rounded-corner branch is active.
    shift = jnp.asarray([0.0, 0.011], dtype=PRECISION)
    model = model._replace(reference=model.reference + shift)
    state = state._replace(
        x=state.x + shift, v=jnp.broadcast_to(jnp.asarray([0.025, -0.01], PRECISION), state.v.shape)
    )
    tc = (
        terrain.TerrainConfig(kind="slopes", blocks=0, height=0.0095, fixed_slopes=(0.03, 0.0))
        if kind == "slope"
        else terrain.TerrainConfig(
            kind="blocks",
            blocks=1,
            rounding=0.001,
            fixed_boxes=((0.0295, 0.002, 0.0, 0.021, 0.009, 0.1, 0.001),),
        )
    )
    model = model._replace(terrain=terrain.generate(tc, dim=2, dtype=np.dtype(PRECISION)))
    # Tight roots make the finite-difference probe meaningful in float32 too;
    # this is a fixture choice, not a change to production solver defaults.
    cfg = replace(cfg, root_tol32=3.0e-8)
    commands = np.asarray([[0.31, -0.22], [-0.13, 0.27]], PRECISION)
    initial_contact, _, _ = spring.contact(model, state.x, state.v, cfg)
    assert np.linalg.norm(np.asarray(initial_contact)) > 0.01
    gap, normal = terrain.gap_normal(state.x, model.terrain)
    if kind == "rounded_step":
        assert np.any((np.asarray(normal)[:, 0] > 0.1) & (np.asarray(normal)[:, 1] > 0.1)), normal

    def objective(actions: jax.Array) -> tuple[jax.Array, tuple[mpm.State, jax.Array, jax.Array]]:
        def advance(
            carry: tuple[mpm.State, jax.Array], action: jax.Array
        ) -> tuple[tuple[mpm.State, jax.Array], jax.Array]:
            old, travel = carry
            new, info = spring.substep(model, old, action, cfg)
            return (new, travel + cfg.dt * new.v), new.status

        (final, travel), statuses = jax.lax.scan(advance, (state, jnp.zeros_like(state.v)), actions)
        value = jnp.sum(model.mass * (-travel[:, 0] / 0.2 + 0.17 * jnp.sum(final.v**2, axis=-1)))
        value = value / jnp.sum(model.mass) + 0.09 * final.work_positive
        return value, (final, statuses, travel)

    compute = jax.jit(jax.value_and_grad(objective, has_aux=True))
    (loss, (final, statuses, travel)), gradient = compute(jnp.asarray(commands))
    assert np.all(np.asarray(statuses) == 0), statuses
    assert np.all(np.isfinite(np.asarray(gradient)))
    bundle = from_jax("spring", model, state, cfg, dt=original.config.dt)
    owner = SolidDeviceModel(bundle, DEVICE)
    episode = SolidEpisode(
        owner, commands, checkpoint_stride=1, velocity_weight=0.17, work_weight=0.09
    )
    result = episode.run(True)
    assert result["status"] == result["adjoint_status"] == 0, result
    assert np.all(episode.recording()["diagnostics"][:, 0] == 0)
    actual_gradient = episode.gradients()["commands"].copy()
    reference = dict(
        u=final.x - model.reference,
        v=final.v,
        actuator=final.actuator,
        rotation=np.broadcast_to(np.eye(2, dtype=PRECISION), (owner.n, 2, 2)),
        omega=np.zeros((owner.n, 2), PRECISION),
        ledger=jnp.stack((final.work_signed, final.work_positive, final.dissipated, final.time)),
    )
    rtol, atol = (5.0e-4, 4.0e-7) if PRECISION == "float32" else (2.0e-8, 2.0e-11)
    for name, expected in reference.items():
        np.testing.assert_allclose(
            getattr(episode.final, name).numpy(), expected, rtol=rtol, atol=atol, err_msg=name
        )
    np.testing.assert_allclose(episode.travel.numpy(), travel, rtol=rtol, atol=atol)
    np.testing.assert_allclose(result["loss"], loss, rtol=rtol, atol=atol)
    # Control derivatives are smaller than the absolute state tolerance; give
    # them a separate meaningful check instead of letting that allowance hide
    # an absent or incorrectly scaled gradient.
    np.testing.assert_allclose(
        actual_gradient,
        gradient,
        rtol=3.0e-3 if PRECISION == "float32" else 2.0e-7,
        atol=1.0e-11 if PRECISION == "float32" else 1.0e-13,
    )
    assert np.linalg.norm(actual_gradient) > 1.0e-8

    # Use a direction with a resolved signal. An arbitrary near-orthogonal
    # direction makes float32 subtraction measure roundoff rather than the VJP.
    direction = actual_gradient.copy()
    direction /= np.linalg.norm(direction)
    h = 0.08 if PRECISION == "float32" else 0.002
    values = []
    for sign in [-1.0, 1.0]:
        episode.set_commands(commands + sign * h * direction)
        perturbed = episode.run(False)
        assert perturbed["status"] == 0, perturbed
        values.append(perturbed["loss"])
    fd = (values[1] - values[0]) / (2.0 * h)
    derivative = np.sum(actual_gradient * direction)
    np.testing.assert_allclose(
        derivative,
        fd,
        rtol=7.0e-3 if PRECISION == "float32" else 2.0e-4,
        atol=1.0e-11 if PRECISION == "float32" else 1.0e-12,
    )
