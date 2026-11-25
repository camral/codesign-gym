"""Embodiment chain rules must include assembly, not just copied field metadata."""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from envs.soft_walker.engine import make
from envs.soft_walker.engine.drivers.task import BodyConfig
from envs.soft_walker.engine.numerics.solvers import NewtonConfig
from envs.soft_walker.engine.physics.common import DynamicsConfig
from envs.soft_walker.engine.physics.contact import PlaneContact
from envs.soft_walker.engine.physics.spring_config import SpringConfig
from envs.soft_walker.engine.warp_native.embodiment_bridge import EmbodimentEpisode


def _fixture(kind: str, dtype: str, decoder: str = "direct") -> tuple:
    """Small contact-free fixtures isolate decoder/assembly and adjoint behavior."""
    identifiers = dict(
        spring="SoftSpring2D-v0",
        beam="SoftBeam3D-v0",
        fem="SoftHexFEM3D-v0",
        mpm2="SoftMPM2D-v0",
        mpm3="SoftMPM3D-v0",
    )
    dim = 2 if kind in ("spring", "mpm2") else 3
    if kind == "spring":
        cfg = SpringConfig(
            grid_shape=(2, 1),
            size=(0.1, 0.05),
            control_dt=0.00025,
            substeps=1,
            settle_steps=0,
            horizon=3,
            ground=False,
            gravity=3.0,
        )
    else:
        cfg = BodyConfig(
            grid_shape=(2,) + (1,) * (dim - 1),
            size=(0.1,) + (0.05,) * (dim - 1),
            control_dt=0.00025,
            substeps=1,
            settle_steps=0,
            horizon=3,
            dynamics=DynamicsConfig(
                max_halvings=0,
                gravity=3.0,
                contact=PlaneContact(enabled=False),
                newton=NewtonConfig(preconditioner_side="right"),
            ),
        )
    env, body = make(
        identifiers[kind],
        config=cfg,
        dtype=dtype,
        device="cpu",
        decoder=decoder,
        design_mode="relaxed",
        max_observation_space_size=10000,
    )
    commands = np.full((2, env.channels), 0.15, dtype)
    if decoder == "primitives":
        # The symmetric seed ties two discrete routing scores. Select a channel
        # before probing continuous derivatives; labels are not differentiable.
        channel_start = 10 + dim - 1  # Features omit the occupancy head.
        body["features"][:, channel_start : env.decoder.spec.poisson_index - 1] = -2
        body["features"][:, channel_start] = 2
    bridge = EmbodimentEpisode(
        env,
        body,
        commands,
        device=os.environ.get("WARP_TEST_DEVICE", "cpu"),
        velocity_weight=0.7,
        checkpoint_stride=2,
    )
    return env, body, commands, bridge


@pytest.mark.parametrize("kind", ("spring", "beam", "fem", "mpm2", "mpm3"))
def test_resolved_assembly_matches_export(kind: str, dtype: np.dtype) -> None:
    """Every consumed coefficient has the original export's value and shape."""
    env, body, _, bridge = _fixture(kind, dtype.name)
    try:
        fields, initial = bridge._assemble(bridge._validate(body))
        for name, value in fields.items():
            np.testing.assert_allclose(
                value, bridge.bundle.fields[name], rtol=3e-6, atol=1e-12, err_msg=name
            )
        for name, value in initial.items():
            np.testing.assert_allclose(
                value, bridge.bundle.initial[name], rtol=3e-6, atol=1e-12, err_msg=name
            )
    finally:
        env.close()


@pytest.mark.parametrize("decoder", ("direct", "bspline", "cppn", "primitives"))
def test_decoder_assembly_pullback(decoder: str, dtype: np.dtype) -> None:
    """A weighted assembled beam loss tests geometry/material/routing chain rules."""
    env, body, _, bridge = _fixture("beam", dtype.name, decoder)
    try:
        parameters = bridge._validate(body)
        (fields, _), pullback = jax.vjp(bridge._assemble, parameters)
        rng = np.random.default_rng(84)
        weights = {
            name: jnp.asarray(rng.normal(size=value.shape), dtype) / (1 + jnp.abs(value))
            for name, value in fields.items()
        }
        initial = bridge._assemble(parameters)[1]
        (grad,) = pullback((weights, jax.tree.map(jnp.zeros_like, initial)))
        direction = {
            name: jnp.asarray(rng.normal(size=value.shape), dtype) * 0.05
            for name, value in parameters.items()
        }
        expected = sum(jnp.vdot(grad[k], direction[k]) for k in grad)

        def loss(p: dict[str, jax.Array]) -> jax.Array:
            f, _ = bridge._assemble(p)
            return sum(jnp.vdot(weights[k], f[k]) for k in f)

        epsilon = 0.01 if dtype.name == "float32" else 1e-4
        plus = jax.tree.map(lambda p, d: p + epsilon * d, parameters, direction)
        minus = jax.tree.map(lambda p, d: p - epsilon * d, parameters, direction)
        observed = (loss(plus) - loss(minus)) / (2 * epsilon)
        np.testing.assert_allclose(
            observed,
            expected,
            rtol=0.025 if dtype.name == "float32" else 2e-4,
            atol=0.001 if dtype.name == "float32" else 1e-6,
        )
    finally:
        env.close()


@pytest.mark.parametrize("kind", ("spring", "beam", "fem", "mpm2", "mpm3"))
def test_native_episode_embodiment_directional_derivative(kind: str, dtype: np.dtype) -> None:
    """An actual native episode closes the decoder-to-physics chain rule."""
    env, body, commands, bridge = _fixture(kind, dtype.name)
    try:
        report, grad, command_grad = bridge.evaluate(body, commands, gradient=True)
        assert report["status"] == report["adjoint_status"] == 0
        assert grad is not None and command_grad is not None
        rng = np.random.default_rng(15)
        directions = {
            key: rng.normal(size=value.shape).astype(dtype) * 0.01 for key, value in grad.items()
        }
        epsilon = 0.02 if dtype.name == "float32" else 0.001
        variants = [
            {
                key: value + sign * epsilon * directions[key] if key in directions else value.copy()
                for key, value in body.items()
            }
            for sign in (1, -1)
        ]
        losses = [bridge.evaluate(candidate, commands)[0]["loss"] for candidate in variants]
        fd = (losses[0] - losses[1]) / (2 * epsilon)
        ad = sum(float(np.vdot(grad[key], directions[key])) for key in grad)
        np.testing.assert_allclose(
            fd,
            ad,
            rtol=0.08 if dtype.name == "float32" else 0.003,
            atol=1e-7 if dtype.name == "float32" else 3e-11,
        )
        with pytest.raises(ValueError, match="shape/dtype"):
            bad = dict(body)
            key = next(iter(bad))
            bad[key] = np.asarray(bad[key]).ravel()[:1]
            bridge.evaluate(bad, commands)
    finally:
        env.close()
