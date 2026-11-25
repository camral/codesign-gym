"""Independent JAX-native Kvaerno reference for native spring stages/adjoints."""

from __future__ import annotations

import os
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

wp = pytest.importorskip("warp")
from envs.soft_walker.engine import make
from envs.soft_walker.engine.physics import spring
from envs.soft_walker.engine.physics.spring_config import SpringConfig
from envs.soft_walker.engine.warp_native.solid_conversion import from_jax
from envs.soft_walker.engine.warp_native.solid_episode import SolidEpisode
from envs.soft_walker.engine.warp_native.solid_model import SolidConfig, SolidDeviceModel

DEVICE = os.environ.get("WARP_TEST_DEVICE", "cpu")
DTYPE = os.environ.get("WARP_TEST_DTYPE", "float64")


@pytest.mark.parametrize("ground", [False, True])
def test_native_kvaerno_matches_jax_forward_and_control_gradient(ground: bool) -> None:
    """Compare changing controls, nonzero initial motion and actual travel loss."""
    cfg = SpringConfig(
        grid_shape=(1, 1),
        size=(0.08, 0.05),
        settle_steps=0,
        substeps=1,
        control_dt=0.002,
        ground=ground,
        gravity=0.7,
        integrator="kvaerno3_native",
        linear_solver="gmres",
    )
    env, fields = make("SoftSpring2D-v0", dtype=DTYPE, device="cpu", config=cfg)
    try:
        assert not env.reset(embodiment=fields)[1].any()
        model = env._model
        state = env._state._replace(
            v=jnp.asarray(
                [[0.02, 0.01], [-0.01, 0.015], [0.03, -0.01], [0.01, -0.02]], dtype=DTYPE
            ),
            actuator=jnp.asarray([0.03, -0.02], dtype=DTYPE),
        )
        bundle = from_jax("spring", model, state, cfg)
        assert bundle.config.integrator == "kvaerno3_native"
        commands = np.asarray([[0.23, -0.11], [-0.16, 0.08], [0.19, -0.07]], dtype=DTYPE)
        direction = np.asarray([0.8, 0.6], dtype=DTYPE)

        def reference(table: jax.Array) -> tuple[jax.Array, spring.State]:
            def step(current: spring.State, command: jax.Array) -> tuple[spring.State, None]:
                return spring.substep(model, current, command, cfg)[0], None

            final, _ = jax.lax.scan(step, state, table)
            loss = -jnp.dot(jnp.asarray(direction), final.com_displacement) / 0.2
            loss += 0.04 * jnp.sum(model.mass * jnp.sum(final.v**2, axis=1)) / jnp.sum(model.mass)
            loss += 0.001 * final.work_positive
            return loss, final

        (loss, expected), gradient = jax.jit(jax.value_and_grad(reference, has_aux=True))(
            jnp.asarray(commands)
        )
        episode = SolidEpisode(
            SolidDeviceModel(bundle, DEVICE),
            commands,
            direction=direction,
            length_ref=0.2,
            velocity_weight=0.04,
            work_weight=0.001,
            checkpoint_stride=2,
        )
        result = episode.run(gradient=True)
        assert result["status"] == result["adjoint_status"] == int(expected.status) == 0
        rtol, atol = (6e-4, 3e-6) if DTYPE == "float32" else (2e-8, 2e-10)
        np.testing.assert_allclose(
            episode.final.u.numpy(), np.asarray(expected.x - model.reference), rtol=rtol, atol=atol
        )
        np.testing.assert_allclose(episode.final.v.numpy(), expected.v, rtol=rtol, atol=atol)
        np.testing.assert_allclose(
            episode.final.actuator.numpy(), expected.actuator, rtol=rtol, atol=atol
        )
        np.testing.assert_allclose(episode.loss.numpy()[0], loss, rtol=rtol, atol=atol)
        np.testing.assert_allclose(
            episode.gradients()["commands"],
            gradient,
            rtol=2e-3 if DTYPE == "float32" else 2e-5,
            atol=3e-7 if DTYPE == "float32" else 2e-10,
        )
        # Whole-step backward Euler displacement would produce the wrong result
        # under gravity, even when the velocity state itself looked plausible.
        average_travel = (
            np.sum(bundle.fields["mass"][:, None] * episode.travel.numpy(), axis=0)
            / bundle.fields["mass"].sum()
        )
        np.testing.assert_allclose(average_travel, expected.com_displacement, rtol=rtol, atol=atol)
        recomputed = SolidEpisode(
            SolidDeviceModel(bundle, DEVICE),
            commands,
            direction=direction,
            length_ref=0.2,
            velocity_weight=0.04,
            work_weight=0.001,
            checkpoint_stride=2,
            reuse_replay_roots=False,
        )
        assert recomputed.run(True)["adjoint_status"] == 0
        np.testing.assert_allclose(
            recomputed.loss.numpy(), episode.loss.numpy(), rtol=rtol, atol=atol
        )
        np.testing.assert_allclose(
            recomputed.gradients()["commands"],
            episode.gradients()["commands"],
            rtol=rtol,
            atol=atol,
        )
        if episode.device.is_cuda and not ground:
            from envs.soft_walker.engine.execution.cupti import DriverAudit

            stream = wp.get_stream(episode.device)
            wp.synchronize_stream(stream)
            with DriverAudit() as observer:
                wp.capture_launch(episode.graphs["gradient"], stream=stream)
                wp.synchronize_stream(stream)
            assert observer.report()["passed"], observer.report()
    finally:
        env.close()


def test_kvaerno_resolved_material_geometry_and_initial_state_derivatives() -> None:
    """Check independent-array sensitivities against JAX and a centered FD ladder."""
    if DTYPE != "float64":
        pytest.skip("Resolved-array FD ladder is a float64 accuracy check")
    cfg = SpringConfig(
        grid_shape=(1, 1),
        size=(0.08, 0.05),
        settle_steps=0,
        substeps=1,
        control_dt=0.002,
        ground=False,
        gravity=0.7,
        integrator="kvaerno3_native",
        linear_solver="gmres",
    )
    env, design = make("SoftSpring2D-v0", dtype=DTYPE, device="cpu", config=cfg)
    try:
        assert not env.reset(embodiment=design)[1].any()
        model, initial = env._model, env._state
        pattern = np.asarray([[1.0, -0.4], [-0.7, 0.3], [0.2, 0.8], [-0.1, -0.5]])
        initial = initial._replace(v=jnp.asarray(0.02 * pattern))
        bundle = from_jax("spring", model, initial, cfg)
        commands = np.asarray([[0.23, -0.11], [-0.16, 0.08], [0.19, -0.07]])
        directions = {
            "mass": 0.2 * bundle.fields["mass"] * np.asarray([1.0, -0.5, 0.3, -0.2]),
            "stiffness": 0.15
            * bundle.fields["stiffness"]
            * np.linspace(-0.6, 0.8, len(model.edges)),
            "damping": 0.15 * bundle.fields["damping"],
            "length": 0.02 * bundle.fields["length"],
            "routing": 0.1 * bundle.fields["routing"] + 0.002,
            "reference": 0.001 * pattern,
        }
        state_directions = {
            "u": 0.001 * pattern[::-1],
            "v": 0.01 * pattern,
            "actuator": np.asarray([0.015, -0.009]),
        }
        keys, state_keys = tuple(directions), tuple(state_directions)

        def reference(theta: jax.Array) -> jax.Array:
            values = {
                key: jnp.asarray(bundle.fields[key]) + theta[i] * jnp.asarray(directions[key])
                for i, key in enumerate(keys)
            }
            varied_model = model._replace(
                reference=values["reference"],
                mass=values["mass"],
                stiffness=values["stiffness"],
                damping=values["damping"],
                length=values["length"],
                logstrain_map=values["routing"],
            )
            states = {
                key: jnp.asarray(bundle.initial[key])
                + theta[len(keys) + i] * jnp.asarray(state_directions[key])
                for i, key in enumerate(state_keys)
            }
            state = initial._replace(
                x=values["reference"] + states["u"],
                v=states["v"],
                actuator=states["actuator"],
            )

            def step(current: spring.State, command: jax.Array) -> tuple[spring.State, None]:
                return spring.substep(varied_model, current, command, cfg)[0], None

            final, _ = jax.lax.scan(step, state, jnp.asarray(commands))
            loss = -jnp.dot(jnp.asarray([0.8, 0.6]), final.com_displacement) / 0.2
            loss += (
                0.04
                * jnp.sum(values["mass"] * jnp.sum(final.v**2, axis=1))
                / jnp.sum(values["mass"])
            )
            return loss + 0.001 * final.work_positive

        theta = jnp.zeros(len(keys) + len(state_keys), dtype=jnp.float64)
        expected = np.asarray(jax.jit(jax.grad(reference))(theta))
        episode = SolidEpisode(
            SolidDeviceModel(bundle, DEVICE),
            commands,
            direction=np.asarray([0.8, 0.6]),
            velocity_weight=0.04,
            work_weight=0.001,
            checkpoint_stride=2,
        )
        assert episode.run(True)["adjoint_status"] == 0
        gradients = episode.gradients()
        actual = np.asarray(
            [np.sum(gradients["model"][key] * directions[key]) for key in keys]
            + [np.sum(gradients["initial"][key] * state_directions[key]) for key in state_keys]
        )
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-9)

        # Independent finite differences perturb all three groups together;
        # the analytic comparison above still checks each direction separately.
        def native_loss(scale: float) -> float:
            for key in keys:
                getattr(episode.owner.model, key).assign(
                    bundle.fields[key] + scale * directions[key]
                )
            for key in state_keys:
                getattr(episode.owner.initial, key).assign(
                    bundle.initial[key] + scale * state_directions[key]
                )
            result = episode.run(False)
            assert result["status"] == 0
            return result["loss"]

        for epsilon in (2e-3, 1e-3, 5e-4):
            finite_difference = (native_loss(epsilon) - native_loss(-epsilon)) / (2 * epsilon)
            np.testing.assert_allclose(finite_difference, actual.sum(), rtol=3e-4, atol=2e-8)
    finally:
        env.close()


def test_failed_kvaerno_stage_rolls_back_the_entire_step() -> None:
    """A deliberately insufficient Newton budget cannot expose a partial stage."""
    from envs.soft_walker.engine.warp_native.solid_builders import regular_solid

    bundle = regular_solid("spring", shape=(1, 1), precision=DTYPE)
    bundle.config = replace(
        bundle.config,
        integrator="kvaerno3_native",
        dt=0.05,
        max_newton=1,
        contact_enabled=False,
        gravity=0.7,
    )
    owner = SolidDeviceModel(bundle, DEVICE)
    owner.initial.v.assign(
        np.asarray([[0.2, 0.1], [-0.1, 0.15], [0.3, -0.1], [0.1, -0.2]], dtype=DTYPE)
    )
    episode = SolidEpisode(owner, np.asarray([[0.95, -0.8]], dtype=DTYPE))
    result = episode.run(False)
    assert result["status"] != 0
    for key in ("u", "v", "rotation", "omega", "actuator", "ledger"):
        np.testing.assert_array_equal(
            getattr(episode.final, key).numpy(), getattr(owner.initial, key).numpy()
        )
    np.testing.assert_array_equal(episode.travel.numpy(), 0.0)


def test_kvaerno_selection_is_explicit_and_spring_only() -> None:
    """No alternate integrator silently replaces the configured physical step."""
    assert SolidConfig().integrator == "backward_euler"
    SolidConfig(integrator="kvaerno3_native")
    with pytest.raises(ValueError, match="integrator|spring"):
        SolidConfig(integrator="kvaerno3")
    for physics in ("beam", "fem"):
        with pytest.raises(ValueError, match="integrator|spring"):
            replace(
                SolidConfig(grid_shape=(4, 4, 4), origin=(0.0,) * 3, physics=physics),
                integrator="kvaerno3_native",
            )
    from envs.soft_walker.engine.warp_native.solid_builders import regular_solid
    from envs.soft_walker.engine.warp_native.solid_step import SolidStep

    bundle = regular_solid("spring", shape=(1, 1), precision=DTYPE)
    bundle.config = replace(bundle.config, integrator="kvaerno3_native")
    with pytest.raises(ValueError, match="SpringKvaernoStep"):
        SolidStep(SolidDeviceModel(bundle, DEVICE))
