"""FEM, beam, and spring local-cache primal and full-parameter derivative parity."""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from envs.soft_walker.engine.embodiments.decoders import Fields
from envs.soft_walker.engine.numerics.solvers import NewtonConfig
from envs.soft_walker.engine.physics import beam, fem, spring
from envs.soft_walker.engine.physics.common import DynamicsConfig
from envs.soft_walker.engine.physics.spring_config import SpringConfig


def _fields(points: jax.Array, parameters: jax.Array) -> Fields:
    """Use nonuniform continuous materials and actuation, including spatial Poisson values."""
    count, dim = points.shape
    shape = 1.0 + 0.1 * jnp.arange(count, dtype=points.dtype)
    direction = jnp.ones((count, dim), points.dtype).at[:, 0].set(0.2 + parameters[5])
    direction = direction / jnp.linalg.norm(direction, axis=1, keepdims=True)
    return Fields(
        0.8 * jnp.exp(0.1 * parameters[0]) * jnp.ones(count, points.dtype),
        20000 * jnp.exp(parameters[1]) * shape,
        1000 * jnp.exp(parameters[2]) * shape,
        0.5 * jnp.exp(parameters[3]) * shape,
        direction,
        jnp.ones(count, jnp.int32),
        jnp.arange(count, dtype=jnp.int32) % 2,
        0.3 + 0.05 * jnp.tanh(parameters[4] + points[:, 0]),
    )


@pytest.mark.parametrize("family", ["spring", "fem", "beam"])
def test_solid_cached_force_all_parameter_derivatives(
    family: str, dtype: np.dtype[np.floating]
) -> None:
    """Compare local caches against ordinary autodiff through geometry/material/control fields."""
    parameters = jnp.asarray(np.random.default_rng(7).normal(scale=0.02, size=12), dtype)
    tangent = jnp.asarray(np.random.default_rng(19).normal(size=12), dtype)
    config = DynamicsConfig(gravity=0)
    spring_config = SpringConfig(grid_shape=(1, 1), ground=False, gravity=0)
    spring_topology = spring.topology(spring_config, dtype)
    mesh = fem.structured_mesh(shape=(1, 1, 1), dtype=dtype)
    beam_topology = beam.topology((2, 1, 1), (0.1, 0.05, 0.05), (0.0, 0.01, 0.0), dtype)

    def response(p: jax.Array, cached: bool) -> jax.Array:
        """Keep all physical inputs live while changing only the local linearization option."""
        command = p[6:8] + jnp.asarray([0.1, -0.05], dtype)
        deformation = 0.01 + 0.02 * p[8]
        if family == "spring":
            model = spring.instantiate(
                _fields(spring_topology.points, p), spring_topology, spring_config, 2
            )
            x = model.reference * (1 + deformation)
            v = model.reference * (0.02 + p[9])
            return spring.forces(
                model, x, v, command, replace(spring_config, spring_cache_tangent=cached)
            )[0].reshape(-1)
        if family == "fem":
            fem_model = fem.instantiate(mesh, _fields(mesh.points, p))
            u = mesh.reference * deformation
            u = u.at[:, 0].add(mesh.reference[:, 1] * (0.02 + p[9]))
            return fem.internal_force(
                fem_model, u, command, replace(config, fem_cache_tangent=cached)
            ).reshape(-1)
        beam_model = beam.instantiate(beam_topology, _fields(beam_topology.points, p))
        u = beam_topology.reference * deformation
        theta = jnp.asarray([0.03 + p[9], 0.01 + p[10], 0.02 + p[11]])
        rotations = beam.exp_so3(jnp.arange(2, dtype=dtype)[:, None] * theta[None])
        f, torque = beam.elastic_force_torque(
            beam_model, u, rotations, command, replace(config, beam_cache_tangent=cached)
        )
        return jnp.concatenate((f.reshape(-1), torque.reshape(-1)))

    expected, expected_jvp = jax.jit(
        lambda p, t: jax.jvp(lambda x: response(x, False), (p,), (t,))
    )(parameters, tangent)
    actual, actual_jvp = jax.jit(lambda p, t: jax.jvp(lambda x: response(x, True), (p,), (t,)))(
        parameters, tangent
    )
    cotangent = jnp.asarray(np.random.default_rng(31).normal(size=actual.shape), dtype)
    expected_vjp = jax.jit(jax.grad(lambda p: jnp.vdot(response(p, False), cotangent)))(parameters)
    actual_vjp = jax.jit(jax.grad(lambda p: jnp.vdot(response(p, True), cotangent)))(parameters)
    tolerance = 3e-5 if dtype.name == "float32" else 2e-9
    absolute = 2e-7 if dtype.name == "float32" else 2e-11
    for a, b in ((actual, expected), (actual_jvp, expected_jvp), (actual_vjp, expected_vjp)):
        assert np.isfinite(a).all() and np.isfinite(b).all()
        np.testing.assert_allclose(a, b, rtol=tolerance, atol=absolute)
    np.testing.assert_allclose(
        jnp.vdot(actual_jvp, cotangent),
        jnp.vdot(tangent, actual_vjp),
        rtol=tolerance,
        atol=absolute,
    )


@pytest.mark.parametrize("family", ["spring", "fem", "beam"])
def test_solid_cached_implicit_state_and_gradient(
    family: str, dtype: np.dtype[np.floating]
) -> None:
    """Material and control gradients through converged implicit steps remain unchanged."""
    parameters = jnp.zeros(12, dtype)
    config = DynamicsConfig(gravity=0, newton=NewtonConfig(gmres_method="fused"))
    spring_config = SpringConfig(
        grid_shape=(1, 1), ground=False, gravity=0, linear_solver="gmres", gmres_method="fused"
    )
    spring_topology = spring.topology(spring_config, dtype)
    mesh = fem.structured_mesh(shape=(1, 1, 1), dtype=dtype)
    beam_topology = beam.topology((2, 1, 1), (0.1, 0.05, 0.05), (0.0, 0.01, 0.0), dtype)

    def objective(p: jax.Array, cached: bool) -> tuple[jax.Array, jax.Array]:
        """Return velocity/work loss and flattened physical state, exposing solve failures."""
        command = p[6:8] + jnp.asarray([0.1, -0.05], dtype)
        if family == "spring":
            model = spring.instantiate(
                _fields(spring_topology.points, p), spring_topology, spring_config, 2
            )
            initial = spring.initial_state(model)
            initial = initial._replace(x=initial.x * 1.001)
            out, _ = spring.substep(
                model, initial, command, replace(spring_config, spring_cache_tangent=cached)
            )
            physical = jnp.concatenate((out.x.reshape(-1), out.v.reshape(-1)))
        elif family == "fem":
            fem_model = fem.instantiate(mesh, _fields(mesh.points, p))
            fem_initial = fem.initial_state(fem_model)
            fem_initial = fem_initial._replace(u=mesh.reference * 0.001)
            out, _ = fem.trial(
                fem_model, fem_initial, command, 0.00025, replace(config, fem_cache_tangent=cached)
            )
            physical = jnp.concatenate((out.u.reshape(-1), out.v.reshape(-1)))
        else:
            beam_model = beam.instantiate(beam_topology, _fields(beam_topology.points, p))
            beam_initial = beam.initial_state(beam_model)
            beam_initial = beam_initial._replace(u=beam_topology.reference * 0.001)
            out, _ = beam.trial(
                beam_model,
                beam_initial,
                command,
                0.00025,
                replace(config, beam_cache_tangent=cached),
            )
            physical = jnp.concatenate(
                (
                    out.u.reshape(-1),
                    out.v.reshape(-1),
                    out.rotation.reshape(-1),
                    out.omega.reshape(-1),
                )
            )
        value = jnp.sum(out.v**2) + out.work_positive
        return jnp.where(out.status == 0, value, jnp.nan), physical

    expected, expected_grad = jax.jit(
        jax.value_and_grad(lambda p: objective(p, False), has_aux=True)
    )(parameters)
    actual, actual_grad = jax.jit(jax.value_and_grad(lambda p: objective(p, True), has_aux=True))(
        parameters
    )
    tolerance = 3e-5 if dtype.name == "float32" else 2e-9
    absolute = 2e-7 if dtype.name == "float32" else 2e-12
    for a, b in zip(
        jax.tree.leaves((actual, actual_grad)), jax.tree.leaves((expected, expected_grad))
    ):
        assert np.isfinite(a).all() and np.isfinite(b).all()
        np.testing.assert_allclose(a, b, rtol=tolerance, atol=absolute)


def test_cached_beam_without_links_is_finite(dtype: np.dtype[np.floating]) -> None:
    """A one-voxel beam has no element cache and retains zero elastic force/torque."""
    topology = beam.topology((1, 1, 1), (0.05, 0.05, 0.05), (0.0, 0.0, 0.0), dtype)
    model = beam.instantiate(topology, _fields(topology.points, jnp.zeros(12, dtype)))
    state = beam.initial_state(model)
    config = DynamicsConfig(beam_cache_tangent=True)
    value, derivative = jax.jit(
        jax.value_and_grad(
            lambda u: sum(
                jnp.sum(load**2)
                for load in beam.elastic_force_torque(
                    model, u, state.rotation, state.actuator, config
                )
            )
        )
    )(state.u)
    np.testing.assert_array_equal(value, 0.0)
    np.testing.assert_array_equal(derivative, jnp.zeros_like(derivative))
