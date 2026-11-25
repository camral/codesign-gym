"""Constitutive, discrete-work, and implicit-adjoint checks for MPM stress control."""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as j
import numpy as np
import pytest

from envs.soft_walker.engine.embodiments.decoders import Fields
from envs.soft_walker.engine.numerics.solvers import NewtonConfig
from envs.soft_walker.engine.physics import material, mpm
from envs.soft_walker.engine.physics.common import DynamicsConfig


def fixture(
    dtype: np.dtype[np.floating] | type[np.floating],
) -> tuple[mpm.Model, mpm.State, mpm.Grid, DynamicsConfig]:
    cloud = mpm.uniform_cloud((1, 1), origin=(0.025, 0.025), dtype=dtype)
    n = len(cloud.reference)
    one = j.ones(n, dtype)
    fields = Fields(
        one,
        one * 12000.0,
        one * 1000.0,
        one * 0.95,
        j.broadcast_to(j.array([0.0, 1.0], dtype), (n, 2)),
        j.ones(n, j.int32),
        j.zeros(n, j.int32),
    )
    model = mpm.instantiate(cloud, fields, channels=1)
    cfg = DynamicsConfig(
        gravity=0,
        actuation_model="active_stress",
        actuator_tau=0,
        max_halvings=0,
        mpm_mass_weighted=True,
        newton=NewtonConfig(preconditioner_side="right", linear_refinements=1),
    )
    return model, mpm.initial_state(model), mpm.Grid(shape=(8, 7)), cfg


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("kind", [0, 1, 2])
@pytest.mark.parametrize("amplitude", [-1700.0, 1300.0])
def test_active_stress_constitutive_derivatives_and_objectivity(
    dim: int, kind: int, amplitude: float, dtype: np.dtype[np.floating]
) -> None:
    rng = np.random.default_rng(301 + dim)
    F = np.eye(dim) + 0.1 * rng.normal(size=(dim, dim))
    n = rng.normal(size=dim)
    n /= np.linalg.norm(n)
    Q = np.zeros((dim, dim)) if kind == 0 else (np.outer(n, n) if kind == 1 else np.eye(dim) / dim)
    S = amplitude * Q
    D = j.array(F - np.eye(dim), dtype)
    stress, direction, k = j.array(amplitude, dtype), j.array(n, dtype), j.array(kind)
    args = (stress, direction, k)
    energy = material.active_stress_potential_from_delta(D, *args)
    P = material.active_stress_from_delta(D, *args)
    tol = 4e-5 if dtype == np.dtype("float32") else 2e-11
    np.testing.assert_allclose(
        energy, 0.5 * np.sum(S * (F.T @ F - np.eye(dim))), rtol=tol, atol=tol
    )
    np.testing.assert_allclose(P, F @ S, rtol=tol, atol=tol)
    np.testing.assert_allclose(
        jax.grad(material.active_stress_potential_from_delta)(D, *args), P, rtol=tol, atol=tol
    )
    H = j.array(rng.normal(size=(dim, dim)), dtype)
    dP = material.active_stress_tangent_action(H, *args)
    np.testing.assert_allclose(dP, np.asarray(H) @ S, rtol=tol, atol=tol)
    np.testing.assert_allclose(
        jax.jvp(lambda d: material.active_stress_from_delta(d, *args), (D,), (H,))[1],
        dP,
        rtol=tol,
        atol=tol,
    )
    R = np.eye(dim)
    theta = 0.47
    R[:2, :2] = [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    rotated = j.array(R @ F - np.eye(dim), dtype)
    np.testing.assert_allclose(
        material.active_stress_potential_from_delta(rotated, *args),
        energy,
        rtol=tol,
        atol=tol * abs(amplitude),
    )
    np.testing.assert_allclose(
        material.active_stress_from_delta(rotated, *args), R @ np.asarray(P), rtol=tol, atol=tol
    )
    if kind == 0:
        np.testing.assert_array_equal(P, np.zeros_like(F))
        assert float(energy) == 0.0


def test_active_stress_work_is_mechanical_and_sign_sensitive(
    dtype: np.dtype[np.floating],
) -> None:
    direction = j.array([0.0, 1.0], dtype)
    kind = j.array(1)
    stress = j.array(1500.0, dtype)
    contraction = j.diag(j.array([0.0, -0.02], dtype))
    expansion = -contraction
    # Constant tensile active stress does positive work while contracting.
    work = material.active_stress_work_density(contraction, contraction, stress, direction, kind)
    np.testing.assert_allclose(work, 1500.0 * 0.98 * 0.02, rtol=2e-6)
    assert float(work) > 0
    assert (
        float(material.active_stress_work_density(expansion, expansion, stress, direction, kind))
        < 0
    )
    assert (
        float(material.active_stress_work_density(expansion, expansion, -stress, direction, kind))
        > 0
    )
    # Changing command with fixed geometry is not mechanical work, even though
    # the command-dependent constitutive potential can change.
    assert (
        float(
            material.active_stress_work_density(
                contraction, j.zeros_like(contraction), stress, direction, kind
            )
        )
        == 0.0
    )
    assert (
        float(material.active_stress_potential_from_delta(contraction, stress, direction, kind))
        != 0.0
    )
    assert (
        float(
            material.active_stress_work_density(
                contraction, contraction, j.array(0.0, dtype), direction, kind
            )
        )
        == 0.0
    )


@pytest.mark.parametrize(
    "settings",
    [
        {"actuation_model": "unknown"},
        {"actuator_tau": 0},
        {"actuation_model": "active_stress", "actuator_tau": -1},
        {"actuation_model": "active_stress", "actuator_tau": float("nan")},
        {"active_stress_ratio": -1},
        {"active_stress_ratio": 1},
        {"active_stress_ratio": float("inf")},
        {"active_stress_ratio": float("nan")},
    ],
)
def test_active_stress_config_rejects_invalid(settings: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        DynamicsConfig(**settings)


def test_active_stress_direct_and_filtered_command(dtype: np.dtype[np.floating]) -> None:
    _, _, _, cfg = fixture(dtype)
    previous = j.array([-0.3], dtype)
    command = j.array([0.6], dtype)
    np.testing.assert_array_equal(mpm.actuator_next(previous, command, 0.0005, cfg), command)
    filtered = mpm.actuator_next(previous, command, 0.0005, replace(cfg, actuator_tau=0.03))
    np.testing.assert_allclose(filtered, 0.6 + (-0.3 - 0.6) * np.exp(-0.0005 / 0.03), rtol=2e-6)
    assert DynamicsConfig().actuation_model == "active_strain"


def test_active_stress_grid_virtual_work_and_passive_energy(dtype: np.dtype[np.floating]) -> None:
    model, state, grid, cfg = fixture(dtype)
    state = state._replace(F=state.F + j.array([[0.06, 0.03], [-0.02, -0.04]], dtype))
    st = mpm.stencil(state.x, grid)
    gv = j.array(np.random.default_rng(913).normal(size=(grid.count, 2)) * 0.03, dtype)
    dt = j.array(0.0005, dtype)
    a = j.array([0.7], dtype)
    _, _, L = mpm.g2p(gv, st, grid)
    increment = dt * (L @ state.F)
    F = state.F + increment
    force = mpm.internal_grid_force(model, F, state.F, a, st, grid, cfg)

    def potential(v: jax.Array) -> jax.Array:
        _, _, grad = mpm.g2p(v, st, grid)
        return mpm.constitutive_potential(model, mpm.deformation_next(state.F, grad, dt), a, cfg)

    tol = 1e-4 if dtype == np.dtype("float32") else 2e-10
    np.testing.assert_allclose(jax.grad(potential)(gv), -dt * force, rtol=tol, atol=tol * 1e-5)
    passive = mpm.internal_grid_force(model, F, state.F, j.zeros_like(a), st, grid, cfg)
    work = (
        model.alpha
        * model.cloud.volume
        * material.active_stress_work_density(
            F - j.eye(2, dtype=dtype),
            increment,
            mpm.stress_amounts(model, a, cfg),
            model.direction,
            model.kind,
        )
    )
    np.testing.assert_allclose(
        j.sum(work), dt * j.sum((force - passive) * gv), rtol=tol, atol=tol * 1e-5
    )
    # Reporting elastic energy never includes command-dependent stress potential.
    np.testing.assert_array_equal(
        mpm.elastic_energy(model, F, a, cfg), mpm.elastic_energy(model, F, -a, cfg)
    )
    strain_cfg = replace(cfg, actuation_model="active_strain", actuator_tau=0.03)
    np.testing.assert_allclose(
        mpm.elastic_energy(model, F, a, cfg),
        mpm.elastic_energy(model, F, j.zeros_like(a), strain_cfg),
        rtol=tol,
    )
    np.testing.assert_allclose(
        passive,
        mpm.internal_grid_force(model, F, state.F, j.zeros_like(a), st, grid, strain_cfg),
        rtol=tol,
        atol=tol * 1e-5,
    )
    np.testing.assert_allclose(
        mpm.stress_amounts(model, a, cfg), 0.4 * (12000 / 2.7) * 0.95 * 0.7, rtol=2e-6
    )


def test_active_stress_implicit_rollout_gradient_and_work(dtype: np.dtype[np.floating]) -> None:
    model, initial, grid, cfg = fixture(dtype)
    dt = j.array(0.0005, dtype)

    def rollout(amplitude: jax.Array) -> tuple[jax.Array, tuple[mpm.State, jax.Array]]:
        def one(state: mpm.State, _: None) -> tuple[mpm.State, jax.Array]:
            next_state, info = mpm.trial(model, state, j.reshape(amplitude, (1,)), dt, grid, cfg)
            return next_state, info.accepted

        state, accepted = jax.lax.scan(one, initial, None, length=2)
        y = state.x[:, 1]
        center = j.sum(model.mass * y) / j.sum(model.mass)
        spread = j.sum(model.mass * (y - center) ** 2) / j.sum(model.mass)
        return spread, (state, accepted)

    value_and_grad = jax.jit(jax.value_and_grad(rollout, has_aux=True))
    evaluate = jax.jit(rollout)
    amplitude = j.array(0.6, dtype)
    (value, (state, accepted)), derivative = value_and_grad(amplitude)
    np.testing.assert_array_equal(accepted, np.ones(2))
    assert int(state.status) == 0
    assert float(derivative) < 0  # positive material tension contracts the body
    step = 2e-2 if dtype == np.dtype("float32") else 1e-3
    plus, aux_plus = evaluate(amplitude + step)
    minus, aux_minus = evaluate(amplitude - step)
    assert np.all(np.asarray(aux_plus[1]) == 1) and np.all(np.asarray(aux_minus[1]) == 1)
    finite_difference = (plus - minus) / (2 * step)
    np.testing.assert_allclose(
        derivative,
        finite_difference,
        rtol=1e-2 if dtype == np.dtype("float32") else 3e-5,
        atol=1e-12,
    )
    # Separate reductions of sum(work) and sum(max(work, 0)) can differ in the last ulp.
    assert float(state.work_signed) > 0 and float(state.work_positive) >= float(
        state.work_signed
    ) * (1 - 1e-12)
    first, info = mpm.trial(model, initial, j.array([0.6], dtype), dt, grid, cfg)
    assert info.accepted == 1
    # Exact accounting check for the accepted implicit state. Float64 subtraction
    # is accurate here; the implementation retains the direct tiny increment.
    if dtype == np.dtype("float64"):
        reference = -j.sum(
            model.alpha
            * model.cloud.volume
            * j.sum(
                material.active_stress_from_delta(
                    first.F - j.eye(2),
                    mpm.stress_amounts(model, first.actuator, cfg),
                    model.direction,
                    model.kind,
                )
                * (first.F - initial.F),
                axis=(-2, -1),
            )
        )
        np.testing.assert_allclose(first.work_signed, reference, rtol=1e-9, atol=1e-14)
    explicit = mpm.explicit_trial(model, initial, j.array([0.6], dtype), dt, grid, cfg)
    assert float(explicit.work_signed) > 0
    np.testing.assert_array_equal(explicit.actuator, j.array([0.6], dtype))
