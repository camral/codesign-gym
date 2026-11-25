"""Positive-work VJPs, including actual spring actuator and routing kernels.

Use WARP_TEST_DTYPE and WARP_TEST_DEVICE to select precision and native device.
The independent JAX work-law comparison always runs on CPU.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import jax

    from envs.soft_walker.engine.warp_native.solid_model import SolidBundle


import os
from functools import lru_cache
from types import SimpleNamespace

import numpy as np
import pytest

wp = pytest.importorskip("warp")

from envs.soft_walker.engine.warp_native.solid_builders import regular_solid
from envs.soft_walker.engine.warp_native.solid_model import SolidConfig, SolidDeviceModel
from envs.soft_walker.engine.warp_native.work import positive_work_function

PRECISION = os.environ.get("WARP_TEST_DTYPE", "float64")
DEVICE = os.environ.get("WARP_TEST_DEVICE", "cpu")
RTOL = 3e-5 if PRECISION == "float32" else 2e-11
ATOL = 2e-8 if PRECISION == "float32" else 2e-13


@lru_cache(maxsize=2)
def scalar_probe(precision: "str") -> "wp.Kernel":
    real = wp.float32 if precision == "float32" else wp.float64
    positive_work = positive_work_function(precision)

    @wp.kernel(module=f"envs.soft_walker.engine.tests.positive_work_{precision}")
    def probe(
        values: wp.array(dtype=real), actual: wp.array(dtype=real), original: wp.array(dtype=real)
    ):
        i = wp.tid()
        actual[i] = positive_work(values[i])
        original[i] = wp.max(values[i], real(0))

    return probe


def test_positive_work_preserves_values_and_matches_jax_vjp() -> None:
    import jax
    import jax.numpy as jnp

    real = wp.float32 if PRECISION == "float32" else wp.float64
    values = np.asarray([-0.25, -1e-6, -0.0, 0.0, 1e-6, 0.25], dtype=PRECISION)
    seeds = np.asarray([0.7, -0.4, 1.0, 2.0, -0.3, 0.8], dtype=PRECISION)
    x = wp.array(values, dtype=real, device=DEVICE, requires_grad=True)
    actual = wp.zeros(len(values), dtype=real, device=DEVICE, requires_grad=True)
    original = wp.zeros_like(actual)
    seed = wp.array(seeds, dtype=real, device=DEVICE)
    with wp.Tape() as tape:
        wp.launch(
            scalar_probe(PRECISION), dim=len(values), inputs=[x, actual, original], device=DEVICE
        )
    tape.backward(grads={actual: seed})
    wp.synchronize_device(DEVICE)
    # Include signed zeros: the custom VJP must not change any forward bits.
    integer = np.uint32 if PRECISION == "float32" else np.uint64
    np.testing.assert_array_equal(actual.numpy().view(integer), original.numpy().view(integer))
    expected = seeds * np.asarray([0.0, 0.0, 0.5, 0.5, 1.0, 1.0], dtype=PRECISION)
    np.testing.assert_array_equal(x.grad.numpy(), expected)
    with jax.enable_x64(PRECISION == "float64"), jax.default_device(jax.devices("cpu")[0]):
        oracle = jax.grad(lambda v: jnp.sum(jnp.maximum(v, 0.0) * jnp.asarray(seeds)))(
            jnp.asarray(values)
        )
    np.testing.assert_array_equal(x.grad.numpy(), np.asarray(oracle))


def spring_fixture() -> "tuple[SolidBundle, SolidDeviceModel]":
    config = SolidConfig(
        physics="spring",
        grid_shape=(4, 4),
        origin=(0.0, 0.0),
        dt=0.004,
        actuator_tau=0.03,
        gravity=0.0,
        contact_enabled=False,
    )
    bundle = regular_solid(
        "spring",
        shape=(1, 1),
        size=(0.05, 0.05),
        origin=(0.0, 0.0),
        channels=2,
        precision=PRECISION,
        config=config,
    )
    # Only the first interaction stores elastic energy. It is stretched at q=0,
    # making work derivatives nonzero even when the actuation change is zero.
    bundle.fields["stiffness"].fill(0.0)
    bundle.fields["stiffness"][0] = 7.0
    bundle.fields["damping"].fill(0.0)
    bundle.fields["length"][0] *= 0.8
    bundle.fields["routing"].fill(0.0)
    owner = SolidDeviceModel(bundle, DEVICE)
    return bundle, owner


@pytest.mark.parametrize("parameter", ["initial_actuator", "routing"])
@pytest.mark.parametrize("value", [-0.02, 0.0, 0.02], ids=["negative", "zero", "positive"])
def test_spring_work_parameter_vjp(parameter: "str", value: "float") -> None:
    import jax
    import jax.numpy as jnp

    from envs.soft_walker.engine.physics import spring

    bundle, owner = spring_fixture()
    k, model, state, cfg = owner.k, owner.model, owner.initial, owner.parameters
    q = wp.zeros(owner.n, dtype=k.qvec, device=DEVICE)
    next_a = wp.zeros(owner.channels, dtype=k.real, device=DEVICE, requires_grad=True)
    command_values = np.asarray(
        [0.0, 0.0] if parameter == "initial_actuator" else [0.35, -0.15], dtype=PRECISION
    )
    command = wp.array(command_values, dtype=k.real, device=DEVICE)
    code = wp.zeros(1, dtype=int, device=DEVICE)
    work = wp.zeros(3, dtype=k.real, device=DEVICE, requires_grad=True)
    seed = wp.array(np.asarray([0.0, 1.0, 0.0], PRECISION), dtype=k.real, device=DEVICE)
    routing = bundle.fields["routing"].copy()
    old = np.zeros(owner.channels, dtype=PRECISION)
    if parameter == "initial_actuator":
        routing[0] = [0.7, -0.2]
        old[0] = value
    else:
        routing[0, 0] = value
    model.routing.assign(routing)
    state.actuator.assign(old)

    def evaluate() -> "None":
        work.zero_()
        wp.launch(
            k.actuator, dim=owner.channels, inputs=[state, command, cfg, next_a], device=DEVICE
        )
        wp.launch(
            k.work_kernel, dim=1, inputs=[model, state, cfg, q, next_a, code, work], device=DEVICE
        )

    with wp.Tape() as tape:
        evaluate()
    tape.backward(grads={work: seed})
    wp.synchronize_device(DEVICE)
    result = work.numpy().copy()
    actual = (
        state.actuator.grad.numpy().copy()
        if parameter == "initial_actuator"
        else model.routing.grad.numpy()[0].copy()
    )

    # Independent derivative of the fixed-position energy change, including the
    # actual actuator filter. At the kink use the reference half subgradient.
    reference = bundle.fields["reference"].astype(np.float64)
    i, j = bundle.fields["entities"][0]
    ell = np.sqrt(np.sum((reference[j] - reference[i]) ** 2) + 1e-24)
    length = float(bundle.fields["length"][0])
    stiffness = float(bundle.fields["stiffness"][0])
    gamma = np.exp(-float(cfg.dt) / float(cfg.tau))
    r, a = routing[0].astype(float), old.astype(float)
    new = command_values.astype(float) + (a - command_values.astype(float)) * gamma
    preferred = length * np.exp(r @ a)
    delta_log = r @ (new - a)
    delta = preferred * np.expm1(delta_log)
    signed = stiffness * delta * (0.5 * delta - (ell - preferred))
    old_derivative = preferred * (r if parameter == "initial_actuator" else a)
    log_derivative = r * (gamma - 1.0) if parameter == "initial_actuator" else new - a
    delta_derivative = (
        old_derivative * np.expm1(delta_log) + preferred * np.exp(delta_log) * log_derivative
    )
    signed_derivative = stiffness * (
        (delta - ell + preferred) * delta_derivative + delta * old_derivative
    )
    expected = (0.5 if signed == 0.0 else float(signed > 0.0)) * signed_derivative
    np.testing.assert_allclose(result[:2], [signed, max(signed, 0.0)], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=ATOL)
    if value == 0.0:
        assert result[0] == result[1] == 0.0
        assert np.linalg.norm(expected) > 1e-5, "The zero-work fixture needs a nonzero derivative"

    with jax.enable_x64(PRECISION == "float64"), jax.default_device(jax.devices("cpu")[0]):
        fixed = dict(
            reference=jnp.asarray(bundle.fields["reference"]),
            active=jnp.asarray(bundle.fields["mass"] > 0),
            edges=jnp.asarray(bundle.fields["entities"][:1]),
            length=jnp.asarray(bundle.fields["length"][:1]),
            stiffness=jnp.asarray(bundle.fields["stiffness"][:1]),
        )

        def objective(variable: "jax.Array") -> "jax.Array":
            initial = variable if parameter == "initial_actuator" else jnp.asarray(old)
            route = (
                jnp.asarray(routing[:1]) if parameter == "initial_actuator" else variable[None, :]
            )
            after = jnp.asarray(command_values) + (initial - jnp.asarray(command_values)) * jnp.exp(
                -jnp.asarray(cfg.dt) / jnp.asarray(cfg.tau)
            )
            oracle_model = SimpleNamespace(**fixed, logstrain_map=route)
            return jnp.sum(
                jnp.maximum(
                    spring.actuation_work(oracle_model, fixed["reference"], initial, after), 0.0
                )
            )

        oracle_input = old if parameter == "initial_actuator" else routing[0]
        oracle_value, oracle_gradient = jax.value_and_grad(objective)(jnp.asarray(oracle_input))
    np.testing.assert_allclose(result[1], np.asarray(oracle_value), rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(actual, np.asarray(oracle_gradient), rtol=RTOL, atol=ATOL)

    # A symmetric difference converges to the same half derivative at zero.
    # No second-order claim is made at this nonsmooth point.
    epsilon = 1e-4 if PRECISION == "float32" else 1e-6
    samples = []
    for offset in (-epsilon, epsilon):
        if parameter == "initial_actuator":
            perturbed = old.copy()
            perturbed[0] += offset
            state.actuator.assign(perturbed)
        else:
            perturbed = routing.copy()
            perturbed[0, 0] += offset
            model.routing.assign(perturbed)
        evaluate()
        wp.synchronize_device(DEVICE)
        samples.append(float(work.numpy()[1]))
    finite_difference = (samples[1] - samples[0]) / (2 * epsilon)
    np.testing.assert_allclose(
        actual[0], finite_difference, rtol=3e-3 if PRECISION == "float32" else 3e-5, atol=ATOL
    )
