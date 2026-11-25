"""Contact derivatives at exact touching and execution-result validation.

Select precision/device with WARP_TEST_DTYPE and WARP_TEST_DEVICE. These probes
exercise local native kernels, without compiling an episode or running JAX.
"""

import os
from dataclasses import replace
from functools import lru_cache

import numpy as np
import pytest

wp = pytest.importorskip("warp")

from envs.soft_walker.engine.warp_native.kernels import kernels
from envs.soft_walker.engine.warp_native.local import functions
from envs.soft_walker.engine.warp_native.model import DeviceModel, uniform_body

DTYPE = os.environ.get("WARP_TEST_DTYPE", "float64")
DEVICE = os.environ.get("WARP_TEST_DEVICE", "cpu")


@lru_cache(maxsize=4)
def probes(dim: "int", precision: "str") -> "tuple[wp.Kernel, wp.Kernel, wp.Kernel]":
    local = functions(dim, precision)
    k = kernels(dim, precision)
    real, vec, Model, Parameters = k.real, k.vec, k.Model, k.Parameters
    softplus, contact, contact_tangent = local.softplus, k.contact, k.contact_tangent
    module = f"envs.soft_walker.engine.tests.contact_d{dim}_{precision}"

    @wp.kernel(module=module)
    def scalar_probe(x: wp.array(dtype=real), result: wp.array(dtype=real)):
        i = wp.tid()
        result[i] = softplus(x[i])

    @wp.kernel(module=module)
    def force_probe(
        x: wp.array(dtype=vec),
        velocity: wp.array(dtype=vec),
        area: real,
        model: Model,
        cfg: Parameters,
        result: wp.array(dtype=vec),
    ):
        i = wp.tid()
        force, dissipation = contact(x[i], velocity[i], area, model, cfg)
        result[i] = force

    @wp.kernel(module=module, enable_backward=False)
    def tangent_probe(
        x: wp.array(dtype=vec),
        velocity: wp.array(dtype=vec),
        dx: wp.array(dtype=vec),
        dv: wp.array(dtype=vec),
        area: real,
        model: Model,
        cfg: Parameters,
        result: wp.array(dtype=vec),
    ):
        i = wp.tid()
        result[i] = contact_tangent(x[i], velocity[i], dx[i], dv[i], area, model, cfg)

    return scalar_probe, force_probe, tangent_probe


def test_softplus_slope_at_zero_and_nearby() -> None:
    real = functions(2, DTYPE).scalar
    values = np.asarray([-1.0e-6, 0.0, 1.0e-6], dtype=DTYPE)
    x = wp.array(values, dtype=real, device=DEVICE, requires_grad=True)
    output = wp.zeros(3, dtype=real, device=DEVICE, requires_grad=True)
    seed = wp.ones(3, dtype=real, device=DEVICE)
    with wp.Tape() as tape:
        wp.launch(probes(2, DTYPE)[0], dim=3, inputs=[x, output], device=DEVICE)
    tape.backward(grads={output: seed})
    wp.synchronize_device(DEVICE)
    tolerance = 3.0e-6 if DTYPE == "float32" else 2.0e-13
    np.testing.assert_allclose(
        output.numpy(), np.logaddexp(0.0, values), rtol=tolerance, atol=tolerance
    )
    np.testing.assert_allclose(
        x.grad.numpy(), 1.0 / (1.0 + np.exp(-values)), rtol=tolerance, atol=tolerance
    )


@pytest.mark.parametrize("dim", [2, 3])
def test_contact_vjp_matches_tangent_and_finite_difference_at_zero_gap(dim: "int") -> None:
    bundle = uniform_body(dim, precision=DTYPE)
    bundle.config = replace(bundle.config, contact_enabled=True, contact_height=0.0)
    model = DeviceModel(bundle, DEVICE)
    k = model.k
    _, force_probe, tangent_probe = probes(dim, DTYPE)
    # Nonzero tangential and normal velocity exercise normal force and friction.
    position = np.zeros(dim, dtype=DTYPE)
    velocity = np.asarray([0.03, -0.02, 0.01][:dim], dtype=DTYPE)
    dx = np.asarray([0.2, 1.0, -0.3][:dim], dtype=DTYPE)
    dv = np.asarray([-0.015, 0.01, 0.007][:dim], dtype=DTYPE)
    cotangent = np.asarray([0.4, 1.0, -0.2][:dim], dtype=DTYPE)
    epsilon = 1.0e-6
    positions = np.stack((position, position + epsilon * dx, position - epsilon * dx))
    velocities = np.stack((velocity, velocity + epsilon * dv, velocity - epsilon * dv))
    x = wp.array(positions, dtype=k.vec, device=DEVICE, requires_grad=True)
    v = wp.array(velocities, dtype=k.vec, device=DEVICE, requires_grad=True)
    force = wp.zeros(3, dtype=k.vec, device=DEVICE, requires_grad=True)
    seeds = np.zeros((3, dim), dtype=DTYPE)
    seeds[0] = cotangent
    seed = wp.array(seeds, dtype=k.vec, device=DEVICE)
    area = k.real(0.001)
    with wp.Tape() as tape:
        wp.launch(
            force_probe,
            dim=3,
            inputs=[x, v, area, model.model, model.parameters, force],
            device=DEVICE,
        )
    tape.backward(grads={force: seed})
    direction_x = wp.array(dx[None], dtype=k.vec, device=DEVICE)
    direction_v = wp.array(dv[None], dtype=k.vec, device=DEVICE)
    tangent = wp.zeros(1, dtype=k.vec, device=DEVICE)
    wp.launch(
        tangent_probe,
        dim=1,
        inputs=[x, v, direction_x, direction_v, area, model.model, model.parameters, tangent],
        device=DEVICE,
    )
    wp.synchronize_device(DEVICE)
    actual = np.dot(x.grad.numpy()[0], dx) + np.dot(v.grad.numpy()[0], dv)
    analytic = np.dot(cotangent, tangent.numpy()[0])
    forces = force.numpy()
    finite_difference = np.dot(cotangent, forces[1] - forces[2]) / (2 * epsilon)
    assert abs(analytic) > 1.0, "The derivative fixture must have a material signal"
    np.testing.assert_allclose(
        actual,
        analytic,
        rtol=3.0e-5 if DTYPE == "float32" else 2.0e-11,
        atol=3.0e-5 if DTYPE == "float32" else 2.0e-11,
    )
    np.testing.assert_allclose(
        actual,
        finite_difference,
        rtol=2.0e-3 if DTYPE == "float32" else 1.0e-5,
        atol=2.0e-3 if DTYPE == "float32" else 1.0e-5,
    )


@pytest.mark.parametrize("gradient", [False, True])
def test_validate_execution_accepts_valid_result(gradient: "bool") -> None:
    from envs.soft_walker.diagnostics.warp.checks import _validate_execution

    result = {"status": 0, "loss": 1.25}
    if gradient:
        result["adjoint_status"] = 0
    _validate_execution(result, gradient=gradient, phase="unit validation")


@pytest.mark.parametrize(
    "gradient, changes",
    [
        (False, {"status": 1}),
        (True, {"status": 1}),
        (False, {"loss": np.nan}),
        (True, {"loss": np.inf}),
        (False, {"loss": -np.inf}),
        (True, {"adjoint_status": 20}),
        (True, {"adjoint_status": 21}),
        (True, {"adjoint_status": None}),
    ],
)
def test_validate_execution_rejects_invalid_result(
    gradient: "bool", changes: "dict[str, object]"
) -> None:
    from envs.soft_walker.diagnostics.warp.checks import _validate_execution

    result = {"status": 0, "loss": 1.25, "adjoint_status": 0}
    result.update(changes)
    if result["adjoint_status"] is None:
        del result["adjoint_status"]
    with pytest.raises(AssertionError):
        _validate_execution(result, gradient=gradient, phase="unit validation")
