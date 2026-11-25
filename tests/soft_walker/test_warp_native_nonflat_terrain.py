"""Shared native terrain/contact: values, VJP and analytic Newton tangent.

These local kernels are shared by MPM, spring, FEM and beam. No episode or
iterative solver is compiled here. Select CPU/CUDA with WARP_TEST_DEVICE.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from envs.soft_walker.engine.warp_native.model import Bundle


import os
from dataclasses import replace
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np
import pytest

wp = pytest.importorskip("warp")

from envs.soft_walker.engine.physics.contact import PlaneContact, response
from envs.soft_walker.engine.physics.terrain import Terrain, gap_normal
from envs.soft_walker.engine.warp_native.kernels import kernels
from envs.soft_walker.engine.warp_native.model import DeviceModel, uniform_body

PRECISION = os.environ.get("WARP_TEST_DTYPE", "float64")
DEVICE = os.environ.get("WARP_TEST_DEVICE", "cpu")


@lru_cache(maxsize=4)
def probes(dim: "int", precision: "str") -> "tuple[wp.Kernel, wp.Kernel]":
    k = kernels(dim, precision)
    real, vec, mat, Model, Parameters = k.real, k.vec, k.mat, k.Model, k.Parameters
    contact, tangent, geometry = k.contact, k.contact_tangent, k.terrain
    module = f"envs.soft_walker.engine.tests.nonflat_d{dim}_{precision}"

    @wp.kernel(module=module)
    def evaluate(
        x: wp.array(dtype=vec),
        v: wp.array(dtype=vec),
        model: Model,
        cfg: Parameters,
        area: real,
        force: wp.array(dtype=vec),
        diss: wp.array(dtype=real),
    ):
        i = wp.tid()
        f, d = contact(x[i], v[i], area, model, cfg)
        force[i] = f
        diss[i] = d

    @wp.kernel(module=module, enable_backward=False)
    def linearize(
        x: wp.array(dtype=vec),
        v: wp.array(dtype=vec),
        dx: wp.array(dtype=vec),
        dv: wp.array(dtype=vec),
        model: Model,
        cfg: Parameters,
        area: real,
        result: wp.array(dtype=vec),
        gaps: wp.array(dtype=real),
        normals: wp.array(dtype=vec),
        hessians: wp.array(dtype=mat),
    ):
        i = wp.tid()
        result[i] = tangent(x[i], v[i], dx[i], dv[i], area, model, cfg)
        g, n, h = geometry(x[i], model)
        gaps[i] = g
        normals[i] = n
        hessians[i] = h

    return evaluate, linearize


@lru_cache(maxsize=6)
def solid_probe(physics: "str", precision: "str") -> "wp.Kernel":
    from envs.soft_walker.engine.warp_native.solid_kernels import solid_kernels

    k = solid_kernels(physics, precision)
    Model, State, Parameters, qvec, real = k.Model, k.State, k.Parameters, k.qvec, k.real
    value = k.contact_value

    @wp.kernel(module=f"envs.soft_walker.engine.tests.nonflat_{physics}_{precision}")
    def evaluate(
        m: Model,
        s: State,
        p: Parameters,
        q: wp.array(dtype=qvec),
        forces: wp.array(dtype=qvec),
        diss: wp.array(dtype=real),
    ):
        i = wp.tid()
        f, e, d = value(i, q[i], m, s, p)
        forces[i] = f
        diss[i] = d

    return evaluate


def fixture(dim: "int", kind: "str") -> "tuple[Bundle, np.ndarray, Terrain]":
    normal = np.asarray([-0.22, 1.0, -0.11][:dim], PRECISION)
    normal /= np.linalg.norm(normal)
    center = np.asarray([0.015, 0.06, 0.025][:dim], PRECISION)
    half = np.asarray([0.02, 0.025, 0.03][:dim], PRECISION)
    if kind == "slope":
        centers = np.empty((0, dim), PRECISION)
        halves = np.empty_like(centers)
        rounding = np.empty(0, PRECISION)
        offset = np.asarray(0.018, PRECISION)
        point = np.asarray([0.023, 0.0, -0.01][:dim], PRECISION)
        point[1] = (offset - 0.0003 - np.dot(point, normal)) / normal[1]
    else:
        normal = np.eye(dim, dtype=PRECISION)[1]
        offset = np.asarray(-1.0 if kind == "tie" else 0.0, PRECISION)
        if kind == "tie":
            center = np.zeros(dim, PRECISION)
            half = np.full(dim, 0.02, PRECISION)
            point = np.asarray([0.005, 0.005, 0.0][:dim], PRECISION)
        else:
            q = np.asarray([0.0012, 0.0021, 0.0015][:dim], PRECISION)
            point = center + (half - 0.003 + q) * np.asarray([-1.0, 1.0, 1.0][:dim], PRECISION)
        centers, halves = center[None], half[None]
        rounding = np.asarray([0.003], PRECISION)
    data = dict(
        normal=normal,
        offset=offset,
        centers=centers,
        half_sizes=halves,
        rounding=rounding,
        active=np.ones(len(centers), np.int32),
    )
    bundle = uniform_body(dim, precision=PRECISION)
    bundle.config = replace(bundle.config, contact_enabled=True, contact_height=0.0)
    bundle.terrain = data
    return bundle, point, Terrain(*(jnp.asarray(data[key]) for key in Terrain._fields))


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("kind", ["slope", "corner", "tie"])
def test_nonflat_native_contact_matches_jax_vjp_and_tangent(dim: "int", kind: "str") -> None:
    bundle, point, terrain = fixture(dim, kind)
    model = DeviceModel(bundle, DEVICE)
    k, cfg = model.k, bundle.config
    evaluate, linearize = probes(dim, PRECISION)
    law = PlaneContact(
        enabled=True,
        stiffness=cfg.contact_stiffness,
        width=cfg.contact_width,
        friction=cfg.friction,
        velocity_width=cfg.friction_width,
    )
    v0 = np.asarray([0.03, -0.02, 0.01][:dim], PRECISION)
    dx0 = np.asarray([0.3, -0.2, 0.4][:dim], PRECISION)
    dv0 = np.asarray([0.1, -0.3, 0.2][:dim], PRECISION)
    seed0 = np.asarray([0.7, -0.4, 0.2][:dim], PRECISION)
    area = 0.001
    # At metre-scale coordinates a 1e-5 step loses several float32 bits during
    # coordinate subtraction. This larger step still resolves the .003 m arc.
    epsilon = 4.0e-5 if PRECISION == "float32" else 1.0e-7
    x = wp.array(
        np.stack([point, point + epsilon * dx0, point - epsilon * dx0]),
        dtype=k.vec,
        device=DEVICE,
        requires_grad=True,
    )
    v = wp.array(
        np.stack([v0, v0 + epsilon * dv0, v0 - epsilon * dv0]),
        dtype=k.vec,
        device=DEVICE,
        requires_grad=True,
    )
    force = wp.zeros(3, dtype=k.vec, device=DEVICE, requires_grad=True)
    diss = wp.zeros(3, dtype=k.real, device=DEVICE)
    seed = np.zeros((3, dim), PRECISION)
    seed[0] = seed0
    with wp.Tape() as tape:
        wp.launch(
            evaluate,
            3,
            inputs=[x, v, model.model, model.parameters, k.real(area), force, diss],
            device=DEVICE,
        )
    tape.backward(grads={force: wp.array(seed, dtype=k.vec, device=DEVICE)})
    dx = wp.array(dx0[None], dtype=k.vec, device=DEVICE)
    dv = wp.array(dv0[None], dtype=k.vec, device=DEVICE)
    tangent = wp.zeros(1, dtype=k.vec, device=DEVICE)
    gaps = wp.zeros(1, dtype=k.real, device=DEVICE)
    normals = wp.zeros(1, dtype=k.vec, device=DEVICE)
    hessians = wp.zeros(1, dtype=k.mat, device=DEVICE)
    wp.launch(
        linearize,
        1,
        inputs=[
            x,
            v,
            dx,
            dv,
            model.model,
            model.parameters,
            k.real(area),
            tangent,
            gaps,
            normals,
            hessians,
        ],
        device=DEVICE,
    )
    wp.synchronize_device(DEVICE)

    def jax_force(x: "jax.Array", v: "jax.Array") -> "jax.Array":
        return response(x, v, jnp.asarray(area, x.dtype), law, terrain)[0]

    jp, jv = jnp.asarray(point), jnp.asarray(v0)
    jf, jt = jax.jvp(jax_force, (jp, jv), (jnp.asarray(dx0), jnp.asarray(dv0)))
    jdx, jdv = jax.grad(lambda x, v: jnp.dot(jax_force(x, v), jnp.asarray(seed0)), argnums=(0, 1))(
        jp, jv
    )
    gap, normal = gap_normal(jp, terrain)
    hessian = jax.jacfwd(lambda x: gap_normal(x, terrain)[1])(jp)
    rtol, atol = (8.0e-5, 3.0e-5) if PRECISION == "float32" else (2.0e-10, 3.0e-10)
    for actual, expected in [
        (force.numpy()[0], jf),
        (tangent.numpy()[0], jt),
        (x.grad.numpy()[0], jdx),
        (v.grad.numpy()[0], jdv),
        (gaps.numpy()[0], gap),
        (normals.numpy()[0], normal),
        (hessians.numpy()[0], hessian),
    ]:
        np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)
    assert np.linalg.norm(force.numpy()[0]) > 1.0e-3
    assert diss.numpy()[0] >= 0.0
    np.testing.assert_allclose(
        np.dot(seed0, tangent.numpy()[0]),
        np.dot(x.grad.numpy()[0], dx0) + np.dot(v.grad.numpy()[0], dv0),
        rtol=rtol,
        atol=atol,
    )
    if kind != "tie":
        # Central differences are inappropriate on a nonsmooth medial-axis tie.
        fd = (force.numpy()[1] - force.numpy()[2]) / (2.0 * epsilon)
        np.testing.assert_allclose(
            tangent.numpy()[0],
            fd,
            rtol=3.0e-3 if PRECISION == "float32" else 2.0e-6,
            atol=3.0e-3 if PRECISION == "float32" else 2.0e-6,
        )


@pytest.mark.parametrize("physics", ["spring", "fem", "beam"])
@pytest.mark.parametrize("kind", ["slope", "box"])
def test_nonflat_native_solid_contact_routing_matches_jax(physics: "str", kind: "str") -> None:
    """Check the exporter/legacy switch and beam marker torque, not just geometry."""
    from envs.soft_walker.diagnostics.warp.solid_fixtures import fixture as solid_fixture
    from envs.soft_walker.engine.physics import beam, spring
    from envs.soft_walker.engine.physics.terrain import TerrainConfig, generate
    from envs.soft_walker.engine.warp_native.solid_conversion import from_jax
    from envs.soft_walker.engine.warp_native.solid_model import SolidDeviceModel

    original, m, s, cfg, _, _ = solid_fixture(physics, PRECISION, contact=True)
    dim = 2 if physics == "spring" else 3
    tc = (
        TerrainConfig(kind="slopes", blocks=0, fixed_slopes=(0.14, 0.06 if dim == 3 else 0.0))
        if kind == "slope"
        else TerrainConfig(
            kind="blocks",
            blocks=1,
            fixed_boxes=((0.0375, 0.006, 0.025, 0.03, 0.011, 0.055, 0.002),),
        )
    )
    m = m._replace(terrain=generate(tc, dim=dim, dtype=np.dtype(PRECISION)))
    cfg = replace(cfg, gravity=0.0)
    bundle = from_jax(physics, m, s, cfg, dt=original.config.dt)
    device = SolidDeviceModel(bundle, DEVICE)
    assert device.model.legacy_contact == 0
    k, dt = device.k, bundle.config.dt
    q = np.random.default_rng(17).normal(size=(device.n, k.Q)).astype(PRECISION) * 0.02
    qdev = wp.array(q, dtype=k.qvec, device=DEVICE)
    force = wp.zeros(device.n, dtype=k.qvec, device=DEVICE)
    diss = wp.zeros(device.n, dtype=k.real, device=DEVICE)
    wp.launch(
        solid_probe(physics, PRECISION),
        device.n,
        inputs=[device.model, device.initial, device.parameters, qdev, force, diss],
        device=DEVICE,
    )
    wp.synchronize_device(DEVICE)
    v = jnp.asarray(q[:, :dim] * cfg.velocity_ref)
    if physics == "spring":
        expected, _, expected_diss = spring.contact(m, s.x + dt * v, v, cfg)
    elif physics == "fem":
        expected, _, ds = response(
            m.mesh.reference + s.u + dt * v, v, m.contact_area, cfg.contact, m.terrain
        )
        expected_diss = jnp.sum(ds)
    else:
        omega = jnp.asarray(q[:, 3:] * cfg.velocity_ref / float(m.topology.h))
        R = beam.exp_so3(dt * omega) @ s.rotation
        f, torque, _, expected_diss, _ = beam.surface(m, s.u + dt * v, R, v, omega, cfg)
        expected = jnp.concatenate((f, torque), axis=-1)
    rtol, atol = (1.0e-4, 3.0e-5) if PRECISION == "float32" else (2.0e-10, 3.0e-10)
    np.testing.assert_allclose(force.numpy(), expected, rtol=rtol, atol=atol)
    np.testing.assert_allclose(np.sum(diss.numpy()), expected_diss, rtol=rtol, atol=atol)
    assert np.linalg.norm(force.numpy()) > 1.0e-3, "Fixture must actually touch the terrain"
