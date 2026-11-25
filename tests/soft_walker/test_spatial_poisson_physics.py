"""Spatial Poisson physics, legacy overrides, and portable resolved models.

These are local constitutive/assembly checks; no trajectory or GPU execution.
The central-force spring model deliberately has no independent Poisson knob.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType

    from numpy.typing import ArrayLike

    from envs.soft_walker.engine.types import BodyDomain


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


import jax
import jax.numpy as jnp
import numpy as np
import pytest

from envs.soft_walker.engine.embodiments.decoders import Fields
from envs.soft_walker.engine.numerics.solvers import NewtonConfig
from envs.soft_walker.engine.physics import beam, fem, mpm, spring
from envs.soft_walker.engine.physics.common import DynamicsConfig
from envs.soft_walker.engine.physics.spring_config import SpringConfig


def material_fields(
    dim: int, dtype: np.dtype[np.floating] | type[np.floating], poisson: ArrayLike | None = None
) -> Fields:
    direction = jnp.broadcast_to(jnp.eye(dim, dtype=dtype)[1], (2, dim))
    # The original seven-argument constructor remains useful for legacy callers.
    fields = Fields(
        jnp.asarray([1.0, 0.8], dtype),
        jnp.asarray([12000.0, 28000.0], dtype),
        jnp.asarray([900.0, 1100.0], dtype),
        jnp.asarray([0.7, 0.9], dtype),
        direction,
        jnp.asarray([1, 1], jnp.int32),
        jnp.asarray([0, 1], jnp.int32),
    )
    return fields if poisson is None else fields._replace(poisson=jnp.asarray(poisson, dtype))


def fixture(
    kind: str, dtype: np.dtype[np.floating] | type[np.floating]
) -> tuple[ModuleType, BodyDomain, Fields]:
    if kind.startswith("mpm"):
        dim = int(kind[-1])
        topology = mpm.uniform_cloud(
            (2,) + (1,) * (dim - 1),
            spacing=0.05,
            origin=(0.02,) * dim,
            particles_per_axis=1,
            dtype=dtype,
        )
        return mpm, topology, material_fields(dim, dtype)
    if kind == "fem":
        return (
            fem,
            fem.structured_mesh((2, 1, 1), (0.1, 0.05, 0.05), (0.0, 0.02, 0.0), dtype),
            material_fields(3, dtype),
        )
    return (
        beam,
        beam.topology((2, 1, 1), (0.1, 0.05, 0.05), (0.0, 0.02, 0.0), dtype),
        material_fields(3, dtype),
    )


@pytest.mark.parametrize("kind", ["mpm2", "mpm3", "fem", "beam"])
def test_spatial_poisson_and_explicit_legacy_override(
    kind: str, dtype: np.dtype[np.floating]
) -> None:
    module, topology, legacy = fixture(kind, dtype)
    fields = legacy._replace(poisson=jnp.asarray([0.26, 0.44], dtype))
    np.testing.assert_array_equal(module.instantiate(topology, fields).poisson, fields.poisson)
    np.testing.assert_array_equal(
        module.instantiate(topology, legacy).poisson, jnp.full((2,), 0.35, dtype)
    )
    np.testing.assert_array_equal(
        module.instantiate(topology, fields, poisson=0.31).poisson, jnp.full((2,), 0.31, dtype)
    )
    override = jnp.asarray([0.29, 0.38], dtype)
    np.testing.assert_array_equal(
        module.instantiate(topology, fields, poisson=override).poisson, override
    )


def reference_density(F: np.ndarray, young: float, poisson: float) -> np.floating:
    """Independent ordinary NumPy neo-Hookean reference (2D is plane strain)."""
    mu = young / (2 * (1 + poisson))
    lam = young * poisson / ((1 + poisson) * (1 - 2 * poisson))
    logj = np.log(np.linalg.det(F))
    return 0.5 * mu * (np.sum(F * F) - F.shape[0] - 2 * logj) + 0.5 * lam * logj**2


@pytest.mark.parametrize("kind", ["mpm2", "mpm3", "fem"])
def test_local_poisson_changes_continuum_energy_and_has_material_gradient(
    kind: str, dtype: np.dtype[np.floating]
) -> None:
    module, topology, fields = fixture(kind, dtype)
    dim = 2 if kind == "mpm2" else 3
    delta = jnp.diag(jnp.asarray([0.065, 0.025, -0.01][:dim], dtype))
    F = jnp.eye(dim, dtype=dtype) + delta
    nu = jnp.asarray([0.26, 0.44], dtype)
    cfg = DynamicsConfig(gravity=0.0)

    def energy(poisson: jax.Array) -> jax.Array:
        model = module.instantiate(topology, fields._replace(poisson=poisson))
        if kind == "fem":
            return fem.elastic_energy(model, topology.reference @ delta.T, model.channels, cfg)
        return mpm.elastic_energy(model, jnp.broadcast_to(F, (2, dim, dim)), model.channels, cfg)

    volume = np.sum(topology.weights, axis=-1) if kind == "fem" else np.asarray(topology.volume)
    weights = np.asarray(fields.occupancy, np.float64) * volume
    E = np.asarray(fields.young, np.float64)
    F64 = np.asarray(F, np.float64)
    nu64 = np.asarray(nu, np.float64)
    reference = np.sum(weights * reference_density(F64, E, nu64))
    expected_gradient = (
        weights
        * (reference_density(F64, E, nu64 + 1e-6) - reference_density(F64, E, nu64 - 1e-6))
        / 2e-6
    )
    value, gradient = jax.value_and_grad(energy)(nu)
    tolerance = 2e-5 if dtype == np.dtype("float32") else 2e-9
    np.testing.assert_allclose(value, reference, rtol=tolerance, atol=1e-12)
    np.testing.assert_allclose(gradient, expected_gradient, rtol=tolerance, atol=1e-12)
    assert np.all(np.isfinite(gradient)) and np.all(np.abs(gradient) > 0)
    assert not np.isclose(float(energy(jnp.full_like(nu, 0.35))), float(value), rtol=1e-3)


@pytest.mark.parametrize("dim", [2, 3])
def test_mpm_active_stress_uses_local_shear_modulus(dim: int, dtype: np.dtype[np.floating]) -> None:
    _, cloud, fields = fixture(f"mpm{dim}", dtype)
    nu = jnp.asarray([0.26, 0.44], dtype)
    cfg = DynamicsConfig(actuation_model="active_stress", active_stress_ratio=0.4)
    commands = jnp.asarray([0.6, -0.3], dtype)
    model = mpm.instantiate(cloud, fields._replace(poisson=nu))
    expected = (
        0.4
        * np.asarray(fields.young)
        / (2 * (1 + np.asarray(nu)))
        * np.asarray(fields.strength)
        * np.asarray(commands)
    )
    np.testing.assert_allclose(mpm.stress_amounts(model, commands, cfg), expected, rtol=2e-6)
    jacobian = jax.jacrev(
        lambda p: mpm.stress_amounts(
            mpm.instantiate(cloud, fields._replace(poisson=p)), commands, cfg
        )
    )(nu)
    np.testing.assert_allclose(jacobian, np.diag(-expected / (1 + np.asarray(nu))), rtol=2e-6)


def test_beam_spatial_poisson_uses_series_shear_compliance(dtype: np.dtype[np.floating]) -> None:
    _, topology, fields = fixture("beam", dtype)
    nu = jnp.asarray([0.26, 0.44], dtype)
    model = beam.instantiate(topology, fields._replace(poisson=nu))
    E = np.asarray(fields.young, np.float64)
    Eeff = 2 / np.sum(1 / E)
    Geff = 1 / np.sum((1 + np.asarray(nu, np.float64)) / E)
    h = float(topology.h)
    second = h**4 / 12
    scale = float(jnp.min(fields.occupancy))
    expected_extension = scale * np.diag(
        [Eeff * h**2, (5 / 6) * Geff * h**2, (5 / 6) * Geff * h**2]
    )
    expected_bend = scale * np.diag([Geff * 2 * second, Eeff * second, Eeff * second])
    np.testing.assert_allclose(model.extension[0], expected_extension, rtol=2e-6, atol=0)
    np.testing.assert_allclose(model.bend[0], expected_bend, rtol=2e-6, atol=0)
    uniform = beam.instantiate(topology, fields, poisson=0.35)
    np.testing.assert_allclose(
        uniform.extension[0, 1, 1], scale * (5 / 6) * Eeff / 2.7 * h**2, rtol=2e-6
    )
    # Axial and flexural stiffness stay fixed; nu changes shear and torsion.
    np.testing.assert_array_equal(model.extension[:, 0, 0], uniform.extension[:, 0, 0])
    np.testing.assert_array_equal(model.bend[:, 1, 1], uniform.bend[:, 1, 1])
    assert not np.isclose(
        float(model.extension[0, 1, 1]), float(uniform.extension[0, 1, 1]), rtol=1e-3
    )


def test_beam_local_poisson_gradient_reaches_shear_and_torsion_energy(
    dtype: np.dtype[np.floating],
) -> None:
    _, topology, fields = fixture("beam", dtype)
    nu = jnp.asarray([0.26, 0.44], dtype)
    cfg = DynamicsConfig(gravity=0.0)
    displacement = jnp.asarray([[0.0, 0.0, 0.0], [0.0, 0.004, 0.0]], dtype)
    rotation = jnp.broadcast_to(jnp.eye(3, dtype=dtype), (2, 3, 3))
    h = float(topology.h)
    E = np.asarray(fields.young, np.float64)
    G = 1 / np.sum((1 + np.asarray(nu, np.float64)) / E)
    # A pure transverse displacement with fixed rotations isolates link shear.
    coefficient = 0.5 * h * float(jnp.min(fields.occupancy)) * (5 / 6) * h**2 * (0.004 / h) ** 2

    def energy(p: jax.Array) -> jax.Array:
        model = beam.instantiate(topology, fields._replace(poisson=p))
        return beam.elastic_energy(model, displacement, rotation, model.channels, cfg)

    value, gradient = jax.value_and_grad(energy)(nu)
    np.testing.assert_allclose(value, coefficient * G, rtol=2e-6)
    np.testing.assert_allclose(gradient, -coefficient * G**2 / E, rtol=2e-6)
    # The independently assembled torsional coefficient has the same local G.
    derivative = jax.jacrev(
        lambda p: beam.instantiate(topology, fields._replace(poisson=p)).bend[0, 0, 0]
    )(nu)
    np.testing.assert_allclose(
        derivative, -float(jnp.min(fields.occupancy)) * (h**4 / 6) * G**2 / E, rtol=2e-6
    )


def test_central_force_springs_have_no_independent_poisson_knob(
    dtype: np.dtype[np.floating],
) -> None:
    cfg = SpringConfig(grid_shape=(2, 1), size=(0.1, 0.05))
    topology = spring.topology(cfg, dtype)
    fields = material_fields(2, dtype)
    models = [
        spring.instantiate(fields._replace(poisson=jnp.asarray(nu, dtype)), topology, cfg, 2)
        for nu in ([0.26, 0.44], [0.44, 0.26])
    ]
    for left, right in zip(models[0], models[1]):
        np.testing.assert_array_equal(left, right)


@pytest.mark.parametrize("kind", ["mpm2", "mpm3", "fem", "beam"])
def test_spatial_poisson_native_export_and_serialization(
    kind: str, dtype: np.dtype[np.floating], tmp_path: Path
) -> None:
    pytest.importorskip("warp")
    module, topology, fields = fixture(kind, dtype)
    nu = jnp.asarray([0.26, 0.44], dtype)
    model = module.instantiate(topology, fields._replace(poisson=nu))
    cfg = DynamicsConfig(max_halvings=0, newton=NewtonConfig(preconditioner_side="right"))
    if kind.startswith("mpm"):
        from envs.soft_walker.engine.warp_native.mpm_conversion import from_jax, to_jax

        dim = int(kind[-1])
        bundle = from_jax(
            model,
            module.initial_state(model),
            mpm.Grid((8,) * dim, 0.05, (-0.05,) * dim),
            cfg,
            dt=0.001,
        )
    else:
        from envs.soft_walker.engine.warp_native.solid_conversion import from_jax

        bundle = from_jax(kind, model, module.initial_state(model), cfg, dt=0.001)
    path = tmp_path / f"{kind}.npz"
    bundle.save(path)
    restored = type(bundle).load(path)
    for key in bundle.fields:
        np.testing.assert_array_equal(restored.fields[key], bundle.fields[key])
    if kind == "beam":
        from envs.soft_walker.engine.warp_native.solid_builders import regular_solid

        properties = {
            key: np.asarray(value) for key, value in fields._replace(poisson=nu)._asdict().items()
        }
        properties["kind"] = properties.pop("actuator_type")
        native = regular_solid(
            "beam",
            shape=(2, 1, 1),
            size=(0.1, 0.05, 0.05),
            origin=(0.0, 0.02, 0.0),
            precision=str(dtype),
            properties=properties,
        )
        for key in ("extension", "bending", "young", "poisson"):
            np.testing.assert_allclose(
                restored.fields[key], native.fields[key], rtol=2e-6, atol=1e-12
            )
        # Effective edge nu is compliance-weighted by E, not the arithmetic mean.
        expected_nu = np.sum(np.asarray(nu) / np.asarray(fields.young)) / np.sum(
            1 / np.asarray(fields.young)
        )
        np.testing.assert_allclose(restored.fields["poisson"], expected_nu, rtol=2e-6)
    else:
        np.testing.assert_array_equal(restored.fields["poisson"], nu)
        if kind.startswith("mpm"):
            roundtrip_model, *_ = to_jax(restored)
            np.testing.assert_array_equal(roundtrip_model.poisson, nu)
        else:
            from envs.soft_walker.engine.warp_native.solid_builders import regular_solid

            native = regular_solid(
                "fem", precision=str(dtype), properties={"poisson": np.asarray(nu)}
            )
            np.testing.assert_array_equal(native.fields["poisson"], nu)
