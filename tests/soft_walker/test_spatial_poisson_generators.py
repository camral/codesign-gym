"""Poisson fields must survive generation, API preparation, and design AD."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


import jax
import jax.numpy as j
import numpy as np
import pytest

from envs.soft_walker.engine import make, make_vec
from envs.soft_walker.engine.body_rollout import prepare
from envs.soft_walker.engine.differentiation import compiled_design_valid
from envs.soft_walker.engine.drivers.task import BodyConfig
from envs.soft_walker.engine.embodiments.decoders import check_embedding
from envs.soft_walker.engine.embodiments.morphology import (
    cell_center_points,
    generate_random_design,
)
from envs.soft_walker.engine.embodiments.serialization import load_design, save_design
from envs.soft_walker.presets.embodiments.chambers import ChamberDecoder
from envs.soft_walker.presets.locomotion import LeggedDecoder
from envs.soft_walker.presets.scenarios.build import decoder_and_embodiment
from envs.soft_walker.presets.scenarios.config import Scenario
from envs.soft_walker.shared import CPPNParameterization, make_implicit_design


@pytest.mark.parametrize("dim", (2, 3))
@pytest.mark.parametrize("representation", ("legged", "direct", "bspline", "cppn", "primitives"))
def test_custom_generators_local_poisson_and_legacy_upgrade(
    dim: int, representation: str, dtype: np.dtype[np.floating], tmp_path: Path
) -> None:
    decoder = (
        LeggedDecoder(dim, dtype)
        if representation == "legged"
        else ChamberDecoder(dim, representation, dtype)
    )
    embedding = decoder.base()
    shape = embedding["poisson_values"].shape
    embedding["poisson_values"] = np.broadcast_to(
        np.linspace(0.27, 0.43, 3, dtype=dtype).reshape((3,) + (1,) * dim), shape
    ).copy()
    points = j.asarray(
        [
            [-0.75, -0.4] + ([-0.66] if dim == 3 else []),
            [0.75, -0.4] + ([-0.66] if dim == 3 else []),
        ],
        dtype,
    )
    fields = decoder.decode(jax.tree.map(j.asarray, embedding), points)
    np.testing.assert_allclose(fields.poisson, [0.29, 0.41], rtol=2e-6)
    derivative = jax.grad(
        lambda values: j.sum(decoder.decode(dict(embedding, poisson_values=values), points).poisson)
    )(j.asarray(embedding["poisson_values"]))
    assert np.isfinite(derivative).all() and np.linalg.norm(derivative) > 0.1
    path = tmp_path / "design.json"
    save_design(path, embedding, decoder)
    loaded, _, _ = load_design(path, decoder=decoder)
    np.testing.assert_array_equal(loaded["poisson_values"], embedding["poisson_values"])
    old = {k: v for k, v in embedding.items() if k != "poisson_values"}
    assert check_embedding(decoder, old) == []
    np.testing.assert_allclose(decoder.decode(old, points).poisson, 0.35, rtol=2e-6)


@pytest.mark.parametrize("dim", (2, 3))
@pytest.mark.parametrize("representation", ("direct", "bspline", "cppn", "primitives"))
def test_random_creatures_have_real_spatial_poisson(
    dim: int, representation: str, dtype: np.dtype[np.floating]
) -> None:
    decoder, embedding, report = generate_random_design(
        representation, dim=dim, seed=7, dtype=dtype.name
    )
    points = j.asarray(cell_center_points(report["grid_shape"]), dtype)
    fields = decoder.decode(jax.tree.map(j.asarray, embedding), points)
    nu = np.asarray(fields.poisson)[np.asarray(fields.occupancy) > 0]
    assert np.isfinite(nu).all() and nu.min() >= 0.25 and nu.max() <= 0.45
    assert np.ptp(nu) > 1e-5


@pytest.mark.parametrize("representation", ("direct", "bspline", "cppn", "primitives"))
def test_scenario_poisson_seed_is_a_field(
    representation: str, dtype: np.dtype[np.floating]
) -> None:
    decoder, embedding = decoder_and_embodiment(
        Scenario(representation=representation, poisson=0.29, dtype=dtype.name)
    )
    assert "poisson" not in embedding
    points = j.asarray(cell_center_points((4, 4)), dtype)
    np.testing.assert_allclose(decoder.decode(embedding, points).poisson, 0.29, rtol=2e-6)


@pytest.mark.parametrize("nu", (0.25, 0.35, 0.45))
def test_implicit_optional_uniform_poisson_seed_and_finite_updates(
    nu: float, dtype: np.dtype[np.floating]
) -> None:
    decoder, embedding, metadata = make_implicit_design(2, dtype=dtype.name, poisson=nu)
    assert "poisson" not in embedding
    coordinates = CPPNParameterization(decoder, embedding)
    assert "poisson" in coordinates.groups
    points = j.asarray(cell_center_points(metadata["grid_shape"]), dtype)
    np.testing.assert_allclose(
        decoder.decode(jax.tree.map(j.asarray, embedding), points).poisson, nu, rtol=2e-6
    )
    for delta in (-1e6, 1e6):
        updated = coordinates.unpack(j.full(coordinates.initial.shape, delta, dtype))
        fields = decoder.decode(updated, points)
        assert np.isfinite(fields.poisson).all()
        assert np.all(np.asarray(fields.poisson) >= 0.25) and np.all(
            np.asarray(fields.poisson) <= 0.45
        )


@pytest.mark.parametrize(
    "eid", ("SoftMPM2D-v0", "SoftMPM3D-v0", "SoftHexFEM3D-v0", "SoftBeam3D-v0")
)
def test_body_api_and_compiled_preparation_use_spatial_poisson(
    eid: str, dtype: np.dtype[np.floating]
) -> None:
    dim = 2 if eid == "SoftMPM2D-v0" else 3
    cfg = BodyConfig(
        grid_shape=(2,) * dim, size=(0.1,) * dim, settle_steps=0, substeps=1, horizon=1
    )
    env, embedding = make(
        eid, dtype=dtype, device="cpu", config=cfg, max_observation_space_size=10000
    )
    try:
        assert "poisson" not in embedding and env.embodiment_space.contains(embedding)
        shape = embedding["values"].shape
        embedding["values"][..., -1] = np.broadcast_to(
            np.linspace(-1.0, 1.0, shape[0], dtype=dtype).reshape((shape[0],) + (1,) * (dim - 1)),
            shape[:-1],
        )
        _, model, errors = env._prepare(embedding)
        assert errors == [] and np.ptp(np.asarray(model.poisson)) > 0.01
        emb = jax.tree.map(j.asarray, embedding)
        actual, state = jax.jit(lambda e: prepare(env, e, env._terrain))(emb)
        assert int(state.status) == 0
        np.testing.assert_allclose(actual.poisson, model.poisson, rtol=2e-6)
        legacy = dict(embedding, poisson=np.asarray(0.31, dtype))
        _, overridden, errors = env._prepare(legacy)
        assert errors == []
        np.testing.assert_allclose(overridden.poisson, 0.31, rtol=2e-6)
        invalid = dict(legacy, poisson=np.asarray(0.9, dtype))
        assert not env.validate_embodiment(invalid)
        fields = env.decoder.decode(emb, env._domain.points)
        assert not bool(compiled_design_valid(env, jax.tree.map(j.asarray, invalid), fields))
    finally:
        env.close()


def test_body_vector_legacy_scalar_overrides_remain_per_lane(
    dtype: np.dtype[np.floating],
) -> None:
    cfg = BodyConfig(grid_shape=(2, 1), size=(0.1, 0.05), settle_steps=0, horizon=1, substeps=1)
    env, embedding = make_vec("SoftMPM2D-v0", num_envs=2, dtype=dtype, device="cpu", config=cfg)
    try:
        legacy = dict(embedding, poisson=np.asarray([0.29, 0.41], dtype))
        candidates = env._candidates(legacy)
        assert len(candidates) == 2
        for lane, candidate in enumerate(candidates):
            _, model, errors = env.envs[lane]._prepare(candidate)
            assert errors == []
            np.testing.assert_allclose(model.poisson, [0.29, 0.41][lane], rtol=2e-6)
    finally:
        env.close()


@pytest.mark.parametrize("family", ("legged", "chamber"))
def test_batched_legacy_custom_designs_upgrade_each_lane(
    family: str, dtype: np.dtype[np.floating]
) -> None:
    decoder = LeggedDecoder(2, dtype) if family == "legged" else ChamberDecoder(2, "direct", dtype)
    cfg = BodyConfig(grid_shape=(2, 1), size=(0.1, 0.05), settle_steps=0, horizon=1, substeps=1)
    env, embedding = make_vec(
        "SoftMPM2D-v0",
        num_envs=2,
        dtype=dtype,
        device="cpu",
        config=cfg,
        decoder=decoder,
        max_observation_space_size=10000,
    )
    try:
        legacy = {k: v for k, v in embedding.items() if k != "poisson_values"}
        legacy["poisson"] = np.asarray([0.29, 0.41], dtype)
        candidates = env._candidates(legacy)
        for lane, candidate in enumerate(candidates):
            assert candidate["poisson_values"].shape == (3, 3, 1)
            _, model, errors = env.envs[lane]._prepare(candidate)
            assert errors == []
            np.testing.assert_allclose(model.poisson, [0.29, 0.41][lane], rtol=2e-6)
    finally:
        env.close()
