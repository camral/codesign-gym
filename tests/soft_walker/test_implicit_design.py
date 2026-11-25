"""Morphology AD checks independent of dynamics or GPU resident execution."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from envs.soft_walker.engine.embodiments.decoders import check_embedding
from envs.soft_walker.engine.embodiments.morphology import (
    cell_center_points,
    generate_random_design,
)
from envs.soft_walker.engine.embodiments.serialization import load_design, save_design
from envs.soft_walker.shared import CPPNParameterization, make_implicit_design


@pytest.mark.parametrize("dim", (2, 3))
def test_implicit_coordinates_reproduce_seed_and_preserve_frozen_fields(
    dim: int, dtype: np.dtype[np.floating], tmp_path: Path
) -> None:
    decoder, embedding, metadata = make_implicit_design(dim, seed=7, dtype=dtype.name)
    coordinates = CPPNParameterization(decoder, embedding)
    initial = coordinates.unpack(coordinates.initial)
    for name in embedding:
        np.testing.assert_array_equal(initial[name], embedding[name])
    assert metadata["seed_generation"]["embedding_valid"]
    points = jnp.asarray(
        cell_center_points(tuple(2 * n for n in metadata["grid_shape"])), dtype=dtype
    )
    hard, raw_seed, _ = generate_random_design(
        "cppn", dim=dim, seed=7, dtype=dtype.name, grid_shape=metadata["grid_shape"]
    )
    seed_fields = hard.decode({k: jnp.asarray(v) for k, v in raw_seed.items()}, points)
    expanded_fields = coordinates.hard_decoder.decode(initial, points)
    for before, after in zip(seed_fields, expanded_fields):
        np.testing.assert_array_equal(before, after)
    updated = jax.jit(coordinates.unpack)(jnp.full_like(jnp.asarray(coordinates.initial), 0.14))
    before, after = decoder.decode(initial, points), decoder.decode(updated, points)
    for name in ("density", "actuator_type", "channel"):
        np.testing.assert_array_equal(getattr(before, name), getattr(after, name))
    assert "poisson" not in updated
    assert np.any(np.asarray(before.poisson) != np.asarray(after.poisson))
    np.testing.assert_array_equal(updated["edge_mask"], embedding["edge_mask"])
    np.testing.assert_array_equal(
        updated["weights"][: decoder.hidden], embedding["weights"][: decoder.hidden]
    )
    np.testing.assert_array_equal(
        updated["bias"][: decoder.hidden], embedding["bias"][: decoder.hidden]
    )
    assert np.any(np.asarray(before.occupancy) != np.asarray(after.occupancy))
    path = tmp_path / "implicit.json"
    save_design(path, embedding, decoder, metadata)
    loaded, restored_decoder, restored_metadata = load_design(path)
    restored = CPPNParameterization(restored_decoder, loaded)
    assert restored_metadata == json.loads(json.dumps(metadata))
    for name, value in restored.unpack(restored.initial).items():
        np.testing.assert_array_equal(value, initial[name])


@pytest.mark.parametrize("dim", (2, 3))
def test_implicit_semantic_groups_have_real_independent_gradients(
    dim: int, dtype: np.dtype[np.floating]
) -> None:
    decoder, embedding, metadata = make_implicit_design(dim, seed=8, dtype=dtype.name)
    coordinates = CPPNParameterization(decoder, embedding)
    points = jnp.asarray(cell_center_points(metadata["grid_shape"]), dtype=dtype)
    probe = jnp.asarray(np.arange(1, dim + 1), dtype=dtype)

    def quantities(theta: jax.Array) -> jax.Array:
        fields = decoder.decode(coordinates.unpack(theta), points)
        return jnp.stack(
            (
                jnp.mean(fields.occupancy),
                jnp.mean(jnp.log(fields.young)),
                jnp.mean(fields.strength),
                jnp.mean(fields.direction @ probe),
                jnp.mean(fields.poisson),
            )
        )

    jacobian = np.asarray(jax.jit(jax.jacrev(quantities))(jnp.asarray(coordinates.initial)))
    assert np.all(np.isfinite(jacobian))
    for index, group in enumerate(coordinates.groups.values()):
        assert np.linalg.norm(jacobian[index, group]) > 1e-5
        other = np.delete(jacobian[:, group], index, axis=0)
        np.testing.assert_array_equal(other, np.zeros_like(other))
    # Compare a genuinely mixed geometry/material/actuation directional derivative.
    direction = np.random.default_rng(17).normal(size=coordinates.initial.shape).astype(dtype)
    direction /= np.linalg.norm(direction)
    h = 2e-3 if dtype == np.dtype("float32") else 1e-4
    finite_difference = (
        np.asarray(quantities(h * direction)) - np.asarray(quantities(-h * direction))
    ) / (2 * h)
    np.testing.assert_allclose(
        jacobian @ direction,
        finite_difference,
        rtol=2e-2 if dtype == np.dtype("float32") else 2e-5,
        atol=2e-4 if dtype == np.dtype("float32") else 2e-8,
    )


@pytest.mark.parametrize("dim", (2, 3))
def test_implicit_bounding_and_hard_geometry_are_explicit(
    dim: int, dtype: np.dtype[np.floating]
) -> None:
    decoder, embedding, metadata = make_implicit_design(dim, dtype=dtype.name)
    coordinates = CPPNParameterization(decoder, embedding)
    assert decoder.spec.design_mode == "relaxed"
    assert coordinates.hard_decoder.spec.design_mode == "hard"
    for shift in (-1e6, 1e6):
        result = coordinates.unpack(np.full_like(coordinates.initial, shift))
        assert (
            check_embedding(
                decoder, {k: np.asarray(v) for k, v in result.items() if k != "poisson"}
            )
            == []
        )
    points = jnp.asarray(cell_center_points(metadata["grid_shape"]), dtype=dtype)
    relaxed = np.asarray(decoder.decode(coordinates.unpack(coordinates.initial), points).occupancy)
    hard = np.asarray(
        coordinates.hard_decoder.decode(coordinates.unpack(coordinates.initial), points).occupancy
    )
    assert np.all((relaxed >= decoder.spec.occupancy_floor) & (relaxed <= 1.0))
    assert set(np.unique(hard)) == {0.0, 1.0}
    with pytest.raises(ValueError, match="expected morphology coordinates"):
        coordinates.unpack(np.zeros(2))


@pytest.mark.parametrize(
    "kwargs", ({"dim": 1}, {"dim": 2, "geometry_temperature": 0}, {"dim": 2, "poisson": 0.7})
)
def test_implicit_design_invalid_configuration(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        make_implicit_design(**kwargs)
