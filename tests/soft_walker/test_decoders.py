"""Decoder field shapes, serialization, validation, and differentiable mappings."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from envs.soft_walker.engine.embodiments.decoders import Fields
    from envs.soft_walker.engine.types import Embedding


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from envs.soft_walker.engine import make
from envs.soft_walker.engine.embodiments.decoders import (
    BSplineDecoder,
    CPPNDecoder,
    FieldSpec,
    GridDecoder,
    PrimitiveDecoder,
    baseline_raw,
    cast_embedding,
    check_embedding,
    interpolate,
    interpret,
    make_decoder,
)
from envs.soft_walker.engine.embodiments.serialization import load_design, save_design
from envs.soft_walker.engine.physics.spring_config import SpringConfig


@pytest.mark.parametrize("name", ["direct", "cppn", "primitives", "bspline"])
@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("mode", ["hard", "relaxed"])
def test_decoder_fields_and_roundtrip(
    name: str, dim: int, mode: str, dtype: np.dtype[np.floating], tmp_path: Path
) -> None:
    spec = FieldSpec(dim=dim, design_mode=mode)
    d = make_decoder(name, spec=spec, dtype=dtype)
    b = d.base()
    assert d.space().contains(b) and not check_embedding(d, b)
    p = jnp.asarray(np.random.default_rng(132).uniform(-0.7, 0.7, (7, dim)), dtype=dtype)
    fields = jax.jit(d.decode)(cast_embedding(d, b), p)
    assert fields.occupancy.shape == (7,) and fields.direction.shape == (7, dim)
    for field in fields:
        assert np.isfinite(field).all()
    np.testing.assert_allclose(np.linalg.norm(fields.direction, axis=-1), 1, atol=2e-6)
    assert (np.asarray(fields.young) >= 5000).all() and (np.asarray(fields.young) <= 100000).all()
    assert (np.asarray(fields.density) >= 800).all() and (np.asarray(fields.density) <= 1250).all()
    if mode == "hard":
        assert np.isin(fields.occupancy, [0, 1]).all()
    else:
        assert ((fields.occupancy >= spec.occupancy_floor) & (fields.occupancy <= 1)).all()
    path = tmp_path / f"{name}_{dim}_{mode}.json"
    save_design(path, b, d, {"units": "SI"})
    loaded, d2, meta = load_design(path)
    assert meta["units"] == "SI"
    actual = d2.decode(cast_embedding(d2, loaded), p)
    for x, y in zip(fields, actual):
        np.testing.assert_allclose(x, y, atol=3e-6 if dtype == np.float32 else 1e-12)


@pytest.mark.parametrize("cubic", [False, True])
@pytest.mark.parametrize("dim", [2, 3])
def test_partition_of_unity_and_boundary_rules(
    cubic: bool, dim: int, dtype: np.dtype[np.floating]
) -> None:
    n = 4 if cubic else 3
    coeff = jnp.full((n,) * dim + (2,), 2.35, dtype=dtype)
    pts = jnp.asarray(
        [[-1.0] * dim, [1.0] * dim, [0.0] * dim, [1.2] * dim, [-1.5] * dim], dtype=dtype
    )
    out = jax.jit(lambda c, p: interpolate(c, p, cubic))(coeff, pts)
    np.testing.assert_allclose(out, 2.35, atol=2e-6 if dtype == np.float32 else 1e-12)


def test_direct_integer_categories(dtype: np.dtype[np.floating]) -> None:
    d = GridDecoder(
        FieldSpec(material_mode="categorical"), shape=(2, 2), dtype=dtype, discrete_labels=True
    )
    b = d.base()
    b["labels"][..., 0] = 2
    b["labels"][..., 1] = 0
    f = d.decode(cast_embedding(d, b), jnp.zeros((3, 2), dtype=dtype))
    np.testing.assert_allclose(f.young, 80000, rtol=1e-7)
    assert np.all(f.strength == 0)
    b["labels"] = b["labels"].astype(dtype)
    assert check_embedding(d, b)


def test_relaxed_material_energy_mixing(dtype: np.dtype[np.floating]) -> None:
    spec = FieldSpec(material_mode="categorical", design_mode="relaxed")
    raw = baseline_raw(np.zeros((2, 2)), spec, dtype)
    raw[:, 6:9] = np.array([-0.4, 0.2, 0.6])
    f = interpret(jnp.asarray(raw), spec)
    w = np.exp(np.array([-0.4, 0.2, 0.6]))
    w = w / w.sum()
    np.testing.assert_allclose(f.young, np.dot(w, spec.palette), rtol=2e-6)


def test_hard_geometry_gradient_is_zero_off_threshold(dtype: np.dtype[np.floating]) -> None:
    d = GridDecoder(dtype=dtype)
    b = cast_embedding(d, d.base())
    p = jnp.asarray([[0.13, 0.24]], dtype=dtype)

    def f(x: jax.Array) -> jax.Array:
        return jnp.sum(d.decode({"values": b["values"].at[..., 0].add(x)}, p).occupancy)

    assert float(jax.grad(f)(jnp.asarray(0.1, dtype))) == 0


def test_cppn_graph_rejection_and_mask_semantics(dtype: np.dtype[np.floating]) -> None:
    d = CPPNDecoder(dtype=dtype)
    b = d.base()
    b["edge_mask"][0, -1] = 1
    assert "CPPN_CYCLE_OR_FORWARD_EDGE" in check_embedding(d, b)
    b = d.base()
    p = jnp.asarray([[0.2, -0.3]], dtype=dtype)
    e = cast_embedding(d, b)
    _baseline = d.raw(e, p)
    # Disabled edges do not affect outputs and their gradient is exactly zero.
    edge = (0, 0)
    e["edge_mask"] = e["edge_mask"].at[edge].set(0)
    g = jax.grad(lambda w: jnp.sum(d.raw({**e, "weights": w}, p)))(e["weights"])
    assert float(g[edge]) == 0


@pytest.mark.parametrize("kind", [0, 1, 2])
def test_each_primitive_kind(dtype: np.dtype[np.floating], kind: int) -> None:
    d = PrimitiveDecoder(dtype=dtype)
    b = d.base()
    b["kind"][:] = kind
    pts = jnp.asarray([[0.0, 0.0], [0.1, 0.1], [1.0, 1.0]], dtype=dtype)
    f = d.decode(cast_embedding(d, b), pts)
    assert f.occupancy[0] == 1 and f.occupancy[-1] == 0
    b["enabled"][:] = 0
    assert "NO_ENABLED_PRIMITIVE" in check_embedding(d, b)
    b = d.base()
    b["axis"][:] = 0
    assert "DEGENERATE_PRIMITIVE_AXIS" in check_embedding(d, b)


def test_custom_decoder_uses_same_pipeline(dtype: np.dtype[np.floating]) -> None:
    class UserDecoder(GridDecoder):
        def decode(self, emb: Embedding, points: jax.Array) -> Fields:
            f = super().decode(emb, points)
            return f._replace(young=jnp.full_like(f.young, 25000.0))

        def descriptor(self) -> dict[str, object]:
            d = super().descriptor()
            d["name"] = "user_constant_stiffness"
            return d

    d = UserDecoder(dtype=dtype)
    e, b = make(dtype=dtype, decoder=d, config=SpringConfig(grid_shape=(2, 1), settle_steps=0))
    assert e.validate_embodiment(b)
    assert not e.reset(embodiment=b)[1][0]
    np.testing.assert_array_equal(e._model.young, np.full(2, 25000.0))


def test_custom_decoder_bad_fields_rejected(dtype: np.dtype[np.floating]) -> None:
    class InvalidDecoder(GridDecoder):
        def decode(self, emb: Embedding, points: jax.Array) -> Fields:
            f = super().decode(emb, points)
            return f._replace(young=jnp.full_like(f.young, jnp.nan))

    d = InvalidDecoder(dtype=dtype)
    e, b = make(dtype=dtype, decoder=d, config=SpringConfig(grid_shape=(2, 1), settle_steps=0))
    assert not e.validate_embodiment(b)
    assert e.reset(embodiment=b)[1][0]


def test_serialization_rejects_descriptor_mismatch(
    dtype: np.dtype[np.floating], tmp_path: Path
) -> None:
    d = GridDecoder(dtype=dtype)
    p = tmp_path / "d.json"
    save_design(p, d.base(), d)
    with pytest.raises(ValueError, match="mismatch"):
        load_design(p, decoder=BSplineDecoder(dtype=dtype))
    doc = json.loads(p.read_text())
    doc["version"] = 900
    p.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="format"):
        load_design(p)


@pytest.mark.parametrize(
    "change",
    [
        {"dim": 1},
        {"channels": 0},
        {"geometry_temperature": 0},
        {"occupancy_floor": 0},
        {"young_bounds": (-1, 2)},
        {"design_mode": "train"},
    ],
)
def test_spec_guards(change: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        FieldSpec(**change)
