"""Spatial material heads and saved-design compatibility, without simulation."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from envs.soft_walker.engine.embodiments.decoders import Decoder
    from envs.soft_walker.engine.types import Embedding


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


import json
from copy import deepcopy

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from envs.soft_walker.engine.embodiments.decoders import (
    CPPNDecoder,
    Fields,
    FieldSpec,
    GridDecoder,
    PrimitiveDecoder,
    baseline_raw,
    cast_embedding,
    check_embedding,
    interpret,
    make_decoder,
    migrate_legacy_embedding,
    set_poisson,
)
from envs.soft_walker.engine.embodiments.serialization import load_design, save_design


def spatial_embedding(decoder: Decoder) -> Embedding:
    emb = decoder.base()
    if isinstance(decoder, GridDecoder):
        values = np.linspace(-2.0, 2.0, decoder.shape[0], dtype=decoder.dtype)
        emb["values"][..., -1] = values.reshape((-1,) + (1,) * (decoder.spec.dim - 1))
    elif isinstance(decoder, CPPNDecoder):
        emb["weights"][-1] = 0
        emb["weights"][-1, 0] = 2.0
        emb["bias"][-1] = 0.0
    else:
        emb["enabled"][:] = 1
        emb["center"][:, 0] = np.linspace(-0.6, 0.6, decoder.count)
        emb["half_size"][:] = 0.45
        emb["features"][:, -1] = np.linspace(-2.0, 2.0, decoder.count)
    return emb


@pytest.mark.parametrize("name", ("direct", "bspline", "cppn", "primitives"))
@pytest.mark.parametrize("dim", (2, 3))
@pytest.mark.parametrize("mode", ("hard", "relaxed"))
def test_spatial_poisson_and_independent_head_gradient(
    name: str, dim: int, mode: str, dtype: np.dtype[np.floating]
) -> None:
    decoder = make_decoder(name, spec=FieldSpec(dim=dim, design_mode=mode), dtype=dtype)
    emb = spatial_embedding(decoder)
    assert not check_embedding(decoder, emb)
    points = np.zeros((3, dim), dtype=dtype)
    points[:, 0] = [-0.6, 0.1, 0.6]
    points = jnp.asarray(points)
    base = cast_embedding(decoder, emb)
    fields = jax.jit(decoder.decode)(base, points)
    assert fields.poisson.shape == (3,)
    assert float(jnp.ptp(fields.poisson)) > 0.05
    assert np.all(np.asarray(fields.poisson) >= np.asarray(0.25, dtype))
    assert np.all(np.asarray(fields.poisson) <= np.asarray(0.45, dtype))

    def shifted(delta: jax.Array) -> Fields:
        changed = dict(base)
        key = (
            "values"
            if isinstance(decoder, GridDecoder)
            else "bias"
            if isinstance(decoder, CPPNDecoder)
            else "features"
        )
        changed[key] = changed[key].at[..., -1].add(delta)
        return decoder.decode(changed, points)

    compiled = jax.jit(shifted)
    unchanged = compiled(jnp.asarray(0.0, dtype))
    other = compiled(jnp.asarray(0.17, dtype))
    for before, after in zip(unchanged[:7], other[:7]):
        np.testing.assert_array_equal(before, after)
    weights = jnp.asarray([0.2, 0.7, 1.3], dtype)
    loss = lambda delta: jnp.sum(shifted(delta).poisson * weights)
    derivative = float(jax.jit(jax.grad(loss))(jnp.asarray(0.0, dtype)))
    epsilon = 2e-3 if dtype == np.dtype("float32") else 1e-4
    fd = float(
        (loss(jnp.asarray(epsilon, dtype)) - loss(jnp.asarray(-epsilon, dtype))) / (2 * epsilon)
    )
    assert derivative > 1e-4 and np.isfinite(derivative)
    np.testing.assert_allclose(
        derivative, fd, rtol=3e-3 if dtype == np.dtype("float32") else 2e-7, atol=1e-8
    )


@pytest.mark.parametrize("name", ("direct", "bspline", "cppn", "primitives"))
def test_uniform_poisson_helper_and_spatial_roundtrip(
    name: str, dtype: np.dtype[np.floating], tmp_path: Path
) -> None:
    decoder = make_decoder(name, dtype=dtype)
    original = spatial_embedding(decoder)
    points = jnp.asarray([[-0.6, 0.2], [0.3, -0.2], [0.6, 0.1]], dtype)
    original_fields = decoder.decode(cast_embedding(decoder, original), points)
    original_copy = deepcopy(original)
    for value in (0.25, 0.31, 0.35, 0.45):
        updated = set_poisson(decoder, original, value)
        assert not check_embedding(decoder, updated)
        if isinstance(decoder, CPPNDecoder):
            prefix = decoder.inputs + decoder.hidden
            assert not np.any(updated["edge_mask"][-1, prefix:])
            np.testing.assert_array_equal(updated["edge_mask"][:-1], original["edge_mask"][:-1])
        fields = decoder.decode(cast_embedding(decoder, updated), points)
        np.testing.assert_allclose(
            fields.poisson, value, rtol=0.0, atol=6e-8 if dtype == np.dtype("float32") else 1e-15
        )
        for before, after in zip(original_fields[:7], fields[:7]):
            np.testing.assert_array_equal(before, after)
    for key in original:
        np.testing.assert_array_equal(original[key], original_copy[key])
    for value in (0.24, 0.46, np.nan, np.inf, [0.35]):
        with pytest.raises(ValueError, match="Poisson"):
            set_poisson(decoder, original, value)
    with pytest.raises(ValueError, match="legacy global"):
        set_poisson(decoder, dict(original, poisson=np.asarray(0.31, dtype)), 0.37)
    path = tmp_path / (name + ".json")
    save_design(path, original, decoder, {"spatial": True})
    document = json.loads(path.read_text())
    assert document["version"] == 3 and document["parameter_schema"] == "soft-spatial-poisson-v1"
    loaded, rebuilt, metadata = load_design(path)
    assert metadata == {"spatial": True}
    for key in original:
        np.testing.assert_array_equal(original[key], loaded[key])
    restored = rebuilt.decode(cast_embedding(rebuilt, loaded), points)
    for before, after in zip(original_fields, restored):
        np.testing.assert_array_equal(before, after)


def legacy_document(decoder: Decoder, version: int) -> tuple[dict[str, object], Embedding]:
    emb = decoder.base()
    if isinstance(decoder, GridDecoder):
        emb["values"] = emb["values"][..., :-1].copy()
    elif isinstance(decoder, PrimitiveDecoder):
        emb["features"] = emb["features"][..., :-1].copy()
    else:
        emb["weights"] = emb["weights"][:-1, :-1].copy()
        emb["edge_mask"] = emb["edge_mask"][:-1, :-1].copy()
        emb["bias"] = emb["bias"][:-1].copy()
    desc = decoder.descriptor()
    desc["spec"].pop("poisson_bounds")
    desc["spec"].pop("poisson_default")
    if version == 2:
        emb["poisson"] = np.asarray(0.31, decoder.dtype)
    arrays = {
        key: {"dtype": str(value.dtype), "shape": list(value.shape), "data": value.tolist()}
        for key, value in emb.items()
    }
    doc = dict(
        format="codesign-soft-design",
        version=version,
        decoder=desc,
        embodiment=arrays,
        metadata={"old": True},
    )
    if version == 2:
        doc["parameter_schema"] = "soft-body-global-poisson-v1"
    return doc, emb


@pytest.mark.parametrize("name", ("direct", "bspline", "cppn", "primitives"))
@pytest.mark.parametrize("version", (1, 2))
def test_legacy_saved_design_migration_preserves_old_coordinates(
    name: str, version: int, dtype: np.dtype[np.floating], tmp_path: Path
) -> None:
    decoder = make_decoder(name, spec=FieldSpec(dim=3), dtype=dtype)
    doc, legacy = legacy_document(decoder, version)
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(doc))
    original_bytes = path.read_bytes()
    loaded, rebuilt, metadata = load_design(path)
    assert path.read_bytes() == original_bytes
    assert metadata == {"old": True}
    if version == 2:
        np.testing.assert_array_equal(loaded["poisson"], legacy["poisson"])
    base = {key: value for key, value in loaded.items() if key != "poisson"}
    assert not check_embedding(rebuilt, base)
    if isinstance(decoder, GridDecoder):
        np.testing.assert_array_equal(loaded["values"][..., :-1], legacy["values"])
    elif isinstance(decoder, PrimitiveDecoder):
        np.testing.assert_array_equal(loaded["features"][..., :-1], legacy["features"])
    else:
        for key in ("weights", "edge_mask"):
            np.testing.assert_array_equal(loaded[key][:-1, :-1], legacy[key])
        np.testing.assert_array_equal(loaded["bias"][:-1], legacy["bias"])
        assert not np.any(loaded["weights"][-1])
    points = jnp.asarray([[-0.7, 0.2, 0.0], [0.7, -0.2, 0.3]], dtype)
    raw = rebuilt.raw(cast_embedding(rebuilt, base), points)
    fields = rebuilt.decode(cast_embedding(rebuilt, base), points)
    old_fields = interpret(raw[..., :-1], rebuilt.spec)
    for old, current in zip(old_fields[:7], fields[:7]):
        np.testing.assert_array_equal(old, current)
    np.testing.assert_array_equal(fields.poisson, np.full(2, 0.35, dtype))
    explicit, _, _ = load_design(path, decoder=decoder)
    for key in loaded:
        np.testing.assert_array_equal(explicit[key], loaded[key])
    save_design(tmp_path / "migrated.json", loaded, rebuilt, metadata)
    saved = json.loads((tmp_path / "migrated.json").read_text())
    assert saved["version"] == 3 and saved["legacy_global_poisson_override"] == (version == 2)
    again, _, _ = load_design(tmp_path / "migrated.json")
    for key in loaded:
        np.testing.assert_array_equal(again[key], loaded[key])


@pytest.mark.parametrize("dim", (2, 3))
@pytest.mark.parametrize("version", (1, 2))
def test_legacy_cppn_load_remains_usable_for_implicit_parameterization(
    dim: int, version: int, dtype: np.dtype[np.floating], tmp_path: Path
) -> None:
    from envs.soft_walker.shared import CPPNParameterization

    decoder = CPPNDecoder(FieldSpec(dim=dim, design_mode="relaxed"), hidden=2, dtype=dtype)
    doc, legacy = legacy_document(decoder, version)
    prefix = decoder.inputs + decoder.hidden
    # The historical implicit/random prior permits input/hidden connections,
    # not output-to-output edges (even when their weights are zero).
    legacy["edge_mask"][:, prefix:] = 0
    doc["embodiment"]["edge_mask"]["data"] = legacy["edge_mask"].tolist()
    path = tmp_path / "implicit_legacy.json"
    path.write_text(json.dumps(doc))
    loaded, rebuilt, _ = load_design(path)
    np.testing.assert_array_equal(loaded["edge_mask"][:-1, :-1], legacy["edge_mask"])
    assert not np.any(loaded["edge_mask"][:, prefix:])
    coordinates = CPPNParameterization(rebuilt, loaded)
    points = jnp.asarray([[-0.6] * dim, [0.3] * dim], dtype)

    def poisson(theta: jax.Array) -> jax.Array:
        return rebuilt.decode(coordinates.unpack(theta), points).poisson

    initial = jnp.asarray(coordinates.initial)
    np.testing.assert_allclose(
        jax.jit(poisson)(initial),
        0.31 if version == 2 else 0.35,
        rtol=0,
        atol=6e-8 if dtype == np.dtype("float32") else 1e-15,
    )
    gradient = jax.jit(jax.grad(lambda theta: jnp.sum(poisson(theta))))(initial)
    assert np.linalg.norm(np.asarray(gradient)[coordinates.groups["poisson"]]) > 1e-5


def test_poisson_head_does_not_compete_with_route_labels(dtype: np.dtype[np.floating]) -> None:
    decoder = GridDecoder(
        FieldSpec(channels=3, material_mode="categorical"), dtype=dtype, discrete_labels=True
    )
    emb = decoder.base()
    emb["labels"][..., 0] = 2
    emb["labels"][..., 1] = 1
    emb["labels"][..., 2] = 2
    emb["values"][..., -1] = 6.0
    fields = decoder.decode(cast_embedding(decoder, emb), jnp.zeros((2, 2), dtype))
    np.testing.assert_array_equal(fields.channel, [2, 2])
    np.testing.assert_array_equal(fields.actuator_type, [1, 1])
    np.testing.assert_array_equal(fields.young, np.full(2, 80000.0, dtype))
    np.testing.assert_array_equal(fields.poisson, np.full(2, 0.45, dtype))
    raw = baseline_raw(np.zeros((2, 2)), decoder.spec, dtype)
    with pytest.raises(ValueError, match="width"):
        interpret(jnp.asarray(raw[..., :-2]), decoder.spec)


@pytest.mark.parametrize(
    "change",
    (
        {"poisson_bounds": (0.1, 0.45)},
        {"poisson_bounds": (0.25, 0.5)},
        {"poisson_bounds": (0.4, 0.3)},
        {"poisson_default": 0.5},
        {"poisson_default": float("nan")},
    ),
)
def test_poisson_spec_rejects_invalid_physical_bounds(change: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="Poisson"):
        FieldSpec(**change)


def test_old_fields_constructor_and_custom_embedding_upgrade(
    dtype: np.dtype[np.floating],
) -> None:
    values = jnp.ones(3, dtype)
    fields = Fields(
        values,
        values,
        values,
        values,
        jnp.ones((3, 2), dtype),
        jnp.ones(3, jnp.int32),
        jnp.zeros(3, jnp.int32),
    )
    assert fields.poisson is None

    class UpgradingGrid(GridDecoder):
        def upgrade_embedding(self, emb: Embedding) -> Embedding:
            if np.shape(emb["values"])[-1] == self.spec.poisson_index:
                return migrate_legacy_embedding(self, emb)
            return emb

    decoder = UpgradingGrid(dtype=dtype)
    _, legacy = legacy_document(decoder, 1)
    assert not check_embedding(decoder, legacy)
    cast = cast_embedding(decoder, legacy)
    assert cast["values"].shape[-1] == decoder.spec.width


def test_serialized_schema_and_partial_migration_fail_closed(
    dtype: np.dtype[np.floating], tmp_path: Path
) -> None:
    decoder = GridDecoder(dtype=dtype)
    doc, _ = legacy_document(decoder, 1)
    doc["embodiment"]["values"]["shape"][-1] -= 1
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="shape"):
        load_design(path)
    save_design(path, decoder.base(), decoder)
    doc = json.loads(path.read_text())
    doc["parameter_schema"] = "unknown"
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="schema"):
        load_design(path)


@pytest.mark.parametrize("version,override", ((1, False), (2, True), (3, False), (3, True)))
def test_custom_descriptor_without_spec_roundtrip(
    version: int, override: bool, dtype: np.dtype[np.floating], tmp_path: Path
) -> None:
    """The shipped custom decoder owns its descriptor and needs no raw migration."""
    from importlib.util import module_from_spec, spec_from_file_location
    from pathlib import Path

    example = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "envs"
        / "soft_walker"
        / "examples"
        / "custom_decoder.py"
    )
    module_spec = spec_from_file_location("_spatial_material_custom_example", example)
    module = module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    decoder = module.UniformMaterialDecoder(dtype=dtype)
    assert "spec" not in decoder.descriptor()
    embedding = decoder.base()
    if override:
        embedding["poisson"] = np.asarray(0.31, dtype)
    path = tmp_path / "custom.json"
    save_design(path, embedding, decoder, {"custom": True})
    document = json.loads(path.read_text())
    if version != 3:
        document["version"] = version
        document.pop("legacy_global_poisson_override")
        if version == 1:
            document.pop("parameter_schema")
        else:
            document["parameter_schema"] = "soft-body-global-poisson-v1"
        path.write_text(json.dumps(document))
    loaded, rebuilt, metadata = load_design(path, decoder=decoder)
    assert rebuilt is decoder and metadata == {"custom": True}
    for name, value in embedding.items():
        np.testing.assert_array_equal(loaded[name], value)
        assert loaded[name].dtype == value.dtype
    points = jnp.asarray([[-0.7, 0.2], [0.7, -0.2]], dtype)
    before = decoder.decode(cast_embedding(decoder, embedding), points)
    after = rebuilt.decode(cast_embedding(rebuilt, loaded), points)
    for expected, actual in zip(before, after):
        np.testing.assert_array_equal(actual, expected)
    assert float(jnp.ptp(after.poisson)) > 0.05
    with pytest.raises(ValueError, match="explicit decoder"):
        load_design(path)
    document["decoder"]["version"] += 1
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="descriptor mismatch"):
        load_design(path, decoder=decoder)


@pytest.mark.parametrize("missing", ("spec", "both_poisson_keys", "one_poisson_key"))
@pytest.mark.parametrize("explicit", (False, True))
def test_builtin_spatial_schema_remains_strict(
    missing: str, explicit: bool, dtype: np.dtype[np.floating], tmp_path: Path
) -> None:
    decoder = GridDecoder(dtype=dtype)
    path = tmp_path / "incomplete_builtin.json"
    save_design(path, decoder.base(), decoder)
    document = json.loads(path.read_text())
    if missing == "spec":
        document["decoder"].pop("spec")
    else:
        document["decoder"]["spec"].pop("poisson_default")
        if missing == "both_poisson_keys":
            document["decoder"]["spec"].pop("poisson_bounds")
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError):
        load_design(path, decoder=decoder if explicit else None)
