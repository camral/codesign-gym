"""Random morphology contracts; no time integration or resident GPU execution."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from envs.soft_walker.engine.embodiments.decoders import Decoder
    from envs.soft_walker.engine.types import Embedding


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


import numpy as np
import pytest

from envs.soft_walker.engine.embodiments import morphology
from envs.soft_walker.engine.embodiments.decoders import check_embedding
from envs.soft_walker.engine.embodiments.serialization import load_design, save_design


@pytest.mark.parametrize("representation", morphology.REPRESENTATIONS)
@pytest.mark.parametrize("dim", (2, 3))
def test_random_morphology_valid_diverse_and_portable(
    representation: str, dim: int, dtype: np.dtype[np.floating], tmp_path: Path
) -> None:
    silhouettes = []
    for seed in range(3):
        decoder, emb, report = morphology.generate_random_design(
            representation, dim=dim, seed=seed, dtype=dtype.name
        )
        assert check_embedding(decoder, emb) == []
        assert report["embedding_valid"]
        assert not report["post_decode_repairs"]
        assert 1 <= report["attempts"] <= 100
        assert all(not row["topology_errors"] for row in report["lattices"])
        assert all(0.08 <= row["occupied_fraction"] <= 0.8 for row in report["lattices"])
        assert all(row["active_channels"] for row in report["lattices"])
        silhouettes.append(report["shape_sha256"])
        shape = (19, 17) if dim == 2 else (11, 9, 9)
        _, original = morphology.sample_fields(decoder, emb, shape)
        path = tmp_path / f"{representation}_{dim}_{seed}.json"
        save_design(path, emb, decoder, metadata=report)
        loaded, rebuilt, metadata = load_design(path)
        _, restored = morphology.sample_fields(rebuilt, loaded, shape)
        for before, after in zip(original, restored):
            np.testing.assert_array_equal(before, after)
        assert metadata["shape_sha256"] == report["shape_sha256"]
    assert len(set(silhouettes)) == len(silhouettes)


@pytest.mark.parametrize("representation", morphology.REPRESENTATIONS)
def test_random_morphology_seed_repeatability_and_parameters_matter(
    representation: str, dtype: np.dtype[np.floating]
) -> None:
    decoder, emb, report = morphology.generate_random_design(
        representation, seed=9, dtype=dtype.name
    )
    _, repeated, repeated_report = morphology.generate_random_design(
        representation, seed=9, dtype=dtype.name
    )
    assert report == repeated_report
    for key in emb:
        np.testing.assert_array_equal(emb[key], repeated[key])
    _, before = morphology.sample_fields(decoder, emb, (40, 32))
    altered = {key: value.copy() for key, value in emb.items()}
    if representation in ("direct", "bspline"):
        altered["values"][..., 0] = np.clip(altered["values"][..., 0] + 1.0, -6.0, 6.0)
    elif representation == "cppn":
        altered["bias"][decoder.hidden] += 1.0
    else:
        altered["radius"] *= 0.65
        altered["half_size"] *= 0.65
        altered["rounding"] *= 0.65
    assert check_embedding(decoder, altered) == []
    _, after = morphology.sample_fields(decoder, altered, (40, 32))
    assert np.any(np.asarray(before.occupancy) != np.asarray(after.occupancy))


def test_random_morphology_rejection_is_bounded_and_reported(
    monkeypatch: pytest.MonkeyPatch, dtype: np.dtype[np.floating]
) -> None:
    calls = []

    def invalid(decoder: Decoder, rng: np.random.Generator) -> Embedding:
        calls.append(1)
        emb = decoder.base()
        emb["enabled"][:] = 0
        return emb

    monkeypatch.setattr(morphology, "_primitives", invalid)
    with pytest.raises(morphology.GenerationError) as failure:
        morphology.generate_random_design("primitives", dtype=dtype.name, max_attempts=3)
    assert len(calls) == 3
    assert failure.value.report["attempts"] == 3
    assert failure.value.report["rejection_reasons"] == {"NO_ENABLED_PRIMITIVE": 3}


@pytest.mark.parametrize(
    "kwargs",
    (
        {"representation": "unknown"},
        {"dim": 1},
        {"seed": -1},
        {"grid_shape": (2, 3)},
        {"grid_shape": (12, 10, 8)},
        {"max_attempts": 0},
        {"dtype": "int32"},
    ),
)
def test_random_morphology_argument_validation(kwargs: dict[str, object]) -> None:
    arguments = {"representation": "direct", **kwargs}
    with pytest.raises(ValueError):
        morphology.generate_random_design(**arguments)
