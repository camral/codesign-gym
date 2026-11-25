"""Portable, non-pickle design serialization; no executable object loading."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import cast

import numpy as np

from ...contracts.fields import POISSON_BOUNDS
from ..types import Embedding, HostEmbedding, JsonValue, PathLike
from .decoders import (
    Decoder,
    FieldSpec,
    check_embedding,
    decoder_from_descriptor,
    migrate_legacy_embedding,
)


def _canonical_descriptor(desc: dict[str, object]) -> dict[str, object]:
    """Only fill documented FieldSpec defaults; keep other descriptor checks strict."""
    desc = dict(desc)
    # Custom decoders own their descriptors; a FieldSpec entry is optional.
    # Without one, preserve exact comparison against the supplied instance.
    if "spec" not in desc:
        return desc
    spec = dict(cast(dict[str, object], desc["spec"]))
    for name in ("palette", "young_bounds", "density_bounds", "poisson_bounds"):
        if name in spec:
            spec[name] = tuple(cast(list[float], spec[name]))
    # Constructor validates decoded JSON values.
    desc["spec"] = asdict(FieldSpec(**spec))  # type: ignore[arg-type]
    return desc


def _errors(decoder: Decoder, embodiment: Embedding, body_parameters: bool = False) -> list[str]:
    """Validate either the current embedding or its explicit legacy Poisson override."""
    if not body_parameters:
        return check_embedding(decoder, embodiment)
    base = {k: v for k, v in embodiment.items() if k != "poisson"}
    errors = check_embedding(decoder, base)
    try:
        nu = np.asarray(embodiment["poisson"])
        valid = (
            nu.shape == ()
            and nu.dtype.kind in "iuf"
            and bool(np.isfinite(nu))
            and POISSON_BOUNDS[0] <= float(nu) <= POISSON_BOUNDS[1]
        )
    except (KeyError, ValueError, TypeError, OverflowError):
        valid = False
    if not valid:
        errors.append("POISSON_OUT_OF_BOUNDS")
    return errors


def save_design(
    path: PathLike,
    embodiment: Embedding,
    decoder: Decoder,
    metadata: dict[str, object] | None = None,
) -> None:
    """Save a validated design as portable JSON including decoder metadata.

    Arrays retain dtype and shape. Parent directories are created; invalid or
    nonfinite designs fail before a usable design artifact is written.
    """
    if hasattr(decoder, "upgrade_embedding"):
        embodiment = decoder.upgrade_embedding(embodiment)
    body_parameters = "poisson" in embodiment and "poisson" not in decoder.space().spaces
    errors = _errors(decoder, embodiment, body_parameters)
    if errors:
        raise ValueError(f"Cannot serialize invalid embedding: {errors}")
    arrays = {
        k: {
            "dtype": str(np.asarray(v).dtype),
            "shape": list(np.shape(v)),
            "data": np.asarray(v).tolist(),
        }
        for k, v in embodiment.items()
    }
    doc = {
        "format": "codesign-soft-design",
        "version": 3,
        "decoder": decoder.descriptor(),
        "embodiment": arrays,
        "parameter_schema": "soft-spatial-poisson-v1",
        "legacy_global_poisson_override": body_parameters,
        "metadata": metadata or {},
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, allow_nan=False) + "\n")


def load_design(
    path: PathLike, decoder: Decoder | None = None
) -> tuple[HostEmbedding, Decoder, dict[str, JsonValue]]:
    """Load and validate a bounded-size design file without executable objects.

    Exact legacy layouts are migrated explicitly. Supplied decoder descriptors
    must match, so a file is never silently reinterpreted with another generator.
    """
    path = Path(path)
    if path.stat().st_size > 32 * 1024 * 1024:
        raise ValueError("Design JSON exceeds 32 MB safety bound")
    d = json.loads(path.read_text())
    if d.get("format") != "codesign-soft-design" or d.get("version") not in (1, 2, 3):
        raise ValueError("Unsupported design format")
    if d["version"] == 2 and d.get("parameter_schema") != "soft-body-global-poisson-v1":
        raise ValueError("Unknown environment parameter schema")
    if d["version"] == 3 and (
        d.get("parameter_schema") != "soft-spatial-poisson-v1"
        or type(d.get("legacy_global_poisson_override")) is not bool
    ):
        raise ValueError("Unknown spatial material parameter schema")
    new_keys = {"poisson_bounds", "poisson_default"}
    has_spec = "spec" in d["decoder"]
    stored_keys = new_keys.intersection(d["decoder"].get("spec", {}))
    if stored_keys and stored_keys != new_keys:
        raise ValueError("Incomplete spatial material decoder schema")
    legacy_raw = has_spec and not stored_keys
    if d["version"] == 3 and legacy_raw:
        raise ValueError("Spatial material schema requires the current decoder descriptor")
    if decoder is None and not has_spec:
        raise ValueError("A custom descriptor without spec requires an explicit decoder instance")
    decoder = decoder or decoder_from_descriptor(d["decoder"])
    if json.dumps(_canonical_descriptor(decoder.descriptor()), sort_keys=True) != json.dumps(
        _canonical_descriptor(d["decoder"]), sort_keys=True
    ):
        raise ValueError(
            "Decoder descriptor mismatch; do not reinterpret saved embeddings silently"
        )
    emb = {}
    for k, a in d["embodiment"].items():
        dt = np.dtype(a["dtype"])
        if dt.kind not in "biuf":
            raise ValueError("Only numeric design arrays are supported")
        value = np.asarray(a["data"], dtype=dt)
        if list(value.shape) != a["shape"]:
            raise ValueError(f"Stored shape mismatch: {k}")
        emb[k] = value
    if legacy_raw:
        emb = (
            decoder.upgrade_embedding(emb)
            if hasattr(decoder, "upgrade_embedding")
            else migrate_legacy_embedding(decoder, emb)
        )
    elif hasattr(decoder, "upgrade_embedding"):
        emb = decoder.upgrade_embedding(emb)
    body_parameters = d["version"] == 2 or (
        d["version"] == 3 and d["legacy_global_poisson_override"]
    )
    errors = _errors(decoder, emb, body_parameters)
    if errors:
        raise ValueError(f"Invalid saved embedding: {errors}")
    return emb, decoder, d.get("metadata", {})
