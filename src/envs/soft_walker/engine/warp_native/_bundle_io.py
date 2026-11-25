"""Pickle-free transport for the distinct native resolved-bundle schemas."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, NotRequired, TypedDict

import numpy as np

from .host_types import FieldArrays, PathLike


class BundleMetadata(TypedDict):
    """JSON metadata interpreted and validated by each concrete bundle class."""

    schema: int | str
    channels: int
    # Dataclass keyword types differ by schema; JSON decoding is the boundary.
    config: dict[str, Any]
    h: NotRequired[float]
    area_ref: NotRequired[float]


def bundle_arrays(
    fields: FieldArrays, terrain: FieldArrays | None, initial: FieldArrays | None
) -> FieldArrays:
    """Pack existing field groups under their stable archive prefixes."""
    arrays = {f"model_{name}": np.asarray(value) for name, value in fields.items()}
    for prefix, values in (("terrain", terrain), ("state", initial)):
        if values is not None:
            arrays.update({f"{prefix}_{name}": np.asarray(v) for name, v in values.items()})
    return arrays


def write_archive(path: Path, metadata: BundleMetadata, arrays: FieldArrays) -> None:
    """Write the prepared metadata and arrays without changing their schema."""
    # Archive keys are model/recording fields, never NumPy's allow_pickle option.
    np.savez_compressed(path, metadata=np.array(json.dumps(metadata)), **arrays)  # type: ignore[arg-type]


def read_archive(
    path: PathLike, *, schema: int | str, schema_error: str
) -> tuple[BundleMetadata, dict[str, FieldArrays]]:
    """Read independent arrays, retaining each caller's schema error contract."""
    with np.load(path, allow_pickle=False) as archive:
        metadata: BundleMetadata = json.loads(str(archive["metadata"]))
        if metadata["schema"] != schema:
            raise ValueError(schema_error)
        config = metadata["config"]
        config["grid_shape"] = tuple(config["grid_shape"])
        config["origin"] = tuple(config["origin"])
        groups = {
            prefix: {
                key[len(prefix) + 1 :]: archive[key].copy()
                for key in archive.files
                if key.startswith(prefix + "_")
            }
            for prefix in ("model", "terrain", "state")
        }
    return metadata, groups
