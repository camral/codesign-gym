"""Backend-independent validation at registered Gym batch boundaries."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import cast

import numpy as np
from numpy.typing import ArrayLike

from .api_types import EmbodimentInput, HostArray, RGBFrame


def stack_info(values: Sequence[object]) -> object:
    """Stack matching host values, recurse through dicts and retain irregular lists."""
    if all(isinstance(value, dict) for value in values):
        mappings = cast("Sequence[dict[str, object]]", values)
        keys: set[str] = set().union(*mappings)
        return {key: stack_info([value.get(key) for value in mappings]) for key in keys}
    if all(
        isinstance(value, (bool, int, float, np.number, np.ndarray)) and not isinstance(value, str)
        for value in values
    ):
        arrays = [np.asarray(value) for value in values]
        if len({array.shape for array in arrays}) == 1:
            return np.stack(arrays)
    return list(values)


def render_lanes(
    count: int, env_idx: int | None, frame: Callable[[int], RGBFrame | None]
) -> RGBFrame | tuple[RGBFrame | None, ...] | None:
    """Select one lane or an ordered tuple after adapter-specific render checks."""
    if env_idx is not None and not 0 <= env_idx < count:
        raise IndexError("env_idx out of range")
    indices = range(count) if env_idx is None else [env_idx]
    frames = tuple(frame(index) for index in indices)
    return frames if env_idx is None else frames[0]


def split_embodiment(value: EmbodimentInput, count: int) -> list[HostArray | None]:
    """Normalize a shared vector or per-lane vectors without copying caller data.

    A missing value reuses that lane's accepted embodiment. Invalid batch rank or
    length raises before any reset or simulation mutation.
    """
    if value is None:
        return [None] * count
    if isinstance(value, np.ndarray):
        if value.ndim == 1:
            return [value] * count
        if value.ndim == 2:
            if value.shape[0] != count:
                raise ValueError(f"batched embodiment length {value.shape[0]} != num_envs {count}")
            return [value[i] for i in range(count)]
        raise ValueError(f"embodiment array must be (D,) or (num_envs, D), got shape {value.shape}")
    values = list(cast(Sequence[ArrayLike | None], value))
    if len(values) != count:
        raise ValueError(f"embodiment sequence length {len(values)} != num_envs {count}")
    return [None if item is None else np.asarray(item) for item in values]


def bounded_actions(value: ArrayLike, shape: tuple[int, ...]) -> HostArray:
    """Validate the entire padded action batch before any lane can advance."""
    actions = np.asarray(value)
    if actions.shape != shape:
        raise ValueError(f"actions must have shape {shape}, got {actions.shape}")
    if (
        actions.dtype.kind not in "fiu"
        or not np.all(np.isfinite(actions))
        or np.any(actions < -1)
        or np.any(actions > 1)
    ):
        raise ValueError("Actions must be finite real numbers bounded by [-1, 1]")
    return actions
