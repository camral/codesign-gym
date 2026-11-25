"""
Authors: Aviraj Newatia

Seed parsing shared by the vector envs.

``reset(seed=...)`` takes an integer (``int`` or a NumPy integer) for the whole
batch or a sequence of ``num_envs`` integers, one per lane. Envs whose lanes own
their own RNG spawn an integer into ``num_envs`` independent lane seeds
(``numpy.random.SeedSequence``), so runs seeded ``s`` and ``s + 1`` share no lane;
a sequence is used as given. Batched envs whose lanes share one stream keep a
single stream for an integer and offset each lane by ``seed_i - seed_0`` for a
sequence.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def _as_int(value: Any, what: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{what} must be an integer, got {value!r}")
    return int(value)


def parse_seed(seed: Any, num_envs: int) -> int | list[int] | None:
    """None, one integer for the batch, or a list of ``num_envs`` per-lane integers."""
    if seed is None:
        return None
    if isinstance(seed, (int, np.integer)) and not isinstance(seed, (bool, np.bool_)):
        return int(seed)
    if isinstance(seed, np.ndarray) and seed.ndim == 0:
        return _as_int(seed.item(), "seed")
    if isinstance(seed, (Sequence, np.ndarray)) and not isinstance(seed, (str, bytes)):
        seeds = [_as_int(s, "each seed") for s in seed]
        if len(seeds) != num_envs:
            raise ValueError(f"seed sequence has {len(seeds)} entries, expected num_envs={num_envs}")
        return seeds
    raise TypeError(f"seed must be an integer or a sequence of {num_envs} integers, got {type(seed).__name__}")


def lane_seeds(seed: Any, num_envs: int) -> list[int | None]:
    """Per-lane seeds for envs whose lanes own their RNG. An integer is spawned into ``num_envs`` independent seeds
    (``SeedSequence(seed).spawn``), so lanes never share a stream with a run seeded differently; a sequence is used as
    given, so ``[s] * num_envs`` still starts every lane the same."""
    parsed = parse_seed(seed, num_envs)
    if parsed is None:
        return [None] * num_envs
    if isinstance(parsed, int):
        children = np.random.SeedSequence(parsed % 2**64).spawn(num_envs)   # SeedSequence takes non-negative entropy
        return [int(child.generate_state(1, np.uint32)[0] >> 1) for child in children]   # 31-bit: valid for every lane RNG
    return parsed
