"""
Authors: Aviraj Newatia

Batched design arguments for the vector envs with a flat design vector.

``reset``/``step``/``validate_embodiment`` take ``None``, one design (broadcast to
every lane), a ``(num_envs, n)`` batch, or a per-lane sequence whose entries are
designs or ``None``. ``None`` means "no design given for this lane": at reset the
lane gets the base design, at step it keeps the one it has.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


def _row(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def design_rows(embodiment: Any, base: np.ndarray, num_envs: int) -> tuple[np.ndarray, np.ndarray]:
    """``(rows, given)``: a ``(num_envs, n)`` float32 batch with ``base`` in the lanes
    no design was given for, and the boolean mask of lanes that were given one.
    Raises ``ValueError`` on a shape that fits neither form."""
    base = np.asarray(base, dtype=np.float32).reshape(-1)
    n = base.shape[0]
    rows = np.tile(base, (num_envs, 1))
    given = np.zeros(num_envs, dtype=bool)
    if embodiment is None:
        return rows, given
    if isinstance(embodiment, (list, tuple)) and len(embodiment) == num_envs and any(e is None for e in embodiment):
        for i, e in enumerate(embodiment):
            if e is not None:
                r = _row(e)
                if r.shape != (n,):
                    raise ValueError(f"lane {i}: embodiment shape {r.shape}, expected ({n},)")
                rows[i], given[i] = r, True
        return rows, given
    arr = np.stack([_row(e) for e in embodiment]) if isinstance(embodiment, (list, tuple)) and embodiment and np.ndim(embodiment[0]) == 1 else _row(embodiment)
    if arr.shape == (n,) or arr.shape == (1, n):
        return np.tile(arr.reshape(1, n), (num_envs, 1)), np.ones(num_envs, dtype=bool)
    if arr.shape == (num_envs, n):
        return arr.copy(), np.ones(num_envs, dtype=bool)
    raise ValueError(f"embodiment shape {arr.shape}; expected ({n},), (1, {n}) or ({num_envs}, {n})")


def split_dict_designs(embodiment: Any, num_envs: int, leaf_ndim: dict[str, int]) -> list:
    """One design (or ``None``) per lane for envs whose design is a dict of arrays.

    Accepts ``None``, a per-lane sequence of dicts (entries may be ``None``), one dict (broadcast to every lane), or a
    stacked dict whose values carry a leading ``num_envs`` axis (a leading axis of 1 broadcasts). ``leaf_ndim`` gives
    each key's native number of dimensions, which tells a single design from a stacked one. A dict that fits neither
    form (unknown or missing keys, mixed shapes) is passed through as one design, so the env's own check rejects it
    rather than this raising. Raises ``ValueError`` only on a batch of the wrong size (a sequence, or a stacked dict)."""
    if embodiment is None:
        return [None] * num_envs
    if isinstance(embodiment, Mapping):
        if set(leaf_ndim) <= set(embodiment) and {np.ndim(embodiment[k]) - leaf_ndim[k] for k in leaf_ndim} == {1}:
            lengths = {np.shape(embodiment[k])[0] for k in leaf_ndim}
            if not (lengths <= {1, num_envs} and len(lengths) == 1):
                raise ValueError(f"stacked embodiment leading sizes {sorted(lengths)}; expected {num_envs} (or 1)")
            pick = (lambda i: i) if lengths == {num_envs} else (lambda i: 0)
            return [{**_copy(embodiment), **{k: np.array(embodiment[k])[pick(i)] for k in leaf_ndim}} for i in range(num_envs)]
        return [_copy(embodiment) for _ in range(num_envs)]
    seq = list(embodiment)
    if len(seq) != num_envs:
        raise ValueError(f"embodiment sequence length {len(seq)} != num_envs {num_envs}")
    return [None if e is None else _copy(e) for e in seq]


def _copy(design: Mapping) -> dict:
    return {k: v.copy() if hasattr(v, "copy") else v for k, v in design.items()}


def lane_valid(rows: np.ndarray, low: np.ndarray, high: np.ndarray, tol: float = 0.0) -> np.ndarray:
    """Per-lane structural validity of a design batch: finite and inside the bounds."""
    low, high = np.asarray(low, np.float32), np.asarray(high, np.float32)
    return np.isfinite(rows).all(1) & (rows >= low - tol).all(1) & (rows <= high + tol).all(1)
