"""Embodiment space for HalfCheetah co-design.

The design is a normalised offset vector (one entry per offsettable slot; see
``offsettable_logic``), ``base_embodiment = zeros``. Validation is structural
only (shape / finite / in-bounds), per the codesign-gym rule.
"""

from __future__ import annotations

import numpy as np
import torch
from gymnasium import spaces


def _tensor(value, device):
    if isinstance(value, (list, tuple)) and value and isinstance(value[0], torch.Tensor):
        return torch.stack([item.to(device=device, dtype=torch.float32) for item in value])
    if isinstance(value, (list, tuple)):
        value = np.asarray(value)
    return torch.as_tensor(value, device=device, dtype=torch.float32)


class EmbodimentSpace(spaces.Box):
    """Normalised offsets, accepting one design or one row per native world."""

    def __init__(self, low, high, num_envs):
        super().__init__(low=low, high=high, dtype=np.float32)
        self.num_envs = num_envs

    def contains(self, value):
        try:
            value = _tensor(value, "cpu").numpy()
        except (TypeError, ValueError, RuntimeError):
            return False
        return value.shape in (self.shape, (1, *self.shape), (self.num_envs, *self.shape)) and bool(
            np.isfinite(value).all() and (value >= self.low).all() and (value <= self.high).all())
