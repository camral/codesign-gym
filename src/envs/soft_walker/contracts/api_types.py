"""NumPy-facing contracts for registered SoftWalker vector environments.

These types do not import either simulator. Embodiments are per-cell C-order
vectors; observations and actions have a leading environment dimension.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias, TypedDict

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .fields import HostArray as HostArray, HostFields

Fields: TypeAlias = HostFields  # Historical wrapper name for the shared host-field mapping.

EmbodimentInput: TypeAlias = ArrayLike | Sequence[ArrayLike | None] | None
SeedInput: TypeAlias = int | Sequence[int | None] | None
StackedInfo: TypeAlias = dict[str, object]
ResetResult: TypeAlias = tuple[HostArray, HostArray, StackedInfo]
StepResult: TypeAlias = tuple[HostArray, HostArray, HostArray, HostArray, StackedInfo]
RGBFrame: TypeAlias = NDArray[np.uint8]


class EnvironmentTiming(TypedDict):
    """Native constructor settings: counts, metre/work scales and energy weight."""

    substeps: int
    settling_steps: int
    horizon: int
    length_ref: float
    work_ref: float
    energy_weight: float
