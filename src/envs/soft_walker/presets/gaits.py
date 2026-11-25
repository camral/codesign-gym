"""Analytic chamber gait seeds; these are candidates rather than certified locomotion."""

from __future__ import annotations

import numpy as np
from numpy.typing import DTypeLike

from envs.soft_walker.engine.types import HostArray


def seed_controller(
    dim: int = 2,
    *,
    amplitude: float = 0.8,
    lift: float = 0.5,
    phase: float = 0.0,
    gait: str = "alternating",
    harmonics: int = 1,
    dtype: DTypeLike = "float64",
) -> HostArray:
    """Analytic chamber-length/bending seed, NOT a pre-certified walking gait.

    x-differential actuation bends the leg; common-mode actuation modulates its
    length a quarter-cycle later. In 3D both lateral chambers start identically
    so the seed does not intentionally roll the torso. Optimization can release
    this symmetry. A sign reversal is an experiment, not a camera trick.

    Pronk gives every leg phase zero. With amplitude=0 it is pure synchronous
    common mode, with no chamber differential/bending in the seed. Under the
    active-stress law positive commands contract: the cosine seed starts with
    contraction and then synchronously releases/extends. "Pronk" initializes
    coefficients only; independent Adam parameters can subsequently break it.
    """
    if dim not in (2, 3) or harmonics < 1:
        raise ValueError("Unsupported controller dimension/harmonics")
    phases = (
        ((0.0,) * (2 if dim == 2 else 4))
        if gait == "pronk"
        else (
            (0.0, np.pi)
            if dim == 2
            else (
                (0.0, np.pi, np.pi, 0.0)
                if gait != "crawl"
                else (0.0, np.pi, 1.5 * np.pi, 0.5 * np.pi)
            )
        )
    )
    channels_per_leg = 2 if dim == 2 else 4
    result = np.zeros((len(phases) * channels_per_leg, 1 + 2 * harmonics), dtype=dtype)
    for leg, offset in enumerate(phases):
        phi = offset + phase
        for chamber in range(channels_per_leg):
            sign = -1.0 if chamber < channels_per_leg // 2 else 1.0
            # sign*A*sin(theta+phi) + lift*cos(theta+phi)
            result[leg * channels_per_leg + chamber, 1] = sign * amplitude * np.cos(
                phi
            ) - lift * np.sin(phi)
            result[leg * channels_per_leg + chamber, 2] = sign * amplitude * np.sin(
                phi
            ) + lift * np.cos(phi)
    return result
