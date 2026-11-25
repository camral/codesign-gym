"""Real Gymnasium spaces in production; an explicit offline test-only adapter.

No fallback is silent. It is enabled only by CODESIGN_OFFLINE_SPACES=1, emits a
warning, and is recorded by the evidence runner. It is not Gymnasium validation.
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Mapping
from typing import TYPE_CHECKING

from numpy.typing import ArrayLike, DTypeLike

from .types import HostArray, HostEmbedding

try:
    from gymnasium import Env
    from gymnasium.spaces import Box, Dict, Space
    from gymnasium.vector import AutoresetMode, VectorEnv

    AUTORESET_NEXT_STEP = AutoresetMode.NEXT_STEP
    from gymnasium.error import ResetNeeded

    USING_GYMNASIUM = True
except ModuleNotFoundError as exc:
    if exc.name != "gymnasium" or os.environ.get("CODESIGN_OFFLINE_SPACES") != "1":
        raise ImportError(
            "Install gymnasium>=1.2.3. The explicit CODESIGN_OFFLINE_SPACES=1 "
            "adapter is for offline numerical development only, not Gym integration."
        ) from exc
    warnings.warn(
        "Using explicit OFFLINE test spaces; Gymnasium integration is unvalidated.", RuntimeWarning
    )
    import numpy as np

    USING_GYMNASIUM = False
    AUTORESET_NEXT_STEP = "NextStep"

    # Explicit offline replacement for the optional Gym import.
    class ResetNeeded(RuntimeError):  # type: ignore[no-redef]
        """Signal that an offline environment needs reset before another transition."""

        pass

    # Explicit offline replacement for the optional Gym import.
    class Env:  # type: ignore[no-redef]
        """Minimal inheritance marker for explicitly enabled offline scalar tests."""

        pass

    # Explicit offline replacement for the optional Gym import.
    class VectorEnv:  # type: ignore[no-redef]
        """Minimal inheritance marker for explicitly enabled offline vector tests."""

        pass

    # Explicit offline replacement for the optional Gym import.
    class Space:  # type: ignore[no-redef]
        """Offline containment interface; concrete spaces provide sampling and bounds."""

        if TYPE_CHECKING:

            def contains(self, value: object) -> bool:
                """Report whether a value satisfies the concrete space's shape and bounds."""
                ...

        def __contains__(self, value: object) -> bool:
            """Delegate containment checks to the concrete offline space."""
            return self.contains(value)

    # Explicit offline replacement for the optional Gym import.
    class Box(Space):  # type: ignore[no-redef]
        """Fixed-shape numeric bounds with a private NumPy sampling generator."""

        def __init__(
            self,
            low: ArrayLike,
            high: ArrayLike,
            shape: tuple[int, ...] | None = None,
            dtype: DTypeLike = np.float32,
            seed: int | None = None,
        ) -> None:
            """Broadcast numeric bounds to a fixed shape for explicitly enabled offline
            diagnostics.
            """
            self.dtype = np.dtype(dtype)
            self.shape = (
                tuple(shape)
                if shape is not None
                else np.broadcast_shapes(np.shape(low), np.shape(high))
            )
            self.low = np.broadcast_to(np.asarray(low, dtype=self.dtype), self.shape).copy()
            self.high = np.broadcast_to(np.asarray(high, dtype=self.dtype), self.shape).copy()
            self.seed(seed)

        def seed(self, seed: int | None = None) -> int | None:
            """Reset only this space sampling generator."""
            self.np_random = np.random.default_rng(seed)
            return seed

        def contains(self, x: object) -> bool:
            """Check shape, dtype, and bounds without coercing the declared space."""
            array = np.asarray(x)
            return bool(
                array.shape == self.shape
                and np.can_cast(array.dtype, self.dtype)
                and np.all(array >= self.low)
                and np.all(array <= self.high)
            )

        def sample(self) -> HostArray:
            """Draw a host sample from the declared offline space."""
            finite_lo, finite_hi = np.isfinite(self.low), np.isfinite(self.high)
            x = self.np_random.normal(size=self.shape)
            both = finite_lo & finite_hi
            if np.issubdtype(self.dtype, np.integer):
                return self.np_random.integers(
                    self.low, self.high + 1, size=self.shape, dtype=self.dtype
                )
            u = self.np_random.random(self.shape)
            x = np.where(
                both,
                np.where(finite_lo, self.low, 0)
                + u * (np.where(finite_hi, self.high, 1) - np.where(finite_lo, self.low, 0)),
                x,
            )
            x = np.where(
                finite_lo & ~finite_hi, self.low + self.np_random.exponential(size=self.shape), x
            )
            x = np.where(
                ~finite_lo & finite_hi, self.high - self.np_random.exponential(size=self.shape), x
            )
            return np.asarray(x, dtype=self.dtype)

    # Explicit offline replacement for the optional Gym import.
    class Dict(Space):  # type: ignore[no-redef]
        """Sorted named offline spaces with exact-key containment and recursive sampling."""

        def __init__(self, spaces: Mapping[str, Space]) -> None:
            """Copy the name-to-space mapping in sorted order while retaining its leaf spaces."""
            self.spaces = dict(sorted(spaces.items()))
            self.shape = None

        def __getitem__(self, key: str) -> Space:
            """Read a named leaf from the offline dictionary space."""
            return self.spaces[key]

        def contains(self, x: object) -> bool:
            """Require exactly the declared keys and containment in every leaf space."""
            return (
                isinstance(x, dict)
                and x.keys() == self.spaces.keys()
                and all(s.contains(x[k]) for k, s in self.spaces.items())
            )

        def seed(self, seed: int | None = None) -> int | None:
            """Spawn an independent deterministic sampling seed for each leaf space."""
            seeds = np.random.SeedSequence(seed).spawn(len(self.spaces))
            for s, ss in zip(self.spaces.values(), seeds):
                s.seed(int(ss.generate_state(1)[0]))
            return seed

        def sample(self) -> HostEmbedding:
            """Draw a host sample from the declared offline space."""
            return {k: s.sample() for k, s in self.spaces.items()}
