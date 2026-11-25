"""Private device copies of the accepted, settled initial state for autoreset."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import warp as wp

if TYPE_CHECKING:
    from .host_types import MPMState, SolidState


@dataclass(frozen=True)
class ResetState:
    """Own independent state buffers; replay never aliases or updates these copies.

    Fields include deformation/rotation, velocity, actuator history, the complete
    work ledger and status. Copies stay on the original state's device and retain
    its dtype and shape. Model parameters remain owned by the accepting lane.
    """

    buffers: tuple[tuple[str, wp.array], ...]

    @classmethod
    def capture(cls, state: MPMState | SolidState, names: tuple[str, ...]) -> ResetState:
        """Copy all physical state fields after a successful explicit reset."""
        return cls(
            tuple((name, wp.clone(getattr(state, name), requires_grad=False)) for name in names)
        )

    def restore(self, state: MPMState | SolidState) -> None:
        """Restore buffers in place, preserving compiled graphs and model ownership."""
        for name, saved in self.buffers:
            wp.copy(getattr(state, name), saved)
