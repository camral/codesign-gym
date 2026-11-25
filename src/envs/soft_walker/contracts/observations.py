"""Flat observation contracts shared by packers and host capacity checks.

All block orders and scales are inherited public formats. These scales are
observation normalizations, independent of decoder seeds and material bounds.
This module imports no numerical or accelerator runtime.
"""

from dataclasses import dataclass

from .metrics import LEDGER_WIDTH

OBSERVATION_YOUNG_SCALE_PA = 20000.0
OBSERVATION_DENSITY_SCALE_KG_M3 = 1000.0
TERRAIN_PROBE_FORWARD_OFFSETS = (0.0, 0.5, 1.0)  # Multiples of configured length_ref.
TERRAIN_PROBE_COUNT = len(TERRAIN_PROBE_FORWARD_OFFSETS)
ACTUATOR_OBSERVATION_WIDTH = 2  # Filtered command and active-channel mask.
SPRING_GLOBAL_WIDTH = 5  # 2D COM, 2D velocity, normalized time.
SPRING_NODE_WIDTH = 5  # 2D displacement, 2D velocity, active flag.
SPRING_FIELD_WIDTH = 4  # Occupancy, log Young modulus, density, strength.
BODY_FIELD_SCALAR_WIDTH = 5  # Occupancy, log Young modulus, density, strength, Poisson.
BEAM_ROTATION_WIDTH = 9
BEAM_MECHANICS_WIDTH = 12  # Row-major rotation deviation followed by 3D angular velocity.


@dataclass(frozen=True)
class ObservationLayout:
    """Ordered flattened blocks; widths include inactive entries before padding."""

    blocks: tuple[tuple[str, int], ...]

    @property
    def width(self) -> int:
        """Return the unpadded observation width."""
        return sum(width for _, width in self.blocks)

    def block(self, name: str) -> slice:
        """Return a named block without reordering any entries."""
        start = 0
        for candidate, width in self.blocks:
            if candidate == name:
                return slice(start, start + width)
            start += width
        raise KeyError(name)


def spring_observation_layout(
    nodes: int, fields: int, channels: int, *, terrain: bool
) -> ObservationLayout:
    """Historical Spring2D policy observation, including optional terrain probes."""
    blocks: tuple[tuple[str, int], ...] = (
        ("global", SPRING_GLOBAL_WIDTH),
        ("nodes", nodes * SPRING_NODE_WIDTH),
        ("fields", fields * SPRING_FIELD_WIDTH),
        ("actuators", channels * ACTUATOR_OBSERVATION_WIDTH),
    )
    if terrain:
        blocks += (("terrain", TERRAIN_PROBE_COUNT * 3),)
    return ObservationLayout(blocks)


def body_observation_layout(
    kind: str, dim: int, nodes: int, fields: int, channels: int, *, profile: str
) -> ObservationLayout:
    """General body-driver policy observation; full MPM includes F and C blocks."""
    blocks: tuple[tuple[str, int], ...] = (
        ("global", 3 * dim + 1),
        ("nodes", nodes * (3 * dim + 1)),
        ("fields", fields * (BODY_FIELD_SCALAR_WIDTH + dim + 3 + channels)),
    )
    if kind == "beam":
        blocks += (("mechanics", nodes * BEAM_MECHANICS_WIDTH),)
    if kind.startswith("mpm") and profile == "full":
        blocks += (("mechanics", nodes * 2 * dim * dim),)
    blocks += (
        ("actuators", channels * ACTUATOR_OBSERVATION_WIDTH),
        ("terrain", TERRAIN_PROBE_COUNT * (dim + 1)),
    )
    return ObservationLayout(blocks)


def resolved_observation_layout(
    kind: str, dim: int, sites: int, channels: int, action_size: int
) -> ObservationLayout:
    """Physical resolved-state observation shared by registered JAX and Warp adapters."""
    blocks: tuple[tuple[str, int], ...] = (("displacement", sites * dim), ("velocity", sites * dim))
    if kind.startswith("mpm"):
        blocks += (("deformation", sites * dim * dim), ("affine_velocity", sites * dim * dim))
    else:
        blocks += (("rotation", sites * dim * dim), ("angular_velocity", sites * dim))
    return ObservationLayout(
        blocks + (("actuators", channels), ("ledger", LEDGER_WIDTH), ("action_mask", action_size))
    )
