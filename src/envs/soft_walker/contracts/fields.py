"""Shared host array contracts and the supported embodiment material envelope.

These are representation/validation bounds, not universal physical limits or
task-specific material choices. Values preserve the original field contract;
their initial calibration was not recorded. FieldSpec may narrow the bounds.
No JAX, Warp, viewer, or simulation module is imported here.
"""

from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

# Mixed float, integer and boolean arrays are intentional at this schema boundary.
HostArray: TypeAlias = NDArray[Any]
HostFields: TypeAlias = dict[str, HostArray]

YOUNG_BOUNDS = (5000.0, 100000.0)  # Pa: supported elastic-modulus envelope.
DENSITY_BOUNDS = (800.0, 1250.0)  # kg/m^3: supported bulk-density envelope.
POISSON_BOUNDS = (0.25, 0.45)  # Dimensionless; excludes the incompressible limit.
POISSON_DEFAULT = 0.35  # Historical midpoint of the supported range.

ACTUATOR_PASSIVE = 0
ACTUATOR_DIRECTIONAL = 1
ACTUATOR_ISOTROPIC = 2
ACTUATOR_TYPE_COUNT = 3
MATERIAL_CATEGORY_COUNT = 3
LABEL_MATERIAL = 0
LABEL_ACTUATOR = 1
LABEL_CHANNEL = 2
LABEL_WIDTH = 3


def resolved_action_mask(
    fields: HostFields, kind: str, channels: int, action_size: int
) -> HostArray:
    """Return padded Boolean channels used by validated resolved actuators.

    Spring and beam routes activate a channel when any signed route is nonzero.
    Continuum actuators also require positive occupancy and strength, and a
    non-passive kind. The caller validates field shapes and channel capacity.
    """
    used: HostArray = np.zeros(channels, bool)
    if kind in ("spring", "beam"):
        used = np.any(np.abs(fields["routing"]) > 0, axis=0)
    else:
        active = (fields["alpha"] > 0) & (fields["strength"] > 0) & (fields["kind"] != 0)
        used[np.asarray(fields["channel"][active], int)] = True
    return np.pad(used, (0, action_size - channels))


@dataclass(frozen=True)
class RawFieldLayout:
    """Raw decoder score positions, including the appended spatial Poisson head.

    Primitive feature arrays omit the occupancy score; all other decoders use
    this complete row. Layouts are computed properties, never descriptor fields.
    """

    dim: int
    channels: int

    occupancy = 0
    young = 1
    density = 2
    strength = 3

    @property
    def direction(self) -> slice:
        """Direction coordinates follow the four scalar scores."""
        return slice(4, 4 + self.dim)

    @property
    def material_logits(self) -> slice:
        """Three material-category scores after the direction coordinates."""
        return slice(self.direction.stop, self.direction.stop + MATERIAL_CATEGORY_COUNT)

    @property
    def actuator_logits(self) -> slice:
        """Passive, directional and isotropic category scores."""
        return slice(self.material_logits.stop, self.material_logits.stop + ACTUATOR_TYPE_COUNT)

    @property
    def channel_logits(self) -> slice:
        """One route score for every configured channel."""
        return slice(self.actuator_logits.stop, self.actuator_logits.stop + self.channels)

    @property
    def poisson(self) -> int:
        """Appended spatial Poisson score; also the exact legacy row width."""
        return self.channel_logits.stop

    @property
    def width(self) -> int:
        """Complete current raw-score row width."""
        return self.poisson + 1

    @property
    def properties(self) -> slice:
        """Primitive properties omit only the first, geometry-generated score."""
        return slice(self.young, self.width)

    @property
    def primitive_poisson(self) -> int:
        """Poisson position relative to a primitive's property row."""
        return self.poisson - self.properties.start
