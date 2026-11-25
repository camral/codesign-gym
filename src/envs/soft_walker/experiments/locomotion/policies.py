"""Inherited legged-experiment policies, separate from physics and solver tolerances.

These values preserve previous qualification and loss behavior. Their original
empirical calibration was not recorded; they are not universal walking/flight
criteria. Named guards retain distinct physical units and are not interchangeable.
"""

from dataclasses import dataclass

MASS_DENOMINATOR_FLOOR = 1e-20  # kg; finite masked means for empty/vanishing groups.
WEIGHT_DENOMINATOR_FLOOR = 1e-20  # N; finite force/weight ratio at zero gravity or mass.
# Orientation direction is sum(mass * reference_axis_offset * deformed_offset),
# in kg m^2. Both its full norm and the lateral norm before atan2 square this
# same vector, so the same kg^2 m^4 guard applies before either square root.
COVARIANCE_NORM2_FLOOR = 1e-30
TRAVEL_HEIGHT_SMOOTHING_FRACTION = 0.05  # Body-height fraction per travel softplus logit.
LOW_BOUNCE_GAP_FRACTION = 0.02  # Body-height fraction per torso-contact softplus logit.
SCORING_WINDOW_ATOL = 1e-10  # s; permits multiplication/division roundoff in fixed windows.
CONTROL_INTERVAL_ATOL = 1e-12  # s; environment/protocol clock equality, separate from windows.
ROLLOUT_TIME_RTOL = 5e-4  # Dimensionless allowance for accumulated control-step arithmetic.
ROLLOUT_TIME_ATOL = 1e-6  # s; original complete-rollout time acceptance floor.
LEGACY_WORK_REFERENCES = {2: 0.4, 3: 4.0}  # J; only fallback for historical rows without config.
LOW_BOUNCE_CHECKPOINT_BLOCK = 3  # Control intervals per rematerialization block; memory policy.


@dataclass(frozen=True)
class FlightQualification:
    """Control-boundary contact evidence required by the existing airborne report."""

    force_fraction: float = 0.01  # Normal force divided by actual body weight.
    surface_clearance: float = 0.003  # m; all exposed contact sites must clear this gap.
    minimum_duration: float = 0.02  # s sampled between first/last qualifying boundaries.
    duration_atol: float = 1e-10  # s; duration arithmetic roundoff, not physical tolerance.
    completed_episodes: int = 2  # Each qualifying span must be followed by observed contact.
    startup_cycles: int = 1  # Excluded only if the replay contains more than one cycle.


@dataclass(frozen=True)
class WalkingQualification:
    """Conservative fixed-replay walking gate; movement alone is insufficient."""

    minimum_swing_clearance: float = 0.001  # m.
    swing_cell_fraction: float = 0.06  # Reference-cell-size fraction; use the larger floor.
    stance_threshold_divisor: int = 2  # Stance must be below half the swing threshold.
    torso_height_fraction: float = 0.35  # Absolute torso height divided by body height.
    minimum_cycles: int = 5  # Complete post-startup cycles.
    minimum_progress: float = 0.25  # Body lengths accumulated after startup.
    positive_cycles: int = 4  # Minimum post-startup cycles with strictly positive progress.
    startup_cycles: int = 1


FLIGHT_QUALIFICATION = FlightQualification()
WALKING_QUALIFICATION = WalkingQualification()
