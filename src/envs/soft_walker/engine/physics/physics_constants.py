"""Fixed physical-model guards shared by ordinary and cached force expressions.

These values are not solver acceptance tolerances. Keeping a guard in one
place prevents an optimized constitutive path from silently changing its law.
The module is host-only and imports no accelerator runtime.

SO(3) and beam guards are inherited implementation policies, not newly
calibrated thresholds. Beam guards restrict the local validity domain of the
corotational model; they are not universal physical strain or rotation limits.
"""

SPRING_SQUARED_LENGTH_FLOOR = 1e-24  # m²; guard a collapsed edge before rejection

# Switch SO(3) series on squared angle or squared sine norm near the identity.
SO3_SERIES_SQUARED_THRESHOLD = 1e-6
# JAX evaluates both where branches, so its denominator needs a separate floor.
# Native Warp branches do not require this inactive-branch guard.
SO3_DENOMINATOR_SQUARED_FLOOR = 1e-12

# Beam admissibility: dimensionless link-length ratio and angles in radians.
BEAM_MIN_LENGTH_RATIO = 0.2
BEAM_MAX_RELATIVE_ROTATION = 1.0
BEAM_MAX_STEP_ROTATION = 0.5
