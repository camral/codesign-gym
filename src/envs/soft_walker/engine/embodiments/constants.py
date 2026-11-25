"""Numerical safeguards and inherited priors for semantic field decoders.

Scores and material coordinates are dimensionless. These choices preserve the
original implementation; no empirical calibration record exists. The limits
stabilize normalization or bound the search space, not physical simulation.
Different squared-norm and host-norm tolerances remain deliberately separate.
"""

RAW_SCORE_LIMIT = 6.0  # Coefficient Box endpoint; also fixes Poisson endpoint mapping.
MAX_CHANNELS = 16  # Historical bounded actuator-routing capacity.
MAX_CPPN_HIDDEN = 32  # Fixed-size acyclic neural-field capacity.
MAX_PRIMITIVES = 32  # Fixed-size primitive union capacity.
DEFAULT_OCCUPANCY_FLOOR = 0.02  # Relaxed geometry retains a small positive weight.
DEFAULT_MATERIAL_PALETTE = (10000.0, 30000.0, 80000.0)  # Pa, three inherited choices.
LABEL_LOGIT = 4  # Integer labels become decisive semantic logits; integer value retained.
CPPN_NORM2_FLOOR = 1e-24  # Keeps the radial-input derivative finite at the origin.
PRIMITIVE_NORM2_FLOOR = 1e-24  # Smooths distance/axis square roots at zero norm.
PRIMITIVE_AXIS_MIN_NORM = 1e-6  # Host structural rejection, in normalized coordinates.
PRIMITIVE_AXIS_MIN_NORM2 = 1e-12  # Compiled squared-axis rejection; same inherited domain.
DISABLED_PRIMITIVE_DISTANCE = 1e3  # Keeps disabled primitives outside the bounded envelope.
PRIMITIVE_HALF_SIZE_BOUNDS = (0.08, 1.5)
PRIMITIVE_RADIUS_BOUNDS = (0.02, 1.0)
PRIMITIVE_HALF_LENGTH_BOUNDS = (0.01, 1.5)
PRIMITIVE_ROUNDING_BOUNDS = (0.0, 0.08)
PRIMITIVE_BASE_HALF_SIZE = 0.9
PRIMITIVE_BASE_RADIUS = 0.65
PRIMITIVE_BASE_HALF_LENGTH = 0.5
PRIMITIVE_BASE_ROUNDING = 0.04
DEFAULT_UNION_BLEND_WIDTH = 0.15  # Normalized-material distance in the soft minimum.
DEFAULT_GEOMETRY_WIDTH = 0.15  # Normalized-material distance per occupancy logit.
DIRECTION_NORM2_FLOOR = 1e-12  # Below this, field interpretation uses a fixed axis.
FIELD_BOUND_RTOL = 1e-6  # Host validation allowance for material arithmetic roundoff.
FIELD_DIRECTION_ATOL = 1e-5  # Host tolerance for a unit actuator direction.
BASE_YOUNG = 20000.0  # Pa, decoder seed stiffness; independent of task preset choices.
BASE_OCCUPANCY_SCORE = 2.0  # Positive score makes the baseline occupied in hard mode.
BASE_CATEGORY_SCORE = 2.0  # Selects middle palette/directional actuator seed categories.
INVERSE_MATERIAL_CLIP = 0.99  # Keeps the inverse tanh away from infinite endpoints.
CPPN_BASE_COORDINATE_WEIGHT = 0.2
CPPN_BASE_RADIUS_WEIGHT = -0.1
CPPN_BASE_HIDDEN_WEIGHT = 0.05
