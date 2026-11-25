"""Host-only initialization profiles shared by native builders and Gym presets.

These values preserve the original ``uniform_body`` / ``regular_solid`` examples
and the registered wrapper defaults that intentionally match them. They are not
universal constitutive constants or measured properties of a named elastomer.
The original tuning/calibration record is unavailable; the implementation plan
describes softened benchmark materials. Decoder seeds remain separate profiles.

Geometry and contact conventions differ between spring, beam, FEM and MPM.
Sharing an initialization value here does not make their discretizations or
force units interchangeable. This module imports no accelerator or wrapper.
"""

# Uniform reference material and actuation profiles. Poisson's ratio is owned by
# contracts.fields.POISSON_DEFAULT rather than copied into this profile.
BODY_DENSITY_KG_M3 = 1000.0
UNIFORM_MPM_YOUNG_PA = 12000.0
UNIFORM_MPM_STRENGTH = 0.95  # Dimensionless multiplier of the actuator command.
REGULAR_SOLID_YOUNG_PA = 20000.0
REGULAR_SOLID_STRENGTH = 0.8  # Dimensionless multiplier of the actuator command.
PLANAR_THICKNESS_M = 0.01  # Physical out-of-plane thickness for 2D volume/area.

# The spring reference builder follows SpringConfig's original damping/contact
# convention; these defaults do not override caller-supplied configurations.
SPRING_DAMPING_RATIO = 0.08  # Dimensionless critical-damping fraction per edge.
SPRING_CONTACT_MODULUS_PA = 200000.0
SPRING_CONTACT_REFERENCE_M = 0.2
# Beam/FEM/MPM retain geometric area (m^2), so this scale is Pa per metre.
# Spring instead folds modulus/reference length into its nodal contact weights.
SOLID_CONTACT_PRESSURE_PER_LENGTH_PA_M = 1e6
EARTH_GRAVITY_M_S2 = 9.81  # Conventional near-surface Earth gravity magnitude.
