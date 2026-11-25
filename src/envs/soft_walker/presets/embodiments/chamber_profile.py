"""Inherited chamber-body preset, independent of any array or physics backend.

These values describe the existing two/four-leg example, not general decoder
limits. Geometry is normalized to [-1, 1]; material values use SI units. The
original seed calibration was not recorded, so these are preserved priors,
not claimed optima. Keep this module host-only for scenario planning.
"""

from dataclasses import dataclass

CHAMBER_CHANNELS = {2: 4, 3: 16}  # Two chambers/leg in 2D; four/leg in 3D.
SHAPE_GRID = (17, 13, 13)  # Fixed coefficient resolution, sliced by dimension.
POISSON_GRID_SIDE = 3  # Trilinear/bilinear spatial material-control lattice.
MATERIAL_REGION_COUNT = 2  # Embedding order: legs, torso.
LEG_MATERIAL_INDEX = 0
TORSO_MATERIAL_INDEX = 1
BASE_MATERIAL_IDS = (2, 2)  # Both regions start at the stiffest palette entry.
BASE_LEG_YOUNG = 80000.0  # Pa; task prior, independent of decoder range limits.
BASE_TORSO_YOUNG = 90000.0  # Pa; slightly stiffer passive torso.
BASE_DENSITY = 900.0  # kg/m^3.
BASE_STRENGTH = 0.95  # Dimensionless actuator strength in the supported [0, 1] range.
PRIMITIVE_HALF_SIZE_BOUNDS = (0.03, 1.5)  # Normalized lengths; allows narrow chambers.
OCCUPANCY_SCORE_WIDTH = 0.04  # Normalized box-score units per relaxed occupancy logit.


@dataclass(frozen=True)
class ChamberGeometry:
    """Box envelope and fixed routing partitions of the inherited legged seed.

    Box coordinates use [-1, 1]. Routing coordinates use q=(x+1)/2, in [0, 1].
    Routing values remain explicit to retain the original arithmetic and exact
    equality-side choices at chamber boundaries. Torso is the upper third of
    the envelope; each leg is half of its x-side's width. 3D legs occupy corners.
    """

    torso_center_y: float = 2.0 / 3.0
    torso_half_y: float = 1.0 / 3.0
    leg_center_x: float = 0.75
    leg_center_y: float = -1.0 / 3.0
    leg_center_z: float = 2.0 / 3.0
    leg_half_x: float = 0.25
    leg_half_y: float = 2.0 / 3.0
    leg_half_z: float = 1.0 / 3.0
    torso_start_qy: float = 2.0 / 3.0
    side_split_q: float = 0.5
    leg_stride_qx: float = 0.75
    chamber_split_qx: float = 0.125
    leg_stride_qz: float = 2.0 / 3.0
    chamber_split_qz: float = 1.0 / 6.0


GEOMETRY = ChamberGeometry()


@dataclass(frozen=True)
class ChamberCPPNSeed:
    """Dimensionless sigmoid gates approximating the chamber box union.

    Gate gain and biases are inherited hand-authored seed coefficients. Their
    precise tuning history is unavailable. The output is linear; hidden nodes
    use sigmoids. Values must not be tied to coefficient bounds merely because
    both currently use six as their magnitude.
    """

    gate_gain: float = 6.0
    x_edge_bias: float = -3.0
    torso_bias: float = -2.0
    x_union_bias: float = -3.03
    torso_sharpen_bias: float = -3.0
    z_edge_bias: float = -2.0
    z_union_bias: float = -3.12
    corner_gain: float = 4.0
    corner_bias: float = -5.65
    output_gain: float = 1.0
    output_bias: float = -0.62


CPPN_SEED = ChamberCPPNSeed()
CPPN_HIDDEN_NODES = {2: 5, 3: 8}  # Directional gates, unions and 3D intersection.
