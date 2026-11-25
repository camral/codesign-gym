"""Stable packed metric, energy-ledger and native solver-counter contracts.

Names describe existing positions; they do not change archive keys, widths or
the meaning of status values. Plain integer constants can be captured by Warp
kernel factories without importing Warp into host recording tools.
"""

from dataclasses import dataclass

METRIC_TIME = 0
METRIC_DX = 1
METRIC_DY = 2
METRIC_DZ = 3
METRIC_WORK_POSITIVE = 4
METRIC_WORK_SIGNED = 5
METRIC_STATUS = 6

# Physical state ledger; the first three entries are also the work-increment array.
LEDGER_WORK_SIGNED = 0
LEDGER_WORK_POSITIVE = 1
LEDGER_DISSIPATED = 2
LEDGER_TIME = 3
LEDGER_WIDTH = 4
WORK_INCREMENT_WIDTH = 3
LEDGER_COLUMNS = ("work_signed_j", "work_positive_j", "dissipated_j", "time_s")

# Newton counts are separate from solver status values and work-operation counters.
COUNT_NEWTON_ITERATIONS = 0
COUNT_BACKTRACKS = 1
COUNT_LINEAR_SOLVES = 2
COUNT_LINE_SEARCH_TRIALS = 3
NEWTON_COUNT_WIDTH = 4

DIAGNOSTIC_STATUS = 0
DIAGNOSTIC_NEWTON_ITERATIONS = 1
DIAGNOSTIC_BACKTRACKS = 2
DIAGNOSTIC_RESIDUAL = 3
DIAGNOSTIC_LINEAR_RESIDUAL = 4
DIAGNOSTIC_LIVE_NODES = 5
NATIVE_DIAGNOSTIC_COLUMNS = (
    "code",
    "newton_iterations",
    "backtracks",
    "root_residual",
    "linear_relative_residual",
    "live_nodes",
)
NATIVE_DIAGNOSTIC_WIDTH = len(NATIVE_DIAGNOSTIC_COLUMNS)

WORK_FORWARD_TANGENT = 0
WORK_TRANSPOSE_TANGENT = 1
WORK_LINEARIZATION_BUILD = 2
WORK_PRIMAL_RESIDUAL = 3
WORK_SUPPORT_BUILD = 4
WORK_REPLAY_ROOT_CHECK = 5
WORK_REPLAY_ROOT_FALLBACK = 6
WORK_COUNT_WIDTH = 7
MPM_WORK_COUNT_COLUMNS = (
    "forward_tangent",
    "transpose_tangent",
    "linearization_build",
    "primal_residual",
    "support_build",
    "replay_root_check",
    "replay_root_fallback",
)
SOLID_WORK_COUNT_COLUMNS = (
    "forward_tangent",
    "transpose_tangent",
    "linearization_build",
    "unused3",
    "unused4",
    "replay_root_check",
    "replay_root_fallback",
)

LOCOMOTION_SOLVER_COLUMNS = (
    "code",
    "root_residual",
    "linear_relative_residual",
    "min_J_or_area_ratio",
    "support_valid",
    "max_penetration_m",
    "newton_iterations",
)
FLIGHT_COLUMNS = (
    "time_s",
    "com_height_m",
    "com_vy_m_s",
    "normal_contact_force_n",
    "weight_n",
    "normal_force_over_weight",
    "all_contact_surface_gap_m",
    "torso_contact_surface_gap_m",
    "torso_angle_rad",
    "torso_upright_cosine",
)


class FlightMetric:
    """Positions in the persisted flight row; names/units match FLIGHT_COLUMNS."""

    time = 0
    com_height = 1
    com_vy = 2
    normal_force = 3
    weight = 4
    normal_force_over_weight = 5
    all_surface_gap = 6
    torso_surface_gap = 7
    torso_angle = 8
    torso_upright = 9
    width = len(FLIGHT_COLUMNS)


class LocomotionSolver:
    """Positions in the persisted locomotion solver row, not native work counters."""

    code = 0
    root_residual = 1
    linear_relative_residual = 2
    min_j_or_area_ratio = 3
    support_valid = 4
    max_penetration = 5
    newton_iterations = 6
    width = len(LOCOMOTION_SOLVER_COLUMNS)
    # Residuals depend on precision; these categorical/geometry counters must
    # agree in repeated evaluations of an identical execution configuration.
    repeatability_columns = (
        code,
        min_j_or_area_ratio,
        support_valid,
        max_penetration,
        newton_iterations,
    )


class ImplicitForwardMetric:
    """Nine-column implicit experiment record: whole-body COM, without leg tags.

    The displacement/work/status prefix matches locomotion, but column seven is
    whole-body COM height, not torso height or a scenario terrain gap.
    """

    time = METRIC_TIME
    dx = METRIC_DX
    dy = METRIC_DY
    dz = METRIC_DZ
    work_positive = METRIC_WORK_POSITIVE
    work_signed = METRIC_WORK_SIGNED
    status = METRIC_STATUS
    com_height = 7
    com_speed = 8
    width = 9


@dataclass(frozen=True)
class _BodyMetricLayout:
    """Shared packed prefix and foot blocks; column seven has workflow-specific meaning."""

    dim: int

    time = METRIC_TIME
    dx = METRIC_DX
    dy = METRIC_DY
    dz = METRIC_DZ
    work_positive = METRIC_WORK_POSITIVE
    work_signed = METRIC_WORK_SIGNED
    status = METRIC_STATUS
    com_speed = 8

    @property
    def legs(self) -> int:
        """Return the existing planar/spatial foot capacity."""
        return 2 if self.dim == 2 else 4

    @property
    def foot_gaps(self) -> slice:
        """Per-foot surface gaps, in the existing anatomical order."""
        return slice(9, 9 + self.legs)

    @property
    def foot_vx(self) -> slice:
        """Per-foot horizontal velocities after the gap block."""
        return slice(9 + self.legs, 9 + 2 * self.legs)


@dataclass(frozen=True)
class LocomotionMetricLayout(_BodyMetricLayout):
    """Legged experiment rows store absolute torso height and per-foot clearances."""

    torso_height = 7

    @property
    def width(self) -> int:
        """Complete locomotion rows end after the foot velocity block."""
        return self.foot_vx.stop


@dataclass(frozen=True)
class ScenarioMetricLayout(_BodyMetricLayout):
    """Scenario rows use a torso-centre gap and append force/surface diagnostics."""

    torso_center_gap = 7

    @property
    def normal_force_over_weight(self) -> int:
        """First scenario-specific column after the foot blocks."""
        return 9 + 2 * self.legs

    @property
    def all_surface_gap(self) -> int:
        """Minimum gap over the complete contact surface."""
        return self.normal_force_over_weight + 1

    @property
    def torso_surface_gap(self) -> int:
        """Minimum gap over the torso contact surface."""
        return self.normal_force_over_weight + 2

    @property
    def com_vy(self) -> int:
        """Vertical center-of-mass velocity."""
        return self.normal_force_over_weight + 3

    @property
    def contact_energy(self) -> int:
        """Stored contact energy in joules."""
        return self.normal_force_over_weight + 4

    @property
    def width(self) -> int:
        """Number of columns in a complete scenario metric row."""
        return self.contact_energy + 1

    def columns(self) -> list[str]:
        """Return the historical scenario archive labels in their unchanged order."""
        return (
            [
                "time_s",
                "dx_m",
                "dy_m",
                "dz_m",
                "work_positive_j",
                "work_signed_j",
                "status",
                "torso_center_gap_m",
                "com_speed_m_s",
            ]
            + [f"foot_{i}_gap_m" for i in range(self.legs)]
            + [f"foot_{i}_vx_m_s" for i in range(self.legs)]
            + [
                "normal_force_over_weight",
                "all_surface_gap_m",
                "torso_surface_gap_m",
                "com_vy_m_s",
                "contact_energy_j",
            ]
        )


def scenario_metric_layout(dim: int) -> ScenarioMetricLayout:
    """Describe the existing scenario record; no accelerator arrays are created."""
    return ScenarioMetricLayout(dim)


def locomotion_metric_columns(dim: int) -> list[str]:
    """Return locomotion labels, retaining their distinct displacement/height names."""
    legs = 2 if dim == 2 else 4
    return (
        [
            "time_s",
            "distance_x_m",
            "distance_y_m",
            "distance_z_m",
            "work_positive_j",
            "work_signed_j",
            "status",
            "torso_height_m",
            "com_speed_m_s",
        ]
        + [f"foot_{k}_clearance_m" for k in range(legs)]
        + [f"foot_{k}_vx_m_s" for k in range(legs)]
    )


def flight_metric_columns() -> list[str]:
    """Return a fresh list of the established flight-assessment labels."""
    return list(FLIGHT_COLUMNS)
