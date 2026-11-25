"""Configuration and geometry constants for NeroGrasp and NeroTools environments."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

BBoxLimits = tuple[tuple[float, float, float], tuple[float, float, float]]

# ---------------------------------------------------------------------------
# LEAP-hand fingertip geometry (NeroGrasp)
# ---------------------------------------------------------------------------
SHAPE_MAX_BBOX_LIMITS_MM: BBoxLimits = ((-8.2, -12.0, -29.0), (8.2, 12.0, 29.0))
SHAPE_MIN_BBOX_LIMITS_MM: BBoxLimits = ((-3.0, -5.0, -3.0), (3.0, 5.0, 3.0))
SNAP_PLANE_CUT_MARGIN_MM = 1.0
SNAP_PLANE_TARGET_XY = (1.0, 1.0)
GEOMETRY_EPSILON = 1e-7
MOUNTING_VERTEX_COUNT = 25

FINGERTIP_BODIES = {"leap_if_tip": "leap_if_tip_trimesh", "leap_th_tip": "leap_th_tip_trimesh"}
FINGERTIP_KEYS = {"leap_if_tip": "index", "leap_th_tip": "thumb"}
FINGERTIP_MESH_ASSETS = ("leap_if_tip_trimesh", "leap_mf_tip_trimesh", "leap_rf_tip_trimesh", "leap_th_tip_trimesh")

# ---------------------------------------------------------------------------
# Parallel-gripper fingertip geometry (NeroTools)
# ---------------------------------------------------------------------------
TOOLS_MAX_BBOX_LIMITS_MM: BBoxLimits = ((-30.0, -30.0, -30.0), (30.0, 30.0, 30.0))
TOOLS_MIN_BBOX_LIMITS_MM: BBoxLimits = ((-1.0, -5.0, -1.0), (1.0, 5.0, 1.0))
TOOLS_SNAP_PLANE_TARGET_XY: tuple[float, float] = (0.0, -1.0)
TOOLS_MOUNTING_VERTEX_COUNT = 12
TOOLS_DEFORMABLE_TIP_Y_PERCENTILE = 0.78
TOOLS_FINGERTIP_BODIES = {
    "nero_gripper_link1_tip": "nero_fingertip_left_tip",
    "nero_gripper_link2_tip": "nero_fingertip_right_tip",
}
TOOLS_FINGERTIP_KEYS = {"nero_gripper_link1_tip": "left", "nero_gripper_link2_tip": "right"}
TOOLS_FINGERTIP_MESH = "robots/deformgrip/ftip_deform/grip_fingertip_tip.obj"
TOOLS_OBJECTS = ("carabiner", "wire_cutters")

# ---------------------------------------------------------------------------
# Convex decomposition settings
# ---------------------------------------------------------------------------
DECOMPOSITION_BACKENDS = ("cuacd", "vhacd", "visacd")
VISACD_OPTIONS = {"concavity": 0.04, "seed": 42}
CUACD_OPTIONS = {
    "threshold": 0.05, "depth": 2, "width": 60, "merge_hulls": True,
    "n_concave_edges": 0, "workers": 8, "pool_bytes": None,
}
DECOMPOSITION_CACHE_BYTES = 512 * 1024 * 1024
MIN_PART_DIM_M = 1e-3


@dataclass(frozen=True)
class FingertipPreset:
    mesh_file: str
    deformable_tip_y_percentile: float


FINGERTIP_PRESETS = {
    "angle_76": FingertipPreset("robots/deformleap/ftip_deform/variants/angle_76/leap_fingertip_tip_angle_76.obj", 0.73),
}


@dataclass(frozen=True)
class MeshRejectorConfig:
    """Optional geometric rejectors; ``None`` disables a threshold-based rejector."""

    min_thickness_mm: float | None = 0.30
    min_volume_ratio: float | None = 0.90
    min_backplate_area_ratio: float | None = 0.30
    reject_degenerate_meshes: bool = True
    enforce_minimum_bbox: bool = True

    def validate(self) -> None:
        if self.min_thickness_mm is not None and (not np.isfinite(self.min_thickness_mm) or self.min_thickness_mm <= 0.0):
            raise ValueError("min_thickness_mm must be positive and finite, or None")
        if self.min_volume_ratio is not None and (not np.isfinite(self.min_volume_ratio) or not 0.0 < self.min_volume_ratio <= 1.0):
            raise ValueError("min_volume_ratio must be in (0, 1], or None")
        if self.min_backplate_area_ratio is not None and (
            not np.isfinite(self.min_backplate_area_ratio) or not 0.0 <= self.min_backplate_area_ratio <= 1.0
        ):
            raise ValueError("min_backplate_area_ratio must be in [0, 1], or None")


def default_tools_rejectors() -> MeshRejectorConfig:
    """Flat parallel-gripper pads permit losing more volume than LEAP wedge tips (0.50 vs 0.90)."""
    return MeshRejectorConfig(min_volume_ratio=0.50)


@dataclass(frozen=True)
class _BaseNeroConfig:
    """Shared task configuration fields and validation for NeroGrasp and NeroTools."""

    object_name: str = "set_screw"
    trajectory: str | None = None
    seed: int = 0
    use_cpu: bool = False
    fidelity: int = 1
    object_position_randomization: float = 0.01
    object_rotation_randomization_deg: float = 5.0
    asymmetric_observations: bool = True
    sample_perturbation_m: float = 0.0001
    snap_plane_clamp: bool = False
    convex_decomposition_backend: str = "cuacd"
    convex_max_parts: int = 16
    convex_target_parts: int | None = None
    rejectors: MeshRejectorConfig = field(default_factory=MeshRejectorConfig)
    runtime_dir: str | None = None
    shape_parallel_workers: int | None = None
    debug: bool = False
    verbose: bool = False
    device: str | None = None

    def _validate_common(self) -> None:
        if self.trajectory not in (None, "demo"):
            raise ValueError("trajectory must be None or 'demo'")
        if not np.isfinite(self.sample_perturbation_m) or self.sample_perturbation_m < 0:
            raise ValueError("sample_perturbation_m must be finite and nonnegative")
        if self.fidelity not in (0, 1):
            raise ValueError("fidelity must be 0 or 1")
        if self.object_position_randomization < 0.0:
            raise ValueError("object_position_randomization must be non-negative")
        if self.object_rotation_randomization_deg < 0.0:
            raise ValueError("object_rotation_randomization_deg must be non-negative")
        if self.convex_max_parts < 2:
            raise ValueError("convex_max_parts must be at least 2")
        if self.convex_target_parts is not None and not (0 < self.convex_target_parts < self.convex_max_parts):
            raise ValueError("convex_target_parts must be positive and below convex_max_parts")
        if self.shape_parallel_workers is not None and self.shape_parallel_workers <= 0:
            raise ValueError("shape_parallel_workers must be positive")
        if self.decomposition_backend not in DECOMPOSITION_BACKENDS:
            raise ValueError(
                f"Unknown convex decomposition backend '{self.convex_decomposition_backend}'. "
                f"Expected one of: {', '.join(DECOMPOSITION_BACKENDS)}."
            )
        self.rejectors.validate()

    @property
    def decomposition_backend(self) -> str:
        return str(self.convex_decomposition_backend).strip().lower()

    @property
    def target_parts(self) -> int:
        return min(8, self.convex_max_parts - 1) if self.convex_target_parts is None else int(self.convex_target_parts)

    def decomposition_options(self) -> dict:
        options = (
            dict(VISACD_OPTIONS) if self.decomposition_backend == "visacd" else
            dict(CUACD_OPTIONS) if self.decomposition_backend == "cuacd" else {}
        )
        options["target_parts"] = self.target_parts
        return options


@dataclass(frozen=True)
class NeroGraspConfig(_BaseNeroConfig):
    """Public configuration for ``NeroGrasp-<Object>``."""

    object_name: str = "set_screw"
    include_object_one_hot: bool = False
    fingertip_preset: str = "angle_76"

    def validate(self) -> None:
        if not self.object_name:
            raise ValueError("object_name must be non-empty")
        if self.fingertip_preset != "angle_76":
            raise ValueError("angle_76 is the only canonical template; supply other meshes through reset")
        self._validate_common()


@dataclass(frozen=True)
class NeroToolsConfig(_BaseNeroConfig):
    """Public configuration for ``NeroGrasp-Tools-<Tool>``."""

    object_name: str = "carabiner"
    gripper_action_noise_std: float = 0.004
    fingertip_preset: str = "gripper"
    rejectors: MeshRejectorConfig = field(default_factory=default_tools_rejectors)

    def validate(self) -> None:
        if self.object_name not in TOOLS_OBJECTS:
            raise ValueError(f"Unknown tool object '{self.object_name}'. Expected one of: {', '.join(TOOLS_OBJECTS)}.")
        if self.fingertip_preset != "gripper":
            raise ValueError("gripper is the only canonical template for NeroGrasp-Tools; supply other meshes through reset")
        if not np.isfinite(self.gripper_action_noise_std) or self.gripper_action_noise_std < 0.0:
            raise ValueError("gripper_action_noise_std must be finite and non-negative")
        self._validate_common()


def resolve_worker_count(requested: int | None) -> int:
    """CPU worker budget: explicit request, else env override, else CPU count."""
    if requested is not None:
        return int(requested)
    for name in ("ADA_GO_PARALLEL_WORKERS", "SLURM_CPUS_PER_TASK"):
        value = os.environ.get(name)
        if value is not None:
            try:
                workers = int(value)
            except ValueError as exc:
                raise ValueError(f"{name} must be a positive integer, got {value!r}") from exc
            if workers <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
            return workers
    return os.cpu_count() or 1
