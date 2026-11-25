"""Geometric rejectors for postprocessed fingertip meshes.

A design that passes ``MeshSpace`` can still be unusable. The optional
rejectors in ``MeshRejectorConfig`` catch the common cases:

- degenerate meshes (fewer than four vertices or faces);
- volume below ``min_volume_ratio`` of the canonical tip;
- a tip smaller than ``SHAPE_MIN_BBOX_LIMITS_MM``;
- local thickness below ``min_thickness_mm``, measured by casting a ray
  inwards from every face centre (needs ``embreex``).

- is_degenerate_mesh, mesh_volume, thickness_metrics: per-mesh measurements;
- rejection_reasons: one fingertip's reasons across a batch of designs.
"""

from __future__ import annotations

import warnings

import numpy as np
import trimesh

from ..config import SNAP_PLANE_CUT_MARGIN_MM
from .adapter import Fingertip

METERS_TO_MM = 1000.0
Y_AXIS = 1
# A snap plane whose normal is this aligned with +-y moves the minimum-box floor to the cut.
SNAP_ALIGN_THRESHOLD = 0.9
FLOAT_TOLERANCE_MM = 1e-3


def is_degenerate_mesh(vertices, faces=None, min_dim_threshold: float | None = None) -> bool:
    """Whether a mesh lacks enough geometry for robust processing."""
    if len(vertices) < 4 or (faces is not None and len(faces) < 4):
        return True
    if min_dim_threshold is None:
        return False
    vertices = np.asarray(vertices)
    return len(vertices) > 0 and np.min(vertices.max(axis=0) - vertices.min(axis=0)) < min_dim_threshold


def mesh_volume(vertices, faces) -> float:
    """Enclosed volume (m^3) by the divergence theorem; 0 for a degenerate mesh."""
    if is_degenerate_mesh(vertices, faces):
        return 0.0
    triangles = np.asarray(vertices, dtype=np.float64)[np.asarray(faces, dtype=np.int64)]
    edges = triangles[:, 1:, :] - triangles[:, :2, :]
    crosses = np.cross(edges[:, 0], edges[:, 1])
    first_integral = crosses[:, 0] * (triangles[:, 0, 0] + triangles[:, 1, 0] + triangles[:, 2, 0])
    volume = float(abs(np.sum(first_integral) / 6.0))
    return volume if np.isfinite(volume) else 0.0


def thickness_metrics(vertices: np.ndarray, faces: np.ndarray) -> dict[str, float | int]:
    """Local ray thickness (mm) at every face centre: its 2nd percentile (p02), and how many samples were usable."""
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    sample_count = int(len(faces))
    empty = {"min_thickness_mm": 0.0, "thickness_sample_count": sample_count,
             "thickness_valid_sample_count": 0, "thickness_invalid_fraction": 1.0}
    if is_degenerate_mesh(vertices, faces) or sample_count == 0:
        return empty
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    points = np.asarray(mesh.triangles_center, dtype=np.float64)
    normals = np.asarray(mesh.face_normals, dtype=np.float64)
    valid = np.all(np.isfinite(points), axis=1) & np.all(np.isfinite(normals), axis=1) & (np.linalg.norm(normals, axis=1) > 1e-12)
    if not np.any(valid):
        return empty
    try:
        from trimesh.ray.ray_pyembree import RayMeshIntersector
    except ImportError as exc:
        raise ImportError("Fingertip thickness testing requires the embreex package.") from exc
    mesh.ray = RayMeshIntersector(mesh)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        thickness_m = trimesh.proximity.thickness(mesh, points[valid], normals=normals[valid], method="ray")
    thickness_mm = np.asarray(thickness_m, dtype=np.float64) * METERS_TO_MM
    finite = thickness_mm[np.isfinite(thickness_mm) & (thickness_mm >= 0.0)]
    valid_count = int(finite.size)
    return {
        "min_thickness_mm": float(np.percentile(finite, 2)) if valid_count else 0.0,
        "thickness_sample_count": sample_count,
        "thickness_valid_sample_count": valid_count,
        "thickness_invalid_fraction": float((sample_count - valid_count) / max(sample_count, 1)),
    }


def min_bbox_limits_mm(fingertip: Fingertip, limits_mm, snap_plane_clamp: bool) -> tuple[np.ndarray, np.ndarray]:
    """The box (mm) a tip's bounding box must cover: ``limits_mm``'s extents, centred on the
    canonical tip in x and z, and starting at its lowest y (at the cut, when it runs along y)."""
    bbox_min_mm, bbox_max_mm = (np.asarray(limit, dtype=np.float64) for limit in limits_mm)
    half_extents_mm = 0.5 * (bbox_max_mm - bbox_min_mm)
    extents_mm = bbox_max_mm - bbox_min_mm
    base = fingertip.vertices
    center_mm = 0.5 * (base.min(axis=0) + base.max(axis=0)) * METERS_TO_MM
    min_limits_mm = center_mm - half_extents_mm
    max_limits_mm = center_mm + half_extents_mm
    min_limits_mm[Y_AXIS] = float(base.min(axis=0)[Y_AXIS] * METERS_TO_MM)
    plane = fingertip.snap_plane
    if snap_plane_clamp and plane is not None and abs(plane.normal[Y_AXIS]) > SNAP_ALIGN_THRESHOLD:
        min_limits_mm[Y_AXIS] = float(plane.support_point[Y_AXIS] * METERS_TO_MM) + SNAP_PLANE_CUT_MARGIN_MM + FLOAT_TOLERANCE_MM
    max_limits_mm[Y_AXIS] = min_limits_mm[Y_AXIS] + extents_mm[Y_AXIS]
    return min_limits_mm, max_limits_mm


def rejection_reasons(
    metrics: list[dict],
    fingertip: Fingertip,
    *,
    base_volume: float,
    min_volume_ratio: float,
    min_bbox_limits_mm_: tuple | None,
    snap_plane_clamp: bool,
    min_thickness_mm: float | None,
) -> list[list[str]]:
    """Reasons each design's mesh of one fingertip is rejected (empty when accepted).

    ``metrics`` holds one dict per design with ``name``, ``volume``,
    ``bbox_min_mm`` and ``bbox_max_mm`` (and ``min_thickness_mm`` and the
    thickness sample counts when ``min_thickness_mm`` is set).
    """
    if not metrics:
        return []
    reasons = [[] for _ in metrics]
    names = [str(metric["name"]) for metric in metrics]
    volumes = np.asarray([metric["volume"] for metric in metrics], dtype=np.float64)
    if base_volume > 0.0:
        min_allowed_volume = min_volume_ratio * base_volume
        for index in np.flatnonzero(volumes < min_allowed_volume):
            reasons[index].append(f"{names[index]} volume {volumes[index]:.6e} < {min_allowed_volume:.6e}")

    if min_bbox_limits_mm_ is not None:
        bbox_min_mm = np.stack([metric["bbox_min_mm"] for metric in metrics]).astype(np.float64)
        bbox_max_mm = np.stack([metric["bbox_max_mm"] for metric in metrics]).astype(np.float64)
        allowed_min, allowed_max = min_bbox_limits_mm(fingertip, min_bbox_limits_mm_, snap_plane_clamp)
        for index in np.flatnonzero(np.any(bbox_min_mm > allowed_min, axis=1)):
            reasons[index].append(f"{names[index]} min bbox min {bbox_min_mm[index].tolist()} mm > {allowed_min.tolist()} mm")
        for index in np.flatnonzero(np.any(bbox_max_mm < allowed_max, axis=1)):
            reasons[index].append(f"{names[index]} min bbox max {bbox_max_mm[index].tolist()} mm < {allowed_max.tolist()} mm")

    if min_thickness_mm is not None and float(min_thickness_mm) > 0.0:
        threshold = float(min_thickness_mm)
        thicknesses = np.asarray([metric.get("min_thickness_mm", 0.0) for metric in metrics], dtype=np.float64)
        for index in np.flatnonzero(~np.isfinite(thicknesses) | (thicknesses < threshold)):
            sample_count = int(metrics[index].get("thickness_sample_count", 0))
            valid_count = int(metrics[index].get("thickness_valid_sample_count", 0))
            reasons[index].append(f"{names[index]} min thickness {thicknesses[index]:.3f} mm < {threshold:.3f} mm "
                                  f"({valid_count}/{sample_count} valid samples)")
    return reasons
