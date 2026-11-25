"""Canonical fingertip geometry: which vertices a design may move, and the tip's support plane.

Each designed fingertip body (``config.FINGERTIP_BODIES``) carries one tip mesh.
Its canonical mesh is split along +y: the vertices at the free end are movable,
the rest (the first ``MOUNTING_VERTEX_COUNT``) mount the tip on the finger. The
seam centre, where the two regions meet, is the point a too-large design is
scaled towards. The support plane is the flat facet facing
``SNAP_PLANE_TARGET_XY``; with ``snap_plane_clamp`` designs are cut at it.

- Fingertip, SnapPlane: per-tip geometry, computed once per env;
- load_fingertips: build them from the robot MJCF and the preset mesh.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Optional

import numpy as np
import trimesh

from ..config import FINGERTIP_BODIES, GEOMETRY_EPSILON, SNAP_PLANE_TARGET_XY


@dataclass(frozen=True)
class SnapPlane:
    """Plane of the fingertip's flat support facet, in mesh coordinates."""
    normal: np.ndarray
    offset: float
    support_point: np.ndarray
    # Which side of the plane the tip lies on (the side kept when cutting).
    keep_negative_halfspace: bool
    # Canonical vertices of the support facet, projected onto the plane when cutting.
    support_vertex_indices: np.ndarray
    support_patch_vertices: np.ndarray


@dataclass(frozen=True)
class Fingertip:
    """One designed fingertip: its body, mesh asset and canonical geometry."""
    name: str                      # body name, e.g. "leap_if_tip"
    mesh_name: str                 # MJCF mesh asset, e.g. "leap_if_tip_trimesh"
    vertices: np.ndarray           # canonical vertices (m), float64
    faces: np.ndarray              # canonical triangles, int64
    non_tip_indices: np.ndarray    # mounting vertices, never moved
    deformable_vertex_indices: np.ndarray
    seam_center: np.ndarray
    snap_plane: Optional[SnapPlane]


def normalize_vector(vec: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float64)
    norm = float(np.linalg.norm(arr))
    if not np.isfinite(norm) or norm <= 1e-12:
        return np.asarray(fallback, dtype=np.float64).copy()
    return arr / norm


def plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = normalize_vector(normal, fallback=np.array([0.0, 0.0, 1.0], dtype=np.float64))
    helper = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(helper, n))) > 0.95:
        helper = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    u = normalize_vector(np.cross(n, helper), fallback=np.array([1.0, 0.0, 0.0], dtype=np.float64))
    v = normalize_vector(np.cross(n, u), fallback=np.array([0.0, 1.0, 0.0], dtype=np.float64))
    return u, v


def load_fingertips(
    robot_root: ET.Element,
    assets_path: str,
    deformable_tip_percentile: float,
    *,
    fingertip_bodies: dict[str, str] = FINGERTIP_BODIES,
    snap_plane_target_xy: tuple[float, float] = SNAP_PLANE_TARGET_XY,
    mounting_vertex_count: int | None = None,
) -> list[Fingertip]:
    """Canonical geometry of every designed fingertip, in ``fingertip_bodies`` order.

    ``robot_root`` is the robot MJCF whose mesh ``file`` attributes are relative
    to ``assets_path`` (or absolute).
    """
    mesh_files = {}
    for mesh_el in robot_root.iter("mesh"):
        name = mesh_el.get("name") or os.path.splitext(os.path.basename(mesh_el.get("file", "")))[0]
        if name and mesh_el.get("file"):
            mesh_files[name] = mesh_el.get("file")

    fingertips = []
    for body_name, mesh_name in fingertip_bodies.items():
        body = robot_root.find(f".//body[@name='{body_name}']")
        if body is None:
            raise ValueError(f"End-effector body '{body_name}' not found in XML.")
        if mesh_name not in mesh_files:
            raise ValueError(f"Tip mesh file not found for effector '{body_name}': tip={mesh_name}")
        mesh = trimesh.load_mesh(os.path.join(assets_path, mesh_files[mesh_name]), process=False)
        vertices = np.array(mesh.vertices, dtype=np.float64)
        faces = np.array(mesh.faces, dtype=np.int64)

        # The free (+y) end is movable, the mounting end is fixed.
        if mounting_vertex_count is not None:
            non_tip = np.arange(int(mounting_vertex_count), dtype=np.int64)
        else:
            pct = float(np.clip(deformable_tip_percentile, 0.0, 1.0)) * 100.0
            tip_mask = vertices[:, 1] >= np.percentile(vertices[:, 1], 100.0 - pct)
            non_tip = np.where(~tip_mask)[0]
        deformable, seam_center = _deformable_seam(vertices, faces, non_tip)
        fingertips.append(Fingertip(
            name=body_name, mesh_name=mesh_name, vertices=vertices, faces=faces, non_tip_indices=non_tip,
            deformable_vertex_indices=deformable, seam_center=seam_center,
            snap_plane=_build_snap_plane(vertices, faces, snap_plane_target_xy=snap_plane_target_xy)))
    return fingertips


def _deformable_seam(vertices: np.ndarray, faces: np.ndarray, non_tip_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Movable vertex indices, and the mean midpoint of edges joining them to the mounting region."""
    deformable_mask = np.ones(len(vertices), dtype=bool)
    deformable_mask[non_tip_indices] = False
    deformable = np.flatnonzero(deformable_mask).astype(np.int64)

    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0), axis=1)
    edges = np.unique(edges, axis=0)
    crossing = edges[deformable_mask[edges[:, 0]] != deformable_mask[edges[:, 1]]]
    if len(crossing):
        seam_center = (0.5 * (vertices[crossing[:, 0]] + vertices[crossing[:, 1]])).mean(axis=0)
    elif deformable.size > 0:
        seam_center = vertices[deformable].mean(axis=0)
    else:
        seam_center = vertices.mean(axis=0)
    return deformable, seam_center.astype(np.float64)


def _build_snap_plane(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    snap_plane_target_xy: tuple[float, float] = SNAP_PLANE_TARGET_XY,
) -> SnapPlane:
    mesh = trimesh.util.concatenate([trimesh.Trimesh(vertices=vertices, faces=faces, process=False)])
    target = np.array([snap_plane_target_xy[0], snap_plane_target_xy[1], 0.0], dtype=np.float64)
    support_dir = normalize_vector(target, fallback=np.array([-1.0, 1.0, 0.0], dtype=np.float64))
    patch = _find_support_patch(mesh, support_dir)
    mesh_vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if patch is not None:
        normal, base_offset, vertex_indices, patch_vertices = patch
        axis_point = _support_rotation_axis_point(patch_vertices)
        support_point = axis_point.copy()
    else:
        normal = normalize_vector(support_dir, fallback=np.array([-1.0, 0.0, 0.0], dtype=np.float64))
        projections = mesh_vertices @ normal
        support_index = int(np.argmax(projections))
        support_point = mesh_vertices[support_index].astype(np.float64, copy=True)
        base_offset = float(projections[support_index])
        vertex_indices = np.zeros((0,), dtype=np.int64)
        patch_vertices = np.zeros((0, 3), dtype=np.float64)
        axis_point = support_point.copy()
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    tol = max(float(np.linalg.norm(bounds[1] - bounds[0])) * 1e-6, GEOMETRY_EPSILON)
    return SnapPlane(
        normal=normal.copy(), offset=float(np.dot(normal, axis_point)), support_point=support_point,
        keep_negative_halfspace=_keep_negative_halfspace(mesh_vertices, normal, base_offset, tol),
        support_vertex_indices=vertex_indices, support_patch_vertices=patch_vertices)


def _keep_negative_halfspace(vertices: np.ndarray, normal: np.ndarray, offset: float, tol: float) -> bool:
    signed = np.asarray(vertices, dtype=np.float64) @ np.asarray(normal, dtype=np.float64) - float(offset)
    if float(np.max(signed)) <= tol:
        return True
    if float(np.min(signed)) >= -tol:
        return False
    return float(np.clip(signed, 0.0, None).sum()) <= float(np.clip(-signed, 0.0, None).sum())


def _find_support_patch(mesh: trimesh.Trimesh, support_dir: np.ndarray):
    """The planar facet best facing ``support_dir``: (normal, offset, vertex indices, patch vertices), or None."""
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if verts.size == 0 or faces.size == 0:
        return None
    facets = mesh.facets
    if len(facets) == 0:
        return None

    facet_normals = np.asarray(mesh.facets_normal, dtype=np.float64)
    facet_areas = np.asarray(mesh.facets_area, dtype=np.float64)
    raw_align = facet_normals @ support_dir
    oriented = np.ones(len(facet_normals), dtype=bool)
    if abs(float(support_dir[0])) > 1e-8:
        oriented &= facet_normals[:, 0] * support_dir[0] > 0.0
    if abs(float(support_dir[1])) > 1e-8:
        oriented &= facet_normals[:, 1] * support_dir[1] > 0.0
    candidates = np.where(oriented & (raw_align > 0.0))[0]
    scores, alignment_dir = raw_align, support_dir
    if candidates.size == 0:
        x_dir = np.array([np.sign(support_dir[0]), 0.0, 0.0], dtype=np.float64)
        if abs(float(x_dir[0])) > 1e-8:
            x_align = facet_normals @ x_dir
            candidates = np.where(x_align > 0.0)[0]
            if candidates.size > 0:
                scores, alignment_dir = x_align, x_dir
    if candidates.size == 0:
        candidates = np.where(raw_align > 0.0)[0]
    if candidates.size == 0:
        return None

    best = float(np.max(scores[candidates]))
    best_candidates = candidates[scores[candidates] >= best - 1e-6]
    if best_candidates.size > 1:
        best_candidates = best_candidates[np.argsort(facet_areas[best_candidates])[::-1]]
    facet = int(best_candidates[0])

    normal = facet_normals[facet].astype(np.float64, copy=True)
    if float(np.dot(normal, alignment_dir)) < 0.0:
        normal = -normal
    normal = normalize_vector(normal, fallback=support_dir)

    selected_faces = np.asarray(facets[facet], dtype=np.int64)
    vertex_indices = np.unique(faces[selected_faces].reshape(-1)).astype(np.int64)
    patch = mesh.submesh([selected_faces], append=True, repair=False)
    patch_vertices = np.asarray(patch.vertices, dtype=np.float64)
    if patch_vertices.size == 0 or np.asarray(patch.faces).size == 0:
        return None
    offset = float(np.mean(patch_vertices @ normal))
    residual = np.abs(patch_vertices @ normal - offset)
    planarity_tol = max(float(np.linalg.norm(np.ptp(patch_vertices, axis=0))) * 3e-3, 1e-6)
    if float(np.max(residual)) > planarity_tol:
        return None
    return normal, offset, vertex_indices, patch_vertices


def _support_rotation_axis_point(patch_vertices: np.ndarray) -> np.ndarray:
    """Midpoint of the support facet's lowest-y edge (its lowest-x edge if that is a single vertex)."""
    verts = np.asarray(patch_vertices, dtype=np.float64)
    bounds = np.stack([verts.min(axis=0), verts.max(axis=0)], axis=0)
    edge_tol = max(float(np.linalg.norm(bounds[1] - bounds[0])) * 1e-5, 1e-6)
    axis_vertices = verts[verts[:, 1] <= float(np.min(verts[:, 1])) + edge_tol]
    if axis_vertices.shape[0] < 2:
        axis_vertices = verts[verts[:, 0] <= float(np.min(verts[:, 0])) + edge_tol]
    if axis_vertices.shape[0] < 2:
        axis_vertices = verts
    return axis_vertices.mean(axis=0)
