"""Fit a fingertip design to the allowed geometry before it is decomposed.

Two steps, both deterministic:

- bounding-box shrink (always on): if the movable vertices leave the allowed
  box (``SHAPE_MAX_BBOX_LIMITS_MM`` centred on the canonical tip), they are
  scaled per axis towards the seam centre until they fit;
- snap-plane cut (``snap_plane_clamp``): the support-facet vertices are
  projected onto the support plane (keeping at least ``min_backplate_area_ratio``
  of the canonical facet area), everything beyond the plane is cut off and the
  opening is capped. Only canonical topology is accepted on this path.

- Postprocessor: holds the per-fingertip invariants and runs both steps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..config import GEOMETRY_EPSILON, SHAPE_MAX_BBOX_LIMITS_MM, SNAP_PLANE_CUT_MARGIN_MM
from .adapter import Fingertip, SnapPlane, plane_basis


@dataclass(frozen=True)
class _Invariants:
    """Arrays that depend only on the canonical fingertip, computed once."""
    unique_edges: np.ndarray
    face_edge_indices: np.ndarray
    plane_basis: Optional[np.ndarray]
    target_patch_2d: Optional[np.ndarray]
    target_patch_extents: Optional[np.ndarray]
    bbox_min_allowed_m: np.ndarray
    bbox_max_allowed_m: np.ndarray
    fixed_bbox_min_m: np.ndarray
    fixed_bbox_max_m: np.ndarray


def centered_bbox_limits_mm(fingertip: Fingertip, bbox_limits_mm) -> tuple[np.ndarray, np.ndarray]:
    """``bbox_limits_mm``'s extents, centred on the canonical tip's bounding-box centre."""
    bbox_min_mm, bbox_max_mm = (np.asarray(limit, dtype=np.float64) for limit in bbox_limits_mm)
    half_extents_mm = 0.5 * (bbox_max_mm - bbox_min_mm)
    center_mm = 0.5 * (fingertip.vertices.min(axis=0) + fingertip.vertices.max(axis=0)) * 1000.0
    return center_mm - half_extents_mm, center_mm + half_extents_mm


def stable_unique_edges(faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unique undirected edges in first-seen face order, and each face's three edge indices."""
    face_edges = np.stack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=1)
    canonical = np.sort(face_edges.reshape(-1, 2), axis=1)
    _, first, inverse_sorted = np.unique(canonical, axis=0, return_index=True, return_inverse=True)
    stable_order = np.argsort(first, kind="stable")
    sorted_to_stable = np.empty(len(stable_order), dtype=np.int64)
    sorted_to_stable[stable_order] = np.arange(len(stable_order), dtype=np.int64)
    first_seen_edges = face_edges.reshape(-1, 2)[first[stable_order]]
    return first_seen_edges, sorted_to_stable[inverse_sorted].reshape(len(faces), 3)


class Postprocessor:
    def __init__(
        self,
        *,
        snap_plane_clamp: bool,
        min_backplate_area_ratio: float,
        max_bbox_limits_mm=SHAPE_MAX_BBOX_LIMITS_MM,
    ):
        self.snap_plane_clamp = bool(snap_plane_clamp)
        self.min_backplate_area_ratio = float(min_backplate_area_ratio)
        self.max_bbox_limits_mm = max_bbox_limits_mm
        self._invariants: dict[str, _Invariants] = {}

    # ----- public -----

    def postprocess(self, fingertip: Fingertip, vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The mesh a design becomes: shrunk to the allowed box and, with ``snap_plane_clamp``, cut at the support plane.

        With ``snap_plane_clamp`` the design must have the canonical topology:
        its mounting vertices are restored and ``faces`` are the canonical ones.
        """
        working = np.asarray(vertices, dtype=np.float64).copy()
        if not self.snap_plane_clamp:
            movable = np.arange(len(fingertip.non_tip_indices), len(working))
            return self.bbox_shrink(fingertip, working, movable), np.asarray(faces, dtype=np.int64).copy()
        working[fingertip.non_tip_indices] = fingertip.vertices[fingertip.non_tip_indices]
        working = self.bbox_shrink(fingertip, working, fingertip.deformable_vertex_indices)
        return self._cut_and_cap(fingertip, working)

    def bbox_shrink(self, fingertip: Fingertip, vertices: np.ndarray, movable: np.ndarray) -> np.ndarray:
        """Scale ``vertices[movable]`` towards the seam centre, per axis, until the tip fits the allowed box.

        Updates and returns ``vertices`` (the caller's working copy).
        """
        verts = vertices
        if verts.ndim != 2 or verts.shape[1] != 3 or verts.shape[0] == 0:
            return verts
        inv = self._get_invariants(fingertip)
        if movable.size == 0 or int(np.max(movable)) >= int(verts.shape[0]):
            return verts

        seam_center = fingertip.seam_center
        movable_vertices = verts[movable]
        scales = np.ones(3, dtype=np.float64)
        tol = max(GEOMETRY_EPSILON, 1e-9)
        movable_min = movable_vertices.min(axis=0)
        movable_max = movable_vertices.max(axis=0)
        bbox_min = np.minimum(inv.fixed_bbox_min_m, movable_min)
        bbox_max = np.maximum(inv.fixed_bbox_max_m, movable_max)
        min_deltas = movable_min - seam_center
        max_deltas = movable_max - seam_center
        for axis in range(3):
            center = float(seam_center[axis])
            lower = float(inv.bbox_min_allowed_m[axis])
            upper = float(inv.bbox_max_allowed_m[axis])
            scale = 1.0
            if float(bbox_min[axis]) < lower - tol:
                if center <= lower + tol:
                    scale = 0.0
                elif float(min_deltas[axis]) < -tol:
                    scale = min(scale, (lower - center) / float(min_deltas[axis]))
            if float(bbox_max[axis]) > upper + tol:
                if center >= upper - tol:
                    scale = 0.0
                elif float(max_deltas[axis]) > tol:
                    scale = min(scale, (upper - center) / float(max_deltas[axis]))
            scales[axis] = min(max(scale, 0.0), 1.0)
        if np.allclose(scales, 1.0, atol=tol, rtol=0.0):
            return verts
        verts[movable] = seam_center.reshape(1, 3) + (movable_vertices - seam_center.reshape(1, 3)) * scales.reshape(1, 3)
        return verts

    # ----- invariants -----

    def _get_invariants(self, fingertip: Fingertip) -> _Invariants:
        cached = self._invariants.get(fingertip.name)
        if cached is not None:
            return cached
        unique_edges, face_edge_indices = stable_unique_edges(fingertip.faces)
        bbox_min_mm, bbox_max_mm = centered_bbox_limits_mm(fingertip, self.max_bbox_limits_mm)
        basis = target_2d = target_extents = None
        if fingertip.snap_plane is not None:
            basis = np.stack(plane_basis(fingertip.snap_plane.normal), axis=1)
            target_2d = fingertip.snap_plane.support_patch_vertices @ basis
            target_extents = np.ptp(target_2d, axis=0) if len(target_2d) > 0 else np.zeros(2, dtype=np.float64)
        fixed = fingertip.vertices[fingertip.non_tip_indices]
        if len(fixed):
            fixed_min, fixed_max = fixed.min(axis=0), fixed.max(axis=0)
        else:
            fixed_min, fixed_max = np.full(3, np.inf), np.full(3, -np.inf)
        cached = _Invariants(
            unique_edges=unique_edges, face_edge_indices=face_edge_indices, plane_basis=basis,
            target_patch_2d=target_2d, target_patch_extents=target_extents,
            bbox_min_allowed_m=bbox_min_mm / 1000.0, bbox_max_allowed_m=bbox_max_mm / 1000.0,
            fixed_bbox_min_m=fixed_min, fixed_bbox_max_m=fixed_max)
        self._invariants[fingertip.name] = cached
        return cached

    # ----- snap-plane cut -----

    def _project_to_snap_plane(self, vertices: np.ndarray, snap_plane: SnapPlane, inv: _Invariants) -> np.ndarray:
        verts = vertices.copy()
        support = np.asarray(snap_plane.support_vertex_indices, dtype=np.int64)
        if support.size == 0 or int(np.max(support)) >= int(verts.shape[0]):
            return verts
        normal = np.asarray(snap_plane.normal, dtype=np.float64)
        signed = verts[support] @ normal - float(snap_plane.offset)
        projected = verts[support] - signed.reshape(-1, 1) * normal.reshape(1, 3)
        verts[support] = self._enforce_min_backplate_area(projected, inv)
        return verts

    def _enforce_min_backplate_area(self, projected: np.ndarray, inv: _Invariants) -> np.ndarray:
        """Grow the projected support facet about its centre to at least ``min_backplate_area_ratio`` of the canonical one."""
        projected = projected.copy()
        ratio = self.min_backplate_area_ratio
        basis, target_2d, target_extents = inv.plane_basis, inv.target_patch_2d, inv.target_patch_extents
        if ratio <= 0.0 or projected.shape[0] < 2 or target_2d is None or target_2d.shape[0] < 2 or basis is None:
            return projected
        projected_2d = projected @ basis
        current_min = projected_2d.min(axis=0)
        current_max = projected_2d.max(axis=0)
        min_area = ratio * float(np.prod(target_extents))
        current_area = float(np.prod(current_max - current_min))
        tol = max(GEOMETRY_EPSILON, 1e-12)
        if min_area <= tol or current_area + tol >= min_area:
            return projected
        center_2d = 0.5 * (current_min + current_max)
        if current_area > tol:
            corrected_2d = center_2d + (projected_2d - center_2d) * float(np.sqrt(min_area / current_area))
        elif target_2d.shape[0] == projected_2d.shape[0]:
            target_center = 0.5 * (target_2d.min(axis=0) + target_2d.max(axis=0))
            corrected_2d = center_2d + (target_2d - target_center) * np.sqrt(ratio)
        else:
            return projected
        return projected + (corrected_2d - projected_2d) @ basis.T

    def _cut_and_cap(self, fingertip: Fingertip, vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Clip the mesh to the tip's side of the snap plane and cap each opening with 2D ear-clipping."""
        snap_plane = fingertip.snap_plane
        faces = fingertip.faces
        if snap_plane is None or faces.size == 0:
            return vertices.copy(), faces.copy()
        inv = self._get_invariants(fingertip)
        working = self._project_to_snap_plane(vertices, snap_plane, inv)
        working[fingertip.non_tip_indices] = fingertip.vertices[fingertip.non_tip_indices]

        normal = np.asarray(snap_plane.normal, dtype=np.float64)
        tol = max(GEOMETRY_EPSILON, 1e-9)
        keep_negative = bool(snap_plane.keep_negative_halfspace)
        signed = working @ normal - float(snap_plane.offset)
        kept_orig = signed <= tol if keep_negative else signed >= -tol
        if bool(np.all(kept_orig)):
            return working, faces.copy()

        margin_m = max(SNAP_PLANE_CUT_MARGIN_MM, 0.0) / 1000.0
        cut_offset = float(snap_plane.offset) - margin_m if keep_negative else float(snap_plane.offset) + margin_m
        signed_cut = working @ normal - cut_offset
        kept = signed_cut <= tol if keep_negative else signed_cut >= -tol
        kept[fingertip.non_tip_indices] = True

        all_vertices, edge_vertex_ids = _intersect_crossing_edges(working, inv.unique_edges, signed_cut, kept, tol)
        faces_out = _clip_triangles_to_halfspace(faces, inv.face_edge_indices, edge_vertex_ids, kept)
        if len(faces_out) == 0:
            return all_vertices, faces_out
        cap_normal = normal if keep_negative else -normal
        basis = inv.plane_basis if inv.plane_basis is not None else np.stack(plane_basis(cap_normal), axis=1)
        faces_out = _cap_boundary_loops(all_vertices, faces_out, cap_normal, basis)
        return _compact_preserving_mounting(all_vertices, faces_out, len(fingertip.non_tip_indices))


def _intersect_crossing_edges(
    working: np.ndarray,
    unique_edges: np.ndarray,
    signed_cut: np.ndarray,
    kept: np.ndarray,
    tol: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute deduplicated intersection vertices for every edge crossing the cut plane."""
    crossing_ids = np.flatnonzero(kept[unique_edges[:, 0]] != kept[unique_edges[:, 1]])
    crossing_edges = unique_edges[crossing_ids]
    if len(crossing_edges) > 0:
        edge_points = []
        for a, b in crossing_edges.tolist():
            s_a, s_b = float(signed_cut[a]), float(signed_cut[b])
            denom = s_a - s_b
            fraction = 0.5 if abs(denom) <= tol else float(np.clip(s_a / denom, 0.0, 1.0))
            edge_points.append(working[a] + fraction * (working[b] - working[a]))
        edge_points_arr = np.asarray(edge_points, dtype=np.float64)
        _, first, inverse_sorted = np.unique(np.round(edge_points_arr, decimals=10), axis=0, return_index=True, return_inverse=True)
        stable_order = np.argsort(first, kind="stable")
        sorted_to_stable = np.empty(len(stable_order), dtype=np.int64)
        sorted_to_stable[stable_order] = np.arange(len(stable_order), dtype=np.int64)
        inverse_stable = sorted_to_stable[inverse_sorted]
        unique_points = edge_points_arr[first[stable_order]]
    else:
        inverse_stable = np.zeros(0, dtype=np.int64)
        unique_points = np.zeros((0, 3), dtype=np.float64)
    edge_vertex_ids = np.full(len(unique_edges), -1, dtype=np.int64)
    edge_vertex_ids[crossing_ids] = len(working) + inverse_stable
    return np.concatenate([working, unique_points], axis=0), edge_vertex_ids


def _clip_triangles_to_halfspace(
    faces: np.ndarray,
    face_edge_indices: np.ndarray,
    edge_vertex_ids: np.ndarray,
    kept: np.ndarray,
) -> np.ndarray:
    """Split triangles crossing the cut plane into 0, 1, or 2 kept triangles."""
    # Encode which of the 3 corners stay kept.
    face_kept = kept[faces]
    face_codes = face_kept[:, 0].astype(np.uint8) + 2 * face_kept[:, 1].astype(np.uint8) + 4 * face_kept[:, 2].astype(np.uint8)
    # 1 kept -> 1 tri; 2 kept -> 2 tris.
    triangle_counts = np.asarray([0, 1, 1, 2, 1, 2, 2, 1], dtype=np.int64)[face_codes]
    face_offsets = np.zeros(len(faces), dtype=np.int64)
    if len(face_offsets) > 1:
        np.cumsum(triangle_counts[:-1], out=face_offsets[1:])
    faces_out = np.empty((int(triangle_counts.sum()), 3), dtype=np.int64)
    # Columns 0..2 are corners; 3..5 are cut vertices.
    tokens = np.concatenate([faces, edge_vertex_ids[face_edge_indices]], axis=1)
    # Lookup templates preserving original triangle winding.
    templates = {
        1: ((3, 5, 0),), 2: ((3, 1, 4),), 3: ((1, 4, 5), (1, 5, 0)), 4: ((4, 2, 5),),
        5: ((3, 4, 2), (3, 2, 0)), 6: ((3, 1, 2), (3, 2, 5)), 7: ((0, 1, 2),),
    }
    for code, code_templates in templates.items():
        face_ids = np.flatnonzero(face_codes == code)
        for local, template in enumerate(code_templates):
            faces_out[face_offsets[face_ids] + local] = tokens[face_ids][:, template]
    if len(faces_out) == 0:
        return np.zeros((0, 3), dtype=np.int64)
    return faces_out[_nondegenerate(faces_out)]


def _cap_boundary_loops(
    all_vertices: np.ndarray,
    faces_out: np.ndarray,
    cap_normal: np.ndarray,
    basis: np.ndarray,
) -> np.ndarray:
    """Find open cut-plane loops and triangulate each cap in 2D via ear clipping."""
    directed = np.stack([faces_out[:, [0, 1]], faces_out[:, [1, 2]], faces_out[:, [2, 0]]], axis=1).reshape(-1, 2)
    unique_boundary, first_boundary, boundary_inverse = np.unique(np.sort(directed, axis=1), axis=0, return_index=True, return_inverse=True)
    boundary_ids = np.flatnonzero(np.bincount(boundary_inverse, minlength=len(unique_boundary)) == 1)
    boundary_edges = unique_boundary[boundary_ids[np.argsort(first_boundary[boundary_ids], kind="stable")]]
    if len(boundary_edges) == 0:
        return faces_out
    cap_faces: list[np.ndarray] = []
    for loop in _closed_edge_loops(boundary_edges):
        if len(loop) < 3:
            continue
        ordered = np.asarray(loop, dtype=np.int64)
        local_tris = _ear_clip_2d(all_vertices[ordered] @ basis)
        if not local_tris:
            continue
        tri_idx = ordered[np.asarray(local_tris, dtype=np.int64)]
        v0, v1, v2 = all_vertices[tri_idx[:, 0]], all_vertices[tri_idx[:, 1]], all_vertices[tri_idx[:, 2]]
        if float(np.sum(np.cross(v1 - v0, v2 - v0) @ cap_normal)) < 0.0:
            tri_idx = tri_idx[:, [0, 2, 1]]
        cap_faces.append(tri_idx)
    if cap_faces:
        faces_out = np.concatenate([faces_out, *cap_faces], axis=0)
    return faces_out[_nondegenerate(faces_out)]


def _compact_preserving_mounting(
    all_vertices: np.ndarray,
    faces_out: np.ndarray,
    fixed_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Drop unreferenced vertices while keeping fixed mounting indices ``0..fixed_count-1`` intact."""
    if len(faces_out) == 0:
        return all_vertices, faces_out
    referenced = np.unique(np.concatenate([np.arange(fixed_count, dtype=np.int64), faces_out.reshape(-1)]))
    compact = np.full(len(all_vertices), -1, dtype=np.int64)
    compact[referenced] = np.arange(len(referenced), dtype=np.int64)
    return all_vertices[referenced], compact[faces_out]


def _nondegenerate(faces: np.ndarray) -> np.ndarray:
    return (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])


def _ear_clip_2d(pts: np.ndarray) -> list[tuple[int, int, int]]:
    """Triangulate a simple 2D polygon loop (convex or concave) by ear clipping."""
    n = len(pts)
    if n < 3:
        return []
    if n == 3:
        return [(0, 1, 2)]
    area2 = float(np.sum(pts[:, 0] * np.roll(pts[:, 1], -1) - np.roll(pts[:, 0], -1) * pts[:, 1]))
    rem = list(range(n)) if area2 >= 0.0 else list(reversed(range(n)))
    tris: list[tuple[int, int, int]] = []

    def _cross2(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))

    def _in_tri(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> bool:
        c1 = _cross2(a, b, p)
        c2 = _cross2(b, c, p)
        c3 = _cross2(c, a, p)
        return c1 >= -1e-12 and c2 >= -1e-12 and c3 >= -1e-12

    while len(rem) > 3:
        m = len(rem)
        ear_found = False
        best_idx = 0
        best_cross = -float("inf")
        for i in range(m):
            ip, ic, in_ = rem[(i - 1) % m], rem[i], rem[(i + 1) % m]
            a, b, c = pts[ip], pts[ic], pts[in_]
            cr = _cross2(a, b, c)
            if cr > best_cross:
                best_cross = cr
                best_idx = i
            if cr <= 1e-14:
                continue
            blocked = False
            for j in range(m):
                ij = rem[j]
                if ij in (ip, ic, in_):
                    continue
                if _in_tri(pts[ij], a, b, c):
                    blocked = True
                    break
            if not blocked:
                tris.append((ip, ic, in_))
                rem.pop(i)
                ear_found = True
                break
        if not ear_found:
            # Degenerate/collinear fallback: clip the most convex remaining corner.
            ip, ic, in_ = rem[(best_idx - 1) % m], rem[best_idx], rem[(best_idx + 1) % m]
            tris.append((ip, ic, in_))
            rem.pop(best_idx)
    tris.append((rem[0], rem[1], rem[2]))
    return tris


def _closed_edge_loops(edge_array: np.ndarray) -> list[list[int]]:
    """Walk boundary edges into closed vertex loops (open chains are dropped)."""
    edges = np.asarray(edge_array, dtype=np.int64)
    if edges.ndim != 2 or edges.shape[1] != 2 or edges.size == 0:
        return []
    adjacency: dict[int, list[int]] = {}
    for a, b in edges.tolist():
        if a == b:
            continue
        adjacency.setdefault(a, []).append(b)
        adjacency.setdefault(b, []).append(a)

    def key(i: int, j: int) -> tuple[int, int]:
        return (i, j) if i < j else (j, i)

    loops: list[list[int]] = []
    visited: set[tuple[int, int]] = set()
    for start, neighbors in adjacency.items():
        for first in neighbors:
            if key(start, first) in visited:
                continue
            loop: list[int] = [start]
            prev, curr = start, first
            visited.add(key(start, first))
            for _ in range(max(len(edges) + 2, 8)):
                loop.append(curr)
                candidates = [n for n in adjacency.get(curr, []) if n != prev]
                if curr == start:
                    break
                if not candidates:
                    loop = []
                    break
                nxt = candidates[0]
                for candidate in candidates:
                    if key(curr, candidate) not in visited:
                        nxt = candidate
                        break
                if key(curr, nxt) in visited and nxt != start:
                    loop = []
                    break
                visited.add(key(curr, nxt))
                prev, curr = curr, nxt
            if len(loop) >= 4 and loop[0] == loop[-1]:
                loops.append(loop[:-1])
    return loops
