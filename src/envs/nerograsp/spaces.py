"""Typed full-mesh embodiments and their Gymnasium space."""

from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
from typing import Any

import numpy as np
from gymnasium.spaces import Space


@dataclass
class Mesh:
    """Full fingertip geometry: XYZ vertices in metres and triangle indices."""

    vertices: np.ndarray
    faces: np.ndarray

    def copy(self) -> Mesh:
        return Mesh(np.array(self.vertices, copy=True), np.array(self.faces, copy=True))


def _oriented_face(face) -> tuple[int, int, int]:
    a, b, c = map(int, face)
    return min((a, b, c), (b, c, a), (c, a, b))


class MeshSpace(Space):
    """One design dictionary of full meshes with immutable indexed mountings.

    Each mesh starts with its protected vertices. Mesh sizes may vary outside
    that prefix. Membership checks structure, mounting faces, watertightness,
    and face connectivity; environment-specific rejectors run separately.
    """

    def __init__(self, base_meshes: dict[str, Mesh], mounting_vertex_counts: dict[str, int], *, sample_perturbation_m: float = 0.0001):
        super().__init__()
        if not np.isfinite(sample_perturbation_m) or sample_perturbation_m < 0:
            raise ValueError("Sampling perturbation must be finite and nonnegative")
        self.sample_perturbation_m = float(sample_perturbation_m)
        if base_meshes.keys() != mounting_vertex_counts.keys():
            raise ValueError("Every mesh needs a mounting vertex count")
        self._base_meshes = {name: mesh.copy() for name, mesh in base_meshes.items()}
        self._mounting_vertex_counts = dict(mounting_vertex_counts)
        self._mounting_faces = {}
        self._mounting_face_sets = {}
        self._boundary_edges = {}
        self._boundary_edge_pairs = {}
        for name, mesh in self._base_meshes.items():
            count = self._mounting_vertex_counts[name]
            if not 0 < count < len(mesh.vertices):
                raise ValueError("Mounting vertex count must leave a replaceable region")
            faces = np.asarray(mesh.faces)
            fixed = faces[np.all(faces < count, axis=1)].copy()
            self._mounting_faces[name] = fixed
            self._mounting_face_sets[name] = Counter(_oriented_face(face) for face in fixed)
            edges = Counter(tuple(sorted((int(face[i]), int(face[(i + 1) % 3])))) for face in fixed for i in range(3))
            b_edges = np.asarray([edge for edge, uses in edges.items() if uses == 1], dtype=np.int64).reshape(-1, 2)
            self._boundary_edges[name] = b_edges
            self._boundary_edge_pairs[name] = (b_edges[:, 0].copy(), b_edges[:, 1].copy()) if len(b_edges) else (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64))
        reasons = self.rejection_reasons(self._base_meshes)
        if reasons:
            raise ValueError("Invalid canonical mesh: " + "; ".join(reasons))

    def __repr__(self) -> str:
        meshes = ", ".join(
            f"{name!r}: (base_vertices={len(mesh.vertices)}, "
            f"base_faces={len(mesh.faces)}, mounting_vertices={self._mounting_vertex_counts[name]})"
            for name, mesh in self._base_meshes.items()
        )
        return f"MeshSpace({{{meshes}}}, sample_perturbation_m={self.sample_perturbation_m!r})"

    @property
    def base_meshes(self) -> dict[str, Mesh]:
        return {name: mesh.copy() for name, mesh in self._base_meshes.items()}

    @property
    def mounting_vertex_counts(self) -> dict[str, int]:
        return dict(self._mounting_vertex_counts)

    @property
    def mounting_faces(self) -> dict[str, np.ndarray]:
        return {name: faces.copy() for name, faces in self._mounting_faces.items()}

    @property
    def boundary_edges(self) -> dict[str, np.ndarray]:
        return {name: edges.copy() for name, edges in self._boundary_edges.items()}

    def contains(self, value: Any) -> bool:
        if isinstance(value, (list, tuple)):
            return len(value) == getattr(self, "batch_size", 1) and all(not self.rejection_reasons(design) for design in value)
        return not self.rejection_reasons(value)

    def sample(self, rng: np.random.Generator | None = None) -> dict[str, Mesh]:
        """Perturb mutable XYZ coordinates uniformly; runtime rejectors remain separate."""
        rng = self.np_random if rng is None else rng
        design = self.base_meshes
        for name, mesh in design.items():
            mesh.vertices = mesh.vertices.astype(np.float64)
            count = self._mounting_vertex_counts[name]
            mesh.vertices[count:] += rng.uniform(-self.sample_perturbation_m, self.sample_perturbation_m, mesh.vertices[count:].shape)
        return design

    def rejection_reasons(self, value: Any) -> tuple[str, ...]:
        if not isinstance(value, dict) or value.keys() != self._base_meshes.keys():
            return ("Expected a design dictionary containing " + ", ".join(self._base_meshes),)
        reasons = []
        for name, mesh in value.items():
            try:
                self._validate_mesh(name, mesh)
            except (ValueError, TypeError, OverflowError) as error:
                reasons.append(f"{name}: {error}")
        return tuple(reasons)

    def _validate_mesh(self, name: str, mesh: Mesh) -> None:
        if not isinstance(mesh, Mesh):
            raise TypeError("Expected Mesh(vertices, faces)")
        base = self._base_meshes[name]
        vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
        if (
            vertices.shape == base.vertices.shape
            and faces.shape == base.faces.shape
            and np.issubdtype(faces.dtype, np.integer)
            and np.array_equal(faces, base.faces)
        ):
            # Fixed-topology fast path: base.faces is already verified watertight, manifold,
            # consistently wound, connected, and attached along mounting edges in __init__.
            if not (np.issubdtype(vertices.dtype, np.floating) or np.issubdtype(vertices.dtype, np.integer)) or not np.isfinite(vertices).all():
                raise ValueError("Vertices must be finite real coordinates")
            count = self._mounting_vertex_counts[name]
            if not np.array_equal(vertices[:count], base.vertices[:count]):
                raise ValueError("Mounting vertices must remain exactly unchanged at their fixed indices")
            tris = vertices.astype(np.float64)[faces]
            signed_vol = float(np.sum(np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])[:, 0] * (tris[:, 0, 0] + tris[:, 1, 0] + tris[:, 2, 0])) / 6.0)
            if not np.isfinite(signed_vol) or signed_vol <= 0.0:
                raise ValueError("Mesh must have positive outward-oriented volume")
            return
        vertices, faces = self._check_mesh_arrays(name, mesh)
        protected = self._check_mounting_preserved(name, vertices, faces)
        self._check_watertight_manifold(name, vertices, faces, protected)

    def _check_mesh_arrays(self, name: str, mesh: Mesh) -> tuple[np.ndarray, np.ndarray]:
        """Verify vertex/face shapes, numeric dtypes, valid indices, and no duplicate or degenerate triangles."""
        if not isinstance(mesh, Mesh):
            raise TypeError("Expected Mesh(vertices, faces)")
        vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
        count = self._mounting_vertex_counts[name]
        if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) <= count:
            raise ValueError("Expected vertices of shape (N, 3) including mounting and replacement vertices")
        if not (np.issubdtype(vertices.dtype, np.floating) or np.issubdtype(vertices.dtype, np.integer)) or not np.isfinite(vertices).all():
            raise ValueError("Vertices must be finite real coordinates")
        if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0 or not np.issubdtype(faces.dtype, np.integer):
            raise ValueError("Expected integer triangle indices of shape (F, 3)")
        if faces.min() < 0 or faces.max() >= len(vertices):
            raise ValueError("Triangle index outside the vertex array")
        if np.any(np.diff(np.sort(faces, axis=1), axis=1) == 0):
            raise ValueError("Triangles must have three distinct vertex indices")
        if len(np.unique(np.sort(faces, axis=1), axis=0)) != len(faces):
            raise ValueError("Duplicate triangles are not allowed")
        if len(np.unique(faces)) != len(vertices):
            raise ValueError("Unreferenced vertices are not allowed")
        return vertices, faces

    def _check_mounting_preserved(self, name: str, vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
        """Ensure the fixed mounting seam vertices and faces match the canonical fingertip exactly."""
        count = self._mounting_vertex_counts[name]
        if not np.array_equal(vertices[:count], self._base_meshes[name].vertices[:count]):
            raise ValueError("Mounting vertices must remain exactly unchanged at their fixed indices")
        protected = np.all(faces < count, axis=1)
        if Counter(_oriented_face(face) for face in faces[protected]) != self._mounting_face_sets[name]:
            raise ValueError("Protected mounting faces must remain unchanged")
        return protected

    def _check_watertight_manifold(self, name: str, vertices: np.ndarray, faces: np.ndarray, protected: np.ndarray) -> None:
        """Verify closed watertight manifold, consistent outward winding, seam attachment, and single component."""
        n_verts = int(len(vertices))
        n_faces = int(len(faces))
        faces_i64 = np.asarray(faces, dtype=np.int64)
        u = faces_i64[:, [0, 1, 2]].reshape(-1)
        v = faces_i64[:, [1, 2, 0]].reshape(-1)
        dir_keys = u * n_verts + v
        dir_sorted = np.sort(dir_keys)
        if dir_sorted.size > 1 and np.any(dir_sorted[1:] == dir_sorted[:-1]):
            raise ValueError("Mesh faces must have consistent orientation (winding)")

        lo = np.minimum(u, v)
        hi = np.maximum(u, v)
        undir_keys = lo * n_verts + hi
        order = np.argsort(undir_keys)
        sorted_undir = undir_keys[order]
        if (sorted_undir.size & 1) != 0 or not np.all(sorted_undir[0::2] == sorted_undir[1::2]):
            raise ValueError("Mesh must be watertight: every edge must have exactly two incident faces")
        if sorted_undir.size > 2 and np.any(sorted_undir[1:-1:2] == sorted_undir[2::2]):
            raise ValueError("Mesh must be watertight: every edge must have exactly two incident faces")

        tris = vertices.astype(np.float64)[faces_i64]
        signed_vol = float(np.sum(np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])[:, 0] * (tris[:, 0, 0] + tris[:, 1, 0] + tris[:, 2, 0])) / 6.0)
        if not np.isfinite(signed_vol) or signed_vol <= 0.0:
            raise ValueError("Mesh must have positive outward-oriented volume")

        face_ids = np.repeat(np.arange(n_faces, dtype=np.int64), 3)[order]
        f0 = face_ids[0::2]
        f1 = face_ids[1::2]
        unique_undir = sorted_undir[0::2]

        b_lo, b_hi = self._boundary_edge_pairs[name]
        if b_lo.size > 0:
            b_keys = b_lo * n_verts + b_hi
            pos = np.searchsorted(unique_undir, b_keys)
            if np.any(pos >= unique_undir.size) or np.any(unique_undir[pos] != b_keys):
                raise ValueError("Replacement must attach along every mounting boundary edge")
            if not np.all(protected[f0[pos]] != protected[f1[pos]]):
                raise ValueError("Replacement must attach along every mounting boundary edge")

        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components

        data = np.ones(f0.size, dtype=np.int8)
        graph = coo_matrix((data, (f0, f1)), shape=(n_faces, n_faces))
        n_comp, _ = connected_components(graph, directed=False, return_labels=True)
        if n_comp != 1:
            raise ValueError("Mesh must be connected to the mounting region; detached components are rejected")
