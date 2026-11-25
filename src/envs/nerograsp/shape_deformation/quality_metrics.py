"""Geometric quality metrics for postprocessed fingertip meshes, reported in ``info``."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import numpy as np

QUALITY_METRIC_KEYS: tuple[str, ...] = (
    "mesh_normal_consistency",
    "mesh_normal_consistency_p95",
    "mesh_laplacian_smoothing",
    "mesh_laplacian_smoothing_p95",
    "mesh_triangle_shape",
    "mesh_edge_length_cv",
)


def compute_final_mesh_quality_metrics(vertices: np.ndarray, faces: np.ndarray) -> dict[str, float]:
    """Compute smoothness/regularity metrics on a postprocessed triangle mesh."""
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    if f.size == 0 or v.size == 0:
        return {k: 0.0 for k in QUALITY_METRIC_KEYS}

    edges = np.unique(np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], axis=0), axis=1), axis=0)
    edge_lengths = np.linalg.norm(v[edges[:, 0]] - v[edges[:, 1]], axis=1)
    normal_cons = _face_normal_consistency(v, f)
    lap = _uniform_laplacian(v, edges)
    shape = _triangle_shape(v, f)

    mean_e = float(edge_lengths.mean()) if edge_lengths.size else 0.0
    return {
        "mesh_normal_consistency": float(normal_cons.mean()) if normal_cons.size else 0.0,
        "mesh_normal_consistency_p95": float(np.percentile(normal_cons, 95.0)) if normal_cons.size else 0.0,
        "mesh_laplacian_smoothing": float(lap.mean()) if lap.size else 0.0,
        "mesh_laplacian_smoothing_p95": float(np.percentile(lap, 95.0)) if lap.size else 0.0,
        "mesh_triangle_shape": float(shape.mean()) if shape.size else 0.0,
        "mesh_edge_length_cv": float(edge_lengths.std() / mean_e) if mean_e > 1e-12 else 0.0,
    }


def summarize_quality_records(
    records: Sequence[Mapping[str, object]],
    *,
    prefix: str = "quality",
    metric_keys: Iterable[str] = QUALITY_METRIC_KEYS,
) -> dict[str, float]:
    """Return mean/std summaries for numeric metric fields."""
    summary: dict[str, float] = {}
    for key in metric_keys:
        vals = [float(r[key]) for r in records if key in r and r[key] is not None and np.isfinite(float(r[key]))]
        if vals:
            arr = np.asarray(vals, dtype=np.float64)
            summary[f"{prefix}/{key}_mean"] = float(arr.mean())
            summary[f"{prefix}/{key}_std"] = float(arr.std())
    return summary


def _face_normal_consistency(v: np.ndarray, f: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n_verts = int(len(v))
    n_faces = int(len(f))
    if n_faces < 2 or n_verts == 0:
        return np.zeros((0,), dtype=np.float64)
    u = f[:, [0, 1, 2]].reshape(-1)
    w = f[:, [1, 2, 0]].reshape(-1)
    valid_e = u != w
    u, w = u[valid_e], w[valid_e]
    fids = np.repeat(np.arange(n_faces, dtype=np.int64), 3)[valid_e]
    keys = np.minimum(u, w) * n_verts + np.maximum(u, w)
    order = np.argsort(keys)
    skeys = keys[order]
    sfids = fids[order]
    # Find runs of length exactly 2
    changes = np.empty(skeys.size + 1, dtype=bool)
    changes[0] = True
    changes[-1] = True
    changes[1:-1] = skeys[1:] != skeys[:-1]
    starts = np.flatnonzero(changes[:-1])
    lengths = np.diff(np.flatnonzero(changes))
    pair_starts = starts[lengths == 2]
    if pair_starts.size == 0:
        return np.zeros((0,), dtype=np.float64)
    fa = sfids[pair_starts]
    fb = sfids[pair_starts + 1]
    raw = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    normals = np.divide(raw, np.maximum(norms, eps))
    na, nb = normals[fa], normals[fb]
    valid = (norms[fa, 0] > eps) & (norms[fb, 0] > eps)
    if not valid.any():
        return np.zeros((0,), dtype=np.float64)
    cos = np.einsum("ij,ij->i", na[valid], nb[valid])
    return 1.0 - np.clip(cos, -1.0, 1.0)


def _uniform_laplacian(v: np.ndarray, edges: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    if edges.size == 0:
        return np.zeros((0,), dtype=np.float64)
    nb_sum = np.zeros_like(v)
    deg = np.zeros(len(v), dtype=np.float64)
    np.add.at(nb_sum, edges[:, 0], v[edges[:, 1]])
    np.add.at(nb_sum, edges[:, 1], v[edges[:, 0]])
    np.add.at(deg, edges[:, 0], 1.0)
    np.add.at(deg, edges[:, 1], 1.0)
    valid = deg > 0.0
    if not valid.any():
        return np.zeros((0,), dtype=np.float64)
    diag = float(np.linalg.norm(v.max(axis=0) - v.min(axis=0)))
    scale = diag if np.isfinite(diag) and diag > eps else 1.0
    return np.linalg.norm(nb_sum[valid] / deg[valid, None] - v[valid], axis=1) / scale


def _triangle_shape(v: np.ndarray, f: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v0, v1, v2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    denom = np.sum((v1 - v0) ** 2 + (v2 - v1) ** 2 + (v0 - v2) ** 2, axis=1)
    q = (2.0 * np.sqrt(3.0) * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)) / np.maximum(denom, eps)
    return np.clip(np.where(np.isfinite(q), q, 0.0), 0.0, 1.0)
