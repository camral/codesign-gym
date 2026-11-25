"""Minimal standalone example: reparameterizing NeroGrasp's mesh embodiment with CBD-SA (+ optional SVD).

NeroGrasp's native embodiment is a dict of raw triangle meshes (``{"index": Mesh, "thumb": Mesh}``).
This script wraps that interface with Cage-Based Deformation + Structural Affinity
(``structural_affinity``) using the same lofted ring cages as ``examples/nerograsp_fasttd3_ngopt/train.py``
(``ring_10x12`` inner + ``ring_6x8`` outer), plus an optional SVD reduction step to compress the
CBD-SA Jacobian into a compact low-dimensional basis for black-box / RL optimizers.

Requirements:
    pip install -e .
    pip install git+https://github.com/camral/structural-affinity.git

Run:
    python examples/nerograsp_cbdsa_reparam.py
"""

from __future__ import annotations

import numpy as np
import trimesh
from structural_affinity import affinity_matrix, mvc_3d, propagation_operator, sparsify

import codesign_gym
from envs.nerograsp.config import MOUNTING_VERTEX_COUNT
from envs.nerograsp.spaces import Mesh


def make_ring_cage(
    verts: np.ndarray, n_rings: int = 6, n_sides: int = 8, clearance: float = 0.0012
) -> tuple[np.ndarray, np.ndarray]:
    """Lofted cross-section ring cage along +y, strictly encapsulating ``verts``."""
    y_min, y_max = float(verts[:, 1].min()), float(verts[:, 1].max())
    y_levels = np.linspace(y_min - clearance, y_max + clearance, n_rings)
    dy = (y_max - y_min) / max(n_rings - 1, 1)
    angles = np.linspace(0.0, 2.0 * np.pi, n_sides, endpoint=False)
    dirs = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    chord_scale = 1.0 / np.cos(np.pi / n_sides)

    rings, centers = [], []
    for y_r in y_levels:
        slab = np.abs(verts[:, 1] - y_r) <= max(0.75 * dy, clearance * 2.0)
        sub = verts[slab] if slab.sum() >= 3 else verts
        cx, cz = float(sub[:, 0].mean()), float(sub[:, 2].mean())
        centers.append((cx, cz))
        rad = np.maximum(((sub[:, [0, 2]] - np.array([cx, cz])) @ dirs.T).max(axis=0), 1e-4) * chord_scale + clearance
        rings.append(np.column_stack([cx + rad * dirs[:, 0], np.full(n_sides, y_r), cz + rad * dirs[:, 1]]))

    poles = np.array([
        [centers[0][0], y_min - 1.5 * clearance, centers[0][1]],
        [centers[-1][0], y_max + 1.5 * clearance, centers[-1][1]],
    ])
    c_verts = np.vstack(rings + [poles])
    bot_pole, top_pole = n_rings * n_sides, n_rings * n_sides + 1

    faces = []
    for r in range(n_rings - 1):
        for s in range(n_sides):
            sn = (s + 1) % n_sides
            a, b, c, d = r * n_sides + s, r * n_sides + sn, (r + 1) * n_sides + s, (r + 1) * n_sides + sn
            faces.extend(([a, c, b], [b, c, d]))
    for s in range(n_sides):
        sn = (s + 1) % n_sides
        top_r = (n_rings - 1) * n_sides
        faces.extend(([bot_pole, s, sn], [top_pole, top_r + sn, top_r + s]))
    c_faces = np.asarray(faces, dtype=np.int64)
    if trimesh.Trimesh(vertices=c_verts, faces=c_faces, process=False).volume < 0:
        c_faces = c_faces[:, [0, 2, 1]]

    center = verts.mean(axis=0)
    for _ in range(15):
        if trimesh.Trimesh(vertices=c_verts, faces=c_faces, process=False).contains(verts).all():
            break
        c_verts = center + (c_verts - center) * 1.05
    return c_verts, c_faces


class NeroGraspCBDSA:
    """Maps a flat vector to a NeroGrasp ``dict[str, Mesh]`` embodiment via CBD-SA (+ optional SVD)."""

    def __init__(
        self,
        base_embodiment: dict[str, Mesh],
        *,
        alpha: float = 0.5,
        density: float = 0.35,
        normalize: bool = True,
        svd_modes: int | None = None,
        mounting_count: int = MOUNTING_VERTEX_COUNT,
    ):
        self.tip_names = sorted(base_embodiment)
        self.base = base_embodiment
        self.mounting_count = mounting_count
        self.operators: dict[str, np.ndarray] = {}
        self.mode_counts: dict[str, int] = {}

        for name in self.tip_names:
            verts = np.asarray(base_embodiment[name].vertices, dtype=np.float64)
            inner_v, inner_f = make_ring_cage(verts, n_rings=10, n_sides=12, clearance=0.0010)
            outer_v, outer_f = make_ring_cage(inner_v, n_rings=6, n_sides=8, clearance=0.0030)

            W_mesh = mvc_3d(verts, inner_v, inner_f)
            W_lu = np.maximum(mvc_3d(inner_v, outer_v, outer_f), 0.0)
            W_lu /= np.maximum(W_lu.sum(axis=1, keepdims=True), 1e-12)
            C_aff = sparsify(affinity_matrix(W_lu), density=density)
            A_op = propagation_operator(C_aff, alpha, normalize=normalize)

            # Combined linear map: cage handle displacements (N_inner, 3) -> deformable vertex offsets
            J = W_mesh[mounting_count:] @ A_op.toarray()

            # Optional SVD reduction: compress N_inner handles -> top-r dominant deformation modes
            if svd_modes is not None and svd_modes > 0:
                r = min(int(svd_modes), J.shape[1])
                _, _, vt = np.linalg.svd(J, full_matrices=False)
                V_r = vt[:r].T
                V_r = V_r / np.maximum(np.max(np.abs(V_r), axis=0, keepdims=True), 1e-8)
                J = J @ V_r

            self.operators[name] = J
            self.mode_counts[name] = J.shape[1]

    @property
    def param_dim(self) -> int:
        return sum(k * 3 for k in self.mode_counts.values())

    def decode(self, theta: np.ndarray) -> dict[str, Mesh]:
        """Convert flat vector ``theta`` (metres) into a NeroGrasp mesh embodiment."""
        theta = np.asarray(theta, dtype=np.float64).reshape(-1)
        out: dict[str, Mesh] = {}
        offset = 0
        for name in self.tip_names:
            k = self.mode_counts[name]
            D = theta[offset : offset + 3 * k].reshape(k, 3)
            offset += 3 * k

            base_mesh = self.base[name]
            verts = np.asarray(base_mesh.vertices, dtype=np.float64).copy()
            verts[self.mounting_count :] += self.operators[name] @ D
            out[name] = Mesh(vertices=verts, faces=np.asarray(base_mesh.faces, dtype=np.int64))
        return out


def main() -> None:
    env, base_emb = codesign_gym.make("NeroGrasp-Thin", use_cpu=True, snap_plane_clamp=True)

    full_reparam = NeroGraspCBDSA(base_emb, alpha=0.5, density=0.35, normalize=True, svd_modes=None)
    svd_reparam = NeroGraspCBDSA(base_emb, alpha=0.5, density=0.35, normalize=True, svd_modes=4)
    print(f"CBD-SA full handles: {full_reparam.param_dim}D -> SVD(4) reduced: {svd_reparam.param_dim}D")

    rng = np.random.default_rng(0)
    theta = rng.normal(0.0, 0.0005, size=svd_reparam.param_dim)
    emb = svd_reparam.decode(theta)

    ok = env.validate_embodiment(emb)
    print(f"validate_embodiment: ok={ok}")
    if not ok:
        emb = base_emb

    obs, rejected, info = env.reset(embodiment=emb)
    obs_desc = {k: tuple(v.shape) for k, v in obs.items()} if isinstance(obs, dict) else obs.shape
    print(f"reset: rejected={rejected}, obs={obs_desc}")

    for step in range(5):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        print(f"step {step}: reward={float(reward):.4f}")
        if terminated or truncated:
            break

    env.close()


if __name__ == "__main__":
    main()
