"""Save and load NeroGrasp-Tools tutorial checkpoints with precomputed fingertip decompositions.

A bundle is a pair of files in one folder:

    <name>.pt               FastTD3 + CMA-ES policy snapshot, copied unchanged from training
    <name>_fingertips.npz   the base and co-designed fingertip meshes, the CBD-SA design vector the
                            policy is conditioned on, the CBD-SA mode matrices, the training command,
                            and the convex parts of every fingertip mesh the env needs

MuJoCo collides each fingertip as a set of convex parts, so a new fingertip mesh normally goes
through a convex decomposition backend (cuacd, vhacd or visacd). Building an env inside
``precomputed_decompositions(...)`` replaces that step with a lookup in the loaded bundles, so the
stored designs can be simulated on any machine without a decomposition backend. Any other design
raises an error.
"""

from __future__ import annotations

import contextlib
import functools
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import warp as wp

import envs.nerograsp.base_env as _base_env
from codesign_gym import make_vec
from envs.nerograsp import Mesh
from envs.nerograsp.shape_deformation.convex_decomposition import (
    ConvexDecomposer,
    ConvexPart,
    _geometry_key,
)

# One stored mesh: post-processed vertices (float64), faces (int32) and its convex parts.
MeshEntry = tuple[np.ndarray, np.ndarray, list[ConvexPart]]


@dataclass
class FingertipBundle:
    env_id: str
    theta: np.ndarray               # (24,) CBD-SA coefficients in metres; the policy observes theta / mesh_bound_m
    mesh_bound_m: float             # CMA-ES search bound on every coefficient
    design: dict[str, Mesh]         # co-designed {"left", "right"} fingertips, ready for env.reset
    base: dict[str, Mesh]           # base fingertips (theta = 0)
    operators: dict[str, np.ndarray]  # CBD-SA mode matrix J per fingertip: (replaceable vertices, modes)
    meshes: list[MeshEntry]         # every decomposed mesh (base and co-designed)
    backend: str                    # decomposition backend that produced the parts
    train_command: str | None       # codesign-baselines command that trained the checkpoint
    checkpoint: Path                # policy snapshot next to the .npz

    def coefficients(self, theta=None) -> dict[str, np.ndarray]:
        """Split a design vector into per-fingertip ``(modes, 3)`` coefficient matrices (metres)."""
        theta = self.theta if theta is None else np.asarray(theta)
        out, start = {}, 0
        for name in sorted(self.operators):
            count = self.operators[name].shape[1] * 3
            out[name] = theta[start:start + count].reshape(-1, 3)
            start += count
        return out


def deform(base_mesh: Mesh, operator: np.ndarray, coefficients) -> Mesh:
    """Apply CBD-SA coefficients ``(modes, 3)`` in metres: ``vertices[mounting:] += operator @ coefficients``."""
    vertices = np.array(base_mesh.vertices, dtype=np.float64)
    vertices[len(vertices) - len(operator):] += operator @ np.asarray(coefficients, dtype=np.float64).reshape(-1, 3)
    return Mesh(vertices, np.array(base_mesh.faces))


# --------------------------------------------------------------------------------------------- #
# Env, policy and rollouts
# --------------------------------------------------------------------------------------------- #
def make_tools_env(env_id, num_envs, *, device="cpu", seed=0, backend=None, render_mode=None, noise=True):
    """Raw NeroGrasp-Tools vector env configured like the training evaluator (``is_eval_env=True``).

    With ``noise=False`` the tool pose and gripper command are not randomized.
    """
    kwargs = {"seed": seed, "is_eval_env": True, "render_mode": render_mode}
    kwargs.update({"use_cpu": True} if str(device) == "cpu" else {"device": str(device)})
    if backend is not None:
        kwargs["convex_decomposition_backend"] = backend
    if not noise:
        kwargs.update(object_position_randomization=0.0, object_rotation_randomization_deg=0.0, gripper_action_noise_std=0.0)
    return make_vec(env_id, num_envs, **kwargs)


def load_tools_policy(path, device="cpu"):
    try:
        from fasttd3_cmaes.normalization import load_policy
    except ImportError as error:
        raise ImportError(
            "codesign-baselines is not installed. See Section 1 of notebooks/morphhand_tutorial.ipynb."
        ) from error

    payload, policy = load_policy(str(path), str(device), compile=False)
    return payload, policy


def design_observation(theta, mesh_bound_m, num_envs, device):
    """The design features the policy sees, scaled to [-1, 1] (same as ``condition_observation``)."""
    scaled = torch.as_tensor(np.asarray(theta) / mesh_bound_m, dtype=torch.float32, device=device).clamp(-1.0, 1.0)
    return scaled.expand(num_envs, -1) if scaled.ndim == 1 else scaled


@torch.no_grad()
def rollout(env, policy, design, theta, mesh_bound_m, *, seed, record=(), record_every=4, frame_scale=1):
    """One synchronized episode in every world.

    ``design`` is one fingertip dict for every world or a list with one per world; ``theta`` is the matching
    ``(24,)`` or ``(num_envs, 24)`` design input for the policy. Worlds in ``record`` are rendered every
    ``record_every`` steps and kept downscaled by ``frame_scale`` (full 1920x1080 frames are 6 MB each).
    Returns per-world results, per-step traces ``(steps, worlds)`` and frames.
    """

    def render(i):
        return np.ascontiguousarray(env.render(env_idx=i)[::frame_scale, ::frame_scale])

    designs = list(design) if isinstance(design, (list, tuple)) else [design] * env.num_embodiments
    torch.manual_seed(seed)
    obs, rejected, info = env.reset(seed=seed, embodiment=designs)
    if bool(torch.as_tensor(rejected).any()):
        raise RuntimeError(f"design rejected: {info['embodiment/rejection_reasons'][0]}")
    design_obs = design_observation(theta, mesh_bound_m, env.num_envs, obs["actor"].device)
    record = list(record)
    frames = {i: [render(i)] for i in record}
    traces = {"articulation": [], "lift_cm": [], "reward": []}
    step = 0
    while True:
        action = policy(torch.cat([obs["actor"], design_obs], dim=-1), deterministic=True, update=False).clamp(-1, 1)
        obs, reward, _, _, info = env.step(action)
        step += 1
        height = wp.to_torch(env.cp_data.xpos)[:, env.obj_body_id, 2]
        if step == 1:
            rest_z = height.clone()  # the tool settles onto the floor during the first step
        lift = height - rest_z
        traces["articulation"].append(info["task/articulation_ratio"].cpu().numpy())
        traces["lift_cm"].append(lift.cpu().numpy() * 100.0)
        traces["reward"].append(reward.cpu().numpy())
        finished = bool(info["finished"].all())
        if step % record_every == 0 or finished:
            for i in record:
                frames[i].append(render(i))
        if finished:
            break
    traces = {key: np.stack(values) for key, values in traces.items()}
    return {
        "returns": info["episode/final_return"].cpu().numpy(),
        "success": env.last_held_at_goal.cpu().numpy().astype(bool),
        "articulation": traces["articulation"][-1],
        "max_lift_cm": traces["lift_cm"].max(axis=0),
        "noise": env.noise_multiplier_t.squeeze(-1).cpu().numpy(),
        "traces": traces,
        "frames": frames,
    }


# --------------------------------------------------------------------------------------------- #
# Precomputed decompositions
# --------------------------------------------------------------------------------------------- #
class PrecomputedDecomposer(ConvexDecomposer):
    """A ``ConvexDecomposer`` that only looks parts up in stored meshes and never runs a backend.

    Lookup is by exact mesh bytes first, then by the closest stored mesh with identical faces,
    accepted when every vertex is within ``atol_m`` (absorbs last-bit differences between platforms).
    """

    def __init__(self, entries: list[MeshEntry], *, atol_m: float = 1e-7, **kwargs):
        super().__init__(**kwargs)
        self._entries = entries
        self._atol_m = float(atol_m)
        self._by_key = {_geometry_key(v, f): parts for v, f, parts in entries}

    def _run(self, meshes):
        return [self._lookup(vertices, faces, name) for vertices, faces, name in meshes]

    def _lookup(self, vertices, faces, mesh_name):
        parts = self._by_key.get(_geometry_key(vertices, faces))
        if parts is not None:
            return list(parts)
        closest = None
        for stored_v, stored_f, stored_parts in self._entries:
            if stored_v.shape == vertices.shape and np.array_equal(stored_f, faces):
                error = float(np.abs(stored_v - vertices).max())
                if closest is None or error < closest[0]:
                    closest = (error, stored_parts)
        if closest is not None and closest[0] <= self._atol_m:
            return list(closest[1])
        detail = ("no stored mesh has the same faces" if closest is None
                  else f"the closest stored mesh differs by {closest[0] * 1e3:.4f} mm")
        raise RuntimeError(
            f"No precomputed convex decomposition for {mesh_name} ({detail}). Without a decomposition "
            "backend, only the fingertips saved in the loaded bundles can be simulated."
        )


@contextlib.contextmanager
def precomputed_decompositions(*bundles: FingertipBundle, atol_m: float = 1e-7):
    """Envs built inside this block take convex parts from ``bundles`` instead of a backend.

    The env keeps its decomposer after the block ends, so later resets use the bundles too.
    """
    entries = [entry for bundle in bundles for entry in bundle.meshes]
    original = _base_env.ConvexDecomposer
    _base_env.ConvexDecomposer = functools.partial(PrecomputedDecomposer, entries, atol_m=atol_m)
    try:
        yield
    finally:
        _base_env.ConvexDecomposer = original


def harvest_decompositions(env) -> list[MeshEntry]:
    """Convex parts of the base fingertips and of the design installed in world 0, from the env's cache."""
    meshes = [env.postprocessor.postprocess(tip, tip.vertices, tip.faces) for tip in env.fingertips]
    meshes += [env.tip_meshes[0][tip.name] for tip in env.fingertips]
    entries, seen = [], set()
    for vertices, faces in meshes:
        vertices = np.ascontiguousarray(vertices, dtype=np.float64)
        faces = np.ascontiguousarray(faces, dtype=np.int32)
        key = _geometry_key(vertices, faces)
        if key in seen:
            continue
        seen.add(key)
        parts = env.decomposer._cache.get(key)
        if parts is None:
            raise RuntimeError("fingertip mesh missing from the decomposition cache; reset the env with the design first")
        entries.append((vertices, faces, list(parts)))
    return entries


# --------------------------------------------------------------------------------------------- #
# Bundle files
# --------------------------------------------------------------------------------------------- #
def bundle_paths(out_dir, name) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    return out_dir / f"{name}.pt", out_dir / f"{name}_fingertips.npz"


def save_fingertips(path, *, env_id, theta, mesh_bound_m, design, base, operators, meshes, backend,
                    train_command=None) -> None:
    arrays = {
        "env_id": np.str_(env_id),
        "theta": np.asarray(theta, dtype=np.float32),
        "mesh_bound_m": np.float64(mesh_bound_m),
        "backend": np.str_(backend),
        "num_meshes": np.int64(len(meshes)),
    }
    if train_command is not None:
        arrays["train_command"] = np.str_(train_command)
    for prefix, tips in (("design", design), ("base", base)):
        for name, mesh in tips.items():
            arrays[f"{prefix}__{name}__vertices"] = np.asarray(mesh.vertices, dtype=np.float64)
            arrays[f"{prefix}__{name}__faces"] = np.asarray(mesh.faces, dtype=np.int64)
    for name, operator in operators.items():
        arrays[f"operator__{name}"] = np.asarray(operator, dtype=np.float64)
    for i, (vertices, faces, parts) in enumerate(meshes):
        arrays[f"mesh{i}__vertices"] = vertices
        arrays[f"mesh{i}__faces"] = faces
        arrays[f"mesh{i}__num_parts"] = np.int64(len(parts))
        for j, part in enumerate(parts):
            arrays[f"mesh{i}__part{j}__vertices"] = np.asarray(part.vertices, dtype=np.float64)
            arrays[f"mesh{i}__part{j}__faces"] = np.asarray(part.faces, dtype=np.int32)
    np.savez_compressed(path, **arrays)


def _frozen(array, dtype):
    out = np.ascontiguousarray(array, dtype=dtype)
    out.setflags(write=False)
    return out


def _tip_meshes(data, prefix) -> dict[str, Mesh]:
    names = sorted({key.split("__")[1] for key in data.files if key.startswith(f"{prefix}__")})
    return {name: Mesh(np.array(data[f"{prefix}__{name}__vertices"]), np.array(data[f"{prefix}__{name}__faces"]))
            for name in names}


def load_bundle(out_dir, name) -> FingertipBundle:
    checkpoint, fingertips = bundle_paths(out_dir, name)
    with np.load(fingertips, allow_pickle=False) as data:
        meshes = []
        for i in range(int(data["num_meshes"])):
            parts = [
                ConvexPart(vertices=_frozen(data[f"mesh{i}__part{j}__vertices"], np.float64),
                           faces=_frozen(data[f"mesh{i}__part{j}__faces"], np.int32))
                for j in range(int(data[f"mesh{i}__num_parts"]))
            ]
            meshes.append((np.ascontiguousarray(data[f"mesh{i}__vertices"], dtype=np.float64),
                           np.ascontiguousarray(data[f"mesh{i}__faces"], dtype=np.int32), parts))
        operators = {key.split("__")[1]: np.array(data[key]) for key in data.files if key.startswith("operator__")}
        return FingertipBundle(
            env_id=str(data["env_id"]),
            theta=np.array(data["theta"], dtype=np.float32),
            mesh_bound_m=float(data["mesh_bound_m"]),
            design=_tip_meshes(data, "design"),
            base=_tip_meshes(data, "base"),
            operators=operators,
            meshes=meshes,
            backend=str(data["backend"]),
            train_command=str(data["train_command"]) if "train_command" in data.files else None,
            checkpoint=checkpoint,
        )
