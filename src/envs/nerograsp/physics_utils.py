"""Stateless MuJoCo collision, visual coloring, and batched tensor geometry helpers."""

from __future__ import annotations

import colorsys

import mujoco
import torch

# Collision group bitmasks shared by NeroGrasp and NeroTools.
BIT_FLOOR = 1 << 0
BIT_OBJECT = 1 << 1
BIT_HAND = 1 << 2
BIT_TIP = 1 << 3
BIT_ARM = 1 << 4

ARM_BODIES = {"nero_base", *(f"nero_link{i}" for i in range(1, 8)), "nero_gripper_link1", "nero_gripper_link2"}
TIP_BODIES = {"leap_if_tip", "leap_th_tip", "leap_mf_tip", "leap_rf_tip"}


def contact_bits(model: mujoco.MjModel, extra_tip_bodies: tuple[str, ...] = ()) -> None:
    """Assign each colliding geom its collision group and affinity bitmask."""
    tip_bodies = TIP_BODIES | set(extra_tip_bodies)
    object_bodies: set[int] = set()
    for joint in range(model.njnt):
        if model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_FREE:
            stack = [int(model.jnt_bodyid[joint])]
            while stack:
                body = stack.pop()
                object_bodies.add(body)
                stack.extend(b for b in range(model.nbody) if int(model.body_parentid[b]) == body)

    unmatched = []
    for geom_id in range(model.ngeom):
        if int(model.geom_contype[geom_id]) == 0 and int(model.geom_conaffinity[geom_id]) == 0:
            continue
        name = model.geom(geom_id).name
        body_id = int(model.geom_bodyid[geom_id])
        body_name = model.body(body_id).name
        if "ground" in name or "floor" in name:
            bits = (BIT_FLOOR, BIT_OBJECT | BIT_HAND | BIT_TIP | BIT_ARM, 3, 1)
        elif body_name in ARM_BODIES:
            bits = (BIT_ARM, BIT_FLOOR | BIT_OBJECT, 3, 1)
        elif "_part_" in name or body_name in tip_bodies:
            bits = (BIT_TIP, BIT_FLOOR | BIT_OBJECT | BIT_TIP, 3, 1)
        elif body_id in object_bodies:
            bits = (BIT_OBJECT, BIT_FLOOR | BIT_HAND | BIT_TIP, 3, 1)
        elif any(prefix in body_name for prefix in ("if_", "mf_", "rf_", "th_", "palm")):
            bits = (BIT_HAND, BIT_FLOOR | BIT_OBJECT, 1, 0)
        else:
            unmatched.append(f"  geom='{name}'  body='{body_name}'")
            continue
        (
            model.geom_contype[geom_id],
            model.geom_conaffinity[geom_id],
            model.geom_condim[geom_id],
            model.geom_priority[geom_id],
        ) = bits
    if unmatched:
        raise ValueError("contact_bits: geom(s) matched no collision group:\n" + "\n".join(unmatched))


def colorize_parts(model: mujoco.MjModel) -> None:
    """Color each active convex part distinctly and hide visual-only geoms in group 4."""
    part_idx = 0
    for geom_id in range(model.ngeom):
        name = model.geom(geom_id).name
        if "_part_" in name:
            is_unused = (
                int(model.geom_contype[geom_id]) == 0
                and int(model.geom_conaffinity[geom_id]) == 0
                and float(model.geom_rgba[geom_id, 3]) <= 0.0
            )
            if is_unused:
                model.geom_group[geom_id] = 4
                model.geom_rgba[geom_id] = (0.0, 0.0, 0.0, 0.0)
                model.geom_matid[geom_id] = -1
                continue
            model.geom_group[geom_id] = 0
            model.geom_rgba[geom_id] = (*colorsys.hsv_to_rgb((part_idx * 0.618033) % 1.0, 0.8, 0.95), 1.0)
            model.geom_matid[geom_id] = -1
            part_idx += 1
        elif "visual" in name:
            model.geom_group[geom_id] = 4


def rotate_points(quat_wxyz: torch.Tensor, local_pts: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
    """Transform batched local 3D points into world coordinates using wxyz quaternions."""
    q_w = quat_wxyz[:, 0:1].unsqueeze(1)
    q_vec = quat_wxyz[:, 1:4].unsqueeze(1).expand(-1, local_pts.shape[1], -1)
    uv = torch.cross(q_vec, local_pts, dim=-1)
    uuv = torch.cross(q_vec, uv, dim=-1)
    return local_pts + 2.0 * (q_w * uv + uuv) + offset.unsqueeze(1)


def world_geom_aabbs(
    geom_xpos: torch.Tensor,
    geom_xmat: torch.Tensor,
    geom_aabb: torch.Tensor,
    geom_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute world-frame min/max AABB corners for selected geoms across all worlds."""
    num_envs = geom_xpos.shape[0]
    rot = geom_xmat[:, geom_indices]
    aabb = geom_aabb.view(num_envs, -1, 2, 3)[:, geom_indices]
    center = geom_xpos[:, geom_indices] + torch.einsum("ngij,ngj->ngi", rot, aabb[:, :, 0])
    extents = torch.einsum("ngij,ngj->ngi", torch.abs(rot), aabb[:, :, 1])
    return center - extents, center + extents
