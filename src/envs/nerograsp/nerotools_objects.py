"""Articulated mesh tool XML loading and split-part FPS keypoints for NeroGraspAlt-Tools.

- load_tool_xml: parse ``carabiner.xml`` or ``wire_cutters.xml`` with absolute mesh
  paths and a normalized ``freejoint`` name, ready for direct MJCF merging;
- generate_tool_keypoints: 12 surface keypoints (6 per interacting part) subsampled
  deterministically via Farthest Point Sampling (FPS).
"""

from __future__ import annotations

import os
import pathlib
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

TOOL_FREEJOINT_NAME = "freejoint"


def load_tool_xml(tools_dir: pathlib.Path, object_name: str) -> ET.Element:
    """Parse ``<tools_dir>/<object_name>/<object_name>.xml`` for direct inclusion into a scene MJCF.

    Resolves each ``<mesh file=...>`` against the tool's ``<compiler meshdir>``,
    strips the ``<compiler>`` tag to avoid meshdir clashes, and renames the root
    ``<freejoint>`` to ``freejoint``.
    """
    xml_path = tools_dir / object_name / f"{object_name}.xml"
    if not xml_path.is_file():
        raise ValueError(f"Requested tool '{object_name}' does not exist at {xml_path}")
    root = ET.parse(xml_path).getroot()
    obj_dir = xml_path.parent
    compiler = root.find("compiler")
    meshdir = obj_dir / compiler.get("meshdir") if (compiler is not None and compiler.get("meshdir")) else obj_dir
    if compiler is not None:
        root.remove(compiler)
    for mesh_el in root.iter("mesh"):
        if "file" in mesh_el.attrib:
            mesh_el.attrib["file"] = os.path.abspath(str(meshdir / mesh_el.attrib["file"]))
    for jnt in root.iter("freejoint"):
        jnt.set("name", TOOL_FREEJOINT_NAME)
        break
    return root


def _subsample_fps(pts: np.ndarray, n: int) -> np.ndarray:
    """Deterministic Farthest Point Sampling starting from the point furthest from the origin."""
    if len(pts) > n:
        fps_pts = np.zeros((n, 3), dtype=np.float32)
        fps_pts[0] = pts[np.argmax(np.linalg.norm(pts, axis=1))]
        distances = np.linalg.norm(pts - fps_pts[0], axis=1)
        for i in range(1, n):
            farthest_idx = int(np.argmax(distances))
            fps_pts[i] = pts[farthest_idx]
            distances = np.minimum(distances, np.linalg.norm(pts - fps_pts[i], axis=1))
        return fps_pts
    if len(pts) == 0:
        return np.zeros((n, 3), dtype=np.float32)
    idx = np.linspace(0, len(pts) - 1, n, dtype=int)
    return np.asarray(pts[idx], dtype=np.float32)


def _geom_world_verts(model: mujoco.MjModel, data: mujoco.MjData, gid: int) -> tuple[np.ndarray, np.ndarray]:
    """Raw local mesh vertices and their current world-frame positions for one mesh geom."""
    mesh_id = int(model.geom_dataid[gid])
    vert_adr = int(model.mesh_vertadr[mesh_id])
    vert_num = int(model.mesh_vertnum[mesh_id])
    verts = model.mesh_vert[vert_adr:vert_adr + vert_num]
    g_pos = data.geom_xpos[gid]
    g_mat = data.geom_xmat[gid].reshape(3, 3)
    return verts, (g_mat @ verts.T).T + g_pos


def generate_tool_keypoints(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    object_name: str,
    obj_body_id: int,
    all_obj_geom_ids: list[int],
    *,
    gate_body_id: int = -1,
    handle1_body_id: int = -1,
    num_keypoints: int = 12,
) -> np.ndarray:
    """Return ``(num_keypoints, 3)`` split-part surface keypoints for ``carabiner`` or ``wire_cutters``."""
    mujoco.mj_kinematics(model, data)
    if object_name == "wire_cutters":
        return _wire_cutters_keypoints(model, data, obj_body_id, all_obj_geom_ids, handle1_body_id, num_keypoints)
    if object_name == "carabiner":
        return _carabiner_keypoints(model, data, obj_body_id, all_obj_geom_ids, gate_body_id, num_keypoints)
    raise ValueError(f"Unsupported tool object for keypoint generation: '{object_name}'")


def _wire_cutters_keypoints(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    obj_body_id: int,
    all_obj_geom_ids: list[int],
    handle1_body_id: int,
    num_keypoints: int,
) -> np.ndarray:
    """6 keypoints on moving handle 1 (in handle1's local frame) + 6 on fixed handle 2 (in root frame)."""
    half = num_keypoints // 2
    if handle1_body_id == -1:
        handle1_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wire_cutters_handle")
    obj_xpos, obj_xmat = data.xpos[obj_body_id], data.xmat[obj_body_id].reshape(3, 3)
    h1_xpos, h1_xmat = data.xpos[handle1_body_id], data.xmat[handle1_body_id].reshape(3, 3)

    left_pts, right_pts = [], []
    for gid in all_obj_geom_ids:
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        if "wire_cutters_handle" not in geom_name or model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        verts, global_verts = _geom_world_verts(model, data, gid)
        if "handle_2" in geom_name:
            local_verts = (global_verts - obj_xpos) @ obj_xmat
            # Lengthwise axis and outward normal of handle 2 in mesh coordinates.
            handle_axis = np.array([-0.0834722, 0.0465408, 0.0])
            outward_dir = np.array([-0.0465408, -0.0834722, 0.0])
        else:
            local_verts = (global_verts - h1_xpos) @ h1_xmat
            # Lengthwise axis and outward normal of handle 1 in mesh coordinates.
            handle_axis = np.array([-0.0465408, 0.0834722, 0.0])
            outward_dir = np.array([-0.0834722, -0.0465408, 0.0])
        proj = verts @ handle_axis
        outward = verts @ outward_dir
        middle_strip = (proj > proj.min() + 0.25 * np.ptp(proj)) & (proj < proj.min() + 0.80 * np.ptp(proj))
        outer_face = outward > (outward.min() + 0.90 * np.ptp(outward))
        (right_pts if "handle_2" in geom_name else left_pts).append(local_verts[outer_face & middle_strip])

    left_arr = np.concatenate(left_pts, axis=0) if left_pts else np.zeros((0, 3), dtype=np.float32)
    right_arr = np.concatenate(right_pts, axis=0) if right_pts else np.zeros((0, 3), dtype=np.float32)
    return np.concatenate([_subsample_fps(left_arr, half), _subsample_fps(right_arr, num_keypoints - half)], axis=0)


def _carabiner_keypoints(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    obj_body_id: int,
    all_obj_geom_ids: list[int],
    gate_body_id: int,
    num_keypoints: int,
) -> np.ndarray:
    """6 keypoints on the middle 35%-65% of moving gate (in gate frame) + 6 on bracket spine (in root frame)."""
    half = num_keypoints // 2
    obj_xpos, obj_xmat = data.xpos[obj_body_id], data.xmat[obj_body_id].reshape(3, 3)
    gate_xpos, gate_xmat = data.xpos[gate_body_id], data.xmat[gate_body_id].reshape(3, 3)
    gate_pts, spine_pts = [], []
    for gid in all_obj_geom_ids:
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        if model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        if geom_name == "carabiner_link_gate_00000_col":
            _, global_verts = _geom_world_verts(model, data, gid)
            gate_pts.append((global_verts - gate_xpos) @ gate_xmat)
        elif geom_name in ("carabiner_link_bracket_00000_col", "carabiner_link_bracket_00001_col"):
            _, global_verts = _geom_world_verts(model, data, gid)
            spine_pts.append((global_verts - obj_xpos) @ obj_xmat)

    if gate_pts:
        gate_arr = np.concatenate(gate_pts, axis=0)
        # Compress gate x-span into its central 35%-65% graspable band.
        x_min, x_range = gate_arr[:, 0].min(), np.ptp(gate_arr[:, 0])
        if x_range > 0:
            gate_arr[:, 0] = x_min + (0.35 + 0.30 * (gate_arr[:, 0] - x_min) / x_range) * x_range
    else:
        gate_arr = np.zeros((0, 3), dtype=np.float32)
    spine_arr = np.concatenate(spine_pts, axis=0) if spine_pts else np.zeros((0, 3), dtype=np.float32)
    return np.concatenate([_subsample_fps(gate_arr, half), _subsample_fps(spine_arr, num_keypoints - half)], axis=0)
