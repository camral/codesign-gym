"""Scripted Nero arm path and LEAP hand / parallel gripper poses."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
ROBOT_XML = ASSETS_DIR / "scenes" / "nero_leap_ftip_swap.xml"
GRIPPER_ROBOT_XML = ASSETS_DIR / "scenes" / "nero_with_gripper.xml"

GRASP_DEMO_DESCEND_FRAC = 0.10
GRASP_DEMO_ASCEND_FRAC = 0.40
DEMO_ARM_EE_ABS_X = 0.36
DEMO_ARM_EE_Y = 0.0
DEMO_ARM_EE_MAX_Z = 0.2
DEMO_ARM_EE_MIN_Z = 0.09
DEMO_ARM_EE_START_QUAT = [0.5, 0.5, 0.5, -0.5]  # wxyz, pointing down

NERO_EE_SITE = "nero_ee_site"
NERO_START_POSE = np.array([0.0, 0.39034993, 0.0, 1.63966067, 0.0, 0.0, 1.11158196])
NERO_DOF = 7
NERO_JOINT_NAMES = tuple(f"nero_joint{i}" for i in range(1, 8))
NERO_NARROW_LIMS_LOW = np.array([-0.50, 0.15, -0.50, 1.35, -0.30, -0.30, 0.40], dtype=np.float32)
NERO_NARROW_LIMS_HIGH = np.array([0.50, 0.85, 0.50, 2.00, 0.30, 0.30, 1.45], dtype=np.float32)
PARALLEL_GRIPPER_MAX_FING_DIST = 0.039

# Middle and ring fingers are held curled (hardware values shifted by pi).
LEAP_HAND_INACTIVE_FINGERS_REST = (
    np.array([3.641593, 3.141593, 4.941593, 4.941593, 3.641593, 3.141593, 4.941593, 4.941593], dtype=np.float32)
    - np.float32(np.pi)
)
LEAP_HAND_START = np.array(
    [1.018563, -0.1, 0.119650, 1.30, *LEAP_HAND_INACTIVE_FINGERS_REST, 1.566194, -0.211689, -0.250000, 1.20],
    dtype=np.float32,
)
LEAP_HAND_CLOSED = np.array(
    [1.224335, 0.076000, 0.460522, 1.009600, *LEAP_HAND_INACTIVE_FINGERS_REST, 1.604914, -0.211689, 0.079120, 1.045120],
    dtype=np.float32,
)


def leap_hand_start_pose() -> np.ndarray:
    return LEAP_HAND_START.copy()


def _open_close_traj(start: np.ndarray, closed: np.ndarray, length: int, descend_frac: float, ascend_frac: float) -> np.ndarray:
    n_descend = int(length * descend_frac)
    n_wait = length - n_descend - int(length * ascend_frac)
    traj = np.tile(closed, (length, 1)).astype(np.float32)
    traj[:n_descend] = start
    close_duration = max(1, n_wait // 2)
    for i in range(close_duration):
        alpha = i / close_duration
        traj[n_descend + i] = (1.0 - alpha) * start + alpha * closed
    return traj


def make_hand_demo_qpos_traj(tgt_traj_len: int) -> np.ndarray:
    return _open_close_traj(LEAP_HAND_START, LEAP_HAND_CLOSED, tgt_traj_len, GRASP_DEMO_DESCEND_FRAC, GRASP_DEMO_ASCEND_FRAC)


def make_gripper_demo_qpos_traj(
    tgt_traj_len: int,
    *,
    descend_frac: float = GRASP_DEMO_DESCEND_FRAC,
    ascend_frac: float = GRASP_DEMO_ASCEND_FRAC,
) -> np.ndarray:
    start = np.array([PARALLEL_GRIPPER_MAX_FING_DIST, -PARALLEL_GRIPPER_MAX_FING_DIST], dtype=np.float32)
    return _open_close_traj(start, np.zeros_like(start), tgt_traj_len, descend_frac, ascend_frac)


def apply_nero_coordinate_transform(pos, quat_wxyz):
    """Flip x and y and rotate 180 degrees about z."""
    new_pos = np.array(pos, dtype=np.float32)
    new_pos[0] *= -1
    new_pos[1] *= -1
    rot = Rotation.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
    new_q = (Rotation.from_euler("z", 180, degrees=True) * rot).as_quat()
    return new_pos, np.array([new_q[3], new_q[0], new_q[1], new_q[2]], dtype=np.float32)


def _basic_ik(model: mujoco.MjModel, pos, quat, q_seed, dt: float = 0.02, n_iters: int = 100, tol: float = 1e-4) -> np.ndarray:
    import mink

    cfg = mink.Configuration(model)
    cfg.data.qpos[:NERO_DOF] = np.asarray(q_seed, dtype=np.float64)[:NERO_DOF]
    cfg.update()
    target = mink.SE3.from_rotation_and_translation(mink.SO3(wxyz=np.asarray(quat, dtype=np.float64)), np.asarray(pos, dtype=np.float64))
    task = mink.FrameTask(NERO_EE_SITE, "site", position_cost=1.0, orientation_cost=5.0)
    task.set_target(target)
    for _ in range(n_iters):
        cfg.integrate_inplace(mink.solve_ik(cfg, [task], dt, solver="quadprog"), dt)
        if np.linalg.norm(cfg.get_transform_frame_to_world(NERO_EE_SITE, "site").translation() - target.translation()) < tol:
            break
    return cfg.data.qpos[:NERO_DOF].copy()


def make_arm_demo_qpos_traj(
    tgt_traj_len: int,
    *,
    robot_xml: Path = ROBOT_XML,
    ee_min_z: float = DEMO_ARM_EE_MIN_Z,
    descend_frac: float = GRASP_DEMO_DESCEND_FRAC,
    ascend_frac: float = GRASP_DEMO_ASCEND_FRAC,
) -> np.ndarray:
    """Joint trajectory (tgt_traj_len, 7) of the descend-hold-lift path."""
    n_descend = int(tgt_traj_len * descend_frac)
    n_ascend = int(tgt_traj_len * ascend_frac)
    n_wait = tgt_traj_len - n_descend - n_ascend
    t = np.arange(n_ascend, dtype=np.float64) / max(n_ascend - 1, 1)
    alpha = t * t * (3.0 - 2.0 * t)
    ascent = (1.0 - alpha) * ee_min_z + alpha * DEMO_ARM_EE_MAX_Z
    heights = np.concatenate([np.linspace(DEMO_ARM_EE_MAX_Z, ee_min_z, n_descend, endpoint=False), np.full(n_wait, ee_min_z), ascent])

    positions = np.zeros((tgt_traj_len, 3), dtype=np.float32)
    positions[:, 0] = -DEMO_ARM_EE_ABS_X
    positions[:, 1] = DEMO_ARM_EE_Y
    positions[:, 2] = heights
    quaternion = np.asarray(DEMO_ARM_EE_START_QUAT, dtype=np.float32)

    model = mujoco.MjModel.from_xml_path(str(robot_xml))
    q_seed, rows = NERO_START_POSE, []
    for pos in positions:
        q_seed = _basic_ik(model, pos, quaternion, q_seed)
        rows.append(q_seed)
    arm_traj = np.array(rows, dtype=np.float32)

    violations = (arm_traj < NERO_NARROW_LIMS_LOW) | (arm_traj > NERO_NARROW_LIMS_HIGH)
    if violations.any():
        details = [f"step {s} J{j + 1}: {arm_traj[s, j]:.5f}" for s, j in zip(*np.where(violations))]
        raise ValueError(f"Demo traj violates NARROW_ARM_LIMS: {', '.join(details[:5])}")
    return arm_traj
