"""Device code for the ball catcher.

Launched at the head of every control step, before the physics substeps:

- tick_kernel: advance the RNG stream index;
- force_kernel (optional): the random push on the base;
- ctrl_kernel: map the policy action onto actuator ctrl.

Launched at the tail of the step program (inside the CUDA graph, after kinematics/com_pos/com_vel):

- obs_state_kernel: computes goal pos/quat/yaw-local vel, yaw-local ball vel & rel pos,
  ball-goal distance, and all-feet-grounded flag in a single pass.

Launched outside the step program, at a reset or autoreset:

- reset_kernel: respawn and apply the design to the worlds in ``mask``.
"""

from __future__ import annotations

import math

import warp as wp

from .offsettable_logic import SegArrays, DeformArrays, apply_offsets

_PI = math.pi

# --- live tunables ----------------------------------------------------- #
T_MAX_OFFSET = 0     # metres per unit of link embodiment action
T_PITCH_VEL = 1
T_TGT_RADIUS = 2     # radius of the sphere the pitch target is sampled in
T_FROM = 3           # pitch origin, 3 floats
T_TO = 6             # pitch target centre, 3 floats
T_FORCE_MAG = 9
T_PITCH_SPREAD = 10  # +/- launch-speed variation
T_ARRIVAL_TIME = 11  # 0 selects speed mode
T_ARRIVAL_SPREAD = 12
T_TARGET_WIDTHS = 13 # arrival-mode target box half-widths, 3 floats
N_TUNABLES = 16


@wp.struct
class SimArrays:
    """Views of the batched mujoco_warp data fields the kernels read/write."""
    qpos: wp.array2d(dtype=wp.float32)
    qvel: wp.array2d(dtype=wp.float32)
    qacc: wp.array2d(dtype=wp.float32)
    qacc_warmstart: wp.array2d(dtype=wp.float32)
    time: wp.array(dtype=wp.float32)
    ctrl: wp.array2d(dtype=wp.float32)
    xfrc_applied: wp.array2d(dtype=wp.spatial_vector)
    xquat: wp.array2d(dtype=wp.quat)
    cvel: wp.array2d(dtype=wp.spatial_vector)
    subtree_com: wp.array2d(dtype=wp.vec3)
    site_xpos: wp.array2d(dtype=wp.vec3)
    geom_xpos: wp.array2d(dtype=wp.vec3)
    geom_xmat: wp.array2d(dtype=wp.mat33)
    geom_size: wp.array2d(dtype=wp.vec3)
    geom_margin: wp.array2d(dtype=wp.float32)


@wp.struct
class ObsArrays:
    """Per-world observation & reward intermediates written by ``obs_state_kernel``."""
    ground_geoms: wp.array(dtype=wp.int32)
    goal_pos: wp.array(dtype=wp.vec3)
    goal_quat: wp.array(dtype=wp.quat)
    goal_vel_local: wp.array(dtype=wp.vec3)
    ball_vel_local: wp.array(dtype=wp.vec3)
    rel_local: wp.array(dtype=wp.vec3)
    dist: wp.array(dtype=wp.float32)
    grounded: wp.array(dtype=wp.bool)


@wp.struct
class EnvArrays:
    """Env-owned per-world buffers."""
    mask: wp.array(dtype=wp.bool)             # worlds to respawn this step
    qpos_init: wp.array(dtype=wp.float32)
    deform: wp.array2d(dtype=wp.float32)      # staged embodiment action
    ctrl_act: wp.array2d(dtype=wp.float32)    # staged policy action, [-1, 1]
    ctrl_scale: wp.array(dtype=wp.float32)    # [-1, 1] -> actuator ctrlrange
    ctrl_mid: wp.array(dtype=wp.float32)
    offsets: wp.array2d(dtype=wp.float32)     # applied design, after clipping
    force_phase: wp.array(dtype=wp.int32)
    force_cycle: wp.array(dtype=wp.int32)
    force_dir: wp.array2d(dtype=wp.float32)
    tick: wp.array(dtype=wp.int32)            # RNG stream index, advanced every step
    lane_seed: wp.array(dtype=wp.int32)        # per-world seed
    ep_seed: wp.array(dtype=wp.int32)         # stream the world's episode was spawned from


@wp.struct
class CatcherConst:
    seed: int
    nq: int
    nv: int
    nu: int
    n_seg: int
    n_slot: int          # link embodiment slots, fewer than n_seg under _equ groups
    ball_qadr: int
    ball_dadr: int
    force_every: int
    force_hold: int
    base_bid: int
    root_bid: int
    goal_sid: int
    goal_bid: int
    floor_gid: int
    n_ground: int
    goal_sq: wp.quat     # site_quat of goal_site in MuJoCo (w, x, y, z) order
    gravity: wp.vec3


@wp.func
def _stream_key(lane_seed: int, tick: int) -> int:
    return wp.randi(wp.rand_init(lane_seed, tick))


@wp.func
def _yaw_local(v: wp.vec3, cy: float, sy: float) -> wp.vec3:
    """Rotate world-frame vector ``v`` into the base's yaw-aligned local frame."""
    return wp.vec3(cy * v[0] + sy * v[1], -sy * v[0] + cy * v[1], v[2])


@wp.func
def _mj_quat_mul(q: wp.quat, r: wp.quat) -> wp.quat:
    """Multiply two MuJoCo (w, x, y, z) quaternions stored in ``wp.quat``."""
    return wp.quat(
        q[0] * r[0] - q[1] * r[1] - q[2] * r[2] - q[3] * r[3],
        q[0] * r[1] + q[1] * r[0] + q[2] * r[3] - q[3] * r[2],
        q[0] * r[2] - q[1] * r[3] + q[2] * r[0] + q[3] * r[1],
        q[0] * r[3] + q[1] * r[2] - q[2] * r[1] + q[3] * r[0],
    )


@wp.kernel
def tick_kernel(env: EnvArrays):
    env.tick[0] = env.tick[0] + 1


@wp.kernel
def reset_kernel(env: EnvArrays, sim: SimArrays, dfm: DeformArrays,
                 seg: SegArrays,
                 c: CatcherConst, tun: wp.array(dtype=wp.float32)):
    """Respawn and apply the embodiment to one world per thread."""
    w = wp.tid()
    if not env.mask[w]:
        return

    for j in range(c.nq):
        sim.qpos[w, j] = env.qpos_init[j]
    for j in range(c.nv):
        sim.qvel[w, j] = 0.0
        sim.qacc[w, j] = 0.0
        sim.qacc_warmstart[w, j] = 0.0
    sim.time[w] = 0.0

    env.force_phase[w] = 0
    env.force_cycle[w] = 0
    env.ep_seed[w] = _stream_key(env.lane_seed[w], env.tick[0])

    rs = wp.rand_init(env.ep_seed[w], w)
    src = wp.vec3(tun[T_FROM], tun[T_FROM + 1], tun[T_FROM + 2])
    tgt = wp.vec3(tun[T_TO], tun[T_TO + 1], tun[T_TO + 2])
    vel = wp.vec3(0.0, 0.0, 0.0)
    if tun[T_ARRIVAL_TIME] > 0.0:
        for j in range(3):
            tgt[j] += (2.0 * wp.randf(rs) - 1.0) * tun[T_TARGET_WIDTHS + j]
        flight = tun[T_ARRIVAL_TIME] + (2.0 * wp.randf(rs) - 1.0) * tun[T_ARRIVAL_SPREAD]
        vel = (tgt - src) / flight - 0.5 * c.gravity * flight
    else:
        tgt += wp.sample_unit_sphere(rs) * tun[T_TGT_RADIUS]
        aim = tgt - src
        dist = wp.length(aim)
        if dist > 1.0e-9:
            speed = tun[T_PITCH_VEL] + (2.0 * wp.randf(rs) - 1.0) * tun[T_PITCH_SPREAD]
            vel = aim / dist * speed

    for j in range(3):
        sim.qpos[w, c.ball_qadr + j] = src[j]
        sim.qvel[w, c.ball_dadr + j] = vel[j]
        sim.qvel[w, c.ball_dadr + 3 + j] = 0.0
    sim.qpos[w, c.ball_qadr + 3] = 1.0
    sim.qpos[w, c.ball_qadr + 4] = 0.0
    sim.qpos[w, c.ball_qadr + 5] = 0.0
    sim.qpos[w, c.ball_qadr + 6] = 0.0

    apply_offsets(dfm, seg, env.deform, env.offsets, w, c.n_seg,
                  tun[T_MAX_OFFSET])


@wp.kernel
def force_kernel(env: EnvArrays, sim: SimArrays, c: CatcherConst,
                 tun: wp.array(dtype=wp.float32)):
    w = wp.tid()
    p = env.force_phase[w]

    if p == 0:
        cyc = env.force_cycle[w]
        rs = wp.rand_init(env.ep_seed[w], env.mask.shape[0] * (cyc + 1) + w)
        ang = wp.randf(rs) * _PI * 2.0
        env.force_dir[w, 0] = wp.cos(ang)
        env.force_dir[w, 1] = wp.sin(ang)
        env.force_cycle[w] = cyc + 1

    mag = tun[T_FORCE_MAG]
    if p >= c.force_every - c.force_hold:
        sim.xfrc_applied[w, c.base_bid] = wp.spatial_vector(
            env.force_dir[w, 0] * mag, env.force_dir[w, 1] * mag,
            0.0, 0.0, 0.0, 0.0)
    else:
        sim.xfrc_applied[w, c.base_bid] = wp.spatial_vector(
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    env.force_phase[w] = (p + 1) % c.force_every


@wp.kernel
def ctrl_kernel(env: EnvArrays, sim: SimArrays, c: CatcherConst):
    w = wp.tid()
    for a in range(c.nu):
        sim.ctrl[w, a] = (wp.clamp(env.ctrl_act[w, a], -1.0, 1.0)
                          * env.ctrl_scale[a] + env.ctrl_mid[a])


@wp.kernel
def obs_state_kernel(sim: SimArrays, out: ObsArrays, c: CatcherConst,
                     mask: wp.array(dtype=wp.bool), only: int):
    """Compute goal pose/velocity, yaw-local ball/goal vectors, distance, and
    grounded-feet flag in one CUDA-graphed kernel."""
    w = wp.tid()
    if only != 0 and not mask[w]:
        return

    # Base yaw frame from base quaternion (w, x, y, z)
    bq = sim.xquat[w, c.base_bid]
    siny = 2.0 * (bq[0] * bq[3] + bq[1] * bq[2])
    cosy = 1.0 - 2.0 * (bq[2] * bq[2] + bq[3] * bq[3])
    ynorm = wp.max(wp.sqrt(cosy * cosy + siny * siny), 1.0e-9)
    cy = cosy / ynorm
    sy = siny / ynorm

    gpos = sim.site_xpos[w, c.goal_sid]
    cvel = sim.cvel[w, c.goal_bid]
    gvel = wp.spatial_bottom(cvel) + wp.cross(
        wp.spatial_top(cvel), gpos - sim.subtree_com[w, c.root_bid])
    gq = _mj_quat_mul(sim.xquat[w, c.goal_bid], c.goal_sq)
    gq = wp.where(gq[0] < 0.0, -gq, gq)

    bpos = wp.vec3(sim.qpos[w, c.ball_qadr], sim.qpos[w, c.ball_qadr + 1], sim.qpos[w, c.ball_qadr + 2])
    bvel = wp.vec3(sim.qvel[w, c.ball_dadr], sim.qvel[w, c.ball_dadr + 1], sim.qvel[w, c.ball_dadr + 2])
    rel = bpos - gpos

    out.goal_pos[w] = gpos
    out.goal_quat[w] = gq
    out.goal_vel_local[w] = _yaw_local(gvel, cy, sy)
    out.ball_vel_local[w] = _yaw_local(bvel, cy, sy)
    out.rel_local[w] = _yaw_local(rel, cy, sy)
    out.dist[w] = wp.length(rel)

    # All feet grounded: exact capsule-plane clearance <= max(foot_margin, floor_margin)
    fg = c.floor_gid
    fpos = sim.geom_xpos[w, fg]
    fmat = sim.geom_xmat[w, fg]
    fnorm = wp.vec3(fmat[0, 2], fmat[1, 2], fmat[2, 2])
    wm = w % sim.geom_margin.shape[0]
    ws = w % sim.geom_size.shape[0]
    fm = sim.geom_margin[wm, fg]
    all_ok = int(1)
    for k in range(c.n_ground):
        gid = out.ground_geoms[k]
        gp = sim.geom_xpos[w, gid]
        gm = sim.geom_xmat[w, gid]
        gaxis = wp.vec3(gm[0, 2], gm[1, 2], gm[2, 2])
        gs = sim.geom_size[ws, gid]
        dist_f = wp.dot(gp - fpos, fnorm) - gs[0] - gs[1] * wp.abs(wp.dot(gaxis, fnorm))
        margin = wp.max(sim.geom_margin[wm, gid], fm)
        if dist_f > margin:
            all_ok = int(0)
    out.grounded[w] = all_ok != 0
