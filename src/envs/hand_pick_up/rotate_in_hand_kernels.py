"""Device code shared by the in-hand envs (rotate_in_hand, hand_pick_up).

Launched at the head of every control step, before the physics substeps:

- tick_kernel: advance the RNG stream index;
- deform_kernel: write the embodiment (pads and links) during the deform
windows, and after every respawn;
- force_kernel (optional): the random push on the object;
- ctrl_kernel: map the policy action onto actuator ctrl.

Launched by the env outside the step program:

- reset_kernel: respawn the worlds in ``mask``, with a new object orientation
and goal, at a reset or after the step program at an autoreset;
- sample_kernel, window_kernel: whether the hand (and the world) touches the
object, remembered over a few steps since contact flickers;
- task_kernel: the object's pose and twist relative to the palm, its distance
to the goal, and scoring.

The object's pose and twist are measured relative to the palm, so turning the
wrist does not count as turning the object. The goal is in the palm's frame
(``EnvConst.goal_world`` 0) or the world's (1), where it stays put while the
hand moves; it is observed in the palm's frame either way. Quaternions are
MuJoCo's w, x, y, z inside a ``wp.quat``. Warp's quaternion helpers read them
as x, y, z, w, so all quaternion math here is written out component-wise.
"""

from __future__ import annotations

import warp as wp

from .hfield_logic import HfieldArrays, apply_hfield_offsets
from .offsettable_logic import DeformArrays, SegArrays, apply_offsets

_PI = 3.14159265358979323846

# --- live tunables ----------------------------------------------------- #
T_MAX_LINK_OFFSET = 0             # metres per unit of link embodiment action
T_HFIELD_OFFSET_FRACTION = 1      # share of each pad's elevation range a cell can use
T_FORCE_MAG = 2
T_GOAL_DIST = 3                   # angle from the current orientation to a new goal [rad]

N_TUNABLES = 4


@wp.struct
class SimArrays:
    """Views of the batched mujoco_warp model and data fields the kernels use."""
    qpos: wp.array2d(dtype=wp.float32)
    qvel: wp.array2d(dtype=wp.float32)
    qacc: wp.array2d(dtype=wp.float32)
    qacc_warmstart: wp.array2d(dtype=wp.float32)
    time: wp.array(dtype=wp.float32)
    ctrl: wp.array2d(dtype=wp.float32)
    xpos: wp.array2d(dtype=wp.vec3)
    xquat: wp.array2d(dtype=wp.quat)
    xfrc_applied: wp.array2d(dtype=wp.spatial_vector)
    cvel: wp.array2d(dtype=wp.spatial_vector)
    subtree_com: wp.array2d(dtype=wp.vec3)


@wp.struct
class EnvArrays:
    """Env-owned per-world buffers."""
    mask: wp.array(dtype=wp.bool)             # worlds to respawn this step
    qpos_init: wp.array(dtype=wp.float32)
    deform: wp.array2d(dtype=wp.float32)      # staged embodiment action
    ctrl_act: wp.array2d(dtype=wp.float32)    # staged policy action, [-1, 1]
    ctrl_scale: wp.array(dtype=wp.float32)    # [-1, 1] -> actuator ctrlrange
    ctrl_mid: wp.array(dtype=wp.float32)
    offsets: wp.array2d(dtype=wp.float32)     # applied embodiment, after clipping
    tick: wp.array(dtype=wp.int32)            # RNG stream index, advanced every step
    lane_seed: wp.array(dtype=wp.int32)        # per-world seed: the run's, or the world's own from a per-lane sequence
    ep_seed: wp.array(dtype=wp.int32)         # stream the world's episode was spawned from
    force_dir: wp.array2d(dtype=wp.float32)
    force_phase: wp.array(dtype=wp.int32)
    force_cycle: wp.array(dtype=wp.int32)
    armed: wp.array(dtype=wp.bool)            # liftoff gate for perturbations (always True without table)
    deform_dirty: wp.array(dtype=wp.int32)    # 1 when staged design changed since last deform


@wp.struct
class GoalArrays:
    """The goal orientation and each world's state relative to it, written by
    ``reset_kernel`` and ``task_kernel``. With ``EnvConst.goal_world`` the goal
    and the position target are world-fixed (``goal_pos_w``).

    ``dwell`` counts steps in tolerance (not necessarily consecutive). ``gate``
    must be open for such a step to count. ``obj_*`` are the object in the
    palm's frame, and ``obj_off_w`` is its offset from the palm in world axes.
    """
    quat: wp.array(dtype=wp.quat)             # current goal, palm or world frame
    quat_obs: wp.array(dtype=wp.quat)         # the same goal in the palm's frame now
    t0: wp.array(dtype=wp.float32)            # sim time the goal was set
    dwell: wp.array(dtype=wp.int32)
    scored: wp.array(dtype=wp.int32)          # goal completed on this step
    rebase: wp.array(dtype=wp.int32)          # goal changed, re-base the shaping
    dist: wp.array(dtype=wp.float32)          # angle to the goal now in force
    reached: wp.array(dtype=wp.float32)       # angle the shaping uses, 0 in tolerance
    paid: wp.array(dtype=wp.int32)            # a dwell step was counted
    elapsed: wp.array(dtype=wp.float32)
    gate: wp.array(dtype=wp.int32)
    pos_dist: wp.array(dtype=wp.float32)      # metres to rel_pos, if pos_on
    obj_pos: wp.array(dtype=wp.vec3)
    obj_quat: wp.array(dtype=wp.quat)
    obj_vel: wp.array(dtype=wp.vec3)          # d/dt of obj_pos, palm axes
    obj_angvel: wp.array(dtype=wp.vec3)       # object's spin relative to the palm
    obj_off_w: wp.array(dtype=wp.vec3)


@wp.struct
class EnvConst:
    seed: int
    nq: int
    nv: int
    nu: int
    n_seg: int
    n_pad: int
    n_cell: int               # pad cells, i.e. the pads' embodiment entries
    obj_bid: int
    obj_qadr: int
    obj_dadr: int
    obj_root: int             # tree roots for the cvel conversions
    palm_bid: int
    palm_root: int
    palm_quat0: wp.quat       # the palm's world orientation at the reset pose
    spawn_pos: wp.vec3        # the object's world origin at respawn
    init_quat: wp.quat        # its orientation then, palm frame, unless randomised
    randomise_init: int
    goal_fixed: int           # 1: one goal per episode, goal_quat
    goal_quat: wp.quat        # palm frame, or the world's with goal_world
    goal_world: int           # 1: goals (and the position target) are world-fixed
    rot_tol: float
    pos_on: int               # 1: the goal also has a position, rel_pos (palm frame)
    rel_pos: wp.vec3
    goal_pos_w: wp.vec3       # rel_pos at the palm's reset pose, world; used with goal_world
    pos_tol: float
    dwell_limit: int
    deform_hfield_s: float    # deform windows, seconds from the respawn
    deform_links_s: float
    force_every: int
    force_hold: int


# Geom classes for the object's contacts
KIND_OTHER = 0
KIND_OBJECT = 1
KIND_HAND = 2
KIND_WORLD = 3


@wp.struct
class ContactArrays:
    """Per-geom classes and the per-world contact memory."""
    kind: wp.array(dtype=wp.int32)
    raw_hand: wp.array(dtype=wp.int32)        # this step's sample
    raw_world: wp.array(dtype=wp.int32)
    hist_hand: wp.array(dtype=wp.int32)       # one bit per remembered sample
    hist_world: wp.array(dtype=wp.int32)
    touch_hand: wp.array(dtype=wp.float32)    # any remembered sample, 0 or 1
    touch_world: wp.array(dtype=wp.float32)
    mult: wp.array(dtype=wp.float32)          # reward multiplier, see window_kernel


@wp.func
def quat_mul(a: wp.quat, b: wp.quat) -> wp.quat:
    return wp.quat(a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
                   a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
                   a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
                   a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0])


@wp.func
def quat_conj(q: wp.quat) -> wp.quat:
    # Inverse of a unit quaternion
    return wp.quat(q[0], -q[1], -q[2], -q[3])


@wp.func
def quat_rotate_inv(q: wp.quat, v: wp.vec3) -> wp.vec3:
    # R(q)^T v: rotating by (s, u) is v + s t + u x t with t = 2 u x v, and the
    # inverse negates u
    u = wp.vec3(-q[1], -q[2], -q[3])
    t = 2.0 * wp.cross(u, v)
    return v + q[0] * t + wp.cross(u, t)


@wp.func
def origin_vel(sim: SimArrays, w: int, b: int, root: int) -> wp.vec3:
    # World velocity of body b's origin. cvel is (angular, linear) at the root's subtree CoM
    cv = sim.cvel[w, b]
    spin = wp.vec3(cv[0], cv[1], cv[2])
    return (wp.vec3(cv[3], cv[4], cv[5])
            + wp.cross(spin, sim.xpos[w, b] - sim.subtree_com[w, root]))


@wp.func
def geodesic(a: wp.quat, b: wp.quat) -> float:
    """Angle of the rotation taking ``a`` to ``b``, in [0, pi]. Squaring the inner
    product makes it the same for ``q`` and ``-q``."""
    dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
    return wp.acos(wp.clamp(2.0 * dot * dot - 1.0, -1.0, 1.0))


@wp.func
def rand_quat(seed: int, idx: int) -> wp.quat:
    """A uniformly random orientation: four normals, normalised.

    The RNG state is built here rather than passed in, since warp's ``rand_*``
    advance their state in place and a shared state would correlate draws.
    """
    rs = wp.rand_init(seed, idx)
    x = wp.randn(rs)
    y = wp.randn(rs)
    z = wp.randn(rs)
    w = wp.randn(rs)
    n = wp.sqrt(x * x + y * y + z * z + w * w)
    if n < 1.0e-9:
        return wp.quat(1.0, 0.0, 0.0, 0.0)
    return wp.quat(x / n, y / n, z / n, w / n)


@wp.func
def goal_at(q: wp.quat, angle: float, seed: int, idx: int) -> wp.quat:
    """An orientation exactly ``angle`` from ``q``, about a random axis, so a
    completed rotation always gives the same shaping return."""
    rs = wp.rand_init(seed, idx)
    x = wp.randn(rs)
    y = wp.randn(rs)
    z = wp.randn(rs)
    n = wp.sqrt(x * x + y * y + z * z)
    if n < 1.0e-9:                 # degenerate draw: any axis will do
        x = 1.0
        y = 0.0
        z = 0.0
        n = 1.0
    s = wp.sin(0.5 * angle) / n
    return quat_mul(q, wp.quat(wp.cos(0.5 * angle), x * s, y * s, z * s))


@wp.func
def _stream_key(lane_seed: int, tick: int) -> int:
    """The random stream of one lane seed at one step: the two hashed together, so runs seeded s and s + 1 never replay
    each other a step apart, as a plain sum seed + tick would."""
    return wp.randi(wp.rand_init(lane_seed, tick))


@wp.func
def goal_in_palm(qp: wp.quat, g: wp.quat, goal_world: int) -> wp.quat:
    """Goal ``g`` in the frame of a palm turned ``qp``: ``g`` itself for a palm-frame goal."""
    if goal_world == 1:
        return quat_mul(quat_conj(qp), g)
    return g


@wp.kernel
def tick_kernel(env: EnvArrays):
    # Launch arguments are frozen in the graph, so the RNG index must advance on device
    env.tick[0] = env.tick[0] + 1


@wp.kernel
def reset_kernel(env: EnvArrays, sim: SimArrays, goal: GoalArrays,
                 c: EnvConst, tun: wp.array(dtype=wp.float32)):
    """Respawn one world per thread, worlds outside ``mask`` return immediately.

    The object spawns at ``spawn_pos``, randomly oriented or at ``init_quat``
    relative to the palm. A randomised goal is ``goal_dist`` from that, in the
    goal's frame.
    """
    w = wp.tid()
    if not env.mask[w]:
        return

    for j in range(c.nq):
        sim.qpos[w, j] = env.qpos_init[j]
    # Clear the solver state too: a warm start carried over from a terminal
    # state can diverge the new episode
    for j in range(c.nv):
        sim.qvel[w, j] = 0.0
        sim.qacc[w, j] = 0.0
        sim.qacc_warmstart[w, j] = 0.0
    sim.time[w] = 0.0
    sim.xfrc_applied[w, c.obj_bid] = wp.spatial_vector(
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    key = _stream_key(env.lane_seed[w], env.tick[0])
    rel = c.init_quat
    if c.randomise_init == 1:
        rel = rand_quat(key, w)
    q = quat_mul(c.palm_quat0, rel)          # palm_quat0, since xquat is stale here
    a = c.obj_qadr
    sim.qpos[w, a + 0] = c.spawn_pos[0]
    sim.qpos[w, a + 1] = c.spawn_pos[1]
    sim.qpos[w, a + 2] = c.spawn_pos[2]
    sim.qpos[w, a + 3] = q[0]
    sim.qpos[w, a + 4] = q[1]
    sim.qpos[w, a + 5] = q[2]
    sim.qpos[w, a + 6] = q[3]

    if c.goal_fixed == 1:
        goal.quat[w] = c.goal_quat
    else:
        # Its own stream, or the goal's axis would depend on the orientation
        start = rel
        if c.goal_world == 1:
            start = q
        goal.quat[w] = goal_at(start, tun[T_GOAL_DIST], key, w + 104729)
    goal.t0[w] = 0.0
    goal.dwell[w] = 0
    goal.scored[w] = 0
    goal.rebase[w] = 1

    env.force_phase[w] = 0
    env.force_cycle[w] = 0
    env.ep_seed[w] = key
    env.force_dir[w, 0] = 0.0
    env.force_dir[w, 1] = 0.0
    env.force_dir[w, 2] = 0.0


@wp.kernel
def deform_kernel(env: EnvArrays, sim: SimArrays, dfm: DeformArrays,
                  seg: SegArrays, hf: HfieldArrays, c: EnvConst,
                  tun: wp.array(dtype=wp.float32)):
    """Write world ``w``'s pads and links from its embodiment entries, when it is
    in ``mask`` or while inside each one's deform window and ``deform_dirty`` is set.
    Both rebuild from the undeformed model, so rewriting is exact."""
    w = wp.tid()
    fresh = env.mask[w]
    if not fresh and env.deform_dirty[0] == 0:
        return
    t = sim.time[w]
    if c.n_cell > 0 and (fresh or t < c.deform_hfield_s):
        apply_hfield_offsets(hf, env.deform, env.offsets, w, c.n_pad,
                             tun[T_HFIELD_OFFSET_FRACTION])
    if c.n_seg > 0 and (fresh or t < c.deform_links_s):
        apply_offsets(dfm, seg, env.deform, env.offsets, w, c.n_seg,
                      tun[T_MAX_LINK_OFFSET])


@wp.kernel
def clear_deform_dirty_kernel(env: EnvArrays):
    env.deform_dirty[0] = 0


@wp.kernel
def ctrl_kernel(env: EnvArrays, sim: SimArrays, c: EnvConst):
    w = wp.tid()
    for a in range(c.nu):
        x = wp.clamp(env.ctrl_act[w, a], -1.0, 1.0)
        sim.ctrl[w, a] = x * env.ctrl_scale[a] + env.ctrl_mid[a]


@wp.kernel
def force_kernel(env: EnvArrays, sim: SimArrays, c: EnvConst,
                 tun: wp.array(dtype=wp.float32)):
    """Random-axis 3D perturbation torque on the object, gated on liftoff.

    While ``env.armed[w]`` is False (or the world is respawning), no torque is
    applied and the cycle clock stays paused. Once armed, a uniformly random 3D
    unit axis is drawn at the start of every ``force_every``-step cycle and a
    pure world-frame torque of magnitude ``force_mag`` [N*m] is applied for the
    cycle's last ``force_hold`` steps.
    """
    w = wp.tid()
    if not env.armed[w] or env.mask[w]:
        sim.xfrc_applied[w, c.obj_bid] = wp.spatial_vector(
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        return

    p = env.force_phase[w]
    if p == 0:
        cyc = env.force_cycle[w]
        # Sequence numbers past num_envs keep this stream apart from the spawn's
        rs = wp.rand_init(env.ep_seed[w], env.mask.shape[0] * (cyc + 1) + w)
        x = wp.randn(rs)
        y = wp.randn(rs)
        z = wp.randn(rs)
        n = wp.sqrt(x * x + y * y + z * z)
        if n < 1.0e-9:
            x = 1.0
            y = 0.0
            z = 0.0
            n = 1.0
        env.force_dir[w, 0] = x / n
        env.force_dir[w, 1] = y / n
        env.force_dir[w, 2] = z / n
        env.force_cycle[w] = cyc + 1

    mag = tun[T_FORCE_MAG]
    if p >= c.force_every - c.force_hold:
        sim.xfrc_applied[w, c.obj_bid] = wp.spatial_vector(
            0.0, 0.0, 0.0,
            env.force_dir[w, 0] * mag,
            env.force_dir[w, 1] * mag,
            env.force_dir[w, 2] * mag)
    else:
        sim.xfrc_applied[w, c.obj_bid] = wp.spatial_vector(
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    env.force_phase[w] = (p + 1) % c.force_every


@wp.kernel
def sample_kernel(nacon: wp.array(dtype=wp.int32),
                  worldid: wp.array(dtype=wp.int32),
                  geom: wp.array(dtype=wp.vec2i), ct: ContactArrays,
                  mask: wp.array(dtype=wp.bool), only_mask: int):
    """Mark the worlds whose object touches the hand, or a world geom. One thread
    per contact slot, every write is the same 1."""
    i = wp.tid()
    if i >= nacon[0]:
        return
    if only_mask == 1 and not mask[worldid[i]]:
        return
    g = geom[i]
    a = ct.kind[g[0]]
    b = ct.kind[g[1]]
    other = KIND_OTHER
    if a == KIND_OBJECT:
        other = b
    elif b == KIND_OBJECT:
        other = a
    w = worldid[i]
    if other == KIND_HAND:
        ct.raw_hand[w] = 1
    elif other == KIND_WORLD:
        ct.raw_world[w] = 1


@wp.kernel
def window_kernel(env: EnvArrays, goal: GoalArrays, ct: ContactArrays,
                  bits: int, table: int, only_mask: int):
    """Shift this step's sample into each world's contact history and read it.

    The hand counts as touching if any remembered sample saw it, and the gate
    opens only then. With ``table``, a world contact as well halves ``mult`` and
    keeps the gate shut, so a goal is scored only lifted off the table. A
    respawned world starts its history over.
    """
    w = wp.tid()
    if only_mask == 1 and not env.mask[w]:
        return
    hh = ct.hist_hand[w]
    hw = ct.hist_world[w]
    if env.mask[w]:
        hh = 0
        hw = 0
    hh = ((hh << 1) | ct.raw_hand[w]) & bits
    hw = ((hw << 1) | ct.raw_world[w]) & bits
    ct.hist_hand[w] = hh
    ct.hist_world[w] = hw
    ct.raw_hand[w] = 0
    ct.raw_world[w] = 0

    th = 0.0
    if hh != 0:
        th = 1.0
    tw = 0.0
    if hw != 0:
        tw = 1.0
    ct.touch_hand[w] = th
    ct.touch_world[w] = tw
    gate = 0
    if hh != 0:
        gate = 1
    if table == 1:
        th = th * (1.0 - 0.5 * tw)
        if hw != 0:
            gate = 0
    ct.mult[w] = th
    goal.gate[w] = gate


@wp.kernel
def task_kernel(env: EnvArrays, sim: SimArrays, goal: GoalArrays,
                c: EnvConst, tun: wp.array(dtype=wp.float32), only_mask: int):
    """Per world: the object's pose and twist in the palm frame, the distance to
    the goal, and scoring once a goal has dwelt ``dwell_limit`` steps.

    ``reached`` and ``paid`` refer to the goal in force when the step began,
    since this step's reward is owed for it. ``dist`` refers to the goal in
    force now, so the observation stays consistent. A scored randomised goal is
    replaced by one ``goal_dist`` from the current orientation. The angle is
    taken in the goal's frame, and ``quat_obs`` is the goal in the palm's frame
    as it stands now.
    """
    w = wp.tid()
    if only_mask == 1 and not env.mask[w]:
        return
    qp = sim.xquat[w, c.palm_bid]
    qo = sim.xquat[w, c.obj_bid]
    off = sim.xpos[w, c.obj_bid] - sim.xpos[w, c.palm_bid]
    q = quat_mul(quat_conj(qp), qo)
    p = quat_rotate_inv(qp, off)
    goal.obj_off_w[w] = off
    goal.obj_pos[w] = p
    goal.obj_quat[w] = q

    cvp = sim.cvel[w, c.palm_bid]
    cvo = sim.cvel[w, c.obj_bid]
    spin_p = wp.vec3(cvp[0], cvp[1], cvp[2])
    spin_o = wp.vec3(cvo[0], cvo[1], cvo[2])
    # The object's velocity minus what it would have if bolted to the palm
    vel = (origin_vel(sim, w, c.obj_bid, c.obj_root)
           - origin_vel(sim, w, c.palm_bid, c.palm_root)
           - wp.cross(spin_p, off))
    goal.obj_vel[w] = quat_rotate_inv(qp, vel)
    goal.obj_angvel[w] = quat_rotate_inv(qp, spin_o - spin_p)

    # The object's orientation in the frame the goal is in.
    qf = q
    if c.goal_world == 1:
        qf = qo
    goal.quat_obs[w] = goal_in_palm(qp, goal.quat[w], c.goal_world)
    dist = geodesic(qf, goal.quat[w])
    goal.elapsed[w] = sim.time[w] - goal.t0[w]
    goal.rebase[w] = 0
    goal.dist[w] = dist
    at_pos = 1
    if c.pos_on == 1:
        pos_dist = wp.length(p - c.rel_pos)
        if c.goal_world == 1:
            pos_dist = wp.length(sim.xpos[w, c.obj_bid] - c.goal_pos_w)
        goal.pos_dist[w] = pos_dist
        if pos_dist >= c.pos_tol:
            at_pos = 0

    # A step in tolerance counts only with the gate open (and at the position, if any)
    dwell = goal.dwell[w]
    if dist < c.rot_tol and at_pos == 1 and goal.gate[w] != 0:
        dwell += 1
        goal.reached[w] = 0.0              # the last few radians are rewarded too
        goal.paid[w] = 1
    else:
        goal.reached[w] = dist
        goal.paid[w] = 0
    goal.dwell[w] = dwell

    if dwell < c.dwell_limit:
        goal.scored[w] = 0
        return

    goal.scored[w] = 1
    if c.goal_fixed == 1:
        return
    newg = goal_at(qf, tun[T_GOAL_DIST], _stream_key(env.lane_seed[w], env.tick[0]) + 7919, w)
    goal.quat[w] = newg
    goal.quat_obs[w] = goal_in_palm(qp, newg, c.goal_world)
    goal.dist[w] = geodesic(qf, newg)
    goal.dwell[w] = 0
    goal.t0[w] = sim.time[w]
    goal.rebase[w] = 1
