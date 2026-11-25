"""Device code for the truck unloader.

Launched at the head of every control step, before the physics substeps:

- tick_kernel: advance the RNG stream index;
- ctrl_kernel: map the policy action onto actuator ctrl.

Launched by the env outside the step program:

- reset_kernel: respawn and apply the embodiment to the worlds in ``mask``, at
a reset or after the step program at an autoreset;
- contact_kernel: which gripper parts and bed geoms each box touches;
- box_kernel: mark boxes lost, and undeliver boxes that left their region;
- task_kernel: resolve each robot's target box, write what the observation
and reward read, and count deliveries.

Robot 0 is the loader, which moves boxes from the container onto the belt.
Robots 1.. are the stackers, each delivering the boxes of its colour from the
belt to its own goal. The structs group the kernel arguments: ``SimArrays``
(views of mujoco_warp model and data), ``EnvArrays`` (env-owned per-world
buffers), ``BoxArrays``, ``RobotArrays``, ``GraspArrays`` and ``BedArrays``
(task tables and state), and ``EnvConst`` (capture-time constants).
"""

from __future__ import annotations

import warp as wp

from .offsettable_logic import DeformArrays, SegArrays, apply_offsets

# Layout of mujoco_warp's geom_surfacevel rows, which Warp matches by length and dtype
vec6f = wp.types.vector(length=6, dtype=wp.float32)

# Goal kinds, see RobotArrays.goal_kind
GOAL_SITE = wp.constant(0)
GOAL_REGION = wp.constant(1)

# --- live tunables ----------------------------------------------------- #
T_MAX_OFFSET = 0     # metres per unit of link embodiment action
T_BELT_VEL = 1       # belt surface speed (m/s) a world's draw is centred on
T_BELT_SPREAD = 2    # +/- fraction of it each respawn draws
T_BOX_JITTER = 3     # +/- metres each container box moves in x and y at respawn
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
    xquat: wp.array2d(dtype=wp.quat)
    cvel: wp.array2d(dtype=wp.spatial_vector)
    subtree_com: wp.array2d(dtype=wp.vec3)
    site_xpos: wp.array2d(dtype=wp.vec3)
    geom_surfacevel: wp.array2d(dtype=vec6f)  # model field, one row per world


@wp.struct
class EnvArrays:
    """Env-owned per-world buffers."""
    mask: wp.array(dtype=wp.bool)             # worlds to respawn this step
    qpos_init: wp.array(dtype=wp.float32)
    deform: wp.array2d(dtype=wp.float32)      # staged embodiment action
    ctrl_act: wp.array2d(dtype=wp.float32)    # staged policy action, [-1, 1]
    ctrl_scale: wp.array(dtype=wp.float32)    # [-1, 1] -> actuator ctrlrange
    ctrl_mid: wp.array(dtype=wp.float32)
    ctrl_deadzone: wp.array(dtype=wp.int32)   # 1: [-1, 0] maps to the bottom of the range
    offsets: wp.array2d(dtype=wp.float32)     # applied embodiment, after clipping
    offset_scale: wp.array(dtype=wp.float32)  # max_offset each world was built with
    tick: wp.array(dtype=wp.int32)
    lane_seed: wp.array(dtype=wp.int32)        # per-world seed: the run's, or the world's own from a per-lane sequence
    belt_gid: wp.array(dtype=wp.int32)        # the belt's moving geoms (surface and rollers)


@wp.struct
class BoxArrays:
    """Every box: its qpos/qvel addresses, owner and spawn data, and its state
    per world (0 live, 1 stacked, 2 lost). A box the loader puts on the belt
    stays live, since a stacker still has to deliver it."""
    qadr: wp.array(dtype=wp.int32)
    dadr: wp.array(dtype=wp.int32)
    color: wp.array(dtype=wp.int32)           # owning stacker id - 1
    spawn_z: wp.array(dtype=wp.float32)
    order: wp.array(dtype=wp.int32)           # loader's visiting order of the container boxes
    state: wp.array2d(dtype=wp.int32)
    left_region: wp.array2d(dtype=wp.int32)   # delivered box that left its region this step


@wp.struct
class RobotArrays:
    """Per robot (0 the loader, then the stackers): addresses and goal
    definitions, the per-world task state, and everything the observation and
    reward read."""
    tip_sid: wp.array(dtype=wp.int32)
    tip_bid: wp.array(dtype=wp.int32)
    root_bid: wp.array(dtype=wp.int32)
    goal_sid: wp.array(dtype=wp.int32)
    grasp_act: wp.array(dtype=wp.int32)       # the robot's grasp actuator
    n_part: wp.array(dtype=wp.int32)          # moving parts of its gripper
    goal_pos0: wp.array(dtype=wp.vec3)        # goal site's local pos at reset
    field_lo: wp.array(dtype=wp.vec3)
    field_hi: wp.array(dtype=wp.vec3)
    # GOAL_SITE: a box within goal_thresh of the goal site, which then rises by
    # goal_step. GOAL_REGION: a box centre in [goal_lo, goal_hi]. The loader's
    # goal is special-cased
    goal_kind: wp.array(dtype=wp.int32)
    goal_step: wp.array(dtype=wp.float32)
    goal_lo: wp.array(dtype=wp.vec3)
    goal_hi: wp.array(dtype=wp.vec3)

    tgt: wp.array2d(dtype=wp.int32)           # target box, -1 for none
    loader_idx: wp.array(dtype=wp.int32)      # loader's position in BoxArrays.order
    dwell: wp.array2d(dtype=wp.int32)
    lost: wp.array2d(dtype=wp.int32)
    scored: wp.array2d(dtype=wp.int32)        # delivered on this step
    undelivered: wp.array2d(dtype=wp.int32)   # delivered boxes that left their region this step
    rebase: wp.array2d(dtype=wp.int32)        # target changed, re-base the shaping

    tip_pos: wp.array2d(dtype=wp.vec3)
    tip_quat: wp.array2d(dtype=wp.quat)
    tip_lvel: wp.array2d(dtype=wp.vec3)
    tip_avel: wp.array2d(dtype=wp.vec3)
    tgt_pos: wp.array2d(dtype=wp.vec3)
    tgt_quat: wp.array2d(dtype=wp.quat)
    tgt_lvel: wp.array2d(dtype=wp.vec3)
    tgt_avel: wp.array2d(dtype=wp.vec3)
    goal_pos: wp.array2d(dtype=wp.vec3)
    d_tip: wp.array2d(dtype=wp.float32)
    d_goal: wp.array2d(dtype=wp.float32)      # target to the goal site, or to the goal region
    grasped: wp.array2d(dtype=wp.float32)
    close: wp.array2d(dtype=wp.float32)
    away: wp.array2d(dtype=wp.float32)        # the tip is clear of the target
    has_tgt: wp.array2d(dtype=wp.float32)


@wp.struct
class GraspArrays:
    """mujoco_warp's contact list, and which box each gripper part touches. A box
    counts as held only if it touches every part of the robot's gripper."""
    contact_geom: wp.array(dtype=wp.vec2i)
    contact_worldid: wp.array(dtype=wp.int32)
    nacon: wp.array(dtype=wp.int32)
    geom_robot: wp.array(dtype=wp.int32)      # robot a grasp geom belongs to, or -1
    geom_part: wp.array(dtype=wp.int32)       # which part of that robot's gripper
    geom_box: wp.array(dtype=wp.int32)        # box a geom belongs to, or -1
    touch: wp.array4d(dtype=wp.int32)         # [world, robot, box, part]


@wp.struct
class BedArrays:
    """The truck bed geoms (ramp, baseplate, lip) and which boxes touch them."""
    geom_bed: wp.array(dtype=wp.int32)
    on_bed: wp.array2d(dtype=wp.int32)


@wp.struct
class EnvConst:
    seed: int
    nq: int
    nv: int
    nu: int
    n_seg: int
    n_robot: int
    n_box: int
    n_grid: int            # boxes in the container, the rest start on the belt
    dwell_limit: int
    goal_thresh: float
    away_xy: float         # radius of the cylinder the tip must leave to be clear of a box
    away_z: float
    belt_x_min: float
    cont_lo: wp.vec3       # the container's AABB
    cont_hi: wp.vec3


@wp.func
def _stream_key(lane_seed: int, tick: int) -> int:
    """The random stream of one lane seed at one step: the two hashed together, so runs seeded s and s + 1 never replay
    each other a step apart, as a plain sum seed + tick would."""
    return wp.randi(wp.rand_init(lane_seed, tick))


@wp.kernel
def tick_kernel(env: EnvArrays):
    # Launch arguments are frozen in the graph, so the RNG index must advance on device
    env.tick[0] = env.tick[0] + 1


@wp.kernel
def reset_kernel(env: EnvArrays, sim: SimArrays, dfm: DeformArrays,
                 seg: SegArrays, box: BoxArrays, rob: RobotArrays,
                 c: EnvConst, tun: wp.array(dtype=wp.float32)):
    """Respawn and apply the embodiment to one world per thread. Worlds outside
    ``mask`` return immediately."""
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

    # Each respawn draws its world's belt speed and nudges the container's boxes in x and y
    rs = wp.rand_init(_stream_key(env.lane_seed[w], env.tick[0]), w)
    speed = tun[T_BELT_VEL] * (1.0 + tun[T_BELT_SPREAD] * (2.0 * wp.randf(rs) - 1.0))
    for k in range(env.belt_gid.shape[0]):
        sv = sim.geom_surfacevel[w, env.belt_gid[k]]
        sv[0] = speed
        sim.geom_surfacevel[w, env.belt_gid[k]] = sv
    for b in range(c.n_grid):
        a = box.qadr[b]
        sim.qpos[w, a] = sim.qpos[w, a] + tun[T_BOX_JITTER] * (2.0 * wp.randf(rs) - 1.0)
        sim.qpos[w, a + 1] = sim.qpos[w, a + 1] + tun[T_BOX_JITTER] * (2.0 * wp.randf(rs) - 1.0)

    apply_offsets(dfm, seg, env.deform, env.offsets, w, c.n_seg,
                  tun[T_MAX_OFFSET])
    env.offset_scale[w] = tun[T_MAX_OFFSET]

    for b in range(c.n_box):
        box.state[w, b] = 0

    rob.loader_idx[w] = 0
    for r in range(c.n_robot):
        rob.tgt[w, r] = -1
        rob.dwell[w, r] = 0
        rob.lost[w, r] = 0
        rob.scored[w, r] = 0
        rob.rebase[w, r] = 1
        dfm.site_pos[w, rob.goal_sid[r]] = rob.goal_pos0[r]   # the stack starts over


@wp.kernel
def ctrl_kernel(env: EnvArrays, sim: SimArrays, c: EnvConst):
    """Map the staged action onto actuator ctrl, one thread per world. A grasp
    actuator has a dead zone: actions at or below 0 give the bottom of its range
    (no grasp), and (0, 1] spans the whole range."""
    w = wp.tid()
    for a in range(c.nu):
        x = wp.clamp(env.ctrl_act[w, a], -1.0, 1.0)
        if env.ctrl_deadzone[a] != 0:
            x = 2.0 * wp.max(x, 0.0) - 1.0
        sim.ctrl[w, a] = x * env.ctrl_scale[a] + env.ctrl_mid[a]


@wp.func
def _box_pos(sim: SimArrays, box: BoxArrays, w: int, b: int) -> wp.vec3:
    a = box.qadr[b]
    return wp.vec3(sim.qpos[w, a + 0], sim.qpos[w, a + 1], sim.qpos[w, a + 2])


@wp.kernel
def clear_task_bufs_kernel(adh: GraspArrays, bed: BedArrays, box: BoxArrays,
                           rob: RobotArrays, c: EnvConst):
    """Zero per-step contact/bookkeeping buffers inside the CUDA-graphed step program."""
    w = wp.tid()
    for b in range(c.n_box):
        bed.on_bed[w, b] = 0
        box.left_region[w, b] = 0
        for r in range(c.n_robot):
            for p in range(rob.n_part[r]):
                adh.touch[w, r, b, p] = 0
    for r in range(c.n_robot):
        rob.undelivered[w, r] = 0


@wp.kernel
def box_kernel(sim: SimArrays, box: BoxArrays, rob: RobotArrays, bed: BedArrays,
               c: EnvConst, mask: wp.array(dtype=wp.bool), only_mask: int):
    """Per (world, box): mark a box lost when it leaves the fields or lands on
    the bed, and make a box delivered to a region live again when it leaves it.

    A box's field is the one holding its x, or the nearest one in x. It is lost
    when it is outside every field in x, outside its field in y, or below its
    field's floor. Touching the bed loses it only outside the container's
    footprint, where the baseplate can be the container's floor.
    """
    w, b = wp.tid()
    if only_mask == 1 and not mask[w]:
        return
    if box.state[w, b] == 2:
        return

    pos = _box_pos(sim, box, w, b)

    # Checked first, so a delivered box knocked out of its field is undelivered
    # before it is marked lost
    if box.state[w, b] == 1:
        r = box.color[b] + 1
        if rob.goal_kind[r] == GOAL_REGION:
            glo = rob.goal_lo[r]
            ghi = rob.goal_hi[r]
            if (pos[0] < glo[0] or pos[0] > ghi[0] or pos[1] < glo[1] or pos[1] > ghi[1]
                    or pos[2] < glo[2] or pos[2] > ghi[2]):
                box.state[w, b] = 0
                wp.atomic_add(rob.undelivered, w, r, 1)
                box.left_region[w, b] = 1

    owner = int(0)
    best = float(1.0e30)
    for r in range(c.n_robot):
        lo = rob.field_lo[r]
        hi = rob.field_hi[r]
        gap = wp.max(lo[0] - pos[0], pos[0] - hi[0])
        if gap < best:
            best = gap
            owner = r
    lo = rob.field_lo[owner]
    hi = rob.field_hi[owner]
    # best > 0 means no field holds the box's x
    on_bed = bed.on_bed[w, b] != 0 and (
        pos[0] < c.cont_lo[0] or pos[0] > c.cont_hi[0]
        or pos[1] < c.cont_lo[1] or pos[1] > c.cont_hi[1])
    if (on_bed or best > 0.0 or pos[1] < lo[1] or pos[1] > hi[1]
            or pos[2] < lo[2]):
        box.state[w, b] = 2
        wp.atomic_add(rob.lost, w, owner, 1)
        return


@wp.kernel
def contact_kernel(grp: GraspArrays, bed: BedArrays, mask: wp.array(dtype=wp.bool), only_mask: int):
    """Per contact: mark the box it joins to a gripper part or to the bed. The
    caller clears ``touch`` and ``on_bed`` first. Every write is the same 1, so
    contacts of one pair don't race."""
    i = wp.tid()
    if i >= grp.nacon[0]:
        return
    if only_mask == 1 and not mask[grp.contact_worldid[i]]:
        return
    g = grp.contact_geom[i]
    if g[0] < 0 or g[1] < 0:        # a flex contact
        return
    r = grp.geom_robot[g[0]]
    part = grp.geom_part[g[0]]
    b = grp.geom_box[g[1]]
    if r < 0 or b < 0:
        r = grp.geom_robot[g[1]]
        part = grp.geom_part[g[1]]
        b = grp.geom_box[g[0]]
    if r >= 0 and b >= 0:
        grp.touch[grp.contact_worldid[i], r, b, part] = 1

    # The same contact, read for the bed
    other = g[0]
    box = grp.geom_box[g[1]]
    if box < 0:
        other = g[1]
        box = grp.geom_box[g[0]]
    if box >= 0 and bed.geom_bed[other] != 0:
        bed.on_bed[grp.contact_worldid[i], box] = 1


@wp.func
def _pick_loader_target(box: BoxArrays, rob: RobotArrays, c: EnvConst,
                        w: int) -> int:
    # The next box in the loader's order that is still live. The index only moves forward
    i = rob.loader_idx[w]
    while i < c.n_grid and box.state[w, box.order[i]] != 0:
        i = i + 1
    rob.loader_idx[w] = i
    if i >= c.n_grid:
        return -1
    return box.order[i]


@wp.func
def _pick_stacker_target(sim: SimArrays, box: BoxArrays, rob: RobotArrays,
                         c: EnvConst, w: int, r: int) -> int:
    # The live box of the robot's colour in its field farthest along the belt, so
    # the target is the most urgent one and never jumps back as boxes travel
    lo = rob.field_lo[r]
    hi = rob.field_hi[r]
    best = int(-1)
    best_x = float(-1.0e30)
    for b in range(c.n_box):
        if box.state[w, b] == 0 and box.color[b] == r - 1:
            p = _box_pos(sim, box, w, b)
            inside = (p[0] >= lo[0] and p[0] <= hi[0] and p[1] >= lo[1]
                      and p[1] <= hi[1] and p[2] >= lo[2] and p[2] <= hi[2])
            if inside and p[0] > best_x:
                best_x = p[0]
                best = b
    return best


@wp.kernel
def task_kernel(sim: SimArrays, dfm: DeformArrays, box: BoxArrays,
                rob: RobotArrays, adh: GraspArrays, c: EnvConst,
                mask: wp.array(dtype=wp.bool), only_mask: int):
    """Per (world, robot): resolve the target box, write what the observation and
    reward read, and count a delivery once the box has dwelt at the goal."""
    w, r = wp.tid()
    if only_mask == 1 and not mask[w]:
        return
    rob.scored[w, r] = 0
    rob.rebase[w, r] = 0

    # Drop a target that is out of play. A stacker also drops one that left its
    # field, the loader keeps its box as the belt carries it away
    t = rob.tgt[w, r]
    if t >= 0:
        if box.state[w, t] != 0:
            t = -1
        elif r > 0:
            p = _box_pos(sim, box, w, t)
            lo = rob.field_lo[r]
            hi = rob.field_hi[r]
            if (p[0] < lo[0] or p[0] > hi[0] or p[1] < lo[1] or p[1] > hi[1]
                    or p[2] < lo[2] or p[2] > hi[2]):
                t = -1
        if t < 0:
            rob.dwell[w, r] = 0

    if t < 0:
        if r == 0:
            t = _pick_loader_target(box, rob, c, w)
        else:
            t = _pick_stacker_target(sim, box, rob, c, w, r)
        if t >= 0:
            rob.rebase[w, r] = 1
            rob.dwell[w, r] = 0
    rob.tgt[w, r] = t

    sid = rob.tip_sid[r]
    bid = rob.tip_bid[r]
    tip = sim.site_xpos[w, sid]
    rob.tip_pos[w, r] = tip
    q = sim.xquat[w, bid]
    rob.tip_quat[w, r] = wp.where(q[0] < 0.0, -q, q)        # w >= 0: one encoding per orientation
    cvel = sim.cvel[w, bid]
    rob.tip_avel[w, r] = wp.spatial_top(cvel)
    # cvel is (angular, linear) at the root's subtree CoM, move it to the tip
    rob.tip_lvel[w, r] = wp.spatial_bottom(cvel) + wp.cross(
        wp.spatial_top(cvel), tip - sim.subtree_com[w, rob.root_bid[r]])
    goal = sim.site_xpos[w, rob.goal_sid[r]]
    rob.goal_pos[w, r] = goal

    if t < 0:
        rob.tgt_pos[w, r] = wp.vec3(0.0, 0.0, 0.0)
        rob.tgt_quat[w, r] = wp.quat(0.0, 0.0, 0.0, 0.0)
        rob.tgt_lvel[w, r] = wp.vec3(0.0, 0.0, 0.0)
        rob.tgt_avel[w, r] = wp.vec3(0.0, 0.0, 0.0)
        rob.d_tip[w, r] = 0.0
        rob.d_goal[w, r] = 0.0
        rob.grasped[w, r] = 0.0
        rob.close[w, r] = 0.0
        rob.away[w, r] = 0.0
        rob.has_tgt[w, r] = 0.0
        return

    a = box.qadr[t]
    d = box.dadr[t]
    pos = _box_pos(sim, box, w, t)
    bq = wp.quat(sim.qpos[w, a + 3], sim.qpos[w, a + 4],
                 sim.qpos[w, a + 5], sim.qpos[w, a + 6])
    rob.tgt_pos[w, r] = pos
    rob.tgt_quat[w, r] = wp.where(bq[0] < 0.0, -bq, bq)
    rob.tgt_lvel[w, r] = wp.vec3(sim.qvel[w, d + 0], sim.qvel[w, d + 1],
                                 sim.qvel[w, d + 2])
    rob.tgt_avel[w, r] = wp.vec3(sim.qvel[w, d + 3], sim.qvel[w, d + 4],
                                 sim.qvel[w, d + 5])
    rob.d_tip[w, r] = wp.length(pos - tip)
    if rob.goal_kind[r] == GOAL_REGION:
        glo = rob.goal_lo[r]
        ghi = rob.goal_hi[r]
        # Distance to the region, 0 anywhere inside it
        out = wp.vec3(wp.max(wp.max(glo[0] - pos[0], pos[0] - ghi[0]), 0.0),
                      wp.max(wp.max(glo[1] - pos[1], pos[1] - ghi[1]), 0.0),
                      wp.max(wp.max(glo[2] - pos[2], pos[2] - ghi[2]), 0.0))
        rob.d_goal[w, r] = wp.length(out)
    else:
        rob.d_goal[w, r] = wp.length(pos - goal)
    rob.has_tgt[w, r] = 1.0
    # Clear: the tip is outside a vertical cylinder around the box, or far
    # enough above or below it
    gap = tip - pos
    clear = (gap[0] * gap[0] + gap[1] * gap[1] > c.away_xy * c.away_xy
             or wp.abs(gap[2]) > c.away_z)
    rob.away[w, r] = wp.where(clear, 1.0, 0.0)
    # Held: the grasp actuator is on and the box touches every gripper part
    held = sim.ctrl[w, rob.grasp_act[r]] > 0.0
    for part in range(rob.n_part[r]):
        if adh.touch[w, r, t, part] == 0:
            held = False
    rob.grasped[w, r] = wp.where(held, 1.0, 0.0)

    # The loader's goal is a line across the moving belt: y and z only, once the
    # box is on the belt
    if r == 0:
        near = (wp.abs(pos[1] - goal[1]) < c.goal_thresh
                and wp.abs(pos[2] - goal[2]) < c.goal_thresh
                and pos[0] > c.belt_x_min)
    elif rob.goal_kind[r] == GOAL_REGION:
        near = rob.d_goal[w, r] <= 0.0
    else:
        near = rob.d_goal[w, r] < c.goal_thresh
    rob.close[w, r] = wp.where(near, 1.0, 0.0)

    # Dwell counts steps (not necessarily consecutive) with the box at the goal,
    # released, and the tip clear. The target is released only after this
    # step's outputs are written, so a delivery is rewarded exactly once
    if near and not held and clear:
        rob.dwell[w, r] = rob.dwell[w, r] + 1
        if rob.dwell[w, r] >= c.dwell_limit:
            rob.scored[w, r] = 1
            if r == 0:
                rob.loader_idx[w] = rob.loader_idx[w] + 1   # the box stays live for its stacker
                rob.tgt[w, r] = -1
                rob.dwell[w, r] = 0
            else:
                box.state[w, t] = 1
                if rob.goal_kind[r] == GOAL_SITE:   # the next box goes on top, or on the next shelf
                    p0 = dfm.site_pos[w, rob.goal_sid[r]]
                    dfm.site_pos[w, rob.goal_sid[r]] = p0 + wp.vec3(0.0, 0.0, rob.goal_step[r])
