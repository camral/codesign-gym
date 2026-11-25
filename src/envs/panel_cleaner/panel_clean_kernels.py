"""Device code for the panel cleaner.

Launched at the head of every control step, before the physics substeps:

- tick_kernel: advance the RNG stream index;
- force_kernel (optional): the random push on the base;
- leg_noise_kernel (optional): random per-step forces on the legs;
- ctrl_kernel: map the policy action onto actuator ctrl.

Launched by the env outside the step program:

- reset_kernel: respawn and apply the embodiment to the worlds in ``mask``, at
a reset or after the step program at an autoreset;
- goal_kernel: where the roller is relative to the current goal, after physics;
- body_panel_contact_kernel: flag worlds where anything but the roller presses
on a panel;
- ep_increment_kernel, terrain_kernel, panel_height_kernel,
midpoint_height_kernel, terrain_lift_kernel: new terrain per episode, with the
panels, midpoints and robot re-seated on it.

The structs group the kernel arguments: ``SimArrays`` (views of mujoco_warp model
and data), ``EnvArrays`` (env-owned per-world buffers), ``PanelArrays``,
``GoalArrays`` and ``LegArrays`` (task tables), and ``EnvConst`` (capture-time
constants). Live tunables are a float array indexed by the ``T_*`` constants.
"""

from __future__ import annotations

import math

import warp as wp

from .offsettable_logic import DeformArrays, SegArrays, LiftArrays, apply_offsets, geom_bottom

_PI = math.pi

# --- live tunables ----------------------------------------------------- #
T_MAX_OFFSET = 0     # metres per unit of link embodiment action
T_FORCE_MAG = 1
T_RAND_POS = 2       # spawn xy jitter, half-side of a square [m]
T_RAND_HEADING = 3   # spawn yaw jitter, +/- [rad]
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
    cvel: wp.array2d(dtype=wp.spatial_vector)
    subtree_com: wp.array2d(dtype=wp.vec3)
    site_xpos: wp.array2d(dtype=wp.vec3)
    site_xmat: wp.array2d(dtype=wp.mat33)
    geom_margin: wp.array2d(dtype=wp.float32)
    geom_rgba: wp.array2d(dtype=wp.vec4)
    body_pos: wp.array2d(dtype=wp.vec3)      # moves the goal marker
    xfrc_applied: wp.array2d(dtype=wp.spatial_vector)


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
    force_phase: wp.array(dtype=wp.int32)
    force_cycle: wp.array(dtype=wp.int32)
    force_dir: wp.array2d(dtype=wp.float32)
    tick: wp.array(dtype=wp.int32)            # RNG stream index, advanced every step
    lane_seed: wp.array(dtype=wp.int32)        # per-world seed: the run's, or the world's own from a per-lane sequence
    ep_seed: wp.array(dtype=wp.int32)         # stream the world's episode was spawned from


@wp.struct
class PanelArrays:
    """Geoms of the panels' visual cells and the colour of an uncleaned cell."""
    cell: wp.array(dtype=wp.int32)
    rgba0: wp.vec4


@wp.struct
class GoalArrays:
    """The goals in visiting order with their acceptance boxes, each world's
    current goal, and the per-world results ``goal_kernel`` writes."""
    site: wp.array(dtype=wp.int32)
    idx: wp.array(dtype=wp.int32)
    hx: wp.array(dtype=wp.float32)         # acceptance half-extents in the goal site's frame
    hy: wp.array(dtype=wp.float32)
    tol: wp.array(dtype=wp.float32)
    standoff: wp.array(dtype=wp.float32)   # target height of the roller above the site
    roller_pos: wp.array(dtype=wp.vec3)
    roller_vel: wp.array(dtype=wp.vec3)
    roller_quat: wp.array(dtype=wp.quat)
    pos: wp.array(dtype=wp.vec3)           # point the roller aims at (world frame)
    normal: wp.array(dtype=wp.vec3)        # goal surface normal (base yaw-local frame)
    rel_local: wp.array(dtype=wp.vec3)     # roller - goal (base yaw-local frame)
    roller_vel_local: wp.array(dtype=wp.vec3)  # roller velocity (base yaw-local frame)
    dist: wp.array(dtype=wp.float32)
    at_goal: wp.array(dtype=wp.bool)
    left_field: wp.array(dtype=wp.bool)


@wp.struct
class LegArrays:
    """Geoms of the switchable leg chains and their compile-time margin and colour."""
    geom: wp.array(dtype=wp.int32)
    margin0: wp.array(dtype=wp.float32)
    rgba0: wp.array(dtype=wp.vec4)
    off_margin: float
    n_geom: int


@wp.struct
class EnvConst:
    seed: int
    nq: int
    nv: int
    nu: int
    n_seg: int
    n_slot: int          # link embodiment slots, groups counted once
    n_legparam: int      # 1 if the embodiment includes the leg switch, else 0
    n_cell: int
    force_every: int
    force_hold: int
    base_bid: int
    base_qadr: int       # qpos address of the base free joint, -1 if none
    roller_bid: int
    root_bid: int        # root of the roller's tree, for its velocity
    marker_bid: int
    n_goals: int
    marker_lift: float
    roller_align_tol: float   # max |dot(roller_y, normal)| for cell cleaning (<= 0 disables)
    field_pos: wp.vec3
    field_size: wp.vec3


@wp.func
def _stream_key(lane_seed: int, tick: int) -> int:
    """The random stream of one lane seed at one step: the two hashed together, so runs seeded s and s + 1 never replay
    each other a step apart, as a plain sum seed + tick would."""
    return wp.randi(wp.rand_init(lane_seed, tick))


@wp.kernel
def tick_kernel(env: EnvArrays):
    # Launch arguments are frozen in the graph, so the RNG index must advance on device
    env.tick[0] = env.tick[0] + 1


@wp.func
def _set_legs(sim: SimArrays, leg: LegArrays, w: int, on: bool):
    """Switch world ``w``'s leg chains on or off.

    contype/conaffinity filtering is baked into the geom-pair list at upload, so
    it cannot vary per world. The margin is read per world, and ``off_margin``
    fails every contact test. Zero alpha hides the chain.
    """
    for k in range(leg.n_geom):
        g = leg.geom[k]
        c = leg.rgba0[k]
        sim.geom_margin[w, g] = wp.where(on, leg.margin0[k], leg.off_margin)
        sim.geom_rgba[w, g] = wp.where(on, c, wp.vec4(c[0], c[1], c[2], 0.0))


@wp.kernel
def reset_kernel(env: EnvArrays, sim: SimArrays, dfm: DeformArrays,
                 seg: SegArrays, pan: PanelArrays, leg: LegArrays,
                 c: EnvConst, tun: wp.array(dtype=wp.float32)):
    """Respawn and apply the embodiment to one world per thread. Worlds outside
    ``mask`` return immediately."""
    w = wp.tid()
    if not env.mask[w]:
        return

    for j in range(c.nq):
        sim.qpos[w, j] = env.qpos_init[j]

    rp = tun[T_RAND_POS]
    rh = tun[T_RAND_HEADING]
    if c.base_qadr >= 0 and (rp > 0.0 or rh > 0.0):
        a = c.base_qadr
        rs = wp.rand_init(_stream_key(env.lane_seed[w], env.tick[0]), w)
        sim.qpos[w, a + 0] = sim.qpos[w, a + 0] + rp * (2.0 * wp.randf(rs) - 1.0)
        sim.qpos[w, a + 1] = sim.qpos[w, a + 1] + rp * (2.0 * wp.randf(rs) - 1.0)
        # Yaw about world z, pre-multiplied onto the keyframe quat (wxyz)
        h = 0.5 * rh * (2.0 * wp.randf(rs) - 1.0)                  # half the yaw angle
        cy, sy = wp.cos(h), wp.sin(h)
        q0 = sim.qpos[w, a + 3]
        q1 = sim.qpos[w, a + 4]
        q2 = sim.qpos[w, a + 5]
        q3 = sim.qpos[w, a + 6]
        sim.qpos[w, a + 3] = cy * q0 - sy * q3
        sim.qpos[w, a + 4] = cy * q1 - sy * q2
        sim.qpos[w, a + 5] = cy * q2 + sy * q1
        sim.qpos[w, a + 6] = cy * q3 + sy * q0
    # Clear the solver state too: a warm start carried over from a terminal
    # state can diverge the new episode
    for j in range(c.nv):
        sim.qvel[w, j] = 0.0
        sim.qacc[w, j] = 0.0
        sim.qacc_warmstart[w, j] = 0.0
    sim.time[w] = 0.0

    env.force_phase[w] = 0
    env.force_cycle[w] = 0
    env.ep_seed[w] = _stream_key(env.lane_seed[w], env.tick[0])

    # Undo the last episode's cleaning
    for k in range(c.n_cell):
        sim.geom_rgba[w, pan.cell[k]] = pan.rgba0

    apply_offsets(dfm, seg, env.deform, env.offsets, w, c.n_seg,
                  tun[T_MAX_OFFSET])

    # On above zero, off otherwise, so the all-zero embodiment has no legs
    if c.n_legparam > 0:
        i = c.n_slot
        on = env.deform[w, i] > 0.0
        env.offsets[w, i] = wp.where(on, 1.0, 0.0)
        _set_legs(sim, leg, w, on)


@wp.kernel
def force_kernel(env: EnvArrays, sim: SimArrays, c: EnvConst,
                 tun: wp.array(dtype=wp.float32)):
    """Random-heading horizontal push on the base, one thread per world.

    A heading is drawn at the start of every ``force_every``-step cycle and
    applied for the cycle's last ``force_hold`` steps.
    """
    w = wp.tid()

    p = env.force_phase[w]

    if p == 0:
        cyc = env.force_cycle[w]
        # Sequence numbers past num_envs keep this stream apart from the spawn's
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
def ctrl_kernel(env: EnvArrays, sim: SimArrays, c: EnvConst):
    w = wp.tid()
    for a in range(c.nu):
        sim.ctrl[w, a] = (wp.clamp(env.ctrl_act[w, a], -1.0, 1.0)
                          * env.ctrl_scale[a] + env.ctrl_mid[a])


@wp.func
def _yaw_local(v: wp.vec3, cy: float, sy: float) -> wp.vec3:
    """Rotate world-frame vector ``v`` into the base's yaw-aligned local frame."""
    return wp.vec3(cy * v[0] + sy * v[1], -sy * v[0] + cy * v[1], v[2])


@wp.kernel
def zero_int_kernel(buf: wp.array(dtype=wp.int32)):
    """Zero an int32 per-world buffer inside the CUDA graph."""
    buf[wp.tid()] = 0


@wp.kernel
def paint_cleaned_kernel(
    cleaned: wp.array(dtype=wp.bool),
    goal_idx: wp.array(dtype=wp.int32),
    goal_is_cell: wp.array(dtype=wp.bool),
    goal_gid: wp.array(dtype=wp.int32),
    n_goals: int,
    clean_rgba: wp.vec4,
    geom_rgba: wp.array2d(dtype=wp.vec4),
):
    """Paint just-cleaned cells in-place on device without host sync or temp tensors."""
    w = wp.tid()
    if not cleaned[w]:
        return
    g = wp.min(goal_idx[w], n_goals - 1)
    if goal_is_cell[g]:
        geom_rgba[w, goal_gid[g]] = clean_rgba


@wp.kernel
def goal_kernel(sim: SimArrays, goal: GoalArrays, c: EnvConst,
                mask: wp.array(dtype=wp.bool), only: int):
    """Roller state relative to each world's current goal, after the physics.

    The goal frame is its site's frame, with z along the surface normal. The
    roller is at the goal when its centre is inside the goal's box, centred
    ``standoff`` above the site, and (for cells) its roller axis is aligned with
    the panel surface. Also moves the goal marker onto the goal and computes
    base yaw-local goal/velocity vectors for rotation-invariant observations.
    With ``only`` set, only the worlds in ``mask`` are updated.
    """
    w = wp.tid()
    if only != 0 and not mask[w]:
        return
    g = wp.min(goal.idx[w], c.n_goals - 1)
    s = goal.site[g]
    origin = sim.site_xpos[w, s]
    rot = sim.site_xmat[w, s]
    normal = wp.vec3(rot[0, 2], rot[1, 2], rot[2, 2])

    sim.body_pos[w, c.marker_bid] = origin + normal * c.marker_lift   # a static body, so a world pose

    # Base yaw frame (cos_yaw, sin_yaw) from base quaternion (w, x, y, z)
    bq = sim.xquat[w, c.base_bid]
    siny = 2.0 * (bq[0] * bq[3] + bq[1] * bq[2])
    cosy = 1.0 - 2.0 * (bq[2] * bq[2] + bq[3] * bq[3])
    ynorm = wp.max(wp.sqrt(cosy * cosy + siny * siny), 1.0e-9)
    cy = cosy / ynorm
    sy = siny / ynorm

    pos = sim.xpos[w, c.roller_bid]
    cvel = sim.cvel[w, c.roller_bid]
    # cvel is (angular, linear) at the root's subtree CoM, move it to the roller
    rvel = wp.spatial_bottom(cvel) + wp.cross(
        wp.spatial_top(cvel), pos - sim.subtree_com[w, c.root_bid])
    goal.roller_vel[w] = rvel
    goal.roller_vel_local[w] = _yaw_local(rvel, cy, sy)
    goal.roller_pos[w] = pos
    q = sim.xquat[w, c.roller_bid]
    goal.roller_quat[w] = wp.where(q[0] < 0.0, -q, q)   # w >= 0: one encoding per orientation

    stand = goal.standoff[g]
    aim = origin + normal * stand
    goal.pos[w] = aim
    goal.normal[w] = _yaw_local(normal, cy, sy)
    goal.rel_local[w] = _yaw_local(pos - aim, cy, sy)
    goal.dist[w] = wp.length(pos - aim)

    d = pos - origin
    lx = wp.dot(wp.vec3(rot[0, 0], rot[1, 0], rot[2, 0]), d)
    ly = wp.dot(wp.vec3(rot[0, 1], rot[1, 1], rot[2, 1]), d)
    lz = wp.dot(normal, d)
    in_box = (wp.abs(lx) <= goal.hx[g] and wp.abs(ly) <= goal.hy[g]
              and wp.abs(lz - stand) <= goal.tol[g])

    # Roller cylinder runs along its body y-axis; require it roughly parallel to panel surface on cells
    roller_y = wp.vec3(
        2.0 * (q[1] * q[2] - q[0] * q[3]),
        1.0 - 2.0 * (q[1] * q[1] + q[3] * q[3]),
        2.0 * (q[2] * q[3] + q[0] * q[1]),
    )
    aligned = (stand <= 0.0 or c.roller_align_tol <= 0.0
               or wp.abs(wp.dot(roller_y, normal)) <= c.roller_align_tol)
    goal.at_goal[w] = in_box and aligned

    off = pos - c.field_pos
    goal.left_field[w] = (wp.abs(off[0]) > c.field_size[0]
                          or wp.abs(off[1]) > c.field_size[1]
                          or wp.abs(off[2]) > c.field_size[2])


@wp.kernel
def body_panel_contact_kernel(
    contact_geom: wp.array(dtype=wp.vec2i),
    contact_worldid: wp.array(dtype=wp.int32),
    contact_dist: wp.array(dtype=wp.float32),
    nacon: wp.array(dtype=wp.int32),
    plate_mask: wp.array(dtype=wp.int32),
    roller_gid: wp.int32,
    depth: float,
    out_bad: wp.array(dtype=wp.int32),
    mask: wp.array(dtype=wp.bool),
    only: int,
):
    """Set ``out_bad[world]`` for every contact deeper than ``depth`` between a
    panel plate and anything but the roller. One thread per contact slot. With
    ``only`` set, only the worlds in ``mask`` are updated."""
    tid = wp.tid()
    if tid >= nacon[0]:
        return
    if only != 0 and not mask[contact_worldid[tid]]:
        return
    if contact_dist[tid] > -depth:                  # contact_dist < 0 is penetration
        return
    pair = contact_geom[tid]
    g0 = pair[0]
    g1 = pair[1]
    if plate_mask[g0] == 0 and plate_mask[g1] == 0:
        return
    if g0 == roller_gid or g1 == roller_gid:
        return
    out_bad[contact_worldid[tid]] = 1


@wp.func
def _hash2(ix: int, iy: int, seed: int) -> float:
    # Integer hash of a lattice point to [0, 1)
    h = wp.uint32(ix) * wp.uint32(374761393)
    h = h + wp.uint32(iy) * wp.uint32(668265263)
    h = h + wp.uint32(seed) * wp.uint32(987654323)
    h = h ^ (h >> wp.uint32(13))
    h = h * wp.uint32(1274126177)
    h = h ^ (h >> wp.uint32(16))
    return float(h & wp.uint32(16777215)) / 16777216.0


@wp.func
def _vnoise(x: float, y: float, seed: int) -> float:
    # Value noise: smoothstep interpolation of the hashed lattice corners
    ix = int(wp.floor(x))
    iy = int(wp.floor(y))
    fx = x - float(ix)
    fy = y - float(iy)
    ux = fx * fx * (3.0 - 2.0 * fx)
    uy = fy * fy * (3.0 - 2.0 * fy)
    a = _hash2(ix, iy, seed)
    b = _hash2(ix + 1, iy, seed)
    c = _hash2(ix, iy + 1, seed)
    d = _hash2(ix + 1, iy + 1, seed)
    return wp.lerp(wp.lerp(a, b, ux), wp.lerp(c, d, ux), uy)


@wp.kernel
def ep_increment_kernel(mask: wp.array(dtype=wp.bool),
                        ep_count: wp.array(dtype=wp.int32)):
    # A separate launch, so terrain_kernel reads the settled count (its seed)
    w = wp.tid()
    if not mask[w]:
        return
    ep_count[w] = ep_count[w] + 1


@wp.kernel
def terrain_kernel(mask: wp.array(dtype=wp.bool),
                   ep_count: wp.array(dtype=wp.int32),
                   hfield_data: wp.array(dtype=wp.float32),
                   hfield_adr: wp.array(dtype=wp.int32),
                   hfield_nrow: wp.array(dtype=wp.int32),
                   hfield_ncol: wp.array(dtype=wp.int32),
                   octaves: int, base_freq: float, lacunarity: float,
                   gain: float, contrast: float, base_seed: int,
                   lane_seed: wp.array(dtype=wp.int32)):
    """Fill each resetting world's heightfield with fractal value noise, seeded
    per world and per episode. One thread per (world, cell)."""
    w, idx = wp.tid()
    if not mask[w]:
        return
    nr = hfield_nrow[w]
    nc = hfield_ncol[w]
    if idx >= nr * nc:
        return
    r = idx // nc
    c = idx % nc
    u = float(c) / float(nc - 1)
    v = float(r) / float(nr - 1)
    seed = _stream_key(lane_seed[w] + base_seed, ep_count[w]) + w * 1000003   # per world seed and episode
    amp = float(1.0)
    freq = float(base_freq)
    total = float(0.0)
    norm = float(0.0)
    for o in range(octaves):
        total += amp * _vnoise(u * freq, v * freq, seed + o * 101)
        norm += amp
        amp *= gain
        freq *= lacunarity
    # fBm clusters around 0.5, so stretch about it before clamping to [0, 1]
    h = (total / wp.max(norm, 1.0e-6) - 0.5) * contrast + 0.5
    hfield_data[hfield_adr[w] + idx] = wp.clamp(h, 0.0, 1.0)


@wp.kernel
def panel_height_kernel(mask: wp.array(dtype=wp.bool),
                        hfield_data: wp.array(dtype=wp.float32),
                        hfield_adr: wp.array(dtype=wp.int32),
                        hfield_nrow: wp.array(dtype=wp.int32),
                        hfield_ncol: wp.array(dtype=wp.int32),
                        hfield_size: wp.array(dtype=wp.vec4),
                        panel_bid: wp.array(dtype=wp.int32),
                        panel_uv: wp.array(dtype=wp.vec2),
                        base_h: float, half_win: int,
                        body_pos: wp.array2d(dtype=wp.vec3)):
    """Set each panel's height from the lowest terrain in a (2 * half_win + 1)^2
    cell window under its support, so the support always reaches the ground."""
    w, p = wp.tid()
    if not mask[w]:
        return
    nr = hfield_nrow[w]
    nc = hfield_ncol[w]
    uv = panel_uv[p]
    c0 = int(wp.clamp(uv[0], 0.0, 1.0) * float(nc - 1) + 0.5)
    r0 = int(wp.clamp(uv[1], 0.0, 1.0) * float(nr - 1) + 0.5)
    lo = float(1.0e9)
    for dr in range(-half_win, half_win + 1):
        for dc in range(-half_win, half_win + 1):
            rr = wp.clamp(r0 + dr, 0, nr - 1)
            cc = wp.clamp(c0 + dc, 0, nc - 1)
            lo = wp.min(lo, hfield_data[hfield_adr[w] + rr * nc + cc])
    bid = panel_bid[p]
    pos = body_pos[w, bid]
    body_pos[w, bid] = wp.vec3(pos[0], pos[1], base_h + lo * hfield_size[w][2])


@wp.kernel
def midpoint_height_kernel(mask: wp.array(dtype=wp.bool), mid_site: wp.array(dtype=wp.int32), mid_panels: wp.array2d(dtype=wp.int32),
                           mid_z0: wp.array(dtype=wp.float32), body_pos: wp.array2d(dtype=wp.vec3), site_pos: wp.array2d(dtype=wp.vec3)):
    """Keep each midpoint site at its compiled height above the mean height of
    the two panels around it. Runs after panel_height_kernel."""
    w, k = wp.tid()
    if not mask[w]:
        return
    s = mid_site[k]
    p = site_pos[w, s]
    zo = body_pos[w, mid_panels[k, 0]][2]
    zi = body_pos[w, mid_panels[k, 1]][2]
    site_pos[w, s] = wp.vec3(p[0], p[1], mid_z0[k] + 0.5 * (zo + zi))


@wp.kernel
def terrain_lift_kernel(la: LiftArrays, zadr: int,
                        hfield_data: wp.array(dtype=wp.float32),
                        hfield_adr: wp.array(dtype=wp.int32),
                        hfield_nrow: wp.array(dtype=wp.int32),
                        hfield_ncol: wp.array(dtype=wp.int32),
                        hfield_size: wp.array(dtype=wp.vec4),
                        x0: float, y0: float, wx: float, wy: float,
                        base_h: float, clearance: float):
    """Shift each resetting robot's root height so its floor-touching geoms clear
    the new terrain by ``clearance``. Runs after terrain_kernel and kinematics."""
    w = wp.tid()
    if not la.mask[w]:
        return
    nr = hfield_nrow[w]
    nc = hfield_ncol[w]
    hz = hfield_size[w][2]
    gap = float(1.0e30)
    for k in range(la.geoms.shape[0]):
        g = la.geoms[k]
        p = la.geom_xpos[w, g]
        ws = w % la.geom_size.shape[0]
        rb = la.geom_rbound[w % la.geom_rbound.shape[0], g]
        # Highest ground under the geom's centre and four points a bounding radius
        # out. Each point takes its cell's highest corner, since the triangulated
        # surface never rises above it
        ground = float(-1.0e30)
        for q in range(5):
            dx = float(0.0)
            dy = float(0.0)
            if q == 1:
                dx = rb
            if q == 2:
                dx = -rb
            if q == 3:
                dy = rb
            if q == 4:
                dy = -rb
            fc = wp.clamp((p[0] + dx - x0) / wx, 0.0, 1.0) * float(nc - 1)
            fr = wp.clamp((p[1] + dy - y0) / wy, 0.0, 1.0) * float(nr - 1)
            c0 = wp.min(int(fc), nc - 2)
            r0 = wp.min(int(fr), nr - 2)
            for cr in range(4):
                v = hfield_data[hfield_adr[w] + (r0 + cr / 2) * nc + c0 + cr % 2]
                ground = wp.max(ground, base_h + v * hz)
        wa = w % la.geom_aabb.shape[0]
        bottom = geom_bottom(la.geom_type[g], p, la.geom_xmat[w, g], la.geom_size[ws, g],
                             la.geom_aabb[wa, g, 0], la.geom_aabb[wa, g, 1])
        gap = wp.min(gap, bottom - ground)
    la.qpos[w, zadr] = la.qpos[w, zadr] + clearance - gap


@wp.kernel
def leg_noise_kernel(tick: wp.array(dtype=wp.int32),
                     lane_seed: wp.array(dtype=wp.int32),
                     leg_bids: wp.array(dtype=wp.int32),
                     mag: float,
                     xfrc: wp.array2d(dtype=wp.spatial_vector)):
    """A new random force on every leg body each step, standing in for a rough
    surface. One thread per (world, leg)."""
    w, i = wp.tid()
    bid = leg_bids[i]
    s = wp.rand_init(_stream_key(lane_seed[w], tick[0]), w * 131 + i)
    fx = (wp.randf(s) * 2.0 - 1.0) * mag
    fy = (wp.randf(s) * 2.0 - 1.0) * mag
    fz = (wp.randf(s) * 2.0 - 1.0) * mag
    xfrc[w, bid] = wp.spatial_vector(fx, fy, fz, 0.0, 0.0, 0.0)
