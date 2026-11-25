"""Per-world link deformation driven by body names, and a design-aware spawn
height.

A body named ``*_offsettable[_<axes>][_pos|_neg][_equ<n>]`` is deformed per
world by its entries of the embodiment action:

- no axes: one entry, the link stretches along its own direction;
- one to three axes from ``xyz``: one entry per letter, each moving the
link's far end along that body-frame axis (taken pointing away from the
joint), so the link turns as well as stretches;
- ``_pos`` / ``_neg``: every entry is clipped to [0, 1] / [-1, 0];
- ``_equ<n>``: all bodies with the tag share one block of entries.

A link is never shortened below its ``min_half`` - further offsets
are clipped.

Code layout:

- find_segments, n_slots, slot_bounds, floor_reports, release_geom_frames:
host side, building the link table from a compiled model;
- SegArrays, DeformArrays, build_seg_arrays, apply_offsets, deform: device
side, rebuilding every link from its undeformed values;
- SpawnLift and the lift_* kernels: shift a respawned robot vertically so its
lowest point sits where the base design's does.
"""

from __future__ import annotations

import re

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp

_NAME_RE = re.compile(
    r"_offsettable(?:_([xyz]{1,3}))?(?:_(pos|neg))?(?:_equ(\d+))?$")
_AXIS = {"x": 0, "y": 1, "z": 2}
_EXTENT = {"pos": (0.0, 1.0), "neg": (-1.0, 0.0), None: (-1.0, 1.0)}

_PI = 3.14159265358979323846

# Warp dtypes of the find_segments columns uploaded by build_seg_arrays
SEG_DTYPE = {"geom": wp.int32, "body": wp.int32, "is_cyl": wp.int32, "tip_geom": wp.int32,
             "axis_adr": wp.int32, "axis_num": wp.int32, "axis_act": wp.int32,
             "child_adr": wp.int32, "child_num": wp.int32, "child_id": wp.int32,
             "site_adr": wp.int32, "site_num": wp.int32, "site_id": wp.int32,
             "dir": wp.vec3, "tip0": wp.vec3, "axis_dir": wp.vec3,
             "child_pos": wp.vec3, "site_pos": wp.vec3}


def unit_mass(half: float, radius: float, is_cyl: int) -> float:
    """Mass of a link's geom at unit density. Host twin of ``mass_inertia``."""
    m_cyl = np.pi * radius ** 2 * 2.0 * half
    return m_cyl if is_cyl else m_cyl + (4.0 / 3.0) * np.pi * radius ** 3


def find_segments(m: mujoco.MjModel,
                  min_link_length: float = 0.0,
                  split_capsules: tuple[str, ...] = (),
                  handled_geoms: tuple[str, ...] = ()) -> dict[str, np.ndarray]:
    """Table of every offsettable link, in body-id order.

    A link's first geom must be a capsule or cylinder running from the body
    origin, so its far end is ``2 * geom_pos``. Child bodies and sites of the
    link move with the far end, so they must sit there. Action slots are
    numbered by first appearance, and an ``_equ`` group shares one block.

    ``min_link_length`` [m] is stored per link as a half-length, capped at the
    undeformed half. ``split_capsules`` names halves of a capsule split at a
    joint (see ``_mark_split_capsules``). A link may carry one more geom, a
    sphere centred on its far end, which rides the tip. Any other geom raises
    unless ``handled_geoms`` names it (its caller deforms it).
    """
    rows, axis_dir, axis_act, axis_lo, axis_hi = [], [], [], [], []
    kid_id, kid_pos, sid, spos = [], [], [], []
    groups, n_slot = {}, 0
    for b in range(m.nbody):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        match = _NAME_RE.search(name)
        if match is None:
            continue
        axes, suffix, equ = match.groups()
        axes = axes or ""
        if len(set(axes)) != len(axes):
            raise RuntimeError(f"{name}: axis letters {axes!r} repeat; each "
                               "names one direction the link's end can move in")
        g = int(m.body_geomadr[b])
        radius, half = float(m.geom_size[g][0]), float(m.geom_size[g][1])
        direction = m.geom_pos[g] / max(half, 1e-9)
        if abs(np.linalg.norm(direction) - 1.0) > 1e-6:
            raise RuntimeError(
                f"{name}: its geom must run from the body origin (got pos "
                f"{m.geom_pos[g]}, half-length {half})")
        is_cyl = int(m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CYLINDER)
        tip_geom, tip_r = -1, 0.0
        for g2 in range(g + 1, g + int(m.body_geomnum[b])):
            g2_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g2) or f"geom {g2}"
            if g2_name in handled_geoms:
                continue
            if (tip_geom < 0 and m.geom_type[g2] == mujoco.mjtGeom.mjGEOM_SPHERE
                    and np.allclose(m.geom_pos[g2], direction * 2.0 * half, atol=1e-6)):
                tip_geom, tip_r = g2, float(m.geom_size[g2][0])
                continue
            raise RuntimeError(f"{name}: extra geom {g2_name!r} would keep its compiled mass and pose when the link "
                               "deforms; only one sphere centred on the link's far end rides it")
        kids = np.flatnonzero(m.body_parentid == b)
        sites = np.flatnonzero(m.site_bodyid == b)
        lo, hi = _EXTENT[suffix]
        n_ax = len(axes) or 1

        if equ is None or equ not in groups:
            acts = list(range(n_slot, n_slot + n_ax))
            n_slot += n_ax
            if equ is not None:
                groups[equ] = (acts, lo, hi, name)
        else:
            acts, glo, ghi, gname = groups[equ]
            if (lo, hi) != (glo, ghi):
                raise RuntimeError(
                    f"{name}: clip range {(lo, hi)} differs from {gname}'s "
                    f"{(glo, ghi)}; an _equ{equ} group shares its offsets, so "
                    "its members must share one pos/neg suffix")
            if len(acts) != n_ax:
                raise RuntimeError(
                    f"{name}: {n_ax} offset(s) against {gname}'s {len(acts)}; "
                    f"an _equ{equ} group shares its offsets, so the i-th offset "
                    "of every member is driven by the same one")
        # Named axes are flipped to point away from the joint, so a positive
        # offset never shortens the link
        dirs = [direction] if not axes else [
            -e if float(direction @ e) < 0.0 else e
            for e in (np.eye(3)[_AXIS[letter]] for letter in axes)]
        for k, d in enumerate(dirs):
            axis_dir.append(d)
            axis_act.append(acts[k])
            axis_lo.append(lo)
            axis_hi.append(hi)

        rows.append((b, g, direction, radius, half,
                     float(m.body_mass[b]) / (unit_mass(half, radius, is_cyl) + (4.0 / 3.0) * np.pi * tip_r ** 3),
                     is_cyl, direction * 2.0 * half,
                     min(0.5 * float(min_link_length), half),
                     len(axis_act) - n_ax, n_ax,
                     len(kid_id), len(kids), len(sid), len(sites), tip_geom, tip_r))
        kid_id += [int(k) for k in kids]
        kid_pos += [m.body_pos[k] for k in kids]
        sid += [int(s) for s in sites]
        spos += [m.site_pos[s] for s in sites]
    if not rows:
        raise RuntimeError("no '*_offsettable*' bodies in the model")

    cols = list(zip(*rows))
    names = ("body", "geom", "dir", "radius", "half0", "density", "is_cyl",
             "tip0", "min_half", "axis_adr", "axis_num",
             "child_adr", "child_num", "site_adr", "site_num", "tip_geom", "tip_r")
    seg = {n: np.asarray(c) for n, c in zip(names, cols)}
    seg["axis_dir"] = np.asarray(axis_dir, dtype=np.float64).reshape(-1, 3)
    seg["axis_act"] = np.asarray(axis_act, dtype=np.int64)
    seg["axis_lo"] = np.asarray(axis_lo, dtype=np.float64)
    seg["axis_hi"] = np.asarray(axis_hi, dtype=np.float64)
    seg["child_id"] = np.asarray(kid_id, dtype=np.int64)
    seg["child_pos"] = np.asarray(kid_pos, dtype=np.float64).reshape(-1, 3)
    seg["site_id"] = np.asarray(sid, dtype=np.int64)
    seg["site_pos"] = np.asarray(spos, dtype=np.float64).reshape(-1, 3)
    _mark_split_capsules(m, seg, split_capsules)
    return seg


def _mark_split_capsules(m: mujoco.MjModel, seg: dict[str, np.ndarray], names: tuple[str, ...]) -> None:
    """Flag the named halves of a capsule split at a joint (``is_cyl`` = 2).

    Two links running in opposite directions from one origin would each count
    the sphere at that origin as a near cap. A flagged half is massed as a
    cylinder plus its far cap only, so the pair weighs what the original capsule
    did. Opt-in by name, because some stock models overlap capsules at one
    origin on purpose (Ant's legs) and those keep both caps.
    """
    index = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(b)): i for i, b in enumerate(seg["body"])}
    for name in names:
        i = index.get(name)
        if i is None or seg["is_cyl"][i] != 0:
            raise RuntimeError(f"split capsule {name!r}: not an offsettable capsule link")
        b = int(seg["body"][i])
        mates = [j for j in (index[n] for n in names if n != name)
                 if m.body_parentid[int(seg["body"][j])] == m.body_parentid[b] and np.allclose(m.body_pos[int(seg["body"][j])], m.body_pos[b])
                 and float(seg["dir"][i] @ seg["dir"][j]) < -0.999 and np.isclose(seg["radius"][i], seg["radius"][j])]
        if len(mates) != 1:
            raise RuntimeError(f"split capsule {name!r}: needs exactly one opposite half from the same origin")
    for name in names:
        seg["is_cyl"][index[name]] = 2


def n_slots(seg: dict[str, np.ndarray]) -> int:
    """Design entries the links take up, groups counted once."""
    return int(seg["axis_act"].max()) + 1


def slot_bounds(seg: dict[str, np.ndarray],
                n_offset: int) -> tuple[np.ndarray, np.ndarray]:
    """(lo, hi) each embodiment entry is clipped to, from its link's pos/neg suffix.
    Entries past the links' (a caller's own) keep [-1, 1]."""
    lo = np.full(n_offset, -1.0, dtype=np.float32)
    hi = np.ones(n_offset, dtype=np.float32)
    lo[seg["axis_act"]] = seg["axis_lo"]   # group members agree, by build check
    hi[seg["axis_act"]] = seg["axis_hi"]
    return lo, hi


def length_safe_lo(seg: dict[str, np.ndarray], axis_lo: np.ndarray,
                   max_offset: float, min_length: float) -> np.ndarray:
    """Per-entry lower clip that keeps every link at least ``min_length`` long.

    A link is at least as long as its far end's projection on its own direction,
    ``L + max_offset * sum_k o_k (a_k . dir)``. Giving every entry with a
    component along the link the floor ``-(L - min_length) / (max_offset *
    sum_k a_k . dir)`` keeps that projection at ``min_length`` or more. An
    ``_equ`` slot takes the tightest floor of its members. The result is
    intersected with ``axis_lo``.
    """
    if max_offset <= 0.0:
        return np.asarray(axis_lo, dtype=np.float64).copy()
    n_slot = int(seg["axis_act"].max()) + 1
    slot_lo = np.full(n_slot, -np.inf)
    for i in range(len(seg["body"])):
        adr, num = int(seg["axis_adr"][i]), int(seg["axis_num"][i])
        proj = np.clip(seg["axis_dir"][adr:adr + num] @ seg["dir"][i], 0.0, None)
        if proj.sum() <= 1e-9:
            continue
        budget = max(2.0 * float(seg["half0"][i]) - float(min_length), 0.0)
        floor = -budget / (float(max_offset) * float(proj.sum()))
        for j in range(adr, adr + num):
            if proj[j - adr] > 1e-9:
                a = int(seg["axis_act"][j])
                slot_lo[a] = max(slot_lo[a], floor)
    return np.maximum(axis_lo, slot_lo[seg["axis_act"]]).astype(np.float64)


def release_geom_frames(m: mujoco.MjModel, seg: dict[str, np.ndarray]) -> None:
    """Make CPU kinematics pose offsettable geoms from their own
    ``geom_pos``/``geom_quat``.

    The compiler marks a geom along a body axis ``mjSAMEFRAME_BODYROT``, and
    ``mj_kinematics`` then ignores its ``geom_quat``, so a link the deform turned
    would be drawn unturned. mujoco_warp ignores ``geom_sameframe``, so only the
    CPU copies used for rendering need this.
    """
    m.geom_sameframe[seg["geom"]] = mujoco.mjtSameFrame.mjSAMEFRAME_NONE


def floor_reports(m: mujoco.MjModel, seg: dict[str, np.ndarray],
                  max_offset: float) -> list[str]:
    """One line per link whose offset range reaches its length floor, where part
    of the embodiment space builds the same link."""
    out = []
    for i, b in enumerate(seg["body"]):
        tip = np.array(seg["tip0"][i], dtype=np.float64)
        adr, num = int(seg["axis_adr"][i]), int(seg["axis_num"][i])
        for j in range(adr, adr + num):
            # The axes are orthogonal, so the shortest end point cancels each
            # axis's component as far as its range allows
            d = np.asarray(seg["axis_dir"][j], dtype=np.float64)
            tip += d * float(np.clip(-np.dot(tip, d),
                                     float(seg["axis_lo"][j]) * max_offset,
                                     float(seg["axis_hi"][j]) * max_offset))
        shortest = float(np.linalg.norm(tip))
        floor = 2.0 * float(seg["min_half"][i])
        if shortest < floor - 1e-9:
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(b))
            out.append(f"{name}: {2 * float(seg['half0'][i]):.3f} m undeformed, "
                       f"{shortest:.3f} m at its shortest, floored at "
                       f"{floor:.3f} m")
    return out


@wp.struct
class SegArrays:
    """The ``find_segments`` table on the device. Ragged blocks (axes, children,
    sites) are indexed by their ``*_adr``/``*_num`` fields."""
    geom: wp.array(dtype=wp.int32)
    body: wp.array(dtype=wp.int32)
    dir: wp.array(dtype=wp.vec3)          # undeformed unit direction, body frame
    radius: wp.array(dtype=wp.float32)
    half0: wp.array(dtype=wp.float32)     # undeformed half-length
    density: wp.array(dtype=wp.float32)
    is_cyl: wp.array(dtype=wp.int32)      # 1 cylinder, 0 capsule, 2 half of a split capsule (no near cap)
    tip_geom: wp.array(dtype=wp.int32)    # sphere riding the far end, -1 for none
    tip_r: wp.array(dtype=wp.float32)
    tip0: wp.array(dtype=wp.vec3)         # undeformed far end = dir * 2 * half0
    min_half: wp.array(dtype=wp.float32)  # shortest this link may be deformed to
    axis_adr: wp.array(dtype=wp.int32)
    axis_num: wp.array(dtype=wp.int32)
    axis_dir: wp.array(dtype=wp.vec3)
    axis_act: wp.array(dtype=wp.int32)    # action slot, an _equ group shares one
    axis_lo: wp.array(dtype=wp.float32)   # clip range, from the pos/neg suffix
    axis_hi: wp.array(dtype=wp.float32)
    child_adr: wp.array(dtype=wp.int32)   # bodies attached at the far end
    child_num: wp.array(dtype=wp.int32)
    child_id: wp.array(dtype=wp.int32)
    child_pos: wp.array(dtype=wp.vec3)    # their undeformed body_pos
    site_adr: wp.array(dtype=wp.int32)    # sites on the link
    site_num: wp.array(dtype=wp.int32)
    site_id: wp.array(dtype=wp.int32)
    site_pos: wp.array(dtype=wp.vec3)


@wp.struct
class DeformArrays:
    """Views of the model fields deformation writes to."""
    geom_pos: wp.array2d(dtype=wp.vec3)
    geom_quat: wp.array2d(dtype=wp.quat)
    geom_size: wp.array2d(dtype=wp.vec3)
    geom_rbound: wp.array2d(dtype=wp.float32)
    geom_aabb: wp.array3d(dtype=wp.vec3)
    body_pos: wp.array2d(dtype=wp.vec3)
    body_ipos: wp.array2d(dtype=wp.vec3)
    body_iquat: wp.array2d(dtype=wp.quat)
    body_mass: wp.array2d(dtype=wp.float32)
    body_inertia: wp.array2d(dtype=wp.vec3)
    site_pos: wp.array2d(dtype=wp.vec3)


def build_seg_arrays(seg: dict[str, np.ndarray], device) -> SegArrays:
    out = SegArrays()
    for name, val in seg.items():
        dtype = SEG_DTYPE.get(name, wp.float32)
        as_np = np.float32 if dtype in (wp.float32, wp.vec3) else np.int32
        setattr(out, name, wp.array(val.astype(as_np), dtype=dtype, device=device))
    return out


@wp.func
def com_shift(half: float, r: float, is_cyl: int) -> float:
    """Distance from a link's geom centre to its CoM, towards the far end:
    non-zero only for a split-capsule half, which has its far cap alone."""
    if is_cyl != 2:
        return 0.0
    m_cyl = _PI * r * r * 2.0 * half
    m_cap = (2.0 / 3.0) * _PI * r * r * r
    return m_cap * (half + 0.375 * r) / (m_cyl + m_cap)


@wp.func
def mass_inertia(half: float, r: float, rho: float, is_cyl: int) -> wp.vec4:
    """(mass, Ixx, Iyy, Izz) of a capsule/cylinder about its CoM, long axis =
    local z, as MuJoCo computes it from geometry."""
    length = 2.0 * half
    m_cyl = rho * _PI * r * r * length
    transverse = m_cyl * (3.0 * r * r + length * length) / 12.0
    if is_cyl == 1:
        return wp.vec4(m_cyl, transverse, transverse, 0.5 * m_cyl * r * r)
    m_cap = rho * (2.0 / 3.0) * _PI * r * r * r
    if is_cyl == 2:
        # Cap centroid 3r/8 past the cylinder's end, own inertia 83/320 m r^2 about it
        z = com_shift(half, r, is_cyl)
        d = half + 0.375 * r
        transverse += m_cyl * z * z + m_cap * (0.259375 * r * r + (d - z) * (d - z))
        return wp.vec4(m_cyl + m_cap, transverse, transverse, 0.5 * m_cyl * r * r + 0.4 * m_cap * r * r)
    transverse += 2.0 * m_cap * (0.4 * r * r + length * length / 4.0
                                 + 0.375 * r * length)
    axial = 0.5 * m_cyl * r * r + 0.8 * m_cap * r * r
    return wp.vec4(m_cyl + 2.0 * m_cap, transverse, transverse, axial)


@wp.func
def aim_quat(u: wp.vec3) -> wp.quat:
    """Rotation taking +z onto the unit vector ``u``.

    Built component-wise in MuJoCo's w, x, y, z order, which mujoco_warp keeps
    inside a ``wp.quat``. Warp's own quaternion helpers read the same floats as
    x, y, z, w, so they must not be used on it.
    """
    s = 1.0 + u[2]
    if s < 1.0e-9:              # u is -z: any half turn about a perpendicular axis
        return wp.quat(0.0, 1.0, 0.0, 0.0)
    inv = 1.0 / wp.sqrt(2.0 * s)
    return wp.quat(s * inv, -u[1] * inv, u[0] * inv, 0.0)


@wp.func
def deform(dfm: DeformArrays, seg: SegArrays, w: int, i: int, delta: wp.vec3):
    """Move the far end of world ``w``'s link ``i`` by ``delta`` metres.

    The link is rebuilt from its undeformed values as the capsule/cylinder from
    the body origin to the new end, so it stretches and turns. Children and
    sites at the far end move with it.
    """
    g = seg.geom[i]
    r = seg.radius[i]
    aim = seg.dir[i]
    tip = seg.tip0[i] + delta
    # Clamp the end point's component along the undeformed aim at zero, so the
    # link cannot fold back through its joint and grow again backwards
    along = wp.dot(tip, aim)
    tip = tip + aim * (wp.max(along, 0.0) - along)
    length = wp.length(tip)
    # If the end collapsed onto the joint, keep the old aim - min_half sets the length
    u = wp.where(length > 1.0e-9, tip / wp.max(length, 1.0e-9), aim)
    half = wp.max(0.5 * length, seg.min_half[i])
    pos = u * half
    quat = aim_quat(u)
    dfm.geom_pos[w, g] = pos
    dfm.geom_quat[w, g] = quat
    dfm.geom_size[w, g] = wp.vec3(r, half, 0.0)
    # Broadphase bounds must follow the size, or a stretched link gets culled
    if seg.is_cyl[i] == 1:
        dfm.geom_rbound[w, g] = wp.sqrt(r * r + half * half)
        dfm.geom_aabb[w, g, 1] = wp.vec3(r, r, half)
    else:
        dfm.geom_rbound[w, g] = half + r
        dfm.geom_aabb[w, g, 1] = wp.vec3(r, r, half + r)
    dfm.geom_aabb[w, g, 0] = wp.vec3(0.0, 0.0, 0.0)

    b = seg.body[i]
    mi = mass_inertia(half, r, seg.density[i], seg.is_cyl[i])
    z = half + com_shift(half, r, seg.is_cyl[i])
    tg = seg.tip_geom[i]
    if tg >= 0:
        # Mass the tip sphere with the link. Both lie on the link axis, so the
        # combined inertia stays diagonal in the geom frame
        rt = seg.tip_r[i]
        dfm.geom_pos[w, tg] = u * (2.0 * half)
        m_s = seg.density[i] * (4.0 / 3.0) * _PI * rt * rt * rt
        zc = (mi[0] * z + m_s * 2.0 * half) / (mi[0] + m_s)
        trans = mi[1] + mi[0] * (z - zc) * (z - zc) + 0.4 * m_s * rt * rt + m_s * (2.0 * half - zc) * (2.0 * half - zc)
        mi = wp.vec4(mi[0] + m_s, trans, trans, mi[3] + 0.4 * m_s * rt * rt)
        z = zc
    dfm.body_mass[w, b] = mi[0]
    dfm.body_inertia[w, b] = wp.vec3(mi[1], mi[2], mi[3])
    dfm.body_ipos[w, b] = u * z
    # The inertia is diagonal in the geom frame, so the principal axes turn with it
    dfm.body_iquat[w, b] = quat

    # Children follow the end as built (after clamping), not as requested
    moved = u * (2.0 * half) - seg.tip0[i]
    for k in range(seg.child_num[i]):
        j = seg.child_adr[i] + k
        dfm.body_pos[w, seg.child_id[j]] = seg.child_pos[j] + moved
    for k in range(seg.site_num[i]):
        j = seg.site_adr[i] + k
        dfm.site_pos[w, seg.site_id[j]] = seg.site_pos[j] + moved


@wp.func
def apply_offsets(dfm: DeformArrays, seg: SegArrays,
                  action: wp.array2d(dtype=wp.float32),
                  offsets: wp.array2d(dtype=wp.float32),
                  w: int, n_seg: int, max_offset: float):
    """Deform every link of world ``w`` by its entries of the embodiment action.

    The action is normalised, and ``max_offset`` [m] scales it here and nowhere
    else. The clipped values are recorded in ``offsets``.
    """
    for i in range(n_seg):
        delta = wp.vec3(0.0, 0.0, 0.0)
        for k in range(seg.axis_num[i]):
            j = seg.axis_adr[i] + k
            a = seg.axis_act[j]
            o = wp.clamp(action[w, a], seg.axis_lo[j], seg.axis_hi[j])
            offsets[w, a] = o
            delta += seg.axis_dir[j] * (o * max_offset)
        deform(dfm, seg, w, i, delta)


# --------------------------------------------------------------------------- #
# Design-aware spawn height
# --------------------------------------------------------------------------- #
# A stretched leg reaches lower, so spawning every design at the stock height
# would bury long designs in the floor. After the deform, the robot is positioned at
# qpos0 (so the drawn reset noise is untouched), the lowest point of every geom
# that can touch the floor is measured, and the root height is shifted so that
# point sits where it does for the base design. The base design therefore spawns
# exactly as before. ``margin`` additionally lifts a base design that starts
# inside the floor

# mjtGeom values, as plain ints for use inside kernels
_PLANE, _HFIELD, _SPHERE, _CAPSULE, _ELLIPSOID, _CYLINDER, _BOX = 0, 1, 2, 3, 4, 5, 6


def floor_contact_geoms(m: mujoco.MjModel, root_body: int | None = None) -> tuple[np.ndarray, float]:
    """Non-world geoms (of ``root_body``'s tree, if given) that can collide with a
    world plane/hfield, and the highest such floor's height."""
    floors = [g for g in range(m.ngeom) if m.geom_bodyid[g] == 0 and m.geom_type[g] in (_PLANE, _HFIELD)]
    if not floors:
        return np.zeros(0, np.int32), 0.0
    hits = [g for g in range(m.ngeom) if m.geom_bodyid[g] != 0 and (root_body is None or m.body_rootid[m.geom_bodyid[g]] == root_body) and any(
        (m.geom_contype[g] & m.geom_conaffinity[f]) or (m.geom_contype[f] & m.geom_conaffinity[g]) for f in floors)]
    return np.asarray(hits, np.int32), float(max(m.geom_pos[f][2] for f in floors))


def root_z_qadr(m: mujoco.MjModel, root_body: int | None = None) -> int:
    """qpos address of the root's vertical coordinate: a free joint's z, or a
    slide joint along world z on a body hanging off the world (``root_body``,
    if given). -1 if none."""
    for j in range(m.njnt):
        b = m.jnt_bodyid[j]
        if m.body_parentid[b] != 0 or (root_body is not None and b != root_body):
            continue
        if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            return int(m.jnt_qposadr[j]) + 2
        if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_SLIDE and np.allclose(np.abs(m.jnt_axis[j]), (0.0, 0.0, 1.0)):
            return int(m.jnt_qposadr[j])
    return -1


@wp.struct
class LiftArrays:
    mask: wp.array(dtype=wp.bool)
    qpos: wp.array2d(dtype=wp.float32)
    qpos0: wp.array(dtype=wp.float32)
    saved: wp.array2d(dtype=wp.float32)
    lowest: wp.array(dtype=wp.float32)
    geoms: wp.array(dtype=wp.int32)
    geom_type: wp.array(dtype=wp.int32)
    geom_size: wp.array2d(dtype=wp.vec3)
    geom_rbound: wp.array2d(dtype=wp.float32)
    geom_aabb: wp.array3d(dtype=wp.vec3)
    geom_xpos: wp.array2d(dtype=wp.vec3)
    geom_xmat: wp.array2d(dtype=wp.mat33)


@wp.kernel
def lift_stash_kernel(la: LiftArrays):
    """Resetting worlds: keep the noised pose, pose them at qpos0 to measure."""
    w, j = wp.tid()
    if not la.mask[w]:
        return
    la.saved[w, j] = la.qpos[w, j]
    la.qpos[w, j] = la.qpos0[j]


@wp.func
def geom_bottom(t: int, p: wp.vec3, R: wp.mat33, s: wp.vec3, lo: wp.vec3, hi: wp.vec3) -> float:
    """World z of the lowest point of a geom (exact for primitives, its AABB otherwise)."""
    a = wp.vec3(R[2, 0], R[2, 1], R[2, 2])
    if t == 2:                  # sphere
        return p[2] - s[0]
    if t == 3:                  # capsule
        return p[2] - wp.abs(a[2]) * s[1] - s[0]
    if t == 4:                  # ellipsoid
        return p[2] - wp.sqrt((a[0] * s[0]) * (a[0] * s[0]) + (a[1] * s[1]) * (a[1] * s[1]) + (a[2] * s[2]) * (a[2] * s[2]))
    if t == 5:                  # cylinder
        return p[2] - wp.abs(a[2]) * s[1] - s[0] * wp.sqrt(wp.max(0.0, 1.0 - a[2] * a[2]))
    if t == 6:                  # box
        return p[2] - wp.abs(a[0]) * s[0] - wp.abs(a[1]) * s[1] - wp.abs(a[2]) * s[2]
    c = p[2] + wp.dot(a, lo)
    return c - wp.abs(a[0]) * hi[0] - wp.abs(a[1]) * hi[1] - wp.abs(a[2]) * hi[2]


@wp.kernel
def lift_lowest_kernel(la: LiftArrays):
    w = wp.tid()
    if not la.mask[w]:
        return
    low = float(1.0e30)
    for k in range(la.geoms.shape[0]):
        g = la.geoms[k]
        ws = w % la.geom_size.shape[0]   # model fields may be one shared row
        wa = w % la.geom_aabb.shape[0]
        low = wp.min(low, geom_bottom(la.geom_type[g], la.geom_xpos[w, g], la.geom_xmat[w, g], la.geom_size[ws, g],
                                      la.geom_aabb[wa, g, 0], la.geom_aabb[wa, g, 1]))
    la.lowest[w] = low


@wp.kernel
def lift_topup_kernel(la: LiftArrays, zadr: int, threshold: wp.array(dtype=wp.float32)):
    """After the shift, at the pose the reset actually drew: a world whose noise
    tilted a geom below ``threshold`` is raised just enough to clear it."""
    w = wp.tid()
    if not la.mask[w]:
        return
    if la.lowest[w] < threshold[0] - 1.0e-5:        # ignore rounding on a noise-free reset
        la.qpos[w, zadr] = la.qpos[w, zadr] + threshold[0] - la.lowest[w]


@wp.kernel
def lift_apply_kernel(la: LiftArrays, zadr: int, target: wp.array(dtype=wp.float32)):
    """Restore the noised pose, shifted so the lowest point lands on ``target``."""
    w, j = wp.tid()
    if not la.mask[w]:
        return
    q = la.saved[w, j]
    if j == zadr:
        q = q + target[0] - la.lowest[w]
    la.qpos[w, j] = q


class SpawnLift:
    """Host side of the spawn lift: buffers, calibration and launches.

    ``launch`` must follow the reset kernel, and only records launches, so it
    can be part of the captured step program. It clears the floor only, limbs
    overlapping each other are left to the solver. Its kinematics passes cover
    every world but change only the derived poses of worlds it does not lift.
    Holds the lift buffers, the target height of the lowest point, and the
    threshold that reset noise may not push a geom below.
    """

    def __init__(self, m: mujoco.MjModel, gpu_model, gpu_data, mask: wp.array, qpos0: wp.array,
                 num_envs: int, device, margin: float | None = None, root_body: int | None = None):
        geoms, self.floor_z = floor_contact_geoms(m, root_body)
        self.zadr = root_z_qadr(m, root_body)
        self.enabled = geoms.size > 0 and self.zadr >= 0
        self.margin = margin
        self.nq = int(m.nq)
        self.gpu_model, self.gpu_data = gpu_model, gpu_data
        la = LiftArrays()
        la.mask = mask
        la.qpos = gpu_data.qpos
        la.qpos0 = qpos0
        la.saved = wp.zeros((num_envs, self.nq), dtype=wp.float32, device=device)
        la.lowest = wp.zeros(num_envs, dtype=wp.float32, device=device)
        la.geoms = wp.array(geoms, dtype=wp.int32, device=device)
        la.geom_type = gpu_model.geom_type
        la.geom_size = gpu_model.geom_size
        la.geom_rbound = gpu_model.geom_rbound
        la.geom_aabb = gpu_model.geom_aabb
        la.geom_xpos = gpu_data.geom_xpos
        la.geom_xmat = gpu_data.geom_xmat
        self.la = la
        self.target = wp.zeros(1, dtype=wp.float32, device=device)
        # The floor, or the base design's own lowest point if that is lower
        # (a model that starts lying on the ground)
        self.threshold = wp.zeros(1, dtype=wp.float32, device=device)
        self.base_bottom = None

    def calibrate(self) -> None:
        """Set the target from world 0's current design, which must be the base
    design: its lowest point at qpos0, raised to the floor plus ``margin`` if
    given. Leaves qpos untouched."""
        if not self.enabled:
            return
        la, n = self.la, self.la.mask.shape[0]
        mask_prev = wp.clone(la.mask)
        la.mask.fill_(True)
        wp.launch(lift_stash_kernel, dim=(n, self.nq), inputs=[la])
        mjw.kinematics(self.gpu_model, self.gpu_data)
        wp.launch(lift_lowest_kernel, dim=n, inputs=[la])
        self.base_bottom = float(la.lowest.numpy()[0])
        self.target.fill_(self.base_bottom if self.margin is None else max(self.base_bottom, self.floor_z + self.margin))
        self.threshold.fill_(min(self.floor_z, float(self.target.numpy()[0])))
        wp.copy(la.qpos, la.saved)
        wp.copy(la.mask, mask_prev)
        mjw.kinematics(self.gpu_model, self.gpu_data)

    def launch(self) -> None:
        if not self.enabled:
            return
        la, n = self.la, self.la.mask.shape[0]
        wp.launch(lift_stash_kernel, dim=(n, self.nq), inputs=[la])
        mjw.kinematics(self.gpu_model, self.gpu_data)
        wp.launch(lift_lowest_kernel, dim=n, inputs=[la])
        wp.launch(lift_apply_kernel, dim=(n, self.nq), inputs=[la, self.zadr, self.target])
        # Re-measure at the drawn pose and raise worlds whose noise dips below the threshold
        mjw.kinematics(self.gpu_model, self.gpu_data)
        wp.launch(lift_lowest_kernel, dim=n, inputs=[la])
        wp.launch(lift_topup_kernel, dim=n, inputs=[la, self.zadr, self.threshold])
