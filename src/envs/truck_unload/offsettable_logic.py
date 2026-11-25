"""Per-world link deformation driven by body names.

A body named ``*_offsettable[_<axes>][_pos|_neg][_equ<n>]`` is deformed per
world by its entries of the embodiment action:

- no axes: one entry, the link stretches along its own direction;
- one to three axes from ``xyz``: one entry per letter, each moving the
link's far end along that body-frame axis (taken pointing away from the
joint), so the link turns as well as stretches;
- ``_pos`` / ``_neg``: every entry is clipped to [0, 1] / [-1, 0];
- ``_equ<n>``: all bodies with the tag share one block of entries.

A link's first geom is rebuilt, and may be a capsule, a cylinder or a box. A box
keeps its cross-section and only changes length. Child bodies and sites at the
far end move with it. A link is never shortened below its ``min_half``.

Code layout:

- find_segments, n_slots, slot_bounds, length_safe_lo, floor_reports,
release_geom_frames: host side, building the link table from a compiled model;
- SegArrays, DeformArrays, build_seg_arrays, apply_offsets, deform: device
side, rebuilding every link from its undeformed values.
"""

from __future__ import annotations

import re

import mujoco
import numpy as np
import warp as wp

_NAME_RE = re.compile(
    r"_offsettable(?:_([xyz]{1,3}))?(?:_(pos|neg))?(?:_equ(\d+))?$")
_AXIS = {"x": 0, "y": 1, "z": 2}
_EXTENT = {"pos": (0.0, 1.0), "neg": (-1.0, 0.0), None: (-1.0, 1.0)}

_PI = 3.14159265358979323846

# Link geom kinds, and which geom_size entry is the half-length
CAPSULE, CYLINDER, BOX = 0, 1, 2
_KIND = {int(mujoco.mjtGeom.mjGEOM_CAPSULE): CAPSULE,
         int(mujoco.mjtGeom.mjGEOM_CYLINDER): CYLINDER,
         int(mujoco.mjtGeom.mjGEOM_BOX): BOX}
_HALF_IDX = {CAPSULE: 1, CYLINDER: 1, BOX: 2}

# Warp dtypes of the find_segments columns uploaded by build_seg_arrays
SEG_DTYPE = {"geom": wp.int32, "body": wp.int32, "kind": wp.int32,
             "axis_adr": wp.int32, "axis_num": wp.int32, "axis_act": wp.int32,
             "child_adr": wp.int32, "child_num": wp.int32, "child_id": wp.int32,
             "site_adr": wp.int32, "site_num": wp.int32, "site_id": wp.int32,
             "dir": wp.vec3, "tip0": wp.vec3, "axis_dir": wp.vec3,
             "child_pos": wp.vec3, "site_pos": wp.vec3, "cross": wp.vec2}


def unit_mass(half: float, cross, kind: int) -> float:
    """Mass of a link's geom at unit density. Host twin of ``mass_inertia``."""
    cx, cy = float(cross[0]), float(cross[1])
    if kind == BOX:
        return 8.0 * cx * cy * half
    m_cyl = np.pi * cx ** 2 * 2.0 * half
    return m_cyl if kind == CYLINDER else m_cyl + (4.0 / 3.0) * np.pi * cx ** 3


def find_segments(m: mujoco.MjModel,
                  min_link_length: float = 0.0) -> dict[str, np.ndarray]:
    """Table of every offsettable link, in body-id order.

    A link's first geom must run from the body origin, so its far end is
    ``2 * geom_pos``. Child bodies and sites of the link move with the far end,
    so they must sit there. Action slots are numbered by first appearance, and
    an ``_equ`` group shares one block. ``min_link_length`` [m] is stored per
    link as a half-length, capped at the undeformed half.
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
        kind = _KIND.get(int(m.geom_type[g]))
        if kind is None:
            raise RuntimeError(
                f"{name}: its first geom is geom type {int(m.geom_type[g])}; an "
                "offsettable link is rebuilt as a capsule, a cylinder or a box")
        size = m.geom_size[g]
        half = float(size[_HALF_IDX[kind]])
        cross = ((float(size[0]), float(size[1])) if kind == BOX     # cross-section half-extents
                 else (float(size[0]), float(size[0])))
        direction = m.geom_pos[g] / max(half, 1e-9)
        if abs(np.linalg.norm(direction) - 1.0) > 1e-6:
            raise RuntimeError(
                f"{name}: its geom must run from the body origin (got pos "
                f"{m.geom_pos[g]}, half-length {half})")
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

        rows.append((b, g, direction, cross, half,
                     float(m.body_mass[b]) / unit_mass(half, cross, kind),
                     kind, direction * 2.0 * half,
                     min(0.5 * float(min_link_length), half),
                     len(axis_act) - n_ax, n_ax,
                     len(kid_id), len(kids), len(sid), len(sites)))
        kid_id += [int(k) for k in kids]
        kid_pos += [m.body_pos[k] for k in kids]
        sid += [int(s) for s in sites]
        spos += [m.site_pos[s] for s in sites]
    if not rows:
        raise RuntimeError("no '*_offsettable*' bodies in the model")

    cols = list(zip(*rows))
    names = ("body", "geom", "dir", "cross", "half0", "density", "kind",
             "tip0", "min_half", "axis_adr", "axis_num",
             "child_adr", "child_num", "site_adr", "site_num")
    seg = {n: np.asarray(c) for n, c in zip(names, cols)}
    seg["cross"] = np.asarray(seg["cross"], dtype=np.float64).reshape(-1, 2)
    seg["axis_dir"] = np.asarray(axis_dir, dtype=np.float64).reshape(-1, 3)
    seg["axis_act"] = np.asarray(axis_act, dtype=np.int64)
    seg["axis_lo"] = np.asarray(axis_lo, dtype=np.float64)
    seg["axis_hi"] = np.asarray(axis_hi, dtype=np.float64)
    seg["child_id"] = np.asarray(kid_id, dtype=np.int64)
    seg["child_pos"] = np.asarray(kid_pos, dtype=np.float64).reshape(-1, 3)
    seg["site_id"] = np.asarray(sid, dtype=np.int64)
    seg["site_pos"] = np.asarray(spos, dtype=np.float64).reshape(-1, 3)
    return seg


def n_slots(seg: dict[str, np.ndarray]) -> int:
    """Embodiment entries the links take up, groups counted once."""
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


def length_safe_lo(seg: dict[str, np.ndarray], axis_lo: np.ndarray, max_offset: float, min_length: float) -> np.ndarray:
    """Per-entry lower clip that keeps every link at least ``min_length`` long.

    A link is at least as long as its far end's projection on its own direction,
    ``L + max_offset * sum_k o_k (a_k . dir)``. Giving every entry with a
    component along the link the floor ``-(L - min_length) / (max_offset *
    sum_k a_k . dir)`` keeps that projection at ``min_length`` or more. An
    ``_equ`` slot takes the tightest floor of its members. The result is
    intersected with ``axis_lo``.
    """
    if max_offset <= 0.0:                  # no entry can change a link
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
    dir: wp.array(dtype=wp.vec3)
    cross: wp.array(dtype=wp.vec2)        # cross-section half-extents (r, r) or (x, y)
    half0: wp.array(dtype=wp.float32)
    density: wp.array(dtype=wp.float32)
    kind: wp.array(dtype=wp.int32)        # CAPSULE, CYLINDER or BOX
    tip0: wp.array(dtype=wp.vec3)
    min_half: wp.array(dtype=wp.float32)
    axis_adr: wp.array(dtype=wp.int32)
    axis_num: wp.array(dtype=wp.int32)
    axis_dir: wp.array(dtype=wp.vec3)
    axis_act: wp.array(dtype=wp.int32)
    axis_lo: wp.array(dtype=wp.float32)   # clip range, from the pos/neg suffix
    axis_hi: wp.array(dtype=wp.float32)
    child_adr: wp.array(dtype=wp.int32)
    child_num: wp.array(dtype=wp.int32)
    child_id: wp.array(dtype=wp.int32)
    child_pos: wp.array(dtype=wp.vec3)
    site_adr: wp.array(dtype=wp.int32)
    site_num: wp.array(dtype=wp.int32)
    site_id: wp.array(dtype=wp.int32)
    site_pos: wp.array(dtype=wp.vec3)


@wp.struct
class DeformArrays:
    """Views of the model fields the link deform writes."""
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
        as_np = np.float32 if dtype in (wp.float32, wp.vec2, wp.vec3) else np.int32
        setattr(out, name, wp.array(val.astype(as_np), dtype=dtype, device=device))
    return out


@wp.func
def mass_inertia(half: float, cross: wp.vec2, rho: float, kind: int) -> wp.vec4:
    """(mass, Ixx, Iyy, Izz) of a link geom about its CoM, long axis = local z,
    as MuJoCo computes it from geometry."""
    length = 2.0 * half
    r = cross[0]
    if kind == 2:
        cx = cross[0]
        cy = cross[1]
        m = rho * 8.0 * cx * cy * half
        third = m / 3.0
        return wp.vec4(m,
                       third * (cy * cy + half * half),
                       third * (cx * cx + half * half),
                       third * (cx * cx + cy * cy))
    m_cyl = rho * _PI * r * r * length
    transverse = m_cyl * (3.0 * r * r + length * length) / 12.0
    if kind == 1:
        return wp.vec4(m_cyl, transverse, transverse, 0.5 * m_cyl * r * r)
    m_cap = rho * (2.0 / 3.0) * _PI * r * r * r
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

    The link is rebuilt from its undeformed values as the geom from the body
    origin to the new end, so it stretches and turns. Children and sites at the
    far end move with it.
    """
    g = seg.geom[i]
    cross = seg.cross[i]
    r = cross[0]
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
    if seg.kind[i] == 2:
        dfm.geom_size[w, g] = wp.vec3(cross[0], cross[1], half)
    else:
        dfm.geom_size[w, g] = wp.vec3(r, half, 0.0)
    # Broadphase bounds must follow the size, or a stretched link gets culled
    if seg.kind[i] == 2:
        dfm.geom_rbound[w, g] = wp.sqrt(cross[0] * cross[0]
                                        + cross[1] * cross[1] + half * half)
        dfm.geom_aabb[w, g, 1] = wp.vec3(cross[0], cross[1], half)
    elif seg.kind[i] == 1:
        dfm.geom_rbound[w, g] = wp.sqrt(r * r + half * half)
        dfm.geom_aabb[w, g, 1] = wp.vec3(r, r, half)
    else:
        dfm.geom_rbound[w, g] = half + r
        dfm.geom_aabb[w, g, 1] = wp.vec3(r, r, half + r)
    dfm.geom_aabb[w, g, 0] = wp.vec3(0.0, 0.0, 0.0)

    b = seg.body[i]
    mi = mass_inertia(half, cross, seg.density[i], seg.kind[i])
    dfm.body_mass[w, b] = mi[0]
    dfm.body_inertia[w, b] = wp.vec3(mi[1], mi[2], mi[3])
    dfm.body_ipos[w, b] = pos
    dfm.body_iquat[w, b] = quat         # the inertia is diagonal in the geom frame

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
