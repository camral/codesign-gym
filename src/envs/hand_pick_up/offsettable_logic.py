"""Per-world link and box deformation driven by body names.

A body named ``*_offsettable[_<axes>][_pos|_neg][_equ<n>]`` is deformed per
world by its entries of the embodiment action:

- no axes: one entry, the link stretches along its own direction;
- one to three axes from ``xyz``: one entry per letter, each moving the
link's far end along that body-frame axis (taken pointing away from the
joint), so the link turns as well as stretches;
- ``_pos`` / ``_neg``: every entry is clipped to [0, 1] / [-1, 0];
- ``_equ<n>``: all bodies with the tag share one block of entries.

A link is a capsule or cylinder from the body origin, and children and sites at
its far end move with it. A body whose first geom is a box is resized instead:
each named axis grows the box evenly about its centre, the heightfield pads on
the body grow with it across their face (per-world ``hfield_size``), and the
body's mass and inertia are rebuilt from the box, its pads and its other geoms.
Links and boxes are never shrunk below their ``min_half`` / ``min_size``.

Code layout:

- find_segments, n_slots, slot_bounds, floor_reports, release_geom_frames:
host side, building the link table from a compiled model;
- SegArrays, DeformArrays, build_seg_arrays, apply_offsets, deform,
deform_box: device side, rebuilding every link and box from its undeformed
values.
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

# Warp dtypes of the find_segments columns uploaded by build_seg_arrays
SEG_DTYPE = {"geom": wp.int32, "body": wp.int32, "is_cyl": wp.int32,
             "is_box": wp.int32,
             "axis_adr": wp.int32, "axis_num": wp.int32, "axis_act": wp.int32,
             "child_adr": wp.int32, "child_num": wp.int32, "child_id": wp.int32,
             "site_adr": wp.int32, "site_num": wp.int32, "site_id": wp.int32,
             "pad_adr": wp.int32, "pad_num": wp.int32, "pad_geom": wp.int32,
             "pad_hfield": wp.int32, "pad_map": wp.vec3i,
             "dir": wp.vec3, "tip0": wp.vec3, "axis_dir": wp.vec3,
             "child_pos": wp.vec3, "site_pos": wp.vec3, "size0": wp.vec3,
             "min_size": wp.vec3, "box_pos": wp.vec3, "box_quat": wp.quat,
             "pad_size0": wp.vec4, "pad_pos": wp.vec3, "pad_quat": wp.quat,
             "fix_com": wp.vec3, "fix_Id": wp.vec3, "fix_Io": wp.vec3}
_FLOATS = (wp.float32, wp.vec3, wp.vec4, wp.quat)
_LINK_TYPES = (int(mujoco.mjtGeom.mjGEOM_CAPSULE),
               int(mujoco.mjtGeom.mjGEOM_CYLINDER))


def unit_mass(half: float, radius: float, is_cyl: int) -> float:
    """Mass of a link's geom at unit density. Host twin of ``mass_inertia``."""
    m_cyl = np.pi * radius ** 2 * 2.0 * half
    return m_cyl if is_cyl else m_cyl + (4.0 / 3.0) * np.pi * radius ** 3


def _geom_axes(m: mujoco.MjModel, g: int) -> np.ndarray:
    # Geom g's axes as the columns of a body-frame rotation
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, np.asarray(m.geom_quat[g], dtype=np.float64))
    return mat.reshape(3, 3)


def _along(v: np.ndarray) -> int | None:
    # The coordinate axis the unit vector v lies along, or None
    k = int(np.argmax(np.abs(v)))
    return k if abs(abs(float(v[k])) - 1.0) < 1e-6 else None


def _geom_label(m: mujoco.MjModel, g: int) -> str:
    body = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g]))
    name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom {g}"
    return f"{name} on {body}"


def _box_pads(m: mujoco.MjModel, b: int, g: int,
              box_axes: list[int], name: str) -> list[tuple]:
    """The heightfield pads on box body ``b``, as (geom, hfield, map, size).

    ``map`` gives per box axis the pad axis (0 or 1) that grows with it, or -1.
    A pad that resizing cannot keep on its face (turned off the box's axes,
    facing along a growing axis, or off-centre across one) raises.
    """
    rot = _geom_axes(m, g)
    out = []
    for p in range(int(m.body_geomadr[b]),
                   int(m.body_geomadr[b] + m.body_geomnum[b])):
        if m.geom_type[p] != mujoco.mjtGeom.mjGEOM_HFIELD:
            continue
        pad = _geom_label(m, p)
        pmap = [-1, -1, -1]
        for k in box_axes:
            u = rot[:, k]
            mp = _along(_geom_axes(m, p).T @ u)
            if mp is None:
                raise RuntimeError(
                    f"{name}: pad {pad} is turned off its box's axes, so the "
                    "box growing along one grows the pad along none")
            if mp == 2:
                raise RuntimeError(
                    f"{name}: pad {pad} faces along an axis its box grows "
                    "along; a pad only follows its box across its face")
            if abs(float((m.geom_pos[p] - m.geom_pos[g]) @ u)) > 1e-6:
                raise RuntimeError(
                    f"{name}: pad {pad} is off its box's centre across an axis "
                    "the box grows along. A pad grows evenly about its own "
                    "centre, which only keeps it on the same part of the face "
                    "when that is the box's centre too.")
            pmap[k] = mp
        h = int(m.geom_dataid[p])
        out.append((p, h, pmap, np.asarray(m.hfield_size[h], dtype=np.float64)))
    return out


def _check_shared_pads(m: mujoco.MjModel, pad_sig: dict[int, tuple]) -> None:
    """Raise if geoms drawing one heightfield would size it differently.
    ``hfield_size`` is one row per heightfield per world, so they must agree."""
    users: dict[int, list[int]] = {}
    for g in range(m.ngeom):
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_HFIELD:
            users.setdefault(int(m.geom_dataid[g]), []).append(g)
    for h, geoms in users.items():
        if len({pad_sig.get(g, ()) for g in geoms}) > 1:
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_HFIELD, h)
            raise RuntimeError(
                f"heightfield {name!r} is drawn by geoms that would size it "
                f"differently: {', '.join(_geom_label(m, g) for g in geoms)}. "
                "Its size is one row per world, so they must all sit on boxes "
                "of one size in one _equ group, or none on an offsettable box.")


def find_segments(m: mujoco.MjModel,
                  min_link_length: float = 0.0,
                  geom_density: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Table of every offsettable link and box, in body-id order.

    A link's first geom must run from the body origin, so its far end is
    ``2 * geom_pos``. Child bodies and sites of the link move with the far end,
    so they must sit there. Action slots are numbered by first appearance, and
    an ``_equ`` group shares one block. ``min_link_length`` [m] is stored per
    link as a half-length (for a box, per axis), capped at the undeformed size.
    ``geom_density`` (per model geom, from the spec) masses a box body as the
    composite of its geoms. Without it, the box carries the body's whole mass.
    """
    rows, axis_dir, axis_act, axis_lo, axis_hi = [], [], [], [], []
    kid_id, kid_pos, sid, spos = [], [], [], []
    pads, pad_sig = [], {}
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
        is_box = int(m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX)
        if not is_box and int(m.geom_type[g]) not in _LINK_TYPES:
            raise RuntimeError(
                f"{name}: its first geom is a "
                f"{mujoco.mjtGeom(m.geom_type[g]).name}; an offsettable body's "
                "first geom is a capsule or cylinder to stretch, or a box to "
                "resize")
        if is_box and not axes:
            raise RuntimeError(
                f"{name}: a box is resized along the axes its name gives, and "
                "this one gives none; add _x, _y or _z")
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

        if is_box:
            rot = _geom_axes(m, g)
            box_axes = [_along(rot[_AXIS[letter]]) for letter in axes]
            if None in box_axes:
                raise RuntimeError(
                    f"{name}: its box is turned off the body's axes, so an "
                    "axis letter names no edge of it to grow")
            # An entry changes the half-size along its box axis by half the
            # offset, since the box grows on both sides of its centre
            dirs = [0.5 * np.eye(3)[k] for k in box_axes]
            size0 = np.asarray(m.geom_size[g], dtype=np.float64)
            min_size = np.minimum(0.5 * float(min_link_length), size0)
            body_pads = _box_pads(m, b, g, box_axes, name)
            for p, _, pmap, _ in body_pads:
                pad_sig[p] = tuple(sorted(
                    (pmap[k], acts[i], lo, hi, float(size0[k]),
                     float(min_size[k])) for i, k in enumerate(box_axes)))
            kids = sites = ()
            radius = half = min_half = 0.0
            direction = tip0 = np.zeros(3)
            is_cyl = 0
            box_pos = np.asarray(m.geom_pos[g], dtype=np.float64)
            box_quat = np.asarray(m.geom_quat[g], dtype=np.float64)
            if geom_density is None:
                density = float(m.body_mass[b]) / (8.0 * float(np.prod(size0)))
                body_pads = [(*p, 0.0) for p in body_pads]
                fix = (0.0, np.zeros(3), np.zeros(3), np.zeros(3))
            else:
                own = range(int(m.body_geomadr[b]), int(m.body_geomadr[b] + m.body_geomnum[b]))
                if any(np.isnan(geom_density[k]) for k in own):
                    raise RuntimeError(f"{name}: a geom on it sets an explicit mass, so it has no density to mass the link by")
                density = float(geom_density[g])
                body_pads = [(*p, float(geom_density[p[0]])) for p in body_pads]
                fix = _fixed_parts(m, b, {g, *(p[0] for p in body_pads)}, geom_density, name)
                _check_composite(m, b, g, density, body_pads, fix, name)
        else:
            radius, half = float(m.geom_size[g][0]), float(m.geom_size[g][1])
            direction = m.geom_pos[g] / max(half, 1e-9)
            if abs(np.linalg.norm(direction) - 1.0) > 1e-6:
                raise RuntimeError(
                    f"{name}: its geom must run from the body origin (got pos "
                    f"{m.geom_pos[g]}, half-length {half})")
            is_cyl = int(m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CYLINDER)
            kids = np.flatnonzero(m.body_parentid == b)
            sites = np.flatnonzero(m.site_bodyid == b)
            # Named axes are flipped to point away from the joint, so a positive
            # offset never shortens the link
            dirs = [direction] if not axes else [
                -e if float(direction @ e) < 0.0 else e
                for e in (np.eye(3)[_AXIS[letter]] for letter in axes)]
            density = float(m.body_mass[b]) / unit_mass(half, radius, is_cyl)
            tip0 = direction * 2.0 * half
            min_half = min(0.5 * float(min_link_length), half)
            size0 = min_size = box_pos = np.zeros(3)
            box_quat = np.array([1.0, 0.0, 0.0, 0.0])
            body_pads = []
            fix = (0.0, np.zeros(3), np.zeros(3), np.zeros(3))
        for k, d in enumerate(dirs):
            axis_dir.append(d)
            axis_act.append(acts[k])
            axis_lo.append(lo)
            axis_hi.append(hi)

        rows.append((b, g, direction, radius, half, density, is_cyl, tip0,
                     min_half, len(axis_act) - n_ax, n_ax,
                     len(kid_id), len(kids), len(sid), len(sites),
                     is_box, size0, min_size, box_pos, box_quat,
                     len(pads), len(body_pads), *fix))
        kid_id += [int(k) for k in kids]
        kid_pos += [m.body_pos[k] for k in kids]
        sid += [int(s) for s in sites]
        spos += [m.site_pos[s] for s in sites]
        pads += body_pads
    if not rows:
        raise RuntimeError("no '*_offsettable*' bodies in the model")
    _check_shared_pads(m, pad_sig)

    cols = list(zip(*rows))
    names = ("body", "geom", "dir", "radius", "half0", "density", "is_cyl",
             "tip0", "min_half", "axis_adr", "axis_num",
             "child_adr", "child_num", "site_adr", "site_num",
             "is_box", "size0", "min_size", "box_pos", "box_quat",
             "pad_adr", "pad_num", "fix_mass", "fix_com", "fix_Id", "fix_Io")
    seg = {n: np.asarray(c) for n, c in zip(names, cols)}
    seg["axis_dir"] = np.asarray(axis_dir, dtype=np.float64).reshape(-1, 3)
    seg["axis_act"] = np.asarray(axis_act, dtype=np.int64)
    seg["axis_lo"] = np.asarray(axis_lo, dtype=np.float64)
    seg["axis_hi"] = np.asarray(axis_hi, dtype=np.float64)
    seg["child_id"] = np.asarray(kid_id, dtype=np.int64)
    seg["child_pos"] = np.asarray(kid_pos, dtype=np.float64).reshape(-1, 3)
    seg["site_id"] = np.asarray(sid, dtype=np.int64)
    seg["site_pos"] = np.asarray(spos, dtype=np.float64).reshape(-1, 3)
    seg["pad_geom"] = np.asarray([p[0] for p in pads], dtype=np.int64)
    seg["pad_hfield"] = np.asarray([p[1] for p in pads], dtype=np.int64)
    seg["pad_map"] = np.asarray([p[2] for p in pads],
                                dtype=np.int64).reshape(-1, 3)
    seg["pad_size0"] = np.asarray([p[3] for p in pads],
                                  dtype=np.float64).reshape(-1, 4)
    seg["pad_rho"] = np.asarray([p[4] for p in pads], dtype=np.float64)
    seg["pad_pos"] = np.asarray([m.geom_pos[p[0]] for p in pads], dtype=np.float64).reshape(-1, 3)
    seg["pad_quat"] = np.asarray([m.geom_quat[p[0]] for p in pads], dtype=np.float64).reshape(-1, 4)
    return seg


def _quat_mat(q: np.ndarray) -> np.ndarray:
    r = np.zeros(9)
    mujoco.mju_quat2Mat(r, np.asarray(q, dtype=np.float64))
    return r.reshape(3, 3)


def _box_part(rho: float, half, pos, quat):
    """(mass, com, inertia about the com) of a solid box, in the body frame."""
    a, b, c = (float(v) for v in half)
    mass = rho * 8.0 * a * b * c
    R = _quat_mat(quat)
    return mass, np.asarray(pos, float), R @ np.diag([mass * (b * b + c * c), mass * (a * a + c * c), mass * (a * a + b * b)]) / 3.0 @ R.T


def _pad_part(rho: float, size, pos, quat):
    # MuJoCo masses a heightfield as a box of its footprint and top/2 + bottom height
    return _box_part(rho, (size[0], size[1], 0.5 * (0.5 * size[2] + size[3])), pos, quat)


def _primitive_part(m: mujoco.MjModel, g: int, rho: float, name: str):
    """(mass, com, inertia) of one fixed geom, by MuJoCo's primitive formulas."""
    t, sz, R = int(m.geom_type[g]), m.geom_size[g], _quat_mat(m.geom_quat[g])
    if t == mujoco.mjtGeom.mjGEOM_SPHERE:
        mass = rho * 4.0 / 3.0 * np.pi * sz[0] ** 3
        diag = np.full(3, 0.4 * mass * sz[0] ** 2)
    elif t in (mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_CAPSULE):
        r, h = sz[0], sz[1]
        m_cyl = rho * np.pi * r * r * 2.0 * h
        trans, axial, mass = m_cyl * (3.0 * r * r + 4.0 * h * h) / 12.0, 0.5 * m_cyl * r * r, m_cyl
        if t == mujoco.mjtGeom.mjGEOM_CAPSULE:
            m_cap = rho * 2.0 / 3.0 * np.pi * r ** 3
            trans += 2.0 * m_cap * (0.4 * r * r + h * h + 0.75 * r * h)
            axial += 0.8 * m_cap * r * r
            mass += 2.0 * m_cap
        diag = np.array([trans, trans, axial])
    elif t == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        mass = rho * 4.0 / 3.0 * np.pi * np.prod(sz)
        diag = mass / 5.0 * np.array([sz[1] ** 2 + sz[2] ** 2, sz[0] ** 2 + sz[2] ** 2, sz[0] ** 2 + sz[1] ** 2])
    elif t == mujoco.mjtGeom.mjGEOM_BOX:
        return _box_part(rho, sz, m.geom_pos[g], m.geom_quat[g])
    elif t == mujoco.mjtGeom.mjGEOM_HFIELD:
        return _pad_part(rho, m.hfield_size[int(m.geom_dataid[g])], m.geom_pos[g], m.geom_quat[g])
    else:
        raise RuntimeError(f"{name}: cannot mass its {mujoco.mjtGeom(t).name} geom {_geom_label(m, g)} as a fixed part")
    return mass, np.asarray(m.geom_pos[g], float), R @ np.diag(diag) @ R.T


def _combine(parts):
    """(mass, com, inertia about the com) of a set of (mass, com, inertia) parts."""
    total = sum(p[0] for p in parts)
    if total <= 0.0:
        return 0.0, np.zeros(3), np.zeros((3, 3))
    com = sum(p[0] * p[1] for p in parts) / total
    inertia = np.zeros((3, 3))
    for mass, c, I in parts:
        d = c - com
        inertia += I + mass * (d @ d * np.eye(3) - np.outer(d, d))
    return total, com, inertia


def _fixed_parts(m: mujoco.MjModel, b: int, moving: set, geom_density: np.ndarray, name: str):
    """The non-deforming geoms of a box body, lumped: (mass, com, diag, off-diag)."""
    parts = [_primitive_part(m, g, float(geom_density[g]), name)
             for g in range(int(m.body_geomadr[b]), int(m.body_geomadr[b] + m.body_geomnum[b]))
             if g not in moving and geom_density[g] > 0.0]
    mass, com, I = _combine(parts)
    return mass, com, np.diag(I).copy(), np.array([I[0, 1], I[0, 2], I[1, 2]])


def _check_composite(m, b, g, rho, pads, fix, name) -> None:
    """Raise if the parts don't rebuild the compiled body, which the deform starts from."""
    parts = [_box_part(rho, m.geom_size[g], m.geom_pos[g], m.geom_quat[g])]
    parts += [_pad_part(prho, size, m.geom_pos[p], m.geom_quat[p]) for p, _, _, size, prho in pads]
    fm, fc, fd, fo = fix
    if fm > 0.0:
        parts.append((fm, fc, np.diag(fd) + np.array([[0, fo[0], fo[1]], [fo[0], 0, fo[2]], [fo[1], fo[2], 0]])))
    mass, com, inertia = _combine(parts)
    R = _quat_mat(m.body_iquat[b])
    compiled = R @ np.diag(m.body_inertia[b]) @ R.T
    if (abs(mass / m.body_mass[b] - 1.0) > 1e-4 or np.abs(com - m.body_ipos[b]).max() > 1e-6
            or np.abs(inertia - compiled).max() > 1e-4 * np.abs(compiled).max()):
        raise RuntimeError(f"{name}: its geoms at the given densities do not rebuild the compiled body "
                           f"(mass {mass:.6g} vs {m.body_mass[b]:.6g}, com {com} vs {m.body_ipos[b]})")


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


def release_geom_frames(m: mujoco.MjModel, seg: dict[str, np.ndarray]) -> None:
    """Make CPU kinematics pose every geom on an offsettable body from its own
    ``geom_pos``/``geom_quat``.

    The compiler marks geoms sharing their body's or inertial frame in
    ``geom_sameframe``, and ``mj_kinematics`` then poses them from that frame. The
    deform turns links and moves inertial frames, so those geoms would be drawn
    wrong. mujoco_warp ignores ``geom_sameframe``, so only the CPU copies used
    for rendering need this.
    """
    mine = np.isin(m.geom_bodyid, seg["body"])
    m.geom_sameframe[mine] = mujoco.mjtSameFrame.mjSAMEFRAME_NONE


def floor_reports(m: mujoco.MjModel, seg: dict[str, np.ndarray],
                  max_offset: float) -> list[str]:
    """One line per link or box axis whose offset range reaches its floor, where
    part of the embodiment space builds the same shape."""
    out = []
    for i, b in enumerate(seg["body"]):
        adr, num = int(seg["axis_adr"][i]), int(seg["axis_num"][i])
        if int(seg["is_box"][i]):
            out += _box_floor_reports(m, seg, i, max_offset)
            continue
        tip = np.array(seg["tip0"][i], dtype=np.float64)
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


def _box_floor_reports(m: mujoco.MjModel, seg: dict[str, np.ndarray], i: int,
                       max_offset: float) -> list[str]:
    out = []
    adr, num = int(seg["axis_adr"][i]), int(seg["axis_num"][i])
    for j in range(adr, adr + num):
        k = int(np.argmax(np.abs(seg["axis_dir"][j])))
        half = float(seg["size0"][i][k])
        narrowest = half + 0.5 * float(seg["axis_lo"][j]) * max_offset
        floor = float(seg["min_size"][i][k])
        if narrowest < floor - 1e-9:
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY,
                                     int(seg["body"][i]))
            out.append(f"{name}: box {2 * half:.3f} m across its axis {k} "
                       f"undeformed, {max(2 * narrowest, 0.0):.3f} m at its "
                       f"narrowest, floored at {2 * floor:.3f} m")
    return out


@wp.struct
class SegArrays:
    """The ``find_segments`` table on the device. Ragged blocks (axes, children,
    sites, pads) are indexed by their ``*_adr``/``*_num`` fields."""
    geom: wp.array(dtype=wp.int32)
    body: wp.array(dtype=wp.int32)
    dir: wp.array(dtype=wp.vec3)
    radius: wp.array(dtype=wp.float32)
    half0: wp.array(dtype=wp.float32)
    density: wp.array(dtype=wp.float32)
    is_cyl: wp.array(dtype=wp.int32)
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
    is_box: wp.array(dtype=wp.int32)      # a box to resize, not a link
    size0: wp.array(dtype=wp.vec3)
    min_size: wp.array(dtype=wp.vec3)
    box_pos: wp.array(dtype=wp.vec3)      # box pose in the body frame
    box_quat: wp.array(dtype=wp.quat)
    pad_adr: wp.array(dtype=wp.int32)
    pad_num: wp.array(dtype=wp.int32)
    pad_geom: wp.array(dtype=wp.int32)
    pad_hfield: wp.array(dtype=wp.int32)
    pad_map: wp.array(dtype=wp.vec3i)     # per box axis, the pad axis that grows with it
    pad_size0: wp.array(dtype=wp.vec4)
    pad_rho: wp.array(dtype=wp.float32)   # 0: massed with the box instead
    pad_pos: wp.array(dtype=wp.vec3)
    pad_quat: wp.array(dtype=wp.quat)
    fix_mass: wp.array(dtype=wp.float32)  # the body's other geoms, lumped: mass,
    fix_com: wp.array(dtype=wp.vec3)      # com, inertia (xx, yy, zz) and (xy, xz, yz)
    fix_Id: wp.array(dtype=wp.vec3)
    fix_Io: wp.array(dtype=wp.vec3)


@wp.struct
class DeformArrays:
    """Views of the model fields the deform writes."""
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
    hfield_size: wp.array2d(dtype=wp.vec4)


def build_seg_arrays(seg: dict[str, np.ndarray], device) -> SegArrays:
    out = SegArrays()
    for name, val in seg.items():
        dtype = SEG_DTYPE.get(name, wp.float32)
        as_np = np.float32 if dtype in _FLOATS else np.int32
        setattr(out, name, wp.array(val.astype(as_np), dtype=dtype, device=device))
    return out


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
def wxyz_to_matrix(q: wp.quat) -> wp.mat33:
    """Rotation matrix of a w, x, y, z quaternion, reordered for Warp's helper."""
    return wp.quat_to_matrix(wp.quat(q[1], q[2], q[3], q[0]))


@wp.func
def matrix_to_wxyz(R: wp.mat33) -> wp.quat:
    """w, x, y, z quaternion of a rotation matrix, reordered from Warp's helper."""
    q = wp.quat_from_matrix(R)
    return wp.quat(q[3], q[0], q[1], q[2])


@wp.func
def box_inertia(mass: float, half: wp.vec3, q: wp.quat) -> wp.mat33:
    """Inertia of a solid box about its centre, turned into the body frame."""
    R = wxyz_to_matrix(q)
    xx = half[0] * half[0]
    yy = half[1] * half[1]
    zz = half[2] * half[2]
    D = wp.mat33(mass * (yy + zz) / 3.0, 0.0, 0.0, 0.0, mass * (xx + zz) / 3.0, 0.0, 0.0, 0.0, mass * (xx + yy) / 3.0)
    return R * D * wp.transpose(R)


@wp.func
def shifted_inertia(I: wp.mat33, mass: float, c: wp.vec3) -> wp.mat33:
    """Parallel-axis shift of an inertia about a part's CoM ``c`` to the body origin."""
    return I + mass * (wp.dot(c, c) * wp.identity(n=3, dtype=float) - wp.outer(c, c))


@wp.func
def deform_box(dfm: DeformArrays, seg: SegArrays, w: int, i: int,
               delta: wp.vec3):
    """Grow world ``w``'s box ``i`` by ``delta`` in half-sizes along its own axes.

    Rebuilt from the undeformed box and floored per axis at ``min_size``. The
    box keeps its centre and axes. Its pads take the half-size change in their
    ``hfield_size`` and broadphase bounds. The body's mass, CoM and principal
    inertia are rebuilt from the box, the pads and the fixed geoms.
    """
    g = seg.geom[i]
    h0 = seg.size0[i]
    lo = seg.min_size[i]
    half = wp.vec3(wp.max(h0[0] + delta[0], lo[0]),
                   wp.max(h0[1] + delta[1], lo[1]),
                   wp.max(h0[2] + delta[2], lo[2]))
    dfm.geom_size[w, g] = half
    dfm.geom_rbound[w, g] = wp.length(half)
    dfm.geom_aabb[w, g, 0] = wp.vec3(0.0, 0.0, 0.0)
    dfm.geom_aabb[w, g, 1] = half

    b = seg.body[i]
    # Accumulate mass, first moment and inertia about the body origin, then
    # move to the composite CoM
    mass = seg.density[i] * 8.0 * half[0] * half[1] * half[2]
    moment = seg.box_pos[i] * mass
    inertia = shifted_inertia(box_inertia(mass, half, seg.box_quat[i]), mass, seg.box_pos[i])

    grow = half - h0
    for k in range(seg.pad_num[i]):
        p = seg.pad_adr[i] + k
        size = seg.pad_size0[p]
        axis = seg.pad_map[p]
        sx = size[0]
        sy = size[1]
        for a in range(3):
            if axis[a] == 0:
                sx = wp.max(size[0] + grow[a], lo[a])
            elif axis[a] == 1:
                sy = wp.max(size[1] + grow[a], lo[a])
        dfm.hfield_size[w, seg.pad_hfield[p]] = wp.vec4(sx, sy, size[2], size[3])
        pg = seg.pad_geom[p]
        aabb = dfm.geom_aabb[w, pg, 1]
        dfm.geom_aabb[w, pg, 1] = wp.vec3(sx, sy, aabb[2])
        top = wp.max(size[2], size[3])
        dfm.geom_rbound[w, pg] = wp.sqrt(sx * sx + sy * sy + top * top)
        hz = 0.5 * (0.5 * size[2] + size[3])     # MuJoCo masses a pad as a box this half-height
        pm = seg.pad_rho[p] * 8.0 * sx * sy * hz
        pc = seg.pad_pos[p]
        mass += pm
        moment += pc * pm
        inertia += shifted_inertia(box_inertia(pm, wp.vec3(sx, sy, hz), seg.pad_quat[p]), pm, pc)

    fm = seg.fix_mass[i]
    fc = seg.fix_com[i]
    fd = seg.fix_Id[i]
    fo = seg.fix_Io[i]
    fixed = wp.mat33(fd[0], fo[0], fo[1], fo[0], fd[1], fo[2], fo[1], fo[2], fd[2])
    mass += fm
    moment += fc * fm
    inertia += shifted_inertia(fixed, fm, fc)

    com = moment / mass
    inertia -= shifted_inertia(wp.mat33(0.0), mass, com)
    # Principal axes, made a proper rotation for matrix_to_wxyz
    Q = wp.mat33()
    d = wp.vec3()
    wp.eig3(inertia, Q, d)
    if wp.determinant(Q) < 0.0:
        Q = wp.mat33(-Q[0, 0], Q[0, 1], Q[0, 2], -Q[1, 0], Q[1, 1], Q[1, 2], -Q[2, 0], Q[2, 1], Q[2, 2])
    dfm.body_mass[w, b] = mass
    dfm.body_ipos[w, b] = com
    dfm.body_inertia[w, b] = d
    dfm.body_iquat[w, b] = matrix_to_wxyz(Q)


@wp.func
def apply_offsets(dfm: DeformArrays, seg: SegArrays,
                  action: wp.array2d(dtype=wp.float32),
                  offsets: wp.array2d(dtype=wp.float32),
                  w: int, n_seg: int, max_offset: float):
    """Deform every link and box of world ``w`` by its entries of the embodiment
    action.

    The action is normalised, and ``max_offset`` [m] scales it here and nowhere
    else. The clipped values are recorded in ``offsets``. A box's ``axis_dir``
    is half a unit vector, so the same sum is its change of half-size.
    """
    for i in range(n_seg):
        delta = wp.vec3(0.0, 0.0, 0.0)
        for k in range(seg.axis_num[i]):
            j = seg.axis_adr[i] + k
            a = seg.axis_act[j]
            o = wp.clamp(action[w, a], seg.axis_lo[j], seg.axis_hi[j])
            offsets[w, a] = o
            delta += seg.axis_dir[j] * (o * max_offset)
        if seg.is_box[i] == 1:
            deform_box(dfm, seg, w, i, delta)
        else:
            deform(dfm, seg, w, i, delta)
