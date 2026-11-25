"""Sphere-radius design axis (the J1 radius mechanism, scoped to Humanoid feet).

The capsule offsettable machinery deforms length/direction only. Humanoid feet
are sphere geoms with no length axis, so this adds a per-world radius scale for
named single-geom sphere bodies: r = r0 * (1 + offset * radius_frac), with mass
and inertia recomputed from the sphere at its captured density. Its design slots
sit AFTER the capsule offsettable slots in the shared design vector.

Only single-geom sphere bodies are supported (exact density from body_mass); the
hands are a second geom on the forearm body and are handled separately.
"""

from __future__ import annotations

import numpy as np
import mujoco
import warp as wp

from .offsettable_logic import DeformArrays

_PI = 3.14159265358979323846


@wp.struct
class RadArrays:
    geom: wp.array(dtype=wp.int32)
    body: wp.array(dtype=wp.int32)
    r0: wp.array(dtype=wp.float32)
    density: wp.array(dtype=wp.float32)
    ipos: wp.array(dtype=wp.vec3)     # sphere centre in the body frame (constant)
    slot: wp.array(dtype=wp.int32)    # design-vector slot for each geom


def find_radius_geoms(m: mujoco.MjModel, names, base_slot: int) -> dict:
    """Table for each named sphere geom, with design slots numbered from ``base_slot``.

    Each geom's body must carry only that geom, so its density is exact
    (body_mass / sphere_volume)."""
    geom, body, r0, dens, ipos, slot = [], [], [], [], [], []
    for k, name in enumerate(names):
        gid = m.geom(name).id
        if m.geom_type[gid] != mujoco.mjtGeom.mjGEOM_SPHERE:
            raise RuntimeError(f"{name}: radius axis needs a sphere geom")
        b = int(m.geom_bodyid[gid])
        if int(m.body_geomnum[b]) != 1:
            raise RuntimeError(f"{name}: radius axis needs a single-geom body (exact density)")
        r = float(m.geom_size[gid][0])
        geom.append(gid); body.append(b); r0.append(r)
        dens.append(float(m.body_mass[b]) / ((4.0 / 3.0) * np.pi * r ** 3))
        ipos.append(m.geom_pos[gid]); slot.append(base_slot + k)
    return {"geom": geom, "body": body, "r0": r0, "density": dens, "ipos": ipos, "slot": slot}


def build_rad_arrays(tab: dict, device) -> RadArrays:
    out = RadArrays()
    out.geom = wp.array(np.asarray(tab["geom"], dtype=np.int32), dtype=wp.int32, device=device)
    out.body = wp.array(np.asarray(tab["body"], dtype=np.int32), dtype=wp.int32, device=device)
    out.r0 = wp.array(np.asarray(tab["r0"], dtype=np.float32), dtype=wp.float32, device=device)
    out.density = wp.array(np.asarray(tab["density"], dtype=np.float32), dtype=wp.float32, device=device)
    out.ipos = wp.array(np.asarray(tab["ipos"], dtype=np.float32).reshape(-1, 3), dtype=wp.vec3, device=device)
    out.slot = wp.array(np.asarray(tab["slot"], dtype=np.int32), dtype=wp.int32, device=device)
    return out


@wp.kernel
def radius_deform(mask: wp.array(dtype=wp.bool),
                  action: wp.array2d(dtype=wp.float32),
                  offsets: wp.array2d(dtype=wp.float32),
                  dfm: DeformArrays, rad: RadArrays,
                  n_rad: int, radius_frac: float):
    """Scale each named sphere's radius by its design slot, per world."""
    w = wp.tid()
    if not mask[w]:
        return
    for i in range(n_rad):
        s = rad.slot[i]
        off = wp.clamp(action[w, s], -1.0, 1.0)
        offsets[w, s] = off
        r = rad.r0[i] * (1.0 + off * radius_frac)
        g = rad.geom[i]
        b = rad.body[i]
        dfm.geom_size[w, g] = wp.vec3(r, 0.0, 0.0)
        dfm.geom_rbound[w, g] = r
        dfm.geom_aabb[w, g, 0] = wp.vec3(0.0, 0.0, 0.0)
        dfm.geom_aabb[w, g, 1] = wp.vec3(r, r, r)
        mass = rad.density[i] * (4.0 / 3.0) * _PI * r * r * r
        inertia = 0.4 * mass * r * r
        dfm.body_mass[w, b] = mass
        dfm.body_inertia[w, b] = wp.vec3(inertia, inertia, inertia)
        dfm.body_ipos[w, b] = rad.ipos[i]
