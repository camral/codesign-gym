"""Forearm + hand design for the Humanoid (multi-geom composite, keeps 376 obs).

The forearm body carries two geoms: the forearm capsule (first geom, made
offsettable so its length/direction is designable via the standard capsule
kernel) and the hand sphere (second geom). This module runs AFTER the capsule
deform and, per world:

  * moves the hand sphere to the deformed capsule's far end (so it tracks the
    forearm instead of detaching),
  * scales the hand radius by its own design slot (the hands-radius axis),
  * recomputes the forearm body's true mass / COM / inertia from BOTH geoms.

Because the capsule and the hand are collinear along the forearm axis u, the
composite inertia is diagonal in the capsule's own frame (aim quaternion), so no
eigen-decomposition is needed: the axial moment adds, the transverse moments add
with a parallel-axis shift along u. Densities are captured equal (the humanoid
geoms share the default density), exact at the base design.
"""

from __future__ import annotations

import numpy as np
import mujoco
import warp as wp

from .offsettable_logic import DeformArrays, mass_inertia

_PI = 3.14159265358979323846


@wp.struct
class ArmArrays:
    cap_geom: wp.array(dtype=wp.int32)     # forearm capsule (body's first geom)
    hand_geom: wp.array(dtype=wp.int32)    # hand sphere (second geom)
    body: wp.array(dtype=wp.int32)
    dir: wp.array(dtype=wp.vec3)           # undeformed unit forearm axis (fallback)
    r_cap: wp.array(dtype=wp.float32)
    r_hand0: wp.array(dtype=wp.float32)
    density: wp.array(dtype=wp.float32)
    gap: wp.array(dtype=wp.float32)        # wrist gap: hand centre beyond the capsule tip, along u
    hand_slot: wp.array(dtype=wp.int32)    # design slot scaling the hand radius


def find_arms(m: mujoco.MjModel, hand_geom_names, base_slot: int) -> dict:
    """Table per arm from each hand geom: its body, that body's capsule (first
    geom), radii, shared density, and the hand-radius design slot (from base_slot)."""
    cap, hand, body, direction, r_cap, r_hand0, dens, gap, slot = [], [], [], [], [], [], [], [], []
    for k, hname in enumerate(hand_geom_names):
        hg = m.geom(hname).id
        if m.geom_type[hg] != mujoco.mjtGeom.mjGEOM_SPHERE:
            raise RuntimeError(f"{hname}: hand radius needs a sphere geom")
        b = int(m.geom_bodyid[hg])
        cg = int(m.body_geomadr[b])        # forearm capsule = body's first geom
        if int(m.body_geomnum[b]) != 2 or m.geom_type[cg] != mujoco.mjtGeom.mjGEOM_CAPSULE:
            raise RuntimeError(f"{hname}: expected a 2-geom (capsule + hand) forearm body")
        rc, half = float(m.geom_size[cg][0]), float(m.geom_size[cg][1])
        u = m.geom_pos[cg] / max(half, 1e-9)
        rh = float(m.geom_size[hg][0])
        cap_vol = _PI * rc ** 2 * 2.0 * half + (4.0 / 3.0) * _PI * rc ** 3
        hand_vol = (4.0 / 3.0) * _PI * rh ** 3
        cap.append(cg); hand.append(hg); body.append(b)
        direction.append(u); r_cap.append(rc); r_hand0.append(rh)
        dens.append(float(m.body_mass[b]) / (cap_vol + hand_vol))  # equal-density capture
        gap.append(float(np.dot(m.geom_pos[hg], u)) - 2.0 * half)   # keep stock hand position at offset 0
        slot.append(base_slot + k)
    return {"cap_geom": cap, "hand_geom": hand, "body": body, "dir": direction,
            "r_cap": r_cap, "r_hand0": r_hand0, "density": dens, "gap": gap, "hand_slot": slot}


def build_arm_arrays(tab: dict, device) -> ArmArrays:
    out = ArmArrays()
    out.cap_geom = wp.array(np.asarray(tab["cap_geom"], np.int32), dtype=wp.int32, device=device)
    out.hand_geom = wp.array(np.asarray(tab["hand_geom"], np.int32), dtype=wp.int32, device=device)
    out.body = wp.array(np.asarray(tab["body"], np.int32), dtype=wp.int32, device=device)
    out.dir = wp.array(np.asarray(tab["dir"], np.float32).reshape(-1, 3), dtype=wp.vec3, device=device)
    out.r_cap = wp.array(np.asarray(tab["r_cap"], np.float32), dtype=wp.float32, device=device)
    out.r_hand0 = wp.array(np.asarray(tab["r_hand0"], np.float32), dtype=wp.float32, device=device)
    out.density = wp.array(np.asarray(tab["density"], np.float32), dtype=wp.float32, device=device)
    out.gap = wp.array(np.asarray(tab["gap"], np.float32), dtype=wp.float32, device=device)
    out.hand_slot = wp.array(np.asarray(tab["hand_slot"], np.int32), dtype=wp.int32, device=device)
    return out


@wp.kernel
def arm_fix(mask: wp.array(dtype=wp.bool),
            action: wp.array2d(dtype=wp.float32),
            offsets: wp.array2d(dtype=wp.float32),
            dfm: DeformArrays, arm: ArmArrays, n_arm: int, radius_frac: float):
    """Track + scale the hand and set the forearm's true composite inertia."""
    w = wp.tid()
    if not mask[w]:
        return
    for i in range(n_arm):
        cg = arm.cap_geom[i]
        hg = arm.hand_geom[i]
        b = arm.body[i]
        gp = dfm.geom_pos[w, cg]              # deformed capsule centre = u*half
        half = wp.length(gp)
        u = wp.where(half > 1.0e-9, gp / wp.max(half, 1.0e-9), arm.dir[i])
        r_cap = arm.r_cap[i]

        s = arm.hand_slot[i]
        off = wp.clamp(action[w, s], -1.0, 1.0)
        offsets[w, s] = off
        r_hand = arm.r_hand0[i] * (1.0 + off * radius_frac)

        hand_d = 2.0 * half + arm.gap[i]         # hand centre distance along u (keeps wrist gap)
        dfm.geom_pos[w, hg] = u * hand_d
        dfm.geom_size[w, hg] = wp.vec3(r_hand, 0.0, 0.0)
        dfm.geom_rbound[w, hg] = r_hand
        dfm.geom_aabb[w, hg, 0] = wp.vec3(0.0, 0.0, 0.0)
        dfm.geom_aabb[w, hg, 1] = wp.vec3(r_hand, r_hand, r_hand)

        rho = arm.density[i]
        mi = mass_inertia(half, r_cap, rho, 0)   # capsule (mass, Ixx, Iyy, Izz), z = long axis
        m_cap = mi[0]
        i_cap_trans = mi[1]
        i_cap_axial = mi[3]
        m_hand = rho * (4.0 / 3.0) * _PI * r_hand * r_hand * r_hand
        i_hand = 0.4 * m_hand * r_hand * r_hand

        total = m_cap + m_hand
        c = (m_cap * half + m_hand * hand_d) / wp.max(total, 1.0e-9)   # COM along u
        d_cap = half - c
        d_hand = hand_d - c
        i_axial = i_cap_axial + i_hand
        i_trans = (i_cap_trans + m_cap * d_cap * d_cap) + (i_hand + m_hand * d_hand * d_hand)

        dfm.body_mass[w, b] = total
        dfm.body_ipos[w, b] = u * c
        dfm.body_iquat[w, b] = dfm.geom_quat[w, cg]   # capsule frame (aim quat of u)
        dfm.body_inertia[w, b] = wp.vec3(i_trans, i_trans, i_axial)
