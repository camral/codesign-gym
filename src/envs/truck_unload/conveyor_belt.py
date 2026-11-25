"""The belt that carries boxes from the loader past the stackers, and the wall
that divides the line.

- add_conveyor: a box geom whose surface velocity carries the boxes, with side
walls, end rollers and the loader's goal site;
- add_divider: a wall across the line with a hole for the belt;
- Conveyor: where the belt ended up.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

_RGBA_BELT = (0.16, 0.18, 0.20, 1.0)
_RGBA_WALLS = (0.52, 0.55, 0.59, 1.0)
_RGBA_DIVIDER = (0.65, 0.78, 0.90, 0.14)    # subtle tinted safety glass, keeps robots clear

BELT_FRICTION = (1.0, 0.005, 0.0001)

# Spelled out rather than inherited - see the note in container.py
_CONTACT = dict(condim=3, margin=0.0, contype=1, conaffinity=1,
                friction=list(BELT_FRICTION))

_CONTACT_CYL = dict(condim=1, margin=0.0, contype=1, conaffinity=1,
                priority = 100)

@dataclass
class Conveyor:
    geom: str
    lower: np.ndarray
    upper: np.ndarray
    outer_half_w: float      # half-width in y, walls included
    goal_site: str
    goal_pos: np.ndarray

    @property
    def top_z(self) -> float:
        return float(self.upper[2])


def add_conveyor(
    spec: mujoco.MjSpec,
    *,
    x_min: float,
    x_max: float,
    center_y: float,
    width: float,
    z_playing_field: float,
    wall_thickness: float,
    belt_velocity: float,
    box_size: float,
    name: str = "conveyor",
    goal_name: str = "loader_goal",
    rgba=_RGBA_BELT,
) -> Conveyor:
    """A belt from ``x_min`` to ``x_max``, top at ``z_playing_field``, carrying
    boxes towards +x at ``belt_velocity``. The ``goal_name`` site is where a box
    resting on the belt has its centre, a box and a half past ``x_min``.

    ``belt_velocity`` must be nonzero: the compiler enables surface velocity only
    if some geom uses it, and per-world writes to ``geom_surfacevel`` are
    otherwise ignored.
    """
    thickness = wall_thickness * 3.0

    half = np.array([(x_max - x_min) / 2.0, width / 2.0, thickness / 2.0])
    pos = np.array([(x_min + x_max) / 2.0, center_y, z_playing_field - thickness / 2.0])

    pos_wall_l = [pos[0], pos[1] + half[1], pos[2] / 2.0]
    pos_wall_r = [pos[0], pos[1] - half[1], pos[2] / 2.0]

    half_wall = [half[0], wall_thickness, pos[2] / 2.0]   # walls from the ground to the belt's centre


    half_cyl = [half[2], half[1]]                          # rollers at both ends of the belt
    pos_cyl_f = [x_min, pos[1], pos[2]]
    pos_cyl_b = [x_max, pos[1], pos[2]]

    spec.worldbody.add_geom(
        name=name, type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=pos.tolist(), size=half.tolist(), rgba=list(rgba),
        surfacevel=[belt_velocity, 0.0, 0.0, 0.0, 0.0, 0.0], **_CONTACT)

    spec.worldbody.add_geom(
        name=name+"_wall_l", type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=pos_wall_l, size=half_wall, rgba=list(_RGBA_WALLS),
        **_CONTACT)

    spec.worldbody.add_geom(
        name=name+"_wall_r", type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=pos_wall_r, size=half_wall, rgba=list(_RGBA_WALLS),
        **_CONTACT)

    spec.worldbody.add_geom(
        name=name+"_cyl_f", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        pos=pos_cyl_f, size=half_cyl, euler=[90, 0, 0], rgba=list(rgba),
        surfacevel=[belt_velocity, 0.0, 0.0, 0.0, 0.0, 0.0], **_CONTACT)

    spec.worldbody.add_geom(
        name=name+"_cyl_b", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        pos=pos_cyl_b, size=half_cyl, euler=[90, 0, 0], rgba=list(rgba),
        surfacevel=[belt_velocity, 0.0, 0.0, 0.0, 0.0, 0.0], **_CONTACT)
    
    goal = np.array([x_min + 1.5 * box_size, center_y,
                     z_playing_field + box_size / 2.0])
    spec.worldbody.add_site(name=goal_name, pos=goal.tolist(), group=4,
                            size=[0.01, 0.01, 0.01])
    return Conveyor(geom=name, lower=pos - half, upper=pos + half,
                    outer_half_w=float(half[1] + wall_thickness),
                    goal_site=goal_name, goal_pos=goal)


def add_divider(
    spec: mujoco.MjSpec,
    *,
    x: float,
    center_y: float,
    opening_half_w: float,
    gap_z: float,
    height: float,
    side_length: float,
    wall_thickness: float,
    name: str = "divider",
    rgba=_RGBA_DIVIDER,
) -> list[str]:
    """A wall across the line at ``x``, ``wall_thickness`` thick and ``height``
    high, with a hole ``opening_half_w`` to each side of ``center_y`` and up to
    ``gap_z``. Made of a panel on each side, one over the hole, and a side
    wall along +x from the right panel. Returns the geom names."""
    if height <= gap_z:
        raise ValueError(f"the divider is {height} high, which leaves no room "
                         f"over a hole that goes up to {gap_z}")
    t = wall_thickness
    side = side_length / 2.0
    parts = [
        ("left", [x, center_y - opening_half_w - side, height / 2.0],
         [t / 2.0, side, height / 2.0]),
        ("right", [x, center_y + opening_half_w + side, height / 2.0],
         [t / 2.0, side, height / 2.0]),
        ("top", [x, center_y, (gap_z + height) / 2.0],
         [t / 2.0, opening_half_w, (height - gap_z) / 2.0]),
        ("side", [x+side, center_y + opening_half_w + side*2, height / 2.0],
         [side, t / 2.0, height / 2.0]),
    ]
    for gname, gpos, gsize in parts:
        spec.worldbody.add_geom(name=f"{name}_{gname}",
                                type=mujoco.mjtGeom.mjGEOM_BOX, pos=gpos,
                                size=gsize, rgba=list(rgba), **_CONTACT)
    return [f"{name}_{gname}" for gname, _, _ in parts]
