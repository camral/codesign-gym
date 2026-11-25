"""The pad a stacker builds on.

A slab whose top face is the playing field, optionally with walls on three sides
(the -y side, facing the robot and the belt, stays open) or a shelf with one
board per level. Legs under the corners are decoration.

- add_platform: a pad, walled pad or shelf, with a goal site that rises per box;
- add_floor_area: a flat pad with a target area anywhere on which a box counts;
- Platform: where the pad ended up and how its goal works.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import mujoco
import numpy as np

_RGBA_PLATFORM = (0.58, 0.61, 0.65, 1.0)

PLATFORM_FRICTION = (1.0, 0.005, 0.0001)

# Spelled out rather than inherited - see the note in container.py
_CONTACT = dict(condim=3, margin=0.0, contype=1, conaffinity=1,
                friction=list(PLATFORM_FRICTION))

_DECOR = dict(contype=0, conaffinity=0, mass=0.0)        # decoration: no contacts, no mass


@dataclass
class Platform:
    """Where a pad ended up and how its goal works. With ``kind`` "site", a box is
    delivered at ``goal_pos``, which then rises by ``goal_step``. With "region",
    a box is delivered with its centre anywhere in [region_lower, region_upper]
    and the goal stays put."""
    geom: str
    lower: np.ndarray       # outer AABB, walls included
    upper: np.ndarray
    pad_lower: np.ndarray
    pad_upper: np.ndarray
    goal_site: str
    goal_pos: np.ndarray
    goal_step: float
    kind: str = "site"
    region_lower: np.ndarray | None = None
    region_upper: np.ndarray | None = None


def add_platform(
    spec: mujoco.MjSpec,
    *,
    name: str,
    center_x: float,
    y_min: float,
    size: float,
    z_playing_field: float,
    box_size: float,
    goal_name: str | None = None,
    walls: bool = False,
    wall_height: float = 0.1,
    wall_thickness: float = 0.025,
    shelf_levels: int = 0,
    box_clearance: float = 0.0,
    rgba=_RGBA_PLATFORM,
) -> Platform:
    """A ``size`` x ``size`` pad, its -y face at ``y_min``, centred on
    ``center_x`` with its top at ``z_playing_field``.

    Walls sit outside the pad and rise from its top, so they don't shrink the
    stacking area. The goal site starts half a box above the pad and rises by
    ``goal_step`` per stacked box. ``shelf_levels`` > 0 makes a shelf three pads
    wide in x, with a back wall and one board per level. Levels are
    ``box_size + box_clearance + wall_thickness`` apart, and that is
    ``goal_step``.
    """
    box = mujoco.mjtGeom.mjGEOM_BOX
    shelf = shelf_levels > 0
    half = np.array([size / 2.0, size / 2.0, wall_thickness / 2.0])

    if shelf:
        half[0] = 3*half[0]

    pos = np.array([center_x, y_min + size / 2.0, z_playing_field])
    pad_lower, pad_upper = pos - half, pos + half

    body = spec.worldbody.add_body(name=name, pos=pos.tolist())
    body.add_geom(name=f"{name}_pad", type=box, size=half.tolist(),
                  rgba=list(rgba), **_CONTACT)
    goal_name = f"{name}_goal" if goal_name is None else goal_name
    goal_local = [0.0, 0.0, half[2] + box_size / 2.0]
    body.add_site(name=goal_name, pos=goal_local, group=4,
                  size=[0.01, 0.01, 0.01])

    t = wall_thickness
    goal_step, h = box_size, wall_height
    if shelf:
        goal_step = box_size + box_clearance + t
        h = shelf_levels * goal_step - t
        for k in range(1, shelf_levels+1):
            body.add_geom(name=f"{name}_board{k}", type=box,
                          pos=[0, 0, half[2] + k * goal_step - t / 2.0],
                          size=[half[0], half[1], t / 2.0],
                          rgba=list(rgba), **_CONTACT)

    lower, upper = pad_lower.copy(), pad_upper.copy()
    z = half[2] + h / 2.0
    parts = []
    if walls or shelf:
        side = t if walls else 0.0           # widen the back wall to close the corners
        parts.append(("back", [0, (size + t) / 2.0, z],
                      [half[0] + side, t / 2.0, h / 2.0 + t]))
        upper += [side, t, h]
        lower -= [side, 0.0, 0.0]
    if walls:
        parts += [("left", [-(size + t) / 2.0, 0, z], [t / 2.0, size / 2.0, h / 2.0]),
                  ("right", [(size + t) / 2.0, 0, z], [t / 2.0, size / 2.0, h / 2.0])]
    for wname, wpos, wsize in parts:
        body.add_geom(name=f"{name}_{wname}", type=box, pos=wpos,
                      size=wsize, rgba=list(rgba), **_CONTACT)

    leg_h = z_playing_field - half[2]
    leg_x, leg_y = half[0] - t / 2.0, half[1] - t / 2.0
    if leg_h > 0.0:
        for gname, gx, gy in (("leg_front_left", -leg_x, -leg_y),
                              ("leg_front_right", leg_x, -leg_y),
                              ("leg_back_left", -leg_x, leg_y),
                              ("leg_back_right", leg_x, leg_y)):
            body.add_geom(name=f"{name}_{gname}", type=box,
                          pos=[gx, gy, -half[2] - leg_h / 2.0],
                          size=[t / 2.0, t / 2.0, leg_h / 2.0],
                          rgba=list(rgba), **_DECOR)

    return Platform(geom=f"{name}_pad", lower=lower, upper=upper,
                    pad_lower=pad_lower, pad_upper=pad_upper,
                    goal_site=goal_name, goal_pos=pos + np.asarray(goal_local),
                    goal_step=float(goal_step))


_RGBA_AREA = (0.18, 0.44, 0.84, 0.42)


def add_floor_area(
    spec: mujoco.MjSpec,
    *,
    name: str,
    center_x: float,
    y_min: float,
    n: int,
    z_playing_field: float,
    box_size: float,
    box_clearance: float,
    wall_thickness: float = 0.025,
    goal_name: str | None = None,
    rgba=_RGBA_PLATFORM,
    area_rgba=_RGBA_AREA,
    height_boxes: float = 3.0,
) -> Platform:
    """A flat pad whose whole top is a target area for ``n`` x ``n`` boxes.

    A box is delivered by resting with its centre over the area, up to
    ``height_boxes`` box heights above the pad. The goal site marks the area's
    centre, for the observation.
    """
    side = float(n) * box_size + (float(n) + 1.0) * box_clearance
    p = add_platform(spec, name=name, center_x=center_x, y_min=y_min, size=side,
                     z_playing_field=z_playing_field, box_size=box_size,
                     goal_name=goal_name, walls=False, wall_thickness=wall_thickness,
                     shelf_levels=0, box_clearance=box_clearance, rgba=rgba)
    top = float(p.pad_upper[2])
    body = spec.body(name)
    # A thin tinted sheet on the pad, drawn only
    body.add_geom(name=f"{name}_area", type=mujoco.mjtGeom.mjGEOM_BOX,
                  pos=[0.0, 0.0, wall_thickness / 2.0 + 0.0005],
                  size=[side / 2.0, side / 2.0, 0.0005], rgba=list(area_rgba), **_DECOR)
    lo = np.array([p.pad_lower[0], p.pad_lower[1], top])
    hi = np.array([p.pad_upper[0], p.pad_upper[1], top + float(height_boxes) * box_size])
    return replace(p, kind="region", region_lower=lo, region_upper=hi, goal_step=0.0)
