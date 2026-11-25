"""The cleaning roller on the arm's tip: a cylinder across the tool site, with
compliant, low-friction contact so pressing it on a panel neither bounces the arm off
nor drags the panel."""

from __future__ import annotations

import mujoco

from .panel import PANEL_BIT

ROLLER_BODY = "roller"
_RGBA = (0.18, 0.62, 0.88, 1.0)            # microfiber cleaning brush cyan-blue
_YOKE_RGBA = (0.20, 0.22, 0.26, 1.0)       # anodized aluminum mounting yoke


def add_roller(
    spec: mujoco.MjSpec,
    *,
    site: str = "arm_tip",
    radius: float = 0.05,
    width: float = 0.25,
    mass: float = 0.4,
    solref=(0.015, 1.0),
    friction=(0.05, 0.005, 0.0001),
    rgba=_RGBA,
) -> str:
    """Attach the roller rigidly to ``site``'s body, at the site's frame, and
    return its body name."""
    s = spec.site(site)
    body = s.parent.add_body(name=ROLLER_BODY, pos=list(s.pos), quat=list(s.quat))
    body.add_geom(name=f"{ROLLER_BODY}_geom",
                  type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                  fromto=[0, -width / 2, 0, 0, width / 2, 0],   # along the site's y, so it rolls along x
                  size=[radius, 0, 0], mass=mass,
                  # priority: its solref and friction decide contacts outright
                  solref=list(solref), friction=list(friction), priority=100,
                  margin=0.0,
                  contype=PANEL_BIT | 1, conaffinity=PANEL_BIT,   # touches the plates and bit-0 geoms
                  rgba=list(rgba))
    # Visual-only end-caps / yoke brackets (group 2, zero mass/collision cost)
    cap_r = radius * 1.08
    for side, sign in (("l", 1.0), ("r", -1.0)):
        y_end = sign * (width / 2 + 0.008)
        body.add_geom(name=f"{ROLLER_BODY}_cap_{side}",
                      type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                      fromto=[0, sign * width / 2, 0, 0, y_end, 0],
                      size=[cap_r, 0, 0], group=2,
                      contype=0, conaffinity=0, density=0.0, mass=0.0,
                      rgba=list(_YOKE_RGBA))
    return ROLLER_BODY
