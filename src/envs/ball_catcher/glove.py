"""The ball-catcher glove: an open box of thin plates attached to the arm's
``arm_tip`` site, with the ``goal_site`` the reward measures to.

- glove_layout: the box layout;
- add_glove: add the glove body and ``goal_site`` to an arm spec.
"""

from __future__ import annotations

import mujoco

_RGBA = (0.55, 0.38, 0.22, 1.0)    # warm saddle-brown catcher's mitt (matches original hue)


def glove_layout(
    width: float,
    height: float,
    wall_h: float,
    thickness: float,
) -> list[tuple[str, list[float], list[float]]]:
    """(name, pos, half-size) of every glove box in the glove body frame."""
    hw, hh, t = width / 2.0, height / 2.0, thickness
    return [
        ("glove_bottom", [0, 0, t / 2], [hh + t, hw + t, t / 2]),
        ("glove_front", [-hh - t / 2, 0, wall_h / 2 + t], [t / 2, hw + t, wall_h / 2]),
        ("glove_right", [t / 2, hw + t / 2, wall_h / 2 + t], [hh + t / 2, t / 2, wall_h / 2]),
    ]


def add_glove(
    spec: mujoco.MjSpec,
    *,
    width: float = 0.32,
    height: float = 0.32,
    wall_h: float = 0.24,
    thickness: float = 0.03,
    goal_radius=0.1,
    rgba=_RGBA,
) -> "mujoco.MjsBody":
    existing_mats = {m.name for m in spec.materials}
    if "glove_leather" not in existing_mats:
        mat = spec.add_material(name="glove_leather")
        mat.rgba = list(rgba)
        mat.metallic = 0.04
        mat.roughness = 0.74
        mat.specular = 0.14

    site = spec.site("arm_tip")
    body = site.parent.add_body(name="glove", pos=site.pos, quat=site.quat)
    for name, pos, size in glove_layout(width, height, wall_h, thickness):
        body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, pos=pos,
                      size=size, material="glove_leather",
                      condim=4, friction=[1.0, 0.01, 0.001], solref=[0.015, 1.0])
    # goal_site sits in the corner between the front and right walls (group 4: semantic/affordance)
    body.add_site(name="goal_site", pos=[-height / 2 + goal_radius, width / 2 - goal_radius, goal_radius],
                  group=4, size=goal_radius, rgba=[0.1, 0.95, 0.55, 0.3])

    return body
