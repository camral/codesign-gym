"""The container the loader empties, the truck bed it stands on, and the boxes
in it.

The container's walls stand on a baseplate whose top surface is the playing
field. A ramp runs from the baseplate's front (+x) edge down to the ground. The
baseplate also carries a transparent truck body (side walls and a ceiling) and
decorative legs. Boxes are laid out on a grid inside the container.

- add_container: build all of it and return a ``Container``;
- add_box: one free box;
- interior_size: the container interior that fits a box grid.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

# Box colours by index. A box's colour index is the id of the stacker that owns it
BOX_RGB = [
    (0.86, 0.22, 0.16), (0.18, 0.68, 0.32), (0.18, 0.44, 0.84), (0.92, 0.76, 0.14),
    (0.76, 0.24, 0.68), (0.16, 0.72, 0.76), (0.92, 0.48, 0.12), (0.55, 0.16, 0.16),
    (0.48, 0.20, 0.52), (0.16, 0.45, 0.24), (0.88, 0.46, 0.46), (0.48, 0.82, 0.52),
    (0.44, 0.56, 0.88), (0.16, 0.18, 0.20), (0.50, 0.52, 0.55), (0.90, 0.92, 0.94),
]
COLOR_NAMES = {"red": 0, "green": 1, "blue": 2, "yellow": 3, "magenta": 4, "cyan": 5, "orange": 6}

_RGBA_WALL = (0.42, 0.45, 0.49, 1.0)
_RGBA_BED = (0.28, 0.31, 0.35, 1.0)
_RGBA_BODY = (0.78, 0.80, 0.84, 0.14)    # frosted trailer shell, keeps loader/boxes clear

BOX_FRICTION = (1.0, 0.005, 0.0001)

# Contact attributes are spelled out on every geom. Attaching a robot spec
# replaces the scene's main default class with the robot's, whose margin and
# conaffinity would float the boxes off the floor
_CONTACT = dict(condim=3, margin=0.0, contype=1, conaffinity=1,
                friction=list(BOX_FRICTION))

_DECOR = dict(contype=0, conaffinity=0, mass=0.0)        # decoration: no contacts, no mass

_NO_TURN = [1.0, 0.0, 0.0, 0.0]


def add_box(spec: mujoco.MjSpec, *, name: str, pos, box_size: float,
            box_mass: float, color: int = 0, palette=BOX_RGB):
    """One free box ``name`` with geom ``<name>_geom``, a child of the worldbody
    (a free joint must hang off the world)."""
    half = box_size / 2.0
    body = spec.worldbody.add_body(name=name, pos=list(pos))
    body.add_freejoint(name=f"{name}_free")
    body.add_geom(name=f"{name}_geom", type=mujoco.mjtGeom.mjGEOM_BOX,
                  size=[half, half, half], mass=box_mass,
                  rgba=list(palette[int(color) % len(palette)]) + [1.0],
                  **_CONTACT)
    return body


def interior_size(box_size: float, box_array, box_clearance: float) -> np.ndarray:
    """Interior extents: n boxes and n + 1 clearances per axis."""
    n = np.asarray(box_array, dtype=float)
    return n * box_size + (n + 1.0) * box_clearance


@dataclass
class Container:
    """Where the container, baseplate and ramp ended up, and the boxes in it.
    ``lower``/``upper`` bound the container alone, from the playing field up."""
    pos: np.ndarray          # interior centre, world
    interior: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    floor_top_z: float       # surface the boxes rest on
    baseplate_lower: np.ndarray
    baseplate_upper: np.ndarray
    ramp_lower: np.ndarray
    ramp_upper: np.ndarray
    bed_geoms: list[str]
    boxes: list[str]
    box_geoms: list[str]
    box_pos: np.ndarray
    box_color: np.ndarray    # (n,) colour, i.e. stacker index


def add_container(
    spec: mujoco.MjSpec,
    *,
    ramp_edge_x: float,
    center_y: float,
    z_playing_field: float,
    box_size: float,
    box_array=(2, 2, 2),
    box_clearance: float = 0.005,
    box_mass: float = 1.0,
    wall_thickness: float = 0.025,
    opening: str = "top",
    baseplate_dims=(2.0, 1.0),
    ramp_tilt_degs: float = 20.0,
    ramp_lip: float = 0.0,
    ramp_x_offset: float = 0.1,
    z_offset: float = 0.0,
    n_colors: int = 1,
    palette=BOX_RGB,
    rng: np.random.Generator | None = None,
    name: str = "container",
    rgba=_RGBA_WALL,
) -> Container:
    """Build the ramp, the baseplate and the container on it, extending -x from
    ``ramp_edge_x``.

    The ramp climbs from ``ramp_edge_x`` (on the ground, or on a lip
    ``ramp_lip`` tall) at ``ramp_tilt_degs`` to the baseplate's front edge. The
    baseplate runs ``baseplate_dims`` (length, width) back from there, and the
    container sits ``ramp_x_offset`` behind its front edge. ``opening`` is "top"
    (no lid), "front" (a lid and no +x wall) or "all" (no walls). ``z_offset``
    raises the container's floor above the baseplate. Box colours are shuffled
    so every stacker owns the same number of boxes, give or take one.
    """
    if opening not in ("top", "front", "all"):
        raise ValueError(f"container_opening must be 'top', 'front' or 'all', got {opening!r}")
    if len(box_array) != 3:
        raise ValueError(f"box_array must be (nx, ny, nz), got {box_array!r}")
    if len(baseplate_dims) != 2:
        raise ValueError(f"baseplate_dims must be (length, width), got {baseplate_dims!r}")
    if not 0.0 < ramp_tilt_degs < 90.0:
        raise ValueError(f"ramp_tilt_degs must be in (0, 90), got {ramp_tilt_degs}")
    if min(ramp_lip, ramp_x_offset, z_offset) < 0.0:
        raise ValueError(f"ramp_lip, ramp_x_offset and z_offset must be >= 0, got "
                         f"{ramp_lip}, {ramp_x_offset}, {z_offset}")

    box = mujoco.mjtGeom.mjGEOM_BOX
    rng = np.random.default_rng() if rng is None else rng
    t, z_pf = wall_thickness, z_playing_field
    if z_pf < t or z_pf <= ramp_lip:
        raise ValueError(f"z_playing_field {z_pf} is too low for a baseplate {t} "
                         f"thick with a ramp coming down to a lip {ramp_lip} tall")
    inner = interior_size(box_size, box_array, box_clearance)
    plate_len, plate_w = (float(v) for v in baseplate_dims)

    tilt = np.radians(ramp_tilt_degs)
    rise = z_pf - ramp_lip
    plate_front = ramp_edge_x - rise / np.tan(tilt)

    front_x = plate_front - ramp_x_offset        # where the container's material starts
    depth = inner[0] + (2.0 if opening == "top" else 1.0) * t
    width = inner[1] + 2.0 * t
    back_x = front_x - depth
    if plate_len < ramp_x_offset + depth or plate_w < width + 2.0 * t:
        raise ValueError(f"baseplate_dims {tuple(baseplate_dims)} cannot hold a "
                         f"{depth:.3f} x {width:.3f} container set "
                         f"{ramp_x_offset} back from its front edge, between two "
                         f"walls {t} thick")

    floor_top_z = z_pf + z_offset
    cx = front_x - inner[0] / 2.0 - (t if opening == "top" else 0.0)
    pos = np.array([cx, center_y, floor_top_z + inner[2] / 2.0])
    slab_x = (back_x + front_x) / 2.0 - cx

    body = spec.worldbody.add_body(name=name, pos=pos.tolist())
    walls = []
    if opening != "all":
        walls = [
            ("back", [-(inner[0] + t) / 2, 0, 0], [t / 2, width / 2, inner[2] / 2]),
            ("left", [0, -(inner[1] + t) / 2, 0], [inner[0] / 2, t / 2, inner[2] / 2]),
            ("right", [0, (inner[1] + t) / 2, 0], [inner[0] / 2, t / 2, inner[2] / 2]),
        ]
        if opening == "top":
            walls.append(("front", [(inner[0] + t) / 2, 0, 0],
                        [t / 2, width / 2, inner[2] / 2]))
        else:
            walls.append(("lid", [slab_x, 0, (inner[2] + t) / 2],
                        [depth / 2, width / 2, t / 2]))
    if z_offset > 0.0:                           # MuJoCo builds no zero-height box
        walls.append(("floor", [slab_x, 0, -(inner[2] + z_offset) / 2],
                      [depth / 2, width / 2, z_offset / 2]))
    for wname, wpos, wsize in walls:
        body.add_geom(name=f"{name}_{wname}", type=box, pos=wpos, size=wsize,
                      rgba=list(rgba), **_CONTACT)

    # The bed, in world coordinates, moved into the body frame as it is added
    bed = [("baseplate", [plate_front - plate_len / 2.0, center_y, z_pf - t / 2.0],
            [plate_len / 2.0, plate_w / 2.0, t / 2.0], _NO_TURN)]
    # The ramp is a slab tilted about y. Its high end tucks into the baseplate
    # and its low end into the ground or the lip, so no corner shows
    up = np.array([np.sin(tilt), 0.0, np.cos(tilt)])     # normal of the ramp's top face
    top_mid = np.array([(plate_front + ramp_edge_x) / 2.0, center_y,
                        (z_pf + ramp_lip) / 2.0])
    bed.append(("ramp", top_mid - up * t / 2.0,
                [rise / np.sin(tilt) / 2.0, plate_w / 2.0, t / 2.0],
                [np.cos(tilt / 2.0), 0.0, np.sin(tilt / 2.0), 0.0]))
    if ramp_lip > 0.0:
        bed.append(("ramp_lip", [ramp_edge_x - t / 2.0, center_y, ramp_lip / 2.0],
                    [t / 2.0, plate_w / 2.0, ramp_lip / 2.0], _NO_TURN))
    bed_geoms = [f"{name}_{gname}" for gname, _, _, _ in bed]
    for gname, gpos, gsize, gquat in bed:
        body.add_geom(name=f"{name}_{gname}", type=box,
                      pos=(np.asarray(gpos) - pos).tolist(), size=list(gsize),
                      quat=list(gquat), rgba=list(_RGBA_BED), **_CONTACT)

    # Truck body: walls from the baseplate's bottom to the ceiling's top, so the
    # corners are square
    plate_cx = plate_front - plate_len / 2.0
    body_h = plate_w + 2.0 * t
    y_off = (plate_w - t) / 2.0
    for gname, gpos, gsize in (
            ("body_left", [plate_cx, center_y - y_off, z_pf - t + body_h / 2.0],
             [plate_len / 2.0, t / 2.0, body_h / 2.0]),
            ("body_right", [plate_cx, center_y + y_off, z_pf - t + body_h / 2.0],
             [plate_len / 2.0, t / 2.0, body_h / 2.0]),
            ("ceiling", [plate_cx, center_y, z_pf + plate_w + t / 2.0],
             [plate_len / 2.0, plate_w / 2.0, t / 2.0])):
        body.add_geom(name=f"{name}_{gname}", type=box,
                      pos=(np.asarray(gpos) - pos).tolist(), size=list(gsize),
                      rgba=list(_RGBA_BODY), **_CONTACT)

    leg_h = z_pf - t
    back_x_leg = plate_front - plate_len + t / 2.0
    if leg_h > 0.0:
        for gname, gx, gy in (("leg_front_left", plate_front - t / 2.0, -y_off),
                              ("leg_front_right", plate_front - t / 2.0, y_off),
                              ("leg_back_left", back_x_leg, -y_off),
                              ("leg_back_right", back_x_leg, y_off)):
            body.add_geom(
                name=f"{name}_{gname}", type=box,
                pos=(np.array([gx, center_y + gy, leg_h / 2.0]) - pos).tolist(),
                size=[t / 2.0, t / 2.0, leg_h / 2.0],
                rgba=list(_RGBA_BED), **_DECOR)

    # The first box sits one clearance off the wall
    step = box_size + box_clearance
    axes = [pos[i] - inner[i] / 2.0 + box_clearance + box_size / 2.0
            + step * np.arange(box_array[i]) for i in range(3)]

    names, geoms, positions = [], [], []
    n_boxes = int(np.prod(box_array))
    colors = rng.permutation(np.arange(n_boxes) % max(int(n_colors), 1))
    i = 0
    for ix in range(box_array[0]):
        for iy in range(box_array[1]):
            for iz in range(box_array[2]):
                bpos = [axes[0][ix], axes[1][iy], axes[2][iz]]
                bname = f"box_{i}"
                add_box(spec, name=bname, pos=bpos, box_size=box_size,
                        box_mass=box_mass, color=int(colors[i]), palette=palette)
                names.append(bname)
                geoms.append(f"{bname}_geom")
                positions.append(bpos)
                i += 1

    top_z = floor_top_z + inner[2] + (t if opening == "front" else 0.0)
    return Container(
        pos=pos, interior=inner,
        lower=np.array([back_x, center_y - width / 2.0, z_pf]),
        upper=np.array([front_x, center_y + width / 2.0, top_z]),
        floor_top_z=float(floor_top_z),
        baseplate_lower=np.array([plate_front - plate_len,
                                  center_y - plate_w / 2.0, z_pf - t]),
        baseplate_upper=np.array([plate_front, center_y + plate_w / 2.0, z_pf]),
        ramp_lower=np.array([plate_front, center_y - plate_w / 2.0, 0.0]),
        ramp_upper=np.array([ramp_edge_x, center_y + plate_w / 2.0, z_pf]),
        bed_geoms=bed_geoms, boxes=names, box_geoms=geoms,
        box_pos=np.asarray(positions), box_color=np.asarray(colors))
