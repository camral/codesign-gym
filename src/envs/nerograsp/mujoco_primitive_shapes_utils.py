"""The graspable objects: templates read from their MJCF, and the one padded object body every world shares.

All worlds share one model layout, so the scene holds a single free body,
``grasp_object``, with as many box and cylinder geom slots as the largest
object in the task needs. Each world fills the slots of its own object and
shrinks the rest to unused placeholders (``UNUSED_GEOM_SIZE``, transparent).
The object is placed at its MJCF pose mapped into the Nero frame, centred on
its centre of mass, and lifted to ``SPAWN_FLOOR_CLEARANCE_M`` above the floor.

The washer is resized at every reset (radius and thickness) without a
recompile; see ``NeroGraspEnv._apply_washers``.

- GeomTemplate, ObjectTemplate, load_object_templates: what each object's MJCF defines;
- object_body_element: the padded object body, as MJCF;
- place_object: fill one world's object body and slots in its ``MjSpec``;
- object_keypoints: surface points of an object, for the distance reward.
"""

from __future__ import annotations

import pathlib
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import mujoco
import numpy as np
import scipy.spatial.transform

from .traj_utils import apply_nero_coordinate_transform

OBJECT_BODY_NAME = "grasp_object"
OBJECT_FREEJOINT_NAME = "grasp_object_freejoint"
OBJECT_BOX_PREFIX = "grasp_object_box"
OBJECT_CYL_PREFIX = "grasp_object_cyl"
UNUSED_GEOM_POS = np.array([0.0, 0.0, 0.0], dtype=np.float32)
UNUSED_GEOM_SIZE = np.array([1e-6, 1e-6, 1e-6], dtype=np.float32)
UNUSED_GEOM_RGBA = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
SPAWN_FLOOR_CLEARANCE_M = 0.001

OBJECT_GROUPS = {
    "small": ["scissors", "tweezers", "spatula", "stick", "screw_long"],
    "tiny": ["set_screw", "t_nut", "lock_nut", "screw"],
    "thin": ["washer"],
}
WASHER_TEMPLATE_RADIUS = 0.015
WASHER_DEFAULT_RADIUS = 0.010
WASHER_MAX_RANDOMIZATION_DELTA = 0.002
WASHER_DEFAULT_HALF_HEIGHT = 0.0015
WASHER_MAX_HALF_HEIGHT_DELTA = 0.0010


def resolve_object_names(selector: str) -> list[str]:
    """``all``, a group name, or one object name -> the object names, in task order."""
    if selector == "all":
        return sum(OBJECT_GROUPS.values(), [])
    return list(OBJECT_GROUPS[selector]) if selector in OBJECT_GROUPS else [selector]


@dataclass(frozen=True)
class GeomTemplate:
    geom_type: str
    size: tuple[float, float, float]
    pos: tuple[float, float, float]
    quat: tuple[float, float, float, float]
    rgba: tuple[float, float, float, float] | None = None
    friction: tuple[float, float, float] | None = None
    solref: tuple[float, float] | None = None
    solimp: tuple[float, float, float, float, float] | None = None


@dataclass(frozen=True)
class ObjectTemplate:
    name: str
    body_pos: tuple[float, float, float]
    body_quat: tuple[float, float, float, float]
    inertial_pos: tuple[float, float, float]
    mass: float
    fullinertia: tuple[float, float, float, float, float, float]
    box_geoms: tuple[GeomTemplate, ...]
    cylinder_geoms: tuple[GeomTemplate, ...]


def _floats(value: str | None, expected_len: int, default: tuple[float, ...]) -> tuple[float, ...]:
    if value is None:
        return default
    parsed = tuple(float(x) for x in value.split())
    if len(parsed) != expected_len:
        raise ValueError(f"Expected {expected_len} values, got {len(parsed)} from '{value}'")
    return parsed


def load_object_templates(objects_dir: pathlib.Path, names: list[str]) -> dict[str, ObjectTemplate]:
    """Parse ``<objects_dir>/<name>/<name>.xml`` for each name."""
    templates = {}
    for name in names:
        xml_path = objects_dir / name / f"{name}.xml"
        if not xml_path.is_file():
            raise ValueError(f"Requested object '{name}' does not exist in {objects_dir}")
        root = ET.fromstring(xml_path.read_text())
        body = root.find(".//worldbody/body")
        inertial = body.find("inertial") if body is not None else None
        if body is None or inertial is None:
            raise ValueError(f"Malformed object XML: {xml_path}")
        boxes, cylinders = [], []
        for geom in body.findall("geom"):
            geom_type = geom.get("type", "sphere")
            if geom_type not in {"box", "cylinder"}:
                raise ValueError(f"Unsupported geom type '{geom_type}' in object XML {xml_path}; objects are built from boxes and cylinders")
            size = tuple(float(x) for x in geom.get("size", "1 1 1").split())
            template = GeomTemplate(
                geom_type=geom_type,
                size=(size + (0.0,) * (3 - len(size)))[:3],
                pos=_floats(geom.get("pos"), 3, (0.0, 0.0, 0.0)),
                quat=_floats(geom.get("quat"), 4, (1.0, 0.0, 0.0, 0.0)),
                rgba=_floats(geom.get("rgba"), 4, ()) if geom.get("rgba") is not None else None,
                friction=_floats(geom.get("friction"), 3, ()) if geom.get("friction") is not None else None,
                solref=_floats(geom.get("solref"), 2, ()) if geom.get("solref") is not None else None,
                solimp=_floats(geom.get("solimp"), 5, ()) if geom.get("solimp") is not None else None,
            )
            (boxes if geom_type == "box" else cylinders).append(template)
        templates[name] = ObjectTemplate(
            name=name,
            body_pos=_floats(body.get("pos"), 3, (0.0, 0.0, 0.0)),
            body_quat=_floats(body.get("quat"), 4, (1.0, 0.0, 0.0, 0.0)),
            inertial_pos=_floats(inertial.get("pos"), 3, (0.0, 0.0, 0.0)),
            mass=float(inertial.get("mass", "0")),
            fullinertia=_floats(inertial.get("fullinertia"), 6, (1e-9, 1e-9, 1e-9, 0.0, 0.0, 0.0)),
            box_geoms=tuple(boxes), cylinder_geoms=tuple(cylinders))
    return templates


def object_body_element(max_boxes: int, max_cylinders: int) -> ET.Element:
    """The shared free object body with ``max_boxes`` box and ``max_cylinders`` cylinder slots."""
    root = ET.Element("mujoco")
    worldbody = ET.SubElement(root, "worldbody")
    body = ET.SubElement(worldbody, "body", {"name": OBJECT_BODY_NAME, "pos": "0.36 0.05 0.003", "quat": "1 0 0 0"})
    ET.SubElement(body, "inertial", {"pos": "0 0 0", "mass": "0.01", "fullinertia": "1e-4 1e-4 1e-4 0 0 0"})
    ET.SubElement(body, "freejoint", {"name": OBJECT_FREEJOINT_NAME})
    for idx in range(max_boxes):
        ET.SubElement(body, "geom", {"name": f"{OBJECT_BOX_PREFIX}_{idx:02d}", "class": "small_obj_collision",
                                     "type": "box", "size": "0.001 0.001 0.001", "mass": "0"})
    for idx in range(max_cylinders):
        ET.SubElement(body, "geom", {"name": f"{OBJECT_CYL_PREFIX}_{idx:02d}", "class": "small_obj_collision",
                                     "type": "cylinder", "size": "0.001 0.001 0", "mass": "0"})
    return root


def _scipy_quat(q):
    return [q[1], q[2], q[3], q[0]]


def _lowest_object_z(template: ObjectTemplate, rot: scipy.spatial.transform.Rotation, ipos: np.ndarray) -> float:
    """Lowest world-frame z coordinate of ``template``'s geoms relative to its center of mass."""
    min_world_z = 0.0
    for box in template.box_geoms:
        w_pos = rot.apply(np.asarray(box.pos) - ipos)
        w_rot = rot * scipy.spatial.transform.Rotation.from_quat(_scipy_quat(box.quat))
        for dx in (-1.0, 1.0):
            for dy in (-1.0, 1.0):
                for dz in (-1.0, 1.0):
                    corner = w_rot.apply([dx * box.size[0], dy * box.size[1], dz * box.size[2]])
                    min_world_z = min(min_world_z, w_pos[2] + corner[2])
    for cyl in template.cylinder_geoms:
        w_pos = rot.apply(np.asarray(cyl.pos) - ipos)
        axis_z = (rot * scipy.spatial.transform.Rotation.from_quat(_scipy_quat(cyl.quat))).apply([0.0, 0.0, 1.0])
        z_extent = cyl.size[1] * abs(axis_z[2]) + cyl.size[0] * np.sqrt(max(0.0, 1.0 - axis_z[2] ** 2))
        min_world_z = min(min_world_z, w_pos[2] - z_extent)
    return min_world_z


def place_object(spec: mujoco.MjSpec, template: ObjectTemplate, max_boxes: int, max_cylinders: int) -> None:
    """Pose, inertia and geom slots of one world's object."""
    body = spec.body(OBJECT_BODY_NAME)
    if body is None:
        raise ValueError(f"Canonical object body '{OBJECT_BODY_NAME}' not found in spec.")
    new_pos_xy, body_q = apply_nero_coordinate_transform([template.body_pos[0], template.body_pos[1]], template.body_quat)
    rot = scipy.spatial.transform.Rotation.from_quat(_scipy_quat(body_q))
    ipos = np.array(template.inertial_pos, dtype=np.float64)
    ipos_global = rot.apply(ipos)
    spawn_z = SPAWN_FLOOR_CLEARANCE_M - _lowest_object_z(template, rot, ipos)

    body.pos = (new_pos_xy[0] + ipos_global[0], new_pos_xy[1] + ipos_global[1], spawn_z + ipos_global[2])
    q = rot.as_quat()
    body.quat = (q[3], q[0], q[1], q[2])
    body.mass = template.mass
    body.ipos = (0.0, 0.0, 0.0)  # geoms are shifted so the body frame is the centre of mass
    body.fullinertia = template.fullinertia

    for prefix, slots, geoms in ((OBJECT_BOX_PREFIX, max_boxes, template.box_geoms),
                                 (OBJECT_CYL_PREFIX, max_cylinders, template.cylinder_geoms)):
        for j in range(slots):
            geom = spec.geom(f"{prefix}_{j:02d}")
            if geom is None:
                raise ValueError(f"Missing canonical object geom slot {prefix}_{j:02d}")
            geom.mass = 0.0
            if j >= len(geoms):
                geom.size = UNUSED_GEOM_SIZE
                geom.pos = UNUSED_GEOM_POS
                geom.quat = (1.0, 0.0, 0.0, 0.0)
                geom.rgba = UNUSED_GEOM_RGBA
                continue
            src = geoms[j]
            if template.name == "washer" and prefix == OBJECT_CYL_PREFIX and j == 0:
                geom.size = np.array([WASHER_DEFAULT_RADIUS, WASHER_DEFAULT_HALF_HEIGHT, src.size[2]], dtype=np.float32)
            else:
                geom.size = src.size
            geom.pos = np.asarray(src.pos) - ipos
            geom.quat = src.quat
            for attr in ("rgba", "friction", "solref", "solimp"):
                if getattr(src, attr) is not None:
                    setattr(geom, attr, getattr(src, attr))


def _quat_to_rot_matrix(q) -> np.ndarray:
    return scipy.spatial.transform.Rotation.from_quat(_scipy_quat(q)).as_matrix().astype(np.float32)


def object_keypoints(template: ObjectTemplate, washer_radius: float | None = None, washer_half_height: float | None = None,
                     num_keypoints: int = 12) -> np.ndarray:
    """``num_keypoints`` surface points in the object's centre-of-mass frame, spread evenly over its geoms.

    Boxes give corners, edge midpoints and face centres; cylinders give three
    rims of 8 points and both cap centres. A washer uses its current size.
    """
    is_washer = template.name == "washer"
    points = []
    for box in template.box_geoms:
        hx, hy, hz = box.size
        pos = np.array(box.pos)
        if is_washer and washer_radius is not None and washer_half_height is not None:
            scale_r = washer_radius / WASHER_TEMPLATE_RADIUS
            hx, hy, hz = hx * scale_r, hy * scale_r, washer_half_height
            pos[0] *= scale_r
            pos[1] *= scale_r
        rot = _quat_to_rot_matrix(box.quat)
        for dx in (-hx, 0, hx):
            for dy in (-hy, 0, hy):
                for dz in (-hz, 0, hz):
                    if abs(dx) == hx or abs(dy) == hy or abs(dz) == hz:
                        points.append(pos + rot.dot(np.array([dx, dy, dz])))
    for cyl in template.cylinder_geoms:
        r = washer_radius if (is_washer and washer_radius is not None) else cyl.size[0]
        hz = washer_half_height if (is_washer and washer_half_height is not None) else cyl.size[1]
        pos = np.array(cyl.pos)
        rot = _quat_to_rot_matrix(cyl.quat)
        for z in (-hz, 0, hz):
            for th in np.linspace(0, 2 * np.pi, 8, endpoint=False):
                points.append(pos + rot.dot(np.array([r * np.cos(th), r * np.sin(th), z])))
        for z in (-hz, hz):
            points.append(pos + rot.dot(np.array([0, 0, z])))

    points = np.array(points, dtype=np.float32)
    if len(points) == 0:
        return np.zeros((num_keypoints, 3), dtype=np.float32)
    points -= np.array(template.inertial_pos, dtype=np.float32)
    idx = np.linspace(0, len(points) - 1, num_keypoints, dtype=int)
    return points[idx][:num_keypoints]
