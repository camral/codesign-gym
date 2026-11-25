#!/usr/bin/env python3
"""Render saved morphology fields in NVIDIA Newton, without importing a solver.

Input: a manifest containing candidates, each with a relative directory holding
design.json and fields.npz. Surfaces are the exposed bounds of occupied sampling
cells. They are deliberately not labelled MPM particles or FEM/spring elements.
Run with the native Newton environment, independently of the JAX environment.
"""

from __future__ import annotations

import argparse
import colorsys
import hashlib
import html
import importlib.metadata
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias, cast
from urllib.parse import quote

import numpy as np

if TYPE_CHECKING or __package__:
    from .view_newton import Studio, world_coordinates
else:
    from view_newton import Studio, world_coordinates

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray
    from PIL import Image, ImageFont
    from warp import array as WarpArray

    from .viewer_types import Candidate

TITLES = {
    "primitives": "Geometric primitives",
    "direct": "Direct grid",
    "bspline": "B-spline field",
    "cppn": "CPPN network",
}
PASSIVE = (0.41, 0.46, 0.50)
CHANNELS = [
    (0.04, 0.61, 0.62),
    (0.95, 0.43, 0.23),
    (0.39, 0.47, 0.86),
    (0.79, 0.33, 0.62),
    (0.72, 0.65, 0.13),
    (0.20, 0.64, 0.39),
]
MATERIAL_STOPS = np.array(
    [
        [0.13, 0.33, 0.50],
        [0.12, 0.59, 0.64],
        [0.69, 0.77, 0.57],
        [0.94, 0.61, 0.31],
        [0.75, 0.27, 0.20],
    ]
)

# Recordings carry floating coordinates and integer connectivity with saved dtypes.
HostArray: TypeAlias = "NDArray[Any]"


# Gallery policies inherited from the existing atlas, not decoder/physics rules.
CELL_CENTER_TOLERANCE = 2e-6  # Normalized [-1, 1] coordinates; accepts saved float32 grids.
DISPLAY_OCCUPANCY_THRESHOLD = 0.5  # Inclusive silhouette classification for this gallery.
MATERIAL_COLOR_BINS = 16  # One shared discretization for faces, legend and provenance.
DISPLAY_FRAME_RATE = 60.0  # Hz of render timestamps only; no simulation is stepped here.
DEFAULT_IMAGE_WIDTH = 1200  # px; also the reference width for proportional label layout.
DEFAULT_IMAGE_HEIGHT = 900  # px.


@dataclass(frozen=True)
class AtlasCamera:
    """Inherited framing choices; visual calibration, independent of physical accuracy.

    Angles are degrees. Clip/distance values use display units after the metre
    scale. Fit fractions reserve screen space for labels. Their original tuning
    was not recorded; preserve them for consistent historical gallery framing.
    """

    display_units_per_metre: float = 10.0
    planar_angles: tuple[float, float] = (-81.0, 9.0)
    spatial_angles: tuple[float, float] = (-57.0, 24.0)
    vertical_fov: float = 32.0
    near_clip: float = 0.005
    far_clip: float = 250.0
    horizontal_fit: float = 0.84
    vertical_fit: float = 0.65
    minimum_distance: float = 0.1
    distance_margin: float = 1.03


ATLAS_CAMERA = AtlasCamera()


def digest(path: str | Path) -> str:
    """Hash file bytes for provenance; propagate I/O failures."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def channel_color(channel: int) -> tuple[float, float, float]:
    """Return a deterministic RGB display color for an actuator channel."""
    return (
        CHANNELS[channel]
        if 0 <= channel < len(CHANNELS)
        else colorsys.hsv_to_rgb((channel * 0.61803398875) % 1.0, 0.61, 0.81)
    )


def material_color(fraction: float) -> HostArray:
    """Interpolate the display palette for a normalized stiffness fraction."""
    coordinate = np.clip(fraction, 0.0, 1.0) * (len(MATERIAL_STOPS) - 1)
    left = min(int(coordinate), len(MATERIAL_STOPS) - 2)
    return MATERIAL_STOPS[left] * (left + 1 - coordinate) + MATERIAL_STOPS[left + 1] * (
        coordinate - left
    )


def load_candidate(root: Path, entry: dict[str, Any]) -> Candidate:
    """Load and validate owned host field arrays beneath root.

    Reject incomplete grids, malformed fields, invalid material values or escaping paths."""
    directory = (root / entry["directory"]).resolve()
    if not directory.is_relative_to(root):
        raise ValueError("Candidate directory must remain inside the input directory")
    with np.load(directory / "fields.npz", allow_pickle=False) as saved:
        fields = {name: saved[name] for name in saved.files}
    required = {
        "points",
        "grid_shape",
        "size",
        "occupancy",
        "young",
        "density",
        "strength",
        "direction",
        "actuator_type",
        "channel",
    }
    if missing := required - fields.keys():
        raise ValueError(f"{directory}: missing fields {sorted(missing)}")
    dim = int(entry["dim"])
    shape = tuple(int(v) for v in fields["grid_shape"])
    size = np.asarray(fields["size"], float)
    points = np.asarray(fields["points"], float)
    if (
        dim not in (2, 3)
        or len(shape) != dim
        or min(shape) < 1
        or size.shape != (dim,)
        or np.any(size <= 0)
    ):
        raise ValueError(f"{directory}: invalid dimension, grid shape or physical size")
    count = int(np.prod(shape))
    if points.shape != (count, dim) or fields["direction"].shape != (count, dim):
        raise ValueError(f"{directory}: inconsistent sample coordinates or directions")
    for name in required - {"points", "grid_shape", "size", "direction"}:
        if fields[name].shape != (count,) or not np.isfinite(fields[name]).all():
            raise ValueError(f"{directory}: invalid {name} array")
    if not np.isfinite(points).all() or not np.isfinite(size).all():
        raise ValueError(f"{directory}: nonfinite coordinates or size")
    coordinates = np.rint((points + 1.0) * np.asarray(shape) / 2.0 - 0.5).astype(int)
    expected = (coordinates + 0.5) * 2.0 / np.asarray(shape) - 1.0
    if (
        not np.allclose(points, expected, rtol=0, atol=CELL_CENTER_TOLERANCE)
        or np.any(coordinates < 0)
        or np.any(coordinates >= np.asarray(shape))
        or len(np.unique(coordinates, axis=0)) != count
    ):
        raise ValueError(f"{directory}: points must be a complete cell-centred sampling grid")
    active = fields["occupancy"] >= DISPLAY_OCCUPANCY_THRESHOLD
    if not active.any() or np.any(fields["young"][active] <= 0):
        raise ValueError(f"{directory}: empty geometry or nonpositive stiffness")
    if not (directory / "design.json").is_file():
        raise ValueError(f"{directory}: missing reusable design.json")
    return cast(
        "Candidate",
        dict(
            entry,
            path=directory,
            fields=fields,
            dim=dim,
            shape=shape,
            size=size,
            coordinates=coordinates,
            active=active,
        ),
    )


def exposed_faces(candidate: Candidate, extrusion: float) -> tuple[HostArray, HostArray]:
    """Return exact occupied-cell exterior quads and source-sample indices.

    Vertex coordinates are physical, Y-up. Each quad has independent vertices,
    so face normals remain flat and material boundaries do not blend.
    """
    shape, dim = candidate["shape"], candidate["dim"]
    step = candidate["size"] / np.asarray(shape)
    active_ids = np.flatnonzero(candidate["active"])
    lookup = {tuple(candidate["coordinates"][index]): index for index in active_ids}
    quads, owners = [], []
    for index in active_ids:
        coord = candidate["coordinates"][index]
        lower = coord * step - candidate["size"] / 2.0
        upper = lower + step
        if dim == 2:
            lower = np.r_[lower, -extrusion / 2.0]
            upper = np.r_[upper, extrusion / 2.0]
        for axis in range(3):
            rest = [d for d in range(3) if d != axis]
            for sign in (-1, 1):
                if axis < dim:
                    neighbor = coord.copy()
                    neighbor[axis] += sign
                    if tuple(neighbor) in lookup:
                        continue
                face = []
                for u, v in ((0, 0), (1, 0), (1, 1), (0, 1)):
                    vertex = lower.copy()
                    vertex[axis] = upper[axis] if sign > 0 else lower[axis]
                    vertex[rest[0]] = upper[rest[0]] if u else lower[rest[0]]
                    vertex[rest[1]] = upper[rest[1]] if v else lower[rest[1]]
                    face.append(vertex)
                face = np.asarray(face)
                normal = np.cross(face[1] - face[0], face[2] - face[0])
                if normal[axis] * sign < 0:
                    face = face[::-1]
                quads.append(face)
                owners.append(index)
    return np.asarray(quads), np.asarray(owners, int)


def face_groups(
    candidate: Candidate,
    owners: HostArray,
    mode: str,
    stiffness_bounds: tuple[float, float],
    bins: int = MATERIAL_COLOR_BINS,
) -> tuple[HostArray, dict[int, HostArray | tuple[float, float, float]]]:
    """Return per-face color groups and their RGB palette for saved field samples."""
    f = candidate["fields"]
    if mode == "material":
        low, high = stiffness_bounds
        fraction = (
            np.zeros(len(owners)) if high == low else (f["young"][owners] - low) / (high - low)
        )
        groups = np.minimum((np.clip(fraction, 0, 1) * bins).astype(int), bins - 1)
        colors: dict[int, HostArray | tuple[float, float, float]] = {
            int(group): material_color((group + 0.5) / bins) for group in np.unique(groups)
        }
    else:
        passive = (f["actuator_type"][owners] == 0) | (f["strength"][owners] <= 0)
        groups = np.where(passive, -1, f["channel"][owners]).astype(int)
        colors = {
            int(group): PASSIVE if group == -1 else channel_color(int(group))
            for group in np.unique(groups)
        }
    return groups, colors


class MorphologyStudio:
    """Own optional Newton buffers for displaying sampled morphology fields."""

    def __init__(self, args: argparse.Namespace, candidates: list[Candidate]) -> None:
        """Allocate graphics resources and fit a shared camera to all candidate envelopes."""
        import warp as wp
        from newton.viewer import ViewerGL
        from pyglet.math import Vec3

        self.wp, self.args = wp, args
        wp.set_device("cuda:0")
        self.viewer = ViewerGL(width=args.width, height=args.height, vsync=False, headless=True)
        self.viewer.renderer.set_title("Codesign | Morphology Atlas")
        self.viewer.show_ui = False
        renderer = self.viewer.renderer
        renderer.draw_shadows = True
        renderer.draw_sky = True
        renderer.sky_upper, renderer.sky_lower = (0.77, 0.83, 0.87), (0.65, 0.72, 0.76)
        renderer.background_color = (0.89, 0.93, 0.94)
        renderer._light_color = (0.47, 0.44, 0.41)
        camera = ATLAS_CAMERA
        self.scale = camera.display_units_per_metre
        self.buffers: dict[str, WarpArray] = {}
        self.indices: dict[str, WarpArray] = {}
        self.color_buffers: dict[str, WarpArray] = {}
        self.identity = wp.array(
            [wp.transform_identity()], dtype=wp.transform, device=self.viewer.device
        )
        self.unit_scale = wp.array([[1.0, 1.0, 1.0]], dtype=wp.vec3, device=self.viewer.device)
        self.material = wp.array([[0.72, 0.04, 0.0, 0.0]], dtype=wp.vec4, device=self.viewer.device)
        sizes = [
            np.r_[c["size"], args.extrusion] if c["dim"] == 2 else c["size"] for c in candidates
        ]
        envelope = np.max(sizes, axis=0)
        self.vertical_offset = envelope[1] / 2.0
        lower = np.array([-envelope[0] / 2.0, -envelope[2] / 2.0, 0.0])
        upper = np.array([envelope[0] / 2.0, envelope[2] / 2.0, envelope[1]])
        lo, hi = lower * self.scale, upper * self.scale
        center = (lo + hi) / 2.0
        corners = np.array(
            [[lo[d] if bits[d] == 0 else hi[d] for d in range(3)] for bits in np.ndindex(2, 2, 2)]
        )
        planar = all(c["dim"] == 2 for c in candidates)
        az, el = np.radians(camera.planar_angles if planar else camera.spatial_angles)
        front = -np.array([np.cos(az) * np.cos(el), np.sin(az) * np.cos(el), np.sin(el)])
        right = np.cross(front, [0.0, 0.0, 1.0])
        right /= np.linalg.norm(right)
        up = np.cross(right, front)
        relative = corners - center
        xx, yy, zz = relative @ right, relative @ up, relative @ front
        tan, aspect = math.tan(math.radians(camera.vertical_fov / 2)), args.width / args.height
        distance = (
            max(
                float(np.max(np.abs(xx) / (tan * aspect * camera.horizontal_fit) - zz)),
                float(np.max(np.abs(yy) / (tan * camera.vertical_fit) - zz)),
                camera.minimum_distance,
            )
            * camera.distance_margin
        )
        position = center - front * distance
        self.viewer.camera.fov = camera.vertical_fov
        self.viewer.camera.near, self.viewer.camera.far = camera.near_clip, camera.far_clip
        self.viewer.set_camera(
            Vec3(*position),
            math.degrees(math.asin(front[2])),
            math.degrees(math.atan2(front[1], front[0])),
        )
        self.camera = {
            "position_display_units": position.tolist(),
            "envelope_m": envelope.tolist(),
            "shared_across_candidates": True,
            "vertical_translation_m": self.vertical_offset,
            "azimuth_degrees": float(np.degrees(az)),
            "elevation_degrees": float(np.degrees(el)),
            "fov_degrees": camera.vertical_fov,
            "display_scale": self.scale,
        }
        self.frame = 0
        self.previous: list[str] = []
        self.viewer.begin_frame(0.0)
        halfx, halfy = envelope[0] * 0.82, max(envelope[2] * 0.9, envelope[0] * 0.30)
        ground = np.array(
            [
                [-halfx, -halfy, -0.001],
                [halfx, -halfy, -0.001],
                [halfx, halfy, -0.001],
                [-halfx, halfy, -0.001],
            ]
        )
        Studio.mesh(
            self, "/stage", ground, [0, 1, 2, 0, 2, 3], self.color("floor", (0.64, 0.70, 0.72))
        )
        self.viewer.end_frame()

    def color(self, name: str, rgb: ArrayLike) -> WarpArray:
        """Cache and return an RGB buffer on the viewer device."""
        if name not in self.color_buffers:
            self.color_buffers[name] = self.wp.array(
                [rgb], dtype=self.wp.vec3, device=self.viewer.device
            )
        return self.color_buffers[name]

    def render(
        self, candidate: Candidate, mode: str, stiffness_bounds: tuple[float, float], index: int
    ) -> tuple[Image.Image, int]:
        """Return a Pillow frame and exposed-quad count for a saved candidate.

        mode selects material or actuator colors; no elastic elements are instantiated."""
        from PIL import Image

        self.viewer.begin_frame(self.frame / DISPLAY_FRAME_RATE)
        for name in self.previous:
            self.viewer.objects[name + "/surface"].hidden = True
        self.previous = []
        quads, owners = exposed_faces(candidate, self.args.extrusion)
        quads[..., 1] += self.vertical_offset
        groups, colors = face_groups(candidate, owners, mode, stiffness_bounds)
        for group in np.unique(groups):
            selected = quads[groups == group]
            points = world_coordinates(selected.reshape(-1, 3))
            indices = (
                np.arange(len(selected))[:, None] * 4 + np.array([0, 1, 2, 0, 2, 3])
            ).reshape(-1)
            name = f"/candidate_{index}/{mode}/{group}"
            Studio.mesh(self, name, points, indices, self.color(name, colors[int(group)]))
            self.previous.append(name)
        self.viewer.end_frame()
        self.frame += 1
        # Two settled frames include the current mesh in Newton's shadow pass.
        for _ in range(2):
            self.viewer.begin_frame(self.frame / DISPLAY_FRAME_RATE)
            self.viewer.end_frame()
            self.frame += 1
        pixels = self.viewer.get_frame().numpy()
        return Image.fromarray(pixels, "RGB"), len(quads)


def font(size: float, bold: bool = False) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    """Load an available display font, falling back to Pillow defaults."""
    from PIL import ImageFont

    path = Path("C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf")
    return (
        ImageFont.truetype(str(path), max(10, int(size)))
        if path.exists()
        else ImageFont.load_default()
    )


def title_card(
    image: Image.Image,
    candidate: Candidate,
    mode: str,
    stiffness_bounds: tuple[float, float],
    channels: list[int],
) -> Image.Image:
    """Return an annotated copy of the supplied Pillow render."""
    from PIL import ImageDraw

    draw = ImageDraw.Draw(image)
    scale = image.width / DEFAULT_IMAGE_WIDTH
    ink, muted, paper = (30, 48, 63), (81, 104, 119), (241, 246, 247)
    top, bottom = int(105 * scale), image.height - int(112 * scale)
    draw.rectangle((0, 0, image.width, top), fill=paper)
    draw.rectangle((0, bottom, image.width, image.height), fill=paper)
    draw.text(
        (32 * scale, 15 * scale),
        "CODESIGN  /  MORPHOLOGY ATLAS",
        fill=muted,
        font=font(15 * scale, True),
    )
    label = TITLES.get(candidate["representation"], candidate["representation"].title())
    draw.text(
        (30 * scale, 43 * scale),
        f"{label}  /  seed {candidate['seed']}",
        fill=ink,
        font=font(30 * scale, True),
    )
    dims = " x ".join(f"{value * 1000:g}" for value in candidate["size"])
    cells = " x ".join(str(v) for v in candidate["shape"])
    draw.text(
        (32 * scale, bottom + 12 * scale),
        f"{candidate['dim']}D  |  {dims} mm envelope  |  {cells} sampling cells  |  {int(candidate['active'].sum())} occupied",
        fill=ink,
        font=font(17 * scale),
    )
    legend_y = bottom + 44 * scale
    if mode == "material":
        draw.text(
            (32 * scale, legend_y), "YOUNG'S MODULUS", fill=muted, font=font(14 * scale, True)
        )
        left = 210 * scale
        for i in range(MATERIAL_COLOR_BINS):
            rgb = tuple((material_color((i + 0.5) / MATERIAL_COLOR_BINS) * 255).astype(int))
            draw.rectangle(
                (
                    left + i * 15 * scale,
                    legend_y + 1 * scale,
                    left + (i + 1) * 15 * scale,
                    legend_y + 19 * scale,
                ),
                fill=rgb,
            )
        low, high = stiffness_bounds
        draw.text(
            (left + 257 * scale, legend_y),
            f"{low / 1000:.1f} to {high / 1000:.1f} kPa  /  shared linear scale",
            fill=ink,
            font=font(15 * scale),
        )
    else:
        cursor = 32 * scale
        for group, label in [(-1, "Passive")] + [(int(c), f"Channel {c}") for c in channels]:
            rgb = PASSIVE if group < 0 else channel_color(group)
            draw.rectangle(
                (cursor, legend_y + 2 * scale, cursor + 16 * scale, legend_y + 18 * scale),
                fill=tuple(int(v * 255) for v in rgb),
            )
            draw.text((cursor + 23 * scale, legend_y), label, fill=ink, font=font(14 * scale))
            cursor += 139 * scale
    draw.text(
        (32 * scale, bottom + 79 * scale),
        "Decoded design geometry / occupied sampling cells / no dynamics or solver elements",
        fill=muted,
        font=font(14 * scale),
    )
    draw.rectangle((0, image.height - 4 * scale, image.width, image.height), fill=(25, 140, 146))
    return image


def write_gallery(
    directory: Path,
    candidates: list[Candidate],
    images: dict[tuple[str, str], Image.Image],
    stiffness_bounds: tuple[float, float],
    channels: list[int],
) -> None:
    """Write the morphology overview images and HTML beside the saved candidate artifacts."""
    from PIL import Image, ImageDraw, ImageOps

    representations = list(dict.fromkeys(c["representation"] for c in candidates))
    columns = max(
        sum(c["representation"] == representation for c in candidates)
        for representation in representations
    )
    tile_width, tile_height = 420, 315
    for mode in ("material", "actuators"):
        sheet = Image.new(
            "RGB", (columns * tile_width, len(representations) * tile_height + 94), (235, 242, 245)
        )
        draw = ImageDraw.Draw(sheet)
        draw.text((25, 14), "Morphology Atlas", fill=(30, 48, 63), font=font(31, True))
        draw.text(
            (27, 54),
            f"{len(candidates)} random decoded designs / {mode} / shared envelope and camera / geometry only",
            fill=(78, 100, 114),
            font=font(18),
        )
        for row, representation in enumerate(representations):
            for column, candidate in enumerate(
                c for c in candidates if c["representation"] == representation
            ):
                image = images[(candidate["id"], mode)]
                tile = ImageOps.contain(
                    image, (tile_width - 8, tile_height - 8), Image.Resampling.LANCZOS
                )
                sheet.paste(
                    tile,
                    (
                        column * tile_width + (tile_width - tile.width) // 2,
                        94 + row * tile_height + (tile_height - tile.height) // 2,
                    ),
                )
        sheet.save(directory / ("overview.png" if mode == "material" else "overview_actuators.png"))
    rows = []
    for representation in representations:
        cards = []
        for candidate in (c for c in candidates if c["representation"] == representation):
            stem = quote(candidate["id"], safe="")
            relative = Path(candidate["path"]).relative_to(directory.parent).as_posix()
            recipe = quote("../" + relative + "/design.json", safe="/.")
            fields = quote("../" + relative + "/fields.npz", safe="/.")
            cards.append(
                f'<article><a class="picture" href="{stem}_material.png"><img data-stem="{stem}" '
                f'src="{stem}_material.png" alt="{html.escape(candidate["id"])} decoded geometry"></a>'
                f'<div class="cardbody"><h3>Seed {candidate["seed"]}</h3><p>{int(candidate["active"].sum())} occupied cells '
                f'&middot; {candidate["dim"]}D</p><a href="{recipe}">Design JSON</a> &middot; '
                f'<a href="{fields}">Decoded fields</a><div class="views"><button onclick="setCard(this,\'material\')">Material</button>'
                "<button onclick=\"setCard(this,'actuators')\">Actuators</button></div></div></article>"
            )
        rows.append(
            f'<section><h2>{html.escape(TITLES.get(representation, representation))}</h2><div class="grid">{"".join(cards)}</div></section>'
        )
    legend = '<span class="swatch" style="background:rgb(105,117,128)"></span>Passive '
    for channel in channels:
        rgb = ",".join(str(round(v * 255)) for v in channel_color(int(channel)))
        legend += f'<span class="swatch" style="background:rgb({rgb})"></span>Channel {channel} '
    low, high = stiffness_bounds
    document = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Morphology Atlas | Codesign</title><style>
:root{color-scheme:light}*{box-sizing:border-box}body{margin:0;background:#eaf1f4;color:#20394b;font:16px system-ui}
header,main,footer{max-width:1700px;margin:auto;padding:30px 34px}header{padding-top:48px}header small{font-weight:700;letter-spacing:2px;color:#137f86}
h1{font-size:clamp(34px,5vw,56px);letter-spacing:-2px;margin:12px 0}p{line-height:1.6;color:#526d7c}header p{max-width:940px;margin:12px 0}
.toolbar{display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin:24px 0 16px}button{border:1px solid #bbd0d8;border-radius:8px;background:white;color:#176c73;font:600 14px system-ui;padding:10px 16px;cursor:pointer}
button:hover,button[aria-pressed=true]{background:#176f77;color:white;border-color:#176f77}.legend{font-size:14px;line-height:2.3;max-width:1100px}.swatch{display:inline-block;width:14px;height:14px;border-radius:3px;vertical-align:-2px;margin:0 5px 0 14px}
main{padding-top:0}section{margin:20px 0 38px}h2{font-size:25px;margin:0 0 17px}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:22px}article{background:white;border:1px solid #d7e3e8;border-radius:14px;overflow:hidden;box-shadow:0 8px 24px #203c4e0a}
img{width:100%;height:auto;display:block}.cardbody{padding:10px 20px 18px}h3{margin:0;font-size:19px}.cardbody p{margin:5px 0;font-size:14px}a{color:#187b82}.cardbody>a{font-size:14px}.views{display:flex;gap:8px;margin-top:14px}.views button{padding:6px 12px;font-size:13px}footer{font-size:14px;color:#526d7c}
@media(min-width:1500px){.grid{grid-template-columns:repeat(6,minmax(0,1fr))}.cardbody{padding:10px 13px 16px}.views{flex-wrap:wrap}}
@media(max-width:1000px){.grid{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:600px){header,main,footer{padding-left:18px;padding-right:18px}.grid{grid-template-columns:1fr}}
</style><header><small>CODESIGN / NVIDIA NEWTON</small><h1>Morphology Atlas</h1>
<p>Four representations generate their own random shapes and material/actuator fields. Each image displays the exposed boundaries of actual occupied sampling cells. These are design previews; no dynamics have been run, and the sampling cells are not solver elements.</p>
<p>The same camera and physical envelope are used throughout. Planar designs have cosmetic display thickness. Open an image for its dimensions and grid resolution, or reuse its exact saved design.</p>
<div class="toolbar"><button id="all-material" aria-pressed="true" onclick="setAll('material')">Material stiffness</button><button id="all-actuators" aria-pressed="false" onclick="setAll('actuators')">Actuator channels</button><a href="overview.png">Material contact sheet</a><a href="overview_actuators.png">Actuator contact sheet</a></div>
"""
    document += f'<div class="legend">Material: {low / 1000:.1f}–{high / 1000:.1f} kPa, shared linear scale in {MATERIAL_COLOR_BINS} color bins. Actuators: {legend}<br>Channel color represents routing, not a random body-part label. Actuator direction and strength are retained in the decoded fields; they are not encoded by color.</div></header><main>'
    document += (
        "".join(rows)
        + """</main><footer>Exact saved fields &middot; deterministic seeds &middot; no physics re-run &middot; <a href="provenance.json">Rendering provenance</a></footer>
<script>function setCard(button,mode){const card=button.closest('article');setImage(card.querySelector('img'),mode)}function setImage(image,mode){image.src=image.dataset.stem+'_'+mode+'.png';image.parentElement.href=image.src}function setAll(mode){document.querySelectorAll('img[data-stem]').forEach(image=>setImage(image,mode));for(const m of ['material','actuators'])document.getElementById('all-'+m).setAttribute('aria-pressed',String(m===mode))}</script></html>"""
    )
    (directory / "index.html").write_text(document, encoding="utf-8")


def main() -> None:
    """Parse and validate CLI options, run the selected display task, and close graphics resources."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Generated morphology directory containing manifest.json",
    )
    parser.add_argument(
        "--out", type=Path, help="Fresh gallery directory (default: INPUT/rendered)"
    )
    parser.add_argument("--width", type=int, default=DEFAULT_IMAGE_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_IMAGE_HEIGHT)
    parser.add_argument(
        "--extrusion", type=float, default=0.006, help="Cosmetic 2D thickness in meters"
    )
    args = parser.parse_args()
    args.input = args.input.resolve()
    args.out = (args.out or args.input / "rendered").resolve()
    if min(args.width, args.height) < 600 or not 0 < args.extrusion < 1:
        parser.error("Image dimensions must be >=600 and extrusion must be in (0,1) meters")
    if args.out.parent != args.input:
        parser.error("Place --out directly inside --input so portable gallery links remain valid")
    if args.out.exists():
        parser.error(f"Refusing to overwrite gallery: {args.out}")
    manifest = json.loads((args.input / "manifest.json").read_text(encoding="utf-8"))
    candidates = [load_candidate(args.input, entry) for entry in manifest["candidates"]]
    if not candidates or len({c["id"] for c in candidates}) != len(candidates):
        parser.error("Manifest needs nonempty, uniquely identified candidates")
    for candidate in candidates:
        name = candidate["id"]
        if (
            not isinstance(name, str)
            or not name
            or any(
                ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
                for ch in name
            )
        ):
            parser.error("Candidate ids must contain only letters, digits, underscores or hyphens")
    stiffness = np.concatenate([c["fields"]["young"][c["active"]] for c in candidates])
    stiffness_bounds = (float(stiffness.min()), float(stiffness.max()))
    channels = sorted(
        {
            int(value)
            for c in candidates
            for value in c["fields"]["channel"][
                c["active"] & (c["fields"]["actuator_type"] != 0) & (c["fields"]["strength"] > 0)
            ]
        }
    )
    args.out.mkdir(parents=True)
    started = time.perf_counter()
    studio, images, details = MorphologyStudio(args, candidates), {}, []
    try:
        for index, candidate in enumerate(candidates):
            for mode in ("material", "actuators"):
                image, face_count = studio.render(candidate, mode, stiffness_bounds, index)
                image = title_card(image, candidate, mode, stiffness_bounds, channels)
                image.save(args.out / f"{candidate['id']}_{mode}.png")
                images[(candidate["id"], mode)] = image
            details.append(
                {
                    "id": candidate["id"],
                    "occupied_cell_count": int(candidate["active"].sum()),
                    "exposed_quad_count": face_count,
                    "dim": candidate["dim"],
                    "grid_shape": list(candidate["shape"]),
                    "size_m": candidate["size"].tolist(),
                    "source_fields_sha256": digest(candidate["path"] / "fields.npz"),
                    "source_design_sha256": digest(candidate["path"] / "design.json"),
                }
            )
            print(
                f"Rendered {index + 1}/{len(candidates)}: {candidate['id']} / material + actuators",
                flush=True,
            )
        write_gallery(args.out, candidates, images, stiffness_bounds, channels)
        provenance = {
            "renderer": "NVIDIA Newton ViewerGL",
            "physics_rerun": False,
            "geometry": "Exposed occupied sampling-cell faces; these are not instantiated solver elements",
            "occupancy_threshold": DISPLAY_OCCUPANCY_THRESHOLD,
            "two_dimensional_extrusion_m": args.extrusion,
            "two_dimensional_extrusion_is_cosmetic": True,
            "material_color": {
                "property": "young",
                "units": "Pa",
                "range": stiffness_bounds,
                "bins": MATERIAL_COLOR_BINS,
                "scale": "linear, shared across candidates",
            },
            "actuator_color": {
                "property": "channel",
                "passive_if": "actuator_type == 0 or strength <= 0",
                "channels": {str(c): list(channel_color(c)) for c in channels},
            },
            "camera": studio.camera,
            "candidates": details,
            "render_device": str(studio.viewer.device),
            "render_wall_s": time.perf_counter() - started,
            "source_manifest_sha256": digest(args.input / "manifest.json"),
            "source_script_sha256": digest(__file__),
            "mesh_helper_sha256": digest(Path(__file__).with_name("view_newton.py")),
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("newton", "warp-lang", "numpy", "pillow")
            },
            "files_sha256": {p.name: digest(p) for p in sorted(args.out.iterdir()) if p.is_file()},
        }
        (args.out / "provenance.json").write_text(
            json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Gallery: {args.out / 'index.html'}", flush=True)
    finally:
        studio.viewer.close()


if __name__ == "__main__":
    main()
