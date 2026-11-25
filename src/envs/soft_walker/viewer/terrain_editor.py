#!/usr/bin/env python3
"""Interactive Newton terrain geometry editor; no simulation or JAX is imported."""

from __future__ import annotations

import argparse
import hashlib
import html
import importlib.metadata
import json
import math
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias, cast

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
# Host-only package namespaces and recipe services do not load JAX. Keep direct
# source-file invocation usable in the separate native Windows Newton viewer.
sys.path.insert(0, str(PROJECT.parents[1]))
from envs.soft_walker.presets.terrain.editor_courses import (  # noqa: E402
    PRESET_DEFAULTS,
    PRESETS,
    make_recipe,
)
from envs.soft_walker.terrain.recipes import (  # noqa: E402
    TerrainRecipe,
    TerrainRecord,
    recipe_terrain,
    validate_recipe,
)

if TYPE_CHECKING or __package__:
    from .view_newton import Studio, rounded_box_surface, terrain_height, world_coordinates
else:
    from view_newton import Studio, rounded_box_surface, terrain_height, world_coordinates

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from PIL import Image, ImageFont
    from warp import array as WarpArray

TITLES = {
    "flat": "Flat / a clean starting point",
    "uphill": "Uphill / a continuous incline",
    "downhill": "Downhill / a continuous descent",
    "cross_slope": "Cross slope / a lateral incline",
    "low_blocks": "Low blocks / a gentle obstacle course",
    "rough_blocks": "Rough blocks / seeded variation",
    "stairs": "Stairs / a rising sequence",
    "hurdles": "Hurdles / repeated clearances",
}


# Recordings carry floating coordinates and integer connectivity with saved dtypes.
HostArray: TypeAlias = "NDArray[Any]"

# Newton's default sidebar is 300 pixels wide. Reserve most of its content
# width for metric labels; the numeric fields remain usable in narrower panes.
_CONTROL_FIELD_WIDTH_FRACTION = 0.4


def unique_directory(base: str | Path) -> Path:
    """Reserve a fresh directory; repeated saves never overwrite earlier recipes."""
    base = Path(base)
    base.parent.mkdir(parents=True, exist_ok=True)
    for number in range(10000):
        candidate = base if not number else base.with_name(f"{base.name}_{number:03d}")
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            continue
    raise RuntimeError(f"No unused save name found beside {base}")


def write_recipe(directory: Path, recipe: object) -> TerrainRecord:
    """Validate and write recipe.json and terrain.json; return the serialized terrain."""
    recipe = validate_recipe(recipe)
    terrain = recipe_terrain(recipe)
    for name, value in (("recipe.json", recipe), ("terrain.json", terrain)):
        (directory / name).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return terrain


class TerrainEditor:
    """Own interactive terrain parameters and Newton display buffers; run no physics."""

    def __init__(self, args: argparse.Namespace) -> None:
        """Allocate the optional CUDA viewer and initialize a validated preset from CLI args."""
        import warp as wp
        from newton.viewer import ViewerGL

        editor = self

        class OrbitViewer(ViewerGL):
            def on_mouse_drag(
                self, x: int, y: int, dx: int, dy: int, buttons: int, modifiers: int
            ) -> None:
                import pyglet

                if self.ui and self.ui.is_capturing():
                    return
                if buttons & pyglet.window.mouse.LEFT:
                    editor.azimuth -= dx * 0.22
                    editor.elevation = float(np.clip(editor.elevation + dy * 0.18, 5.0, 85.0))

            def on_mouse_scroll(self, x: int, y: int, scroll_x: float, scroll_y: float) -> None:
                if self.ui and self.ui.is_capturing():
                    return
                editor.zoom = float(np.clip(editor.zoom * math.exp(-scroll_y * 0.09), 0.3, 3.0))

        self.wp, self.args = wp, args
        wp.set_device("cuda:0")
        self.viewer = OrbitViewer(
            width=args.width,
            height=args.height,
            vsync=not bool(args.gallery or args.headless),
            headless=bool(args.gallery or args.headless),
        )
        self.viewer.renderer.set_title("Codesign | Terrain Studio")
        renderer = self.viewer.renderer
        renderer.draw_shadows = True
        renderer.draw_sky = True
        renderer.sky_upper = (0.76, 0.82, 0.87)
        renderer.sky_lower = (0.61, 0.67, 0.70)
        renderer.background_color = (0.88, 0.91, 0.93)
        renderer._light_color = (0.43, 0.41, 0.38)
        self.viewer.show_ui = not bool(args.gallery)
        self.scale = 10.0
        self.buffers: dict[str, WarpArray] = {}
        self.indices: dict[str, WarpArray] = {}
        self.topology: dict[str, tuple[int, bytes]] = {}
        self.identity = wp.array(
            [wp.transform_identity()], dtype=wp.transform, device=self.viewer.device
        )
        self.unit_scale = wp.array([[1.0, 1.0, 1.0]], dtype=wp.vec3, device=self.viewer.device)
        self.material = wp.array([[0.84, 0.02, 0.0, 0.0]], dtype=wp.vec4, device=self.viewer.device)
        self.palette = {
            "floor": self.color((0.58, 0.60, 0.59)),
            "teal": self.color((0.12, 0.52, 0.52)),
            "clay": self.color((0.73, 0.40, 0.27)),
            "slate": self.color((0.26, 0.43, 0.59)),
        }
        self.dim, self.seed = args.dim, args.seed
        self.show_grid, self.show_scale = True, True
        self.azimuth, self.elevation, self.zoom = -62.0, 29.0, 1.0
        self.status = "Adjust geometry, then save a recipe for legacy replay."
        self.error = ""
        self.frame = 0
        self.dirty = True
        self.set_preset(args.preset)
        self.viewer.camera.fov = 38.0
        self.viewer.camera.near = 0.005
        self.viewer.camera.far = 250.0
        self.viewer.register_ui_callback(self.ui, position="side")

    def color(self, rgb: tuple[float, float, float]) -> WarpArray:
        """Allocate one RGB color on the viewer device."""
        return self.wp.array([rgb], dtype=self.wp.vec3, device=self.viewer.device)

    def set_preset(self, name: str) -> None:
        """Replace editing parameters with a preset and mark display geometry for rebuilding."""
        self.preset = name
        if name == "cross_slope":
            self.dim = 3
        self.parameters = dict(PRESET_DEFAULTS[name])
        self.dirty = True

    def recipe(self) -> TerrainRecipe:
        """Return a validated portable recipe from the current editing parameters."""
        return make_recipe(
            self.preset,
            dim=self.dim,
            seed=self.seed,
            count=cast(int, self.parameters["count"]),
            slope_degrees=self.parameters["slope_degrees"],
            start=self.parameters["start"],
            spacing=self.parameters["spacing"],
            width=self.parameters["width"],
            height=self.parameters["height"],
            rounding=self.parameters["rounding"],
        )

    def mesh(self, name: str, points: HostArray, indices: HostArray, color: WarpArray) -> None:
        """Update terrain vertices; replace device mesh resources only if topology changes."""
        # ViewerGL has fixed mesh topology. Recreate only when an edited rounded
        # box changes its tessellation, rather than assigning mismatched buffers.
        topology = (len(points), np.asarray(indices, np.int32).tobytes())
        if name in self.topology and self.topology[name] != topology:
            self.viewer.objects.pop(name + "/surface", None)
            old = self.viewer.objects.pop(name, None)
            if old is not None:
                old.destroy()
            self.buffers.pop(name, None)
            self.indices.pop(name, None)
        self.topology[name] = topology
        Studio.mesh(self, name, np.asarray(points), indices, color)
        self.used_meshes.add(name)

    def lines(
        self, name: str, starts: HostArray, ends: HostArray, color: WarpArray, hidden: bool = False
    ) -> None:
        """Update owned device line buffers and their display visibility."""
        wp = self.wp
        self.viewer.log_lines(
            name,
            wp.array(np.asarray(starts) * self.scale, dtype=wp.vec3, device=self.viewer.device),
            wp.array(np.asarray(ends) * self.scale, dtype=wp.vec3, device=self.viewer.device),
            color,
            width=0.008,
            hidden=hidden,
        )
        self.viewer.lines[name].hidden = hidden

    def rebuild(self) -> None:
        """Rebuild the display mesh from current recipe parameters; no simulation is run."""
        recipe = self.recipe()
        terrain = recipe_terrain(recipe)
        self.current_recipe, self.terrain = recipe, terrain
        self.used_meshes: set[str] = set()
        boxes = [
            (c, b, r)
            for c, b, r, active in zip(
                terrain["centers"], terrain["half_sizes"], terrain["rounding"], terrain["active"]
            )
            if active
        ]
        self.xlo = -0.075
        self.xhi = max(0.78, max((c[0] + b[0] + 0.10 for c, b, _ in boxes), default=0.78))
        self.depth = 0.21 if self.dim == 2 else 0.31
        plane = dict(terrain, active=[])
        x = np.array([self.xlo, self.xhi, self.xhi, self.xlo])
        z = np.array([-self.depth, -self.depth, self.depth, self.depth])
        heights = terrain_height(x, z, plane)
        top = world_coordinates(np.stack((x, heights - 0.00015, z), axis=1))
        bottom = top.copy()
        bottom[:, 2] -= 0.021
        points = np.concatenate((top, bottom))
        faces = [0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7]
        for k in range(4):
            nxt = (k + 1) % 4
            faces.extend((k, nxt, nxt + 4, k, nxt + 4, k + 4))
        self.mesh("/terrain/stage", points, faces, self.palette["floor"])
        bounds = [points]
        accent = (
            "clay"
            if self.preset == "rough_blocks"
            else "slate"
            if self.preset == "hurdles"
            else "teal"
        )
        for index, (center, half_size, radius) in enumerate(boxes):
            # 2D obstacles are infinite out of plane physically; their finite
            # display depth is cosmetic. 3D boxes retain their actual dimensions.
            vertices, indices = rounded_box_surface(center, half_size, radius, self.depth)
            self.mesh(f"/terrain/block_{index}", vertices, indices, self.palette[accent])
            bounds.append(vertices)
        for name in self.topology:
            self.viewer.objects[name + "/surface"].hidden = name not in self.used_meshes

        starts, ends = [], []
        for xx in np.arange(math.ceil(self.xlo / 0.05) * 0.05, self.xhi, 0.05):
            yy = terrain_height(np.full(2, xx), np.array([-self.depth, self.depth]), plane) + 0.0002
            starts.append([xx, self.depth, yy[0]])
            ends.append([xx, -self.depth, yy[1]])
        for zz in np.arange(-self.depth + 0.01, self.depth, 0.05):
            yy = terrain_height(np.array([self.xlo, self.xhi]), np.full(2, zz), plane) + 0.0002
            starts.append([self.xlo, -zz, yy[0]])
            ends.append([self.xhi, -zz, yy[1]])
        self.lines("/terrain/grid", starts, ends, (0.42, 0.46, 0.47), hidden=not self.show_grid)

        # A 16 cm ruler references robot length; no fabricated motion or contact
        # state is shown in this geometry-only editor.
        lateral = -self.depth + 0.04
        rr = np.array([[0.0, lateral], [0.16, lateral]])
        yy = terrain_height(rr[:, 0], rr[:, 1], plane) + 0.003
        a, b = np.array([0.0, -lateral, yy[0]]), np.array([0.16, -lateral, yy[1]])
        tick = np.array([0.0, 0.008, 0.0])
        self.lines(
            "/reference/robot_length",
            [a, a - tick, b - tick],
            [b, a + tick, b + tick],
            (0.94, 0.48, 0.20),
            hidden=not self.show_scale,
        )
        bounds.append(np.array([a, b]))
        all_points = np.concatenate(bounds)
        self.lo, self.hi = all_points.min(0), all_points.max(0)
        self.dirty = False

    def fit_camera(self) -> None:
        """Fit the orbit camera to the current terrain bounds and zoom."""
        from pyglet.math import Vec3

        lo, hi = self.lo * self.scale, self.hi * self.scale
        center = (lo + hi) * 0.5
        corners = np.array(
            [[lo[d] if bit[d] == 0 else hi[d] for d in range(3)] for bit in np.ndindex(2, 2, 2)]
        )
        az, el = math.radians(self.azimuth), math.radians(self.elevation)
        front = -np.array([math.cos(az) * math.cos(el), math.sin(az) * math.cos(el), math.sin(el)])
        right = np.cross(front, [0.0, 0.0, 1.0])
        right /= np.linalg.norm(right)
        up = np.cross(right, front)
        relative = corners - center
        xx, yy, zz = relative @ right, relative @ up, relative @ front
        tan = math.tan(math.radians(38.0 / 2))
        aspect = self.args.width / self.args.height
        xlimit, ylimit = (0.87, 0.64) if self.args.gallery else (0.57, 0.82)
        distance = (
            max(
                float(np.max(np.abs(xx) / (tan * aspect * xlimit) - zz)),
                float(np.max(np.abs(yy) / (tan * ylimit) - zz)),
                0.1,
            )
            * 1.08
            * self.zoom
        )
        # Leave room for Newton's left-hand editor panel in interactive mode.
        if not self.args.gallery:
            center -= right * distance * tan * aspect * 0.22
        self.viewer.set_camera(
            Vec3(*(center - front * distance)),
            math.degrees(math.asin(front[2])),
            math.degrees(math.atan2(front[1], front[0])),
        )

    def render(self) -> None:
        """Rebuild dirty geometry as needed, then draw and advance the display frame counter."""
        self.viewer.begin_frame(self.frame / 60.0)
        if self.dirty:
            try:
                self.rebuild()
                self.error = ""
            except ValueError as exc:
                self.error = str(exc)
                self.dirty = False
                if not hasattr(self, "lo"):
                    raise
        self.fit_camera()
        self.viewer.end_frame()
        self.frame += 1

    def save(self) -> None:
        """Write a fresh recipe directory and update UI status; preserve previous saves."""
        recipe = self.recipe()
        directory = unique_directory(self.args.out)
        write_recipe(directory, recipe)
        (directory / "README.txt").write_text(
            "Terrain geometry created in NVIDIA Newton Terrain Studio.\n"
            "No simulation or locomotion result is implied. Dimensions are meters.\n"
            "recipe.json preserves the exact explicit geometry for simulation.\n"
            "Use scripts/scenarios.py replay --terrain-file <recipe.json> --execution legacy.\n",
            encoding="utf-8",
        )
        self.status = f"Saved: {directory}"
        print(self.status, flush=True)

    def ui(self, ui: Any) -> None:
        """Apply ImGui parameter edits and save actions, reporting validation failures in the UI."""
        ui.text("CODESIGN / TERRAIN STUDIO")
        ui.text_wrapped(
            "Geometry preview. Create a course, save its recipe, then replay with the legacy solver."
        )
        ui.separator()
        control_width = ui.get_content_region_avail().x * _CONTROL_FIELD_WIDTH_FRACTION
        ui.set_next_item_width(control_width)
        changed, index = ui.combo(
            "Preset",
            list(PRESETS).index(self.preset),
            [name.replace("_", " ").title() for name in PRESETS],
        )
        if changed:
            self.set_preset(PRESETS[index])
        ui.set_next_item_width(control_width)
        changed, index = ui.combo("Dimension", self.dim - 2, ["2D / planar", "3D"])
        if changed:
            if self.preset == "cross_slope" and index == 0:
                self.status = "Cross slope needs 3D; choose another preset for 2D."
            else:
                self.dim = index + 2
                self.dirty = True
        if self.preset in ("uphill", "downhill", "cross_slope"):
            ui.set_next_item_width(control_width)
            changed, value = ui.slider_float(
                "Slope (degrees)", self.parameters["slope_degrees"], 0.0, 24.0, "%.1f"
            )
            if changed:
                self.parameters["slope_degrees"] = value
                self.dirty = True
        if self.preset in ("low_blocks", "rough_blocks", "stairs", "hurdles"):
            ui.set_next_item_width(control_width)
            changed, value = ui.slider_int("Obstacle count", self.parameters["count"], 1, 12)
            if changed:
                self.parameters["count"] = value
                self.dirty = True
            for key, label, low, high in (
                (
                    "height",
                    "Rise per step (mm)" if self.preset == "stairs" else "Height (mm)",
                    5.0,
                    60.0,
                ),
                ("width", "Width (mm)", 12.0, 120.0),
                ("spacing", "Center spacing (mm)", 25.0, 220.0),
                ("start", "First center X (mm)", 220.0, 450.0),
                ("rounding", "Edge radius (mm)", 0.0, min(2.0, self.parameters["height"] * 240.0)),
            ):
                ui.set_next_item_width(control_width)
                changed, value = ui.slider_float(
                    label, self.parameters[key] * 1000.0, low, high, "%.1f"
                )
                if changed:
                    self.parameters[key] = value / 1000.0
                    self.parameters["rounding"] = min(
                        self.parameters["rounding"],
                        self.parameters["height"] * 0.24,
                        self.parameters["width"] * 0.24,
                    )
                    self.dirty = True
            ui.set_next_item_width(control_width)
            changed, value = ui.slider_int("Seed", self.seed, 0, 999)
            if changed:
                self.seed, self.dirty = value, True
            if ui.button("Next seed"):
                self.seed += 1
                self.dirty = True
        ui.separator()
        for name, attr in (("5 cm grid", "show_grid"), ("16 cm robot-length ruler", "show_scale")):
            changed, value = ui.checkbox(name, getattr(self, attr))
            if changed:
                setattr(self, attr, value)
                self.dirty = True
        if ui.button("Reset view"):
            self.azimuth, self.elevation, self.zoom = -62.0, 29.0, 1.0
        ui.text_wrapped(
            "Drag to orbit. Wheel to zoom. Orange ruler = robot length, not a simulated robot."
        )
        ui.text_wrapped("2D obstacle depth is display-only; 3D uses the actual box depth.")
        ui.separator()
        if ui.button("Save recipe + terrain"):
            try:
                self.save()
                self.error = ""
            except (ValueError, OSError) as exc:
                self.error = str(exc)
        ui.text_wrapped(self.error or self.status)


def title_card(image: Image.Image, editor: TerrainEditor) -> Image.Image:
    """Return an annotated copy of the supplied Pillow render."""
    from PIL import ImageDraw, ImageFont

    draw = ImageDraw.Draw(image)
    scale = image.width / 1280

    def font(size: float, bold: bool = False) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
        path = Path("C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf")
        return (
            ImageFont.truetype(str(path), max(11, round(size * scale)))
            if path.exists()
            else ImageFont.load_default()
        )

    ink, muted = (30, 48, 62), (80, 104, 118)
    top, bottom = round(105 * scale), image.height - round(88 * scale)
    draw.rectangle((0, 0, image.width, top), fill=(241, 245, 246))
    draw.rectangle((0, bottom, image.width, image.height), fill=(241, 245, 246))
    draw.text(
        (36 * scale, 16 * scale), "CODESIGN  /  TERRAIN STUDIO", fill=muted, font=font(16, True)
    )
    draw.text((35 * scale, 43 * scale), TITLES[editor.preset], fill=ink, font=font(29, True))
    draw.text(
        (36 * scale, bottom + 14 * scale),
        f"{editor.dim}D  |  5 cm grid  |  orange ruler: 16 cm robot length  |  seed {editor.seed}",
        fill=ink,
        font=font(18),
    )
    draw.text(
        (36 * scale, bottom + 46 * scale),
        "NVIDIA Newton geometry preview  /  Save a recipe, then simulate in legacy mode",
        fill=muted,
        font=font(15),
    )
    draw.rectangle((0, image.height - 4 * scale, image.width, image.height), fill=(26, 131, 132))
    return image


def gallery(editor: TerrainEditor, directory: Path) -> None:
    """Render preset screenshots and a portable HTML gallery with source provenance."""
    from PIL import Image, ImageOps

    directory = Path(directory)
    if directory.exists():
        raise FileExistsError(f"Refusing to overwrite gallery: {directory}")
    directory.mkdir(parents=True)
    started = time.perf_counter()
    images, cards = [], []
    original_dim = editor.args.dim
    for name in PRESETS:
        editor.dim = 3 if name == "cross_slope" else original_dim
        editor.set_preset(name)
        editor.render()
        editor.render()
        image = title_card(Image.fromarray(editor.viewer.get_frame().numpy(), "RGB"), editor)
        image.save(directory / f"{name}.png")
        recipe_dir = directory / name
        recipe_dir.mkdir()
        write_recipe(recipe_dir, editor.recipe())
        images.append(image)
        cards.append(
            f'<article><a href="{name}.png"><img src="{name}.png" alt="{html.escape(TITLES[name])}"></a>'
            f"<h2>{html.escape(name.replace('_', ' ').title())}</h2>"
            f'<a href="{name}/recipe.json">Recipe JSON</a> &middot; '
            f'<a href="{name}/terrain.json">Terrain JSON</a></article>'
        )
        print(f"Saved {name}.png and replayable recipe ({editor.dim}D)", flush=True)
    sheet = Image.new("RGB", (1280, 1600), (235, 241, 243))
    for index, image in enumerate(images):
        tile = ImageOps.contain(image, (640, 400), Image.Resampling.LANCZOS)
        left, top = (index % 2) * 640, (index // 2) * 400
        sheet.paste(tile, (left + (640 - tile.width) // 2, top))
    sheet.save(directory / "overview.png")
    (directory / "index.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
        "<title>Codesign Terrain Studio</title><style>body{margin:0;background:#eaf0f2;color:#233646;"
        "font:16px system-ui}header,main,footer{max-width:1400px;margin:auto;padding:28px}"
        "h1{font-size:38px;margin:0 0 12px}p{line-height:1.6;max-width:850px}main{display:grid;"
        "grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:24px;padding-top:0}"
        "article{background:#fff;border-radius:12px;overflow:hidden;padding-bottom:20px;"
        "box-shadow:0 8px 28px #243d4a12}img{width:100%;display:block}h2{margin:16px 20px 8px}"
        "article>a:not(:first-child){margin-left:20px}a{color:#127b7d}"
        "@media(max-width:500px){main{grid-template-columns:1fr}}</style>"
        "<header><h1>Terrain Studio</h1><p>Eight editable terrain families, rendered in NVIDIA Newton. "
        "These are geometry previews. The orange ruler references the 16 cm robot length; "
        "no simulated motion is shown. Save a recipe in the desktop editor and use it with legacy replay.</p>"
        "<p>Drag in the desktop viewer to orbit, and scroll to zoom. Sizes are in meters in JSON and "
        "millimeters in the editor. Cross slope requires 3D.</p></header><main>"
        + "".join(cards)
        + "</main><footer>Codesign / exact analytical plane and rounded-box geometry</footer></html>",
        encoding="utf-8",
    )
    provenance = {
        "renderer": "NVIDIA Newton ViewerGL",
        "physics_rerun": False,
        "purpose": "Terrain geometry preview and recipe creation; not simulation evidence",
        "presets": list(PRESETS),
        "render_wall_s": time.perf_counter() - started,
        "render_device": str(editor.viewer.device),
        "source_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "units": "meters",
        "grid_spacing_m": 0.05,
        "robot_length_reference_m": 0.16,
        "planar_display_depth_is_cosmetic": True,
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("newton", "warp-lang", "numpy", "pillow")
        },
        "files_sha256": {
            str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(directory.rglob("*"))
            if path.is_file()
        },
    }
    (directory / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved gallery: {directory / 'index.html'}", flush=True)


def main() -> None:
    """Parse and validate CLI options, run the selected display task, and close graphics resources."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=PRESETS, default="stairs")
    parser.add_argument("--dim", type=int, choices=(2, 3), default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=PROJECT / "runs" / "terrain_demo" / "custom")
    parser.add_argument(
        "--gallery", type=Path, help="Render a hidden-window PNG/HTML gallery; no simulation"
    )
    parser.add_argument(
        "--save-current",
        action="store_true",
        help="Save the initial recipe into a fresh --out directory",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Hide the window for a bounded --max-frames UI smoke check",
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument(
        "--max-frames", type=int, help="Close the interactive viewer after this many frames"
    )
    args = parser.parse_args()
    if min(args.width, args.height) < 400 or (args.max_frames is not None and args.max_frames < 1):
        parser.error("Image dimensions must be >=400 and max-frames must be positive")
    if not 0 <= args.seed <= 2**32 - 1:
        parser.error("seed must be in [0, 2**32-1]")
    if args.gallery and args.gallery.exists():
        parser.error(f"Refusing to overwrite gallery: {args.gallery}")
    if args.headless and not args.gallery and args.max_frames is None:
        parser.error("--headless requires --max-frames or --gallery")
    editor = TerrainEditor(args)
    try:
        if args.save_current:
            editor.save()
        if args.gallery:
            gallery(editor, args.gallery)
        else:
            while editor.viewer.is_running():
                editor.render()
                if args.max_frames is not None and editor.frame >= args.max_frames:
                    break
    finally:
        editor.viewer.close()


if __name__ == "__main__":
    main()
