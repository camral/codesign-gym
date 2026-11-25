#!/usr/bin/env python3
"""Newton studio playback of saved trajectories, without JAX or a physics solver.

Run with the existing native Newton virtual environment. Geometry follows saved
samples exactly. Thin extrusion and reference-lattice surfaces are display skin.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from PIL import Image, ImageFont
    from warp import array as WarpArray

    from .viewer_types import MeshHost

# Preserve the established viewer imports for scripts and notebook callers.
if TYPE_CHECKING or __package__:
    from .recording import (
        COLORS as COLORS,
        DetailGeometry as DetailGeometry,
        Recording as Recording,
        Skin as Skin,
        digest as digest,
        lattice_cells as lattice_cells,
        ordered_polygon as ordered_polygon,
        read_json as read_json,
        rounded_box_surface as rounded_box_surface,
        terrain_height as terrain_height,
        world_coordinates as world_coordinates,
    )
else:
    from recording import (
        COLORS as COLORS,
        DetailGeometry as DetailGeometry,
        Recording as Recording,
        Skin as Skin,
        digest as digest,
        lattice_cells as lattice_cells,
        ordered_polygon as ordered_polygon,
        read_json as read_json,
        rounded_box_surface as rounded_box_surface,
        terrain_height as terrain_height,
        world_coordinates as world_coordinates,
    )

# Recordings carry floating coordinates and integer connectivity with saved dtypes.
HostArray: TypeAlias = "NDArray[Any]"


class Studio:
    """Own Newton graphics buffers and playback controls for a fixed saved recording."""

    def __init__(self, recording: Recording, skin: Skin, args: argparse.Namespace) -> None:
        """Allocate optional CUDA graphics resources and fit the recording bounds.

        args comes from main; the caller must close viewer after playback or export."""
        import newton
        import warp as wp
        from newton.viewer import ViewerGL

        self.wp, self.newton = wp, newton
        wp.set_device("cuda:0")
        self.viewer = ViewerGL(
            width=args.width, height=args.height, vsync=not bool(args.gif), headless=bool(args.gif)
        )
        self.viewer.renderer.set_title("Codesign | Newton Studio")
        self.viewer.renderer.draw_shadows = True
        self.viewer.renderer.draw_sky = True
        self.viewer.renderer.sky_upper = (0.76, 0.80, 0.83)
        self.viewer.renderer.sky_lower = (0.60, 0.64, 0.66)
        self.viewer.renderer.background_color = (0.84, 0.88, 0.91)
        # This installed Newton release exposes light color through its renderer
        # field rather than the newer public exposure/diffuse controls.
        self.viewer.renderer._light_color = (0.43, 0.40, 0.37)
        self.viewer.show_ui = not bool(args.gif)
        self.rec, self.skin, self.args = recording, skin, args
        self.detail = DetailGeometry(recording, getattr(args, "marker_radius", None))
        self.show_surface = not getattr(args, "no_surface", False)
        self.show_detail = bool(
            getattr(args, "particles", False) or getattr(args, "elements", False)
        )
        self.show_nodes = True
        self.detail_has_been_enabled = self.show_detail
        self.scale = args.display_scale
        self.fixed = True
        self.speed = 1.0
        self.loop = True
        self.index = 0
        self.clock = float(recording.times[0])
        self.last_wall = time.perf_counter()
        self.buffers: dict[str, WarpArray] = {}
        self.indices: dict[str, WarpArray] = {}
        self.identity = wp.array(
            [wp.transform_identity()], dtype=wp.transform, device=self.viewer.device
        )
        self.unit_scale = wp.array([[1.0, 1.0, 1.0]], dtype=wp.vec3, device=self.viewer.device)
        self.material = wp.array([[0.62, 0.05, 0.0, 0.0]], dtype=wp.vec4, device=self.viewer.device)
        self.colors = {
            g: wp.array([c], dtype=wp.vec3, device=self.viewer.device) for g, c in enumerate(COLORS)
        }
        # Fit the full recorded bounds in camera coordinates. Width/height alone
        # miss the perspective contribution of 3D depth and oblique viewpoints.
        # Include cosmetic skin thickness and leave the export HUD unobstructed.
        padding = np.full(3, max(0.0015 if skin.beam else 0.0, self.detail.radius))
        if recording.dim == 2:
            padding[1] += skin.thickness / 2
        lo = (recording.lo - padding) * self.scale
        hi = (recording.hi + padding) * self.scale
        center = (lo + hi) * 0.5
        corners = np.array(
            [[lo[d] if bit[d] == 0 else hi[d] for d in range(3)] for bit in np.ndindex(2, 2, 2)]
        )
        az = math.radians(args.azimuth)
        el = math.radians(float(np.clip(args.elevation, -89.0, 89.0)))
        front = -np.array([math.cos(az) * math.cos(el), math.sin(az) * math.cos(el), math.sin(el)])
        right = np.cross(front, [0.0, 0.0, 1.0])
        right /= np.linalg.norm(right)
        up = np.cross(right, front)
        relative = corners - center
        camera_x, camera_y, camera_z = relative @ right, relative @ up, relative @ front
        tan_half_fov = math.tan(math.radians(38 / 2))
        aspect = args.width / args.height
        overlay_scale = args.width / 1280
        top_margin = (94 + 30) * overlay_scale
        bottom_margin = (82 + 30) * overlay_scale
        horizontal_margin = 40 * overlay_scale
        x_limit = 1 - 2 * horizontal_margin / args.width
        y_limit = min(1 - 2 * top_margin / args.height, 1 - 2 * bottom_margin / args.height)
        if y_limit <= 0.05:
            raise ValueError("Image is too short for the title/footer; increase --height")
        distance = (
            max(
                float(np.max(np.abs(camera_x) / (tan_half_fov * aspect * x_limit) - camera_z)),
                float(np.max(np.abs(camera_y) / (tan_half_fov * y_limit) - camera_z)),
                float(0.01 - camera_z.min()),
                0.05,
            )
            * 1.04
        )
        self.camera_position = center - front * distance
        self.camera_pitch = math.degrees(math.asin(front[2]))
        self.camera_yaw = math.degrees(math.atan2(front[1], front[0]))
        span = float(np.linalg.norm(hi - lo))
        depth = distance + camera_z
        pixel_x = (camera_x / (depth * tan_half_fov * aspect) + 1) * args.width / 2
        pixel_y = (1 - camera_y / (depth * tan_half_fov)) * args.height / 2
        self.camera_fit = {
            "method": "Perspective fit of all eight recorded-bounds corners, including display skin and cosmetic markers",
            "pixel_bounds": [
                float(pixel_x.min()),
                float(pixel_y.min()),
                float(pixel_x.max()),
                float(pixel_y.max()),
            ],
            "usable_pixel_bounds": [
                horizontal_margin,
                top_margin,
                args.width - horizontal_margin,
                args.height - bottom_margin,
            ],
            "world_bounds_display_units": [lo.tolist(), hi.tolist()],
        }
        self.viewer.camera.fov = 38.0
        self.viewer.camera.near = 0.005
        self.viewer.camera.far = 250.0
        self.viewer._cam_speed = max(0.05, span * 0.08)
        self.reset_camera()
        self.add_ground()
        if not args.gif:
            self.viewer.register_ui_callback(self.ui, position="side")

    def reset_camera(self) -> None:
        """Restore the fitted camera without altering saved simulation coordinates."""
        from pyglet.math import Vec3

        self.viewer.set_camera(Vec3(*self.camera_position), self.camera_pitch, self.camera_yaw)

    def mesh(
        self: MeshHost,
        name: str,
        positions: HostArray,
        indices: HostArray,
        color: WarpArray,
        material: WarpArray | None = None,
        hidden: bool = False,
    ) -> None:
        """Update owned device vertices and log a mesh with fixed topology.

        Positions are host metres; scale is cosmetic. Existing index buffers are reused."""
        wp = self.wp
        points = np.asarray(positions * self.scale, np.float32)
        if name not in self.buffers:
            self.buffers[name] = wp.array(points, dtype=wp.vec3, device=self.viewer.device)
            self.indices[name] = wp.array(
                np.asarray(indices, np.int32), dtype=wp.int32, device=self.viewer.device
            )
        else:
            self.buffers[name].assign(points)
        self.viewer.log_mesh(
            name, self.buffers[name], self.indices[name], hidden=True, backface_culling=False
        )
        self.viewer.log_instances(
            name + "/surface",
            name,
            self.identity,
            self.unit_scale,
            color,
            self.material if material is None else material,
            hidden=hidden,
        )

    def set_detail_visible(self, visible: bool) -> None:
        """Set detail visibility; first enabling detail hides the cosmetic surface."""
        if visible and not self.detail_has_been_enabled:
            self.show_surface = False
            self.detail_has_been_enabled = True
        self.show_detail = bool(visible)

    def detail_geometry(self, frame: int) -> None:
        """Update recorded site buffers even while hidden, so scrubbing is exact."""
        wp, detail = self.wp, self.detail
        if not detail.available:
            return

        def buffer(name: str, values: HostArray, dtype: Any) -> WarpArray:
            values = np.asarray(values, np.float32)
            if name not in self.buffers:
                self.buffers[name] = wp.array(values, dtype=dtype, device=self.viewer.device)
            else:
                self.buffers[name].assign(values)
            return self.buffers[name]

        if self.show_detail or "/detail/nodes" in self.viewer.objects:
            points = buffer(
                "detail_points", self.rec.world[frame, detail.node_ids] * self.scale, wp.vec3
            )
            radius = detail.radius * (1.0 if detail.particles else 0.45)
            radii = buffer(
                "detail_radii", np.full(len(detail.node_ids), radius * self.scale), wp.float32
            )
            colors = buffer("detail_node_colors", detail.node_colors, wp.vec3)
            self.viewer.log_points(
                "/detail/nodes",
                points,
                radii,
                colors,
                hidden=not (self.show_detail and (detail.particles or self.show_nodes)),
            )
        if len(detail.edges) and (self.show_detail or "/detail/edges" in self.viewer.lines):
            starts = buffer(
                "detail_starts", self.rec.world[frame, detail.edges[:, 0]] * self.scale, wp.vec3
            )
            ends = buffer(
                "detail_ends", self.rec.world[frame, detail.edges[:, 1]] * self.scale, wp.vec3
            )
            colors = buffer("detail_edge_colors", detail.edge_colors, wp.vec3)
            self.viewer.log_lines(
                "/detail/edges", starts, ends, colors, hidden=not self.show_detail
            )
            # This Newton release only applies hidden at line creation.
            self.viewer.lines["/detail/edges"].hidden = not self.show_detail

    def visibility(self) -> dict[str, bool]:
        """Return stable export/UI flags for surfaces, particles, elements and nodes."""
        return {
            "solid_surface": self.show_surface,
            "particles": self.show_detail and self.detail.particles,
            "elements": self.show_detail and not self.detail.particles,
            "element_nodes": self.show_detail and not self.detail.particles and self.show_nodes,
        }

    def add_ground(self) -> None:
        """Allocate a display stage matching saved terrain, with metre-based grid references."""
        # A finite matte stage, aligned with the saved terrain. Small grid lines
        # are physical world references and stay fixed as the robot moves.
        wp = self.wp
        xmid = (self.rec.lo[0] + self.rec.hi[0]) * 0.5
        extent = max(2.0, (self.rec.hi[0] - self.rec.lo[0]) * 8.0)
        xs = np.linspace(xmid - extent, xmid + extent, 81)
        zs = np.linspace(-extent, extent, 41)
        x: HostArray
        z: HostArray
        x, z = np.meshgrid(xs, zs, indexing="ij")
        plane = dict(self.rec.terrain, active=[]) if self.rec.terrain else None
        y = terrain_height(x, z, plane) - 0.00015
        p = world_coordinates(np.stack((x, y, z), -1).reshape(-1, 3))
        triangles: list[int] = []
        for i in range(len(xs) - 1):
            for j in range(len(zs) - 1):
                a = i * len(zs) + j
                b = a + len(zs)
                triangles.extend((a, b + 1, b, a, a + 1, b + 1))
        color = wp.array([[0.56, 0.57, 0.55]], dtype=wp.vec3, device=self.viewer.device)
        matte = wp.array([[0.96, 0.0, 0.0, 0.0]], dtype=wp.vec4, device=self.viewer.device)
        self.mesh("/stage", p, triangles, color, matte)
        if self.rec.terrain:
            t = self.rec.terrain
            for index, (c, b, r, active) in enumerate(
                zip(
                    t.get("centers", []),
                    t.get("half_sizes", []),
                    t.get("rounding", []),
                    t.get("active", []),
                )
            ):
                if active:
                    vertices, faces = rounded_box_surface(c, b, r, extent)
                    self.mesh(f"/terrain/block_{index}", vertices, faces, color, matte)
        # Fixed 5 cm metric grid; buried portions are occluded by terrain meshes.
        starts, ends = [], []
        xlo, xhi = self.rec.lo[0] - 0.25, self.rec.hi[0] + 0.25
        zlo, zhi = -0.40, 0.40
        for xx in np.arange(math.floor(xlo / 0.05) * 0.05, xhi, 0.05):
            yy = terrain_height(np.full(2, xx), np.array([zlo, zhi]), plane) + 0.0001
            starts.append([xx, -zlo, yy[0]])
            ends.append([xx, -zhi, yy[1]])
        for zz in np.arange(zlo, zhi + 0.025, 0.05):
            yy = terrain_height(np.array([xlo, xhi]), np.full(2, zz), plane) + 0.0001
            starts.append([xlo, -zz, yy[0]])
            ends.append([xhi, -zz, yy[1]])
        if starts:
            self.viewer.log_lines(
                "/ruler",
                wp.array(np.asarray(starts) * self.scale, dtype=wp.vec3, device=self.viewer.device),
                wp.array(np.asarray(ends) * self.scale, dtype=wp.vec3, device=self.viewer.device),
                (0.43, 0.47, 0.48),
            )

    def render(self, frame: int) -> None:
        """Draw exactly one saved frame and synchronize graphics buffers for capture."""
        self.index = int(frame)
        if self.fixed:
            self.reset_camera()
        self.viewer.begin_frame(float(self.rec.times[self.index]))
        for group, (points, indices) in self.skin.geometry(self.index).items():
            self.mesh(
                f"/robot/region_{group}",
                points,
                indices,
                self.colors[group],
                hidden=not self.show_surface,
            )
        self.detail_geometry(self.index)
        self.viewer.end_frame()

    def ui(self, ui: Any) -> None:
        """Apply optional ImGui controls to playback and visibility; leave recording data intact."""
        ui.separator()
        ui.text("CODESIGN / SAVED MOTION")
        ui.text(f"{self.rec.driver.upper()}   |   {len(self.rec.times)} physical samples")
        ui.text_wrapped("Thin display skin follows saved positions. No physics is rerun.")
        _, self.show_surface = ui.checkbox("Solid surface", self.show_surface)
        if self.detail.available:
            changed, visible = ui.checkbox(self.detail.label, self.show_detail)
            if changed:
                self.set_detail_visible(visible)
            if self.show_detail:
                if not self.detail.particles:
                    _, self.show_nodes = ui.checkbox("Element nodes", self.show_nodes)
                ui.text_wrapped(self.detail.description)
                ui.text_wrapped(
                    "Marker size is cosmetic. Toggle Solid surface to expose or cover detail."
                )
        else:
            ui.text_wrapped("No recorded particle type or element connectivity is available.")
        if ui.button("Play" if self.viewer.is_paused() else "Pause"):
            self.viewer._paused = not self.viewer.is_paused()
        ui.same_line()
        if ui.button("Restart"):
            self.clock = float(self.rec.times[0])
            self.index = 0
        changed, frame = ui.slider_int("Frame", self.index, 0, len(self.rec.times) - 1)
        if changed:
            self.index = frame
            self.clock = float(self.rec.times[frame])
            self.viewer._paused = True
        _, self.speed = ui.slider_float("Speed", self.speed, 0.1, 3.0)
        _, self.loop = ui.checkbox("Loop", self.loop)
        _, self.fixed = ui.checkbox("Fixed camera", self.fixed)
        if ui.button("Reset view"):
            self.reset_camera()
        ui.text(f"Physical time: {self.rec.times[self.index]:.3f} s")
        ui.text(f"Displacement: {1000 * self.rec.displacements_x[self.index]:+.1f} mm")
        ui.text_wrapped(self.rec.verdict())

    def interactive(self) -> None:
        """Play saved samples until the window closes or the configured frame limit is reached."""
        rendered = 0
        while self.viewer.is_running():
            now = time.perf_counter()
            elapsed = min(now - self.last_wall, 0.1)
            self.last_wall = now
            if not self.viewer.is_paused():
                self.clock += elapsed * self.speed
                if self.clock > self.rec.times[-1]:
                    if self.loop:
                        self.clock = float(self.rec.times[0])
                    else:
                        self.clock = float(self.rec.times[-1])
                        self.viewer._paused = True
                self.index = int(np.searchsorted(self.rec.times, self.clock, side="right") - 1)
            self.render(max(0, self.index))
            rendered += 1
            if self.args.max_frames is not None and rendered >= self.args.max_frames:
                break


def overlay(
    image: Image.Image,
    recording: Recording,
    frame: int,
    args: argparse.Namespace,
    visibility: dict[str, bool] | None = None,
) -> Image.Image:
    """Return a copied Pillow image annotated with saved physical time and assessment."""
    from PIL import ImageDraw, ImageFont

    image = image.copy()
    draw = ImageDraw.Draw(image)
    scale = args.width / 1280

    def font(size: float, bold: bool = False) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
        for path in [
            Path("C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        ]:
            if path.exists():
                return ImageFont.truetype(str(path), max(11, round(size * scale)))
        return ImageFont.load_default()

    ink = (33, 49, 64)
    muted = (86, 105, 118)
    draw.rectangle((0, 0, image.width, round(94 * scale)), fill=(238, 242, 243))
    draw.rounded_rectangle(
        (round(35 * scale), round(29 * scale), round(43 * scale), round(67 * scale)),
        radius=3,
        fill=(17, 142, 143),
    )
    draw.text(
        (round(57 * scale), round(19 * scale)),
        "CODESIGN  /  MOTION STUDY",
        font=font(25, True),
        fill=ink,
    )
    draw.text(
        (round(58 * scale), round(54 * scale)),
        f"{recording.driver.upper()}   •   NVIDIA Newton   •   saved simulation states",
        font=font(17),
        fill=muted,
    )
    y = image.height - round(82 * scale)
    draw.rectangle((0, y, image.width, image.height), fill=(238, 242, 243))
    draw.text(
        (round(37 * scale), y + round(13 * scale)),
        f"{recording.times[frame]:5.2f} s",
        font=font(26, True),
        fill=ink,
    )
    draw.text(
        (round(170 * scale), y + round(20 * scale)),
        f"{1000 * recording.displacements_x[frame]:+.1f} mm displacement",
        font=font(18),
        fill=ink,
    )
    verdict = recording.verdict()
    draw.text((round(650 * scale), y + round(20 * scale)), verdict, font=font(17), fill=muted)
    modes = []
    if visibility is None or visibility["solid_surface"]:
        modes.append("DISPLAY SKIN")
    if visibility and visibility["particles"]:
        modes.append("SAVED PARTICLES / COSMETIC SIZE")
    if visibility and visibility["elements"]:
        modes.append(
            "SAVED SURFACE ELEMENTS" if "faces" in recording.arrays else "SAVED ELEMENT LINKS"
        )
    draw.text(
        (round(37 * scale), y + round(51 * scale)),
        "FIXED CAMERA  /  5 cm GRID  /  EXACT SAMPLES  /  " + " + ".join(modes),
        font=font(12),
        fill=muted,
    )
    span = recording.times[-1] - recording.times[0]
    progress = (recording.times[frame] - recording.times[0]) / span if span else 1.0
    draw.rectangle(
        (0, image.height - round(3 * scale), round(image.width * progress), image.height),
        fill=(17, 142, 143),
    )
    return image


def export(studio: Studio, args: argparse.Namespace) -> None:
    """Write GIF, poster and provenance from saved samples using the caller-owned viewer.

    Rendering and file errors propagate; the caller closes graphics resources."""
    from PIL import Image

    rec = studio.rec
    indices = rec.frame_indices(args.fps, args.max_frames)
    images = []
    started = time.perf_counter()
    for number, frame in enumerate(indices):
        studio.render(frame)
        rgb = studio.viewer.get_frame().numpy()
        images.append(overlay(Image.fromarray(rgb, "RGB"), rec, frame, args, studio.visibility()))
        if number % 20 == 0 or number == len(indices) - 1:
            print(f"Rendered {number + 1}/{len(indices)} saved frames", flush=True)
    durations = [max(10, round(float(dt) * 100) * 10) for dt in np.diff(rec.times[indices])]
    final_hold_ms = max(10, round(100 / args.fps) * 10)
    durations.append(final_hold_ms)
    images[0].save(
        args.gif,
        format="GIF",
        save_all=True,
        append_images=images[1:],
        duration=durations,
        loop=0,
        disposal=2,
        optimize=False,
    )
    poster_index = min(len(images) - 1, len(images) // 4)
    images[poster_index].save(args.gif.with_suffix(".png"))
    inputs = [
        p
        for p in [
            rec.path,
            rec.directory / "topology.npz",
            rec.directory / "terrain.json",
            rec.directory / "case.json",
            rec.directory / "protocol.json",
            rec.directory / "assessment.json",
            rec.directory / "flight_assessment.json",
            rec.directory / "objective.json",
        ]
        if p.exists()
    ]
    report = {
        "renderer": "NVIDIA Newton ViewerGL",
        "source_run": str(rec.directory),
        "source_sha256": {p.name: digest(p) for p in inputs},
        "viewer_script_sha256": digest(Path(__file__)),
        "packages": {
            p: importlib.metadata.version(p)
            for p in ("newton", "warp-lang", "pyglet", "imgui-bundle", "numpy", "pillow")
        },
        "source_indices": indices.tolist(),
        "physical_frame_times_s": rec.times[indices].tolist(),
        "gif_durations_ms": durations,
        "poster_source_index": int(indices[poster_index]),
        "intentional_endpoint_hold_ms": final_hold_ms,
        "source_frame_count": rec.source_count,
        "valid_prefix_frames": len(rec.times),
        "excluded_tail_reason": rec.excluded_reason,
        "time_interpolation": False,
        "geometry": studio.skin.description,
        "visual_extrusion_m": studio.skin.thickness,
        "visibility": studio.visibility(),
        "detail_geometry": studio.detail.provenance(),
        "detail_first_enable_behavior": "First enabling detail in the UI hides the surface; surface can then be independently re-enabled",
        "region_colors": "Anatomical torso/leg identities; not actuator or stress values",
        "terrain_source": "saved terrain.json"
        if rec.terrain is not None
        else "flat display stage; no terrain artifact",
        "render_device": str(studio.viewer.device),
        "display_scale": studio.scale,
        "display_axis_mapping": "(x,y,z) -> (x,-z,y); 2D -> (x,extrusion,y)",
        "fixed_camera": True,
        "camera_position_display_units": studio.camera_position.tolist(),
        "camera_pitch_degrees": studio.camera_pitch,
        "camera_yaw_degrees": studio.camera_yaw,
        "camera_fit": studio.camera_fit,
        "width": args.width,
        "height": args.height,
        "requested_fps": args.fps,
        "render_wall_s": time.perf_counter() - started,
        "physics_rerun": False,
        "assessment": rec.assessment,
        "flight_assessment": rec.flight_assessment,
        "objective": rec.objective,
    }
    args.gif.with_suffix(".provenance.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved {args.gif}", flush=True)


def main() -> None:
    """Parse and validate CLI options, run the selected display task, and close graphics resources."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument(
        "--gif",
        type=Path,
        help="Render a hidden-window GIF, poster PNG and provenance JSON instead of opening the viewer",
    )
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument(
        "--max-frames",
        "--frames",
        type=int,
        help="Limit export samples across the full duration, or stop interactive playback after N draws",
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument(
        "--thickness",
        type=float,
        default=0.009,
        help="Cosmetic thickness for planar display skin, in meters",
    )
    parser.add_argument(
        "--particles",
        action="store_true",
        help="Show recorded active MPM particles; use --no-surface for an unobstructed cloud",
    )
    parser.add_argument(
        "--elements",
        action="store_true",
        help="Show recorded spring/beam links or FEM boundary-face perimeters and nodes",
    )
    parser.add_argument("--no-surface", action="store_true", help="Hide the cosmetic solid surface")
    parser.add_argument(
        "--marker-radius",
        type=float,
        help="Cosmetic particle radius in meters (default 18%% of minimum reference spacing); element nodes use 45%% of this radius",
    )
    parser.add_argument(
        "--display-scale",
        type=float,
        default=10.0,
        help="Uniform graphics-only scale; metric labels stay in physical units",
    )
    parser.add_argument("--azimuth", type=float, default=-62.0)
    parser.add_argument("--elevation", type=float, default=22.0)
    args = parser.parse_args()
    if (
        not all(
            math.isfinite(v)
            for v in (args.fps, args.thickness, args.display_scale, args.azimuth, args.elevation)
        )
        or args.fps <= 0
        or args.thickness <= 0
        or args.display_scale <= 0
        or min(args.width, args.height) < 200
        or (args.max_frames is not None and args.max_frames < 1)
    ):
        parser.error(
            "FPS, thickness and scale must be positive; image dimensions >=200; max-frames >=1"
        )
    if args.marker_radius is not None and (
        not math.isfinite(args.marker_radius) or args.marker_radius <= 0
    ):
        parser.error("Marker radius must be finite and positive")
    if args.particles and args.elements:
        parser.error("Choose --particles for MPM or --elements for the other recorded models")
    if args.gif:
        args.gif = args.gif.resolve()
        if args.gif.suffix.lower() != ".gif":
            parser.error("--gif output must have a .gif extension")
        for path in (
            args.gif,
            args.gif.with_suffix(".png"),
            args.gif.with_suffix(".provenance.json"),
        ):
            if path.exists():
                parser.error(f"Refusing to overwrite existing output: {path}")
        args.gif.parent.mkdir(parents=True, exist_ok=True)
    rec = Recording(args.run)
    detail = DetailGeometry(rec, args.marker_radius)
    if args.particles and not detail.particles:
        parser.error("--particles requires a recording explicitly identified as MPM")
    if args.elements and (detail.particles or not detail.available):
        parser.error("--elements requires saved spring/beam links or FEM boundary faces")
    skin = Skin(rec, args.thickness)
    studio = Studio(rec, skin, args)
    try:
        export(studio, args) if args.gif else studio.interactive()
    finally:
        studio.viewer.close()


if __name__ == "__main__":
    main()
