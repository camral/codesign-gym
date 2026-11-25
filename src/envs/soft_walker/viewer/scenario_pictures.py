"""Static scene diagrams sampled from actual decoders; no simulated-motion claim.

Pillow output is only a debug picture. A local renderer can consume saved physical
trajectories/terrain instead; no graphics package is required by the simulators.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy.typing import ArrayLike

from envs.soft_walker.engine.types import PathLike
from envs.soft_walker.presets.scenarios.config import Scenario

if TYPE_CHECKING:
    from PIL import ImageFont

    from envs.soft_walker.engine.physics.terrain import Terrain

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from envs.soft_walker.presets.scenarios.build import decoder_and_embodiment
from envs.soft_walker.presets.terrain.scenario_courses import case_terrain, resolved_terrain
from envs.soft_walker.viewer.terrain_geometry import height_at

COLORS = ((43, 137, 165), (218, 132, 67), (104, 160, 107), (158, 109, 172))

# Inherited diagnostic style, not physics or material configuration. The inclusive
# occupancy threshold and spawn translation are returned in picture provenance.
DISPLAY_OCCUPANCY_THRESHOLD = 0.5
SPAWN_CLEARANCE_M = 0.001
_CAMERA_SCALE_PX_PER_M = {2: 2500.0, 3: 2000.0}
_CAMERA_ORIGIN_FRACTIONS = (0.23, 0.77)
_CAMERA_DEPTH_PROJECTION = (-0.42, -0.23)
_TERRAIN_PROFILE_RANGE_M = (-0.035, 0.36)
_TERRAIN_PROFILE_SAMPLES = 180
_TERRAIN_DEPTH_SECTIONS_M = (-0.06, 0.06)
_QUAD_PERIMETER_SIGNS = ((-1, -1), (1, -1), (1, 1), (-1, 1))
# Three visible cube faces in perimeter order, with decorative shading factors.
_CELL_FACES = (
    (((-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1)), 0.8),
    (((1, -1, -1), (1, -1, 1), (1, 1, 1), (1, 1, -1)), 0.9),
    (((-1, 1, -1), (1, 1, -1), (1, 1, 1), (-1, 1, 1)), 1.12),
)
# The overview uses the historical seed and metric window for reproducibility.
_SHEET_SEED = 7
_SHEET_RANGE_M = (-0.05, 0.70)
_SHEET_SAMPLES = 400
_SHEET_CROSS_SECTION_ORIGIN_M = 0.325  # Midpoint of the longitudinal window.


def _terrain_section_label(name: str, terrain: Terrain) -> str:
    """Describe the realized plane angle or the exact sampled travel window."""
    if name == "cross_slope":
        angle = np.degrees(np.arctan2(-float(terrain.normal[2]), float(terrain.normal[1])))
        # Round the display angle only; normalized float32 plane normals can carry
        # tiny angle error. Geometry still uses the unrounded terrain coefficients.
        return f"{abs(round(float(angle), 2)):g} degree lateral section"
    return f"Travel x: {_SHEET_RANGE_M[0]:.2f} to {_SHEET_RANGE_M[1]:.2f} m | seed {_SHEET_SEED}"


def font(size: int = 16, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a readable preview font with the built-in Pillow fallback."""
    # Pillow's portable bundled font; no machine-dependent font file required.
    return ImageFont.load_default(size=size)


def scene_picture(
    case: Scenario,
    output: PathLike,
    *,
    pixels: tuple[int, int] = (1000, 650),
    design_mode: str = "hard",
    material_mode: str = "continuous",
) -> dict[str, object]:
    """Render occupied physical cells, before settling; return image metadata."""
    import jax.numpy as jnp

    decoder, embodiment = decoder_and_embodiment(
        case, design_mode=design_mode, material_mode=material_mode
    )
    shape = case.grid_shape
    indices = np.stack(np.meshgrid(*[np.arange(n) for n in shape], indexing="ij"), axis=-1).reshape(
        -1, case.dim
    )
    q = 2.0 * (indices + 0.5) / np.asarray(shape) - 1.0
    fields = decoder.decode(embodiment, jnp.asarray(q, case.dtype))
    occupied = np.asarray(fields.occupancy) >= DISPLAY_OCCUPANCY_THRESHOLD
    torso, leg, _ = decoder.regions(jnp.asarray(q))
    torso, leg = np.asarray(torso), np.asarray(leg)
    centers = (indices + 0.5) * case.cell_size
    if case.dim == 3:
        centers[:, 2] -= case.size[2] / 2
    terrain = case_terrain(case)
    # Display spawn translation matches the driver convention, not a settled pose.
    from envs.soft_walker.engine.drivers.environment import spawn_shift

    shift = float(spawn_shift(terrain, case.size)) + SPAWN_CLEARANCE_M
    centers[:, 1] += shift
    image = Image.new("RGB", pixels, (247, 249, 251))
    draw = ImageDraw.Draw(image)
    width, height = pixels
    is3 = case.dim == 3
    scale = _CAMERA_SCALE_PX_PER_M[case.dim]
    origin = np.array([width * _CAMERA_ORIGIN_FRACTIONS[0], height * _CAMERA_ORIGIN_FRACTIONS[1]])

    def project(point: ArrayLike) -> tuple[float, float]:
        """Project a world point with the fixed preview camera."""
        p = np.pad(point, (0, 3 - len(point)))
        return tuple(
            origin
            + scale
            * np.array(
                [
                    p[0] + _CAMERA_DEPTH_PROJECTION[0] * p[2],
                    -p[1] + _CAMERA_DEPTH_PROJECTION[1] * p[2],
                ]
            )
        )

    terrain_x = np.linspace(*_TERRAIN_PROFILE_RANGE_M, _TERRAIN_PROFILE_SAMPLES)
    terrain_y = height_at(terrain_x, np.zeros_like(terrain_x), terrain)
    profile = [project(np.array([x, y])) for x, y in zip(terrain_x, terrain_y)]
    draw.polygon(profile + [(width, height - 50), (0, height - 50)], fill=(222, 229, 235))
    draw.line(profile, fill=(93, 112, 124), width=3)
    if is3:
        for z in _TERRAIN_DEPTH_SECTIONS_M:
            hh = height_at(terrain_x, np.full_like(terrain_x, z), terrain)
            draw.line(
                [project(np.array([x, y, z])) for x, y in zip(terrain_x, hh)],
                fill=(167, 178, 188),
                width=1,
            )
    half = case.cell_size / 2
    order = np.argsort(centers[:, 2] if is3 else centers[:, 1])
    for i in order:
        if not occupied[i]:
            continue
        color = (69, 86, 110) if torso[i] else COLORS[int(leg[i])]
        c = centers[i]
        if not is3:
            points = [project(c + np.array(v) * half) for v in _QUAD_PERIMETER_SIGNS]
            draw.polygon(points, fill=color, outline=(35, 54, 67))
        else:
            for face, shade in _CELL_FACES:
                points = [project(c + np.asarray(v) * half) for v in face]
                rgb = tuple(min(255, int(channel * shade)) for channel in color)
                draw.polygon(points, fill=rgb, outline=(36, 60, 73))
    draw.rectangle((0, 0, width, 100), fill=(247, 249, 251))
    draw.text(
        (30, 20),
        f"{case.driver.upper()}  /  {case.representation}  /  {case.terrain_label}",
        font=font(26),
        fill=(19, 44, 63),
    )
    draw.text(
        (30, 59),
        "INITIAL DESIGN - sampled cells, not a simulation or a successful gait",
        font=font(17),
        fill=(83, 103, 117),
    )
    y = height - 42
    draw.text(
        (28, y),
        f"Envelope {case.size} m | h={1000 * case.cell_size:.0f} mm | {case.channels} internal channels",
        font=font(17),
        fill=(44, 67, 83),
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    return {
        "file": output.name,
        "kind": "sampled_initial_design",
        "occupied_cells": int(occupied.sum()),
        "not_a_simulation": True,
        "spawn_vertical_translation_m": shift,
        "display_occupancy_threshold": DISPLAY_OCCUPANCY_THRESHOLD,
        "design_mode": design_mode,
        "material_mode": material_mode,
    }


def terrain_sheet(output: PathLike) -> None:
    """Eight metric terrain profiles, identical to scenario SDF geometry."""
    from envs.soft_walker.presets.scenarios.config import TERRAINS

    image = Image.new("RGB", (1400, 1100), "white")
    draw = ImageDraw.Draw(image)
    for i, name in enumerate(TERRAINS):
        col, row = i % 2, i // 2
        x0, y0 = 35 + col * 700, 20 + row * 270
        draw.rounded_rectangle(
            (x0, y0, x0 + 630, y0 + 240), radius=12, fill=(244, 247, 250), outline=(210, 221, 229)
        )
        draw.text(
            (x0 + 20, y0 + 16), name.replace("_", " ").upper(), font=font(23), fill=(20, 46, 65)
        )
        t = resolved_terrain(name, dim=3, seed=_SHEET_SEED)
        xx = np.linspace(*_SHEET_RANGE_M, _SHEET_SAMPLES)
        if name == "cross_slope":
            heights = height_at(np.zeros_like(xx), xx - _SHEET_CROSS_SECTION_ORIGIN_M, t)
        else:
            heights = height_at(xx, np.zeros_like(xx), t)
        points = [(x0 + 25 + (x + 0.05) * 770, y0 + 157 - y * 1400) for x, y in zip(xx, heights)]
        draw.polygon(points + [(x0 + 602, y0 + 201), (x0 + 25, y0 + 201)], fill=(202, 220, 228))
        draw.line(points, fill=(39, 105, 131), width=3)
        label = _terrain_section_label(name, t)
        draw.text((x0 + 22, y0 + 210), label, font=font(16), fill=(69, 90, 106))
    image.save(output)
