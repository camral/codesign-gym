"""Actual-state Pillow animation with a fixed world camera and a metric ruler.

This renderer never integrates, smooths, interpolates or invents robot motion.
It consumes saved physical positions. Frame selection uses recorded physical
clock times, and the camera is fixed over the entire movie. No Matplotlib,
GPU graphics context, ffmpeg, or GUI is required.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from numpy.typing import ArrayLike

from envs.soft_walker.engine.types import HostArray, PathLike, PhysicsModel

if TYPE_CHECKING:
    from envs.soft_walker.engine.drivers.spring_environment import Spring2DEnv
    from envs.soft_walker.engine.physics import beam, fem, spring
    from envs.soft_walker.engine.physics.terrain import Terrain

import math
from pathlib import Path

import numpy as np

PALETTE = [(48, 132, 169), (233, 145, 70), (109, 166, 103), (172, 114, 167)]


def topology_record(env: Spring2DEnv, model: PhysicsModel) -> dict[str, HostArray]:
    "Record fixed active sites and exposed connectivity for trajectory rendering."
    from envs.soft_walker.engine.drivers.model_access import (
        task_reference_positions as reference_positions,
    )
    from envs.soft_walker.shared import region_tags

    reference = np.asarray(reference_positions(env, model))
    active = np.asarray(model.active)
    torso, feet = region_tags(env, model)
    topology = {
        "reference": reference,
        "active": active,
        "torso": np.asarray(torso),
        "feet": np.asarray(feet),
    }
    if not hasattr(env, "kind"):
        model = cast("spring.Model", model)
        topology["cells"] = np.asarray(model.cells)[np.asarray(model.occupancy) > 0]
        topology["edges"] = np.asarray(model.edges)[np.asarray(model.stiffness) > 0]
    elif env.kind == "beam":
        model = cast("beam.Model", model)
        edges = np.asarray(model.topology.edges)
        topology["edges"] = edges[active[edges].all(axis=1)]
    elif env.kind == "fem":
        model = cast("fem.Model", model)
        faces = np.asarray(model.mesh.faces)
        neighbors = np.asarray(model.mesh.neighbors)
        has_neighbor = np.asarray(model.mesh.has_neighbor)
        alpha = np.asarray(model.alpha)
        expose = (alpha[:, None] > 0) & (~has_neighbor | (alpha[neighbors] <= 0))
        topology["faces"] = faces[expose][:, [0, 1, 3, 2]]
    return topology


def render_animation(
    positions: ArrayLike,
    metrics: ArrayLike,
    topology: dict[str, HostArray],
    output: PathLike,
    *,
    title: str,
    period: float,
    fps: int = 25,
    pixels: tuple[int, int] = (840, 500),
    verdict: str = "candidate; not gait-certified",
    terrain: Terrain | None = None,
) -> dict[str, object]:
    """Write GIF + frame provenance; return a small render report.

    Invalid/repeated-time tails after a failed interval are removed. The failure
    is explicitly annotated. Frame indices always refer to original samples.
    The initial t=0 state should be supplied by the caller as sample zero.
    """
    from PIL import Image, ImageDraw

    from envs.soft_walker.shared import write_json

    positions = np.asarray(positions)
    rows = np.asarray(metrics)
    if positions.ndim != 3 or len(positions) != len(rows) or len(rows) < 1:
        raise ValueError("Positions and metric rows must have matching time axes")
    if fps <= 0:
        raise ValueError("fps must be positive")
    if not np.isfinite(positions).all():
        raise ValueError("Nonfinite positions cannot be rendered as a valid trajectory")
    times = rows[:, 0]
    if not np.isfinite(times).all() or np.any(np.diff(times) < -1e-9):
        raise ValueError("Physical sample times must be finite and monotonic")
    end = next((i for i in range(1, len(rows)) if rows[i, 6] != 0), len(rows))
    failed = end < len(rows)
    positions, rows, times = positions[:end], rows[:end], times[:end]
    active = np.asarray(topology["active"], bool)
    if not active.any():
        raise ValueError("No active physical sites")
    targets = np.arange(times[0], times[-1] + 0.5 / fps, 1.0 / fps)
    indices = np.searchsorted(times, targets, side="left").clip(0, len(times) - 1)
    indices = np.unique(np.r_[0, indices, len(times) - 1])
    dim = positions.shape[-1]
    xyz = positions if dim == 3 else np.pad(positions, ((0, 0), (0, 0), (0, 1)))
    basis = np.array([[1.0, 0.0, -0.38], [0.0, 1.0, 0.23]])
    projected = xyz @ basis.T
    all_xy = projected[:, active].reshape(-1, 2)
    lo = all_xy.min(axis=0)
    hi = all_xy.max(axis=0)
    lo[1] = min(lo[1], -0.025)
    width, height = pixels
    scale = min((width - 120) / max(hi[0] - lo[0], 0.15), (height - 175) / max(hi[1] - lo[1], 0.15))
    camera = (lo + hi) / 2

    def project(q: ArrayLike) -> tuple[float, float]:
        "Project one recorded world point using the fixed metric camera."
        q = np.asarray(q)
        if q.shape[-1] == 2:
            q = np.r_[q, 0.0]
        uv = (basis @ q - camera) * scale
        return (width / 2 + float(uv[0]), height * 0.53 - float(uv[1]))

    torso = np.asarray(topology["torso"])
    ref = np.asarray(topology["reference"])
    xmid = (ref[:, 0].min() + ref[:, 0].max()) / 2
    zmid = (ref[:, -1].min() + ref[:, -1].max()) / 2 if dim == 3 else 0
    leg = (ref[:, 0] > xmid).astype(int)
    if dim == 3:
        leg = 2 * leg + (ref[:, 2] > zmid)
    colors = [(67, 82, 110) if torso[k] else PALETTE[leg[k]] for k in range(len(ref))]
    if "colors" in topology:
        supplied = np.asarray(topology["colors"])
        if supplied.shape != (len(ref), 3) or not np.isfinite(supplied).all():
            raise ValueError("Topology colors must be one finite RGB triple per site")
        colors = [tuple(int(c) for c in row) for row in np.clip(supplied, 0, 255)]
    frames = []
    bounds_x = (float(xyz[..., 0].min()) - 0.1, float(xyz[..., 0].max()) + 0.1)
    for k in indices:
        image = Image.new("RGB", pixels, (248, 249, 251))
        draw = ImageDraw.Draw(image)
        # Static world ruler and ground. The robot never moves the camera.
        a, b = project([bounds_x[0], 0, 0]), project([bounds_x[1], 0, 0])
        if terrain is None:
            draw.polygon([a, b, (width, height), (0, height)], fill=(226, 232, 235))
            draw.line([a, b], fill=(99, 117, 130), width=2)
        else:
            from envs.soft_walker.viewer.terrain_geometry import height_at

            xx = np.linspace(bounds_x[0], bounds_x[1], 240)
            yy = height_at(xx, np.zeros_like(xx), terrain)
            profile = [project([x, y, 0.0]) for x, y in zip(xx, yy)]
            draw.polygon(profile + [(width, height), (0, height)], fill=(226, 232, 235))
            draw.line(profile, fill=(99, 117, 130), width=2)
            if dim == 3:
                for z in (-0.06, 0.06):
                    zz = np.full_like(xx, z)
                    draw.line(
                        [project([x, y, z]) for x, y in zip(xx, height_at(xx, zz, terrain))],
                        fill=(150, 166, 176),
                        width=1,
                    )
        for ruler_x in np.arange(math.floor(bounds_x[0] / 0.025) * 0.025, bounds_x[1], 0.025):
            floor_y = 0.0 if terrain is None else float(height_at(ruler_x, 0.0, terrain))
            px, py = project([ruler_x, floor_y, 0])
            draw.line([(px, py), (px, py + 6)], fill=(84, 103, 115))
            draw.text((px - 12, py + 8), f"{100 * ruler_x:.1f}", fill=(84, 103, 115))
        points = xyz[k]
        if "cells" in topology:
            for cell in topology["cells"]:
                color = colors[int(cell[0])]
                draw.polygon([project(points[n]) for n in cell], fill=color, outline=(32, 54, 67))
        elif "faces" in topology:
            faces = topology["faces"]
            for face in sorted(faces, key=lambda f: np.mean(points[f, 2])):
                draw.polygon(
                    [project(points[n]) for n in face],
                    fill=colors[int(face[0])],
                    outline=(40, 67, 78),
                )
        elif "edges" in topology:
            for i, j in topology["edges"]:
                draw.line([project(points[i]), project(points[j])], fill=colors[int(i)], width=5)
        if "cells" not in topology and "faces" not in topology:
            order = np.argsort(points[:, 2])
            radius = max(2, min(5, scale * 0.0035))
            for n in order:
                if not active[n]:
                    continue
                px, py = project(points[n])
                draw.ellipse(
                    (px - radius, py - radius, px + radius, py + radius),
                    fill=colors[n],
                    outline=(50, 70, 84),
                )
        label = "NUMERICAL FAILURE — last valid samples" if failed else verdict
        draw.rectangle((0, 0, width, 62), fill=(248, 249, 251))
        draw.text((18, 12), title, fill=(20, 42, 59))
        draw.text((18, 34), label, fill=(168, 48, 39) if failed else (61, 80, 95))
        draw.rectangle((0, height - 56, width, height), fill=(248, 249, 251))
        draw.text(
            (18, height - 44),
            f"t={times[k]:.3f} s | cycle={times[k] / period:.2f} | "
            f"dx={1000 * rows[k, 1]:+.2f} mm | W+={rows[k, 4]:.5f} J",
            fill=(20, 42, 59),
        )
        draw.text(
            (18, height - 24),
            "Actual simulation states | fixed camera | ruler in cm | +x forward",
            fill=(75, 90, 100),
        )
        frames.append(image)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    durations = [max(10, round(1000 * d / 10) * 10) for d in np.diff(times[indices])]
    durations += [durations[-1] if durations else 1000]
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        loop=0,
        duration=durations,
        optimize=False,
        disposal=2,
    )
    report = {
        "frames": len(frames),
        "source_indices": indices,
        "physical_frame_times_s": times[indices],
        "gif_durations_ms": durations,
        "fixed_camera_projected_center": camera,
        "pixels_per_meter": scale,
        "status": "failure_preview" if failed else "simulation_replay",
        "verdict": verdict,
        "no_state_interpolation": True,
    }
    write_json(output.with_suffix(".frames.json"), report)
    return report
