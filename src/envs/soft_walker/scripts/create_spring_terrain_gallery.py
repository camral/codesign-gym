#!/usr/bin/env python3
"""Simulate and render small spring meshes contacting three analytical terrains.

The images show recorded implicit-solver states, never keyframed motion. This is
a passive contact demonstration, not a trained gait or a performance benchmark.
Run with JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 for the small dense reference solver.
"""

from __future__ import annotations

import argparse

# Resolve integrated imports independently of the caller's working directory.
import sys as _script_sys
from pathlib import Path, Path as _ScriptPath
from typing import TYPE_CHECKING, Any, TypedDict, cast

_source_root = str(_ScriptPath(__file__).resolve().parents[3])
if _source_root not in _script_sys.path:
    _script_sys.path.insert(0, _source_root)
_script_dir = str(_ScriptPath(__file__).resolve().parent)
if _script_dir not in _script_sys.path:
    _script_sys.path.insert(0, _script_dir)
import json
import sys
import time
from dataclasses import asdict

from envs.soft_walker.shared import (
    ENGINE_ROOT as ENGINE_ROOT,
    REPO_ROOT as REPO_ROOT,
    SCRIPTS_ROOT as SCRIPTS_ROOT,
    SOFT_ROOT as SOFT_ROOT,
    SRC_ROOT as SRC_ROOT,
    TESTS_ROOT as TESTS_ROOT,
)

ROOT = REPO_ROOT
sys.path.insert(0, str(ROOT / "src"))

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from envs.soft_walker.engine.embodiments.decoders import Fields
from envs.soft_walker.engine.physics import spring, terrain
from envs.soft_walker.engine.physics.spring_config import SpringConfig

if TYPE_CHECKING:
    from pathlib import Path

    from PIL import ImageFont

    from envs.soft_walker.engine.types import (
        Array,
    )


class GalleryCase(TypedDict):
    """One passive landing setup; SI coordinates are fixed before simulation."""

    name: str
    title: str
    subtitle: str
    tc: terrain.TerrainConfig
    velocity: tuple[float, float]
    clearance: float


CASES: tuple[GalleryCase, ...] = (
    dict(
        name="flat_landing",
        title="Flat landing",
        subtitle="Reference contact on a horizontal surface",
        tc=terrain.TerrainConfig(kind="flat", blocks=0),
        velocity=(0.18, -0.03),
        clearance=0.075,
    ),
    dict(
        name="downhill",
        title="Downhill contact",
        subtitle="The contact normal follows the tilted ground",
        tc=terrain.TerrainConfig(kind="slopes", blocks=0, height=0.025, fixed_slopes=(-0.28, 0.0)),
        velocity=(0.10, -0.03),
        clearance=0.085,
    ),
    dict(
        name="rounded_step",
        title="Rounded step",
        subtitle="A landing straddles the step edge",
        tc=terrain.TerrainConfig(
            kind="blocks",
            blocks=1,
            rounding=0.001,
            fixed_boxes=((0.175, 0.0, 0.0, 0.085, 0.032, 0.1, 0.003),),
        ),
        velocity=(0.24, -0.03),
        clearance=0.085,
    ),
)

# Serialized reports can contain backend-specific auxiliary diagnostic trees.
Report = dict[str, Any]


def pad_terrain(t: terrain.Terrain, count: int = 3) -> terrain.Terrain:
    """Same array shapes across cases lets JAX reuse a single compilation."""
    n = count - len(t.active)
    return t._replace(
        centers=jnp.pad(t.centers, ((0, n), (0, 0))),
        half_sizes=jnp.pad(t.half_sizes, ((0, n), (0, 0)), constant_values=1.0),
        rounding=jnp.pad(t.rounding, (0, n)),
        active=jnp.pad(t.active, (0, n)),
    )


def font(size: int, bold: bool = False) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    """Load an available display font, falling back to Pillow's default font."""
    paths = [
        f"/usr/share/fonts/truetype/dejavu/DejaVuSans{'-Bold' if bold else ''}.ttf",
        f"C:/Windows/Fonts/{'segoeuib' if bold else 'segoeui'}.ttf",
    ]
    for p in paths:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def render(
    case: GalleryCase,
    model: spring.Model,
    positions: Array,
    gaps: Array,
    times: Array,
    out: Path,
    step_dt: float,
    record_stride: int,
) -> Report:
    """Render the supplied saved/sample geometry and diagnostics into image or animation artifacts."""
    width, height = 840, 560
    frames = []
    xlo = min(-0.035, float(positions[..., 0].min()) - 0.025)
    xhi = max(0.265, float(positions[..., 0].max()) + 0.035)
    ylo = min(-0.032, float(positions[..., 1].min()) - 0.015)
    yhi = max(0.19, float(positions[..., 1].max()) + 0.015)
    scale = min((width - 90) / (xhi - xlo), 340 / (yhi - ylo))
    center_x = 0.5 * (xlo + xhi)

    def xy(p: Array) -> tuple[float, float]:
        return (width / 2 + (p[0] - center_x) * scale, 443 - (p[1] - ylo) * scale)

    cells, edges = np.asarray(model.cells), np.asarray(model.edges)
    reference = np.asarray(model.reference)
    areas = (
        np.sum(
            reference[cells, 0] * np.roll(reference[cells, 1], -1, axis=1)
            - reference[cells, 1] * np.roll(reference[cells, 0], -1, axis=1),
            axis=1,
        )
        / 2
    )
    polygons = positions[:, cells]
    current_areas = (
        np.sum(
            polygons[..., 0] * np.roll(polygons[..., 1], -1, axis=2)
            - polygons[..., 1] * np.roll(polygons[..., 0], -1, axis=2),
            axis=2,
        )
        / 2
    )
    deformation = np.max(abs(current_areas / areas - 1.0), axis=1)
    active_surface = np.asarray(model.contact_k) > 0
    contacts = (gaps < 0.0015) & active_surface[None]
    palette = [(52, 174, 176), (47, 156, 179), (47, 135, 181), (61, 113, 175)]
    title_font, subtitle_font, small_font = font(26, True), font(16), font(14)
    t = cast(terrain.Terrain, model.terrain)
    tn = np.asarray(t.normal)
    to = float(t.offset)
    for k, positions_k in enumerate(positions):
        im = Image.new("RGB", (width, height), "#eff4f5")
        draw = ImageDraw.Draw(im)
        draw.text((34, 23), case["title"], fill="#193641", font=title_font)
        draw.text((35, 62), case["subtitle"], fill="#526a73", font=subtitle_font)
        for gx in np.arange(np.floor(xlo / 0.05) * 0.05, xhi + 0.05, 0.05):
            draw.line((xy((gx, ylo)), xy((gx, yhi))), fill="#e0e8eb", width=1)
        for gy in np.arange(np.floor(ylo / 0.05) * 0.05, yhi + 0.05, 0.05):
            draw.line((xy((xlo, gy)), xy((xhi, gy))), fill="#e0e8eb", width=1)
        plane_y = lambda x: (to - tn[0] * x) / tn[1]
        draw.polygon(
            [
                xy((xlo - 1, plane_y(xlo - 1))),
                xy((xhi + 1, plane_y(xhi + 1))),
                xy((xhi + 1, ylo - 1)),
                xy((xlo - 1, ylo - 1)),
            ],
            fill="#8497a1",
        )
        draw.line((xy((xlo, plane_y(xlo))), xy((xhi, plane_y(xhi)))), fill="#526975", width=3)
        for c, h, radius, active in zip(t.centers, t.half_sizes, t.rounding, t.active):
            if not bool(active):
                continue
            c, h = np.asarray(c), np.asarray(h)
            p1, p2 = xy((c[0] - h[0], c[1] + h[1])), xy((c[0] + h[0], c[1] - h[1]))
            # The analytical terrain is a UNION with the ground half-space;
            # do not show box undersides that are buried below the plane.
            overlay = Image.new("RGBA", im.size, (0, 0, 0, 0))
            odraw = ImageDraw.Draw(overlay)
            odraw.rounded_rectangle(
                [p1, p2], radius=float(radius) * scale, fill="#93a6ae", outline="#526975", width=2
            )
            ground_pixels = [
                xy((xlo - 1, plane_y(xlo - 1))),
                xy((xhi + 1, plane_y(xhi + 1))),
                xy((xhi + 1, ylo - 1)),
                xy((xlo - 1, ylo - 1)),
            ]
            odraw.polygon(ground_pixels, fill=(0, 0, 0, 0))
            im.paste(overlay, (0, 0), overlay)
        for i, poly in enumerate(positions_k[cells]):
            draw.polygon([xy(p) for p in poly], fill=palette[(i // 2) % len(palette)])
        for e in edges:
            draw.line([xy(positions_k[e[0]]), xy(positions_k[e[1]])], fill="#d4eef0", width=2)
        for n, p in enumerate(positions_k):
            x, y = xy(p)
            radius = 4 if contacts[k, n] else 3
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                fill="#f3b658" if contacts[k, n] else "#205465",
            )
        draw.rounded_rectangle((28, 467, width - 28, 535), radius=13, fill="white")
        draw.text(
            (44, 477),
            f"t = {times[k]:.3f} s   |   {int(contacts[k].sum())} near-contact nodes",
            fill="#264653",
            font=small_font,
        )
        draw.text(
            (44, 504),
            "Blue shades: cell columns  /  amber: near contact  /  passive mesh",
            fill="#627781",
            font=small_font,
        )
        draw.text((width - 184, 477), "JAX / float64 / CPU", fill="#627781", font=small_font)
        draw.text((width - 184, 504), "Playback: 0.25x", fill="#627781", font=small_font)
        frames.append(im)
    # Saving only recorded states retains the true timing (10 ms per sample,
    # shown at 40 ms per frame for a 4x slow-motion presentation).
    duration_ms = max(20, round(step_dt * record_stride * 4000))
    frames[0].save(
        out / f"{case['name']}.gif",
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    thumb_index = int(np.argmax(deformation + 0.003 * contacts.sum(axis=1)))
    frames[thumb_index].save(out / f"{case['name']}.png")
    return dict(
        animation=f"{case['name']}.gif",
        thumbnail=f"{case['name']}.png",
        frames=len(frames),
        thumbnail_frame=thumb_index,
        max_cell_area_change=float(np.max(deformation)),
        maximum_near_contact_nodes=int(contacts.sum(axis=1).max()),
        playback_speed=0.25,
        frame_duration_ms=duration_ms,
    )


def main() -> None:
    """Command entry point. Simulate and render small spring meshes contacting three analytical terrains."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=ROOT / "runs/spatial_material_terrain_gallery/terrain"
    )
    parser.add_argument("--seconds", type=float, default=0.9)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--record-stride", type=int, default=10)
    parser.add_argument("--case", choices=[c["name"] for c in CASES], action="append")
    args = parser.parse_args()
    if not jax.config.x64_enabled:
        parser.error("Set JAX_ENABLE_X64=1 before launching this float64 gallery")
    if args.seconds <= 0 or args.dt <= 0 or args.record_stride < 1:
        parser.error("duration, dt and record stride must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    cfg = SpringConfig(
        grid_shape=(4, 2),
        size=(0.12, 0.06),
        thickness=0.01,
        control_dt=args.dt,
        substeps=1,
        settle_steps=0,
        initial_clearance=0.0,
        gravity=9.81,
        friction=0.12,
        damping_ratio=0.055,
        contact_modulus=120000.0,
        root_tol64=2.0e-10,
        newton_max_steps=30,
        line_search_steps=10,
    )
    topo = spring.topology(cfg, np.float64)
    n = len(topo.points)
    fields = Fields(
        jnp.ones(n),
        jnp.full(n, 12000.0),
        jnp.full(n, 850.0),
        jnp.zeros(n),
        jnp.broadcast_to(jnp.asarray([1.0, 0.0]), (n, 2)),
        jnp.zeros(n, jnp.int32),
        jnp.zeros(n, jnp.int32),
        jnp.full(n, 0.35),
    )
    base = spring.instantiate(fields, topo, cfg, channels=1)
    steps = round(args.seconds / args.dt)

    @jax.jit
    def simulate(
        model: spring.Model, state: spring.State
    ) -> tuple[spring.State, tuple[Array, Array, Array, spring.Diagnostics]]:
        def step(
            state: spring.State, _: None
        ) -> tuple[spring.State, tuple[Array, Array, Array, spring.Diagnostics]]:
            new, info = spring.substep(model, state, jnp.zeros(1), cfg)
            return new, (new.x, new.time, new.status, info)

        return jax.lax.scan(step, state, None, length=steps)

    report: Report = dict(
        schema="codesign_spring_terrain_gallery_v1",
        backend="jax",
        dtype="float64",
        device=str(jax.devices()[0]),
        integrator="backward_euler",
        linear_solver="dense",
        description="Actual passive spring-lattice trajectories; no actuation or optimization.",
        poisson_note="This central-force spring lattice has no independent continuum Poisson control.",
        material=dict(young_pa=12000.0, density_kg_m3=850.0),
        spring_config=asdict(cfg),
        cases=[],
    )
    previous_metadata = args.out / "metadata.json"
    if args.case and previous_metadata.exists():
        previous = json.loads(previous_metadata.read_text())
        report["cases"] = [c for c in previous.get("cases", []) if c["name"] not in args.case]
    for case in CASES:
        if args.case and case["name"] not in args.case:
            continue
        t = pad_terrain(terrain.generate(case["tc"], dim=2, dtype=np.float64))
        model = base._replace(
            reference=base.reference + jnp.asarray([0.0, case["clearance"]]), terrain=t
        )
        state = spring.initial_state(model)
        state = state._replace(v=jnp.broadcast_to(jnp.asarray(case["velocity"]), state.v.shape))
        print(f"Simulating {case['name']}: {steps} steps", flush=True)
        started = time.perf_counter()
        final, (positions, times, statuses, info) = jax.block_until_ready(simulate(model, state))
        elapsed = time.perf_counter() - started
        positions, times, statuses = map(np.asarray, (positions, times, statuses))
        diagnostic_data = {k: np.asarray(v) for k, v in info._asdict().items()}
        first_failed = np.flatnonzero(statuses)
        valid_steps = int(first_failed[0]) if len(first_failed) else steps
        ids = np.arange(args.record_stride - 1, valid_steps, args.record_stride)
        frame_positions = np.concatenate((np.asarray(state.x)[None], positions[ids]))
        frame_times = np.concatenate(([0.0], times[ids]))
        gaps = np.asarray(terrain.gap_normal(jnp.asarray(frame_positions), t)[0])
        entry: Report = dict(
            name=case["name"],
            title=case["title"],
            subtitle=case["subtitle"],
            terrain_config=asdict(case["tc"]),
            terrain=terrain.serialize(t),
            initial_clearance_m=case["clearance"],
            initial_velocity_m_s=list(case["velocity"]),
            requested_steps=steps,
            accepted_steps=int(diagnostic_data["accepted"].sum()),
            failed_steps=int(diagnostic_data["failed"].sum()),
            status=int(final.status),
            first_failed_step=int(first_failed[0]) + 1 if len(first_failed) else None,
            maximum_residual=float(diagnostic_data["max_residual"].max()),
            maximum_penetration_m=float(diagnostic_data["max_penetration"].max()),
            minimum_area_ratio=float(diagnostic_data["min_area_ratio"].min()),
            minimum_length_ratio=float(diagnostic_data["min_length_ratio"].min()),
            simulated_seconds=float(final.time),
            wall_seconds_including_first_compile=elapsed,
            com_displacement_m=np.asarray(final.com_displacement).tolist(),
            source_script="src/envs/soft_walker/scripts/create_spring_terrain_gallery.py",
        )
        entry["spring_config"] = asdict(cfg)
        np.savez_compressed(
            args.out / f"{case['name']}.npz",
            positions=positions,
            times=times,
            statuses=statuses,
            reference=np.asarray(model.reference),
            cells=np.asarray(model.cells),
            edges=np.asarray(model.edges),
            mass=np.asarray(model.mass),
            **diagnostic_data,
        )
        entry.update(
            render(
                case,
                model,
                frame_positions,
                gaps,
                frame_times,
                args.out,
                args.dt,
                args.record_stride,
            )
        )
        report["cases"].append(entry)
        (args.out / "metadata.json").write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps(
                {
                    k: entry[k]
                    for k in (
                        "name",
                        "accepted_steps",
                        "status",
                        "maximum_penetration_m",
                        "simulated_seconds",
                    )
                }
            ),
            flush=True,
        )
    manifest: Report = dict(
        schema="codesign_visual_gallery_manifest_v1",
        completed=all(c["status"] == 0 for c in report["cases"]),
        items=[],
    )
    for c in report["cases"]:
        if c["status"] != 0:
            continue
        manifest["items"].append(
            dict(
                id=c["name"],
                title=c["title"],
                description=c["subtitle"]
                + ". Passive spring lattice; blue shades identify cell columns, amber nodes indicate near contact.",
                dim=2,
                kind="physics",
                image=c["thumbnail"],
                animation=c["animation"],
                links=[
                    dict(label="Simulation evidence", href="metadata.json"),
                    dict(label="Recorded trajectory", href=f"{c['name']}.npz"),
                ],
                steps=c["accepted_steps"],
                requested_steps=c["requested_steps"],
                status=c["status"],
                simulated_seconds=c["simulated_seconds"],
            )
        )
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    thumbs = [
        Image.open(args.out / c["thumbnail"]).convert("RGB")
        for c in report["cases"]
        if c["status"] == 0
    ]
    if thumbs:
        contact_sheet = Image.new(
            "RGB", (840 * min(2, len(thumbs)), 560 * ((len(thumbs) + 1) // 2)), "#eff4f5"
        )
        for i, thumb in enumerate(thumbs):
            contact_sheet.paste(thumb, ((i % 2) * 840, (i // 2) * 560))
        if len(thumbs) == 3:
            d = ImageDraw.Draw(contact_sheet)
            d.text((886, 642), "TERRAIN CONTACT", fill="#193641", font=font(28, True))
            d.text(
                (886, 700),
                "Actual simulated spring-lattice trajectories",
                fill="#526a73",
                font=font(20),
            )
            d.text(
                (886, 750),
                "Gravity + elasticity + regularized terrain contact",
                fill="#526a73",
                font=font(18),
            )
            d.text(
                (886, 792),
                "Blue shades identify columns; amber marks near contact.",
                fill="#526a73",
                font=font(18),
            )
            d.text(
                (886, 850),
                "No trained gait. No prescribed or keyframed motion.",
                fill="#526a73",
                font=font(18),
            )
            d.text(
                (886, 910),
                "JAX  |  float64  |  backward Euler  |  CPU",
                fill="#526a73",
                font=font(18),
            )
        contact_sheet.save(args.out / "contact_sheet.png")
    if any(c["status"] != 0 for c in report["cases"]):
        raise SystemExit("A trajectory failed; metadata and rendered accepted prefix were retained")
    print(f"Saved {len(report['cases'])} simulated terrain demos to {args.out}", flush=True)


if __name__ == "__main__":
    main()
