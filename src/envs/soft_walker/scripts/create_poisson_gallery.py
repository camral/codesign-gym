#!/usr/bin/env python3
"""Render reproducible decoder fields; these pictures do not run dynamics.

Every color is read from Fields.poisson at the displayed material sample.
The sweep changes only LeggedDecoder.poisson_values and verifies that all
other decoded fields are unchanged. Output designs and NPZs accompany images.
"""

from __future__ import annotations

import argparse

# Resolve integrated imports independently of the caller's working directory.
import sys as _script_sys
from pathlib import Path, Path as _ScriptPath
from typing import TYPE_CHECKING, Any, Callable, cast

_source_root = str(_ScriptPath(__file__).resolve().parents[3])
if _source_root not in _script_sys.path:
    _script_sys.path.insert(0, _source_root)
_script_dir = str(_ScriptPath(__file__).resolve().parent)
if _script_dir not in _script_sys.path:
    _script_sys.path.insert(0, _script_dir)
import json
import os
import sys
import time

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
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image

from envs.soft_walker.engine.embodiments.decoders import grid_points
from envs.soft_walker.engine.embodiments.morphology import generate_random_design, sample_fields
from envs.soft_walker.engine.embodiments.serialization import load_design, save_design
from envs.soft_walker.presets.locomotion import LeggedDecoder

if TYPE_CHECKING:
    from pathlib import Path

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from envs.soft_walker.engine.embodiments.decoders import Decoder
    from envs.soft_walker.engine.types import (
        Array,
        PathLike,
    )

BG = "#0c1526"
PANEL = "#132239"
FG = "#ecf3ff"
MUTED = "#9bb0c9"
CMAP = LinearSegmentedColormap.from_list(
    "material_nu", ["#436bd7", "#28b5c4", "#f0dc79", "#f47852"]
)
NORM = Normalize(0.25, 0.45)
LABELS = {
    "direct": "Direct grid",
    "bspline": "B-spline",
    "cppn": "Implicit CPPN",
    "primitives": "Primitives",
}

# Serialized reports can contain backend-specific auxiliary diagnostic trees.
Report = dict[str, Any]


def write(path: PathLike, data: Any) -> None:
    """Serialize evidence to the requested path using this command's existing JSON encoder."""
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def style() -> None:
    """Return the shared CSS used by the generated offline gallery."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "text.color": FG,
            "axes.labelcolor": MUTED,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "figure.facecolor": BG,
            "savefig.facecolor": BG,
        }
    )


def colorbar(
    fig: Figure, rect: tuple[float, float, float, float] = (0.23, 0.11, 0.54, 0.028)
) -> None:
    """Add the shared material/Poisson color scale to a Matplotlib figure."""
    axis = fig.add_axes(rect)
    bar = fig.colorbar(
        plt.cm.ScalarMappable(norm=NORM, cmap=CMAP), cax=axis, orientation="horizontal"
    )
    bar.set_ticks([0.25, 0.30, 0.35, 0.40, 0.45])
    bar.outline.set_visible(False)
    bar.ax.tick_params(length=0, pad=7, labelsize=9)
    return bar


def draw2d(axis: Axes, shape: tuple[int, ...], arrays: Array) -> None:
    """Draw sampled two-dimensional material fields on the supplied axes."""
    mask = arrays["occupancy"].reshape(shape) > 0
    values = np.ma.array(arrays["poisson"].reshape(shape), mask=~mask)
    axis.set_facecolor(PANEL)
    axis.imshow(
        values.T,
        origin="lower",
        extent=(-1, 1, -1, 1),
        cmap=CMAP,
        norm=NORM,
        interpolation="nearest",
    )
    axis.set_xlim(-1.18, 1.18)
    axis.set_ylim(-1.18, 1.18)
    axis.set_aspect("equal")
    for location in (-1.0, 0.0, 1.0):
        axis.axvline(location, color="#34506b", lw=0.5, alpha=0.26, zorder=0)
        axis.axhline(location, color="#34506b", lw=0.5, alpha=0.26, zorder=0)
    axis.set_axis_off()


def draw3d(axis: Axes, shape: tuple[int, ...], arrays: Array) -> None:
    """Draw sampled three-dimensional material fields on the supplied axes."""
    mask = arrays["occupancy"].reshape(shape) > 0
    nu = arrays["poisson"].reshape(shape)
    spacing = 2.0 / np.asarray(shape)
    # Material coordinates are x,y,z with y vertical; matplotlib displays x,z,y.
    faces, colors = [], []
    for cell in np.argwhere(mask):
        center = -1.0 + (cell + 0.5) * spacing
        for dim in range(3):
            other = [d for d in range(3) if d != dim]
            for sign in (-1, 1):
                neighbor = cell.copy()
                neighbor[dim] += sign
                if np.all(neighbor >= 0) and np.all(neighbor < shape) and mask[tuple(neighbor)]:
                    continue
                corners = []
                for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                    point = center.copy()
                    point[dim] += sign * spacing[dim] / 2
                    point[other[0]] += a * spacing[other[0]] / 2
                    point[other[1]] += b * spacing[other[1]] / 2
                    corners.append(point[[0, 2, 1]])
                faces.append(corners)
                color = np.array(CMAP(NORM(nu[tuple(cell)])))
                shade = {
                    (0, -1): 0.79,
                    (0, 1): 0.94,
                    (1, -1): 0.64,
                    (1, 1): 1.0,
                    (2, -1): 0.88,
                    (2, 1): 0.73,
                }[(dim, sign)]
                color[:3] *= shade
                colors.append(color)
    axis.add_collection3d(
        Poly3DCollection(
            faces, facecolors=colors, edgecolors=(0.04, 0.08, 0.13, 0.10), linewidths=0.18
        )
    )
    for location in np.linspace(-1.2, 1.2, 9):
        axis.plot(
            [-1.2, 1.2], [location, location], [-1.06, -1.06], color="#314860", lw=0.55, alpha=0.55
        )
        axis.plot(
            [location, location], [-1.2, 1.2], [-1.06, -1.06], color="#314860", lw=0.55, alpha=0.55
        )
    axis.set(xlim=(-1.2, 1.2), ylim=(-1.2, 1.2), zlim=(-1.1, 1.2))
    axis.set_box_aspect((1, 1, 1))
    axis.view_init(elev=24, azim=-48)
    axis.set_axis_off()
    axis.set_facecolor(BG)


def render_card(
    path: Path, title: str, subtitle: str, shape: tuple[int, ...], arrays: Array, note: str
) -> None:
    """Save a labeled material-field preview with its discretization and explanatory note."""
    fig = plt.figure(figsize=(7.2, 7.2), dpi=140)
    fig.text(
        0.07, 0.944, "SPATIAL MATERIALS / POISSON RATIO", color="#78c7dc", fontsize=9, weight="bold"
    )
    fig.text(0.07, 0.895, title, fontsize=23, weight="bold")
    fig.text(0.07, 0.852, subtitle, fontsize=11, color=MUTED)
    dim = len(shape)
    axis = fig.add_axes((0.10, 0.215, 0.80, 0.63), projection="3d" if dim == 3 else None)
    (draw2d if dim == 2 else draw3d)(axis, shape, arrays)
    occupied = arrays["occupancy"] > 0
    nu = arrays["poisson"][occupied]
    fig.text(
        0.07,
        0.195,
        f"{np.count_nonzero(occupied):,} occupied samples    |    observed ν {nu.min():.3f}–{nu.max():.3f}",
        color=MUTED,
        fontsize=10,
    )
    colorbar(fig)
    fig.text(0.5, 0.038, note, ha="center", color=MUTED, fontsize=9)
    fig.savefig(path)
    plt.close(fig)


def sample_and_save(
    out: Path,
    identifier: str,
    title: str,
    subtitle: str,
    decoder: Decoder,
    embedding: Array,
    shape: tuple[int, ...],
    metadata: Report,
) -> Report:
    """Decode a candidate, save its fields/metadata, and produce its preview card."""
    directory = out / identifier
    directory.mkdir()
    design = directory / "design.json"
    save_design(design, embedding, decoder, metadata)
    # Custom decoders require their explicit constructor on reload.
    restored, restored_decoder, _ = load_design(design, decoder=decoder)
    points, fields = sample_fields(decoder, embedding, shape)
    _, other = sample_fields(restored_decoder, restored, shape)
    for actual, expected in zip(fields, other):
        np.testing.assert_array_equal(actual, expected)
    arrays = {key: np.asarray(value) for key, value in fields._asdict().items()}
    # Known field keys cannot collide with NumPy's optional allow_pickle keyword.
    cast(Callable[..., None], np.savez_compressed)(
        directory / "fields.npz", points=points, shape=np.array(shape), **arrays
    )
    occupied = arrays["occupancy"] > 0
    assert np.any(occupied) and np.isfinite(arrays["poisson"]).all()
    assert np.ptp(arrays["poisson"][occupied]) > 0.001
    render_card(
        directory / "preview.png",
        title,
        subtitle,
        shape,
        arrays,
        "Decoded material field · fixed reference geometry · no dynamics",
    )
    manifest: Report = dict(
        id=identifier,
        title=title,
        description=subtitle,
        dimension=len(shape),
        image=f"{identifier}/preview.png",
        design=f"{identifier}/design.json",
        fields=f"{identifier}/fields.npz",
        sample_shape=list(shape),
        occupied_samples=int(np.count_nonzero(occupied)),
        poisson_min=float(arrays["poisson"][occupied].min()),
        poisson_max=float(arrays["poisson"][occupied].max()),
        color_scale=[0.25, 0.45],
        dynamics_run=False,
        roundtrip_fields_identical=True,
        metadata=metadata,
    )
    write(directory / "manifest.json", manifest)
    print(
        json.dumps({"completed": identifier, "occupied_samples": manifest["occupied_samples"]}),
        flush=True,
    )
    return manifest


def sweep(out: Path, frame_count: int) -> Report:
    """Render a sequence of spatial Poisson fields and package the sequence metadata."""
    identifier = "biped_field_sweep"
    directory = out / identifier
    directory.mkdir()
    decoder = LeggedDecoder(dim=2, dtype="float64")
    embedding = decoder.base()
    coefficient_points = grid_points((3, 3))
    shape = (90, 90)
    _, baseline = sample_fields(decoder, embedding, shape)
    frames, coefficients, values = [], [], []
    for index in range(frame_count):
        phase = 2 * np.pi * index / frame_count
        # A rotating gradient is specified at the physical 3x3 material coefficients.
        field = 0.35 + 0.065 * (
            coefficient_points[..., 0] * np.cos(phase) + coefficient_points[..., 1] * np.sin(phase)
        )
        embedding["poisson_values"] = field[..., None]
        points, fields = sample_fields(decoder, embedding, shape)
        for name in baseline._fields:
            if name != "poisson":
                np.testing.assert_array_equal(getattr(fields, name), getattr(baseline, name))
        arrays = {key: np.asarray(value) for key, value in fields._asdict().items()}
        coefficients.append(field)
        values.append(arrays["poisson"])
        fig = plt.figure(figsize=(10, 5.65), dpi=105)
        fig.text(0.06, 0.935, "PROGRAM A MATERIAL FIELD", fontsize=21, weight="bold")
        fig.text(
            0.06,
            0.885,
            "Change 3 × 3 coefficients → decode local Poisson ratio on the same biped",
            fontsize=11,
            color=MUTED,
        )
        left = fig.add_axes((0.09, 0.22, 0.30, 0.53))
        left.imshow(field.T, origin="lower", cmap=CMAP, norm=NORM, extent=(-1, 1, -1, 1))
        for x in range(3):
            for y in range(3):
                left.text(
                    -2 / 3 + 2 * x / 3,
                    -2 / 3 + 2 * y / 3,
                    f"{field[x, y]:.2f}",
                    ha="center",
                    va="center",
                    color=BG,
                    weight="bold",
                    fontsize=12,
                )
        left.set_axis_off()
        right = fig.add_axes((0.56, 0.19, 0.34, 0.59))
        draw2d(right, shape, arrays)
        fig.text(
            0.24,
            0.79,
            "EMBEDDING COEFFICIENTS",
            ha="center",
            fontsize=10,
            color="#78c7dc",
            weight="bold",
        )
        fig.text(
            0.735,
            0.79,
            "ACTUAL DECODED FIELD",
            ha="center",
            fontsize=10,
            color="#78c7dc",
            weight="bold",
        )
        fig.text(0.46, 0.48, "→", fontsize=34, color="#78c7dc")
        colorbar(fig, (0.24, 0.115, 0.52, 0.03))
        fig.text(
            0.5,
            0.035,
            "Material-field sweep, not a physics simulation. Shape, stiffness and actuation remain fixed.",
            ha="center",
            fontsize=10,
            color=MUTED,
        )
        fig.canvas.draw()
        frame = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())
        frames.append(frame)
        if index == 0:
            frame.save(directory / "preview.png")
            save_design(
                directory / "design.json",
                embedding,
                decoder,
                {"demo": "first frame of material field sweep", "dynamics_run": False},
            )
        plt.close(fig)
    frames[0].save(
        directory / "animation.gif",
        save_all=True,
        append_images=frames[1:],
        duration=100,
        loop=0,
        optimize=False,
        disposal=2,
    )
    np.savez_compressed(
        directory / "fields.npz",
        points=points,
        shape=np.array(shape),
        poisson_coefficients=np.asarray(coefficients),
        poisson_frames=np.asarray(values),
        occupancy=np.asarray(baseline.occupancy),
    )
    manifest: Report = dict(
        id=identifier,
        title="Program the biped’s material field",
        description="A rotating Poisson-ratio gradient on fixed geometry. Left: physical material coefficients; right: real decoder output.",
        image=f"{identifier}/preview.png",
        animation=f"{identifier}/animation.gif",
        design=f"{identifier}/design.json",
        fields=f"{identifier}/fields.npz",
        frame_count=frame_count,
        duration_seconds=frame_count / 10,
        color_scale=[0.25, 0.45],
        dynamics_run=False,
        geometry_and_other_fields_identical_all_frames=True,
    )
    write(directory / "manifest.json", manifest)
    print(json.dumps({"completed": identifier, "frames": frame_count}), flush=True)
    return manifest


def overview(out: Path, cards: list[Report]) -> None:
    """Compose the generated candidate cards into a gallery overview."""
    # Composite original rendered panels without changing any material colors.
    width, height = 504, 504
    canvas = Image.new("RGB", (width * 4, height * 2), BG)
    for index, card in enumerate(cards[:8]):
        picture = Image.open(out / card["image"]).convert("RGB")
        picture.thumbnail((width, height), Image.Resampling.LANCZOS)
        canvas.paste(picture, ((index % 4) * width, (index // 4) * height))
    canvas.save(out / "all_representations.png")
    canvas.save(out / "contact_sheet.png")


def main() -> None:
    """Command entry point. Render reproducible decoder fields; these pictures do not run dynamics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=ROOT / "runs/spatial_material_terrain_gallery/materials"
    )
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--sweep-frames", type=int, default=40)
    args = parser.parse_args()
    if args.sweep_frames < 4:
        parser.error("--sweep-frames must be at least 4")
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error(f"Use a fresh output folder; preserving existing files at {out}")
    out.mkdir(parents=True, exist_ok=True)
    style()
    started = time.perf_counter()
    cards = []
    for dim in (2, 3):
        shape = (100, 84) if dim == 2 else (28, 22, 22)
        for representation in ("direct", "bspline", "cppn", "primitives"):
            decoder, embedding, report = generate_random_design(
                representation, dim=dim, seed=args.seed, dtype="float64", max_attempts=100
            )
            cards.append(
                sample_and_save(
                    out,
                    f"{representation}_{dim}d",
                    f"{LABELS[representation]} · {dim}D",
                    f"Random creature / seed {args.seed} / local material values",
                    decoder,
                    embedding,
                    shape,
                    {"generator": report, "demo": "sampled random material field"},
                )
            )
    for dim, name in ((2, "biped"), (3, "quadruped")):
        decoder = LeggedDecoder(dim=dim, dtype="float64")
        embedding = decoder.base()
        p = grid_points((3,) * dim)
        embedding["poisson_values"] = (0.35 + 0.045 * p[..., 0] + 0.045 * p[..., 1])[..., None]
        cards.append(
            sample_and_save(
                out,
                f"stock_{name}",
                f"Stock {name}",
                "Programmed 3 × 3 material grid"
                if dim == 2
                else "Programmed 3 × 3 × 3 material grid",
                decoder,
                embedding,
                (96, 96) if dim == 2 else (24, 24, 24),
                {"demo": "fixed stock geometry, spatial physical Poisson coefficients"},
            )
        )
    cards.append(sweep(out, args.sweep_frames))
    overview(out, cards)
    for card in cards:
        card["kind"] = "field"
        card["dim"] = card.get("dimension", 2)
        card["links"] = [
            {"label": "Design JSON", "href": card["design"]},
            {"label": "Decoded fields NPZ", "href": card["fields"]},
        ]
    write(
        out / "manifest.json",
        {
            "schema": "codesign_material_gallery_v1",
            "completed": True,
            "items": cards,
            "overview": "contact_sheet.png",
            "dynamics_run": False,
            "description": "All colors are actual decoded Fields.poisson values. Three-dimensional face shading changes brightness only. Display sample cells are not solver elements.",
            "source_script": "src/envs/soft_walker/scripts/create_poisson_gallery.py",
            "wall_seconds": time.perf_counter() - started,
        },
    )


if __name__ == "__main__":
    main()
