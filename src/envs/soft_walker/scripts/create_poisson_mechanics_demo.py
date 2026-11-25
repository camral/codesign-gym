#!/usr/bin/env python3
"""Render real HexFEM3D quasi-static relaxation with programmable Poisson ratio.

This is displacement-controlled elastic energy minimization, not a trajectory
integrated in time. The animation traverses and reverses the solved loading
states for readability; its frame duration has no physical-time interpretation.
"""

from __future__ import annotations

import argparse

# Resolve integrated imports independently of the caller's working directory.
import sys as _script_sys
from pathlib import Path, Path as _ScriptPath
from typing import TYPE_CHECKING, Any

_source_root = str(_ScriptPath(__file__).resolve().parents[3])
if _source_root not in _script_sys.path:
    _script_sys.path.insert(0, _source_root)
_script_dir = str(_ScriptPath(__file__).resolve().parent)
if _script_dir not in _script_sys.path:
    _script_sys.path.insert(0, _script_dir)
import json
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import solve
from scipy.optimize import brentq

from envs.soft_walker.engine.embodiments.decoders import FieldSpec, GridDecoder, set_poisson
from envs.soft_walker.engine.embodiments.serialization import save_design
from envs.soft_walker.engine.physics import fem, material
from envs.soft_walker.engine.physics.common import DynamicsConfig
from envs.soft_walker.shared import (
    ENGINE_ROOT as ENGINE_ROOT,
    REPO_ROOT as REPO_ROOT,
    SCRIPTS_ROOT as SCRIPTS_ROOT,
    SOFT_ROOT as SOFT_ROOT,
    SRC_ROOT as SRC_ROOT,
    TESTS_ROOT as TESTS_ROOT,
)

if TYPE_CHECKING:
    from pathlib import Path

    from PIL import ImageFont

    from envs.soft_walker.engine.types import (
        Array,
        HostArray,
    )

# Serialized reports can contain backend-specific auxiliary diagnostic trees.
Report = dict[str, Any]


def render(
    output: Path,
    reference: Array,
    front_faces: Array,
    states: Array,
    field_values: Array,
    strains: Array,
    diagnostics: list[list[Report]],
) -> None:
    """Render the supplied saved/sample geometry and diagnostics into image or animation artifacts."""
    from PIL import Image, ImageDraw, ImageFont

    def font(size: int, bold: bool = False) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
        names = (
            f"C:/Windows/Fonts/{'segoeuib' if bold else 'segoeui'}.ttf",
            f"/mnt/c/Windows/Fonts/{'segoeuib' if bold else 'segoeui'}.ttf",
            f"/usr/share/fonts/truetype/dejavu/DejaVuSans{'-Bold' if bold else ''}.ttf",
        )
        for name in names:
            if Path(name).is_file():
                return ImageFont.truetype(name, size)
        return ImageFont.load_default()

    stops = np.asarray([[67, 107, 215], [40, 181, 196], [240, 220, 121], [244, 120, 82]])

    def color(value: float) -> tuple[np.integer, ...]:
        t = np.clip((float(value) - 0.25) / 0.20, 0, 1) * 3
        lo = min(int(t), 2)
        return tuple(np.rint(stops[lo] * (1 - t + lo) + stops[lo + 1] * (t - lo)).astype(int))

    titles = [
        "Uniform · ν = 0.25",
        "Uniform · ν = 0.35",
        "Uniform · ν = 0.45",
        "Spatial · graded ν",
    ]
    subtitles = [
        "Lower transverse contraction",
        "Middle reference material",
        "Higher transverse contraction",
        "Local contraction follows the material field",
    ]
    length, width = 0.12, 0.048
    frames = []
    for frame, strain in enumerate(strains):
        canvas = Image.new("RGB", (1600, 1030), "#eef3f7")
        draw = ImageDraw.Draw(canvas)
        draw.text((55, 33), "ONE SHAPE. FOUR MATERIAL FIELDS.", font=font(23, True), fill="#466984")
        draw.text(
            (53, 79),
            "Watch Poisson’s ratio change the lateral response",
            font=font(40, True),
            fill="#18324a",
        )
        draw.text(
            (55, 141),
            f"Axial extension: {100 * strain:4.1f}%    |    free lateral contraction    |    float64 / CPU",
            font=font(23),
            fill="#48657a",
        )
        for index in range(4):
            x0, y0 = 50 + (index % 2) * 770, 200 + (index // 2) * 337
            draw.rounded_rectangle((x0, y0, x0 + 730, y0 + 313), radius=16, fill="#ffffff")
            draw.text((x0 + 20, y0 + 15), titles[index], font=font(25, True), fill="#18324a")
            scale = 3900.0

            def transform(points: Array) -> HostArray:
                points = np.asarray(points)
                return np.stack(
                    [x0 + 48 + points[:, 0] * scale, y0 + 168 - points[:, 1] * scale], axis=1
                )

            positions = reference + states[index, frame]
            for face, nu in zip(front_faces, field_values[index]):
                vertices = [tuple(point) for point in transform(positions[face, :2])]
                draw.polygon(vertices, fill=color(nu))
                draw.line(vertices + [vertices[0]], fill="#f8fafb", width=2)
            outline = transform(
                [
                    [0, -width / 2],
                    [length, -width / 2],
                    [length, width / 2],
                    [0, width / 2],
                    [0, -width / 2],
                ]
            )
            for start, end in zip(outline[:-1], outline[1:]):
                distance = np.linalg.norm(end - start)
                for offset in np.arange(0, distance, 13):
                    a = start + (end - start) * (offset / distance)
                    b = start + (end - start) * (min(distance, offset + 7) / distance)
                    draw.line([tuple(a), tuple(b)], fill="#7e909e", width=2)
            width_ratio = (positions[:, 1].max() - positions[:, 1].min()) / width
            record = diagnostics[index][frame]
            draw.text(
                (x0 + 710, y0 + 20),
                f"Height {100 * width_ratio:5.1f}%",
                font=font(22),
                fill="#24485f",
                anchor="ra",
            )
            draw.text((x0 + 20, y0 + 282), subtitles[index], font=font(19), fill="#547083")
            if not record["converged"]:
                draw.text(
                    (x0 + 710, y0 + 58),
                    "NOT CONVERGED",
                    font=font(20, True),
                    fill="#c43228",
                    anchor="ra",
                )
        draw.text(
            (55, 906),
            "HexFEM3D  ·  64 elements per specimen  ·  E = 20 kPa",
            font=font(21, True),
            fill="#24485f",
        )
        draw.text(
            (55, 943),
            "Quasi-static equilibrium; true-scale displacement. No gravity or contact.",
            font=font(20),
            fill="#48657a",
        )
        draw.text(
            (55, 979),
            "Dashed = unloaded shape. Animation speed does not represent elapsed physical time.",
            font=font(18),
            fill="#567184",
        )
        for column in range(420):
            draw.line(
                [(1100 + column, 905), (1100 + column, 925)], fill=color(0.25 + 0.20 * column / 419)
            )
        for nu in [0.25, 0.30, 0.35, 0.40, 0.45]:
            draw.text(
                (1100 + (nu - 0.25) / 0.20 * 420, 936),
                f"{nu:.2f}",
                font=font(18),
                fill="#48657a",
                anchor="ma",
            )
        draw.text((1310, 970), "Local Poisson ratio ν", font=font(20), fill="#24485f", anchor="ma")
        frames.append(canvas)
        if frame == len(strains) - 1:
            frames[-1].save(output / "poisson_comparison.png")
    # A visual loading/unloading loop, not simulated elapsed time.
    sequence = frames + [frames[-1]] * 7 + frames[-2:0:-1] + [frames[0]] * 7
    sequence[0].save(
        output / "poisson_comparison.gif",
        save_all=True,
        append_images=sequence[1:],
        duration=110,
        loop=0,
        optimize=False,
    )


def main() -> None:
    """Command entry point. Render real HexFEM3D quasi-static relaxation with programmable Poisson ratio."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=REPO_ROOT / "runs/spatial_material_terrain_gallery/mechanics"
    )
    parser.add_argument("--frames", type=int, default=21)
    args = parser.parse_args()
    if args.frames < 3:
        parser.error("--frames must be at least 3")
    jax.config.update("jax_enable_x64", True)
    if any(device.platform != "cpu" for device in jax.devices()):
        raise RuntimeError("Run with JAX_PLATFORMS=cpu; this small demo is intended for CPU")
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    shape, size, origin = (8, 4, 2), (0.12, 0.048, 0.028), (0.0, -0.024, -0.014)
    mesh = fem.structured_mesh(shape, size, origin, np.float64)
    reference = np.asarray(mesh.reference)
    decoder = GridDecoder(FieldSpec(dim=3), shape=(3, 3, 3), dtype="float64")
    names = ["uniform_025", "uniform_035", "uniform_045", "graded"]
    models, fields_list, embeddings = [], [], []
    for index, name in enumerate(names):
        embedding = decoder.base()
        # Passive specimen: no actuation or body force participates in the solve.
        embedding["values"][..., 10:13] = [-2.0, -2.0, -2.0]
        embedding["values"][..., 10] = 2.0
        if index < 3:
            embedding = set_poisson(decoder, embedding, [0.25, 0.35, 0.45][index])
        else:
            embedding["values"][..., decoder.spec.poisson_index] = np.linspace(-1.4, 1.4, 3)[
                :, None, None
            ]
        fields = decoder.decode(jax.tree.map(jnp.asarray, embedding), mesh.points)
        model = fem.instantiate(mesh, fields)
        fields_list.append(fields)
        models.append(model)
        embeddings.append(embedding)
        save_design(output / f"{name}_design.json", embedding, decoder)

    # Plate constraints prescribe only axial displacement, allowing transverse
    # sliding on both end faces. Three transverse scalar anchors remove rigid
    # translation and rotation without clamping a face against Poisson motion.
    fixed = np.zeros(reference.shape, dtype=bool)
    left = np.isclose(reference[:, 0], 0.0)
    right = np.isclose(reference[:, 0], size[0])
    fixed[left | right, 0] = True
    center = int(np.argmin(np.sum((reference - np.array([0.0, 0.0, 0.0])) ** 2, axis=1)))
    axis_node = int(np.argmin(np.sum((reference - np.array([0.0, 0.0, size[2] / 2])) ** 2, axis=1)))
    fixed[center, 1:] = True
    fixed[axis_node, 1] = True
    free = np.flatnonzero(~fixed.reshape(-1))
    prescribed = np.zeros_like(reference)
    prescribed[right, 0] = size[0]
    prescribed = jnp.asarray(prescribed)
    free_indices = jnp.asarray(free)
    cfg = DynamicsConfig(gravity=0.0)
    scale = 20000.0 * np.prod(size)
    base_model = models[0]

    def unpack(z: Array, strain: float | Array) -> Array:
        return (
            (prescribed * strain)
            .reshape(-1)
            .at[free_indices]
            .set(z * size[0])
            .reshape(reference.shape)
        )

    def objective(z: Array, strain: float | Array, poisson: float | Array) -> Array:
        model = base_model._replace(poisson=poisson)
        return fem.elastic_energy(model, unpack(z, strain), model.channels, cfg) / scale

    value_grad = jax.jit(jax.value_and_grad(objective))
    hessian = jax.jit(jax.hessian(objective))
    minimum_j = jax.jit(
        lambda z, s: jnp.min(material.determinant(fem.deformation(base_model, unpack(z, s))))
    )
    strains = np.linspace(0.0, 0.30, args.frames)
    all_states, diagnostics = [], []
    for index, (name, model) in enumerate(zip(names, models)):
        states, records = [], []
        z: HostArray = np.zeros(len(free), dtype=np.float64)
        for frame, strain in enumerate(strains):
            frame_started = time.perf_counter()

            def fun(candidate: Array) -> tuple[float, HostArray]:
                energy, gradient = value_grad(
                    jnp.asarray(candidate), jnp.asarray(strain), model.poisson
                )
                return float(energy), np.asarray(gradient, dtype=np.float64)

            if frame:
                increment = np.zeros_like(reference)
                increment[:, 0] = reference[:, 0] * (strain - strains[frame - 1])
                z = z + increment.reshape(-1)[free] / size[0]
            message = "Iteration limit"
            backtracks = 0
            for iteration in range(50):
                value, gradient = fun(z)
                if np.max(np.abs(gradient)) < 1e-9:
                    message = "Free-DOF gradient converged"
                    break
                matrix = np.asarray(hessian(jnp.asarray(z), jnp.asarray(strain), model.poisson))
                direction = solve(matrix, -gradient, assume_a="sym")
                if not np.all(np.isfinite(direction)) or np.dot(direction, gradient) >= 0:
                    message = "Non-descent Newton direction"
                    break
                step = 1.0
                accepted = False
                for trial in range(30):
                    candidate = z + step * direction
                    if float(minimum_j(jnp.asarray(candidate), strain)) > 0.1:
                        candidate_value, _ = fun(candidate)
                        if np.isfinite(
                            candidate_value
                        ) and candidate_value <= value + 1e-4 * step * np.dot(gradient, direction):
                            z = candidate
                            accepted = True
                            break
                    step *= 0.5
                    backtracks += 1
                if not accepted:
                    message = "Backtracking failed"
                    break
            value, gradient = fun(z)
            displacement = np.asarray(unpack(jnp.asarray(z), strain))
            min_j = float(
                jnp.min(material.determinant(fem.deformation(model, jnp.asarray(displacement))))
            )
            residual = float(np.max(np.abs(gradient)))
            converged = bool(np.isfinite(value) and min_j > 0 and residual < 2e-7)
            states.append(displacement)
            records.append(
                {
                    "frame": frame,
                    "axial_strain": float(strain),
                    "converged": converged,
                    "optimizer_success": message == "Free-DOF gradient converged",
                    "optimizer_message": message,
                    "iterations": int(iteration),
                    "line_search_backtracks": backtracks,
                    "energy_joule": float(value * scale),
                    "free_gradient_inf_normalized": residual,
                    "min_deformation_determinant": min_j,
                    "wall_seconds": time.perf_counter() - frame_started,
                }
            )
            print(
                f"{name} frame {frame + 1}/{len(strains)} strain={strain:.3f} residual={residual:.3e} minJ={min_j:.4f} converged={converged}",
                flush=True,
            )
        all_states.append(states)
        diagnostics.append(records)
    all_states = np.asarray(all_states)
    # Independent homogeneous neo-Hookean solution: zero transverse stress
    # gives mu*(lambda_t**2-1)+lambda*log(lambda_x*lambda_t**2)=0.
    uniform_reference_errors = []
    for index, nu in enumerate((0.25, 0.35, 0.45)):
        mu = 20000.0 / (2 * (1 + nu))
        lame_lambda = 20000.0 * nu / ((1 + nu) * (1 - 2 * nu))
        errors: list[float] = []
        for frame, strain in enumerate(strains):
            lateral = brentq(
                lambda t: mu * (t * t - 1) + lame_lambda * np.log((1 + strain) * t * t),
                0.5,
                1.1,
                xtol=1e-14,
            )
            expected = reference * np.asarray([strain, lateral - 1, lateral - 1])
            errors.append(float(np.max(np.abs(all_states[index, frame] - expected))))
        uniform_reference_errors.append(max(errors))
    if max(uniform_reference_errors) > 1e-8:
        raise RuntimeError(f"Homogeneous analytical reference mismatch: {uniform_reference_errors}")
    faces = np.asarray(mesh.faces)
    # Side view of the exterior positive-z faces; each Hex8 face is stored in
    # tensor order, so [0,1,3,2] gives a non-crossing polygon perimeter.
    front_elements = np.flatnonzero(~np.asarray(mesh.has_neighbor)[:, 5])
    front_faces = faces[front_elements, 5][:, [0, 1, 3, 2]]
    poisson = np.stack([np.asarray(fields.poisson) for fields in fields_list])
    np.savez_compressed(
        output / "solved_states.npz",
        reference=reference,
        cells=np.asarray(mesh.cells),
        front_faces=front_faces,
        displacements=all_states,
        strains=strains,
        poisson=poisson,
        young=np.asarray(fields_list[0].young),
        density=np.asarray(fields_list[0].density),
        occupancy=np.asarray(fields_list[0].occupancy),
        strength=np.asarray(fields_list[0].strength),
        direction=np.asarray(fields_list[0].direction),
        actuator_type=np.asarray(fields_list[0].actuator_type),
        channel=np.asarray(fields_list[0].channel),
        fixed_dof=fixed,
        free_dof_indices=free,
        coefficient_values=np.stack([x["values"] for x in embeddings]),
    )
    report: Report = {
        "method": "HexFEM3D elastic_energy minimized with damped Newton, exact JAX gradients/Hessians, SciPy symmetric linear solves, and determinant-aware Armijo backtracking",
        "physical_time_integration": False,
        "backend": "JAX CPU float64",
        "mesh_shape": shape,
        "size_m": size,
        "origin_m": origin,
        "young_pa": 20000.0,
        "density_kg_m3": 1000.0,
        "axial_strain_range": [0.0, 0.30],
        "frames": len(strains),
        "case_names": names,
        "boundary_conditions": "x=0 end has ux=0; x=L end has ux=strain*L; end faces slide transversely; left center uy=uz=0, second left node uy=0 removes axial rigid rotation",
        "convergence_criterion": "finite energy; min det(F)>0; max absolute normalized free-DOF energy gradient < 2e-7",
        "energy_normalization_joule": float(scale),
        "coordinate_normalization_m": size[0],
        "all_frames_converged": all(
            record["converged"] for group in diagnostics for record in group
        ),
        "poisson_ranges": [[float(x.min()), float(x.max())] for x in poisson],
        "diagnostics": diagnostics,
    }
    report["uniform_analytical_reference_max_displacement_errors_m"] = uniform_reference_errors
    (output / "convergence.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    render(
        output, reference, front_faces, all_states, poisson[:, front_elements], strains, diagnostics
    )
    items = [
        {
            "id": "poisson_mechanics",
            "title": "How local Poisson ratio changes real deformation",
            "description": "Four identical 3D HexFEM specimens: ν=0.25, 0.35, 0.45, and an embedding-defined gradient. Prescribed axial extension with free lateral contraction; every frame is an elastic energy equilibrium, shown at true scale. This is a quasi-static loading loop, not time integration.",
            "dim": 3,
            "kind": "quasistatic",
            "image": "poisson_comparison.png",
            "animation": "poisson_comparison.gif",
            "links": [
                {"label": "Convergence and setup", "href": "convergence.json"},
                {"label": "Solved states and fields", "href": "solved_states.npz"},
            ],
        }
    ]
    manifest: Report = {
        "items": items,
        "all_frames_converged": report["all_frames_converged"],
        "wall_seconds": time.perf_counter() - started,
        "render_note": "Orthographic front view of full three-dimensional solutions. Animation reverses computed equilibrium frames; frame duration is not physical time.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)
    if not report["all_frames_converged"]:
        raise SystemExit(
            "At least one frame did not meet equilibrium checks; inspect convergence.json"
        )


if __name__ == "__main__":
    main()
