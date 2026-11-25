"""Headless diagnostic raster previews with an explicit host-data boundary.

The body adapter copies device geometry into a :class:`BodyPreviewScene`; the
rasterizer consumes only NumPy arrays and Pillow. Neither path advances physics.
Pillow is imported only when rendering is explicitly requested. Camera and color
policies preserve the original preview appearance; they are not physics settings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from numpy.typing import ArrayLike

from .types import BodyModel, HostArray

if TYPE_CHECKING:
    from PIL import ImageDraw

    from .drivers.environment import BodyEnv
    from .drivers.task import TaskState
    from .physics import beam, fem, mpm, spring, terrain
    from .physics.spring_config import SpringConfig

import numpy as np

# These display policies were inherited from the original diagnostic renderer;
# they are visual choices, not calibrated material or contact parameters.
_BACKGROUND = (246, 248, 251)
_BLOCK_COLOR = (169, 184, 193)
# Padding is the total consumed span on each image axis, not a per-edge margin.
_SPRING_VIEWPORT_PADDING_PX = (80, 60)
_SPRING_SPAN_MULTIPLIERS = (1.5, 1.8)
_SPRING_CAMERA_HEIGHT_FRACTION = 0.45
_SPRING_VERTICAL_ANCHOR = 0.55
_SPRING_GROUND_COLOR = (217, 226, 229)
_SPRING_INK_COLOR = (20, 56, 75)
_SPRING_NODE_RADIUS_PX = 2
_SPRING_LABEL_ORIGIN_PX = (12, 10)
_BODY_PROJECTION = np.array([[1.0, 0.0, -0.42], [0.0, 1.0, 0.28]])
_BODY_FIT_FRACTIONS = (0.72, 0.65)
_BODY_VERTICAL_ANCHOR = 0.53
_BODY_GROUND_EXTENT_M = (0.3, 0.5)
_BODY_GROUND_SAMPLES = 80
_BODY_GROUND_STROKE_PX = 12
_BEAM_STROKE_PX = 6
_BEAM_MARKER_MIN_RADIUS_PX = 3.0
_BEAM_MARKER_SPACING_FRACTION = 0.28
_PARTICLE_MIN_RADIUS_PX = 2.0
_PARTICLE_SPACING_FRACTION = 0.3
# The logarithmic palette spans the historical 5--100 kPa material range.
_PALETTE_YOUNG_MIN_PA = 5000.0
_PALETTE_YOUNG_RATIO = 20.0
_QUAD_PERIMETER_SIGNS = ((-1, -1), (1, -1), (1, 1), (-1, 1))
_TENSOR_FACE_PERIMETER = np.array([0, 1, 3, 2])


@dataclass(frozen=True)
class TerrainPreview:
    """Host plane and box geometry in metres, without contact-model behavior.

    ``normal`` has shape ``(dimension,)``; centers and half-sizes have shape
    ``(boxes, dimension)``. ``active`` selects visible boxes. The legacy preview
    draws rectangular silhouettes even when the physical box has rounded edges.
    Arrays are read-only inputs to the renderer, owned by the scene's caller.
    """

    normal: HostArray
    offset: float
    centers: HostArray
    half_sizes: HostArray
    active: HostArray


@dataclass(frozen=True)
class SurfacePreview:
    """FEM host topology: tensor-ordered faces, neighbor links, and material fields.

    Faces have shape ``(elements, faces_per_element, 4)``; neighbor arrays have
    shape ``(elements, faces_per_element)``. ``alpha`` and ``young`` have one value
    per element; Young's modulus is in Pa. Exposed occupied faces are drawn only.
    """

    faces: HostArray
    neighbors: HostArray
    has_neighbor: HostArray
    alpha: HostArray
    young: HostArray


@dataclass(frozen=True)
class BeamPreview:
    """Host node pairs ``(edges, 2)`` and reference node spacing in metres."""

    edges: HostArray
    spacing: float


@dataclass(frozen=True)
class ParticlePreview:
    """MPM cell spacing in metres and samples per grid axis for marker sizing."""

    grid_spacing: float
    particles_per_axis: int


@dataclass(frozen=True)
class BodyPreviewScene:
    """Backend-free inputs for a single body frame, with no live environment.

    Positions have shape ``(nodes_or_particles, 2 | 3)`` in world metres; active
    has shape ``(nodes_or_particles,)``. ``length_ref`` is the minimum camera span
    in metres. At least one active point and a nonvertical terrain plane are
    required, as in the environment preview. The rasterizer never mutates arrays.
    ``body_preview_scene`` creates owned host copies; direct callers own theirs.
    """

    positions: HostArray
    active: HostArray
    terrain: TerrainPreview
    geometry: SurfacePreview | BeamPreview | ParticlePreview
    length_ref: float
    label: str


@dataclass(frozen=True)
class _BodyCamera:
    """Fixed oblique projection with a pixel scale fitted to active points."""

    center: HostArray
    scale: float
    width: int
    height: int

    def project(self, point: ArrayLike) -> tuple[float, float]:
        """Map one 2D/3D world point to image pixels without modifying it."""
        point = np.asarray(point)
        if point.shape[-1] == 2:
            point = np.r_[point, 0.0]
        offset = (_BODY_PROJECTION @ point - self.center) * self.scale
        return (
            float(self.width / 2 + offset[0]),
            float(self.height * _BODY_VERTICAL_ANCHOR - offset[1]),
        )


def _young_shade(young: float) -> float:
    """Map Pa to the clipped historical logarithmic palette coordinate."""
    return float(
        np.clip(np.log(young / _PALETTE_YOUNG_MIN_PA) / np.log(_PALETTE_YOUNG_RATIO), 0, 1)
    )


def render_rgb(
    model: spring.Model, state: spring.State, cfg: SpringConfig, width: int = 640, height: int = 320
) -> HostArray:
    "Render current spring geometry to an RGB host image without advancing physics."
    try:
        from PIL import Image, ImageDraw
    except ImportError as e:
        raise ImportError(
            "RGB preview requires Pillow, included in codesign-gym dependencies; "
            "install it with 'python -m pip install pillow'."
        ) from e
    image = Image.new("RGB", (width, height), _BACKGROUND)
    draw = ImageDraw.Draw(image)
    x = np.asarray(state.x)
    active = np.asarray(model.active)
    occupied = np.asarray(model.occupancy) > 0
    center = (x[active].min(axis=0) + x[active].max(axis=0)) / 2
    scale = min(
        (width - _SPRING_VIEWPORT_PADDING_PX[0]) / (cfg.size[0] * _SPRING_SPAN_MULTIPLIERS[0]),
        (height - _SPRING_VIEWPORT_PADDING_PX[1]) / (cfg.size[1] * _SPRING_SPAN_MULTIPLIERS[1]),
    )
    camera = np.array([center[0], cfg.ground_height + cfg.size[1] * _SPRING_CAMERA_HEIGHT_FRACTION])

    def project(q: ArrayLike) -> tuple[float, float]:
        "Project one world point to fixed image pixel coordinates."
        p = (q - camera) * scale
        return (float(width / 2 + p[0]), float(height * _SPRING_VERTICAL_ANCHOR - p[1]))

    gy = project(np.array([0.0, cfg.ground_height]))[1]
    if cfg.ground:
        if model.terrain is None:
            draw.rectangle((0, gy, width, height), fill=_SPRING_GROUND_COLOR)
        else:
            t = model.terrain
            lo = center[0] - width / (2 * scale)
            hi = center[0] + width / (2 * scale)
            y = lambda xx: (float(t.offset) - float(t.normal[0]) * xx) / float(t.normal[1])
            draw.polygon(
                [
                    project(np.array([lo, y(lo)])),
                    project(np.array([hi, y(hi)])),
                    (width, height),
                    (0, height),
                ],
                fill=_SPRING_GROUND_COLOR,
            )
            for c, h, on in zip(
                np.asarray(t.centers), np.asarray(t.half_sizes), np.asarray(t.active)
            ):
                if on:
                    draw.polygon(
                        [project(c + h * np.asarray(sign)) for sign in _QUAD_PERIMETER_SIGNS],
                        fill=_BLOCK_COLOR,
                    )
    for i, cell in enumerate(np.asarray(model.cells)):
        if not occupied[i]:
            continue
        shade = _young_shade(float(model.young[i]))
        color = (int(38 + 100 * shade), int(162 - 45 * shade), int(171 + 35 * shade))
        points = [project(q) for q in x[cell]]
        draw.polygon(points, fill=color, outline=_SPRING_INK_COLOR)
    for q in x[active]:
        px, py = project(q)
        radius = _SPRING_NODE_RADIUS_PX
        draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=_SPRING_INK_COLOR)
    draw.text(
        _SPRING_LABEL_ORIGIN_PX,
        f"Spring2D / t={float(state.time):.3f}s / {cfg.integrator}",
        fill=_SPRING_INK_COLOR,
    )
    return np.asarray(image)


def render_body_rgb(env: BodyEnv, width: int = 720, height: int = 420) -> HostArray:
    """Download a reset environment's geometry and return a host ``uint8`` RGB frame.

    Rendering is diagnostic and does not advance or mutate physics. The simple
    painter's algorithm is not an occlusion-exact engine. Device-to-host transfers
    occur in ``body_preview_scene``; direct host callers can reuse its output.
    """
    return render_body_scene_rgb(body_preview_scene(env), width, height)


def body_preview_scene(env: BodyEnv) -> BodyPreviewScene:
    """Snapshot a reset body environment into owned host arrays without stepping it.

    This adapter is the only body-rendering function that accesses environment
    internals or imports driver modules. Conversion may synchronize the device.
    As with the original renderer, calling before a successful reset is invalid.
    """
    from .drivers.environment import positions

    model, state = cast(BodyModel, env._model), cast("TaskState", env._state)
    geometry: SurfacePreview | BeamPreview | ParticlePreview
    if env.kind == "fem":
        fem_model = cast("fem.Model", model)
        geometry = SurfacePreview(
            faces=np.array(fem_model.mesh.faces, copy=True),
            neighbors=np.array(fem_model.mesh.neighbors, copy=True),
            has_neighbor=np.array(fem_model.mesh.has_neighbor, copy=True),
            alpha=np.array(fem_model.alpha, copy=True),
            young=np.array(fem_model.young, copy=True),
        )
    elif env.kind == "beam":
        beam_model = cast("beam.Model", model)
        geometry = BeamPreview(
            edges=np.array(beam_model.topology.edges, copy=True),
            spacing=float(beam_model.topology.h),
        )
    else:
        geometry = ParticlePreview(
            grid_spacing=cast("mpm.Grid", env._grid).spacing,
            particles_per_axis=env.config.particles_per_axis,
        )
    ground = cast("terrain.Terrain", env._terrain)
    return BodyPreviewScene(
        positions=np.array(positions(model, state.body, env.kind), copy=True),
        active=np.array(model.active, copy=True),
        terrain=TerrainPreview(
            normal=np.array(ground.normal, copy=True),
            offset=float(ground.offset),
            centers=np.array(ground.centers, copy=True),
            half_sizes=np.array(ground.half_sizes, copy=True),
            active=np.array(ground.active, copy=True),
        ),
        geometry=geometry,
        length_ref=env.config.length_ref,
        label=f"{env.env_id} | t={float(state.time):.3f}s | {env.dtype}",
    )


def _body_camera(scene: BodyPreviewScene, width: int, height: int) -> _BodyCamera:
    """Fit the inherited camera to active points and the physical reference length."""
    x = scene.positions
    xyz = x if x.shape[-1] == 3 else np.pad(x, ((0, 0), (0, 1)))
    projected = xyz @ _BODY_PROJECTION.T
    visible = projected[scene.active]
    center = (visible.min(0) + visible.max(0)) / 2
    span = max(float(np.max(np.ptp(visible, axis=0))), scene.length_ref)
    scale = min(width * _BODY_FIT_FRACTIONS[0], height * _BODY_FIT_FRACTIONS[1]) / span
    return _BodyCamera(center, scale, width, height)


def _draw_body_terrain(
    draw: ImageDraw.ImageDraw, scene: BodyPreviewScene, camera: _BodyCamera
) -> None:
    """Draw the original sampled plane and rectangular box silhouettes in place."""
    x, t, project = scene.positions, scene.terrain, camera.project
    dim = x.shape[-1]
    for xx in np.linspace(
        float(x[:, 0].min()) - _BODY_GROUND_EXTENT_M[0],
        float(x[:, 0].max()) + _BODY_GROUND_EXTENT_M[1],
        _BODY_GROUND_SAMPLES,
    ):
        yy = (float(t.offset) - float(t.normal[0]) * xx) / float(t.normal[1])
        p = project(np.array([xx, yy] + ([0.0] if dim == 3 else [])))
        draw.line((p, (p[0], camera.height)), fill=(224, 232, 233), width=_BODY_GROUND_STROKE_PX)
    for c, h, on in zip(t.centers, t.half_sizes, t.active):
        if not on:
            continue
        corners = [
            c + np.asarray([sx * h[0], sy * h[1]] + ([0.0] if dim == 3 else []))
            for sx, sy in _QUAD_PERIMETER_SIGNS
        ]
        draw.polygon([project(q) for q in corners], fill=_BLOCK_COLOR)


def _draw_surface(
    draw: ImageDraw.ImageDraw,
    scene: BodyPreviewScene,
    geometry: SurfacePreview,
    camera: _BodyCamera,
) -> None:
    """Draw occupied exposed faces in the original stable depth order."""
    x = scene.positions
    items: list[tuple[float, HostArray, int]] = []
    for element in range(len(geometry.alpha)):
        if geometry.alpha[element] <= 0:
            continue
        for face, indices in enumerate(geometry.faces[element]):
            if (
                geometry.has_neighbor[element, face]
                and geometry.alpha[geometry.neighbors[element, face]] > 0
            ):
                continue
            indices = indices[_TENSOR_FACE_PERIMETER]
            items.append((float(x[indices, 2].mean()), indices, element))
    for _, indices, element in sorted(items, key=lambda item: item[0]):
        shade = _young_shade(float(geometry.young[element]))
        draw.polygon(
            [camera.project(point) for point in x[indices]],
            fill=(int(42 + 95 * shade), 145, 180),
            outline=(28, 63, 79),
        )


def _draw_beams(
    draw: ImageDraw.ImageDraw, scene: BodyPreviewScene, geometry: BeamPreview, camera: _BodyCamera
) -> None:
    """Draw active links followed by cosmetic square node markers."""
    x, active, project = scene.positions, scene.active, camera.project
    for i, j in geometry.edges:
        if active[i] and active[j]:
            draw.line((project(x[i]), project(x[j])), fill=(38, 108, 143), width=_BEAM_STROKE_PX)
    for index, point in enumerate(x):
        if not active[index]:
            continue
        px, py = project(point)
        radius = max(
            _BEAM_MARKER_MIN_RADIUS_PX,
            geometry.spacing * camera.scale * _BEAM_MARKER_SPACING_FRACTION,
        )
        draw.rectangle(
            (px - radius, py - radius, px + radius, py + radius),
            fill=(52, 159, 168),
            outline=(25, 68, 75),
        )


def _draw_particles(
    draw: ImageDraw.ImageDraw,
    scene: BodyPreviewScene,
    geometry: ParticlePreview,
    camera: _BodyCamera,
) -> None:
    """Draw active particles by depth, preserving the original marker radius formula."""
    x = scene.positions
    for index in np.argsort(x[:, 2] if x.shape[-1] == 3 else x[:, 1]):
        if not scene.active[index]:
            continue
        px, py = camera.project(x[index])
        radius = max(
            _PARTICLE_MIN_RADIUS_PX,
            geometry.grid_spacing
            / geometry.particles_per_axis
            * camera.scale
            * _PARTICLE_SPACING_FRACTION,
        )
        draw.ellipse(
            (px - radius, py - radius, px + radius, py + radius),
            fill=(45, 156, 173),
            outline=(34, 100, 126),
        )


def render_body_scene_rgb(
    scene: BodyPreviewScene, width: int = 720, height: int = 420
) -> HostArray:
    """Rasterize a host scene into ``(height, width, 3)`` ``uint8`` RGB pixels.

    This path imports no accelerator or driver modules, performs no transfers,
    and does not modify scene arrays. Pillow is required only for this call.
    Invalid topology or an empty active set raises from the corresponding NumPy
    operation, matching the environment preview's existing preconditions.
    """
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (width, height), _BACKGROUND)
    draw = ImageDraw.Draw(image)
    camera = _body_camera(scene, width, height)
    _draw_body_terrain(draw, scene, camera)
    if isinstance(scene.geometry, SurfacePreview):
        _draw_surface(draw, scene, scene.geometry, camera)
    elif isinstance(scene.geometry, BeamPreview):
        _draw_beams(draw, scene, scene.geometry, camera)
    else:
        _draw_particles(draw, scene, scene.geometry, camera)
    draw.text((12, 12), scene.label, fill=(25, 50, 68))
    draw.text((12, height - 24), "Development renderer | terrain contact only", fill=(25, 50, 68))
    return np.asarray(image)
