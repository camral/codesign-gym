"""SoftWalker 3D frames: the live body drawn with Andy's NVIDIA Newton studio viewer (``viewer/view_newton.py``).

``SoftWalkerVectorEnv.render`` uses this by default and falls back to the Pillow side view (``render.py``) when no
OpenGL context can be opened (e.g. a headless Linux GPU box without EGL; see the README). Andy's viewer is imported
unchanged and extended here: the live state is wrapped as a one-frame recording, the camera tracks the body, and the
stage is restyled (blue checkered floor, black pits under gap platforms, plain grey stairs, white beam torso, touching
MPM particle markers). Warp's default device is left alone (the viewer's ``set_device("cuda:0")`` is skipped).
"""

from __future__ import annotations

import importlib
import math
import os
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from . import config as C
from . import embodiment as E

WIDTH, HEIGHT = 1280, 800
FLOOR = ((.2, .3, .4), (.3, .4, .5))  # checker pair
PIT, STAIRS = (0., 0., 0.), (.42, .47, .52)
BEAM_TORSO = (1., 1., 1.)
CHECK = .05                           # m, checker size
MARKER = .66                          # MPM marker radius / particle spacing: neighbouring spheres touch
CAMERA = {"flat": (-62., 22.), "stairs": (-75., 20.), "gaps": (-90., 8.)}  # (azimuth, elevation) degrees
EGL_HINT = ("install EGL for headless OpenGL matching the NVIDIA driver, e.g. `sudo apt-get install -y libegl1 libnvidia-gl-595-server` "
            "(see src/envs/soft_walker/README.md)")

_viewer, _unavailable, _anchor = None, None, None


def viewer_module():
    """Andy's ``viewer/view_newton.py``, imported once as ``envs.soft_walker.viewer.view_newton`` (its ``.recording``
    import needs the package)."""
    global _viewer
    if _viewer is None:
        _viewer = importlib.import_module("envs.soft_walker.viewer.view_newton")
    return _viewer


def terrain_name(terrain: dict | None) -> str:
    if terrain is None:
        return "flat"
    return "gaps" if float(terrain["offset"]) < 0 else "stairs"


def topology(driver: str, fields: dict, design: np.ndarray) -> dict:
    """Viewer sites for one resolved design: active mask, torso (top occupied cell row) and the drawn connectivity."""
    d, g = C.dim_of(driver), C.grid(driver)
    ref = np.asarray(fields["reference"], float)
    out = {}
    if driver.startswith("mpm"):
        active = np.asarray(fields["alpha"]) > 0
    elif driver == "fem":  # hexahedra with alpha > 0; their outward boundary quads
        hexes = np.asarray(fields["entities"])[np.asarray(fields["alpha"]) > 0]
        active = np.zeros(len(ref), bool)
        active[np.unique(hexes)] = True
        out["faces"] = _hex_faces(ref, hexes)
    elif driver == "spring":  # empty cells: cell_alpha 0; their corner nodes are massless unless shared
        active = np.asarray(fields["mass"]) > 0
        out["cells"] = np.asarray(fields["cells"])[np.asarray(fields["cell_alpha"]) > 0].astype(np.int32)
    else:  # beam: link alpha is 1 everywhere; empty cells are massless nodes
        active = np.asarray(fields["mass"]) > 0
        links = np.asarray(fields["entities"])
        out["edges"] = links[active[links].all(1)].astype(np.int32)
    rows = np.flatnonzero((E.to_cells(design, d)["occupancy"] >= E.OCCUPANCY_THRESHOLD).reshape(g).any(axis=tuple(a for a in range(d) if a != 1)))
    torso = active & (ref[:, 1] >= C.origin(driver)[1] + rows.max() * C.CELL - 1e-9)
    return dict(active=active, torso=torso, **out)


def _hex_faces(ref: np.ndarray, hexes: np.ndarray) -> np.ndarray:
    """Outward boundary quads of hexahedra, from node coordinates (faces shared by two hexes are interior)."""
    faces, counts = {}, {}
    for h in hexes:
        p, centre = ref[h], ref[h].mean(0)
        for axis in range(3):
            for side in (p[:, axis].min(), p[:, axis].max()):
                f = [n for n, q in zip(h, p) if np.isclose(q[axis], side)]
                q = ref[f] - ref[f].mean(0)
                a, b = [k for k in range(3) if k != axis]
                f = [f[i] for i in np.argsort(np.arctan2(q[:, b], q[:, a]))]
                r = ref[f]
                if np.dot(np.cross(r[1] - r[0], r[2] - r[0]), r.mean(0) - centre) < 0:
                    f.reverse()
                key = tuple(sorted(f))
                faces[key], counts[key] = f, counts.get(key, 0) + 1
    return np.array([f for k, f in faces.items() if counts[k] == 1], np.int32).reshape(-1, 4)


class NewtonRenderer:
    """One lane's Newton studio for one design; ``draw(positions)`` returns an RGB frame (HEIGHT, WIDTH, 3)."""

    def __init__(self, driver: str, fields: dict, design: np.ndarray, terrain: dict | None, *, width: int = WIDTH, height: int = HEIGHT, human: bool = False):
        if human and sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            raise RuntimeError('render_mode="human" requires a display')
        if not human and sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
            import pyglet
            pyglet.options["headless"] = True  # EGL, no X server
        V = viewer_module()
        self.driver, self.width, self.height, name = driver, width, height, terrain_name(terrain)
        self.recording = _live_recording(V, driver, fields, design, terrain)
        particles = driver.startswith("mpm")
        azimuth, elevation = CAMERA[name]
        args = SimpleNamespace(width=width, height=height, gif=not human, thickness=.009, particles=particles, elements=False, no_surface=particles,
                               marker_radius=MARKER * C.CELL / C.PARTICLES_PER_AXIS if particles else None, display_scale=10.,
                               azimuth=azimuth, elevation=elevation, max_frames=None, fps=25.)
        skin = V.Skin(self.recording, args.thickness)
        torso, V.COLORS[0] = V.COLORS[0], BEAM_TORSO if driver == "beam" else V.COLORS[0]
        try:
            self.studio = _studio_class(V)(self.recording, skin, args)
        finally:
            V.COLORS[0] = torso
        self.origin_x = float(self.recording.world[0, self.recording.active, 0].mean())

    def draw(self, positions: np.ndarray, label: str = "", *, capture: bool = True) -> np.ndarray | None:
        V, rec = viewer_module(), self.recording
        rec.positions[0] = np.asarray(positions, float)
        rec.world[0] = V.world_coordinates(rec.positions[0])
        self.studio.track = float(rec.world[0, rec.active, 0].mean()) - self.origin_x
        self.studio.render(0)
        if not capture:
            return None
        from PIL import Image, ImageDraw
        image = Image.fromarray(np.ascontiguousarray(self.studio.viewer.get_frame().numpy()[..., :3]))
        if image.size != (self.width, self.height):  # HiDPI framebuffers are larger than the window
            image = image.resize((self.width, self.height), Image.LANCZOS)
        if label:
            ImageDraw.Draw(image).text((12, 10), label, fill=(20, 32, 48))
        return np.asarray(image)

    def close(self):
        self.studio.viewer.close()


def _live_recording(V, driver: str, fields: dict, design: np.ndarray, terrain: dict | None):
    """A one-frame ``view_newton.Recording`` over the resolved design; ``draw`` overwrites its frame."""
    rec = V.Recording.__new__(V.Recording)
    ref = np.asarray(fields["reference"], float)
    topo = topology(driver, fields, design)
    rec.arrays = {k: v for k, v in topo.items() if k not in ("active", "torso")}
    rec.reference, rec.active, rec.torso = ref, topo["active"], topo["torso"]
    rec.positions, rec.metrics, rec.times = ref[None].copy(), np.zeros((1, 7)), np.zeros(1)
    rec.terrain = None if terrain is None else {k: np.asarray(v).tolist() for k, v in terrain.items()}
    rec.driver, rec.dim = driver, ref.shape[1]
    rec.protocol, rec.assessment, rec.flight_assessment, rec.objective = {}, {}, {}, {}
    rec.source_count, rec.excluded_reason, rec.directory = 1, None, Path(".")
    rec.world = V.world_coordinates(rec.positions)
    visible = rec.world[:, rec.active].reshape(-1, 3)
    rec.lo, rec.hi = visible.min(0), visible.max(0)
    return rec


_studios = {}


def _studio_class(V):
    if V in _studios:
        return _studios[V]

    class StyledStudio(V.Studio):
        """Andy's Studio with a body-tracking camera, the codesign-gym stage styling and Warp's default device untouched."""
        track = 0.

        def __init__(self, recording, skin, args):
            import warp as wp
            set_device, wp.set_device = wp.set_device, lambda *a, **k: None
            try:
                super().__init__(recording, skin, args)
            finally:
                wp.set_device = set_device

        def reset_camera(self):
            from pyglet.math import Vec3
            position = self.camera_position + np.array([self.track * self.scale, 0., 0.])
            self.viewer.set_camera(Vec3(*position), self.camera_pitch, self.camera_yaw)

        def add_ground(self):
            wp, dev, t = self.wp, self.viewer.device, self.rec.terrain
            colour = lambda rgb: wp.array([rgb], dtype=wp.vec3, device=dev)
            matte = wp.array([[.96, 0., 0., 0.]], dtype=wp.vec4, device=dev)
            pit = t is not None and float(t.get("offset", 0.)) < 0  # plane under gap platforms: a black void
            xmid = (self.rec.lo[0] + self.rec.hi[0]) * .5
            extent = max(2., (self.rec.hi[0] - self.rec.lo[0]) * 8.)
            xs = np.arange(math.floor((xmid - extent) / CHECK), math.ceil((xmid + extent) / CHECK) + 1) * CHECK
            zs = np.arange(-math.ceil(extent / CHECK), math.ceil(extent / CHECK) + 1) * CHECK
            x, z = np.meshgrid(xs, zs, indexing="ij")
            y = V.terrain_height(x, z, dict(t, active=[]) if t else None) - .00015
            points = V.world_coordinates(np.stack((x, y, z), -1).reshape(-1, 3))
            for k, cells in enumerate(_checkers(xs, zs)):
                self.mesh(f"/stage_{k}", points, cells, colour(PIT if pit else FLOOR[k]), matte)
            for index, (c, b, r, active) in enumerate(zip(*(t.get(key, []) for key in ("centers", "half_sizes", "rounding", "active"))) if t else ()):
                if not active:
                    continue
                vertices, faces = V.rounded_box_surface(c, b, r, extent)
                self.mesh(f"/terrain/block_{index}", vertices, faces, colour(PIT if pit else STAIRS), matte)
                if pit:  # gap platforms are the floor: checker their tops on the stage lattice
                    self._checker_top(V, index, c, b, r, extent, matte, colour)

        def _checker_top(self, V, index, c, b, r, extent, matte, colour):
            top, x0, x1 = c[1] + b[1] + .0002, c[0] - b[0] + r, c[0] + b[0] - r
            zh = (b[2] if len(b) > 2 else extent) - r  # 2D boxes span the stage depth (rounded_box_surface)
            grid = lambda lo, hi: np.unique(np.r_[lo, np.arange(math.ceil(lo / CHECK), math.floor(hi / CHECK) + 1) * CHECK, hi])
            xs, zs = grid(x0, x1), grid(-zh, zh)
            x, z = np.meshgrid(xs, zs, indexing="ij")
            points = V.world_coordinates(np.stack((x, np.full(x.shape, top), z), -1).reshape(-1, 3))
            for k, cells in enumerate(_checkers(xs, zs)):
                if cells:
                    self.mesh(f"/terrain/top_{index}_{k}", points, cells, colour(FLOOR[k]), matte)

    _studios[V] = StyledStudio
    return StyledStudio


def _checkers(xs: np.ndarray, zs: np.ndarray) -> tuple[list, list]:
    """Triangle indices of an (xs, zs) vertex lattice, split by global checker parity."""
    cells = ([], [])
    for i in range(len(xs) - 1):
        for j in range(len(zs) - 1):
            a = i * len(zs) + j; b = a + len(zs)
            parity = (math.floor((xs[i] + xs[i + 1]) / (2 * CHECK)) + math.floor((zs[j] + zs[j + 1]) / (2 * CHECK))) % 2
            cells[parity].extend((a, b + 1, b, a, a + 1, b + 1))
    return cells


def make_renderer(driver: str, fields: dict, design: np.ndarray, terrain: dict | None, *, human: bool = False) -> NewtonRenderer | None:
    """A Newton renderer, or None (warning once per process) when Newton or an OpenGL context is unavailable."""
    global _unavailable, _anchor
    if _unavailable is not None and not human:
        return None
    try:
        renderer = NewtonRenderer(driver, fields, design, terrain, human=human)
        if _anchor is None:  # Newton's GL objects live in pyglet's shared context, which dies with the last window
            import pyglet
            _anchor = pyglet.window.Window(width=1, height=1, visible=False)
        return renderer
    except Exception as exc:  # no GL context / EGL, no display, missing package: fall back to the 2D renderer
        if human:
            raise
        root = exc
        while root.__cause__ is not None or root.__context__ is not None:  # Newton re-raises pyglet's EGL ImportError as a generic hint
            root = root.__cause__ or root.__context__
        _unavailable = f"{type(root).__name__}: {root}"
        warnings.warn(f"SoftWalker 3D (Newton) rendering unavailable ({_unavailable}); using the 2D Pillow side view. To enable it, {EGL_HINT}.",
                      RuntimeWarning, stacklevel=3)
        return None
