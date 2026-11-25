"""Saved simulation samples and display geometry, independent of the viewer.

Coordinates and connectivity originate in saved files. Extruded surfaces and
particle marker radii are display geometry, never a new physics discretization.
Loading this module needs NumPy but neither Newton, Warp, nor JAX.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

# Standalone viewer scripts import recording without a package installation.
if TYPE_CHECKING or __package__:
    from ..contracts.metrics import METRIC_DX, METRIC_STATUS, METRIC_TIME
    from ..host_terrain import height_field
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from contracts.metrics import METRIC_DX, METRIC_STATUS, METRIC_TIME
    from host_terrain import height_field

COLORS = [
    (0.16, 0.22, 0.29),
    (0.09, 0.59, 0.59),
    (0.91, 0.43, 0.19),
    (0.31, 0.48, 0.78),
    (0.71, 0.39, 0.62),
]


# Recordings carry floating coordinates and integer connectivity with saved dtypes.
HostArray: TypeAlias = "NDArray[Any]"


# Display reconstruction policies inherited from the original recorder. These
# are not solver tolerances; the original visual calibration was not recorded.
REFERENCE_DECIMAL_PLACES = 10  # m coordinates rounded to a 0.1 nm lattice key.
MIN_DISPLAY_SPACING = 1e-9  # m; discard sub-nanometre gaps before choosing markers.
PARTICLE_RADIUS_FRACTION = 0.18  # Fraction of the smallest retained lattice spacing.
FALLBACK_MARKER_RADIUS = 0.0015  # m; used when spacing cannot be inferred.
ELEMENT_MARKER_RADIUS_FRACTION = 0.45  # Relative to the chosen particle marker radius.
BEAM_MIN_LENGTH = 1e-12  # m; reject coincident saved endpoints before normalization.
BEAM_AXIS_ALIGNMENT_LIMIT = 0.9  # Unit-vector component; choose a nonparallel helper.
BEAM_MIN_RADIUS = 0.0015  # m, cosmetic minimum independent of beam mechanics.
BEAM_RADIAL_SEGMENTS = 8  # Octagonal display cylinders, including matching cap indices.
MIN_HEIGHT_FIELD_NORMAL = 1e-10  # Dimensionless vertical normal component.
BOX_FACE_SAMPLES = 13  # Samples per face axis before adding the rounded-core boundaries.
BOX_DIRECTION_NORM_FLOOR = 1e-30  # m; protects normalization at a coincident core point.


def read_json(path: Path, default: Any = None) -> Any:
    """Read an optional JSON artifact, returning default only when the path is absent.

    Malformed JSON and I/O errors propagate. Historical artifact schemas are heterogeneous."""
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def digest(path: Path) -> str:
    """Hash file bytes for provenance; propagate I/O failures."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def world_coordinates(points: ArrayLike) -> HostArray:
    """Rotate physical Y-up into viewer Z-up, preserving handedness in 3D."""
    points = np.asarray(points)
    if points.shape[-1] == 2:
        return np.stack((points[..., 0], np.zeros(points.shape[:-1]), points[..., 1]), axis=-1)
    return np.stack((points[..., 0], -points[..., 2], points[..., 1]), axis=-1)


def recording_metric_indices(directory: Path, width: int) -> dict[str, int | None]:
    """Resolve saved column names, retaining positional fallback only without metadata.

    Scenario sidecars take precedence over locomotion replay metadata. Explicit
    malformed or ambiguous columns fail clearly instead of selecting another metric.
    Historical recordings without column metadata retain time/distance/status slots.
    """
    columns = None
    declared = False
    sidecar = directory / "metric_columns.json"
    if sidecar.exists():
        columns, declared = read_json(sidecar), True
    else:
        replay = read_json(directory / "replay.json", {})
        if not isinstance(replay, dict):
            raise ValueError("replay.json must contain an object")
        if "columns" in replay:
            columns, declared = replay["columns"], True
    if not declared:
        if width <= METRIC_TIME:
            raise ValueError("Legacy metrics require a physical time column")
        return {
            "time": METRIC_TIME,
            "distance_x": METRIC_DX if width > METRIC_DX else None,
            "status": METRIC_STATUS if width > METRIC_STATUS else None,
        }
    if (
        not isinstance(columns, list)
        or any(not isinstance(name, str) or not name for name in columns)
        or len(columns) != width
        or len(set(columns)) != width
    ):
        raise ValueError("Metric column metadata must contain one unique name per saved column")
    result: dict[str, int | None] = {}
    for semantic, aliases in (
        ("time", ("time_s",)),
        ("distance_x", ("dx_m", "distance_x_m")),
        ("status", ("status",)),
    ):
        matches = [columns.index(name) for name in aliases if name in columns]
        if len(matches) > 1 or (not matches and semantic != "status"):
            raise ValueError(f"Metric metadata has missing or ambiguous {semantic} columns")
        result[semantic] = matches[0] if matches else None
    return result


class Recording:
    """Owned host copies of a saved trajectory and its valid renderable prefix.

    Positions have shape (frames, sites, 2|3) in metres; metrics[:, 0] is
    physical time in seconds in legacy files. Named metadata controls newer files.
    Invalid trailing samples are excluded, never repaired."""

    def __init__(self, directory: str | Path) -> None:
        """Load NPZ/JSON files without pickle; reject missing geometry or an empty valid prefix."""
        self.directory = Path(directory).resolve()
        self.path = self.directory / "trajectory.npz"
        with np.load(self.path, allow_pickle=False) as saved:
            self.arrays = {key: saved[key] for key in saved.files}
        if (self.directory / "topology.npz").exists():
            with np.load(self.directory / "topology.npz", allow_pickle=False) as saved:
                for key in saved.files:
                    self.arrays.setdefault(key, saved[key])
        a = self.arrays
        self.positions, self.metrics = a["positions"], a["metrics"]
        if self.metrics.ndim != 2:
            raise ValueError("Expected metrics[frame, column]")
        self.metric_indices = recording_metric_indices(self.directory, self.metrics.shape[1])
        time_index = self.metric_indices["time"]
        status_index = self.metric_indices["status"]
        assert time_index is not None
        if self.positions.ndim != 3 or self.positions.shape[-1] not in (2, 3):
            raise ValueError("Expected positions[frame, site, dimension=2 or 3]")
        if len(self.metrics) != len(self.positions) or not len(self.positions):
            raise ValueError("Positions and metrics need matching, nonempty time axes")
        self.reference = np.asarray(a["reference"])
        self.active = np.asarray(a["active"], bool)
        self.torso = np.asarray(a.get("torso", np.zeros(len(self.active))), bool)
        if self.reference.shape != self.positions.shape[1:] or not self.active.any():
            raise ValueError("Reference coordinates/active mask do not match saved positions")
        self.protocol = read_json(self.directory / "case.json") or read_json(
            self.directory / "protocol.json", {}
        )
        self.assessment = read_json(self.directory / "assessment.json", {})
        self.flight_assessment = read_json(self.directory / "flight_assessment.json", {})
        self.objective = read_json(self.directory / "objective.json", {})
        self.terrain = read_json(self.directory / "terrain.json")
        if self.terrain is None and self.protocol.get("terrain") not in (None, "flat"):
            raise ValueError("The recording declares nonflat terrain but terrain.json is missing")
        self.driver = str(self.protocol.get("driver", "saved robot"))
        self.source_count = len(self.positions)
        self.excluded_reason = None
        stop = self.source_count
        for k in range(self.source_count):
            row = self.metrics[k]
            if not np.isfinite(self.positions[k, self.active]).all() or not np.isfinite(
                row[time_index]
            ):
                stop, self.excluded_reason = k, "nonfinite saved positions/time"
                break
            if status_index is not None and row[status_index] != 0:
                stop, self.excluded_reason = k, "nonzero saved solver status"
                break
            if k and row[time_index] <= self.metrics[k - 1, time_index]:
                stop, self.excluded_reason = k, "non-increasing physical time"
                break
        if stop < 1:
            raise ValueError("No valid saved sample is available to render")
        self.positions, self.metrics = self.positions[:stop], self.metrics[:stop]
        self.times = self.metrics[:, time_index]
        self.world = world_coordinates(self.positions)
        visible = self.world[:, self.active].reshape(-1, 3)
        self.lo, self.hi = visible.min(0), visible.max(0)
        self.dim = self.positions.shape[-1]

    @property
    def displacements_x(self) -> HostArray:
        """Read saved horizontal displacement when that metric exists."""
        index = self.metric_indices["distance_x"]
        if index is None:
            raise ValueError("Saved recording has no horizontal displacement column")
        return self.metrics[:, index]

    def group(self, nodes: ArrayLike) -> int:
        """Return an anatomical display-color index for the supplied saved site indices."""
        nodes = np.asarray(nodes, dtype=int)
        if self.torso[nodes].mean() >= 0.5:
            return 0
        q = self.reference[nodes].mean(0)
        mid = (self.reference.min(0) + self.reference.max(0)) * 0.5
        return 1 + int(q[0] > mid[0]) + (2 * int(q[2] > mid[2]) if self.dim == 3 else 0)

    def verdict(self) -> str:
        """Describe the saved assessment; rendering does not qualify locomotion."""
        if self.excluded_reason:
            return "Failure preview: valid prefix"
        if self.objective.get("name") == "hop":
            return (
                "Repeated flight observed"
                if self.flight_assessment.get("repeated_airborne_observed")
                else "Pronk candidate / repeated flight not observed"
            )
        return (
            "Walking gate passed"
            if self.assessment.get("walking_gate_passed")
            else "Candidate / walking not qualified"
        )

    def frame_indices(self, fps: float, maximum: int | None = None) -> HostArray:
        """Select existing samples nearest a requested FPS, including both endpoints.

        No interpolation changes positions or timestamps. maximum limits exported samples."""
        targets = np.arange(self.times[0], self.times[-1] + 0.5 / fps, 1 / fps)
        indices = np.unique(
            np.r_[
                0,
                np.searchsorted(self.times, targets, side="left").clip(0, len(self.times) - 1),
                len(self.times) - 1,
            ]
        )
        if maximum is not None and len(indices) > maximum:
            indices = indices[
                np.unique(np.round(np.linspace(0, len(indices) - 1, maximum)).astype(int))
            ]
        return indices


def lattice_cells(reference: HostArray, active: HostArray) -> list[list[int]]:
    """Connect only adjacent Cartesian reference sites whose corners are active."""
    dim = reference.shape[1]
    axes = [np.unique(np.round(reference[:, d], REFERENCE_DECIMAL_PLACES)) for d in range(dim)]
    coordinates = np.array(
        [
            np.searchsorted(axis, np.round(reference[:, d], REFERENCE_DECIMAL_PLACES))
            for d, axis in enumerate(axes)
        ]
    ).T
    lookup = {tuple(c): i for i, c in enumerate(coordinates)}
    if len(lookup) != len(reference) or np.prod([len(a) for a in axes]) != len(reference):
        raise ValueError(
            "No saved connectivity and reference sites are not a complete Cartesian lattice"
        )
    offsets = np.array(list(np.ndindex(*(2,) * dim)))
    result = []
    for origin in np.ndindex(*(len(a) - 1 for a in axes)):
        nodes = [lookup[tuple(np.asarray(origin) + off)] for off in offsets]
        if active[nodes].all():
            result.append(nodes)
    return result


def ordered_polygon(nodes: ArrayLike, reference: HostArray) -> list[int]:
    """Order 2D site indices around their reference centroid for display triangulation."""
    nodes = np.asarray(nodes, dtype=int)
    q = reference[nodes, :2]
    center = q.mean(0)
    return nodes[np.argsort(np.arctan2(q[:, 1] - center[1], q[:, 0] - center[0]))].tolist()


class DetailGeometry:
    """Recorded particles/connectivity only; never reconstruct solver elements."""

    def __init__(self, recording: Recording, marker_radius: float | None = None) -> None:
        """Extract saved connectivity or MPM particle markers; reject invalid site indices.

        marker_radius is a cosmetic radius in metres, independent of the physics model."""
        self.rec = recording
        self.particles = recording.driver.lower().startswith("mpm")
        self.label = (
            "Particles"
            if self.particles
            else ("Surface elements" if "faces" in recording.arrays else "Elements")
        )
        self.active_ids = np.flatnonzero(recording.active)
        spacing = [
            np.diff(np.unique(np.round(recording.reference[:, d], REFERENCE_DECIMAL_PLACES)))
            for d in range(recording.dim)
        ]
        spacing = np.concatenate([v[v > MIN_DISPLAY_SPACING] for v in spacing])
        self.radius = float(
            marker_radius
            if marker_radius is not None
            else (
                PARTICLE_RADIUS_FRACTION * spacing.min() if len(spacing) else FALLBACK_MARKER_RADIUS
            )
        )
        links = set()

        def add(a: int, b: int) -> None:
            a, b = int(a), int(b)
            if min(a, b) < 0 or max(a, b) >= len(recording.active):
                raise ValueError("Saved connectivity contains an out-of-range site")
            if a != b and recording.active[[a, b]].all():
                links.add(tuple(sorted((a, b))))

        if self.particles:
            self.description = "Recorded active MPM particle centers; sphere radius is cosmetic"
        elif "faces" in recording.arrays:
            for face in recording.arrays["faces"]:
                for i, a in enumerate(face):
                    add(a, face[(i + 1) % len(face)])
            self.description = (
                "Recorded boundary-face perimeters; interior FEM elements are unavailable"
            )
        else:
            for a, b in recording.arrays.get("edges", []):
                add(a, b)
            for cell in recording.arrays.get("cells", []):
                for i, a in enumerate(cell):
                    add(a, cell[(i + 1) % len(cell)])
            self.description = (
                "Recorded links and cell perimeters; no inferred diagonals or elements"
            )
        self.edges: NDArray[np.int32] = np.asarray(sorted(links), dtype=np.int32).reshape(-1, 2)
        self.node_ids = self.active_ids if self.particles else np.unique(self.edges)
        self.node_colors: HostArray = np.asarray(
            [COLORS[recording.group([n])] for n in self.node_ids], np.float32
        ).reshape(-1, 3)
        edge_colors: list[tuple[float, float, float]] = [
            COLORS[recording.group(self.edges[index])] for index in range(len(self.edges))
        ]
        self.edge_colors: HostArray = np.asarray(edge_colors, np.float32).reshape(-1, 3)
        self.available = self.particles or bool(len(self.edges))

    def provenance(self) -> dict[str, Any]:
        """Return JSON-compatible metadata distinguishing solver connectivity from display markers."""
        return {
            "label": self.label,
            "description": self.description,
            "active_saved_sites": len(self.active_ids),
            "particle_count": len(self.node_ids) if self.particles else 0,
            "unique_edge_count": len(self.edges),
            "connected_node_count": len(self.node_ids) if not self.particles else 0,
            "particle_marker_radius_m": self.radius if self.particles else None,
            "element_node_marker_radius_m": self.radius * ELEMENT_MARKER_RADIUS_FRACTION
            if not self.particles
            else None,
            "marker_radius_is_cosmetic": True,
            "position_offset_m": 0.0,
            "source_connectivity_sha256": hashlib.sha256(
                self.edges.astype("<i4").tobytes()
            ).hexdigest()
            if len(self.edges)
            else None,
            "fem_interior_elements_available": False if "faces" in self.rec.arrays else None,
        }


class Skin:
    """Display-only extrusions and surfaces evaluated at saved sample positions."""

    def __init__(self, recording: Recording, thickness: float) -> None:
        """Build fixed display connectivity from saved geometry; reject an empty surface.

        Two-dimensional thickness is in metres. This never changes solver elements."""
        self.recording = recording
        self.thickness = thickness if recording.dim == 2 else 0.0
        self.parts: dict[int, tuple[HostArray, HostArray, HostArray]] = {}
        self.beam = (
            "edges" in recording.arrays
            and "cells" not in recording.arrays
            and recording.driver == "beam"
        )
        self.edges = np.asarray(recording.arrays.get("edges", []), dtype=int)
        faces = []
        if self.beam:
            self.description = "Cylinders following saved beam edges; radius is display-only"
            return
        if recording.dim == 2:
            cells = recording.arrays.get("cells")
            if cells is None:
                cells = lattice_cells(recording.reference, recording.active)
                self.description = "Thin extruded skin on adjacent active reference-lattice sites; not solver elements"
            else:
                self.description = "Thin extrusion of saved spring cells; thickness is display-only"
            polygons = [ordered_polygon(c, recording.reference) for c in cells]
            polygons = [p for p in polygons if recording.active[p].all()]
            counts = Counter(
                tuple(sorted((p[i], p[(i + 1) % len(p)]))) for p in polygons for i in range(len(p))
            )
            for p in polygons:
                group = recording.group(p)
                faces.append((p, [-thickness / 2] * len(p), group))
                faces.append((p[::-1], [thickness / 2] * len(p), group))
                for i, a in enumerate(p):
                    b = p[(i + 1) % len(p)]
                    if counts[tuple(sorted((a, b)))] == 1:
                        faces.append(
                            (
                                [a, a, b, b],
                                [-thickness / 2, thickness / 2, thickness / 2, -thickness / 2],
                                group,
                            )
                        )
        elif "faces" in recording.arrays:
            self.description = "Saved FEM boundary faces at saved vertex positions"
            faces = [
                (list(f), [0.0] * len(f), recording.group(f)) for f in recording.arrays["faces"]
            ]
        else:
            self.description = (
                "Boundary skin of adjacent active 3D reference-lattice sites; not solver elements"
            )
            cells = lattice_cells(recording.reference, recording.active)
            all_faces = {}
            counts = Counter()
            # np.ndindex order is 000,001,010,011,100,101,110,111.
            patterns = (
                (0, 1, 3, 2),
                (4, 6, 7, 5),
                (0, 4, 5, 1),
                (2, 3, 7, 6),
                (0, 2, 6, 4),
                (1, 5, 7, 3),
            )
            for cell in cells:
                center = recording.reference[cell].mean(0)
                for pattern in patterns:
                    f = [cell[i] for i in pattern]
                    p = recording.reference[f]
                    if np.dot(np.cross(p[1] - p[0], p[2] - p[0]), p.mean(0) - center) < 0:
                        f.reverse()
                    key = tuple(sorted(f))
                    all_faces[key] = f
                    counts[key] += 1
            faces = [
                (f, [0.0] * len(f), recording.group(f))
                for key, f in all_faces.items()
                if counts[key] == 1
            ]
        by_group = defaultdict(list)
        for face_nodes, face_depth, group in faces:
            by_group[group].append((face_nodes, face_depth))
        for group, group_faces in by_group.items():
            nodes: list[int] = []
            depth: list[float] = []
            triangles: list[tuple[int, int, int]] = []
            for polygon, offsets in group_faces:
                base = len(nodes)
                nodes.extend(polygon)
                depth.extend(offsets)
                triangles.extend((base, base + j, base + j + 1) for j in range(1, len(polygon) - 1))
            self.parts[group] = (
                np.asarray(nodes),
                np.asarray(depth),
                np.asarray(triangles, np.int32).reshape(-1),
            )
        if not self.parts:
            raise ValueError("Saved active geometry has no connected surface to display")

    def geometry(self, frame: int) -> dict[int, tuple[HostArray, HostArray]]:
        """Return group-to-(vertices, triangle indices) host arrays for one saved frame.

        Coordinates are Z-up in metres; vertices are fresh copies. Degenerate beams fail."""
        if not self.beam:
            result = {}
            for group, (nodes, depth, indices) in self.parts.items():
                p = self.recording.world[frame, nodes].copy()
                p[:, 1] += depth
                result[group] = p, indices
            return result
        groups: dict[int, tuple[list[HostArray], list[tuple[int, int, int]]]] = defaultdict(
            lambda: ([], [])
        )
        for a, b in self.edges:
            if not self.recording.active[[a, b]].all():
                continue
            pa, pb = self.recording.world[frame, [a, b]]
            delta = pb - pa
            length = np.linalg.norm(delta)
            if length < BEAM_MIN_LENGTH:
                raise ValueError("Degenerate saved beam edge")
            axis = delta / length
            helper = (
                np.array([0.0, 0.0, 1.0])
                if abs(axis[2]) < BEAM_AXIS_ALIGNMENT_LIMIT
                else np.array([0.0, 1.0, 0.0])
            )
            u = np.cross(axis, helper)
            u /= np.linalg.norm(u)
            v = np.cross(axis, u)
            radius = max(self.thickness / 2, BEAM_MIN_RADIUS)
            ring = np.array(
                [
                    radius * (math.cos(t) * u + math.sin(t) * v)
                    for t in np.arange(BEAM_RADIAL_SEGMENTS) * (2 * math.pi / BEAM_RADIAL_SEGMENTS)
                ]
            )
            points, triangles = groups[self.recording.group([a, b])]
            base = len(points)
            points.extend(pa + ring)
            points.extend(pb + ring)
            for j in range(BEAM_RADIAL_SEGMENTS):
                k = (j + 1) % BEAM_RADIAL_SEGMENTS
                triangles.extend(
                    (
                        (base + j, base + k, base + BEAM_RADIAL_SEGMENTS + k),
                        (
                            base + j,
                            base + BEAM_RADIAL_SEGMENTS + k,
                            base + BEAM_RADIAL_SEGMENTS + j,
                        ),
                    )
                )
            triangles.extend(
                (base, base + j + 1, base + j) for j in range(1, BEAM_RADIAL_SEGMENTS - 1)
            )
            triangles.extend(
                (
                    base + BEAM_RADIAL_SEGMENTS,
                    base + BEAM_RADIAL_SEGMENTS + j,
                    base + BEAM_RADIAL_SEGMENTS + j + 1,
                )
                for j in range(1, BEAM_RADIAL_SEGMENTS - 1)
            )
        return {
            g: (np.asarray(p), np.asarray(t, np.int32).reshape(-1)) for g, (p, t) in groups.items()
        }


def terrain_height(x: HostArray, z: HostArray, terrain: dict[str, Any] | None) -> HostArray:
    """Evaluate the saved plane and rounded obstacles as a display height field.

    x and z are horizontal host coordinates in metres. Reject a vertical base plane."""
    if terrain is None:
        return np.zeros_like(x)
    normal = np.asarray(terrain.get("normal", [0.0, 1.0, 0.0]), float)
    offset = float(terrain.get("offset", 0.0))
    if abs(normal[1]) < MIN_HEIGHT_FIELD_NORMAL:
        raise ValueError("Terrain normal is not a height field")
    return height_field(
        x,
        z,
        normal,
        offset,
        zip(
            terrain.get("centers", []),
            terrain.get("half_sizes", []),
            terrain.get("rounding", []),
            terrain.get("active", []),
        ),
    )


def rounded_box_surface(
    center: ArrayLike, half_size: ArrayLike, radius: float, depth: float
) -> tuple[HostArray, HostArray]:
    """Static rounded-box display mesh; separate walls never form fake ramps."""
    center, half_size = np.asarray(center, float), np.asarray(half_size, float)
    if len(center) == 2:
        center = np.r_[center, 0.0]
        half_size = np.r_[half_size, depth]
    vertices: list[HostArray] = []
    indices: list[int] = []
    core = np.maximum(half_size - radius, 0.0)
    for axis in range(3):
        other = [d for d in range(3) if d != axis]
        for sign in (-1.0, 1.0):
            grids = [
                np.unique(
                    np.r_[
                        np.linspace(-half_size[d], half_size[d], BOX_FACE_SAMPLES),
                        -core[d],
                        core[d],
                    ]
                )
                for d in other
            ]
            base = len(vertices)
            for a in grids[0]:
                for b in grids[1]:
                    p = np.zeros(3)
                    p[axis] = sign * half_size[axis]
                    p[other] = (a, b)
                    if radius > 0:
                        q = np.clip(p, -core, core)
                        delta = p - q
                        p = q + radius * delta / max(
                            np.linalg.norm(delta), BOX_DIRECTION_NORM_FLOOR
                        )
                    vertices.append(p + center)
            for i in range(len(grids[0]) - 1):
                for j in range(len(grids[1]) - 1):
                    a = base + i * len(grids[1]) + j
                    b = a + len(grids[1])
                    indices.extend((a, b, b + 1, a, b + 1, a + 1))
    return world_coordinates(np.asarray(vertices)), np.asarray(indices, np.int32)
