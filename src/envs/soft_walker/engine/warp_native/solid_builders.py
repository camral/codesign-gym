"""Readable NumPy-only construction of regular native Lagrangian bodies.

NumPy is used only to validate/instantiate a design at a reset boundary. Forces,
Newton/Krylov iterations, time stepping, and reverse differentiation use Warp.
These are the same topology and reference-quadrature conventions as JAX.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

    from numpy.typing import ArrayLike, DTypeLike

    from .host_types import FieldArrays, HostArray

from itertools import product

import numpy as np

from ...contracts.fields import POISSON_DEFAULT
from ..physics.construction_defaults import (
    BODY_DENSITY_KG_M3,
    EARTH_GRAVITY_M_S2,
    PLANAR_THICKNESS_M,
    REGULAR_SOLID_STRENGTH,
    REGULAR_SOLID_YOUNG_PA,
    SOLID_CONTACT_PRESSURE_PER_LENGTH_PA_M,
    SPRING_CONTACT_MODULUS_PA,
    SPRING_CONTACT_REFERENCE_M,
    SPRING_DAMPING_RATIO,
)
from .solid_model import SolidBundle, SolidConfig

# Historical small construction fixture: two cells along x, one on other axes.
_REFERENCE_SHAPE = (2, 1, 1)
_REFERENCE_CELL_SIZE_M = 0.05
_REFERENCE_CLEARANCE_M = 0.002
# Cubic beam cells are a model requirement. This tolerance accommodates only
# floating-point representation of equal dimensions, not meaningful anisotropy.
_CUBIC_CELL_RELATIVE_TOLERANCE = 1e-12


def _properties(
    count: "int",
    dim: "int",
    channels: "int",
    dtype: "DTypeLike",
    supplied: "Mapping[str, ArrayLike] | None",
) -> "FieldArrays":
    """Broadcast per-cell material/actuator fields and validate their bounds.

    Return independent host arrays; directional fibres are normalized once at
    instantiation. Missing material fields use the reference construction profile.
    Fibres default to +y and channels cycle over cells; beam construction may
    align unspecified fibres with the longest envelope axis. Unknown fields,
    invalid material bounds or a zero directional fibre raise ``ValueError``.
    """
    direction = np.zeros((count, dim), dtype=dtype)
    direction[:, 1] = 1
    data: dict[str, ArrayLike] = dict(
        occupancy=np.ones(count),
        young=np.full(count, REGULAR_SOLID_YOUNG_PA),
        density=np.full(count, BODY_DENSITY_KG_M3),
        strength=np.full(count, REGULAR_SOLID_STRENGTH),
        direction=direction,
        kind=np.ones(count, np.int32),
        channel=np.arange(count, dtype=np.int32) % channels,
        poisson=np.full(count, POISSON_DEFAULT),
    )
    if supplied:
        if set(supplied) - set(data):
            raise ValueError(f"Unknown design fields: {set(supplied) - set(data)}")
        data.update(supplied)
    out = {}
    for name, value in data.items():
        shape = (count, dim) if name == "direction" else (count,)
        out[name] = np.array(
            np.broadcast_to(np.asarray(value), shape),
            dtype=np.int32 if name in ("kind", "channel") else dtype,
        )
    if not all(np.all(np.isfinite(v)) for v in out.values()):
        raise ValueError("Nonfinite design")
    if np.any((out["occupancy"] < 0) | (out["occupancy"] > 1)) or out["occupancy"].sum() <= 0:
        raise ValueError("Empty body or invalid occupancy")
    if min(out["young"].min(), out["density"].min()) <= 0 or np.any(
        (out["poisson"] <= -1) | (out["poisson"] >= 0.5)
    ):
        raise ValueError("Invalid compressible material")
    if np.any((out["kind"] < 0) | (out["kind"] > 2)) or np.any(
        (out["channel"] < 0) | (out["channel"] >= channels)
    ):
        raise ValueError("Invalid actuator type or channel")
    if np.any((out["strength"] < 0) | (out["strength"] > 1)):
        raise ValueError("Actuator strength must be in [0,1]")
    length = np.linalg.norm(out["direction"], axis=-1)
    if np.any((out["kind"] == 1) & (length == 0)):
        raise ValueError("Directional actuators require a nonzero fiber")
    out["direction"] /= np.where(length > 0, length, 1)[:, None]
    return out


def _empty(
    reference: "HostArray", entities: "HostArray", channels: "int", dtype: "DTypeLike"
) -> "FieldArrays":
    """Allocate a resolved-solid field dictionary with fixed topology shapes."""
    n, d = reference.shape
    ne = len(entities)
    return dict(
        reference=reference,
        mass=np.zeros(n, dtype),
        inertia=np.ones(n, dtype),
        free=np.ones((n, d), dtype),
        entities=entities,
        cells=np.empty((0, 0), np.int32),
        cell_alpha=np.empty(0, dtype),
        stiffness=np.zeros(ne, dtype),
        damping=np.zeros(ne, dtype),
        length=np.ones(ne, dtype),
        routing=np.zeros((ne, channels), dtype),
        extension=np.zeros((ne, d, d), dtype),
        bending=np.zeros((ne, d, d), dtype),
        axis=np.zeros((ne, d), dtype),
        young=np.ones(ne, dtype),
        poisson=np.full(ne, POISSON_DEFAULT, dtype),
        strength=np.zeros(ne, dtype),
        direction=np.zeros((ne, d), dtype),
        kind=np.zeros(ne, np.int32),
        channel=np.zeros(ne, np.int32),
        alpha=np.ones(ne, dtype),
        gradients=np.empty((0, 0, d), dtype),
        weights=np.empty((ne, 0), dtype),
        contact_area=np.zeros((n, 1), dtype),
        offsets=np.empty((0, d), dtype),
    )


def regular_solid(
    physics: "str" = "spring",
    *,
    shape: "tuple[int, ...] | None" = None,
    size: "ArrayLike | None" = None,
    origin: "ArrayLike | None" = None,
    channels: "int" = 2,
    precision: "str" = "float64",
    properties: "Mapping[str, ArrayLike] | None" = None,
    config: "SolidConfig | None" = None,
    thickness: "float" = PLANAR_THICKNESS_M,
    damping_ratio: "float" = SPRING_DAMPING_RATIO,
    contact_modulus: "float" = SPRING_CONTACT_MODULUS_PA,
    length_ref: "float" = SPRING_CONTACT_REFERENCE_M,
    fixed: "ArrayLike | None" = None,
    terrain: "FieldArrays | None" = None,
) -> "SolidBundle":
    """Build owned host arrays for spring, beam or Hex8 FEM reference geometry.

    ``shape`` counts cells; ``size``, ``origin``, ``thickness`` and ``length_ref``
    are metres. Defaults describe two 50-mm cells with 2-mm ground clearance.
    Per-cell ``properties`` accepts occupancy, Young modulus (Pa), density
    (kg/m^3), strength, unit direction, actuator kind/channel and Poisson ratio.
    Omitted values use the documented reference construction profile; partial
    arrays are broadcast to the fixed cell count. No device storage is allocated.

    Spring edges use ``damping_ratio`` as a dimensionless critical-damping
    fraction. Their stored ``contact_area`` is actually nodal area multiplied by
    ``contact_modulus / length_ref`` (N/m), not raw area. Legacy flat contact
    multiplies this weight by penetration; general terrain additionally applies
    the configuration's dimensionless contact multiplier (default 1). Beam/FEM
    store geometric contact area (m^2) and use ``config.contact_stiffness`` in
    Pa/m (default 1e6). ``contact_modulus`` and ``length_ref`` therefore affect
    spring assembly only; they are not alternate beam/FEM stiffness controls.

    No topology repair, stiffness rescaling, or imported JAX executable is hidden
    here. ``fixed`` is a node-by-coordinate mask (FEM only). Geometry changes may
    change active mass/connectivity; invalid capacity, bounds or geometry raises
    ``ValueError`` rather than silently clipping. Beam cells must be cubic.
    """
    if physics not in ("spring", "beam", "fem") or precision not in ("float32", "float64"):
        raise ValueError("Unsupported physical model or precision")
    d = 2 if physics == "spring" else 3
    shape = tuple(shape or _REFERENCE_SHAPE[:d])
    if len(shape) != d or any(type(n) is not int or n < 1 for n in shape):
        raise ValueError("Positive integer shape required")
    size = np.asarray(
        size if size is not None else np.asarray(shape) * _REFERENCE_CELL_SIZE_M, dtype=float
    )
    origin = np.asarray(
        origin if origin is not None else (0.0, _REFERENCE_CLEARANCE_M, 0.0)[:d],
        dtype=float,
    )
    if (
        size.shape != (d,)
        or origin.shape != (d,)
        or np.any(size <= 0)
        or not np.all(np.isfinite(size))
        or not np.all(np.isfinite(origin))
    ):
        raise ValueError("Invalid reference envelope")
    if (
        type(channels) is not int
        or channels < 1
        or thickness <= 0
        or damping_ratio < 0
        or contact_modulus < 0
        or length_ref <= 0
    ):
        raise ValueError("Invalid capacity or mechanical scale")
    h = size / shape
    idx: HostArray = np.asarray(list(product(*(range(n) for n in shape))), dtype=np.int32)
    count = len(idx)
    prop = _properties(count, d, channels, precision, properties)
    if physics == "beam" and (properties is None or "direction" not in properties):
        prop["direction"].fill(0)
        prop["direction"][:, int(np.argmax(size))] = 1
    a, E, rho = prop["occupancy"], prop["young"], prop["density"]
    config = config or SolidConfig(
        physics=physics,
        grid_shape=(4,) * d,
        origin=(0.0,) * d,
        gravity=EARTH_GRAVITY_M_S2,
        contact_enabled=True,
        contact_stiffness=1.0 if physics == "spring" else SOLID_CONTACT_PRESSURE_PER_LENGTH_PA_M,
        linear_work_reuse=True,
    )
    if config.physics != physics:
        raise ValueError("Configuration physics differs from requested body")
    routing = np.eye(channels, dtype=precision)[prop["channel"]] * prop["strength"][:, None]
    if physics == "spring":
        nx, ny = shape
        reference = (
            np.asarray(list(product(range(nx + 1), range(ny + 1))), dtype=precision) * h + origin
        )

        def node_index(i: int, j: int) -> int:
            """Flatten a 2D corner index in the fixed reference grid."""
            return i * (ny + 1) + j

        cells: HostArray = np.asarray(
            [
                [
                    node_index(i, j),
                    node_index(i + 1, j),
                    node_index(i + 1, j + 1),
                    node_index(i, j + 1),
                ]
                for i, j in idx
            ],
            np.int32,
        )
        owners: dict[tuple[int, int], list[int]] = {}
        for cell, nodes in enumerate(cells):
            for i, j in ((0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3)):
                owners.setdefault(tuple(sorted((nodes[i], nodes[j]))), []).append(cell)
        edges: HostArray = np.asarray(sorted(owners), np.int32)
        f = _empty(reference, edges, channels, precision)
        f["cells"], f["cell_alpha"] = cells, a
        volume = np.prod(h) * thickness
        np.add.at(f["mass"], cells.ravel(), np.repeat(a * rho * volume / 4, 4))
        delta = reference[edges[:, 1]] - reference[edges[:, 0]]
        f["length"] = np.linalg.norm(delta, axis=-1)
        direction = delta / f["length"][:, None]
        for e, edge in enumerate(edges):
            member = np.asarray(owners[tuple(edge)])
            weights = a[member]
            total = weights.sum()
            compliance = np.sum(weights / E[member])
            effective = total / compliance if compliance > 0 else 0.0
            f["stiffness"][e] = (
                effective * total * thickness * np.prod(h) / (2 * f["length"][e] ** 2)
            )
            projection = np.sum(prop["direction"][member] * direction[e], axis=-1) ** 2
            projection = np.where(
                prop["kind"][member] == 1, projection, np.where(prop["kind"][member] == 2, 0.5, 0.0)
            )
            f["routing"][e] = (
                config.max_log_stretch
                * np.sum((weights * projection)[:, None] * routing[member], axis=0)
                / (total if total > 0 else 1.0)
            )
        mi, mj = f["mass"][edges[:, 0]], f["mass"][edges[:, 1]]
        reduced = mi * mj / np.where(mi + mj > 0, mi + mj, 1.0)
        f["damping"] = 2 * damping_ratio * np.sqrt(f["stiffness"] * reduced)
        surface: HostArray = np.zeros(len(reference), precision)
        for cell, (i, j) in enumerate(idx):
            neighbors = ((i, j - 1), (i + 1, j), (i, j + 1), (i - 1, j))
            for face, (ni, nj) in enumerate(neighbors):
                other = a[ni * ny + nj] if 0 <= ni < nx and 0 <= nj < ny else 0.0
                area = max(a[cell] - other, 0.0) * h[face % 2] * thickness
                np.add.at(surface, cells[cell, [[0, 1], [1, 2], [2, 3], [3, 0]][face]], area / 2)
        f["contact_area"] = surface[:, None] * contact_modulus / length_ref
    else:
        bits = np.asarray(list(product((0, 1), repeat=3)))
        node_shape = tuple(n + 1 for n in shape)
        reference = (
            np.asarray(list(product(*(range(n) for n in node_shape))), precision) * h + origin
        )
        ci = idx[:, None, :] + bits[None, :, :]
        cells = np.ravel_multi_index(tuple(ci[..., i] for i in range(3)), node_shape).astype(
            np.int32
        )
        exposure: HostArray = np.zeros((count, 6), precision)
        offsets: list[HostArray] = []
        face_areas = []
        face_nodes: list[HostArray] = []
        for axis in range(3):
            for sign in (-1, 1):
                other = idx.copy()
                other[:, axis] += sign
                good = np.all((other >= 0) & (other < np.asarray(shape)), axis=-1)
                clamped = np.clip(other, 0, np.asarray(shape) - 1)
                neighbor_ids = np.ravel_multi_index(tuple(clamped[:, k] for k in range(3)), shape)
                face = len(offsets)
                exposure[:, face] = np.maximum(a - np.where(good, a[neighbor_ids], 0.0), 0.0)
                off = np.zeros(3)
                off[axis] = sign * h[axis] / 2
                offsets.append(off)
                face_areas.append(np.prod(h[np.arange(3) != axis]))
                face_nodes.append(cells[:, np.flatnonzero(bits[:, axis] == (sign + 1) // 2)])
        if physics == "beam":
            if not np.allclose(h, h[0], rtol=_CUBIC_CELL_RELATIVE_TOLERANCE, atol=0):
                raise ValueError("Beam voxels must be cubic")
            reference = (idx + 0.5) * h + origin
            edge_pairs: list[tuple[int, int]] = []
            axes = []
            for axis in range(3):
                for i, index in enumerate(idx):
                    other = index.copy()
                    other[axis] += 1
                    if other[axis] < shape[axis]:
                        edge_pairs.append((i, int(np.ravel_multi_index(tuple(other), shape))))
                        axes.append(np.eye(3)[axis])
            edges = np.asarray(edge_pairs, np.int32).reshape(-1, 2)
            f = _empty(reference, edges, channels, precision)
            f["axis"] = np.asarray(axes, precision).reshape(-1, 3)
            f["mass"] = a * rho * np.prod(h)
            # Cubic voxel inertia: m*(h^2+h^2)/12 about each centroidal axis.
            f["inertia"] = f["mass"] * h[0] ** 2 / 6
            f["offsets"] = np.asarray(offsets, precision)
            f["contact_area"] = exposure * np.asarray(face_areas)[None, :]
            source_nodes: HostArray
            target_nodes: HostArray
            source_nodes, target_nodes = edges.T
            effective = 2 / (1 / E[source_nodes] + 1 / E[target_nodes])
            # Equal half-links act in series: average shear compliance itself,
            # rather than averaging nu or deriving G from only one endpoint.
            local_shear = E / (2 * (1 + prop["poisson"]))
            shear = 2 / (1 / local_shear[source_nodes] + 1 / local_shear[target_nodes])
            # Resolved bundles store edge material data (not voxel properties).
            f["young"] = effective
            f["poisson"] = effective / (2 * shear) - 1
            weight = np.minimum(a[source_nodes], a[target_nodes])
            axial = f["axis"][..., :, None] * f["axis"][..., None, :]
            perp = np.eye(3) - axial
            # Square-section second moment; 5/6 is the inherited rectangular
            # Timoshenko shear correction, matching physics.beam's convention.
            second = h[0] ** 4 / 12
            f["extension"] = weight[:, None, None] * (
                effective[:, None, None] * h[0] ** 2 * axial
                + (5 / 6) * shear[:, None, None] * h[0] ** 2 * perp
            )
            f["bending"] = weight[:, None, None] * (
                shear[:, None, None] * 2 * second * axial + effective[:, None, None] * second * perp
            )
            for e, (left_node, right_node) in enumerate(edges):
                pr = []
                for node in (left_node, right_node):
                    pr.append(
                        np.dot(prop["direction"][node], f["axis"][e]) ** 2
                        if prop["kind"][node] == 1
                        else (1 / 3 if prop["kind"][node] == 2 else 0.0)
                    )
                f["routing"][e] = 0.5 * (pr[0] * routing[left_node] + pr[1] * routing[right_node])
        else:
            f = _empty(reference, cells, channels, precision)
            f["alpha"] = a
            f["young"] = E
            f["poisson"] = prop["poisson"]
            f["strength"] = prop["strength"]
            f["direction"] = prop["direction"]
            f["kind"] = prop["kind"]
            f["channel"] = prop["channel"]
            np.add.at(f["mass"], cells.ravel(), np.repeat(a * rho * np.prod(h) / 8, 8))
            signs = 2 * bits - 1
            xis = np.asarray(list(product((-1 / np.sqrt(3), 1 / np.sqrt(3)), repeat=3)))
            gradients: HostArray = np.zeros((8, 8, 3), precision)
            for q, xi in enumerate(xis):
                for node, sign in enumerate(signs):
                    for axis in range(3):
                        gradients[q, node, axis] = (
                            sign[axis]
                            / (4 * h[axis])
                            * np.prod(1 + sign[np.arange(3) != axis] * xi[np.arange(3) != axis])
                        )
            f["gradients"] = gradients
            f["weights"] = np.full((count, 8), np.prod(h) / 8, dtype=precision)
            area = np.zeros(len(reference), precision)
            for face, nodes in enumerate(face_nodes):
                np.add.at(
                    area, nodes.ravel(), np.repeat(exposure[:, face] * face_areas[face] / 4, 4)
                )
            f["contact_area"] = area[:, None]
            if fixed is not None:
                f["free"] = ~np.broadcast_to(np.asarray(fixed, dtype=bool), (len(reference), 3))
    f["free"] = f["free"] * (f["mass"] > 0)[:, None]
    f = {
        k: np.asarray(
            v, dtype=np.int32 if k in ("entities", "cells", "kind", "channel") else precision
        )
        for k, v in f.items()
    }
    return SolidBundle(
        f, config, channels, float(h.min()), float(np.prod(h[:2])), terrain
    ).validate()
