"""Per-world heightfield pads driven by the embodiment action.

Every ``<hfield>`` asset of the hand is a deformable pad. Each of its cells takes
one embodiment entry in [0, 1], written to the cell as ``action * fraction`` of
the pad's own elevation range. Pads are not identified by name, and several
geoms may draw one pad.

mujoco_warp shares ``hfield_data`` and ``hfield_size`` across worlds, so the
device model carries a copy of every heightfield per world, laid out world
after world (``tile_hfields``), and each world's geoms draw their own copies
through a per-world ``geom_dataid`` (``world_geom_dataid``). Both arrays then
reshape to one row per world, which is how the kernels index them. The copies
exist on the device only; the compiled ``MjModel`` keeps one set, which is what
the render copies draw a world from.

With ``grid_subd`` > 1 the heightfields are ``grid_subd`` times finer each way
than the MJCF grid (``subdivide_hfields``), but the embodiment keeps one entry
per MJCF cell: each cell becomes a ``grid_subd`` x ``grid_subd`` patch sampled
from a degree-``spline_degree`` spline through the pad's cells, clipped to
[0, 1]. The spline is linear and separable, so it is two small matrices per pad
(``spline_matrix``), applied on the device as ``rows @ cells @ cols.T`` -- the
samples ``scipy.interpolate.RectBivariateSpline`` gives with ``s=0``.

With ``relief_length`` > 0 the pads that run along a finger link are tapered
towards the joints at both ends, so curled links do not press into each other.

- tile_hfields, world_geom_dataid: the per-world copies;
- spline_matrix, subdivide_hfields: subdivision;
- find_pads, empty_pads, n_cells, overflow_reports, relief_weights: host side,
building the pad table from a compiled model;
- HfieldArrays, build_pad_arrays, apply_hfield_offsets: device side.
"""

from __future__ import annotations

import mujoco
import numpy as np
import warp as wp
from scipy.interpolate import splev, splrep

# nrow x ncol is a pad's cell grid, the deform action's; hrow x hcol is its
# heightfield's samples, the same unless the pads are subdivided.
PAD_KEYS = ("hfield", "geom", "adr", "nrow", "ncol", "ncell", "col0", "range",
            "hrow", "hcol")


def spline_matrix(n: int, subd: int, degree: int) -> np.ndarray:
    """``[n * subd, n]``: the samples of a subdivided axis of ``n`` cells, as a
    linear map of the cell values.

    The cells sit at 0 .. n-1 and a degree-``degree`` spline interpolates them
    (FITPACK with ``s=0``; ``RectBivariateSpline`` is this spline along each
    axis). Sample ``j`` is at the centre of its sub-cell, ``(j + 0.5) / subd -
    0.5`` in cell units, clamped to the outer cells, so the outermost half cell
    repeats the edge value instead of extrapolating into spikes. The spline is
    linear in the values, so column ``i`` is the spline through cell ``i``
    alone. The end result is analogous to RectBivariateSpline, but on the live 1D array.

    An axis with too few cells for ``degree`` (a spline needs more cells than
    its degree) is interpolated at the highest degree it does have: linearly
    with 2 cells, and a single cell is held constant across its samples.
    """
    degree = min(int(degree), n - 1)
    if degree == 0:
        return np.ones((subd, 1))
    cells = np.arange(n, dtype=np.float64)
    at = np.clip((np.arange(n * subd) + 0.5) / subd - 0.5, 0.0, n - 1.0)
    return np.stack([splev(at, splrep(cells, unit, k=degree, s=0))
                     for unit in np.eye(n)], axis=1)


def subdivide_hfields(spec: mujoco.MjSpec, subd: int, degree: int) -> None:
    """Make every heightfield of ``spec`` ``subd`` times finer each way, its
    samples the spline through the old ones (see ``spline_matrix``).

    Runs on the assembled spec, before it is compiled. The old grid is what
    ``find_pads`` reads back as the cells, given the same ``subd``. A pad
    loaded from a file has a grid of its own the MJCF does not state, so it is
    refused rather than guessed at.
    """
    for h in spec.hfields:
        if h.file:
            raise RuntimeError(
                f"heightfield {h.name!r} is loaded from {h.file!r}; only "
                "heightfields with nrow/ncol in the MJCF can be subdivided")
        nrow, ncol = int(h.nrow), int(h.ncol)
        cells = np.asarray(h.userdata, dtype=np.float64)
        cells = (cells.reshape(nrow, ncol) if cells.size == nrow * ncol
                 else np.zeros((nrow, ncol)))
        fine = (spline_matrix(nrow, subd, degree) @ cells
                @ spline_matrix(ncol, subd, degree).T)
        h.nrow, h.ncol = nrow * subd, ncol * subd
        h.userdata = np.clip(fine, 0.0, 1.0).ravel()


def tile_hfields(gm, m: mujoco.MjModel,
                 n_world: int) -> tuple[wp.array, wp.array]:
    """Give the device model ``gm`` a copy of every heightfield of ``m`` for
    each of ``n_world`` worlds, and return per-world views of the copies:
    ``hfield_data`` as ``[world, cell]`` and ``hfield_size`` as
    ``[world, hfield]``.

    World ``w``'s copy of heightfield ``h`` is id ``w * m.nhfield + h``, and
    its cells sit ``w * m.nhfielddata`` further into ``hfield_data`` than the
    original's, so each world's block is one contiguous row laid out exactly
    like the compiled model's. Nothing draws a copy until ``world_geom_dataid``
    points each world's geoms at them.

    Built here rather than in the spec: adding heightfields to an ``MjSpec``
    costs time in the number it already holds, which at a few thousand worlds
    is minutes of model building. The kernels never size anything off
    ``nhfield``, only index by ``geom_dataid``, so longer arrays are all they
    need. A mujoco_warp that already batches a field per world keeps its
    leading axis, with the one row holding every world's copies.
    """
    nh, nd = m.nhfield, m.nhfielddata
    rows = {
        "hfield_size": np.tile(np.asarray(m.hfield_size, dtype=np.float32), (n_world, 1)),
        "hfield_nrow": np.tile(m.hfield_nrow, n_world),
        "hfield_ncol": np.tile(m.hfield_ncol, n_world),
        "hfield_adr": (m.hfield_adr[None] + nd * np.arange(n_world)[:, None]).ravel(),
        "hfield_data": np.tile(np.asarray(m.hfield_data, dtype=np.float32), n_world),
    }
    for name, host in rows.items():
        old = getattr(gm, name)
        if old.ndim == 2:
            host = host[None]
        setattr(gm, name, wp.array(host, dtype=old.dtype, device=old.device))
    gm.nhfield, gm.nhfielddata = nh * n_world, nd * n_world
    return (gm.hfield_data.reshape((n_world, nd)),
            gm.hfield_size.reshape((n_world, nh)))


def world_geom_dataid(m: mujoco.MjModel, n_world: int) -> np.ndarray:
    """``geom_dataid`` per world, ``[world, geom]``: every heightfield geom of
    world ``w`` draws ``w``'s copy of its heightfield (see ``tile_hfields``).
    Every other geom, and every geom of world 0, keeps its compiled id."""
    out = np.repeat(np.asarray(m.geom_dataid, dtype=np.int32)[None], n_world, axis=0)
    hf = m.geom_type == mujoco.mjtGeom.mjGEOM_HFIELD
    out[:, hf] += (np.arange(n_world, dtype=np.int32) * m.nhfield)[:, None]
    return out


def find_pads(m: mujoco.MjModel, col0: int = 0,
              subd: int = 1) -> dict[str, np.ndarray]:
    """Table of every pad, in hfield-id order.

    ``col0`` is the deform-action column the first pad's first cell reads; pads
    follow one another from there, in the same order their cells sit in
    ``hfield_data``. Each pad records the geom that draws it, so the render
    copies know which pads to re-upload, and its elevation range, which is
    what a cell's data means in metres.

    ``subd`` is the subdivision ``subdivide_hfields`` applied: a pad's cells
    are its heightfield's samples ``subd`` to a side, so the deform action
    keeps the MJCF grid's size whatever the heightfield's.
    """
    subd = max(int(subd), 1)
    geom_of: dict[int, int] = {}
    for g in range(m.ngeom):
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_HFIELD:
            geom_of.setdefault(int(m.geom_dataid[g]), g)

    rows, col = [], int(col0)
    for i in range(m.nhfield):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_HFIELD, i)
        if i not in geom_of:
            raise RuntimeError(
                f"heightfield {name!r} is never drawn by a geom, so nothing "
                "would collide with the cells the action deforms")
        if float(m.hfield_size[i][2]) <= 0.0:
            raise RuntimeError(
                f"heightfield {name!r} has elevation range 0, so no cell of it "
                "can ever leave its base plane")
        hrow, hcol = int(m.hfield_nrow[i]), int(m.hfield_ncol[i])
        if hrow % subd or hcol % subd:
            raise RuntimeError(
                f"heightfield {name!r} is {hrow}x{hcol}, which is not a grid "
                f"subdivided {subd} times")
        nrow, ncol = hrow // subd, hcol // subd
        rows.append((i, geom_of[i], int(m.hfield_adr[i]), nrow, ncol,
                     nrow * ncol, col, float(m.hfield_size[i][2]), hrow, hcol))
        col += nrow * ncol
    if not rows:
        raise RuntimeError(
            "the model has no heightfields to deform, but "
            "hfield_offset_fraction is > 0. Set it to 0 to run this hand as it "
            "is -- with max_link_offset at 0 too the hand is rigid and the "
            "deform action empty.")
    return {n: np.asarray(c) for n, c in zip(PAD_KEYS, zip(*rows))}


def empty_pads() -> dict[str, np.ndarray]:
    """An empty table, for a run without pad deformation."""
    return {n: np.zeros(0, dtype=np.float64) for n in PAD_KEYS}


def n_cells(pads: dict[str, np.ndarray]) -> int:
    """Embodiment entries the pads take up, one per cell."""
    return int(pads["ncell"].sum()) if len(pads["ncell"]) else 0


def overflow_reports(m: mujoco.MjModel, pads: dict[str, np.ndarray],
                     obj_bid: int) -> list[str]:
    """One line per pad whose cells are fine enough to overflow a contact pair.

    mujoco_warp stops recording a pair's contacts at ``mjMAXCONPAIR``, so a pad
    with cells much smaller than the object silently loses contacts. The cap
    is per pair, so splitting a pad into several geoms helps too.
    """
    reach = max((float(m.geom_rbound[g]) for g in range(m.ngeom)
                 if int(m.geom_bodyid[g]) == obj_bid), default=0.0)
    out = []
    for i, h in enumerate(pads["hfield"]):
        size = m.hfield_size[int(h)]
        nrow, ncol = int(pads["hrow"][i]), int(pads["hcol"][i])   # samples: what CCD runs against
        cols = min(ncol - 1, int(np.ceil(2.0 * reach * (ncol - 1)
                                         / max(2.0 * size[0], 1e-9))) + 1)
        rows = min(nrow - 1, int(np.ceil(2.0 * reach * (nrow - 1)
                                         / max(2.0 * size[1], 1e-9))) + 1)
        prisms = 2 * max(rows, 1) * max(cols, 1)      # two triangles per cell under the object
        if prisms > mujoco.mjMAXCONPAIR:
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_HFIELD, int(h))
            out.append(
                f"{name}: {nrow}x{ncol} samples, the object spans up to "
                f"{rows}x{cols} of them -> {prisms} candidate contacts against "
                f"a cap of {mujoco.mjMAXCONPAIR} per geom pair")
    return out


# Pads whose columns (heightfield x) run along a finger link. Double-ended links
# meet a joint at both column ends (c = 0 and c = hcol - 1); distal tip links
# meet a joint only at the proximal end (c = hcol - 1, body z = 0) and are free
# at the fingertip (c = 0, body +z).
RELIEF_PADS = ("proximal_x", "middle_x")
TIP_RELIEF_PADS = ("tip_x",)


def relief_weights(m: mujoco.MjModel, pads: dict[str, np.ndarray],
                   relief_length: int) -> list[np.ndarray]:
    """Per pad, one weight per heightfield column that tapers finger-link pads
    towards their joints.

    On ``RELIEF_PADS`` (joints at both ends), column ``c`` of ``hcol`` is
    ``d = min(c, hcol - 1 - c)`` samples from the nearer joint and is weighted
    ``min(1, d / relief_length)``, so both end columns are 0. On
    ``TIP_RELIEF_PADS`` (joint only at ``c = hcol - 1``), ``d = hcol - 1 - c``,
    leaving the fingertip end untapered. Other pads, and every pad at
    ``relief_length`` 0, are weighted 1.
    """
    out = []
    for h, hcol in zip(pads["hfield"], pads["hcol"]):
        hcol = int(hcol)
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_HFIELD, int(h)) or ""
        if relief_length > 0 and name.endswith(RELIEF_PADS):
            c = np.arange(hcol)
            out.append(np.minimum(1.0, np.minimum(c, hcol - 1 - c) / relief_length))
        elif relief_length > 0 and name.endswith(TIP_RELIEF_PADS):
            c = np.arange(hcol)
            out.append(np.minimum(1.0, (hcol - 1 - c) / relief_length))
        else:
            out.append(np.ones(hcol))
    return out


@wp.struct
class HfieldArrays:
    """The pad table on the device, and a per-world view of ``hfield_data``.
    With ``spline`` set the pads are subdivided: ``ay``/``ax`` hold each pad's
    row and column ``spline_matrix``, one after another, and ``tmp`` is
    per-world scratch for the half-applied product. ``cw`` holds each pad's
    ``relief_weights``, pad after pad."""
    data: wp.array2d(dtype=wp.float32)    # [world, sample], in [0, 1], see tile_hfields
    adr: wp.array(dtype=wp.int32)
    ncell: wp.array(dtype=wp.int32)
    col0: wp.array(dtype=wp.int32)        # pad's first embodiment column
    spline: int                           # 1: samples are the spline through the cells
    nrow: wp.array(dtype=wp.int32)        # cells
    ncol: wp.array(dtype=wp.int32)
    hrow: wp.array(dtype=wp.int32)        # samples
    hcol: wp.array(dtype=wp.int32)
    ay_adr: wp.array(dtype=wp.int32)      # pad's [hrow, nrow] matrix in ay
    ax_adr: wp.array(dtype=wp.int32)      # pad's [hcol, ncol] matrix in ax
    ay: wp.array(dtype=wp.float32)
    ax: wp.array(dtype=wp.float32)
    tmp: wp.array2d(dtype=wp.float32)     # [world, nrow * hcol of the largest pad]
    cw_adr: wp.array(dtype=wp.int32)      # pad's first column weight in cw
    cw: wp.array(dtype=wp.float32)


def build_pad_arrays(pads: dict[str, np.ndarray], data: wp.array, device,
                     n_world: int, subd: int = 1, degree: int = 2,
                     col_weights: list[np.ndarray] | None = None) -> HfieldArrays:
    """Upload the pad table, with the spline matrices of subdivided pads and
    the column weights (all 1 if None)."""
    out = HfieldArrays()
    out.data = data
    i32 = lambda a: wp.array(np.asarray(a, dtype=np.int32).reshape(-1),
                             dtype=wp.int32, device=device)
    for name in ("adr", "ncell", "col0", "nrow", "ncol", "hrow", "hcol"):
        setattr(out, name, i32(pads[name]))
    out.spline = int(subd > 1 and len(pads["hfield"]) > 0)
    ay, ax, ay_adr, ax_adr, tmp = [np.zeros(1)], [np.zeros(1)], [], [], 1
    if out.spline:
        ay, ax, a_y, a_x = [], [], 0, 0
        for nrow, ncol in zip(pads["nrow"], pads["ncol"]):
            nrow, ncol = int(nrow), int(ncol)
            ay.append(spline_matrix(nrow, subd, degree).ravel())
            ax.append(spline_matrix(ncol, subd, degree).ravel())
            ay_adr.append(a_y)
            ax_adr.append(a_x)
            a_y, a_x = a_y + ay[-1].size, a_x + ax[-1].size
            tmp = max(tmp, nrow * ncol * subd)
    out.ay_adr, out.ax_adr = i32(ay_adr), i32(ax_adr)
    out.ay = wp.array(np.concatenate(ay).astype(np.float32), dtype=wp.float32, device=device)
    out.ax = wp.array(np.concatenate(ax).astype(np.float32), dtype=wp.float32, device=device)
    out.tmp = wp.zeros((n_world if out.spline else 1, tmp), dtype=wp.float32, device=device)
    if col_weights is None:
        col_weights = [np.ones(int(h)) for h in pads["hcol"]]
    cw_adr = np.cumsum([0] + [len(c) for c in col_weights])[:-1]
    out.cw_adr = i32(cw_adr if len(col_weights) else [0])
    out.cw = wp.array(np.concatenate(col_weights + [np.ones(1)]).astype(np.float32),   # never empty
                      dtype=wp.float32, device=device)
    return out


@wp.func
def apply_hfield_offsets(hf: HfieldArrays,
                         action: wp.array2d(dtype=wp.float32),
                         offsets: wp.array2d(dtype=wp.float32),
                         w: int, n_pad: int, fraction: float):
    """Write world ``w``'s pads from its embodiment entries. The clipped action is
    recorded in ``offsets``, untapered. A subdivided pad's samples are the spline
    through its clipped cells, clipped to [0, 1] before the fraction. Every
    sample is then scaled by its column's ``cw`` weight."""
    scale = wp.clamp(fraction, 0.0, 1.0)
    for i in range(n_pad):
        adr = hf.adr[i]
        col = hf.col0[i]
        cw0 = hf.cw_adr[i]
        for k in range(hf.ncell[i]):
            a = wp.clamp(action[w, col + k], 0.0, 1.0)
            if hf.spline == 0:                   # one sample per cell, row-major
                hf.data[w, adr + k] = a * scale * hf.cw[cw0 + k % hf.ncol[i]]
            offsets[w, col + k] = a
        if hf.spline != 0:
            nr = hf.nrow[i]
            nc = hf.ncol[i]
            hr = hf.hrow[i]
            hc = hf.hcol[i]
            ay0 = hf.ay_adr[i]
            ax0 = hf.ax_adr[i]
            for r in range(nr):
                for c in range(hc):
                    acc = float(0.0)
                    for j in range(nc):
                        acc += offsets[w, col + r * nc + j] * hf.ax[ax0 + c * nc + j]
                    hf.tmp[w, r * hc + c] = acc
            for r in range(hr):
                for c in range(hc):
                    acc = float(0.0)
                    for j in range(nr):
                        acc += hf.ay[ay0 + r * nr + j] * hf.tmp[w, j * hc + c]
                    hf.data[w, adr + r * hc + c] = wp.clamp(acc, 0.0, 1.0) * scale * hf.cw[cw0 + c]
