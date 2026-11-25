"""The solar-panel array the roller cleans, and the order it is cleaned in.

Each panel is a support, one collision plate (brushed aluminum frame), and a grid
of visual-only photovoltaic cells, each with a site at its centre on the plate's
top surface. Slightly inset tiles reveal the metallic plate between them as crisp
grid lines without separate grout geoms. The env tests the roller against those
sites, so per-cell resolution costs no extra contacts.

Cells are indexed globally as ``(ix, iy)``: ``ix`` grows away from the robot
(+x), ``iy`` from -y to +y. A traversal visits all cells of one panel before
moving on. Between panels the sweep can stop at a midpoint, a free-space goal in
the gap that makes the robot walk around the finished panel.

- add_panels: build the array and return its ``PanelGrid``;
- add_midpoints: add the midpoint sites for a traversal;
- add_goal_marker: the translucent box marking the current goal;
- PanelGrid, Goal: the built array and one stop of the sweep;
- TRAVERSALS: the visiting orders, by name.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import mujoco
import numpy as np

RGBA_CELL = (0.28, 0.34, 0.42, 1.0)      # dusty, uncleaned solar cell
RGBA_CLEAN = (0.04, 0.24, 0.64, 1.0)     # clean, deep photovoltaic navy
_RGBA_COLUMN = (0.30, 0.32, 0.36, 1.0)   # galvanized steel support
_RGBA_PLATE = (0.72, 0.75, 0.79, 1.0)    # anodized aluminum frame/backing
_RGBA_MARKER = (0.10, 0.95, 0.65, 0.28)  # translucent cyan-green target marker

# Contact bit of the plates. Everything else uses bit 0, so a plate with only
# this bit touches nothing but geoms with it in their conaffinity (the roller)
PANEL_BIT = 1 << 2

MARKER_BODY = "goal_marker"
_MARKER_SCALE = 0.75                     # marker size as a fraction of a cell
_TILE_LIFT = 2.0e-4                      # against z-fighting of tiles and plate


def _serpentine(n_x: int, n_y: int, *, y_major: bool,
                start: tuple[int, int] = (0, 0)) -> list[tuple[int, int]]:
    """Boustrophedon over an ``n_x`` x ``n_y`` grid from the ``start`` corner.
    ``y_major`` sweeps along y and steps along x, otherwise the reverse."""
    sx, sy = start
    xs = list(range(n_x) if sx == 0 else range(n_x - 1, -1, -1))
    ys = list(range(n_y) if sy == 0 else range(n_y - 1, -1, -1))
    order: list[tuple[int, int]] = []
    if y_major:
        for i, ix in enumerate(xs):
            order += [(ix, iy) for iy in (ys if i % 2 == 0 else ys[::-1])]
    else:
        for i, iy in enumerate(ys):
            order += [(ix, iy) for ix in (xs if i % 2 == 0 else xs[::-1])]
    return order


def _panel_serpentine(n_x: int, n_y: int, cell_rows: int, cell_cols: int, *,
                      y_major: bool) -> list[tuple[int, int]]:
    """Serpentine over the panels, and over each panel's cells in turn. Each
    panel starts at the corner nearest where the previous one ended."""
    order: list[tuple[int, int]] = []
    corners = [(x, y) for x in (0, cell_rows - 1) for y in (0, cell_cols - 1)]
    for pr, pc in _serpentine(n_x // cell_rows, n_y // cell_cols, y_major=y_major):
        base_x, base_y = pr * cell_rows, pc * cell_cols
        if order:
            last_x, last_y = order[-1]
            start = min(corners, key=lambda c: (abs(base_x + c[0] - last_x)
                                                + abs(base_y + c[1] - last_y)))
        else:
            start = (0, 0)
        order += [(base_x + lx, base_y + ly)
                  for lx, ly in _serpentine(cell_rows, cell_cols,
                                            y_major=y_major, start=start)]
    return order


TRAVERSALS = {
    "horizontal": lambda n_x, n_y, c_r, c_c: _panel_serpentine(
        n_x, n_y, c_r, c_c, y_major=True),
    "vertical": lambda n_x, n_y, c_r, c_c: _panel_serpentine(
        n_x, n_y, c_r, c_c, y_major=False),
}


@dataclass(frozen=True)
class Goal:
    """One stop of the sweep: its site, its cell geom (None for a midpoint), the
    panel ``key`` it belongs to (unique for a midpoint), and its world position."""
    site: str
    geom: str | None
    key: tuple
    pos: tuple[float, float, float]

    @property
    def is_midpoint(self) -> bool:
        return self.geom is None


@dataclass
class PanelGrid:
    """The built array: grid shape and one panel's share of it, a cell's
    half-extent and the panels' tilt, each cell's site and geom names and
    position, each panel's foot, and the midpoints added per traversal."""
    n_x: int
    n_y: int
    cell_rows: int
    cell_cols: int
    half: tuple[float, float, float]
    tilt_quat: tuple[float, float, float, float]
    cells: dict[tuple[int, int], tuple[str, str]]
    cell_pos: dict[tuple[int, int], tuple[float, float, float]]
    panel_pos: dict[tuple[int, int], tuple[float, float, float]]
    midpoints: dict[str, list[tuple[int, str, tuple[float, float, float]]]] = (
        field(default_factory=dict))

    def _order(self, name: str) -> list[tuple[int, int]]:
        if name not in TRAVERSALS:
            raise ValueError(f"unknown panel_traversal {name!r}; "
                             f"have {sorted(TRAVERSALS)}")
        return TRAVERSALS[name](self.n_x, self.n_y,
                                self.cell_rows, self.cell_cols)

    def goals(self, name: str) -> list[Goal]:
        """Every stop of traversal ``name`` in visiting order, midpoints included."""
        out = [Goal(*self.cells[k],
                    (k[0] // self.cell_rows, k[1] // self.cell_cols),
                    self.cell_pos[k])
               for k in self._order(name)]
        # From the back, so each recorded index still points at its cell
        for i, site, pos in reversed(self.midpoints.get(name, [])):
            out.insert(i, Goal(site, None, ("midpoint", i), pos))
        return out

    @property
    def geoms(self) -> list[str]:
        return [g for _, g in self.cells.values()]

    @property
    def marker_half(self) -> list[float]:
        return [_MARKER_SCALE * h for h in self.half]


def _ensure_panel_materials(spec: mujoco.MjSpec) -> None:
    """Add reusable PBR materials for solar panels, frames, and supports."""
    existing = {mat.name for mat in spec.materials}
    if "panel_steel" not in existing:
        m = spec.add_material(name="panel_steel")
        m.rgba = list(_RGBA_COLUMN)
        m.metallic = 0.65
        m.roughness = 0.42
        m.specular = 0.4
    if "panel_frame" not in existing:
        m = spec.add_material(name="panel_frame")
        m.rgba = list(_RGBA_PLATE)
        m.metallic = 0.78
        m.roughness = 0.28
        m.specular = 0.65
    if "panel_cell_glass" not in existing:
        # White basecoat so per-geom rgba controls dusty -> clean tint while keeping glossy glass reflection
        m = spec.add_material(name="panel_cell_glass")
        m.rgba = [1.0, 1.0, 1.0, 1.0]
        m.metallic = 0.20
        m.roughness = 0.18
        m.specular = 0.80


def add_panels(
    spec: mujoco.MjSpec,
    *,
    rows: int = 1,
    cols: int = 1,
    first_pos=(1.2, 0.0, 0.0),
    length: float = 0.6,
    width: float = 0.75,
    height: float = 0.7,
    tilt_degs: float = 20.0,
    n_sensor_rows: int = 2,
    n_sensor_cols: int = 2,
    thickness: float = 0.03,
    gap_x: float = 1.0,
    gap_y: float = 0.0,
    collide_roller_only: bool = True,
) -> PanelGrid:
    """Build the ``rows`` x ``cols`` array and return its grid. ``first_pos`` is
    the near panel's foot, centred in y."""
    _ensure_panel_materials(spec)
    box = mujoco.mjtGeom.mjGEOM_BOX
    tilt = math.radians(tilt_degs)
    quat = (math.cos(-tilt / 2.0), 0.0, math.sin(-tilt / 2.0), 0.0)
    cell_x, cell_y = length / n_sensor_rows, width / n_sensor_cols

    # Support: a stub down the plate's normal, a vertical column to the ground,
    # and a cylinder of the stub's radius rounding the elbow between them
    stub = length / 5.0
    n_x, n_z = -math.sin(tilt), math.cos(tilt)          # plate normal in the xz plane
    plate_z = height - thickness / 2.0
    elbow_x, elbow_z = -n_x * stub, plate_z - n_z * stub
    if elbow_z <= 0.0:
        raise ValueError(
            f"a panel {height} tall tilted {tilt_degs} deg has its support elbow "
            f"at z={elbow_z:.3f}, at or below the ground: raise panel_height or "
            f"shorten the panel (the stub is panel_length / 5 = {stub:.3f})")
    tile_h = thickness * 0.15
    # The plate's contact bits. With collide_roller_only, only the roller touches it
    con = PANEL_BIT if collide_roller_only else PANEL_BIT | 1
    aff = 0 if collide_roller_only else PANEL_BIT | 1

    cells: dict[tuple[int, int], tuple[str, str]] = {}
    cell_pos: dict[tuple[int, int], tuple[float, float, float]] = {}
    panel_pos: dict[tuple[int, int], tuple[float, float, float]] = {}
    ct, st = math.cos(tilt), math.sin(tilt)
    for r in range(rows):
        for c in range(cols):
            name = f"panel_r{r}_c{c}"
            pos = [first_pos[0] + (length + gap_x) * r,
                   first_pos[1] + (width + gap_y) * (c - (cols - 1) / 2),
                   first_pos[2]]
            panel_pos[(r, c)] = tuple(pos)
            body = spec.worldbody.add_body(name=name, pos=pos)
            body.add_geom(name=f"{name}_column", type=box,
                          pos=[elbow_x, 0, elbow_z / 2],
                          size=[stub, width / 5, elbow_z / 2],
                          material="panel_steel", solref=[0.01, 1.0],
                          contype=1, conaffinity=1)
            body.add_geom(name=f"{name}_elbow",
                          type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                          fromto=[elbow_x, -width / 5, elbow_z,
                                  elbow_x, width / 5, elbow_z],
                          size=[stub, 0, 0],
                          material="panel_steel", solref=[0.01, 1.0],
                          contype=1, conaffinity=1)
            top = body.add_body(name=f"{name}_top",
                                pos=[0, 0, plate_z], quat=list(quat))
            top.add_geom(name=f"{name}_stub", type=box, pos=[0, 0, -stub / 2],
                         size=[stub, width / 5, stub / 2],
                         material="panel_steel", solref=[0.01, 1.0],
                         contype=1, conaffinity=1)
            top.add_geom(name=f"{name}_plate", type=box, pos=[0, 0, 0],
                         size=[length / 2, width / 2, thickness / 2],
                         material="panel_frame", solref=[0.01, 1.0],
                         contype=con, conaffinity=aff)

            # Inset photovoltaic cell tiles (group 2 visual only): gaps reveal the metallic frame
            for lr in range(n_sensor_rows):
                for lc in range(n_sensor_cols):
                    cx = -length / 2 + (lr + 0.5) * cell_x
                    cy = -width / 2 + (lc + 0.5) * cell_y
                    tile = f"{name}_cell_{lr}_{lc}"
                    top.add_geom(name=tile, type=box, group=2,
                                 pos=[cx, cy, thickness / 2 - tile_h / 2 + _TILE_LIFT],
                                 size=[0.93 * cell_x / 2, 0.93 * cell_y / 2, tile_h / 2],
                                 material="panel_cell_glass",
                                 rgba=list(RGBA_CELL),
                                 contype=0, conaffinity=0, density=0.0, mass=0.0)
                    top.add_site(name=f"{tile}_site", pos=[cx, cy, thickness / 2],
                                 size=[0.005, 0, 0], group=4,
                                 rgba=[1.0, 0.9, 0.1, 0.5])
                    key = (r * n_sensor_rows + lr, c * n_sensor_cols + lc)
                    cells[key] = (f"{tile}_site", tile)
                    # World position of the site, computed here because the
                    # midpoints are placed from it before compiling
                    cz = thickness / 2
                    cell_pos[key] = (pos[0] + ct * cx - st * cz,
                                     pos[1] + cy,
                                     pos[2] + plate_z + st * cx + ct * cz)

    return PanelGrid(n_x=rows * n_sensor_rows, n_y=cols * n_sensor_cols,
                     cell_rows=n_sensor_rows, cell_cols=n_sensor_cols,
                     half=(cell_x / 2, cell_y / 2, thickness / 2),
                     tilt_quat=quat, cells=cells, cell_pos=cell_pos,
                     panel_pos=panel_pos)


def add_midpoints(spec: mujoco.MjSpec, grid: PanelGrid, traversal: str, *,
                  y_offset: float = 0.3) -> int:
    """Add a midpoint site before each new panel of ``traversal``, and return
    how many there are.

    x and z are halfway between the last cell left and the first cell entered.
    For panels side by side, y is halfway between them. For panels one behind
    the other, y is ``y_offset`` further out than the cell just left, measured
    from that panel's centre line, so the robot goes around it. The sites carry
    the panels' tilt, so a midpoint's normal matches the cells'.
    """
    grid.midpoints.pop(traversal, None)       # stale midpoints would read as more panel changes
    goals = grid.goals(traversal)
    marks: list[tuple[int, str, tuple[float, float, float]]] = []
    for i in range(1, len(goals)):
        out_g, in_g = goals[i - 1], goals[i]
        if out_g.key == in_g.key:
            continue
        out_, in_ = out_g.pos, in_g.pos
        if out_g.key[1] != in_g.key[1]:      # different panel columns, so side by side
            y = 0.5 * (grid.panel_pos[out_g.key][1]
                       + grid.panel_pos[in_g.key][1])
        else:
            out_y = out_[1] - grid.panel_pos[out_g.key][1]
            y = out_[1] + (y_offset if out_y >= 0.0 else -y_offset)
        pos = (0.5 * (out_[0] + in_[0]), y, 0.5 * (out_[2] + in_[2]))
        name = f"midpoint_{len(marks)}_site"
        spec.worldbody.add_site(name=name, pos=list(pos),
                                quat=list(grid.tilt_quat), size=[0.02, 0, 0],
                                group=4, rgba=[1.0, 0.4, 0.1, 0.6])
        marks.append((i, name, pos))
    grid.midpoints[traversal] = marks
    return len(marks)


def add_goal_marker(spec: mujoco.MjSpec, *, half, quat) -> str:
    """Add the translucent goal marker as its own static body, parked below the
    floor. The env moves it through ``body_pos``."""
    body = spec.worldbody.add_body(name=MARKER_BODY, pos=[0, 0, -10.0],
                                   quat=list(quat))
    body.add_geom(name=f"{MARKER_BODY}_geom", type=mujoco.mjtGeom.mjGEOM_BOX,
                  size=list(half), group=2, contype=0, conaffinity=0,
                  density=0.0, mass=0.0, rgba=list(_RGBA_MARKER))
    return MARKER_BODY
