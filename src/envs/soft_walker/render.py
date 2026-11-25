"""SoftWalker frames: a 2D side view (oblique for 3D) of the live body over its terrain, extending Andy's Pillow renderer.

The camera tracks the body horizontally at a fixed zoom. Only solid cells are drawn, each coloured by its actuation
channel (grey when passive); terrain boxes, the ground plane (and pit floors on ``gaps``) are filled; a text overlay
gives the preset, step, travel and return. 3D bodies use Andy's oblique projection ``(x + 0.38 z, y + 0.13 z)``; terrain
is drawn as its x-y profile.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from .contracts.api_types import Fields, HostArray, RGBFrame

import colorsys

import numpy as np

from . import config as C, embodiment as E

WIDTH, HEIGHT = 720, 420  # px, render_state's default frame
VIEW_WIDTH = 0.6  # m of world shown horizontally (the body is 0.16 m long)
VIEW_BOTTOM = -0.08  # m, below the 5 cm pit floor
BACKGROUND, GROUND, TERRAIN, PASSIVE, OUTLINE, TEXT = (
    (245, 247, 250),
    (160, 165, 175),
    (125, 130, 140),
    (175, 178, 184),
    (40, 48, 60),
    (20, 32, 48),
)


def channel_colours(channels: int) -> list[tuple[int, int, int]]:
    """Evenly spaced hues, one per actuation channel."""
    return [
        cast(
            "tuple[int, int, int]",
            tuple(int(255 * c) for c in colorsys.hsv_to_rgb(k / channels, 0.65, 0.85)),
        )
        for k in range(channels)
    ]


def _project(x: HostArray) -> HostArray:
    """Map metre coordinates into a side view or the shared oblique 3D projection."""
    return (
        x[:, :2]
        if x.shape[-1] == 2
        else np.stack((x[:, 0] + 0.38 * x[:, 2], x[:, 1] + 0.13 * x[:, 2]), axis=-1)
    )


def _hull(points: HostArray) -> HostArray:
    """Convex hull (monotone chain) of 2D points, for drawing projected hexahedra."""
    pts = sorted(map(tuple, np.round(points, 12)))
    if len(pts) <= 2:
        return np.asarray(pts)

    def cross(o: tuple[float, ...], a: tuple[float, ...], b: tuple[float, ...]) -> float:
        """Signed projected triangle area used by the monotone-chain hull."""
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, ...]] = []
    upper: list[tuple[float, ...]] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.asarray(lower[:-1] + upper[:-1])


def cell_shapes(driver: str, positions: HostArray, fields: Fields) -> list[tuple[int, HostArray]]:
    """(cell index, world-space polygon or particle position) for every drawable primitive, back-to-front for 3D."""
    g, d = C.grid(driver), C.dim_of(driver)
    if driver == "spring" or driver == "fem":
        # spring: 4-node admissibility cells; FEM: 8-node hexahedra are its entities (its `cells` is an empty placeholder)
        corners = positions[
            np.asarray(fields["cells"] if driver == "spring" else fields["entities"])
        ]  # (cells, 4 or 8, dim)
        order = np.argsort(-corners.mean(1)[:, 2]) if d == 3 else np.arange(len(corners))
        return [(int(c), corners[c]) for c in order]
    if driver == "beam":  # one node per cell, at its centre
        h = C.CELL / 2
        square = np.asarray([[-h, -h, 0], [h, -h, 0], [h, h, 0], [-h, h, 0]])
        order = np.argsort(-positions[:, 2])
        return [(int(c), positions[c] + square) for c in order]
    ppa = C.PARTICLES_PER_AXIS  # MPM particles, mapped back to cells
    index = np.array(np.unravel_index(np.arange(len(positions)), tuple(n * ppa for n in g))).T
    cells = cast("HostArray", np.ravel_multi_index(tuple((index // ppa).T), g))
    order = np.argsort(-positions[:, 2]) if d == 3 else np.arange(len(positions))
    return [(int(cells[p]), positions[p][None]) for p in order]


def draw(
    driver: str,
    positions: HostArray,
    fields: Fields,
    design: HostArray,
    terrain: Fields | None,
    *,
    label: str = "",
) -> RGBFrame:
    """RGB frame (HEIGHT, WIDTH, 3) of the body at ``positions`` with per-cell ``design`` over ``terrain``."""
    from PIL import Image, ImageDraw

    d, channels = C.dim_of(driver), C.CHANNELS[C.dim_of(driver)]
    props = E.to_cells(np.asarray(design, float), d)
    solid = props["occupancy"] >= E.OCCUPANCY_THRESHOLD
    active = (np.rint(props["kind"]) != 0) & (props["strength"] > 0)
    palette = channel_colours(channels)
    shapes = [
        (c, s) for c, s in cell_shapes(driver, np.asarray(positions, float), fields) if solid[c]
    ]
    centre = np.mean(np.concatenate([_project(s) for _, s in shapes]), axis=0)[0] if shapes else 0.0
    scale = WIDTH / VIEW_WIDTH
    left = centre - VIEW_WIDTH / 2
    px = lambda xy: [
        (float((x - left) * scale), float(HEIGHT - (y - VIEW_BOTTOM) * scale)) for x, y in xy
    ]
    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    canvas = ImageDraw.Draw(image)
    ground = 0.0 if terrain is None else float(terrain["offset"])  # horizontal plane (normal = +y)
    canvas.rectangle(
        px([(left, ground)])[0] + px([(left + VIEW_WIDTH, VIEW_BOTTOM - 1.0)])[0], fill=GROUND
    )
    if terrain is not None:
        for c, h in zip(np.asarray(terrain["centers"]), np.asarray(terrain["half_sizes"])):
            (x0, y0), (x1, y1) = px([(c[0] - h[0], c[1] + h[1]), (c[0] + h[0], c[1] - h[1])])
            canvas.rectangle((x0, y0, x1, y1), fill=TERRAIN)
    radius = max(2.0, C.CELL / C.PARTICLES_PER_AXIS * scale / 2)
    for c, shape in shapes:
        colour = palette[int(np.rint(props["channel"][c]))] if active[c] else PASSIVE
        xy = _project(shape)
        if len(xy) == 1:
            ((x, y),) = px(xy)
            canvas.ellipse((x - radius, y - radius, x + radius, y + radius), fill=colour)
        else:
            canvas.polygon(px(_hull(xy) if len(xy) > 4 else xy), fill=colour, outline=OUTLINE)
    canvas.text((12, 10), label, fill=TEXT)
    return np.asarray(image)
