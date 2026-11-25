"""A faithful, fast reproduction of the canonical f1tenth_gym viewer.

f1tenth_gym_jax ships only a web dashboard, and the original ``f110_gym`` pyglet renderer
cannot run alongside our numpy-2 / JAX stack (it pins numpy<=1.22, gym 0.19, numba, and
pyglet<1.5 -- which cannot even open an OpenGL context on modern macOS). So we redraw the
classic look ourselves from the same data, matching its exact palette and geometry:

  - background  RGB (9, 32, 87)     -- dark navy (``glClearColor`` in f110_gym/rendering.py)
  - wall points RGB (183, 193, 222) -- occupied map cells (``map_img == 0``)
  - ego car     RGB (172, 97, 185)  -- a ``0.58 x 0.31`` m oriented rectangle (``get_vertices``)
  - a zoom-follow camera centred on the car.

Speed: the whole track is rasterised to a static navy+walls image **once**; each frame just
crops the follow window and paints the car + trail with PIL -- no per-frame matplotlib, so the
live view runs at tens of fps.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

NAVY = (9, 32, 87)
LAVENDER = (183, 193, 222)
EGO = (172, 97, 185)
TRAIL = (150, 80, 165)
CAR_LENGTH, CAR_WIDTH = 0.58, 0.31


def _car_corners(x: float, y: float, theta: float) -> np.ndarray:
    c = np.array([[CAR_LENGTH / 2, CAR_WIDTH / 2], [CAR_LENGTH / 2, -CAR_WIDTH / 2],
                  [-CAR_LENGTH / 2, -CAR_WIDTH / 2], [-CAR_LENGTH / 2, CAR_WIDTH / 2]])
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return c @ R.T + (x, y)


class CanonicalViewer:
    """Fast rasterised f1tenth track view: static walls + a zoom-follow crop with the car."""

    def __init__(self, track, zoom: float = 10.0, ppm: int = 15, out_px: int = 420):
        self._ppm = int(ppm)                       # pixels per metre
        self._zoom = float(zoom)                   # half-window in metres
        self._out = int(out_px)
        self._half = int(round(self._zoom * self._ppm))   # crop half-size in px

        occ = np.asarray(track.occ_map)
        res, ox, oy = float(track.resolution), float(track.ox), float(track.oy)
        rows, cols = np.where(occ == 0.0)
        wx = ox + cols * res
        wy = oy + rows * res
        self._xmin, self._ymax = wx.min(), wy.max()

        w = int((wx.max() - self._xmin) * self._ppm) + 1
        h = int((self._ymax - wy.min()) * self._ppm) + 1
        pad = self._half
        # static canvas (padded so a follow crop is always in bounds)
        img = np.empty((h + 2 * pad, w + 2 * pad, 3), np.uint8)
        img[:] = NAVY
        px = ((wx - self._xmin) * self._ppm).astype(int) + pad
        py = ((self._ymax - wy) * self._ppm).astype(int) + pad
        for dx in (-1, 0, 1):                      # thicken walls to ~3px
            for dy in (-1, 0, 1):
                img[np.clip(py + dy, 0, img.shape[0] - 1), np.clip(px + dx, 0, img.shape[1] - 1)] = LAVENDER
        self._img = img
        self._pad = pad

    def _to_px(self, x: float, y: float):
        return ((x - self._xmin) * self._ppm) + self._pad, ((self._ymax - y) * self._ppm) + self._pad

    def frame(self, x: float, y: float, theta: float, lines: Optional[List[str]] = None,
              trail: Optional[np.ndarray] = None) -> np.ndarray:
        from PIL import Image, ImageDraw

        cxp, cyp = self._to_px(x, y)
        x0, y0 = int(round(cxp)) - self._half, int(round(cyp)) - self._half
        crop = self._img[y0:y0 + 2 * self._half, x0:x0 + 2 * self._half].copy()
        im = Image.fromarray(crop)
        draw = ImageDraw.Draw(im)

        if trail is not None and len(trail) > 1:
            pts = [(self._to_px(px, py)[0] - x0, self._to_px(px, py)[1] - y0) for px, py in trail]
            draw.line(pts, fill=TRAIL, width=2)

        corners = _car_corners(x, y, theta)
        poly = [(self._to_px(px, py)[0] - x0, self._to_px(px, py)[1] - y0) for px, py in corners]
        draw.polygon(poly, fill=EGO)

        if lines:
            draw.multiline_text((8, 6), "\n".join(lines), fill=(235, 240, 250), spacing=3)

        if im.size != (self._out, self._out):
            im = im.resize((self._out, self._out), Image.NEAREST)
        return np.asarray(im)
