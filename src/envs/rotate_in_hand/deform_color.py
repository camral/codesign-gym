"""Colour each heightfield pad by how far its cells are raised.

A pad's colour comes from a 2D texture on it (``geom_rgba`` can only tint a
whole geom). A texel's value is the relief under it as a share of the pad's
current reach, mapped through ``COLOR_SCALE``. Measured facts about MuJoCo's
hfield texture mapping that the code relies on:

- texel row ``i`` is hfield row ``nrow - 1 - i``, columns agree;
- only ``mjTEXROLE_RGB`` is sampled, so a texture carries no alpha;
- ``geom_rgba`` modulates the texture, so pad geoms must be white.

Code layout:

- add_pad_textures: give every hfield a texture and material in the spec;
- PadPainter: repaint a render copy's textures from its ``hfield_data``. The
caller uploads them, since the two render paths reach their GL context
differently;
- parse_scale, sample: the colour scale.
"""

from __future__ import annotations

import mujoco
import numpy as np

TEXTURE_SIZE = 64                           # texels across a pad, not a physics resolution

TEX_PREFIX = "deform_color_"

# [keypoint, r, g, b], keypoint as a share of a pad's reach. 0 and 1 are required
COLOR_SCALE = [
    [0.00, 0.16, 0.20, 0.36],   # slate, a pad on its base plane
    [0.25, 0.13, 0.47, 0.71],
    [0.50, 0.20, 0.73, 0.55],
    [0.75, 0.95, 0.80, 0.22],
    [1.00, 0.86, 0.20, 0.18],   # red, as high as the pad can go
]


def parse_scale(scale) -> tuple[np.ndarray, np.ndarray]:
    """Validate a colour scale and return ``(keys, rgb)`` sorted by keypoint."""
    rows = list(scale)
    if len(rows) < 2:
        raise ValueError("a colour scale needs at least the two endpoints, "
                         f"got {len(rows)}")
    keys, rgb = [], []
    for i, row in enumerate(rows):
        row = [float(v) for v in row]
        if len(row) != 4:
            raise ValueError(
                f"colour scale entry {i} has {len(row)} numbers; an entry is "
                "[keypoint, r, g, b]. A pad is always opaque, so a keypoint "
                "carries no alpha")
        if not 0.0 <= row[0] <= 1.0:
            raise ValueError(f"colour scale entry {i} has keypoint {row[0]}; "
                             "keypoints are shares of a pad's reach and "
                             "live in [0, 1]")
        if any(not 0.0 <= c <= 1.0 for c in row[1:]):
            raise ValueError(
                f"colour scale entry {i} has rgb {tuple(row[1:])}; MuJoCo's "
                "rgb components are 0-1 floats, not 0-255 bytes")
        keys.append(row[0])
        rgb.append(row[1:])

    keys = np.asarray(keys, dtype=np.float64)
    order = np.argsort(keys, kind="stable")
    keys, rgb = keys[order], np.asarray(rgb, dtype=np.float32)[order]
    if np.any(np.diff(keys) == 0.0):
        dupes = sorted({float(k) for k in keys[:-1][np.diff(keys) == 0.0]})
        raise ValueError(
            f"colour scale has two entries at keypoint(s) {dupes}; which one "
            "wins would depend on list order")
    for end in (0.0, 1.0):
        if not np.any(keys == end):
            raise ValueError(
                f"colour scale has no keypoint at {end:.0f}. Both ends are "
                "required: an undeformed pad and one at its full reach "
                f"each need a colour, and this scale runs "
                f"{keys[0]:.3g}..{keys[-1]:.3g}")
    return keys, rgb


def sample(t: np.ndarray, keys: np.ndarray, rgb: np.ndarray,
           interp: bool) -> np.ndarray:
    """``t`` in [0, 1] -> uint8 RGB. ``interp`` blends between keypoints, else
    ``t`` snaps to the nearest one (ties go to the lower)."""
    t = np.clip(np.asarray(t, dtype=np.float64), 0.0, 1.0)
    if interp:
        out = np.stack([np.interp(t, keys, rgb[:, c]) for c in range(3)], -1)
    else:
        hi = np.clip(np.searchsorted(keys, t, side="left"), 1, len(keys) - 1)
        nearer_hi = (t - keys[hi - 1]) > (keys[hi] - t)
        out = rgb[np.where(nearer_hi, hi, hi - 1)]
    return (np.clip(out, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def add_pad_textures(spec: mujoco.MjSpec, size: int = TEXTURE_SIZE) -> None:
    """Give every hfield of the assembled ``spec`` its own texture and material,
    and bind and whiten every pad geom."""
    for hf in spec.hfields:
        tex = spec.add_texture()
        tex.name = TEX_PREFIX + hf.name
        tex.type = mujoco.mjtTexture.mjTEXTURE_2D
        tex.builtin = mujoco.mjtBuiltin.mjBUILTIN_FLAT
        tex.width, tex.height = int(size), int(size)
        tex.rgb1 = [1.0, 1.0, 1.0]
        mat = spec.add_material()
        mat.name = TEX_PREFIX + hf.name
        mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = tex.name
        mat.texuniform = False
        mat.texrepeat = [1.0, 1.0]
        mat.specular, mat.shininess = 0.0, 0.0     # a highlight would read as another keypoint

    for geom in spec.geoms:
        if geom.type == mujoco.mjtGeom.mjGEOM_HFIELD:
            geom.material = TEX_PREFIX + geom.hfieldname
            geom.rgba = [1.0, 1.0, 1.0, 1.0]


class PadPainter:
    """Turns a render copy's ``hfield_data`` into its ``tex_data``.

    Holds the colour scale, the env's current ``hfield_offset_fraction``, and per
    pad a fixed plan: the four cells around every texel and the bilinear weights.
    """

    def __init__(self, m: mujoco.MjModel, pads: dict[str, np.ndarray],
                 fraction: float, scale=None, interp: bool = True):
        self.keys, self.rgb = parse_scale(
            COLOR_SCALE if scale is None else scale)
        self.interp = bool(interp)
        self.fraction = float(fraction)          # kept current by the env
        if self.fraction <= 0.0:
            raise ValueError("colouring reads a cell as a share of its pad's "
                             "reach, and hfield_offset_fraction must be > 0 "
                             "for there to be any")
        self._plan = []
        for k, hid in enumerate(pads["hfield"]):
            hid = int(hid)
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_HFIELD, hid)
            tid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_TEXTURE,
                                    TEX_PREFIX + name)
            if tid < 0:
                raise RuntimeError(
                    f"heightfield {name!r} has no {TEX_PREFIX} texture, so "
                    "there is nothing to paint its relief onto. "
                    "add_pad_textures has to run on the spec before it is "
                    "compiled.")
            self._plan.append(self._pad_plan(m, tid, pads, k))

    @staticmethod
    def _pad_plan(m, tid, pads, k):
        tw, th = int(m.tex_width[tid]), int(m.tex_height[tid])
        nch, adr = int(m.tex_nchannel[tid]), int(m.tex_adr[tid])
        # The heightfield's own samples, which a subdivided pad has more of
        # than it has cells.
        nrow, ncol = int(pads["hrow"][k]), int(pads["hcol"][k])
        base = int(pads["adr"][k])

        # Texel centres in the vertex grid, rows flipped (see the module docstring)
        v = (np.arange(th) + 0.5) / th
        u = (np.arange(tw) + 0.5) / tw
        rf = np.clip((1.0 - v) * (nrow - 1), 0.0, nrow - 1)[:, None]
        cf = np.clip(u * (ncol - 1), 0.0, ncol - 1)[None, :]
        r0 = np.minimum(rf.astype(np.int64), max(nrow - 2, 0))
        c0 = np.minimum(cf.astype(np.int64), max(ncol - 2, 0))
        r1, c1 = np.minimum(r0 + 1, nrow - 1), np.minimum(c0 + 1, ncol - 1)
        dr, dc = rf - r0, cf - c0

        idx = np.stack([np.broadcast_to(base + r * ncol + c, (th, tw))
                        for r, c in ((r0, c0), (r0, c1), (r1, c0), (r1, c1))])
        return (tid, adr, th, tw, nch, idx,
                np.broadcast_to(dr, (th, tw)).astype(np.float32),
                np.broadcast_to(dc, (th, tw)).astype(np.float32))

    @property
    def tex_ids(self) -> tuple[int, ...]:
        """Textures to re-upload after ``paint``."""
        return tuple(p[0] for p in self._plan)

    def paint(self, m: mujoco.MjModel) -> None:
        """Repaint every pad of the render copy ``m`` from the relief written into it."""
        data = m.hfield_data
        for _, adr, th, tw, nch, idx, dr, dc in self._plan:
            v00, v01, v10, v11 = (data[i] for i in idx)
            # Bilinear as lerps, not a weighted sum: the weights don't sum to
            # exactly 1, which would dither a flat pad across a byte boundary
            top = v00 + (v01 - v00) * dc
            bot = v10 + (v11 - v10) * dc
            reach = (top + (bot - top) * dr) / self.fraction   # share of the pad's current reach
            rgb = sample(reach, self.keys, self.rgb, self.interp)
            if nch > 3:
                rgb = np.concatenate(
                    [rgb, np.full((th, tw, nch - 3), 255, np.uint8)], axis=-1)
            m.tex_data[adr: adr + th * tw * nch] = rgb.ravel()
