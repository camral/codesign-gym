"""Small diagnostic renderer. Receives actual states only, after simulation."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

    from .host_types import HostArray

from itertools import combinations

import numpy as np


def render_state(
    positions: "ArrayLike",
    reference: "ArrayLike",
    entities: "HostArray | None" = None,
    *,
    label: "str" = "",
    bounds: "ArrayLike | None" = None,
    width: "int" = 720,
    height: "int" = 420,
) -> "NDArray[np.uint8]":
    """Project world-space nodes/particles into a diagnostic uint8 RGB image.

    Positions/reference are (count, 2 or 3) metre coordinates. Optional
    entities contain node indices; bounds are projected (low, high) corners.
    Pillow is imported only when rendering is requested."""
    from PIL import Image, ImageDraw

    positions = np.asarray(positions)
    reference = np.asarray(reference)

    def project(x: "HostArray") -> "HostArray":
        """Use the standard side-view/oblique projection without changing the state."""
        return (
            x[:, :2]
            if x.shape[-1] == 2
            else np.stack((x[:, 0] + 0.38 * x[:, 2], x[:, 1] + 0.13 * x[:, 2]), axis=-1)
        )

    xy = project(positions)
    ref = project(reference)
    if bounds is None:
        lo, hi = ref.min(0) - 0.05, ref.max(0) + 0.05
    else:
        lo, hi = np.asarray(bounds)
    scale = min((width - 60) / max(hi[0] - lo[0], 1e-6), (height - 65) / max(hi[1] - lo[1], 1e-6))
    pixels = np.stack(
        (30 + (xy[:, 0] - lo[0]) * scale, height - 30 - (xy[:, 1] - lo[1]) * scale), axis=-1
    )
    image = Image.new("RGB", (width, height), (245, 247, 250))
    draw = ImageDraw.Draw(image)
    floor = height - 30 + lo[1] * scale
    draw.line((0, floor, width, floor), fill=(125, 130, 140), width=2)
    if entities is not None:
        edges = set()
        for cell in entities:
            for a, b in combinations(map(int, cell), 2):
                if (
                    len(cell) == 2
                    or np.count_nonzero(np.abs(reference[a] - reference[b]) > 1e-9) == 1
                ):
                    edges.add(tuple(sorted((a, b))))
        for a, b in sorted(edges):
            draw.line((*pixels[a], *pixels[b]), fill=(43, 99, 149), width=2)
    for x, y in pixels:
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(26, 76, 122))
    draw.text((18, 12), label, fill=(20, 32, 48))
    return np.asarray(image)
