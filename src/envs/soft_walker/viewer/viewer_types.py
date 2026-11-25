"""Host contracts for saved geometry and the optional Newton display adapter.

No graphics package is imported at runtime. Newton's viewer and Warp's generated
vector dtypes remain dynamic; geometry and buffer ownership are explicit.
"""

from __future__ import annotations

from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, Protocol, TypedDict

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from warp import array as WarpArray


class Candidate(TypedDict):
    """Decoded host samples; coordinates use metres and the physical Y-up frame."""

    id: str
    representation: str
    seed: int
    path: Path
    fields: dict[str, NDArray[Any]]
    dim: int
    shape: tuple[int, ...]
    size: NDArray[Any]
    coordinates: NDArray[Any]
    active: NDArray[Any]


class MeshHost(Protocol):
    """Own device buffers reused by Studio, morphology previews and terrain editing."""

    wp: ModuleType
    viewer: Any  # Optional Newton ViewerGL exposes version-specific display calls.
    scale: float
    buffers: dict[str, WarpArray]
    indices: dict[str, WarpArray]
    identity: WarpArray
    unit_scale: WarpArray
    material: WarpArray
