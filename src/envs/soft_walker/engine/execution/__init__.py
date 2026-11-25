"""Whole-episode execution and CUDA residency certification.

JAX program builders are loaded only when requested. The shared CUDA observer
can therefore audit an independent Warp runtime without importing JAX.
"""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .compiled import compile_program as compile_program
    from .program import (
        EpisodeProgram as EpisodeProgram,
        bind_episode as bind_episode,
    )

from .policy import ResidencyError, ResidentPolicy

_LAZY = {
    "EpisodeProgram": ".program",
    "bind_episode": ".program",
    "compile_program": ".compiled",
}
__all__ = ["ResidentPolicy", "ResidencyError", *_LAZY]


def __getattr__(name: str) -> object:
    """Lazily resolve and memoize public JAX helpers; unknown names raise AttributeError."""
    if name not in _LAZY:
        raise AttributeError(f"{__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_LAZY[name], __name__), name)
    globals()[name] = value
    return value
