"""Resolve JAX precision and device settings without eager runtime imports.

Importing this module requires only NumPy. Each resolver loads JAX when called;
precision checks retain the configured x64 policy and device requests never
silently fall back to a different backend.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import DTypeLike

if TYPE_CHECKING:
    import jax


def resolve_dtype(dtype: DTypeLike) -> np.dtype:
    """Resolve float32/float64 and reject float64 when JAX x64 is disabled."""
    import jax

    d = np.dtype(dtype)
    if d not in (np.dtype("float32"), np.dtype("float64")):
        raise ValueError("dtype must be float32 or float64")
    if d == np.dtype("float64") and not jax.config.x64_enabled:
        raise ValueError(
            "float64 requires JAX_ENABLE_X64=1 before process initialization; silent truncation is forbidden"
        )
    return d


def resolve_device(device: str | jax.Device | None = None) -> jax.Device:
    """Resolve an explicit device; unavailable devices raise without fallback."""
    import jax

    if device is None:
        return jax.devices()[0]
    if hasattr(device, "platform"):
        return device
    name, _, index = str(device).partition(":")
    if name == "cuda":
        name = "gpu"
    if name not in ("cpu", "gpu"):
        raise ValueError("device must be cpu[:index] or cuda[:index]")
    devices = jax.devices(name)  # Fails rather than silently falling back to CPU.
    i = int(index or 0)
    if not 0 <= i < len(devices):
        raise ValueError("Device index out of range")
    return devices[i]
