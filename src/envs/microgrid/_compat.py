"""Compatibility shims. Imported before pymgrid anywhere it is used.

python-microgrid 1.4.1 calls ``np.product`` in its core module step path, but
``np.product`` was removed in NumPy 2.0. We alias it to ``np.prod`` rather than
pinning NumPy < 2, because this environment shares a NumPy-2 install with
pandas / gymnasium. This is the only NumPy-2 breakage in pymgrid's active code.
"""

from __future__ import annotations

import numpy as np

if not hasattr(np, "product"):  # pragma: no cover - trivial shim
    np.product = np.prod  # type: ignore[attr-defined]


def import_pymgrid():
    """Load the legacy dependency without Gym's generic migration banner.

    Codesign Gym exposes Gymnasium spaces and handles pymgrid's NumPy-2 shim
    above. The banner is printed by gym_notices rather than warnings.warn, so
    silence only that known notice during this dependency import. Restore the
    notice registry and leave every other warning/error untouched.
    """
    import importlib
    try:
        from gym_notices.notices import notices
    except ImportError:
        return importlib.import_module("pymgrid")
    removed = {version: message for version, message in notices.items()
               if message.startswith("Gym has been unmaintained since 2022")}
    try:
        for version in removed:
            del notices[version]
        return importlib.import_module("pymgrid")
    finally:
        notices.update(removed)


pymgrid = import_pymgrid()
