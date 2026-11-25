"""Select MuJoCo's GL backend for headless rendering.

Must run before mujoco is first imported anywhere in the process, afterwards
the backend is fixed.

- setup_mujoco_headless: force an offscreen backend (EGL on Linux, CGL on macOS);
- auto_configure_gl: call it only when no display and no explicit backend exist.
"""

import os
import sys


def setup_mujoco_headless(require_headless_flag: bool = False) -> None:
    """Point MuJoCo at an offscreen GL backend unless ``MUJOCO_GL`` is already set.

    Args:
        require_headless_flag: only apply when ``--headless`` is in ``sys.argv``.
    """
    if require_headless_flag and "--headless" not in sys.argv:
        return

    if os.environ.get("MUJOCO_GL"):
        return

    if sys.platform != "darwin":
        os.environ["MUJOCO_GL"] = "egl"
        # EGL ignores CUDA_VISIBLE_DEVICES and puts every context on GPU 0, so
        # parallel workers would all render on device 0
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            cvd = os.environ["CUDA_VISIBLE_DEVICES"].split(",")[0]
            os.environ["MUJOCO_EGL_DEVICE_ID"] = cvd
    else:
        os.environ["MUJOCO_GL"] = "cgl"


def auto_configure_gl() -> None:
    """Go headless only when there is no display and no explicit ``MUJOCO_GL``,
    so ``mujoco.viewer`` keeps working wherever a window can open."""
    if os.environ.get("MUJOCO_GL"):
        return
    if sys.platform == "darwin":
        # macOS always has a window server, and CGL would break the GLFW viewer
        return
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        return
    setup_mujoco_headless()
