"""Configure MuJoCo's headless GL backend before MuJoCo is imported.

This MUST run before any ``import mujoco`` anywhere in the process, otherwise
the GL backend is already fixed and offscreen rendering fails on a headless
machine.
"""

import os
import sys


def setup_mujoco_headless(require_headless_flag: bool = False) -> None:
    """Point MuJoCo at an offscreen GL backend (EGL on Linux, CGL on macOS).

    Args:
        require_headless_flag: only apply the config when ``--headless`` is
            present in ``sys.argv`` (so an interactive viewer run is untouched).
    """
    if require_headless_flag and "--headless" not in sys.argv:
        return

    # An explicit MUJOCO_GL in the environment wins (e.g. osmesa on a box with
    # no EGL, or glfw for a debugging run with a display attached).
    if os.environ.get("MUJOCO_GL"):
        return

    if sys.platform != "darwin":
        os.environ["MUJOCO_GL"] = "egl"
        # EGL ignores CUDA_VISIBLE_DEVICES and routes every context to GPU 0 by
        # default; pin it to this worker's GPU so parallel workers don't all
        # pile onto device 0 and OOM the render context.
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            cvd = os.environ["CUDA_VISIBLE_DEVICES"].split(",")[0]
            os.environ["MUJOCO_EGL_DEVICE_ID"] = cvd
    else:
        os.environ["MUJOCO_GL"] = "cgl"


def auto_configure_gl() -> None:
    """Pick a GL backend for a process that never said which one it wants.

    Called at import time by the package ``__init__``, so any script that just imports
    the env (a training run, a notebook) can render on a headless box without
    repeating the setup call. A display -- or an explicit ``MUJOCO_GL`` -- means
    the caller has a windowing system to render into, so the default backend is
    left alone and ``mujoco.viewer`` keeps working.
    """
    if os.environ.get("MUJOCO_GL"):
        return
    if sys.platform == "darwin":
        # macOS always has a window server; forcing CGL here would break the
        # interactive viewer, which needs GLFW.
        return
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        return
    setup_mujoco_headless()
