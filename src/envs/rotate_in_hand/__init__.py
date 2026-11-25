"""GPU-parallel in-hand reorientation env with a per-world deformable hand.

Importing this package chooses MuJoCo's GL backend, which is only possible
*before* mujoco is first imported. Python runs a package's ``__init__`` before
any of its submodules, so importing anything under ``envs.rotate_in_hand`` gets
that ordering for free.
"""

from .setup_headless import auto_configure_gl

auto_configure_gl()
