"""GPU-parallel pick-up-and-turn env with a per-world deformable hand.

The modules it shares with ``rotate_in_hand`` are copied into this package,
assets included, so it stands alone as the other envs do.

Importing this package chooses MuJoCo's GL backend, which is only possible
*before* mujoco is first imported. Python runs a package's ``__init__`` before
any of its submodules, so importing anything under ``envs.hand_pick_up`` gets
that ordering for free.
"""

from .setup_headless import auto_configure_gl

auto_configure_gl()
