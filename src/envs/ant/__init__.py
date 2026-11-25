"""
Authors: Aviraj Newatia

GPU-parallel Ant co-design env (port of Gymnasium's Ant-v4).

Importing this package chooses MuJoCo's GL backend, which is only possible
*before* mujoco is first imported. Python runs a package's ``__init__`` before
any of its submodules, so importing anything under ``envs.ant`` gets that
ordering for free.
"""

from .setup_headless import auto_configure_gl

auto_configure_gl()
