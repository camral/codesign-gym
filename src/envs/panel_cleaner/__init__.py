"""GPU-parallel solar-panel-cleaning env.

Importing this package chooses MuJoCo's GL backend, which is only possible
*before* mujoco is first imported. Python runs a package's ``__init__`` before
any of its submodules, so importing anything under ``envs.panel_cleaner`` gets
that ordering for free -- which is also why the env class is not re-exported
here: a driver that wants the offscreen backend imports ``setup_headless``
first, and pulling mujoco in from this file would beat it to the decision.
"""

from .setup_headless import auto_configure_gl

auto_configure_gl()
