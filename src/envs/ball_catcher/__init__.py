"""Ball-catcher co-design env.

Importing the package selects MuJoCo's GL backend, which only works before
mujoco is first imported. A package's ``__init__`` runs before any of its
submodules, which guarantees that order.
"""

from .setup_headless import auto_configure_gl

auto_configure_gl()
