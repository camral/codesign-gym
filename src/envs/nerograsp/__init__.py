"""NeroGrasp & NeroTools: designed fingertips for a Nero arm (native MuJoCo Warp batch).

Importing this package chooses MuJoCo's GL backend, which is only possible
*before* mujoco is first imported.
"""

from .setup_headless import auto_configure_gl

auto_configure_gl()

from .config import MeshRejectorConfig, NeroGraspConfig, NeroToolsConfig  # noqa: E402
from .spaces import Mesh, MeshSpace, Space  # noqa: E402

__all__ = ["NeroGraspConfig", "NeroToolsConfig", "MeshRejectorConfig", "Mesh", "MeshSpace", "Space"]
