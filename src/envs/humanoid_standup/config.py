"""HumanoidStandup co-design config: v4 task constants + design bounds (same as Humanoid).

Same model + offsettable machinery as Humanoid; the task differs: the body starts
SUPINE (baked into the XML's qpos0) and is rewarded for raising its torso, with no
healthy termination. Uses solver="Newton" (approved, as for Humanoid) since
mujoco_warp lacks the stock PGS solver.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import os

_ASSETS = os.path.join(os.path.dirname(__file__), "assets")


@dataclass(frozen=True)
class HumanoidStandupConfig:
    xml: str = "humanoidstandup.xml"
    frame_skip: int = 5                     # v4 HumanoidStandup frame_skip
    reset_noise_scale: float = 1e-2         # v4 reset noise (uniform, qpos + qvel)
    ctrl_cost_weight: float = 0.1           # v4 quad_ctrl_cost weight (on applied ctrl)
    impact_cost_weight: float = 0.5e-6      # v4 quad_impact_cost weight (on cfrc_ext)
    impact_cost_max: float = 10.0           # v4 impact-cost clamp
    max_episode_steps: int = 1000           # v4 episode length (no task termination)
    # Design bounds shared with Humanoid (same geometry -> same derived values):
    max_offset: float = 0.144               # 0.5 x median segment length
    min_link_length: float = 0.12           # 2 x max segment radius (0.060)
    radius_geoms: tuple = ("right_foot", "left_foot")
    hand_geoms: tuple = ("right_hand", "left_hand")
    radius_frac: float = 0.5
    njmax: int = 160
    nconmax: int | None = 40

    @property
    def xml_path(self) -> str:
        return os.path.join(_ASSETS, self.xml)


def make_config(setup: str | None = None, **overrides) -> HumanoidStandupConfig:
    """Build a config. ``setup`` is accepted for registry symmetry (one setup)."""
    return replace(HumanoidStandupConfig(), **overrides)
