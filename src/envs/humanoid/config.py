"""Humanoid co-design config: v4 task constants (incl. healthy termination) + design bounds.

NOTE: the ported XML uses solver="Newton" (approved) because mujoco_warp does not
implement the PGS solver the stock Humanoid-v4 XML requests; contact dynamics
therefore differ from stock, while the obs/reward/termination formulas are v4-exact.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import os

_ASSETS = os.path.join(os.path.dirname(__file__), "assets")


@dataclass(frozen=True)
class HumanoidConfig:
    xml: str = "humanoid.xml"
    frame_skip: int = 5                     # v4 Humanoid frame_skip
    reset_noise_scale: float = 1e-2         # v4 reset noise (uniform, qpos + qvel)
    ctrl_cost_weight: float = 0.1           # v4 ctrl_cost_weight (on applied ctrl)
    forward_reward_weight: float = 1.25     # v4 forward_reward_weight (COM x-velocity)
    healthy_reward: float = 5.0             # v4 healthy_reward (per healthy step)
    # v4 healthy range (kept FIXED regardless of design, per G2); torso height:
    healthy_z_min: float = 1.0
    healthy_z_max: float = 2.0
    max_episode_steps: int = 1000           # v4 episode length (control steps)
    # Design (offsettable) bounds: a +1 offset moves a link's far end this many
    # metres along a named body axis; min_link_length floors a shrunk link.
    max_offset: float = 0.144       # 0.5 x median segment length (0.289)
    min_link_length: float = 0.12   # 2 x max segment radius (0.060)
    # Sphere-radius design axis (J1): r = r0*(1 + off*radius_frac). Feet are their
    # own single-geom bodies; hands are the second geom on the forearm body and are
    # handled by the arm mechanism (which also makes the forearm length designable).
    radius_geoms: tuple = ("right_foot", "left_foot")
    hand_geoms: tuple = ("right_hand", "left_hand")
    radius_frac: float = 0.5
    njmax: int = 128
    nconmax: int | None = 32

    @property
    def xml_path(self) -> str:
        return os.path.join(_ASSETS, self.xml)


def make_config(setup: str | None = None, **overrides) -> HumanoidConfig:
    """Build a config. ``setup`` is accepted for registry symmetry (one setup)."""
    return replace(HumanoidConfig(), **overrides)
