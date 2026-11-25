"""Walker2d co-design config: v4 task constants (incl. healthy termination) + design bounds."""

from __future__ import annotations

from dataclasses import dataclass, replace
import os

_ASSETS = os.path.join(os.path.dirname(__file__), "assets")


@dataclass(frozen=True)
class Walker2dConfig:
    xml: str = "walker2d.xml"
    frame_skip: int = 4                    # v4 Walker2d frame_skip
    reset_noise_scale: float = 5e-3        # v4 reset noise (uniform, qpos + qvel)
    ctrl_cost_weight: float = 1e-3         # v4 ctrl_cost_weight
    forward_reward_weight: float = 1.0     # v4 forward_reward_weight
    healthy_reward: float = 1.0            # v4 healthy_reward (per healthy step)
    # v4 healthy ranges (kept FIXED regardless of design, per G2):
    healthy_z_min: float = 0.8             # torso height floor
    healthy_z_max: float = 2.0             # torso height ceiling (two-sided, unlike Hopper)
    healthy_angle: float = 1.0             # |torso angle| ceiling
    max_episode_steps: int = 1000          # v4 episode length (control steps)
    # Design (offsettable) bounds: a +1 offset moves a link's far end this many
    # metres along a named body axis; min_link_length floors a shrunk link.
    max_offset: float = 0.163       # 0.5 x median segment length (0.325)
    min_link_length: float = 0.12   # 2 x max segment radius (0.060)
    # Links that are halves of one stock capsule split at a joint: massed once, as the stock capsule.
    split_capsules: tuple = ("torso_upper_offsettable_z", "torso_lower_offsettable_z")
    njmax: int = 32
    nconmax: int | None = 16

    @property
    def xml_path(self) -> str:
        return os.path.join(_ASSETS, self.xml)


def make_config(setup: str | None = None, **overrides) -> Walker2dConfig:
    """Build a config. ``setup`` is accepted for registry symmetry (one setup)."""
    return replace(Walker2dConfig(), **overrides)
