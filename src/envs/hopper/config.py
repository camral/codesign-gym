"""Hopper co-design config: v4 task constants (incl. healthy termination) + design bounds."""

from __future__ import annotations

from dataclasses import dataclass, replace
import os

_ASSETS = os.path.join(os.path.dirname(__file__), "assets")


@dataclass(frozen=True)
class HopperConfig:
    xml: str = "hopper.xml"
    frame_skip: int = 4                    # v4 Hopper frame_skip
    reset_noise_scale: float = 5e-3        # v4 reset noise (uniform, qpos + qvel)
    ctrl_cost_weight: float = 1e-3         # v4 ctrl_cost_weight
    forward_reward_weight: float = 1.0     # v4 forward_reward_weight
    healthy_reward: float = 1.0            # v4 healthy_reward (per healthy step)
    # v4 healthy ranges (kept FIXED regardless of design, per G2):
    healthy_z_min: float = 0.7             # torso height floor
    healthy_angle: float = 0.2             # |torso angle| ceiling
    healthy_state: float = 100.0           # |state[2:]| ceiling
    max_episode_steps: int = 1000          # v4 episode length (control steps)
    # Design (offsettable) bounds: a +1 offset moves a link's far end this many
    # metres along a named body axis; min_link_length floors a shrunk link.
    max_offset: float = 0.115       # 0.5 x median segment length (0.230)
    min_link_length: float = 0.12   # 2 x max segment radius (0.060)
    # Links that are halves of one stock capsule split at a joint: massed once, as the stock capsule.
    split_capsules: tuple = ("torso_upper_offsettable_z", "torso_lower_offsettable_z", "foot_toe_offsettable_x", "foot_heel_offsettable_x")
    njmax: int = 24
    nconmax: int | None = 12

    @property
    def xml_path(self) -> str:
        return os.path.join(_ASSETS, self.xml)


def make_config(setup: str | None = None, **overrides) -> HopperConfig:
    """Build a config. ``setup`` is accepted for registry symmetry (one setup)."""
    return replace(HopperConfig(), **overrides)
