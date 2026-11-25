"""Ant co-design config: v4 task constants (incl. healthy termination) + design bounds."""

from __future__ import annotations

from dataclasses import dataclass, replace
import os

_ASSETS = os.path.join(os.path.dirname(__file__), "assets")


@dataclass(frozen=True)
class AntConfig:
    xml: str = "ant.xml"
    frame_skip: int = 5                     # v4 Ant frame_skip
    reset_noise_scale: float = 0.1          # v4 reset noise (qpos uniform, qvel gaussian)
    ctrl_cost_weight: float = 0.5           # v4 ctrl_cost_weight
    forward_reward_weight: float = 1.0      # v4 forward_reward_weight
    healthy_reward: float = 1.0             # v4 healthy_reward (per healthy step)
    # v4 healthy range (kept FIXED regardless of design, per G2); torso height:
    healthy_z_min: float = 0.2
    healthy_z_max: float = 1.0
    # v4 default use_contact_forces=False -> no contact-force obs or contact cost.
    max_episode_steps: int = 1000           # v4 episode length (control steps)
    # Design (offsettable) bounds: a +1 offset moves a link's far end this many
    # metres along a named body axis; min_link_length floors a shrunk link.
    max_offset: float = 0.141       # 0.5 x median segment length (0.283)
    min_link_length: float = 0.16   # 2 x max segment radius (0.080)
    njmax: int = 48
    nconmax: int | None = 16

    @property
    def xml_path(self) -> str:
        return os.path.join(_ASSETS, self.xml)


def make_config(setup: str | None = None, **overrides) -> AntConfig:
    """Build a config. ``setup`` is accepted for registry symmetry (one setup)."""
    return replace(AntConfig(), **overrides)
