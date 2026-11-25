"""Swimmer co-design config: v4 task constants + design bounds."""

from __future__ import annotations

from dataclasses import dataclass, replace
import os

_ASSETS = os.path.join(os.path.dirname(__file__), "assets")


@dataclass(frozen=True)
class SwimmerConfig:
    xml: str = "swimmer.xml"
    frame_skip: int = 4                    # v4 Swimmer frame_skip
    reset_noise_scale: float = 0.1         # v4 reset noise (uniform, qpos + qvel)
    ctrl_cost_weight: float = 1e-4         # v4 ctrl_cost_weight
    forward_reward_weight: float = 1.0     # v4 forward_reward_weight
    max_episode_steps: int = 1000          # v4 episode length (control steps)
    # Design (offsettable) bounds: a +1 offset moves a link's far end this many
    # metres along a named body axis; min_link_length floors a shrunk link.
    max_offset: float = 0.5         # 0.5 x median segment length (1.000)
    min_link_length: float = 0.2    # 2 x max segment radius (0.100)
    njmax: int = 16
    nconmax: int | None = 4

    @property
    def xml_path(self) -> str:
        return os.path.join(_ASSETS, self.xml)


def make_config(setup: str | None = None, **overrides) -> SwimmerConfig:
    """Build a config. ``setup`` is accepted for registry symmetry (one setup)."""
    return replace(SwimmerConfig(), **overrides)
