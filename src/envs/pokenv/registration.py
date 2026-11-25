"""Conventional builder consumed by ``codesign_gym``'s registry.

``codesign_gym.make_vec`` looks up ``envs.pokenv.registration:build_vec`` and calls it with a batch
size plus the common codesign-gym kwargs. A single builder serves every opponent via the
``opponent`` argument (mirroring how lq_structure and microgrid use ``setup``); it returns the
env's native vectorized form (:class:`~envs.pokenv.vector_env.PokenvVectorEnv`), which owns its
Showdown cluster and exposes ``base_embodiment``.

Opponents (matching the Pokemon package's ``OPPONENTS``):
  - ``random``            : a player choosing uniformly among legal moves.
  - ``max_base_power``    : a player picking the highest base-power move.
  - ``simple_heuristics`` : poke-env's heuristic player.
  - ``league``            : a fixed or randomly selected five-battle Pokemon League gauntlet.

Remaining keyword args are forwarded to ``pokemon_codesign_env.make_vec`` -- e.g. ``shards``,
``envs_per_worker``, ``step_timeout``, ``format_config``, ``embodiment_config``,
``observation_config`` -- so a caller can tune the cluster and the design space through
``codesign_gym.make``/``make_vec``.
"""

from __future__ import annotations

from typing import Any

from .vector_env import PokenvVectorEnv

# Opponent ids this builder accepts, mirroring pokemon_codesign_env's OPPONENTS. Kept as a literal
# so registration does not import the (heavy) Pokemon factory just to list them; validated against
# the package at build time.
OPPONENTS = ("random", "max_base_power", "simple_heuristics", "league")


def build_vec(
    num_envs: int,
    *,
    opponent: str = "random",
    is_eval_env: bool = False,
    render_mode: str | None = None,
    **overrides: Any,
) -> PokenvVectorEnv:
    """Build the vectorized Pokemon Showdown env for a given opponent.

    ``opponent`` selects the task suite's opponent; remaining keyword args are forwarded to
    ``pokemon_codesign_env.make_vec`` (cluster sizing, format, and design-space configuration).
    """
    if opponent not in OPPONENTS:
        raise ValueError(f"unknown opponent {opponent!r}; choose from {list(OPPONENTS)}")
    return PokenvVectorEnv(
        num_envs=num_envs,
        opponent=opponent,
        is_eval_env=is_eval_env,
        render_mode=render_mode,
        **overrides,
    )
