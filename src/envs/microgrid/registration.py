"""Conventional builder consumed by ``codesign_gym``'s registry.

``codesign_gym.make_vec`` looks up ``envs.microgrid.registration:build_vec`` and
calls it with a batch size plus the common codesign-gym kwargs. A single builder
serves all setups via the ``setup`` argument (mirroring lq_structure); it returns
the env's native vectorized form (:class:`MicrogridVectorEnv`), which exposes
``base_embodiment``.

Setups:
  - ``random``        : single building, grid-connected, randomized demand &
    irradiance (24 h).
  - ``forecast``      : as random, with forecast-augmented observations and a
    tunable ``forecast_noise``.
  - ``campus``        : several buildings summed into one larger grid-connected
    microgrid (bigger load and design ranges).
  - ``offgrid``       : single building, NO utility grid -- demand must be met by
    solar+battery+genset (multi-day, 72 h).
  - ``offgridcampus`` : multi-building AND off-grid: a remote community microgrid
    (multi-day, 72 h).

All setups accept transition-level embodiment changes in training; an
``is_eval_env`` freezes the design mid-episode (handled by the CoDesignEnv base).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .vector_env import MicrogridVectorEnv

# Per-setup default env kwargs. User kwargs passed through ``make`` override these.
SETUP_CONFIGS: Dict[str, Dict[str, Any]] = {
    "random": {"horizon": 24},
    "forecast": {"horizon": 24, "forecast_horizon": 12, "forecast_noise": 0.25},
    "campus": {
        "horizon": 24,
        "n_buildings": 4,
        "demand_peak_kw": 1200.0,   # 4 x ~300 kW; ranges/grid scale up with this
    },
    "offgrid": {
        "horizon": 72,             # multi-day: cloudy-run resilience
        "off_grid": True,
    },
    "offgridcampus": {"horizon": 72, "n_buildings": 4, "demand_peak_kw": 1200.0, "off_grid": True},
}


def build_vec(
    num_envs: int,
    *,
    setup: str = "random",
    is_eval_env: bool = False,
    render_mode: Optional[str] = None,
    **overrides: Any,
) -> MicrogridVectorEnv:
    """Build the vectorized microgrid env for a given setup.

    ``setup`` selects the scenario; remaining keyword args override the setup's
    default :class:`~envs.microgrid.env.MicrogridCodesignEnv` kwargs.
    """
    if setup not in SETUP_CONFIGS:
        raise ValueError(f"unknown setup {setup!r}; choose from {sorted(SETUP_CONFIGS)}")
    env_kwargs = {**SETUP_CONFIGS[setup], **overrides}
    return MicrogridVectorEnv(
        num_envs=num_envs,
        is_eval_env=is_eval_env,
        render_mode=render_mode,
        **env_kwargs,
    )
