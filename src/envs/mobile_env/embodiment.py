"""Embodiment design layer for the mobile-env co-design environment.

The embodiment is the **network infrastructure design**: how many base stations
to deploy, where, and at what transmit power. It is exposed as a dict of two
keys (see the Codesign Gym API contract):

    - ``num_bs``  : number of active base stations (int, in ``[min_bs, max_bs]``)
    - ``stations``: an ``(max_bs, 3)`` float array; row ``i`` is ``(x, y, tx)``
                    for base station ``i``. Only the first ``num_bs`` rows are
                    active; the remaining rows are padding (a valid sentinel).

Because mobile-env fixes the number/positions of base stations at construction
(and the observation/action sizes scale with the BS count), the design is a
*variable-size* set. We represent it as a fixed ``max_bs`` padded array so the
``embodiment_space`` has a fixed shape; ``num_bs`` says how many rows count.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping

import numpy as np
from gymnasium import spaces

from .config import MobileEnvConfig

# Ordered keys of the embodiment dict.
EMBODIMENT_KEYS = ("num_bs", "stations")

# Column indices within a station row.
_X, _Y, _TX = 0, 1, 2

EmbType = Dict[str, Any]


def _dbm_to_watts(dbm: np.ndarray | float) -> np.ndarray:
    """Convert transmit power from dBm to watts (``W = 10^(dBm/10) / 1000``)."""
    return np.power(10.0, np.asarray(dbm, dtype=np.float64) / 10.0) / 1000.0


def tx_budget_watts(cfg: MobileEnvConfig) -> float:
    """Total radiated-power budget (W): ``max_bs * W(tx_max) * tx_budget_fraction``.

    A fraction < 1 makes it impossible to run the whole ``max_bs`` fleet at
    ``tx_max``, so transmit power trades off against station count.
    """
    return float(cfg.max_bs * _dbm_to_watts(cfg.tx_max) * cfg.tx_budget_fraction)


def make_embodiment_space(cfg: MobileEnvConfig) -> spaces.Dict:
    """Build the ``embodiment_space``: ``num_bs`` (Discrete) + padded ``stations`` Box.

    ``num_bs`` is ``Discrete(max_bs + 1)`` (values ``0..max_bs``); the ``min_bs``
    floor is enforced in :func:`validate_embodiment`, not by the space itself.
    """
    m = cfg.max_bs
    low = np.tile(np.array([0.0, 0.0, cfg.tx_min], dtype=np.float32), (m, 1))
    high = np.tile(np.array([cfg.width, cfg.height, cfg.tx_max], dtype=np.float32), (m, 1))
    return spaces.Dict({
        "num_bs": spaces.Discrete(m + 1),
        "stations": spaces.Box(low=low, high=high, shape=(m, 3), dtype=np.float32),
    })


def base_embodiment(cfg: MobileEnvConfig) -> EmbType:
    """The reference deployment: the preset's seed positions at ``seed_tx``.

    Padding rows are a valid sentinel ``(0, 0, tx_min)`` so the whole array lies
    inside the embodiment Box.
    """
    m = cfg.max_bs
    seeds = cfg.seed_positions
    n = len(seeds)
    if not (cfg.min_bs <= n <= m):
        raise ValueError(
            f"preset {cfg.name!r} has {n} seed positions, outside [min_bs, max_bs]=[{cfg.min_bs}, {m}]"
        )
    stations = np.zeros((m, 3), dtype=np.float32)
    stations[:, _TX] = cfg.tx_min  # sentinel tx for all rows
    for i, (x, y) in enumerate(seeds):
        stations[i] = (x, y, cfg.seed_tx)
    return {"num_bs": int(n), "stations": stations}


def validate_embodiment(embodiment: Mapping[str, Any], cfg: MobileEnvConfig) -> bool:
    """Return True iff ``embodiment`` is a valid infrastructure design for ``cfg``.

    Checks: exact keys; ``num_bs`` an int in ``[min_bs, max_bs]``; ``stations`` of
    shape ``(max_bs, 3)`` and finite; every *active* station within the map and
    tx range; the summed active radiated power within the tx budget; and (if
    enabled) a minimum separation between active stations.
    """
    if set(embodiment.keys()) != set(EMBODIMENT_KEYS):
        return False

    try:
        num_bs = int(embodiment["num_bs"])
    except (TypeError, ValueError):
        return False
    if not (cfg.min_bs <= num_bs <= cfg.max_bs):
        return False

    stations = np.asarray(embodiment["stations"], dtype=np.float32)
    if stations.shape != (cfg.max_bs, 3) or not np.all(np.isfinite(stations)):
        return False

    active = stations[:num_bs]
    if np.any(active[:, _X] < 0) or np.any(active[:, _X] > cfg.width):
        return False
    if np.any(active[:, _Y] < 0) or np.any(active[:, _Y] > cfg.height):
        return False
    if np.any(active[:, _TX] < cfg.tx_min) or np.any(active[:, _TX] > cfg.tx_max):
        return False

    if cfg.tx_budget_fraction > 0 and _dbm_to_watts(active[:, _TX]).sum() > tx_budget_watts(cfg):
        return False

    if cfg.min_separation > 0 and num_bs > 1:
        pos = active[:, :2]
        d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
        d[np.diag_indices(num_bs)] = np.inf
        if np.min(d) < cfg.min_separation:
            return False

    return True


def embodiments_equal(a: Mapping[str, Any] | None, b: Mapping[str, Any] | None) -> bool:
    """Structural equality of two embodiments (``num_bs`` + station array)."""
    if a is None or b is None:
        return a is b
    if int(a["num_bs"]) != int(b["num_bs"]):
        return False
    return np.array_equal(np.asarray(a["stations"]), np.asarray(b["stations"]))


def stations_from_embodiment(embodiment: Mapping[str, Any], cfg: MobileEnvConfig):
    """Instantiate mobile-env ``BaseStation`` objects for the active rows.

    Imported lazily so the module stays importable without mobile-env installed.
    """
    from mobile_env.core.entities import BaseStation

    num_bs = int(embodiment["num_bs"])
    stations = np.asarray(embodiment["stations"], dtype=np.float32)
    return [
        BaseStation(
            i,
            (float(stations[i, _X]), float(stations[i, _Y])),
            bw=cfg.bw,
            freq=cfg.freq,
            tx=float(stations[i, _TX]),
            height=cfg.bs_height,
        )
        for i in range(num_bs)
    ]
