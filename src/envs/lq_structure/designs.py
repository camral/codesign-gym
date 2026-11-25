"""Embodiment space, base embodiment, validation, and padding helpers.

The embodiment is a dict of design parameters (Option A):

    Dict({
        'masses':    Box(n),          continuous
        'stiffness': Box(n),          continuous
        'damping':   Box(n),          continuous
        'actuators': MultiBinary(n),  actuator placement
        'sensors':   MultiBinary(n),  sensor placement
    })

Setups that do not design placement (e.g. LQR) still expose the keys in the space for
a uniform contract, but freeze them to their :func:`base_embodiment` value (all-ones by
default) as an int8 ``Box`` with ``low == high``; ``validate_embodiment`` rejects a
placement that differs from it, plus the constraints that actually apply to the setup
(budgets, ``>=1`` actuator, action-slot capacity).

Coupled constraints (mass/authority budget, actuator capacity) cannot live in a
gymnasium Space, so they are enforced here in :func:`validate_embodiment`.
"""

from __future__ import annotations

import numpy as np
from gymnasium import spaces

from .config import LQConfig

EMB_KEYS = ("masses", "stiffness", "damping", "actuators", "sensors")


def build_embodiment_space(cfg: LQConfig) -> spaces.Dict:
    """Construct the gymnasium Dict embodiment space for a config."""
    n = cfg.n
    base = base_embodiment(cfg)

    def placement(key: str, designed: bool) -> spaces.Space:
        # A placement the setup does not design is pinned to its base value, so
        # sampling the space never proposes a placement the setup would reject.
        return spaces.MultiBinary(n) if designed else spaces.Box(low=base[key], high=base[key], dtype=np.int8)

    return spaces.Dict(
        {
            "masses": spaces.Box(
                low=cfg.mass_low, high=cfg.mass_high, shape=(n,), dtype=np.float64
            ),
            "stiffness": spaces.Box(
                low=cfg.stiffness_low, high=cfg.stiffness_high, shape=(n,), dtype=np.float64
            ),
            "damping": spaces.Box(
                low=cfg.damping_low, high=cfg.damping_high, shape=(n,), dtype=np.float64
            ),
            "actuators": placement("actuators", cfg.design_actuators),
            "sensors": placement("sensors", cfg.design_sensors),
        }
    )


def base_embodiment(cfg: LQConfig) -> dict[str, np.ndarray]:
    """Default embodiment used at construction / for reset-to-known-state.

    Continuous parameters default to the midpoint of their bounds (or config overrides).
    Placement defaults to all-ones (fully actuated / sensed) unless overridden. Setups
    that do not design a given placement keep it fixed at this base value.
    """
    n = cfg.n

    def mid(lo, hi):
        return np.full(n, 0.5 * (lo + hi), dtype=np.float64)

    masses = cfg.base_masses if cfg.base_masses is not None else mid(cfg.mass_low, cfg.mass_high)
    stiffness = (
        cfg.base_stiffness if cfg.base_stiffness is not None
        else mid(cfg.stiffness_low, cfg.stiffness_high)
    )
    damping = (
        cfg.base_damping if cfg.base_damping is not None
        else mid(cfg.damping_low, cfg.damping_high)
    )
    if cfg.base_actuators is not None:
        actuators = cfg.base_actuators
    elif cfg.actuator_budget is not None and cfg.actuator_budget < n:
        # Respect the setup's actuator budget: actuate the first ``budget`` masses so
        # the base embodiment is itself valid (it is the reset-to-known-state default).
        actuators = np.zeros(n, dtype=np.int8)
        actuators[: cfg.actuator_budget] = 1
    else:
        actuators = np.ones(n, dtype=np.int8)
    sensors = (
        cfg.base_sensors if cfg.base_sensors is not None
        else np.ones(n, dtype=np.int8)
    )
    return {
        "masses": np.asarray(masses, dtype=np.float64).copy(),
        "stiffness": np.asarray(stiffness, dtype=np.float64).copy(),
        "damping": np.asarray(damping, dtype=np.float64).copy(),
        "actuators": np.asarray(actuators, dtype=np.int8).copy(),
        "sensors": np.asarray(sensors, dtype=np.int8).copy(),
    }


def coerce_embodiment(embodiment: dict, cfg: LQConfig) -> dict[str, np.ndarray]:
    """Normalize an embodiment dict to canonical dtypes/shapes, filling missing keys.

    Missing keys are taken from :func:`base_embodiment`. Raises ``KeyError`` for
    unknown keys and ``ValueError`` for wrong shapes.
    """
    base = base_embodiment(cfg)
    out: dict[str, np.ndarray] = {}
    for key in embodiment:
        if key not in EMB_KEYS:
            raise KeyError(f"unknown embodiment key: {key!r}")
    for key in EMB_KEYS:
        if key in embodiment and embodiment[key] is not None:
            val = np.asarray(embodiment[key])
            if val.shape != (cfg.n,):
                raise ValueError(
                    f"embodiment[{key!r}] must have shape ({cfg.n},), got {val.shape}"
                )
            dtype = np.int8 if key in ("actuators", "sensors") else np.float64
            out[key] = val.astype(dtype).copy()
        else:
            out[key] = base[key]
    return out


def validate_embodiment(embodiment: dict, cfg: LQConfig) -> tuple[bool, str]:
    """Check an embodiment against space bounds and setup constraints.

    Returns ``(ok, reason)``. ``reason`` is empty when ``ok`` is True. This is the
    single source of truth used by the env's ``validate_embodiment`` and reset/step.
    """
    try:
        emb = coerce_embodiment(embodiment, cfg)
    except (KeyError, ValueError) as exc:
        return False, str(exc)

    # Per-element bounds (what the gymnasium Space would check).
    bounds = {
        "masses": (cfg.mass_low, cfg.mass_high),
        "stiffness": (cfg.stiffness_low, cfg.stiffness_high),
        "damping": (cfg.damping_low, cfg.damping_high),
    }
    for key, (lo, hi) in bounds.items():
        v = emb[key]
        if np.any(v < lo - 1e-9) or np.any(v > hi + 1e-9):
            return False, f"{key} out of bounds [{lo}, {hi}]"
    for key in ("actuators", "sensors"):
        v = emb[key]
        if not np.all(np.isin(v, (0, 1))):
            return False, f"{key} must be binary"
    base = base_embodiment(cfg)
    for key, designed in (("actuators", cfg.design_actuators), ("sensors", cfg.design_sensors)):
        if not designed and not np.array_equal(emb[key], base[key]):
            return False, f"{key} placement is fixed in this setup (must equal the base placement)"

    n_act = int(emb["actuators"].sum())
    # Need at least one actuator to be controllable at all.
    if n_act < 1:
        return False, "at least one actuator is required"
    # Partial-observation setups need at least one sensor.
    if cfg.observation_mode == "partial" and int(emb["sensors"].sum()) < 1:
        return False, "at least one sensor is required for partial observation"

    # Budget constraints (Setup 3).
    if cfg.mass_budget is not None and emb["masses"].sum() > cfg.mass_budget + 1e-9:
        return False, f"mass budget exceeded: {emb['masses'].sum():.3f} > {cfg.mass_budget}"
    if cfg.actuator_budget is not None and n_act > cfg.actuator_budget:
        return False, f"actuator budget exceeded: {n_act} > {cfg.actuator_budget}"

    return True, ""


# --- observation padding / masking -------------------------------------------------

def observation_dim(cfg: LQConfig) -> int:
    """Fixed observation length: [measurement(2n)] + [act_mask(n)] + [sen_mask(n)]."""
    return 4 * cfg.n


def build_observation(
    cfg: LQConfig,
    x: np.ndarray,
    emb: dict[str, np.ndarray],
    measurement_noise: np.ndarray | None = None,
) -> np.ndarray:
    """Assemble the padded observation vector.

    - Full mode: the measurement block is the clean full state (2n).
    - Partial mode: the measurement block holds noisy (pos, vel) at sensed masses,
      zero at unsensed masses (fixed 2n layout, masked by the sensor mask).
    Then the actuator and sensor masks are appended, so the policy always knows the
    live layout regardless of placement changes.
    """
    n = cfg.n
    meas = np.zeros(2 * n, dtype=np.float64)
    if cfg.observation_mode == "full":
        meas[:] = x
    else:
        sensors = emb["sensors"].astype(bool)
        sen_idx = np.flatnonzero(sensors)
        for k, i in enumerate(sen_idx):
            noise_p = measurement_noise[2 * k] if measurement_noise is not None else 0.0
            noise_v = measurement_noise[2 * k + 1] if measurement_noise is not None else 0.0
            meas[i] = x[i] + noise_p            # position slot for mass i
            meas[n + i] = x[n + i] + noise_v    # velocity slot for mass i
    obs = np.concatenate([meas, emb["actuators"].astype(np.float64), emb["sensors"].astype(np.float64)])
    return obs.astype(np.float64)
