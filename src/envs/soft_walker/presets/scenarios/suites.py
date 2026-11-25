"""Small named scenario suites for demonstrations and orthogonal coverage."""

from __future__ import annotations

from dataclasses import replace

from envs.soft_walker.presets.scenarios.config import (
    CONTROLLERS,
    DRIVERS,
    REPRESENTATIONS,
    TERRAINS,
    Scenario,
)

# Inherited pulse-controller guesses, not calibrated hopping guarantees.
AIRBORNE_AMPLITUDE = -0.2  # Dimensionless command amplitude.
AIRBORNE_LIFT = 1.6  # Dimensionless common-mode lift multiplier.


def suite(
    name: str = "drivers", *, tier: str = "coarse", dtype: str = "float64", seed: int = 0
) -> list[Scenario]:
    """Small orthogonal suites, never the full factorial by default."""
    base = Scenario(tier=tier, dtype=dtype, seed=seed)
    if name == "drivers":
        return [
            replace(base, driver=d, gait="walk" if d in ("spring", "mpm2") else "trot")
            for d in DRIVERS
        ]
    if name == "terrains":
        return [replace(base, driver="mpm3", gait="crawl", terrain=t) for t in TERRAINS]
    if name == "representations":
        return [
            replace(base, driver=d, representation=r, gait="walk" if d == "mpm2" else "trot")
            for d in ("mpm2", "mpm3")
            for r in REPRESENTATIONS
        ]
    if name == "controllers":
        return [replace(base, controller=c) for c in CONTROLLERS]
    if name == "airborne":
        return [
            replace(
                base,
                driver=d,
                controller="pulse",
                gait=g,
                amplitude=AIRBORNE_AMPLITUDE,
                lift=AIRBORNE_LIFT,
            )
            for d in DRIVERS
            for g in (("pronk",) if d in ("spring", "mpm2") else ("bound", "pronk"))
        ]
    raise ValueError("suite must be drivers, terrains, representations, controllers, or airborne")
