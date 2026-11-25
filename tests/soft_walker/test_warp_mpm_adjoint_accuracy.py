"""MPM adjoint refinement accuracy with unchanged gradient acceptance limits."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from envs.soft_walker.engine.warp_native.model import Bundle


import os
from dataclasses import replace

import numpy as np
import pytest

from envs.soft_walker.diagnostics.warp.checks import contact_fixture
from envs.soft_walker.engine.warp_native.episode import Episode
from envs.soft_walker.engine.warp_native.model import DeviceModel
from envs.soft_walker.engine.warp_native.step import ImplicitStep

DEVICE = os.environ.get("WARP_TEST_DEVICE", "cpu")
PRECISION = os.environ.get("WARP_TEST_DTYPE", "float64")


def promote(bundle: Bundle) -> Bundle:
    """Preserve the actual float32 physical inputs in the float64 reference."""

    def group(values: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {
            key: np.asarray(value).astype(
                np.float64 if np.asarray(value).dtype.kind == "f" else np.asarray(value).dtype,
                copy=True,
            )
            for key, value in values.items()
        }

    names = (
        "spacing",
        "dt",
        "gravity",
        "velocity_ref",
        "actuator_tau",
        "max_log_stretch",
        "active_stress_ratio",
        "min_J",
        "contact_height",
        "contact_stiffness",
        "contact_width",
        "friction",
        "friction_width",
        "armijo",
    )
    cfg = {name: float(np.float32(getattr(bundle.config, name))) for name in names}
    cfg["origin"] = tuple(float(np.float32(x)) for x in bundle.config.origin)
    return replace(
        bundle,
        fields=group(bundle.fields),
        initial=group(bundle.initial),
        terrain=group(bundle.terrain),
        config=replace(bundle.config, **cfg),
    ).validate()


@pytest.mark.parametrize("dim", [2, 3])
def test_changed_command_volume_adjoint_against_promoted_precision(dim: int) -> None:
    """The old float32 path misses the changed-table volume-gradient target."""
    if PRECISION != "float32":
        pytest.skip("This comparison specifically checks float32 against promoted float64")
    b = contact_fixture(dim, "float32", "active_strain", "slope")
    b.config = replace(
        b.config,
        krylov_method="adaptive",
        cache_tangent=True,
        mass_weighted=True,
        contact_enabled=False,
    )
    cmd = np.asarray([[0.3], [-0.2]], np.float32)
    weights = {
        k: float(np.float32(v))
        for k, v in [("length_ref", 0.2), ("velocity_weight", 0.13), ("work_weight", 0.07)]
    }
    reference = Episode(
        DeviceModel(promote(b), DEVICE),
        cmd.astype("float64"),
        settling_steps=1,
        checkpoint_stride=2,
        reuse_replay_roots=True,
        **weights,
    )
    variants = [
        b,
        replace(
            b,
            config=replace(
                b.config,
                compact_nodes=24 if dim == 2 else 96,
                linear_work_reuse=True,
                block_preconditioner=True,
            ),
        ),
    ]
    episodes = [
        Episode(
            DeviceModel(v, DEVICE),
            cmd,
            settling_steps=1,
            checkpoint_stride=2,
            reuse_replay_roots=True,
            **weights,
        )
        for v in variants
    ]
    for table_index, table in enumerate((cmd, cmd * np.float32(-0.6), cmd)):
        reference.commands.assign(table.astype("float64"))
        result = reference.run(True)
        assert result["status"] == result["adjoint_status"] == 0
        expected = reference.gradients()["model"]["volume"].copy()
        for variant, episode in enumerate(episodes):
            episode.commands.assign(table)
            result = episode.run(True)
            assert result["status"] == result["adjoint_status"] == 0
            actual = episode.gradients()["model"]["volume"].astype("float64").copy()
            # Same existing MPM gradient norm policy, with no added floor.
            assert np.linalg.norm(actual - expected) <= 3e-3 * np.linalg.norm(expected) + 1e-24, (
                dim,
                table_index,
                variant,
                actual,
                expected,
            )


def test_refinement_target_is_mpm_precision_specific() -> None:
    bundle = contact_fixture(2, PRECISION)
    owner = DeviceModel(bundle, DEVICE)
    step = ImplicitStep(owner)
    expected = (
        min(owner.parameters.linear_limit, 4 * np.finfo(np.float32).eps)
        if PRECISION == "float32"
        else owner.parameters.linear_limit
    )
    assert step.adjoint_refinement_target == expected
    assert step.config.linear_refinements == bundle.config.linear_refinements
    assert step.cfg.linear_limit == owner.parameters.linear_limit
