"""Additional production-solver and captured compact-capacity regression checks."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable


import os
from dataclasses import replace
from itertools import product

import numpy as np
import pytest
import warp as wp

PRECISION = os.environ.get("WARP_TEST_DTYPE", "float64")
DEVICE = os.environ.get("WARP_TEST_DEVICE", "cpu")
RTOL = 4e-4 if PRECISION == "float32" else 2e-8
ATOL = 3e-7 if PRECISION == "float32" else 2e-11


def assert_saved_array_close(actual: np.ndarray, reference: np.ndarray, key: str) -> None:
    if PRECISION == "float32" and key == "C":
        # Near-zero affine off-diagonals accumulate float32 reduction error.
        # Promoted-input float64 diagnostics support this per-particle tensor
        # allowance; preserve the component rtol and all other state policies.
        reference64 = reference.astype(np.float64)
        error = np.abs(actual.astype(np.float64) - reference64)
        scale = np.max(np.abs(reference64), axis=(-2, -1), keepdims=True)
        absolute = np.maximum(ATOL, 32 * np.finfo(np.float32).eps * scale)
        assert np.all(error <= absolute + RTOL * np.abs(reference64)), (key, error, absolute)
    else:
        np.testing.assert_allclose(actual, reference, rtol=RTOL, atol=ATOL, err_msg=key)


def execute(fn: Callable[[], None], device: wp.context.Device) -> None:
    if device.is_cuda:
        with wp.ScopedCapture(device=device, force_module_load=True) as capture:
            fn()
        wp.capture_launch(capture.graph)
    else:
        fn()
    wp.synchronize_device(device)


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("law,contact", [("active_strain", False), ("active_stress", True)])
def test_production_adaptive_all_options_and_changed_commands(
    dim: int, law: str, contact: bool
) -> None:
    from envs.soft_walker.diagnostics.warp.checks import contact_fixture
    from envs.soft_walker.engine.warp_native.episode import Episode
    from envs.soft_walker.engine.warp_native.model import DeviceModel

    bundle = contact_fixture(dim, PRECISION, law, "slope")
    bundle.config = replace(
        bundle.config,
        krylov_method="adaptive",
        cache_tangent=True,
        mass_weighted=True,
        contact_enabled=contact,
    )
    commands = np.array([[0.3], [-0.2]], dtype=PRECISION)
    candidate = replace(
        bundle,
        config=replace(
            bundle.config,
            compact_nodes=24 if dim == 2 else 96,
            linear_work_reuse=True,
            block_preconditioner=True,
        ),
    )
    episodes = [
        Episode(
            DeviceModel(b, DEVICE),
            commands,
            settling_steps=1,
            checkpoint_stride=2,
            velocity_weight=0.13,
            work_weight=0.07,
            reuse_replay_roots=True,
        )
        for b in (bundle, candidate)
    ]
    first = None
    for table in (commands, commands * -0.6, commands):
        results = []
        for episode in episodes:
            episode.commands.assign(table)
            result = episode.run(True)
            assert result["status"] == result["adjoint_status"] == 0, result
            # CPU .numpy() results alias Warp storage; retain independent
            # snapshots before changing commands and replaying the episode.
            arrays = {
                k: getattr(episode.final, k).numpy().copy()
                for k in ("x", "v", "F", "C", "actuator", "ledger")
            }
            gradients = episode.gradients()
            arrays["loss"] = np.array(result["loss"])
            arrays["commands_gradient"] = gradients["commands"].copy()
            arrays.update(
                {
                    f"{g}_{k}": v.copy()
                    for g in ("model", "initial")
                    for k, v in gradients[g].items()
                }
            )
            results.append(arrays)
        for key in results[0]:
            if PRECISION == "float32" and (
                key.startswith(("model_", "initial_")) or key == "commands_gradient"
            ):
                # Preserve the existing float32 MPM gradient norm policy.
                # Volume derivatives subtract nearly equal normalization
                # terms; the changed-command discrepancy remains visible
                # under this policy. Keep the floor for cancellation zeros.
                reference = results[0][key].astype(np.float64)
                error = np.linalg.norm(results[1][key].astype(np.float64) - reference)
                assert (
                    np.allclose(results[1][key], results[0][key], rtol=RTOL, atol=ATOL)
                    or error <= 3e-3 * np.linalg.norm(reference) + 1e-24
                ), (key, error)
            else:
                assert_saved_array_close(results[1][key], results[0][key], key)
        if first is None:
            first = results[1]
    for key in first:
        assert_saved_array_close(results[1][key], first[key], key)


@pytest.mark.parametrize("dim", [2, 3])
def test_captured_exact_capacity_overflow_then_recovery(dim: int) -> None:
    from envs.soft_walker.engine.warp_native.episode import Episode
    from envs.soft_walker.engine.warp_native.model import DeviceModel, uniform_body
    from envs.soft_walker.engine.warp_native.step import ImplicitStep

    bundle = uniform_body(dim, precision=PRECISION)
    probe_owner = DeviceModel(bundle, DEVICE)
    probe = ImplicitStep(probe_owner)
    execute(probe.prepare, probe_owner.device)
    capacity = int(probe.st.live.numpy()[0])
    bundle.config = replace(
        bundle.config,
        compact_nodes=capacity,
        krylov_method="adaptive",
        cache_tangent=True,
        linear_work_reuse=True,
        block_preconditioner=True,
    )
    owner = DeviceModel(bundle, DEVICE)
    initial = {
        k: getattr(owner.initial, k).numpy().copy()
        for k in ("x", "v", "F", "C", "actuator", "ledger")
    }
    episode = Episode(
        owner,
        np.array([[0.1]], dtype=PRECISION),
        checkpoint_stride=1,
        reuse_replay_roots=True,
        velocity_weight=0.1,
    )
    good = episode.run(True)
    assert good["status"] == good["adjoint_status"] == 0, good
    assert int(episode.step.st.live.numpy()[0]) == capacity
    reference = {k: getattr(episode.final, k).numpy().copy() for k in initial}
    reference_gradient = episode.gradients()["commands"].copy()
    spread = np.asarray(list(product((2.5, 5.5), repeat=dim)), dtype=PRECISION)
    spread = (
        np.asarray(bundle.config.origin, dtype=PRECISION) + bundle.config.spacing * spread
    ).astype(PRECISION)
    owner.initial.x.assign(spread)
    invalid = episode.run(True)
    assert invalid["status"] != 0 and invalid["adjoint_status"] != 0
    assert int(episode.step.st.live.numpy()[0]) > capacity
    assert np.all(episode.step.st.ids.numpy() < capacity)
    for key in initial:
        np.testing.assert_array_equal(
            getattr(episode.final, key).numpy(), spread if key == "x" else initial[key]
        )
    with pytest.raises(RuntimeError):
        episode.gradients()
    owner.initial.x.assign(initial["x"])
    recovered = episode.run(True)
    assert recovered["status"] == recovered["adjoint_status"] == 0, recovered
    for key in reference:
        np.testing.assert_allclose(
            getattr(episode.final, key).numpy(), reference[key], rtol=RTOL, atol=ATOL
        )
    np.testing.assert_allclose(
        episode.gradients()["commands"], reference_gradient, rtol=RTOL, atol=ATOL
    )


@pytest.mark.parametrize("dim", [2, 3])
def test_bad_node_blocks_use_scalar_fallback(dim: int) -> None:
    from envs.soft_walker.diagnostics.warp.checks import contact_fixture
    from envs.soft_walker.engine.warp_native.model import DeviceModel
    from envs.soft_walker.engine.warp_native.step import ImplicitStep

    bundle = contact_fixture(dim, PRECISION, "active_stress", "slope")
    bundle.config = replace(
        bundle.config,
        krylov_method="adaptive",
        cache_tangent=True,
        block_preconditioner=True,
        linear_work_reuse=True,
    )
    owner = DeviceModel(bundle, DEVICE)
    step = ImplicitStep(owner)

    def prepare() -> None:
        step.prepare()
        wp.copy(step.q, step.guess)
        step.refresh_linearization()

    execute(prepare, owner.device)
    # Inversion adds identity to the force-derived block. Feed three distinct
    # pathological final blocks and verify finite scalar fallbacks explicitly.
    matrices = np.tile(np.eye(dim, dtype=PRECISION), (step.capacity, 1, 1))
    mass = step.st.mass.numpy()
    rows = np.flatnonzero(mass > 0)[:3]
    assert len(rows) == 3
    matrices[rows[0]] = 0
    matrices[rows[1]].fill(np.nan)
    matrices[rows[2]] = np.diag(np.array([1] + [0] * (dim - 1), dtype=PRECISION))
    # Set dt/m exactly to one for this isolated kernel probe so a singular
    # matrix stays exactly singular rather than becoming roundoff-scaled I.
    mass[rows] = bundle.config.dt
    step.st.mass.assign(mass)
    force_blocks = (np.eye(dim, dtype=PRECISION) - matrices).astype(PRECISION)
    step.block_matrix.assign(force_blocks)
    step.block_fallbacks.zero_()
    threshold = 1e-5 if PRECISION == "float32" else 1e-12
    execute(
        lambda: step.launch(
            step.linearization.kernels.invert_blocks,
            step.capacity,
            step.st,
            step.cfg,
            step.block_matrix,
            step.block_inverse,
            owner.k.real(threshold),
            step.block_fallbacks,
        ),
        owner.device,
    )
    inverse = step.block_inverse.numpy()
    assert int(step.block_fallbacks.numpy()[0]) == 3
    diagonal = step.st.diagonal.numpy()
    assert np.isfinite(inverse).all()
    for row in rows:
        np.testing.assert_allclose(inverse[row], np.eye(dim) / diagonal[row], rtol=RTOL, atol=ATOL)
