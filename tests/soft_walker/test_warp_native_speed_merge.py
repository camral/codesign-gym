"""Cache lifetime, saved-root validation and solver-selection merge regressions.

The native runner selects WARP_TEST_DEVICE and WARP_TEST_DTYPE in separate
processes. CUDA tests capture complete programs; no host-stepped CUDA fallback.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from envs.soft_walker.engine.warp_native.model import Bundle


import os
from dataclasses import replace

import numpy as np
import pytest

wp = pytest.importorskip("warp")

from envs.soft_walker.diagnostics.warp.checks import compare_array, contact_fixture
from envs.soft_walker.engine.warp_native.adaptive_gmres import AdaptiveGMRES
from envs.soft_walker.engine.warp_native.adaptive_krylov import AdaptiveCapturableGMRES
from envs.soft_walker.engine.warp_native.config import MPMConfig
from envs.soft_walker.engine.warp_native.krylov import CapturableGMRES
from envs.soft_walker.engine.warp_native.model import DeviceModel, uniform_body
from envs.soft_walker.engine.warp_native.step import ImplicitStep

PRECISION = os.environ.get("WARP_TEST_DTYPE", "float64")
DEVICE = os.environ.get("WARP_TEST_DEVICE", "cpu")


def captured_program(program: Callable[[], None], device: wp.context.Device) -> Callable[[], None]:
    """Return a reusable whole-program launch for either test device."""
    if device.is_cuda:
        assert wp.is_conditional_graph_supported(), "Conditional CUDA graphs required"
        with wp.ScopedCapture(device=device, force_module_load=True) as capture:
            program()

        def run() -> None:
            wp.capture_launch(capture.graph)
            wp.synchronize_device(device)
    else:

        def run() -> None:
            program()
            wp.synchronize_device(device)

    return run


def cached_fixture(dim: int) -> Bundle:
    """Use nonuniform deformation so stale and current tangent operators differ."""
    bundle = contact_fixture(dim, PRECISION, "active_stress")
    bundle.config = replace(bundle.config, cache_tangent=True, krylov_method="adaptive")
    # Nonuniform deformation and velocity make the root-dependent tangent
    # different from a zero-strain/predictor tangent, without changing the law.
    bundle.initial["F"][:, 0, 0] = 1.015
    bundle.initial["F"][:, 1, 1] = 0.985
    bundle.initial["F"][:, 0, 1] = 0.02
    bundle.initial["v"][:, 0] += np.linspace(-0.003, 0.004, len(bundle.initial["v"]))
    return bundle.validate()


@pytest.mark.parametrize(
    "method,expected",
    [
        (None, AdaptiveCapturableGMRES),
        ("legacy", CapturableGMRES),
        ("adaptive", AdaptiveCapturableGMRES),
        ("adaptive_dynamic", AdaptiveGMRES),
    ],
)
def test_solver_factory_selection_preserves_established_default(
    method: None | str, expected: type
) -> None:
    assert MPMConfig().krylov_method == "adaptive"
    bundle = uniform_body(2, precision=PRECISION)
    if method is not None:
        bundle.config = replace(bundle.config, krylov_method=method)
    step = ImplicitStep(DeviceModel(bundle, DEVICE))
    assert type(step.linear) is expected


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize(
    "invalidate_with_prepare", [False, True], ids=["completed_solve", "prepare"]
)
def test_direct_final_root_products_ignore_stale_linearization(
    dim: int, invalidate_with_prepare: bool
) -> None:
    step = ImplicitStep(DeviceModel(cached_fixture(dim), DEVICE))
    step.command.fill_(0.35)
    rng = np.random.default_rng(581 + dim)
    direction = wp.array(
        rng.normal(size=(step.capacity, dim)).astype(PRECISION),
        dtype=step.k.vec,
        device=step.device,
    )
    actual = [wp.zeros_like(direction), wp.zeros_like(direction)]
    reference = [wp.zeros_like(direction), wp.zeros_like(direction)]
    source_force = wp.zeros_like(direction)

    def program() -> None:
        step.st.work_counts.zero_()
        step.forward()
        if invalidate_with_prepare:
            step.refresh_linearization()
            step.prepare()
        # A dead cache must never influence public products after a completed
        # solve/prepare. Poisoning makes a missed invalidation deterministic.
        step.linearization.data.mu.fill_(float("nan"))
        for transpose in (0, 1):
            step.apply_tangent(direction, actual[transpose], actual[transpose], 1.0, 0.0, transpose)
            source_force.zero_()
            step.launch(
                step.k.force_tangent,
                step.owner.n,
                step.model,
                step.state,
                step.st,
                step.cfg,
                step.q,
                direction,
                step.next_a,
                transpose,
                source_force,
            )
            step.launch(
                step.k.tangent_grid,
                step.capacity,
                step.st,
                step.cfg,
                direction,
                source_force,
                reference[transpose],
                step.k.real(1),
                step.k.real(0),
                transpose,
                reference[transpose],
            )

    run = captured_program(program, step.device)
    for command in (0.35, -0.2):
        step.command.fill_(command)
        run()
        assert int(step.code.numpy()[0]) == 0
        assert int(step.output.status.numpy()[0]) == 0
        assert int(step.st.work_counts.numpy()[2]) > 0, "Fixture must exercise tangent caching"
        for transpose in (0, 1):
            compare_array(
                actual[transpose].numpy(),
                reference[transpose].numpy(),
                name="direct_final_root_" + str(transpose),
                precision=PRECISION,
            )


def seeds_for(step: ImplicitStep) -> dict[wp.array, wp.array]:
    """Create deterministic adjoint seeds for position, velocity, actuator, and ledger."""
    rng = np.random.default_rng(743 + step.config.dim)
    x_seed = wp.array(
        rng.normal(size=(step.owner.n, step.config.dim)).astype(PRECISION),
        dtype=step.k.vec,
        device=step.device,
    )
    v_seed = wp.array(
        (0.1 * rng.normal(size=(step.owner.n, step.config.dim))).astype(PRECISION),
        dtype=step.k.vec,
        device=step.device,
    )
    actuator_seed = wp.ones_like(step.output.actuator)
    ledger_seed = wp.array(
        np.array([0.1, 0.2, 0.1, 0.05], PRECISION), dtype=step.k.real, device=step.device
    )
    return {
        step.output.x: x_seed,
        step.output.v: v_seed,
        step.output.actuator: actuator_seed,
        step.output.ledger: ledger_seed,
    }


def input_gradients(step: ImplicitStep) -> dict[str, np.ndarray]:
    """Copy command, initial-state, and material adjoints before buffers are reused."""
    values = {"command": step.command.grad.numpy().copy()}
    for group, obj in (("state", step.state), ("model", step.model)):
        for name, array in vars(obj).items():
            if isinstance(array, wp.array) and array.grad is not None:
                values[group + "_" + name] = array.grad.numpy().copy()
    return values


@pytest.mark.parametrize("dim", [2, 3])
def test_saved_root_reconstruction_checks_and_recovers_before_implicit_vjp(dim: int) -> None:
    reference = ImplicitStep(DeviceModel(cached_fixture(dim), DEVICE))
    reference.command.fill_(0.3)
    reference_seeds = seeds_for(reference)

    def normal_step() -> None:
        reference.forward(record_derivatives=True)
        reference.backward(reference_seeds)

    captured_program(normal_step, reference.device)()
    assert int(reference.code.numpy()[0]) == 0
    assert int(reference.output.status.numpy()[0]) == 0
    accepted_root = reference.q.numpy().copy()
    expected_state = {
        name: getattr(reference.output, name).numpy().copy()
        for name in ("x", "v", "F", "C", "actuator", "ledger")
    }
    expected_gradient = input_gradients(reference)
    assert all(np.isfinite(v).all() for v in expected_gradient.values())
    assert np.linalg.norm(expected_gradient["command"]) > 0

    replay = ImplicitStep(DeviceModel(cached_fixture(dim), DEVICE))
    replay.command.fill_(0.3)
    saved_q = wp.array(accepted_root, dtype=replay.k.vec, device=replay.device)
    saved_code = wp.zeros(1, dtype=int, device=replay.device)
    replay_seeds = seeds_for(replay)

    def load_saved_root() -> None:
        wp.copy(replay.q, saved_q)
        wp.copy(replay.code, saved_code)

    def reconstructed_step() -> None:
        replay.st.work_counts.zero_()
        replay.forward(record_derivatives=True, solved_root_loader=load_saved_root)
        replay.backward(replay_seeds)

    run = captured_program(reconstructed_step, replay.device)
    cases = [
        ("valid", accepted_root, 0),
        ("finite_wrong_root", np.zeros_like(accepted_root), 1),
        ("outside_support", np.full_like(accepted_root, 1.0e6), 1),
        ("nonfinite_root", np.full_like(accepted_root, np.nan), 1),
        ("valid_after_corruption", accepted_root, 0),
    ]
    for label, root, fallback in cases:
        saved_q.assign(root)
        saved_code.zero_()
        run()
        assert int(replay.code.numpy()[0]) == 0, label
        assert int(replay.output.status.numpy()[0]) == 0, label
        counts = replay.st.work_counts.numpy()
        assert counts[5] == 1, (label, "saved root was not checked", counts)
        assert counts[6] == fallback, (label, "unexpected fallback count", counts)
        assert np.isfinite(replay.residual_norm.numpy()).all(), label
        for name, expected in expected_state.items():
            compare_array(
                getattr(replay.output, name).numpy(),
                expected,
                name="reconstructed_" + label + "_" + name,
                precision=PRECISION,
            )
        gradients = input_gradients(replay)
        assert gradients.keys() == expected_gradient.keys()
        for name, expected in expected_gradient.items():
            compare_array(
                gradients[name],
                expected,
                name="gradient_reconstructed_" + label + "_" + name,
                precision=PRECISION,
            )

    # A failed source solve must remain a failure even if its q happens to be
    # finite and accurate. Reconstruction cannot silently rehabilitate status.
    saved_q.assign(accepted_root)
    saved_code.fill_(3)
    run()
    assert int(replay.code.numpy()[0]) != 0
    assert int(replay.output.status.numpy()[0]) != 0
    assert not np.isfinite(replay.command.grad.numpy()).all()
