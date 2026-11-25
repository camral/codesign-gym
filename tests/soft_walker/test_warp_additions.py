"""Focused port/API and compact/preconditioner regressions, not a full-suite run."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from envs.soft_walker.engine.warp_native.model import DeviceModel
    from envs.soft_walker.engine.warp_native.step import ImplicitStep


import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import warp as wp

PRECISION = os.environ.get("WARP_TEST_DTYPE", "float64")
DEVICE = os.environ.get("WARP_TEST_DEVICE", "cpu")
RTOL = 4e-4 if PRECISION == "float32" else 2e-8
ATOL = 3e-7 if PRECISION == "float32" else 2e-11


def execute(fn: Callable[[], None], device: wp.context.Device) -> None:
    if device.is_cuda:
        with wp.ScopedCapture(device=device, force_module_load=True) as capture:
            fn()
        wp.capture_launch(capture.graph)
    else:
        fn()
    wp.synchronize_device(device)


def mpm_fixture(dim: int = 2, **options: object) -> tuple[DeviceModel, ImplicitStep]:
    from envs.soft_walker.diagnostics.warp.checks import contact_fixture
    from envs.soft_walker.engine.warp_native.model import DeviceModel
    from envs.soft_walker.engine.warp_native.step import ImplicitStep

    bundle = contact_fixture(dim, PRECISION, "active_stress", "slope")
    bundle.config = replace(
        bundle.config, krylov_method="adaptive_dynamic", cache_tangent=True, **options
    )
    owner = DeviceModel(bundle, DEVICE)
    return owner, ImplicitStep(owner)


@pytest.mark.parametrize("dim", [2, 3])
def test_compact_block_episode_state_and_gradient_parity(dim: int) -> None:
    from envs.soft_walker.engine.warp_native.episode import Episode

    compact = 24 if dim == 2 else 96
    values = []
    for options in (
        {},
        dict(compact_nodes=compact, linear_work_reuse=True, block_preconditioner=True),
    ):
        owner, _ = mpm_fixture(dim, **options)
        episode = Episode(
            owner,
            np.array([[0.3], [-0.2]], PRECISION),
            settling_steps=1,
            checkpoint_stride=2,
            velocity_weight=0.13,
            work_weight=0.07,
            reuse_replay_roots=True,
        )
        result = episode.run(True)
        assert result["status"] == 0 and result["adjoint_status"] == 0, result
        gradient = episode.gradients()
        values.append(
            (
                {
                    key: getattr(episode.final, key).numpy()
                    for key in ("x", "v", "F", "C", "actuator", "ledger", "status")
                },
                gradient,
                result,
            )
        )
    for key in values[0][0]:
        np.testing.assert_allclose(values[0][0][key], values[1][0][key], rtol=RTOL, atol=ATOL)
    for group in ("model", "initial"):
        for key in values[0][1][group]:
            np.testing.assert_allclose(
                values[0][1][group][key], values[1][1][group][key], rtol=RTOL, atol=ATOL
            )
    np.testing.assert_allclose(
        values[0][1]["commands"], values[1][1]["commands"], rtol=RTOL, atol=ATOL
    )
    assert values[1][2]["solver_capacity"] <= compact


@pytest.mark.parametrize("dim", [2, 3])
def test_compact_overflow_is_transactional_not_truncation(dim: int) -> None:
    owner, step = mpm_fixture(dim, compact_nodes=1, linear_work_reuse=True)
    before = {
        key: getattr(owner.initial, key).numpy()
        for key in ("x", "v", "F", "C", "actuator", "ledger", "status")
    }
    execute(lambda: step.forward(), owner.device)
    assert step.st.live.numpy()[0] > step.capacity
    assert step.output.status.numpy()[0] != 0
    for key in before:
        if key != "status":
            np.testing.assert_array_equal(getattr(step.output, key).numpy(), before[key])
    assert np.all(step.st.ids.numpy() < step.capacity)


@pytest.mark.parametrize("dim", [2, 3])
def test_node_block_is_actual_diagonal_and_transpose(dim: int) -> None:
    owner, step = mpm_fixture(
        dim, compact_nodes=24 if dim == 2 else 96, linear_work_reuse=True, block_preconditioner=True
    )

    def prepare() -> None:
        step.prepare()
        wp.copy(step.q, step.guess)
        step.refresh_linearization()

    execute(prepare, owner.device)
    inverse = step.block_inverse.numpy()
    # Probe two nontrivial positive-mass rows, not every global basis column.
    live_rows = np.flatnonzero(step.st.mass.numpy() > 0)
    for row in live_rows[[0, len(live_rows) // 2]]:
        block = np.empty((dim, dim), PRECISION)
        x = wp.zeros_like(step.q)
        y = wp.zeros_like(step.q)
        for axis in range(dim):
            basis = np.zeros((step.capacity, dim), PRECISION)
            basis[row, axis] = 1
            x.assign(basis)
            execute(lambda: step.matvec(x, y, y), owner.device)
            block[:, axis] = y.numpy()[row]
        np.testing.assert_allclose(inverse[row] @ block, np.eye(dim), rtol=RTOL, atol=ATOL)
    rng = np.random.default_rng(3)
    x = wp.array(
        rng.normal(size=(step.capacity, dim)).astype(PRECISION),
        dtype=owner.k.vec,
        device=owner.device,
    )
    y = wp.zeros_like(x)
    step._transpose_solve = True
    execute(lambda: step.precondition(x, y, y), owner.device)
    np.testing.assert_allclose(
        y.numpy(), np.einsum("nji,nj->ni", inverse, x.numpy()), rtol=RTOL, atol=ATOL
    )


def test_refinement_reports_accepted_solution_true_residual() -> None:
    owner, step = mpm_fixture(2, linear_work_reuse=True, linear_refinements=3)
    step.cfg.linear_limit = 1e-13 if PRECISION == "float64" else 1e-10
    rng = np.random.default_rng(211)
    rhs = rng.normal(size=(step.capacity, 2)).astype(PRECISION)
    step.rhs.assign(rhs)

    def solve() -> None:
        step.prepare()
        wp.copy(step.q, step.guess)
        step.linear_solve()

    execute(solve, owner.device)
    assert step.refine_count.numpy()[0] > 0
    reported = step.sums.numpy().copy()

    # The accepted-solution check uses the same fixed-q cached operator as
    # GMRES. Do not compare a squared roundoff-level residual to a separately
    # ordered uncached product with an arbitrarily tight relative tolerance.
    def accepted_product() -> None:
        step.refresh_linearization()
        step.matvec(step.delta, step.linear_ax, step.linear_ax)

    execute(accepted_product, owner.device)
    accepted_ax = step.linear_ax.numpy().copy()
    true = np.sum((accepted_ax.astype(np.float64) - rhs.astype(np.float64)) ** 2)
    np.testing.assert_allclose(
        reported[0],
        true,
        rtol=3e-3 if PRECISION == "float32" else 2e-5,
        atol=1e-18 if PRECISION == "float32" else 1e-25,
    )
    # Also check the original, uncached expression, with a normwise allowance
    # only for the different floating-point ordering. Solver tolerances remain
    # unchanged; this artificial refinement test requested sub-roundoff error.
    step._linearization_ready = False
    execute(lambda: step.matvec(step.delta, step.linear_ax, step.linear_ax), owner.device)
    difference = np.linalg.norm(
        step.linear_ax.numpy().astype(np.float64) - accepted_ax.astype(np.float64)
    )
    assert difference <= 64 * np.finfo(PRECISION).eps * max(np.linalg.norm(rhs), 1.0)


@pytest.mark.parametrize("kind", ["spring", "beam", "fem"])
def test_native_solid_bundle_roundtrip_and_history_gradient(kind: str, tmp_path: Path) -> None:
    from envs.soft_walker.engine.warp_native.solid_builders import regular_solid
    from envs.soft_walker.engine.warp_native.solid_episode import SolidEpisode
    from envs.soft_walker.engine.warp_native.solid_model import SolidBundle, SolidDeviceModel

    bundle = regular_solid(kind, precision=PRECISION)
    bundle.save(tmp_path / "model.npz")
    restored = SolidBundle.load(tmp_path / "model.npz")
    for key in bundle.fields:
        np.testing.assert_array_equal(restored.fields[key], bundle.fields[key])
    owner = SolidDeviceModel(restored, DEVICE)
    commands = np.resize(np.array([0.3, -0.2], PRECISION), (2, owner.channels))
    episode = SolidEpisode(
        owner,
        commands,
        settling_steps=1,
        checkpoint_stride=2,
        velocity_weight=0.1,
        work_weight=0.05,
    )
    result = episode.run(True)
    assert result["status"] == 0 and result["adjoint_status"] == 0, result
    first = episode.gradients()["commands"].copy()
    result = episode.run(True)
    np.testing.assert_allclose(episode.gradients()["commands"], first, rtol=RTOL, atol=ATOL)
    assert np.all(np.isfinite(first))
    assert episode.recording()["u"].shape[0] == episode.blocks + 1


@pytest.mark.parametrize(
    "kind,env_id",
    [
        ("spring", "SoftSpring2D-v0"),
        ("beam", "SoftBeam3D-v0"),
        ("fem", "SoftHexFEM3D-v0"),
        ("mpm2", "SoftMPM2D-v0"),
        ("mpm3", "SoftMPM3D-v0"),
    ],
)
def test_all_driver_public_api_and_rejected_reset(kind: str, env_id: str) -> None:
    from envs.soft_walker.engine import make

    env, base = make(
        env_id,
        backend="warp",
        device=DEVICE,
        dtype=PRECISION,
        horizon=2,
        substeps=1,
        settling_steps=0,
        observation_profile="resolved",
    )
    obs, rejected, info = env.reset(embodiment=base, seed=13)
    assert not rejected and np.all(np.isfinite(obs))
    action = np.zeros(env.action_space.shape, PRECISION)
    obs, reward, terminated, truncated, info = env.step(action)
    assert not terminated and not truncated and info["valid"]
    assert np.all(np.isfinite(obs)) and np.isfinite(reward)
    invalid = {k: v.copy() for k, v in base.items()}
    key = "young" if kind.startswith("mpm") else "mass"
    invalid[key].fill(-1)
    _, rejected, _ = env.reset(embodiment=invalid)
    assert rejected
    with pytest.raises(Exception):
        env.step(action)
    env.close()


def test_native_namespace_does_not_import_jax() -> None:
    code = """import sys
class Block:
    def find_spec(self,name,path=None,target=None):
        if name=="jax" or name.startswith("jax."):
            raise RuntimeError("Native path imported JAX")
sys.meta_path.insert(0,Block())
from envs.soft_walker.engine import make
from envs.soft_walker.engine.warp_native.solid_builders import regular_solid
from envs.soft_walker.engine.warp_native.solid_episode import SolidEpisode
for kind in ("spring","beam","fem"):
    regular_solid(kind)
assert "jax" not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_native_vector_next_step_autoreset() -> None:
    from envs.soft_walker.engine import make_vec

    env, base = make_vec(
        "SoftSpring2D-v0",
        backend="warp",
        num_envs=2,
        dtype=PRECISION,
        device=DEVICE,
        horizon=1,
        vectorization_mode="warp",
    )
    obs, rejected, info = env.reset(embodiment=base, seed=91)
    assert not rejected.any()
    action = np.zeros(env.action_space.shape, PRECISION)
    first = env.step(action)
    assert first[3].all() and not first[2].any()
    second = env.step(action)
    assert not second[2].any() and not second[3].any()
    assert np.all(second[1] == 0.0)
    assert all(v["autoreset"] for v in second[4]["per_env"])
    env.close()


@pytest.mark.parametrize("kind", ["spring", "beam", "fem"])
def test_local_force_rigid_motion_objectivity(kind: str) -> None:
    from envs.soft_walker.engine.warp_native.solid_builders import regular_solid
    from envs.soft_walker.engine.warp_native.solid_model import SolidDeviceModel
    from envs.soft_walker.engine.warp_native.solid_step import SolidStep

    bundle = regular_solid(kind, precision=PRECISION)
    bundle.config = replace(bundle.config, gravity=0.0, contact_enabled=False)
    owner = SolidDeviceModel(bundle, DEVICE)
    d = owner.config.dim
    angle = 0.21
    rotation = np.eye(d, dtype=PRECISION)
    rotation[:2, :2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    ref = bundle.fields["reference"]
    owner.initial.u.assign((ref @ rotation.T + 0.02 - ref).astype(PRECISION))
    owner.initial.rotation.assign(np.tile(rotation, (owner.n, 1, 1)))
    step = SolidStep(owner)

    def program() -> None:
        step.prepare()
        step.q.zero_()
        step.evaluate(step.q, step.residual)

    execute(program, owner.device)
    force = step.force_buffer.numpy()
    # The test's forces are O(10 N) under a macroscopic strain. A rigid motion
    # should leave only arithmetic roundoff, not an orientation-dependent spring.
    assert np.max(np.abs(force)) < (4e-5 if PRECISION == "float32" else 1e-11), force


def test_zero_start_does_not_execute_empty_restart_budget() -> None:
    from envs.soft_walker.engine.warp_native.model import DeviceModel
    from envs.soft_walker.engine.warp_native.step import ImplicitStep

    owner, _ = mpm_fixture(2, linear_work_reuse=True)
    bundle = owner.bundle if hasattr(owner, "bundle") else None
    from envs.soft_walker.diagnostics.warp.checks import contact_fixture

    bundle = contact_fixture(2, PRECISION, "active_stress", "slope")
    bundle.config = replace(
        bundle.config, krylov_method="adaptive", cache_tangent=True, linear_work_reuse=True
    )
    owner = DeviceModel(bundle, DEVICE)
    step = ImplicitStep(owner)

    def program() -> None:
        step.prepare()
        wp.copy(step.q, step.guess)
        step.rhs.zero_()
        step.st.work_counts.zero_()
        step.linear_solve()

    execute(program, owner.device)
    # No A@0 and no Arnoldi/restart work. Retain the independent true check.
    assert step.st.work_counts.numpy()[0] == 1
    np.testing.assert_array_equal(step.delta.numpy(), 0.0)
    assert step.sums.numpy()[0] == 0.0


def test_vector_rejected_reset_never_autorestores_old_embodiment() -> None:
    from envs.soft_walker.engine import make_vec
    from envs.soft_walker.engine.spaces import ResetNeeded

    env, base = make_vec(
        "SoftSpring2D-v0", backend="warp", num_envs=2, dtype=PRECISION, device=DEVICE, horizon=1
    )
    env.reset(embodiment=base)
    actions = np.zeros(env.action_space.shape, PRECISION)
    env.step(actions)
    invalid = {key: value.copy() for key, value in base.items()}
    invalid["mass"][1] = 0
    _, rejected, _ = env.reset(embodiment=invalid)
    assert not rejected[0] and rejected[1]
    with pytest.raises(ResetNeeded):
        env.step(actions)
    assert env.envs[0]._steps == 0
    env.close()
