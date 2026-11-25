"""Native vector wrapper parity and lane isolation on small real physics models.

Set CODESIGN_PARALLEL_TEST_DEVICE=cuda to exercise concurrent CUDA streams;
the default CPU run checks serial fallback and the same public contract.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

wp = pytest.importorskip("warp")
pytest.importorskip("gymnasium")

# Optional backend dependencies are checked before importing the wrapper.
import codesign_gym as cg  # noqa: E402
from envs.soft_walker import config as C  # noqa: E402
from envs.soft_walker.vector_env import SoftWalkerVectorEnv  # noqa: E402

DEVICE = os.environ.get("CODESIGN_PARALLEL_TEST_DEVICE", "cpu")


@pytest.fixture
def tiny_task(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reduce fixture geometry and physical duration without changing wrapper behavior."""
    # Preserve the builders, residuals, tolerances and actual solver; reduce
    # only topology/episode length so wrapper regressions finish promptly.
    monkeypatch.setattr(C, "BASE_GRID", {2: (2, 1), 3: (2, 1, 1)})
    monkeypatch.setattr(C, "CONTROL_DT", 0.0001)
    monkeypatch.setattr(C, "SUBSTEPS", 1)
    monkeypatch.setattr(C, "SETTLE_SECONDS", 0.0)
    monkeypatch.setattr(C, "HORIZON", 3)
    monkeypatch.setattr(C, "PARTICLES_PER_AXIS", 1)
    monkeypatch.setattr(
        C,
        "mpm_grid",
        lambda driver, terrain_name="flat": ((9,) * C.dim_of(driver), (-0.06,) * C.dim_of(driver)),
    )


def create(
    driver: str = "spring", *, execution: str = "auto", device: str = DEVICE, **kwargs: object
) -> SoftWalkerVectorEnv:
    return SoftWalkerVectorEnv(2, driver=driver, device=device, execution=execution, **kwargs)


@pytest.mark.parametrize("driver", C.DRIVERS)
def test_auto_batch_matches_serial_physics(driver: str, tiny_task: None) -> None:
    serial, batch = create(driver, execution="serial"), create(driver)
    try:
        # Different materials and controls must remain independent per lane.
        designs = np.stack([serial.base_embodiment.copy()] * 2)
        # The material layout is public and tested separately; use the actual
        # codec here so this test also remains valid if that layout grows.
        from envs.soft_walker import embodiment as E

        properties = E.to_cells(designs[1], serial.dim)
        properties["young"] *= 1.1
        designs[1] = E.from_cells(properties, serial.dim, serial.dtype)
        assert not serial.reset(seed=11, embodiment=designs)[1].any()
        assert not batch.reset(seed=11, embodiment=designs)[1].any()
        actions = np.zeros(serial.action_space.shape, dtype=serial.dtype)
        actions[0, 0], actions[1, 0] = 0.15, -0.1
        for _ in range(2):
            expected, actual = serial.step(actions), batch.step(actions)
            np.testing.assert_allclose(actual[0], expected[0], rtol=2e-8, atol=2e-10)
            np.testing.assert_allclose(actual[1], expected[1], rtol=2e-8, atol=2e-10)
            np.testing.assert_array_equal(actual[2], expected[2])
            np.testing.assert_array_equal(actual[3], expected[3])
            assert not actual[2].any()
            assert actual[4]["valid"].all()
            expected_mode = "parallel" if wp.get_device(DEVICE).is_cuda else "serial"
            assert np.all(np.asarray(actual[4]["execution_mode"]) == expected_mode)
    finally:
        serial.close()
        batch.close()


@pytest.mark.parametrize("bad", [np.nan, np.inf, 1.01, "bad", 1j])
def test_invalid_actions_do_not_step_or_autoreset_any_lane(
    bad: complex | float | str, tiny_task: None
) -> None:
    env = create()
    try:
        env.reset(seed=3)
        env._pending[0] = True
        owners = tuple(lane.owner for lane in env.lanes)
        steps = tuple(lane._steps for lane in env.lanes)
        actions = np.zeros(
            env.action_space.shape, dtype=complex if isinstance(bad, complex) else object
        )
        actions[1, 0] = bad
        # Numeric cases should exercise the finite/bounds predicates as well.
        if not isinstance(bad, (str, complex)):
            actions = actions.astype(float)
        with pytest.raises(ValueError, match="finite"):
            env.step(actions)
        assert tuple(lane._steps for lane in env.lanes) == steps
        assert all(lane.owner is owner for lane, owner in zip(env.lanes, owners))
        np.testing.assert_array_equal(env._pending, [True, False])
    finally:
        env.close()


def test_mixed_autoreset_and_masked_reset_preserve_lane_order(
    tiny_task: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(C, "HORIZON", 2)
    env = create()
    try:
        env.reset(seed=1)
        actions = np.zeros(env.action_space.shape)
        env.step(actions)
        untouched = env.lanes[0].owner
        env.reset(seed=[7, 9], options={"reset_mask": np.array([False, True])})
        assert env.lanes[0].owner is untouched and env.lanes[0]._steps == 1
        np.testing.assert_array_equal(env.step(actions)[3], [True, False])
        result = env.step(actions)
        np.testing.assert_array_equal(result[4]["autoreset"], [True, False])
        np.testing.assert_array_equal(result[3], [False, True])
        assert result[1][0] == 0.0
        assert [lane._steps for lane in env.lanes] == [0, 2]
    finally:
        env.close()


def test_minimum_integer_action_is_rejected_before_mutation(tiny_task: None) -> None:
    env = create()
    try:
        env.reset(seed=3)
        actions = np.zeros(env.action_space.shape, dtype=np.int64)
        actions[1, 0] = np.iinfo(np.int64).min
        with pytest.raises(ValueError, match="bounded"):
            env.step(actions)
        assert [lane._steps for lane in env.lanes] == [0, 0]
    finally:
        env.close()


def test_invalid_lane_lifecycle_does_not_reset_other_lanes(tiny_task: None) -> None:
    from gymnasium.error import ResetNeeded

    env = create()
    try:
        env.reset(seed=3)
        env._pending[0] = True
        owner = env.lanes[0].owner
        env.lanes[1].close()
        with pytest.raises(ResetNeeded):
            env.step(np.zeros(env.action_space.shape))
        assert env.lanes[0].owner is owner and env._pending[0]
    finally:
        env.close()


def test_all_autoreset_skips_physics_batch(
    tiny_task: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    from envs.soft_walker.engine.warp_native import vector_env as native

    monkeypatch.setattr(C, "HORIZON", 1)
    env = create()
    try:
        env.reset(seed=1)
        actions = np.zeros(env.action_space.shape)
        assert env.step(actions)[3].all()

        def forbidden(*args: object, **kwargs: object) -> None:
            raise AssertionError("No active lanes may submit a physics batch")

        monkeypatch.setattr(native, "step_lanes", forbidden)
        result = env.step(actions)
        assert result[4]["autoreset"].all()
        assert not result[1].any() and not result[2].any() and not result[3].any()
    finally:
        env.close()


def test_failed_lane_does_not_stop_healthy_lane(tiny_task: None) -> None:
    env = create()
    try:
        env.reset(seed=2)
        # A persisted failure enters through the actual episode's state/status
        # path. It must not poison another lane's solver or autoreset lifecycle.
        failed = env.lanes[0].owner.initial
        before = failed.u.numpy().copy()
        wp.copy(failed.status, wp.array([1], dtype=wp.int32, device=DEVICE))
        result = env.step(np.zeros(env.action_space.shape))
        np.testing.assert_array_equal(result[2], [True, False])
        np.testing.assert_array_equal(result[4]["valid"], [False, True])
        np.testing.assert_array_equal(env.lanes[0].owner.initial.u.numpy(), before)
        assert env.lanes[1]._steps == 1
        healthy = env.lanes[1].owner
        next_result = env.step(np.zeros(env.action_space.shape))
        np.testing.assert_array_equal(next_result[4]["autoreset"], [True, False])
        assert env.lanes[1].owner is healthy and env.lanes[1]._steps == 2
    finally:
        env.close()


def test_execution_option_and_partial_constructor_cleanup(tiny_task: None) -> None:
    for execution in ("bad", "parallel"):
        with pytest.raises(ValueError):
            create(execution=execution, device="cpu")
    with pytest.raises(ValueError):
        SoftWalkerVectorEnv(0, driver="spring", device="cpu")
    env, _ = cg.make_vec("SoftWalker-Spring2D", num_envs=2, device="cpu", execution="serial")
    assert env.execution == "serial"
    env.close()
    env.close()
    with pytest.raises(RuntimeError, match="closed"):
        env.reset()
    with pytest.raises(RuntimeError, match="closed"):
        env.step(np.zeros(env.action_space.shape))


def test_render_design_and_returns_survive_batched_partial_reset(
    tiny_task: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Upstream rendering keeps the right design and return for each lane."""
    from envs.soft_walker import embodiment as E, render as R, render3d as R3
    monkeypatch.setattr(C, "HORIZON", 2)
    env, base = cg.make_vec("SoftWalker-Spring2D", num_envs=2, device=DEVICE,
                            execution="auto", render_mode="rgb_array")
    try:
        props = E.to_cells(base.copy(), env.dim)
        props["channel"][:] = 2
        variant = E.from_cells(props, env.dim, env.dtype)
        designs = np.stack([base, variant])
        assert not env.reset(seed=3, embodiment=designs)[1].any()
        actions = np.zeros(env.action_space.shape)
        actions[:, : env.channels] = 0.1
        first = env.step(actions)
        np.testing.assert_allclose(env._returns, first[1])
        frames = env.render()
        shape = frames[0].shape  # Newton 3D where an OpenGL context opens, else the Pillow side view
        assert shape in ((R3.HEIGHT, R3.WIDTH, 3), (R.HEIGHT, R.WIDTH, 3))
        assert len(frames) == 2 and all(frame.shape == shape for frame in frames)
        np.testing.assert_array_equal(env.render(env_idx=1), frames[1])
        env.reset(options={"reset_mask": np.array([False, True])}, embodiment=designs)
        np.testing.assert_allclose(env._returns, [first[1][0], 0.0])
        np.testing.assert_array_equal(env._designs, designs)
        second = env.step(actions)
        np.testing.assert_allclose(env._returns, [first[1][0] + second[1][0], second[1][1]])
        third = env.step(actions)
        np.testing.assert_array_equal(third[4]["autoreset"], [True, False])
        np.testing.assert_allclose(env._returns, [0.0, second[1][1] + third[1][1]])
        np.testing.assert_array_equal(env._designs, designs)
        assert env.render(env_idx=0).shape == shape
    finally:
        env.close()
    single, base = cg.make("SoftWalker-Spring2D", device=DEVICE, render_mode="rgb_array", execution="auto")
    try:
        single.reset(seed=3, embodiment=base)
        assert single.render().shape == shape
    finally:
        single.close()
