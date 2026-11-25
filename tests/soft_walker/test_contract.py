"""Public scalar/vector environment contracts and rejected-input behavior."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from envs.soft_walker.engine.drivers.spring_environment import Spring2DEnv


from copy import deepcopy
from dataclasses import replace

import jax
import numpy as np
import pytest

from envs.soft_walker.engine import make, make_vec, registered_envs
from envs.soft_walker.engine.embodiments.decoders import FieldSpec, GridDecoder
from envs.soft_walker.engine.physics.spring_config import SpringConfig
from envs.soft_walker.engine.spaces import USING_GYMNASIUM, ResetNeeded


def make_small(
    dtype: np.dtype[np.floating] | type[np.floating], small_config: SpringConfig, **kw: object
) -> tuple[Spring2DEnv, dict[str, np.ndarray]]:
    return make(dtype=dtype, config=small_config, max_observation_space_size=128, **kw)


def test_factory_registry(dtype: np.dtype[np.floating], small_config: SpringConfig) -> None:
    assert len(registered_envs()) == 5
    with pytest.raises(ValueError):
        make("not-an-env")
    for id in list(registered_envs())[1:]:
        assert "implemented" in registered_envs()[id]
    # backend="warp" routes to the native Warp implementation (no JAX fallback)
    native, native_base = make(backend="warp", dtype=dtype.name, device="cpu")
    assert type(native).__name__ == "NativeEnv" and native.validate_embodiment(native_base)
    native.close()
    env, base = make_small(dtype, small_config)
    assert env.embodiment_space.contains(base)
    assert env.validate_embodiment(base)
    assert type(env.validate_embodiment(base)) is bool
    base["values"][...] = 6
    assert not np.all(env._base["values"] == 6)
    env.close()
    env.close()
    with pytest.raises(RuntimeError):
        env.reset()


def test_arities_observations_and_metrics(
    dtype: np.dtype[np.floating], small_config: SpringConfig
) -> None:
    env, base = make_small(dtype, small_config)
    out = env.reset(seed=11, embodiment=base)
    assert len(out) == 3 and out[1].shape == (1,) and out[1].dtype == np.bool_
    assert not out[1][0] and env.observation_space.contains(out[0])
    assert out[0].dtype == dtype
    assert out[2]["observation_mask"].shape == env.observation_space.shape
    rewards = []
    for k in range(small_config.horizon):
        result = env.step(np.array([0.2, -0.1, 0.6, -0.3], dtype=dtype))
        assert len(result) == 5
        obs, reward, term, trunc, info = result
        rewards.append(reward)
        assert not term and trunc == (k == small_config.horizon - 1)
        assert env.observation_space.contains(obs)
        assert info["valid"] and info["work_positive_j"] >= 0
        assert info["solver"]["accepted"] == small_config.substeps
    expected = -info["objective"][0] - small_config.energy_weight * info["objective"][1]
    np.testing.assert_allclose(sum(rewards), expected, atol=2e-7 if dtype == np.float32 else 1e-13)
    with pytest.raises(ResetNeeded):
        env.step(np.zeros(4, dtype=dtype))


@pytest.mark.parametrize("bad", ["nan", "bounds", "shape", "keys", "dtype"])
def test_rejection_cannot_reuse_state(
    dtype: np.dtype[np.floating], small_config: SpringConfig, bad: str
) -> None:
    env, base = make_small(dtype, small_config)
    env.reset(embodiment=base)
    b = deepcopy(base)
    if bad == "nan":
        b["values"].flat[0] = np.nan
    elif bad == "bounds":
        b["values"].flat[0] = 7
    elif bad == "shape":
        b["values"] = b["values"][:-1]
    elif bad == "keys":
        b["poisson_ratio"] = 0.35
    else:
        b["values"] = b["values"].astype(complex)
    assert not env.validate_embodiment(b)
    obs, rejected, info = env.reset(embodiment=b)
    assert rejected[0] and info["embodiment_errors"] and np.all(obs == 0)
    with pytest.raises(ResetNeeded):
        env.step(np.zeros(4, dtype=dtype))
    assert not env.reset(embodiment=base)[1][0]


@pytest.mark.parametrize("capacity", ["action", "observation"])
def test_capacity_errors(
    dtype: np.dtype[np.floating], small_config: SpringConfig, capacity: str
) -> None:
    with pytest.raises(ValueError, match="CAPACITY"):
        make(
            dtype=dtype,
            config=small_config,
            **(
                {"max_action_space_size": 1}
                if capacity == "action"
                else {"max_observation_space_size": 1}
            ),
        )


@pytest.mark.parametrize(
    "action",
    [np.zeros(3), np.ones(4) * 1.1, np.array([0, np.nan, 0, 0]), np.array([0, 0, 0, np.inf])],
)
def test_actions_rejected_atomically(
    dtype: np.dtype[np.floating], small_config: SpringConfig, action: np.ndarray | list[float]
) -> None:
    env, base = make_small(dtype, small_config)
    env.reset(embodiment=base)
    old = env._state
    with pytest.raises(ValueError):
        env.step(action)
    np.testing.assert_array_equal(env._state.x, old.x)


@pytest.mark.parametrize("evaluation", [False, True])
def test_unsupported_step_update_warns_and_ignores(
    dtype: np.dtype[np.floating],
    small_config: SpringConfig,
    evaluation: bool,
    tolerance: float,
) -> None:
    a, base = make_small(dtype, small_config, is_eval_env=evaluation)
    b, _ = make_small(dtype, small_config, is_eval_env=evaluation)
    a.reset(embodiment=base)
    b.reset(embodiment=base)
    u = np.array([0.1, 0.2, 0, 0], dtype=dtype)
    with pytest.warns(UserWarning, match="ignored"):
        actual = a.step(u, embodiment={"arbitrary": "even invalid, unsupported"})
    expected = b.step(u)
    np.testing.assert_allclose(actual[0], expected[0], rtol=tolerance, atol=tolerance)
    np.testing.assert_array_equal(a._model.young, b._model.young)


def test_padding_invariance(
    dtype: np.dtype[np.floating], small_config: SpringConfig, tolerance: float
) -> None:
    a, base = make(
        dtype=dtype, config=small_config, max_action_space_size=2, max_observation_space_size=128
    )
    b, _ = make(
        dtype=dtype, config=small_config, max_action_space_size=20, max_observation_space_size=1024
    )
    ra = a.reset(seed=7, embodiment=base)
    rb = b.reset(seed=7, embodiment=base)
    np.testing.assert_allclose(ra[0], rb[0][:128], rtol=tolerance, atol=tolerance)
    for _ in range(2):
        x = a.step(np.array([0.2, -0.1], dtype=dtype))
        y = b.step(np.array([0.2, -0.1] + [0.93] * 18, dtype=dtype))
        np.testing.assert_allclose(x[0], y[0][:128], rtol=tolerance, atol=tolerance)
        assert np.all(y[0][128:] == 0)
        np.testing.assert_allclose(
            x[4]["objective"], y[4]["objective"], rtol=tolerance, atol=tolerance
        )
        assert not np.any(y[4]["action_mask"][2:])


def test_seed_replay_and_validation_side_effect_free(
    dtype: np.dtype[np.floating], small_config: SpringConfig, tolerance: float
) -> None:
    env, b = make_small(dtype, small_config)
    outputs = []
    for repeat in range(2):
        env.reset(seed=132, embodiment=b)
        x = []
        for _ in range(2):
            state = env._state
            assert env.validate_embodiment(b)
            assert env._state is state
            x.append(env.step(np.array([0.1, -0.2, 0, 0], dtype=dtype))[0])
        outputs.append(np.stack(x))
    if jax.default_backend() == "cpu":
        np.testing.assert_array_equal(*outputs)
    else:
        np.testing.assert_allclose(*outputs, rtol=tolerance, atol=tolerance)


def test_empty_disconnected_and_vertex_geometry(dtype: np.dtype[np.floating]) -> None:
    d = GridDecoder(FieldSpec(), shape=(2, 2), dtype=dtype)
    cfg = SpringConfig(grid_shape=(2, 2), settle_steps=0)
    e, b = make(dtype=dtype, decoder=d, config=cfg)
    b["values"][..., 0] = -6
    assert "EMPTY_GEOMETRY" in e.validation_report(b)
    b["values"][..., 0] = np.array([[6, -6], [-6, 6]], dtype=dtype)
    reasons = e.validation_report(b)
    assert "DISCONNECTED_GEOMETRY" in reasons and "VERTEX_ONLY_JUNCTION" in reasons


def test_vector_scalar_and_next_step_autoreset(
    dtype: np.dtype[np.floating], small_config: SpringConfig, tolerance: float
) -> None:
    cfg = replace(small_config, horizon=2)
    vector, base = make_vec(num_envs=2, dtype=dtype, config=cfg, max_observation_space_size=128)
    base["values"][1, ..., 1] += 0.15
    ro = vector.reset(seed=[3, 9], embodiment=base)
    assert ro[1].shape == (2,) and not ro[1].any()
    assert vector.embodiment_space.contains(base)
    scalar = []
    for i in range(2):
        e, b = make_small(dtype, cfg)
        e.reset(seed=[3, 9][i], embodiment={k: v[i] for k, v in base.items()})
        scalar.append(e)
    u = np.array([[0.2, -0.1, 0, 0], [-0.1, 0.3, 0, 0]], dtype=dtype)
    for _ in range(2):
        out = vector.step(u)
        for i, e in enumerate(scalar):
            s = e.step(u[i])
            np.testing.assert_allclose(out[0][i], s[0], rtol=tolerance, atol=tolerance)
            np.testing.assert_allclose(
                out[4]["objective"][i], s[4]["objective"], rtol=tolerance, atol=tolerance
            )
    assert out[3].all()
    nxt = vector.step(u)
    assert (
        np.all(nxt[1] == 0) and not nxt[2].any() and not nxt[3].any() and nxt[4]["autoreset"].all()
    )
    for i, e in enumerate(scalar):
        expected = e.reset()[0]
        np.testing.assert_allclose(nxt[0][i], expected, rtol=tolerance, atol=tolerance)


def test_vector_rejection_and_partial_reset(
    dtype: np.dtype[np.floating], small_config: SpringConfig
) -> None:
    v, b = make_vec(num_envs=2, dtype=dtype, config=small_config, max_observation_space_size=128)
    v.reset(embodiment=b)
    valid = deepcopy(b)
    b["values"][1, ..., 0] = -6
    o, r, i = v.reset(embodiment=b)
    assert r.tolist() == [False, True]
    first = v.envs[0]._state
    with pytest.raises(ResetNeeded):
        v.step(np.zeros((2, 4), dtype=dtype))
    v.reset(options={"reset_mask": np.array([False, True])}, embodiment=valid)
    assert v.envs[0]._state is first
    assert not v.step(np.zeros((2, 4), dtype=dtype))[2].any()


def test_controlled_solver_failure_preserves_state(dtype: np.dtype[np.floating]) -> None:
    c = SpringConfig(grid_shape=(2, 1), settle_steps=0, substeps=1, newton_max_steps=1)
    e, b = make(dtype=dtype, config=c)
    e.reset(embodiment=b)
    old = e._state
    o, r, term, trunc, info = e.step(np.array([1, -1, 0, 0], dtype=dtype))
    assert term and not trunc and not info["valid"] and not info["completed"]
    assert info["status_name"] == "NONLINEAR_SOLVE_FAILED"
    np.testing.assert_array_equal(e._state.x, old.x)
    assert float(e._state.time) == 0 and float(e._state.work_positive) == 0


def test_render_lifecycle(dtype: np.dtype[np.floating], small_config: SpringConfig) -> None:
    e, b = make_small(dtype, small_config)
    e.reset(embodiment=b)
    with pytest.warns(UserWarning):
        assert e.render() is None
    e.close()
    e.close()


@pytest.mark.integration
def test_real_gymnasium_types(dtype: np.dtype[np.floating], small_config: SpringConfig) -> None:
    if not USING_GYMNASIUM:
        pytest.skip("Real Gymnasium unavailable; offline adapter is NOT integration evidence")
    import gymnasium as gym

    e, b = make_small(dtype, small_config)
    assert isinstance(e, gym.Env) and isinstance(e.embodiment_space, gym.spaces.Dict)
    assert isinstance(e.action_space, gym.spaces.Box)
    e.reset(seed=31, embodiment=b)
    assert e.np_random_seed == 31
    gym.spaces.utils.flatten(e.embodiment_space, b)
    v, bv = make_vec(num_envs=2, dtype=dtype, config=small_config, max_observation_space_size=128)
    assert isinstance(v, gym.vector.VectorEnv)
    assert v.metadata["autoreset_mode"] == gym.vector.AutoresetMode.NEXT_STEP
    assert v.embodiment_space.contains(bv)


def test_seed_metadata_vector_attributes_and_closed(
    dtype: np.dtype[np.floating], small_config: SpringConfig
) -> None:
    e, b = make_small(dtype, small_config)
    e.reset(seed=23, embodiment=b)
    assert e._np_random_seed == 23
    rng = e.np_random
    e.reset(seed=None)
    assert e.np_random is rng and e._np_random_seed == 23
    if USING_GYMNASIUM:
        assert e.np_random_seed == 23
    v, bv = make_vec(num_envs=2, dtype=dtype, config=small_config, max_observation_space_size=128)
    _, r, info = v.reset(seed=40, embodiment=bv)
    assert v.np_random_seed == (40, 41) and len(v.np_random) == 2
    assert info["_valid"].all() and info["_objective"].all()
    v.set_attr("test_attribute", [1, 2])
    assert v.get_attr("test_attribute") == (1, 2)
    assert not v.closed
    v.close()
    assert v.closed
