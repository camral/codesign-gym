"""Cross-driver API, pure-rollout, reset, and observation contracts."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from envs.soft_walker.engine.spaces import Dict


from copy import deepcopy
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from envs.soft_walker.engine import make, make_vec
from envs.soft_walker.engine.differentiation import bind_rollout
from envs.soft_walker.engine.drivers.task import BodyConfig
from envs.soft_walker.engine.physics.spring_config import SpringConfig
from envs.soft_walker.engine.physics.terrain import TerrainConfig
from envs.soft_walker.engine.spaces import ResetNeeded

IDS = ["SoftHexFEM3D-v0", "SoftMPM2D-v0", "SoftMPM3D-v0", "SoftBeam3D-v0"]


def config(eid: str, **kw: object) -> BodyConfig:
    args = dict(substeps=1, control_dt=0.00025, settle_steps=1, horizon=2, grid_margin=1)
    if eid == "SoftMPM2D-v0":
        args.update(grid_shape=(2, 1), size=(0.1, 0.05))
    args.update(kw)
    return BodyConfig(**args)


@pytest.mark.parametrize("eid", IDS)
def test_m3_driver_api_metrics_and_pure_rollout(eid: str, dtype: np.dtype[np.floating]) -> None:
    e, b = make(
        eid, dtype=dtype, config=config(eid), design_mode="relaxed", render_mode="rgb_array"
    )
    obs, rejected, info = e.reset(seed=17, embodiment=b)
    assert not rejected[0] and e.observation_space.contains(obs)
    assert e.embodiment_space.contains(b)
    assert info["mass_kg"] > 0 and info["volume_m3"] > 0
    assert e.observation_size <= e.max_observation_space_size
    actions = np.asarray([[0.24, -0.17, 0.1, 0.0], [0.22, -0.15, 0.0, 0.1]], dtype)
    function = jax.jit(bind_rollout(e, record_trajectory=True))
    out = jax.block_until_ready(function(jax.tree.map(jnp.asarray, b), jnp.asarray(actions)))
    assert bool(out["metrics"]["valid"])
    rewards = []
    for k, a in enumerate(actions):
        o, r, term, trunc, inf = e.step(a)
        rewards.append(r)
        assert not term and trunc == (k == 1)
        np.testing.assert_allclose(
            o,
            out["observations"][k],
            rtol=4e-4 if dtype == np.float32 else 1e-9,
            atol=3e-6 if dtype == np.float32 else 1e-10,
        )
        np.testing.assert_allclose(
            np.sum(inf["accepted_dt_trace_s"]), e.config.control_dt, rtol=2e-6
        )
    np.testing.assert_allclose(
        sum(rewards),
        -inf["objective"][0] - e.config.energy_weight * inf["objective"][1],
        atol=2e-9 if dtype == np.float32 else 1e-14,
    )
    np.testing.assert_allclose(inf["objective"], out["metrics"]["objective"], rtol=4e-4, atol=1e-9)
    assert inf["completed"] and bool(out["completed"])
    frame = e.render()
    assert frame.shape == (420, 720, 3) and frame.dtype == np.uint8
    with pytest.raises(ResetNeeded):
        e.step(np.zeros(4))
    e.close()


@pytest.mark.parametrize("eid", IDS)
@pytest.mark.parametrize("decoder", ["direct", "cppn", "primitives", "bspline"])
def test_m3_decoder_instantiation_matrix(
    eid: str, decoder: str, dtype: np.dtype[np.floating]
) -> None:
    e, b = make(eid, dtype=dtype, config=config(eid, settle_steps=0), decoder=decoder)
    assert e.validate_embodiment(b)
    _, m, errors = e._prepare(b)
    assert not errors
    assert float(jnp.sum(m.mass)) > 0
    e.close()


@pytest.mark.parametrize("eid", IDS)
def test_m3_rejection_and_capacity(eid: str, dtype: np.dtype[np.floating]) -> None:
    cfg = config(eid)
    with pytest.raises(ValueError, match="OBSERVATION_CAPACITY"):
        make(eid, dtype=dtype, config=cfg, max_observation_space_size=2)
    with pytest.raises(ValueError, match="ACTION_CAPACITY"):
        make(eid, dtype=dtype, config=cfg, max_action_space_size=1)
    e, b = make(eid, dtype=dtype, config=cfg)
    assert not e.reset(seed=3, embodiment=b)[1][0]
    invalid = deepcopy(b)
    invalid["poisson"] = np.asarray(0.499, dtype)
    assert e.reset(embodiment=invalid)[1][0]
    with pytest.raises(ResetNeeded):
        e.step(np.zeros(4, dtype))
    assert not e.reset(seed=3, embodiment=b)[1][0]
    before = e._state
    with pytest.raises(ValueError):
        e.step(np.asarray([0.0, np.nan, 0.0, 0.0]))
    assert e._state is before
    with pytest.warns(UserWarning):
        e.step(np.zeros(4, dtype), embodiment=invalid)
    e.close()


@pytest.mark.parametrize("eid", IDS)
def test_m3_vector_scalar_autoreset(eid: str, dtype: np.dtype[np.floating]) -> None:
    cfg = config(eid, settle_steps=0, horizon=1)
    vec, base = make_vec(eid, num_envs=2, dtype=dtype, config=cfg)
    o, r, i = vec.reset(seed=20, embodiment=base)
    assert not np.any(r)
    scalar, b = make(eid, dtype=dtype, config=cfg)
    scalar.reset(seed=20, embodiment=b)
    action = np.asarray([0.2, -0.1, 0.0, 0.0], dtype)
    a = scalar.step(action)
    v = vec.step(np.stack([action, action]))
    np.testing.assert_allclose(
        v[0][0],
        a[0],
        rtol=3e-4 if dtype == np.float32 else 1e-9,
        atol=2e-6 if dtype == np.float32 else 1e-10,
    )
    assert np.all(v[3])
    reset = vec.step(np.zeros((2, 4), dtype))
    assert not np.any(reset[2]) and not np.any(reset[3]) and np.all(reset[1] == 0)
    assert np.all(reset[4]["autoreset"])
    invalid = {k: np.array(v, copy=True) for k, v in base.items()}
    invalid["values"][0, ..., -1] = np.nan
    _, reject, _ = vec.reset(embodiment=invalid, options={"reset_mask": np.array([True, False])})
    assert reject[0] and not reject[1]
    with pytest.raises(ResetNeeded):
        vec.step(np.zeros((2, 4), dtype))
    scalar.close()
    vec.close()


def test_m3_observation_hidden_state_and_masks(dtype: np.dtype[np.floating]) -> None:
    e, b = make("SoftMPM2D-v0", dtype=dtype, config=config("SoftMPM2D-v0", settle_steps=0))
    obs, _, i = e.reset(seed=2, embodiment=b)
    s = e._state
    altered = s._replace(body=s.body._replace(C=s.body.C.at[0, 0, 1].set(0.2)))
    other = np.asarray(e._observe(e._model, altered)[0])
    assert not np.array_equal(obs, other)
    assert not np.any(obs[e.observation_size :]) and not np.any(
        i["observation_mask"][e.observation_size :]
    )
    assert i["action_mask"].shape == (4,)
    e.close()


def test_m3_padding_invariance(dtype: np.dtype[np.floating]) -> None:
    cfg = config("SoftHexFEM3D-v0", settle_steps=0)
    e, b = make("SoftHexFEM3D-v0", dtype=dtype, config=cfg, max_observation_space_size=256)
    big, _ = make(
        "SoftHexFEM3D-v0",
        dtype=dtype,
        config=cfg,
        max_observation_space_size=800,
        max_action_space_size=9,
    )
    e.reset(seed=1, embodiment=b)
    big.reset(seed=1, embodiment=b)
    a = e.step(np.asarray([0.2, -0.1, 0.0, 0.0], dtype))
    z = big.step(np.asarray([0.2, -0.1, 0.0, 0.0, 0.7, 0.7, 0.7, 0.7, 0.7], dtype))
    np.testing.assert_allclose(a[0], z[0][:256], rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(a[4]["objective"], z[4]["objective"], atol=1e-12)
    e.close()
    big.close()


@pytest.mark.parametrize("kind", ["slopes", "blocks"])
def test_m3_seeded_spring_terrain_replay_and_gradient_path(
    kind: str, dtype: np.dtype[np.floating]
) -> None:
    cfg = SpringConfig(
        grid_shape=(2, 1),
        substeps=1,
        control_dt=0.00025,
        settle_steps=0,
        horizon=2,
        terrain=TerrainConfig(kind=kind),
    )
    e, b = make(dtype=dtype, config=cfg, max_observation_space_size=256)
    first = e.reset(seed=8, embodiment=b)
    saved = deepcopy(first[2]["terrain"])
    result = e.step(np.asarray([0.2, -0.1, 0.0, 0.0], dtype))
    again = e.reset(seed=8, embodiment=b)
    assert saved == again[2]["terrain"]
    np.testing.assert_array_equal(first[0], again[0])
    pure = jax.jit(bind_rollout(e))(
        jax.tree.map(jnp.asarray, b), jnp.asarray([[0.2, -0.1, 0.0, 0.0]], dtype)
    )
    np.testing.assert_allclose(pure["observations"][0], result[0], atol=2e-6, rtol=3e-4)
    assert not result[2]
    e.close()


def test_m3_failure_transaction_and_invalid_objective(dtype: np.dtype[np.floating]) -> None:
    eid = "SoftHexFEM3D-v0"
    cfg = config(eid, settle_steps=0, substeps=2)
    cfg = replace(
        cfg,
        dynamics=replace(
            cfg.dynamics,
            newton=replace(cfg.dynamics.newton, max_steps=1, atol64=1e-20, atol32=1e-20),
        ),
    )
    e, b = make(eid, dtype=dtype, config=cfg)
    e.reset(seed=2, embodiment=b)
    old = e._state
    _, _, term, _, info = e.step(np.asarray([0.9, -0.8, 0.0, 0.0], dtype))
    assert term and not info["valid"] and np.all(np.isnan(info["objective"]))
    np.testing.assert_array_equal(e._state.body.u, old.body.u)
    np.testing.assert_array_equal(e._state.displacement, old.displacement)
    assert np.sum(info["accepted_dt_trace_s"]) == 0
    e.close()


@pytest.mark.parametrize("value", ["bad", complex(0.35, 0.1), np.nan, [0.35], None])
def test_m3_malformed_poisson_is_rejection_not_exception(
    value: None | complex | float | list[float] | str, dtype: np.dtype[np.floating]
) -> None:
    e, b = make("SoftHexFEM3D-v0", dtype=dtype, config=config("SoftHexFEM3D-v0", settle_steps=0))
    b["poisson"] = value
    assert not e.validate_embodiment(b)
    assert e.reset(embodiment=b)[1][0]
    e.close()


@pytest.mark.parametrize("eid", ["SoftHexFEM3D-v0", "SoftSpring2D-v0"])
def test_m3_closed_loop_policy_same_actions_and_gradient(
    eid: str, dtype: np.dtype[np.floating]
) -> None:
    from envs.soft_walker.engine.differentiation import bind_policy

    cfg = (
        SpringConfig(
            grid_shape=(2, 1),
            size=(0.1, 0.05),
            control_dt=0.00025,
            substeps=1,
            settle_steps=0,
            horizon=2,
        )
        if eid == "SoftSpring2D-v0"
        else config(eid, settle_steps=0)
    )
    e, b = make(eid, dtype=dtype, config=cfg, design_mode="relaxed")
    e.reset(seed=4, embodiment=b)
    emb = jax.tree.map(jnp.asarray, b)

    # External controller with memory, no required optimizer or NN library.
    def policy(p: jax.Array, obs: jax.Array, carry: jax.Array) -> tuple[jax.Array, jax.Array]:
        action = jnp.tanh(p + 0.03 * obs[:2] + carry)
        return action, carry + 0.01

    run = bind_policy(e, policy, steps=2)
    p = jnp.asarray([0.23, -0.16], dtype)
    carry = jnp.zeros(2, dtype)
    result = jax.block_until_ready(jax.jit(run)(emb, p, carry))
    replay = jax.block_until_ready(jax.jit(bind_rollout(e))(emb, result["actions"]))
    np.testing.assert_allclose(
        result["metrics"]["objective"], replay["metrics"]["objective"], rtol=2e-5, atol=1e-10
    )
    loss = lambda p: run(emb, p, carry)["metrics"]["scalar_objective"]
    gradient = jax.jit(jax.grad(loss))(p)
    assert np.isfinite(gradient).all() and np.linalg.norm(gradient) > 0
    # Closed-loop derivative independently checked against a directional FD.
    d = jnp.asarray([0.7, -0.3], dtype)
    h = 0.02 if dtype == np.float32 else 1e-3
    f = jax.jit(loss)
    fd = (float(f(p + h * d)) - float(f(p - h * d))) / (2 * h)
    np.testing.assert_allclose(
        fd,
        jnp.vdot(gradient, d),
        rtol=5e-3 if dtype == np.float32 else 4e-5,
        atol=3e-10 if dtype == np.float32 else 1e-13,
    )
    e.close()


def test_m3_reserved_decoder_key_is_not_silently_overwritten(
    dtype: np.dtype[np.floating],
) -> None:
    from envs.soft_walker.engine.embodiments.decoders import FieldSpec, GridDecoder
    from envs.soft_walker.engine.spaces import Box, Dict

    class ConflictingDecoder(GridDecoder):
        def space(self) -> Dict:
            return Dict(dict(super().space().spaces, poisson=Box(-1.0, 1.0, (), dtype)))

    decoder = ConflictingDecoder(spec=FieldSpec(dim=3), shape=(2, 2, 2), dtype=dtype)
    with pytest.raises(ValueError, match="RESERVED_EMBODIMENT_KEY"):
        make("SoftHexFEM3D-v0", dtype=dtype, decoder=decoder, config=config("SoftHexFEM3D-v0"))


def test_m3_mpm3_uniform_freeflight_reference(dtype: np.dtype[np.floating]) -> None:
    from envs.soft_walker.engine.physics.common import DynamicsConfig
    from envs.soft_walker.engine.physics.contact import PlaneContact

    cfg = config(
        "SoftMPM3D-v0",
        settle_steps=0,
        dynamics=DynamicsConfig(gravity=0.0, contact=PlaneContact(enabled=False), max_halvings=0),
    )
    e, b = make("SoftMPM3D-v0", dtype=dtype, config=cfg)
    e.reset(seed=2, embodiment=b)
    state = e._state
    velocity = jnp.broadcast_to(jnp.asarray([0.07, -0.03, 0.02], dtype), state.v.shape)
    state = state._replace(body=state.body._replace(v=velocity))
    new, diag = e._advance(e._model, state, jnp.zeros(2, dtype))
    assert int(new.status) == 0
    np.testing.assert_allclose(new.body.v, velocity, rtol=5e-6, atol=3e-8)
    np.testing.assert_allclose(
        new.body.x, state.body.x + cfg.control_dt * velocity, rtol=1e-6, atol=1e-8
    )
    np.testing.assert_allclose(new.body.F, state.body.F, rtol=3e-6, atol=1e-7)
    np.testing.assert_allclose(
        new.displacement, cfg.control_dt * velocity[0], rtol=4e-6, atol=1e-10
    )
    e.close()
