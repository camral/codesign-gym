"""Walker2d-v4 with physics="native": C MuJoCo (mujoco.rollout) steps the physics on the CPU instead of MuJoCo Warp.

Resets still run the Warp kernels, so reset observations and deformed designs must match the Warp path exactly. Steps
use a different engine (float64 C MuJoCo vs float32 Warp), so trajectories agree to the step-1 bar the Warp port meets
against stock Gymnasium (1e-3, tests/test_gym_fidelity.py) and then drift apart chaotically; episodes must still end on
the same step and earn the same return. Measured on Apple M5 (16 random designs, sinusoid gait): step-1 obs 4e-4,
step-5 6e-3, step-10 8e-2; 15/16 first terminations on the same step, one a step later; returns within 0.3.
"""

import numpy as np
import pytest

pytest.importorskip("warp")
pytest.importorskip("mujoco_warp")
gymnasium = pytest.importorskip("gymnasium")

import codesign_gym as cg  # noqa: E402

ENV_ID = "Walker2d-v4"


def _pair(n, **cfg):
    warp, base = cg.make_vec(ENV_ID, n, device="cpu", use_graph=False, **cfg)
    native, _ = cg.make_vec(ENV_ID, n, device="cpu", physics="native", **cfg)
    return warp, native, base


def _designs(base, n, seed=1):
    d = np.random.default_rng(seed).uniform(-1, 1, (n, base.size)).astype(np.float32)
    d[0] = base
    return d


def _gait(t, n):
    """An open-loop sinusoid gait, the same for every world."""
    return np.tile(np.sin(0.3 * t + np.linspace(0, 3, 6)).astype(np.float32), (n, 1))


def test_native_builds_on_cpu_and_steps():
    env, _ = cg.make_vec(ENV_ID, 3, device="cpu", physics="native")
    assert env.physics == "native"
    obs, rejected, _ = env.reset(seed=0)
    assert obs.shape == (3, 17) and not bool(rejected.any())
    for t in range(5):
        obs, reward, terminated, truncated, _ = env.step(_gait(t, 3))
    assert np.isfinite(obs.numpy()).all() and np.isfinite(reward.numpy()).all()
    env.close()


def test_native_rejects_other_devices_and_unknown_backends():
    from envs.walker2d.walker2d_env import Walker2dParallelEnv
    with pytest.raises(ValueError, match="device='cpu'"):
        Walker2dParallelEnv(2, device="cuda", physics="native")
    with pytest.raises(ValueError, match="physics must be"):
        cg.make_vec(ENV_ID, 2, device="cpu", physics="jax")


def test_reset_and_designs_match_warp_exactly():
    n = 16
    warp, native, base = _pair(n)
    d = _designs(base, n)
    ow, _, iw = warp.reset(seed=3, embodiment=d)
    on, _, inn = native.reset(seed=3, embodiment=d)
    np.testing.assert_array_equal(ow.numpy(), on.numpy())
    np.testing.assert_array_equal(iw["embodiment"].numpy(), inn["embodiment"].numpy())
    # Every per-world CPU model carries exactly its world's deformed design and set_const values.
    for name in native._DEFORM_FIELDS + native._CONST_FIELDS:
        rows = getattr(native.gpu_model, name).numpy()
        if rows.size == 0 or rows.shape[0] < n:
            continue
        for w in range(n):
            np.testing.assert_array_equal(rows[w].reshape(-1), getattr(native.variants[w], name).reshape(-1),
                                          err_msg=f"{name}, world {w}")
    warp.close()
    native.close()


@pytest.mark.parametrize("noise", [0.0, 5e-3])
def test_steps_match_warp(noise):
    n = 16
    warp, native, base = _pair(n, reset_noise_scale=noise)
    d = _designs(base, n)
    warp.reset(seed=3, embodiment=d)
    native.reset(seed=3, embodiment=d)
    bounds = {1: 1e-3, 5: 2e-2, 10: 2.5e-1}           # measured 4e-4, 6e-3, 8e-2
    for t in range(1, 11):
        ow, rw, tw, _, _ = warp.step(_gait(t - 1, n))
        on, rn, tn, _, _ = native.step(_gait(t - 1, n))
        if t in bounds:
            dev = float((ow - on).abs().max())
            assert dev < bounds[t], f"step {t}: obs deviates {dev:.2e} > {bounds[t]}"
            assert float((rw - rn).abs().max()) < bounds[t]
    warp.close()
    native.close()


def test_episodes_match_warp():
    """Same designs, seed and gait: worlds fall on the same step (±1) and earn the same first-episode return."""
    n = 64
    warp, native, base = _pair(n)
    d = _designs(base, n, seed=7)
    warp.reset(seed=11, embodiment=d)
    native.reset(seed=11, embodiment=d)
    ended = {k: np.full(n, -1) for k in ("w", "n")}
    ret = {k: np.full(n, np.nan) for k in ("w", "n")}
    for t in range(300):
        for key, env in (("w", warp), ("n", native)):
            _, _, terminated, truncated, info = env.step(_gait(t, n))
            first = (ended[key] < 0) & (terminated | truncated).numpy()
            ended[key][first] = t
            if first.any():
                ret[key][first] = info["final_return"].numpy()[first]
        if (ended["w"] >= 0).all() and (ended["n"] >= 0).all():
            break
    assert (ended["w"] >= 0).all() and (ended["n"] >= 0).all(), "not every world ended within 300 steps"
    same = np.abs(ended["w"] - ended["n"]) <= 1
    assert same.mean() >= 0.9, f"termination steps agree for only {same.mean():.0%}: {ended['w']} vs {ended['n']}"
    assert np.abs(ret["w"] - ret["n"])[same].max() < 1.5
    assert abs(ret["w"].mean() - ret["n"].mean()) < 0.5
    warp.close()
    native.close()


def test_native_matches_gym_step1():
    """Native C MuJoCo is Gymnasium's own engine, so the port meets the same 1e-3 step-1 bar as the Warp path."""
    my, _ = cg.make_vec(ENV_ID, 1, device="cpu", physics="native", reset_noise_scale=0.0)
    g = gymnasium.make(ENV_ID, reset_noise_scale=0.0).unwrapped
    gobs, _ = g.reset(seed=0)
    mobs, _, _ = my.reset(seed=0)
    assert float(np.abs(mobs[0].numpy() - gobs).max()) < 1e-3
    a = np.random.default_rng(0).uniform(-1, 1, 6).astype(np.float32)
    mo, mr, _, _, _ = my.step(a[None])
    go, gr, _, _, _ = g.step(a * my.ctrl_scale.numpy() + my.ctrl_mid.numpy())
    assert float(np.abs(mo[0].numpy() - go).max()) < 1e-3
    assert abs(float(mr[0]) - float(gr)) < 1e-3
    my.close()


def test_direct_state_writes_reach_native_physics():
    """Writing through qpos_torch() between steps (as tests and scripts do) moves the native world too."""
    env, _ = cg.make_vec(ENV_ID, 2, device="cpu", physics="native", reset_noise_scale=0.0)
    env.reset(seed=0)
    env.qpos_torch()[0, 2] = 1.5                       # torso angle far outside the healthy range
    _, r, terminated, _, _ = env.step(np.zeros((2, 6), np.float32))
    assert bool(terminated[0]) and not bool(terminated[1])
    assert abs(float(r[0]) - env.cfg.healthy_reward) < 0.1 * env.cfg.healthy_reward
    env.close()


def test_render_uses_each_worlds_design():
    pytest.importorskip("mujoco")
    env, base = cg.make_vec(ENV_ID, 2, device="cpu", physics="native", render_mode="rgb_array")
    long_legs = np.full_like(base, 0.5)
    try:
        env.reset(seed=0, embodiment=np.stack([base, long_legs]))
        frames = env.render()
    except Exception as e:  # no OpenGL context on this machine
        if "GL" in type(e).__name__ or "GL" in str(e):
            pytest.skip(f"no OpenGL context: {e}")
        raise
    assert frames[0].shape == (480, 480, 3) and not np.array_equal(frames[0], frames[1])
    env.close()
