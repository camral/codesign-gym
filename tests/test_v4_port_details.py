"""v4 port details: gym's per-step info keys, the respawn step's real forward
velocity, final_steps in control steps, close() and renders that show the design."""

from __future__ import annotations

import gymnasium
import numpy as np
import pytest

import codesign_gym as cg

PORTS = ["HalfCheetah-v4", "Swimmer-v4", "Hopper-v4", "Walker2d-v4", "Ant-v4", "Humanoid-v4", "HumanoidStandup-v4"]


def _port(env_id, n=1, **kw):
    return cg.make_vec(env_id, num_envs=n, device="cpu", use_graph=False, **kw)[0]


@pytest.mark.parametrize("env_id", PORTS)
def test_step_info_has_gyms_v4_keys(env_id):
    my = _port(env_id)
    my.reset(seed=0)
    _, _, _, _, info = my.step(np.zeros((1, my.single_action_space.shape[0]), np.float32))
    g = gymnasium.make(env_id).unwrapped
    g.reset(seed=0)
    _, _, _, _, ginfo = g.step(np.zeros(g.action_space.shape, np.float32))
    assert set(ginfo) <= set(info), set(ginfo) - set(info)


def test_hopper_step_info_matches_gym():
    my = _port("Hopper-v4", reset_noise_scale=0.0)
    g = gymnasium.make("Hopper-v4", reset_noise_scale=0.0).unwrapped
    my.reset(seed=0)
    g.reset(seed=0)
    a = np.random.default_rng(0).uniform(-1, 1, my.single_action_space.shape[0]).astype(np.float32)
    _, _, _, _, info = my.step(a[None])
    _, _, _, _, ginfo = g.step(a * my.ctrl_scale.cpu().numpy() + my.ctrl_mid.cpu().numpy())
    for k in ("x_position", "x_velocity"):
        assert float(info[k][0]) == pytest.approx(ginfo[k], abs=1e-4), k


def test_first_step_after_autoreset_pays_the_real_forward_velocity_and_final_steps_count_control_steps():
    """NEXT_STEP: the step after an episode ends only respawns the world (reward 0). The first real step after it
    measures its displacement from the respawn (qpos0 here, x = 0), not from where the last episode ended."""
    my = _port("Hopper-v4", reset_noise_scale=0.0)
    my.reset(seed=0)
    a = np.zeros((1, my.single_action_space.shape[0]), np.float32)
    for _ in range(3):
        my.step(a)
    my.qpos_torch()[0, 0] += 5.0                    # far from the respawn point
    my.qpos_torch()[0, 2] = 1.0                     # unhealthy angle: this step terminates
    _, _, terminated, _, info = my.step(a)
    assert bool(terminated[0]) and int(info["final_steps"][0]) == 4
    _, reward, terminated, _, _ = my.step(a)        # the respawn step
    assert float(reward[0]) == 0.0 and not bool(terminated[0])
    _, reward, _, _, info = my.step(a)              # the first real step of the new episode
    dt = my.phys_substeps * float(my.m.opt.timestep)
    x, v = float(info["x_position"][0]), float(info["x_velocity"][0])
    assert v == pytest.approx(x / dt, abs=1e-4) and abs(v) < 1.0 and v != 0.0


@pytest.mark.parametrize("env_id", ["Hopper-v4", "Humanoid-v4"])
def test_close_frees_the_renderers(env_id):
    my = _port(env_id)
    assert type(my).close is not gymnasium.Env.close
    my.close()


def test_render_model_mirrors_each_worlds_design():
    my = _port("Hopper-v4", n=2)
    lo, hi = my.offset_bounds
    my.reset(embodiment=np.stack([np.zeros_like(hi), hi]))
    render_m = my._make_render_model()
    my._mirror_render_model(1, render_m)
    np.testing.assert_allclose(render_m.geom_size, my.gpu_model.geom_size.numpy()[1], atol=1e-6)
    np.testing.assert_allclose(render_m.body_pos, my.gpu_model.body_pos.numpy()[1], atol=1e-6)
    # A tracking camera aims at the subtree COM, so the subtree masses follow the mirrored masses.
    np.testing.assert_allclose(render_m.body_subtreemass[0], render_m.body_mass.sum(), rtol=1e-6)


@pytest.mark.parametrize("env_id", ["HalfCheetah-v4", "Hopper-v4", "Walker2d-v4"])
def test_extreme_designs_spawn_clear_of_the_floor_under_reset_noise(env_id):
    """The spawn lift measures at qpos0 and tops up at the drawn pose, so reset noise
    that tilts a long design never starts it inside the floor."""
    my = _port(env_id, n=4)
    lo, hi = my.offset_bounds
    floor = my.m.geom("floor").id
    for seed in range(3):
        my.reset(seed=seed, embodiment=np.stack([hi, lo, hi, lo]))
        d = my.gpu_data
        n = int(d.nacon.numpy()[0])
        geoms, dist = d.contact.geom.numpy()[:n], d.contact.dist.numpy()[:n]
        on_floor = (geoms == floor).any(axis=1)
        assert (dist[on_floor] > -1e-3).all(), float(dist[on_floor].min())


@pytest.mark.parametrize("env_id", PORTS)
def test_noise_free_base_design_spawns_at_stock_qpos0(env_id):
    my = _port(env_id, n=2, reset_noise_scale=0.0)
    my.reset(seed=0)
    np.testing.assert_array_equal(my.qpos_torch().numpy(), np.broadcast_to(my.m.qpos0.astype(np.float32), (2, my.m.nq)))
