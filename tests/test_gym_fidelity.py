"""End-to-end fidelity: each mujoco_warp port vs the stock Gymnasium mujoco env.

For each env we reset BOTH with reset noise disabled (deterministic qpos0), then
apply a matched control sequence -- the port takes a normalised action in
[-1, 1] which its ctrl kernel maps to the actuator ctrlrange, and we feed the
stock gym env that same *applied* ctrl -- and compare observation/reward.

This measures the real gap between mujoco_warp (the port's engine) and mujoco
(gym's engine), plus any geometry change the port introduced. The per-env
tolerances below record the measured step-1 deviation (with headroom); see the
printed report for the actual numbers. HalfCheetah's tolerance is loose because
its offsettable port re-authors leg geoms (pos/axisangle -> fromto-from-origin),
a small but real geometry change; Humanoid's is loose because it uses Newton
(mujoco_warp lacks the stock PGS solver). The others -- Hopper/Walker2d included,
whose split capsules are massed as the stock ones -- reproduce gym to ~machine
precision per control step.
"""

import numpy as np
import pytest

pytest.importorskip("warp")
pytest.importorskip("mujoco_warp")
gymnasium = pytest.importorskip("gymnasium")

import codesign_gym as cg  # noqa: E402

# env_id -> (init/rest obs tolerance, step-1 obs tolerance). Rename/relocation-only
# ports reproduce gym to ~machine precision; ports with a hub-root torso split, geom
# re-authoring, Newton-vs-PGS, or a re-anchored forearm deviate more but stay within
# the accepted 0.2 step bound. Humanoid's init tolerance is looser because its obs
# uniquely includes per-body COM inertia, and re-anchoring the forearm capsule to the
# body origin shifts it ~1.7cm (can't be exact while keeping the forearm designable).
CASES = {
    "Swimmer-v4": (1e-3, 1e-3),      # front link moved whole -> step-exact
    "Ant-v4": (1e-3, 1e-3),          # rename-only
    "Hopper-v4": (1e-3, 1e-3),       # hub-root torso + foot split, massed as the stock capsules -> step-exact
    "Walker2d-v4": (1e-3, 1e-3),     # hub-root torso split, massed as the stock capsule -> step-exact
    "HalfCheetah-v4": (1e-3, 2e-1),  # leg geom re-authoring + torso split
    "Humanoid-v4": (2e-2, 2e-1),     # Newton vs PGS + re-anchored forearms (obs has cinert)
}


@pytest.mark.parametrize("env_id,tols", list(CASES.items()))
def test_port_matches_gym_step1(env_id, tols):
    init_tol, tol = tols
    my, _ = cg.make_vec(env_id, num_envs=1, device="cpu", use_graph=False, reset_noise_scale=0.0)
    g = gymnasium.make(env_id, reset_noise_scale=0.0).unwrapped

    gobs, _ = g.reset(seed=0)
    mobs, _, _ = my.reset(seed=0)
    mobs = mobs[0].cpu().numpy()
    init_dev = float(np.abs(mobs - gobs).max())

    scale = my.ctrl_scale.cpu().numpy()
    mid = my.ctrl_mid.cpu().numpy()
    act_dim = my.single_action_space.shape[0]

    rng = np.random.default_rng(0)
    a = rng.uniform(-1, 1, act_dim).astype(np.float32)
    applied = a * scale + mid                      # what both engines actually apply
    mo, mr, _, _, _ = my.step(a[None])
    go, gr, _, _, _ = g.step(applied)
    step1_dev = float(np.abs(mo[0].cpu().numpy() - go).max())

    print(f"\n[{env_id}] init_obs_dev={init_dev:.2e} step1_obs_dev={step1_dev:.2e} "
          f"step1_rew_dev={abs(float(mr[0]) - float(gr)):.2e}")

    assert init_dev < init_tol, f"{env_id}: reset states differ ({init_dev} > {init_tol})"
    assert step1_dev < tol, f"{env_id}: step-1 obs deviates {step1_dev} > {tol}"


def test_humanoidstandup_kinematic_parity():
    """HumanoidStandup starts fully supine on the ground, so its contact-force /
    velocity obs channels diverge hugely from the stock PGS solver (mujoco_warp uses
    Newton). Assert parity on the KINEMATIC + inertia + actuator channels (which are
    faithful, ~1e-2), and only that the contact channels stay finite (their
    magnitude is checked in test_humanoidstandup_impact_cost_is_paid). gym's
    HumanoidStandup-v4 doesn't accept reset_noise_scale, so force its state to qpos0."""
    my, _ = cg.make_vec("HumanoidStandup-v4", num_envs=1, device="cpu",
                        use_graph=False, reset_noise_scale=0.0)
    g = gymnasium.make("HumanoidStandup-v4").unwrapped
    g.reset(seed=0)
    g.set_state(g.init_qpos, g.init_qvel)          # deterministic supine init
    gobs = g._get_obs()
    mobs, _, _ = my.reset(seed=0)
    mobs = mobs[0].cpu().numpy()

    nq, nv, nb = int(my.m.nq), int(my.m.nv), int(my.m.nbody)
    # obs layout: qpos[2:], qvel, cinert, cvel, qfrc_actuator, cfrc_ext
    b = np.cumsum([0, nq - 2, nv, nb * 10, nb * 6, nv, nb * 6])
    names = ["qpos", "qvel", "cinert", "cvel", "qfrc", "cfrc"]
    kinematic = {"qpos", "cinert", "qfrc"}          # faithful channels (bounded)

    scale = my.ctrl_scale.cpu().numpy(); mid = my.ctrl_mid.cpu().numpy()
    rng = np.random.default_rng(0)
    a = rng.uniform(-1, 1, my.single_action_space.shape[0]).astype(np.float32)
    mo, _, _, _, _ = my.step(a[None]); mo = mo[0].cpu().numpy()
    go, _, _, _, _ = g.step(a * scale + mid)

    assert np.abs(mobs - gobs).max() < 2e-2, "supine reset states differ"
    for k, nm in enumerate(names):
        dev = float(np.abs(mo[b[k]:b[k + 1]] - go[b[k]:b[k + 1]]).max())
        if nm in kinematic:
            assert dev < 0.2, f"{nm} channel deviates {dev} > 0.2"
        else:
            assert np.isfinite(dev), f"{nm} channel not finite"


def _zero_action_rollout(env_id, steps):
    """Port and stock gym from the same noise-free start under zero control; yields
    (step, port reward, gym reward, port sum|cfrc_ext|, gym sum|cfrc_ext|)."""
    my, _ = cg.make_vec(env_id, num_envs=1, device="cpu", use_graph=False, reset_noise_scale=0.0)
    g = gymnasium.make(env_id).unwrapped
    g.reset(seed=0)
    g.set_state(g.init_qpos, g.init_qvel)
    my.reset(seed=0)
    assert np.abs(my.gpu_data.cfrc_ext.numpy()).sum() == 0.0 == np.abs(g.data.cfrc_ext).sum()   # v4 reset: no forces yet
    a = np.zeros(my.single_action_space.shape[0], np.float32)
    applied = a * my.ctrl_scale.cpu().numpy() + my.ctrl_mid.cpu().numpy()
    for t in range(1, steps + 1):
        _, mr, _, _, _ = my.step(a[None])
        _, gr, _, _, _ = g.step(applied)
        yield t, float(mr[0]), float(gr), float(np.abs(my.gpu_data.cfrc_ext.numpy()).sum()), float(np.abs(g.data.cfrc_ext).sum())


def test_humanoid_contact_forces_match_gym():
    """cfrc_ext is computed once per control step (mj_rnePostConstraint in v4), so the
    84 contact-force obs dims are live: they match gym once the feet land."""
    for t, mr, gr, pc, gc in _zero_action_rollout("Humanoid-v4", 25):
        if t in (15, 25):
            assert gc > 100.0 and abs(pc - gc) < 0.05 * gc, f"step {t}: sum|cfrc_ext| {pc} vs gym {gc}"
            assert abs(mr - gr) < 1e-2, f"step {t}: reward {mr} vs gym {gr}"


def test_humanoidstandup_impact_cost_is_paid():
    """Standup's quad_impact_cost reads cfrc_ext; with it computed, the supine step-1
    reward matches gym's (it was over-paid by the whole impact cost before)."""
    t, mr, gr, pc, gc = next(_zero_action_rollout("HumanoidStandup-v4", 1))
    assert 0.5 * gc < pc < 2.0 * gc, f"sum|cfrc_ext| {pc} vs gym {gc}"
    assert abs(mr - gr) < 0.05, f"step-1 reward {mr} vs gym {gr}"


# env_id -> (qpos index, value) putting the torso outside its healthy range while airborne
UNHEALTHY = {"Hopper-v4": (2, 1.0), "Walker2d-v4": (2, 1.5), "Ant-v4": (2, 1.5), "Humanoid-v4": (2, 2.5)}


@pytest.mark.parametrize("env_id", list(UNHEALTHY))
def test_terminating_step_pays_healthy_reward(env_id):
    """v4 pays healthy_reward as float(is_healthy or terminate_when_unhealthy), so the
    step that terminates an episode still earns it (v5 does not)."""
    my, _ = cg.make_vec(env_id, num_envs=1, device="cpu", use_graph=False, reset_noise_scale=0.0)
    my.reset(seed=0)
    j, v = UNHEALTHY[env_id]
    my.qpos_torch()[0, j] = v
    _, r, terminated, _, _ = my.step(np.zeros((1, my.single_action_space.shape[0]), np.float32))
    assert bool(terminated[0])
    assert abs(float(r[0]) - my.cfg.healthy_reward) < 0.1 * my.cfg.healthy_reward, float(r[0])


@pytest.mark.parametrize("env_id,tol", [("Hopper-v4", 0.3), ("Humanoid-v4", 0.5)])
def test_terminal_reward_matches_gym(env_id, tol):
    """Under a matched control both engines terminate within a step of each other; each
    one's terminating-step reward agrees with the other's (bonus included)."""
    my, _ = cg.make_vec(env_id, num_envs=1, device="cpu", use_graph=False, reset_noise_scale=0.0)
    g = gymnasium.make(env_id, reset_noise_scale=0.0).unwrapped
    g.reset(seed=0)
    my.reset(seed=0)
    a = np.random.default_rng(1).uniform(-1, 1, my.single_action_space.shape[0]).astype(np.float32)
    applied = a * my.ctrl_scale.cpu().numpy() + my.ctrl_mid.cpu().numpy()
    mine = theirs = None
    for _ in range(300):
        if mine is None:
            _, mr, mt, _, _ = my.step(a[None])
            mine = float(mr[0]) if bool(mt[0]) else None
        if theirs is None:
            _, gr, gt, _, _ = g.step(applied)
            theirs = float(gr) if gt else None
        if mine is not None and theirs is not None:
            break
    assert mine is not None and theirs is not None, "no termination within 300 steps"
    assert abs(mine - theirs) < tol, f"terminal reward {mine} vs gym {theirs}"


@pytest.mark.parametrize("env_id", [*CASES, "HumanoidStandup-v4"])
def test_base_design_mass_matches_stock(env_id):
    """At design 0 every port weighs what stock gym does. Hopper/Walker2d split a stock
    capsule at a joint; its halves are massed without the shared near cap."""
    my, _ = cg.make_vec(env_id, num_envs=1, device="cpu", use_graph=False, reset_noise_scale=0.0)
    my.reset(seed=0)
    stock = gymnasium.make(env_id).unwrapped.model.body_mass.sum()
    assert my.gpu_model.body_mass.numpy()[0].sum() == pytest.approx(stock, rel=1e-5)


def test_split_capsule_halves_weigh_one_capsule_when_stretched():
    """For any design, a split pair weighs one capsule spanning both halves (no double cap)."""
    my, _ = cg.make_vec("Hopper-v4", num_envs=1, device="cpu", use_graph=False)
    lo, hi = my.offset_bounds
    my.reset(embodiment=hi[None])
    seg, mass = my._segs, my.gpu_model.body_mass.numpy()[0]
    halves = [i for i, c in enumerate(seg["is_cyl"]) if c == 2][:2]      # torso upper + lower
    size = my.gpu_model.geom_size.numpy()[0]
    r = float(seg["radius"][halves[0]])
    length = sum(2.0 * float(size[int(seg["geom"][i])][1]) for i in halves)
    rho = float(seg["density"][halves[0]])
    one_capsule = rho * (np.pi * r * r * length + 4.0 / 3.0 * np.pi * r ** 3)
    assert sum(mass[int(seg["body"][i])] for i in halves) == pytest.approx(one_capsule, rel=1e-5)
