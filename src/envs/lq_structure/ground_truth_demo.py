"""Exact per-design ground truth for the LQ structure co-design environment.

This script is the sanity-checker referenced in the README: for a *fixed* embodiment the
optimal controller is computable exactly, so a co-design algorithm can be checked against
a known best-achievable cost. The ground-truth solver lives here (not on the env) because
it does not fit the gym API shape.

For each setup it:
  1. builds the exact discrete-time plant for a design (reusing the package dynamics so
     the ground truth uses the same discretization the env steps with),
  2. solves the optimal controller (LQR / LQG+Kalman / saturated QP-MPC),
  3. rolls that controller out in the actual environment and reports the achieved cost,
  4. compares against an analytic expected cost (LQR) and a do-nothing baseline.

Run::

    python -m envs.lq_structure.ground_truth_demo                 # all three setups
    python -m envs.lq_structure.ground_truth_demo --setup lqg --n 6 --episodes 200

By default it also renders one optimal-policy rollout per setup to a GIF
(lq_<setup>_optimal.gif). Control it with --render {off,gif,human}, --render-out,
--fps, and --render-stride, e.g.::

    python -m envs.lq_structure.ground_truth_demo --setup lqr --n 6 --episodes 2000
    python -m envs.lq_structure.ground_truth_demo --setup lqr --render human
    python -m envs.lq_structure.ground_truth_demo --setup all --render off
"""

from __future__ import annotations

import argparse

import numpy as np
from scipy.linalg import solve_discrete_are

import codesign_gym as cg
from envs.lq_structure.dynamics import Plant


# --- exact solvers -----------------------------------------------------------------

def dlqr(Ad, Bd, Q, R):
    """Discrete-time infinite-horizon LQR. Returns (K, P) with optimal u = -K x."""
    P = solve_discrete_are(Ad, Bd, Q, R)
    K = np.linalg.solve(R + Bd.T @ P @ Bd, Bd.T @ P @ Ad)
    return K, P


def analytic_lqr_return(env, design, K):
    """Exact expected episodic return of the closed loop u = -K x.

    Uses the state-covariance recursion under the env's own initial-state and
    process-noise distributions, so it is directly comparable to Monte-Carlo rollouts.
    """
    cfg = env.cfg
    plant = Plant(
        design["masses"], design["stiffness"], design["damping"],
        design["actuators"], design["sensors"], cfg.dt,
    )
    Ad, Bd, Gd = plant.Ad, plant.Bd, plant.Gd
    n = cfg.n
    Q = cfg.Q()
    R = cfg.R(plant.n_actuators)

    Acl = Ad - Bd @ K
    S = cfg.dt * (Q + K.T @ R @ K)
    Sigma_w = plant.n_actuators and (Gd @ (cfg.process_noise_std**2 * np.eye(n)) @ Gd.T)
    if not isinstance(Sigma_w, np.ndarray):
        Sigma_w = Gd @ (cfg.process_noise_std**2 * np.eye(n)) @ Gd.T

    cov0 = np.diag(
        np.concatenate([np.full(n, cfg.init_pos_std**2), np.full(n, cfg.init_vel_std**2)])
    )
    cov = cov0.copy()
    total = 0.0
    for _ in range(cfg.horizon):
        total += np.trace(S @ cov)
        cov = Acl @ cov @ Acl.T + Sigma_w
    return -total


class KalmanController:
    """LQG controller: LQR gain on a Kalman-filtered state estimate (separation)."""

    def __init__(self, env, design):
        cfg = env.cfg
        self.cfg = cfg
        self.plant = Plant(
            design["masses"], design["stiffness"], design["damping"],
            design["actuators"], design["sensors"], cfg.dt,
        )
        self.K, _ = dlqr(self.plant.Ad, self.plant.Bd, cfg.Q(), cfg.R(self.plant.n_actuators))
        n = cfg.n
        self.Qk = self.plant.Gd @ (cfg.process_noise_std**2 * np.eye(n)) @ self.plant.Gd.T
        n_meas = self.plant.Cs.shape[0]
        self.Rk = cfg.measurement_noise_std**2 * np.eye(max(n_meas, 1))
        self.sen_idx = np.flatnonzero(np.asarray(design["sensors"]).astype(bool))
        self.reset()

    def reset(self):
        self.xhat = np.zeros(self.cfg.state_dim)
        self.P = np.eye(self.cfg.state_dim)
        self._last_u = np.zeros(self.plant.n_actuators)

    def _measurement_from_obs(self, obs):
        n = self.cfg.n
        meas = obs[: 2 * n]
        y = np.zeros(2 * self.sen_idx.size)
        for k, i in enumerate(self.sen_idx):
            y[2 * k] = meas[i]
            y[2 * k + 1] = meas[n + i]
        return y

    def act(self, obs):
        Ad, Bd, Cs = self.plant.Ad, self.plant.Bd, self.plant.Cs
        # Predict with the previously applied control.
        xpred = Ad @ self.xhat + (Bd @ self._last_u if Bd.shape[1] else 0.0)
        Ppred = Ad @ self.P @ Ad.T + self.Qk
        # Update with the current measurement.
        y = self._measurement_from_obs(obs)
        S = Cs @ Ppred @ Cs.T + self.Rk
        Kf = Ppred @ Cs.T @ np.linalg.inv(S)
        self.xhat = xpred + Kf @ (y - Cs @ xpred)
        self.P = (np.eye(self.cfg.state_dim) - Kf @ Cs) @ Ppred
        u = -self.K @ self.xhat
        self._last_u = u
        return u


class QPController:
    """Saturated receding-horizon (MPC) controller solved as a QP with cvxpy."""

    def __init__(self, env, design, mpc_horizon=15):
        import cvxpy as cp
        self.cp = cp
        cfg = env.cfg
        self.cfg = cfg
        self.plant = Plant(
            design["masses"], design["stiffness"], design["damping"],
            design["actuators"], design["sensors"], cfg.dt,
        )
        self.Ad, self.Bd = self.plant.Ad, self.plant.Bd
        self.Q = cfg.Q()
        self.R = cfg.R(self.plant.n_actuators)
        _, self.Pterm = dlqr(self.Ad, self.Bd, self.Q, self.R)
        self.H = mpc_horizon
        self.u_max = cfg.u_max
        self.n_act = self.plant.n_actuators

    def reset(self):
        pass

    def act(self, obs):
        cp = self.cp
        nx = self.cfg.state_dim
        x0 = obs[:nx]
        x = cp.Variable((nx, self.H + 1))
        u = cp.Variable((self.n_act, self.H))
        cost = 0
        cons = [x[:, 0] == x0]
        for t in range(self.H):
            cost += cp.quad_form(x[:, t], self.Q) + cp.quad_form(u[:, t], self.R)
            cons += [x[:, t + 1] == self.Ad @ x[:, t] + self.Bd @ u[:, t]]
            cons += [cp.abs(u[:, t]) <= self.u_max]
        cost += cp.quad_form(x[:, self.H], self.Pterm)
        cp.Problem(cp.Minimize(cost), cons).solve(solver=cp.OSQP, warm_start=True)
        return np.asarray(u.value[:, 0]).reshape(-1)


# --- rollout ------------------------------------------------------------------------

def rollout(env, design, policy, episodes, seed):
    """Mean episodic return of ``policy`` over ``episodes`` episodes.

    ``env`` is a single (squeezed) codesign_gym env: unbatched observation and scalar
    reward, so the policy consumes ``obs`` directly.
    """
    returns = []
    for ep in range(episodes):
        obs, embodiment_rejected, info = env.reset(seed=seed + ep, embodiment=design)
        if embodiment_rejected:
            raise ValueError("design rejected")
        if hasattr(policy, "reset"):
            policy.reset()
        total = 0.0
        done = False
        while not done:
            a = policy(obs) if callable(policy) else policy.act(obs)
            action = np.zeros(env.action_space.shape)          # (A,)
            a = np.asarray(a).reshape(-1)
            action[: a.size] = a
            obs, r, term, trunc, info = env.step(action)
            total += float(r)
            done = bool(term or trunc)
        returns.append(total)
    return float(np.mean(returns)), float(np.std(returns))


class LinearPolicy:
    """u = -K x, reading the full-state measurement block from the observation."""

    def __init__(self, K, nx):
        self.K = K
        self.nx = nx

    def __call__(self, obs):
        return -self.K @ obs[: self.nx]


def zero_policy(obs):
    return np.zeros(1)


# --- rendering ----------------------------------------------------------------------

def save_gif(frames, path, fps):
    """Save a list of (H, W, 3) uint8 frames as an animated GIF."""
    from PIL import Image
    if not frames:
        return None
    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(path, save_all=True, append_images=imgs[1:],
                 duration=int(1000 / max(fps, 1)), loop=0)
    return path


def render_rollout(env_id, n, design, policy, seed, mode, out_path, fps, stride):
    """Roll the optimal policy once with rendering; save a GIF (or show a window).

    Uses a fresh render-enabled env so it is independent of the Monte-Carlo eval env.
    """
    renv, _ = cg.make(
        env_id, n=n, is_eval_env=True,
        render_mode=("human" if mode == "human" else "rgb_array"),
    )
    obs, embodiment_rejected, info = renv.reset(seed=seed, embodiment=design)
    if embodiment_rejected:
        raise ValueError("ground-truth design was rejected")
    if hasattr(policy, "reset"):
        policy.reset()
    frames = []
    f0 = renv.render()                                      # single frame (squeezed)
    if mode != "human" and f0 is not None:
        frames.append(f0)
    t, done = 0, False
    while not done:
        a = policy(obs) if callable(policy) else policy.act(obs)
        action = np.zeros(renv.action_space.shape)          # (A,)
        a = np.asarray(a).reshape(-1)
        action[: a.size] = a
        obs, r, term, trunc, info = renv.step(action)
        t += 1
        frame = renv.render()
        if mode != "human" and frame is not None and t % stride == 0:
            frames.append(frame)
        done = bool(term or trunc)
    renv.close()
    if mode == "human":
        return None
    saved = save_gif(frames, out_path, fps)
    if saved:
        print(f"  rendered optimal rollout -> {saved} ({len(frames)} frames)")
    return saved


def _render_opts(kwargs):
    """Pull render options out of a demo function's kwargs (with defaults)."""
    return dict(
        mode=kwargs.get("render_mode", "off"),
        out=kwargs.get("render_out", None),
        fps=kwargs.get("fps", 20),
        stride=kwargs.get("render_stride", 1),
    )


# --- per-setup demos ----------------------------------------------------------------

def _maybe_render(setup, env_id, n, base, policy, seed, opts):
    if opts["mode"] == "off":
        return
    out = opts["out"] or f"lq_{setup}_optimal.gif"
    render_rollout(env_id, n, base, policy, seed, opts["mode"], out,
                   opts["fps"], opts["stride"])


def demo_lqr(n, episodes, seed, **kwargs):
    print("\n=== Setup 1: LQR (continuous design, full-state feedback) ===")
    env, base = cg.make("LQStructure-LQR", n=n, is_eval_env=True)
    plant = Plant(base["masses"], base["stiffness"], base["damping"],
                  base["actuators"], base["sensors"], env.cfg.dt)
    K, _ = dlqr(plant.Ad, plant.Bd, env.cfg.Q(), env.cfg.R(plant.n_actuators))
    opt = LinearPolicy(K, env.cfg.state_dim)

    analytic = analytic_lqr_return(env, base, K)
    mc_opt, sd = rollout(env, base, opt, episodes, seed)
    mc_zero, _ = rollout(env, base, zero_policy, episodes, seed)
    print(f"  analytic optimal return : {analytic:10.3f}")
    print(f"  MC optimal return       : {mc_opt:10.3f} +/- {sd:.3f}")
    print(f"  MC do-nothing baseline  : {mc_zero:10.3f}")
    rel = abs(mc_opt - analytic) / max(abs(analytic), 1e-9)
    print(f"  |MC - analytic| / |analytic| = {rel:.3%}  (sanity check)")
    _maybe_render("lqr", "LQStructure-LQR", n, base, opt, seed, _render_opts(kwargs))


def demo_lqg(n, episodes, seed, **kwargs):
    print("\n=== Setup 2: LQG (placement design, partial noisy obs + Kalman) ===")
    env, base = cg.make("LQStructure-LQG", n=n, is_eval_env=True)
    ctrl = KalmanController(env, base)
    mc_opt, sd = rollout(env, base, ctrl, episodes, seed)
    mc_zero, _ = rollout(env, base, zero_policy, episodes, seed)
    print(f"  MC LQG return           : {mc_opt:10.3f} +/- {sd:.3f}")
    print(f"  MC do-nothing baseline  : {mc_zero:10.3f}")
    print(f"  improvement over baseline: {mc_opt - mc_zero:10.3f}")
    _maybe_render("lqg", "LQStructure-LQG", n, base, ctrl, seed, _render_opts(kwargs))


def demo_qp(n, episodes, seed, **kwargs):
    print("\n=== Setup 3: QP (actuator saturation under mass/authority budget) ===")
    env, base = cg.make("LQStructure-QP", n=n, is_eval_env=True)
    print(f"  design: actuators={base['actuators']} (budget "
          f"{env.cfg.actuator_budget}), u_max={env.cfg.u_max}")
    ctrl = QPController(env, base)
    mc_opt, sd = rollout(env, base, ctrl, episodes, seed)
    mc_zero, _ = rollout(env, base, zero_policy, episodes, seed)
    print(f"  MC saturated-QP return  : {mc_opt:10.3f} +/- {sd:.3f}")
    print(f"  MC do-nothing baseline  : {mc_zero:10.3f}")
    print(f"  improvement over baseline: {mc_opt - mc_zero:10.3f}")
    _maybe_render("qp", "LQStructure-QP", n, base, ctrl, seed, _render_opts(kwargs))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--setup", choices=["lqr", "lqg", "qp", "all"], default="all")
    p.add_argument("--n", type=int, default=6)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--render", choices=["off", "gif", "human"], default="gif",
                   help="render one optimal-policy rollout per setup (default: gif)")
    p.add_argument("--render-out", default=None,
                   help="output GIF path (single setup); defaults to lq_<setup>_optimal.gif")
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--render-stride", type=int, default=1,
                   help="keep every k-th frame in the GIF")
    args = p.parse_args()

    rkw = dict(render_mode=args.render, render_out=args.render_out,
               fps=args.fps, render_stride=args.render_stride)

    if args.setup in ("lqr", "all"):
        demo_lqr(args.n, args.episodes, args.seed, **rkw)
    if args.setup in ("lqg", "all"):
        demo_lqg(args.n, args.episodes, args.seed, **rkw)
    if args.setup in ("qp", "all"):
        demo_qp(args.n, max(20, args.episodes // 5), args.seed, **rkw)  # QP-MPC is slower


if __name__ == "__main__":
    main()
