# PPO co-design for the Codesign Gym MuJoCo ports: the RL agent chooses the embodiment as well as the control.
#
# A PPO port of the "design as action" scheme in codesign-baselines' FastTD3
# (fasttd3_rl/fast_td3/codesign/joint.py and gym_train.py):
#   * Every world starts each episode with one zero-time *design step*. The policy outputs a design in normalized
#     [-1, 1] coordinates; it is mapped to the env's embodiment bounds and applied with a masked `reset`.
#   * Policy and critic see x = [normalized obs ‖ normalized design ‖ design-phase flag] (FastTD3's
#     JointRollout.state()), so the controller is conditioned on the body it is driving.
#   * The design step is a transition with reward 0 (or `rejection_reward` if the simulator rejects the design)
#     and discount 1 into the episode's first control state, so the episode's return is credited to the design
#     choice (FastTD3: design_transition_discount = 1.0). Here that credit flows through GAE:
#         A_design = r_design + V(x_t0) - V(x_design) + lambda * A_control[t0]
#   * Rejected designs fall back to the base design, as in FastTD3.
#
# Differences from FastTD3 (beyond PPO vs TD3):
#   * Design and control have separate Gaussian heads and log-stds (FastTD3 uses one actor for [design ‖ control]).
#   * The design-phase state is [0 ‖ 0 ‖ 1] rather than [previous terminal obs ‖ previous design ‖ 1]. FastTD3's
#     evaluation already queries the design with a fresh state, so this keeps training and evaluation identical.
#
# Python use:
#     from ppo_mujoco_codesign import train_ppo_codesign          # with notebooks/helpers on sys.path
#     result = train_ppo_codesign("Walker2d-v4", total_timesteps=2_000_000)
#     result.design            # the chosen design (env units), result.eval_returns on that design
#
# Command line:
#     python helpers/ppo_mujoco_codesign.py --env-id Walker2d-v4 --total-timesteps 2000000
from __future__ import annotations

import dataclasses
import math
import os
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal

from ppo_mujoco import (
    PPOConfig,
    RewardNormalizer,
    RunningMeanStd,
    _config,
    _seed_everything,
    _writer,
    build_parser,
    evaluate_policy,
    make_env,
    mlp,
)

DEFAULT_REJECTION_REWARD = -250.0  # codesign_adapter.default_rejection_reward for MuJoCo envs


@dataclass
class CodesignPPOConfig(PPOConfig):
    exp_name: str = "ppo_mujoco_codesign"
    """the name of this experiment"""
    num_envs: int = 64
    """more worlds -> more design samples per PPO iteration"""
    num_steps: int = 64
    """steps per world per rollout (batch = num_envs * num_steps)"""
    design_coef: float = 1.0
    """weight of the design policy + design value loss"""
    design_ent_coef: float = 0.0
    """entropy bonus on the design distribution"""
    design_init_std: float = 0.3
    """initial std of the design distribution, in normalized [-1, 1] design units"""
    rejection_reward: float = DEFAULT_REJECTION_REWARD
    """reward (raw units, normalized like the env reward) for a design the simulator rejects"""
    eval_on_base: bool = True
    """also evaluate the trained controller on the base design, for comparison"""


class CodesignAgent(nn.Module):
    """Control head pi(a | x), design head pi(d | x_design), one critic V(x); x = [obs ‖ design ‖ phase]."""

    def __init__(self, obs_dim: int, act_dim: int, design_dim: int, design_init_std: float = 0.3):
        super().__init__()
        self.obs_dim, self.act_dim, self.design_dim = obs_dim, act_dim, design_dim
        in_dim = obs_dim + design_dim + 1
        self.critic = mlp(in_dim, 1, 1.0)
        self.actor_mean = mlp(in_dim, act_dim, 0.01)
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))
        self.design_mean = mlp(in_dim, design_dim, 0.01)  # ~0 at init: starts at the centre of the design range
        self.design_logstd = nn.Parameter(torch.full((1, design_dim), math.log(design_init_std)))

    def state(self, norm_obs: torch.Tensor, design: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        return torch.cat([norm_obs, design, phase.float().reshape(-1, 1)], dim=-1)

    def design_state(self, n: int, device) -> torch.Tensor:
        return torch.cat([torch.zeros(n, self.obs_dim + self.design_dim, device=device), torch.ones(n, 1, device=device)], -1)

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        mean = self.actor_mean(x)
        probs = Normal(mean, torch.exp(self.actor_logstd.expand_as(mean)))
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)

    def get_design_and_value(self, x, design=None):
        mean = self.design_mean(x)
        probs = Normal(mean, torch.exp(self.design_logstd.expand_as(mean)))
        if design is None:
            design = probs.sample()
        return design, probs.log_prob(design).sum(1), probs.entropy().sum(1), self.critic(x)

    def deterministic_design(self, device) -> torch.Tensor:
        """The mean design (normalized, clipped to [-1, 1])."""
        with torch.no_grad():
            return self.design_mean(self.design_state(1, device)).clamp(-1, 1)[0]


@dataclass
class CodesignResult:
    agent: CodesignAgent
    obs_rms: RunningMeanStd
    env_id: str
    design: np.ndarray
    """the agent's deterministic design, in env units (pass to `env.reset(embodiment=...)`)"""
    design_normalized: np.ndarray
    eval_returns: np.ndarray
    """deterministic controller on `design`"""
    base_eval_returns: np.ndarray
    """deterministic controller on the base design (if `eval_on_base`)"""
    history: list = field(default_factory=list)
    config: Optional[CodesignPPOConfig] = None

    def policy_for(self, design_normalized):
        """A raw-obs -> action function for a fixed (normalized) design, for `evaluate_policy`."""
        agent, rms = self.agent, self.obs_rms
        device = next(agent.parameters()).device
        d = torch.as_tensor(design_normalized, dtype=torch.float32, device=device)

        def act(obs):
            with torch.no_grad():
                obs = obs.to(device)
                x = agent.state(rms.normalize(obs), d.expand(obs.shape[0], -1), torch.zeros(obs.shape[0], device=device))
                return agent.actor_mean(x)

        return act

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save({"agent": self.agent.state_dict(), "obs_rms": self.obs_rms.state_dict(), "env_id": self.env_id,
                    "design": self.design, "design_normalized": self.design_normalized,
                    "dims": (self.agent.obs_dim, self.agent.act_dim, self.agent.design_dim),
                    "eval_returns": self.eval_returns, "base_eval_returns": self.base_eval_returns,
                    "config": dataclasses.asdict(self.config) if self.config else None}, path)


def _ppo_loss(newlogprob, oldlogprob, adv, newvalue, oldvalue, ret, entropy, args, ent_coef, norm_adv=True):
    logratio = newlogprob - oldlogprob
    ratio = logratio.exp()
    if norm_adv and adv.numel() > 1:
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    pg_loss = torch.max(-adv * ratio, -adv * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)).mean()
    newvalue = newvalue.view(-1)
    if args.clip_vloss:
        v_clipped = oldvalue + torch.clamp(newvalue - oldvalue, -args.clip_coef, args.clip_coef)
        v_loss = 0.5 * torch.max((newvalue - ret) ** 2, (v_clipped - ret) ** 2).mean()
    else:
        v_loss = 0.5 * ((newvalue - ret) ** 2).mean()
    with torch.no_grad():
        approx_kl = ((ratio - 1) - logratio).mean()
        clipfrac = ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
    loss = pg_loss - ent_coef * entropy.mean() + args.vf_coef * v_loss
    return loss, pg_loss, v_loss, approx_kl, clipfrac


def train_ppo_codesign(env_id: Optional[str] = None, config: Optional[CodesignPPOConfig] = None, **overrides) -> CodesignResult:
    """Jointly learn a design and a design-conditioned controller with PPO on a Codesign Gym MuJoCo env.

    Any CodesignPPOConfig field can be passed as a keyword, e.g. `total_timesteps=2_000_000, num_envs=128`.
    """
    if env_id is not None:
        overrides["env_id"] = env_id
    args = _config(config, overrides, cls=CodesignPPOConfig)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    _seed_everything(args.seed, args.torch_deterministic)

    envs, base = make_env(args.env_id, args.num_envs, device=args.device, physics=args.physics)
    device = envs.device
    writer = _writer(args, run_name)
    N = args.num_envs
    try:
        obs_dim = int(np.prod(envs.single_observation_space.shape))
        act_dim = int(np.prod(envs.single_action_space.shape))
        design_dim = len(base)
        act_low = torch.as_tensor(envs.single_action_space.low, device=device)
        act_high = torch.as_tensor(envs.single_action_space.high, device=device)
        d_low = torch.as_tensor(np.asarray(envs.embodiment_space.low, dtype=np.float32).reshape(-1, design_dim)[0], device=device)
        d_high = torch.as_tensor(np.asarray(envs.embodiment_space.high, dtype=np.float32).reshape(-1, design_dim)[0], device=device)
        d_span = (d_high - d_low).clamp_min(1e-12)
        base_t = torch.as_tensor(base, dtype=torch.float32, device=device)
        base_norm = torch.where(d_high > d_low, 2 * (base_t - d_low) / d_span - 1, torch.zeros_like(base_t))

        def to_env_units(d_norm):
            actual = d_low + (d_norm.clamp(-1, 1) + 1) * 0.5 * (d_high - d_low)
            return torch.minimum(torch.maximum(actual, d_low), d_high)  # guard float round-off at the bounds

        agent = CodesignAgent(obs_dim, act_dim, design_dim, args.design_init_std).to(device)
        optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
        obs_rms = RunningMeanStd((obs_dim,), device=device)
        reward_norm = RewardNormalizer(N, args.gamma, device=device, clip=args.reward_clip)
        x_dim = obs_dim + design_dim + 1

        # Control-step storage (as in CleanRL) ...
        xs = torch.zeros((args.num_steps, N, x_dim), device=device)
        actions = torch.zeros((args.num_steps, N, act_dim), device=device)
        logprobs = torch.zeros((args.num_steps, N), device=device)
        rewards = torch.zeros((args.num_steps, N), device=device)
        dones = torch.zeros((args.num_steps, N), device=device)
        values = torch.zeros((args.num_steps, N), device=device)

        global_step = 0
        start_time = time.time()
        raw_obs = envs.reset(seed=args.seed, embodiment=base)[0]  # every world gets a design before its first step
        designs = base_norm.expand(N, -1).clone()                 # current normalized design of each world
        need_design = torch.ones(N, dtype=torch.bool, device=device)
        next_done = torch.zeros(N, device=device)
        history = []
        total_candidates = total_rejections = 0

        for iteration in range(1, args.num_iterations + 1):
            if args.anneal_lr:
                optimizer.param_groups[0]["lr"] = (1.0 - (iteration - 1.0) / args.num_iterations) * args.learning_rate

            # ... plus a side buffer of design steps: (x_d, d, logp, V(x_d), reward, step t0, world, rejected)
            d_x, d_act, d_logp, d_val, d_rew, d_t, d_env, d_rej = [], [], [], [], [], [], [], []
            episode_returns, episode_lengths = [], []
            for step in range(args.num_steps):
                if bool(need_design.any()):
                    # Zero-time design step for every world that is starting an episode.
                    idx = torch.nonzero(need_design).flatten()
                    x_d = agent.design_state(len(idx), device)
                    with torch.no_grad():
                        d, logp, _, v = agent.get_design_and_value(x_d)
                    designs[idx] = d.clamp(-1, 1)
                    raw_obs, rejected, _ = envs.reset(embodiment=to_env_units(designs).cpu().numpy(),
                                                      options={"reset_mask": need_design})
                    rejected = torch.as_tensor(rejected, dtype=torch.bool, device=device)
                    if bool(rejected.any()):
                        designs[rejected] = base_norm
                        raw_obs, again, _ = envs.reset(embodiment=to_env_units(designs).cpu().numpy(),
                                                       options={"reset_mask": rejected})
                        if bool(torch.as_tensor(again).any()):
                            raise RuntimeError("simulator rejected the base design")
                    rej = rejected[idx]
                    scale = torch.sqrt(reward_norm.rms.var + 1e-8).float()
                    r_d = torch.where(rej, torch.full_like(v.flatten(), args.rejection_reward) / scale, torch.zeros_like(v.flatten()))
                    d_x.append(x_d); d_act.append(d); d_logp.append(logp); d_val.append(v.flatten()); d_rew.append(r_d)
                    d_t.append(torch.full_like(idx, step)); d_env.append(idx); d_rej.append(rej)
                    total_candidates += len(idx)
                    total_rejections += int(rej.sum())
                    need_design.zero_()
                    obs_rms.update(raw_obs)
                    next_done[idx] = 1.0  # the control step at t0 starts a new episode

                global_step += N
                x = agent.state(obs_rms.normalize(raw_obs, args.obs_clip), designs, torch.zeros(N, device=device))
                xs[step] = x
                dones[step] = next_done
                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(x)
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

                raw_obs, reward, terminated, truncated, info = envs.step(torch.clamp(action, act_low, act_high))
                done = terminated | truncated
                reward = reward_norm(reward, done)
                if bool(done.any()):
                    trunc_only = truncated & ~terminated
                    if bool(trunc_only.any()):
                        with torch.no_grad():
                            fx = agent.state(obs_rms.normalize(info["final_obs"][trunc_only], args.obs_clip),
                                             designs[trunc_only], torch.zeros(int(trunc_only.sum()), device=device))
                            reward[trunc_only] += args.gamma * agent.get_value(fx).flatten()
                    episode_returns += info["final_return"][done].tolist()
                    episode_lengths += info["final_steps"][done].tolist()
                    need_design |= done  # these worlds pick a new design (and reset) before their next step
                rewards[step] = reward
                if not bool(done.all()):
                    obs_rms.update(raw_obs[~done])
                next_done = done.float()

            # GAE over control steps. A world that finished at the last step has next_done = 1, so its
            # (stale) next value is masked out; a world mid-episode bootstraps from V(x_next).
            with torch.no_grad():
                next_x = agent.state(obs_rms.normalize(raw_obs, args.obs_clip), designs, torch.zeros(N, device=device))
                next_value = agent.get_value(next_x).reshape(1, -1)
                advantages = torch.zeros_like(rewards)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal, nextvalues = 1.0 - next_done, next_value
                    else:
                        nextnonterminal, nextvalues = 1.0 - dones[t + 1], values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values

                # Design steps: discount 1 into the first control state x_t0 (no bootstrap if rejected).
                n_design = sum(len(a) for a in d_act)
                if n_design:
                    bx, bd, blogp = torch.cat(d_x), torch.cat(d_act), torch.cat(d_logp)
                    bval, brew, bt, benv, brej = torch.cat(d_val), torch.cat(d_rew), torch.cat(d_t), torch.cat(d_env), torch.cat(d_rej)
                    cont = (~brej).float()
                    b_dadv = brew + cont * (values[bt, benv] - bval) + cont * args.gae_lambda * advantages[bt, benv] - (1 - cont) * bval
                    b_dret = b_dadv + bval

            b_x = xs.reshape(-1, x_dim)
            b_logprobs, b_actions = logprobs.reshape(-1), actions.reshape(-1, act_dim)
            b_advantages, b_returns, b_values = advantages.reshape(-1), returns.reshape(-1), values.reshape(-1)

            b_inds = np.arange(args.batch_size)
            clipfracs, d_clipfracs = [], []
            d_pg = d_v = d_kl = torch.tensor(0.0)
            for epoch in range(args.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, args.batch_size, args.minibatch_size):
                    mb = b_inds[start:start + args.minibatch_size]
                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_x[mb], b_actions[mb])
                    loss, pg_loss, v_loss, approx_kl, cf = _ppo_loss(
                        newlogprob, b_logprobs[mb], b_advantages[mb], newvalue, b_values[mb], b_returns[mb], entropy,
                        args, args.ent_coef, args.norm_adv)
                    clipfracs.append(cf)
                    if n_design:
                        # The design batch is small (one sample per episode), so it joins every minibatch update;
                        # the PPO clip keeps the repeated updates within the trust region.
                        _, dlogp_new, dent, dval_new = agent.get_design_and_value(bx, bd)
                        dloss, d_pg, d_v, d_kl, dcf = _ppo_loss(dlogp_new, blogp, b_dadv, dval_new, bval, b_dret, dent,
                                                                args, args.design_ent_coef, args.norm_adv)
                        d_clipfracs.append(dcf)
                        loss = loss + args.design_coef * dloss
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()
                if args.target_kl is not None and approx_kl > args.target_kl:
                    break

            sps = int(global_step / (time.time() - start_time))
            mean_design = agent.deterministic_design(device)
            design_std = torch.exp(agent.design_logstd).mean().item()
            row = {"global_step": global_step, "sps": sps, "episodes": len(episode_returns),
                   "train/reward": float(np.mean(episode_returns)) if episode_returns else None,
                   "episodic_length": float(np.mean(episode_lengths)) if episode_lengths else None,
                   "embodiment_candidates": n_design,
                   "embodiment_rejection_rate": float(torch.cat(d_rej).float().mean()) if n_design else 0.0,
                   "design/mean_abs": float(mean_design.abs().mean()), "design/std": design_std,
                   "design/value": float(bval.mean()) if n_design else None,
                   "design/policy_loss": d_pg.item(), "design/value_loss": d_v.item(), "design/approx_kl": d_kl.item(),
                   "value_loss": v_loss.item(), "policy_loss": pg_loss.item(), "approx_kl": approx_kl.item(),
                   "clipfrac": float(np.mean(clipfracs))}
            history.append(row)
            if writer is not None:
                for k, v in row.items():
                    if v is not None and k != "global_step":
                        writer.add_scalar(k if "/" in k else f"charts/{k}", v, global_step)
                writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            if args.verbose and (iteration % 10 == 0 or iteration == args.num_iterations):
                ret = f"{row['train/reward']:.1f}" if row["train/reward"] is not None else "n/a"
                print(f"[{args.env_id} codesign] iter {iteration}/{args.num_iterations} step {global_step} return {ret} "
                      f"(n={row['episodes']}) |design| {row['design/mean_abs']:.3f} design std {design_std:.3f} "
                      f"rejected {total_rejections}/{total_candidates} SPS {sps}", flush=True)
    finally:
        envs.close()

    design_norm = agent.deterministic_design(device)
    result = CodesignResult(agent=agent, obs_rms=obs_rms, env_id=args.env_id,
                            design=to_env_units(design_norm).cpu().numpy(), design_normalized=design_norm.cpu().numpy(),
                            eval_returns=np.zeros(0), base_eval_returns=np.zeros(0), history=history, config=args)
    if args.save_path:
        result.save(args.save_path)  # save before evaluating, so a failed evaluation never loses the agent
    if args.eval_episodes > 0:
        eval_kw = dict(num_episodes=args.eval_episodes, device=args.device, physics=args.physics, seed=args.seed + 10_000)
        result.eval_returns = evaluate_policy(result.policy_for(design_norm), args.env_id, result.design, **eval_kw)
        if args.eval_on_base:
            result.base_eval_returns = evaluate_policy(result.policy_for(base_norm), args.env_id, base, **eval_kw)
        if args.verbose:
            print(f"[{args.env_id} codesign] chosen design (env units): {np.round(result.design, 3).tolist()}")
            print(f"[{args.env_id} codesign] eval return on chosen design {result.eval_returns.mean():.1f} "
                  f"± {result.eval_returns.std():.1f}", flush=True)
            if args.eval_on_base:
                print(f"[{args.env_id} codesign] same controller on base design {result.base_eval_returns.mean():.1f} "
                      f"± {result.base_eval_returns.std():.1f}", flush=True)
        if writer is not None:
            writer.add_scalar("eval/reward", float(result.eval_returns.mean()), global_step)
    if writer is not None:
        writer.close()
    if args.save_path:
        result.save(args.save_path)
    return result


if __name__ == "__main__":
    parser = build_parser(CodesignPPOConfig, defaults={"log_dir": os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs")})
    train_ppo_codesign(config=CodesignPPOConfig(**vars(parser.parse_args())))
