# PPO (continuous actions) for the Codesign Gym MuJoCo ports, trained on one fixed design.
# Adapted from CleanRL's ppo_continuous_action.py
# (https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy).
#
# Differences from CleanRL, all forced by the Codesign Gym API:
#   * `make_vec` returns (env, base_embodiment) and the env is natively batched: obs, rewards and flags are torch
#     tensors on `env.device`, so the Gymnasium wrappers are replaced by small torch equivalents below
#     (RunningMeanStd, RewardNormalizer, action clipping, episode stats from `info["final_return"]`).
#   * `reset(embodiment=...)` returns (obs, rejected, info); the design is fixed for the whole run.
#   * Autoreset is NEXT_STEP. Finished copies are reset straight away with `reset_mask`, so the buffer never holds
#     the fake respawn transition and `next_obs` is the new episode's first obs (CleanRL's semantics).
#   * Truncated (time-limit) episodes are bootstrapped with V(final_obs); terminated ones are not.
#
# Python use:
#     from ppo_mujoco import train_ppo          # with notebooks/helpers on sys.path
#     result = train_ppo("Walker2d-v4", embodiment=None, total_timesteps=1_000_000)   # None = base design
#     print(result.eval_returns.mean())
#
# Command line:
#     python helpers/ppo_mujoco.py --env-id Walker2d-v4 --total-timesteps 1000000
#     python helpers/ppo_mujoco.py --env-id Hopper-v4 --embodiment random
#     python helpers/ppo_mujoco.py --env-id Ant-v4 --embodiment 0.1,0,0,...        (one value per design slot)
from __future__ import annotations

import argparse
import dataclasses
import os
import random
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal

from codesign_gym import make_vec

MUJOCO_ENVS = ("HalfCheetah-v4", "Swimmer-v4", "Hopper-v4", "Walker2d-v4", "Ant-v4", "Humanoid-v4", "HumanoidStandup-v4")
NATIVE_PHYSICS_ENVS = ("Walker2d-v4",)  # envs that accept physics="native" (C MuJoCo on the CPU)


@dataclass
class PPOConfig:
    env_id: str = "Walker2d-v4"
    """the id of the Codesign Gym environment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=True`"""
    device: Optional[str] = None
    """simulator and network device: "cpu", "cuda", "cuda:N"; None picks CUDA if available"""
    physics: Optional[str] = None
    """"warp" or "native" (Walker2d only); None uses native C MuJoCo when running on the CPU, as it is much faster"""
    log_dir: Optional[str] = None
    """tensorboard root directory; None disables tensorboard"""
    exp_name: str = "ppo_mujoco"
    """the name of this experiment"""
    save_path: Optional[str] = None
    """if set, save the agent and observation statistics here after training"""
    verbose: bool = True
    """print progress"""

    # Algorithm specific arguments
    total_timesteps: int = 1_000_000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 16
    """the number of parallel simulator worlds"""
    num_steps: int = 128
    """the number of steps to run in each world per policy rollout (batch = num_envs * num_steps)"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: Optional[float] = None
    """the target KL divergence threshold"""
    obs_clip: float = 10.0
    """clip normalized observations to [-obs_clip, obs_clip]"""
    reward_clip: float = 10.0
    """clip normalized rewards to [-reward_clip, reward_clip]"""
    eval_episodes: int = 10
    """deterministic evaluation episodes after training (0 disables)"""

    # to be filled in runtime
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


# --------------------------------------------------------------------------------------------------------------------
# Torch replacements for the Gymnasium wrappers
# --------------------------------------------------------------------------------------------------------------------


class RunningMeanStd:
    """Torch port of gymnasium's RunningMeanStd (parallel Welford update)."""

    def __init__(self, shape=(), device="cpu", epsilon=1e-4):
        self.mean = torch.zeros(shape, dtype=torch.float64, device=device)
        self.var = torch.ones(shape, dtype=torch.float64, device=device)
        self.count = epsilon

    def update(self, x: torch.Tensor) -> None:
        x = x.detach().to(torch.float64).reshape(-1, *self.mean.shape)
        batch_mean, batch_var, batch_count = x.mean(0), x.var(0, unbiased=False), x.shape[0]
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean = self.mean + delta * batch_count / total
        m2 = self.var * self.count + batch_var * batch_count + delta.pow(2) * self.count * batch_count / total
        self.var = m2 / total
        self.count = total

    def normalize(self, x: torch.Tensor, clip: float = 10.0, eps: float = 1e-8) -> torch.Tensor:
        out = (x.to(torch.float64) - self.mean) / torch.sqrt(self.var + eps)
        return out.clamp(-clip, clip).to(torch.float32)

    def state_dict(self) -> dict:
        return {"mean": self.mean.cpu(), "var": self.var.cpu(), "count": self.count}

    @classmethod
    def from_state_dict(cls, state: dict, device="cpu") -> "RunningMeanStd":
        rms = cls(tuple(state["mean"].shape), device=device)
        rms.mean, rms.var, rms.count = state["mean"].to(device), state["var"].to(device), state["count"]
        return rms


class RewardNormalizer:
    """Torch port of gymnasium's NormalizeReward: scale rewards by the std of the discounted return, then clip.
    Unlike gymnasium's version the return accumulator is cleared on truncation as well as termination."""

    def __init__(self, num_envs: int, gamma: float, device="cpu", clip: float = 10.0, eps: float = 1e-8):
        self.rms = RunningMeanStd((), device=device)
        self.returns = torch.zeros(num_envs, dtype=torch.float64, device=device)
        self.gamma, self.clip, self.eps = gamma, clip, eps

    def __call__(self, reward: torch.Tensor, done: torch.Tensor) -> torch.Tensor:
        self.returns = self.returns * self.gamma + reward.to(torch.float64)
        self.rms.update(self.returns)
        self.returns[done] = 0.0
        out = reward.to(torch.float64) / torch.sqrt(self.rms.var + self.eps)
        return out.clamp(-self.clip, self.clip).to(torch.float32)


# --------------------------------------------------------------------------------------------------------------------
# Environment helpers
# --------------------------------------------------------------------------------------------------------------------


def resolve_device(device: Optional[str]) -> str:
    from core.device import resolve_mujoco_device  # codesign-gym's own CUDA/Warp check

    return str(resolve_mujoco_device(device))


def resolve_physics(env_id: str, device: str, physics: Optional[str]) -> Optional[str]:
    """Native C MuJoCo is ~10x faster than Warp's CPU backend, so use it where the env offers it."""
    if physics is not None:
        if physics != "warp" and env_id not in NATIVE_PHYSICS_ENVS:
            raise ValueError(f"{env_id} only supports physics='warp'")
        return physics
    if env_id in NATIVE_PHYSICS_ENVS and device == "cpu":
        return "native"
    return None


def make_env(env_id: str, num_envs: int, *, device: Optional[str] = None, physics: Optional[str] = None,
             is_eval_env: bool = False, **kwargs):
    """`make_vec` plus device/physics resolution. Returns (env, base_embodiment)."""
    device = resolve_device(device)
    physics = resolve_physics(env_id, device, physics)
    if physics is not None:
        kwargs["physics"] = physics
    return make_vec(env_id, num_envs, device=device, is_eval_env=is_eval_env, **kwargs)


def check_embodiment(env, embodiment) -> np.ndarray:
    """Return the design as a float32 array; raise if it is outside the env's embodiment space."""
    embodiment = np.asarray(embodiment, dtype=np.float32)
    valid = np.asarray(env.validate_embodiment(embodiment))
    if not valid.all():
        space = env.embodiment_space
        raise ValueError(f"embodiment {embodiment} is not valid for this env: expected shape {space.shape[-1:]} "
                         f"within [{space.low.min()}, {space.high.max()}]")
    return embodiment


def full_reset(env, embodiment, seed=None) -> torch.Tensor:
    obs, rejected, info = env.reset(seed=seed, embodiment=embodiment)
    if bool(torch.as_tensor(rejected).any()):
        raise ValueError(f"embodiment rejected by the simulator: {embodiment}")
    return obs


# --------------------------------------------------------------------------------------------------------------------
# Agent
# --------------------------------------------------------------------------------------------------------------------


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def mlp(in_dim: int, out_dim: int, out_std: float, hidden: int = 64) -> nn.Sequential:
    return nn.Sequential(
        layer_init(nn.Linear(in_dim, hidden)),
        nn.Tanh(),
        layer_init(nn.Linear(hidden, hidden)),
        nn.Tanh(),
        layer_init(nn.Linear(hidden, out_dim), std=out_std),
    )


class Agent(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.obs_dim, self.act_dim = obs_dim, act_dim
        self.critic = mlp(obs_dim, 1, 1.0)
        self.actor_mean = mlp(obs_dim, act_dim, 0.01)
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_std = torch.exp(self.actor_logstd.expand_as(action_mean))
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)

    def act_deterministic(self, x):
        return self.actor_mean(x)


@dataclass
class PPOResult:
    agent: Agent
    obs_rms: RunningMeanStd
    env_id: str
    embodiment: np.ndarray
    eval_returns: np.ndarray
    history: list = field(default_factory=list)
    """one row per iteration: global_step, episodic_return (mean of episodes that ended), sps, ..."""
    config: Optional[PPOConfig] = None

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        """Deterministic action for raw (unnormalized) observations."""
        device = next(self.agent.parameters()).device
        with torch.no_grad():
            return self.agent.act_deterministic(self.obs_rms.normalize(obs.to(device)))

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save({"agent": self.agent.state_dict(), "obs_rms": self.obs_rms.state_dict(), "env_id": self.env_id,
                    "embodiment": self.embodiment, "obs_dim": self.agent.obs_dim, "act_dim": self.agent.act_dim,
                    "eval_returns": self.eval_returns, "config": dataclasses.asdict(self.config) if self.config else None},
                   path)


# --------------------------------------------------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------------------------------------------------


def evaluate_policy(policy: Callable[[torch.Tensor], torch.Tensor], env_id: str, embodiment=None, *,
                    num_episodes: int = 10, device: Optional[str] = None, physics: Optional[str] = None,
                    seed: int = 12345) -> np.ndarray:
    """Run `num_episodes` episodes in parallel (one per world) and return their undiscounted returns.

    `policy` maps a batch of raw observations (torch, on the env's device) to actions, e.g. `PPOResult.act`.
    `embodiment` is one design for every world, a (num_episodes, n) batch, or None for the base design.
    """
    env, base = make_env(env_id, num_episodes, device=device, physics=physics, is_eval_env=True)
    try:
        embodiment = base if embodiment is None else check_embodiment(env, embodiment)
        obs = full_reset(env, embodiment, seed=seed)
        low = torch.as_tensor(env.single_action_space.low, device=env.device)
        high = torch.as_tensor(env.single_action_space.high, device=env.device)
        returns = torch.zeros(num_episodes, device=env.device)
        finished = torch.zeros(num_episodes, dtype=torch.bool, device=env.device)
        while not bool(finished.all()):
            action = torch.clamp(policy(obs).to(env.device), low, high)
            obs, _, terminated, truncated, info = env.step(action)
            newly = (terminated | truncated) & ~finished
            if bool(newly.any()):
                returns[newly] = info["final_return"][newly].to(returns.dtype)
                finished |= newly
        return returns.cpu().numpy()
    finally:
        env.close()


# --------------------------------------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------------------------------------


def _config(config: Optional[PPOConfig], overrides: dict, cls=PPOConfig):
    config = dataclasses.replace(config) if config is not None else cls()
    for key, value in overrides.items():
        if not hasattr(config, key):
            raise TypeError(f"unknown {cls.__name__} option {key!r}")
        setattr(config, key, value)
    config.batch_size = int(config.num_envs * config.num_steps)
    config.minibatch_size = int(config.batch_size // config.num_minibatches)
    config.num_iterations = max(1, config.total_timesteps // config.batch_size)
    return config


def _seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = deterministic


def _writer(config, run_name):
    if config.log_dir is None:
        return None
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(os.path.join(config.log_dir, run_name))
    writer.add_text("hyperparameters", "|param|value|\n|-|-|\n%s" % (
        "\n".join([f"|{key}|{value}|" for key, value in dataclasses.asdict(config).items()])))
    return writer


def train_ppo(env_id: Optional[str] = None, embodiment=None, config: Optional[PPOConfig] = None, **overrides) -> PPOResult:
    """Train PPO on one fixed design of a Codesign Gym MuJoCo env.

    Args:
        env_id: e.g. "Walker2d-v4" (defaults to `config.env_id`).
        embodiment: a design from `env.embodiment_space` (numpy array of offsets), or None for the base design.
        config: a PPOConfig; any field can also be passed as a keyword, e.g. `total_timesteps=2_000_000`.

    Returns a PPOResult with the trained agent, observation statistics, deterministic eval returns and history.
    """
    if env_id is not None:
        overrides["env_id"] = env_id
    args = _config(config, overrides)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    _seed_everything(args.seed, args.torch_deterministic)

    envs, base = make_env(args.env_id, args.num_envs, device=args.device, physics=args.physics)
    device = envs.device

    writer = _writer(args, run_name)
    try:
        embodiment = base if embodiment is None else check_embodiment(envs, embodiment)
        obs_dim = int(np.prod(envs.single_observation_space.shape))
        act_dim = int(np.prod(envs.single_action_space.shape))
        act_low = torch.as_tensor(envs.single_action_space.low, device=device)
        act_high = torch.as_tensor(envs.single_action_space.high, device=device)

        agent = Agent(obs_dim, act_dim).to(device)
        optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
        obs_rms = RunningMeanStd((obs_dim,), device=device)
        reward_norm = RewardNormalizer(args.num_envs, args.gamma, device=device, clip=args.reward_clip)

        # ALGO Logic: Storage setup
        obs = torch.zeros((args.num_steps, args.num_envs, obs_dim), device=device)
        actions = torch.zeros((args.num_steps, args.num_envs, act_dim), device=device)
        logprobs = torch.zeros((args.num_steps, args.num_envs), device=device)
        rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
        dones = torch.zeros((args.num_steps, args.num_envs), device=device)
        values = torch.zeros((args.num_steps, args.num_envs), device=device)

        global_step = 0
        start_time = time.time()
        raw_obs = full_reset(envs, embodiment, seed=args.seed)
        obs_rms.update(raw_obs)
        next_obs = obs_rms.normalize(raw_obs, args.obs_clip)
        next_done = torch.zeros(args.num_envs, device=device)
        history = []

        for iteration in range(1, args.num_iterations + 1):
            if args.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / args.num_iterations
                optimizer.param_groups[0]["lr"] = frac * args.learning_rate

            episode_returns, episode_lengths = [], []
            for step in range(0, args.num_steps):
                global_step += args.num_envs
                obs[step] = next_obs
                dones[step] = next_done

                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(next_obs)
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

                # ClipAction: the env's ctrl cost uses the action as given, so clip before stepping.
                raw_obs, reward, terminated, truncated, info = envs.step(torch.clamp(action, act_low, act_high))
                done = terminated | truncated
                reward = reward_norm(reward, done)

                if bool(done.any()):
                    # Bootstrap time-limit truncations from the true final observation.
                    trunc_only = truncated & ~terminated
                    if bool(trunc_only.any()):
                        with torch.no_grad():
                            final_obs = obs_rms.normalize(info["final_obs"][trunc_only], args.obs_clip)
                            reward[trunc_only] += args.gamma * agent.get_value(final_obs).flatten()
                    episode_returns += info["final_return"][done].tolist()
                    episode_lengths += info["final_steps"][done].tolist()
                    # Reset finished worlds now (replaces the NEXT_STEP respawn transition).
                    raw_obs, rejected, _ = envs.reset(embodiment=embodiment, options={"reset_mask": done})
                    if bool(torch.as_tensor(rejected).any()):
                        raise RuntimeError("simulator rejected the embodiment on reset")

                rewards[step] = reward
                obs_rms.update(raw_obs)
                next_obs = obs_rms.normalize(raw_obs, args.obs_clip)
                next_done = done.float()

            if episode_returns and writer is not None:
                for r, l in zip(episode_returns, episode_lengths):
                    writer.add_scalar("charts/episodic_return", r, global_step)
                    writer.add_scalar("charts/episodic_length", l, global_step)

            # bootstrap value if not done
            with torch.no_grad():
                next_value = agent.get_value(next_obs).reshape(1, -1)
                advantages = torch.zeros_like(rewards)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values

            # flatten the batch
            b_obs = obs.reshape((-1, obs_dim))
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1, act_dim))
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            # Optimizing the policy and value network
            b_inds = np.arange(args.batch_size)
            clipfracs = []
            for epoch in range(args.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, args.batch_size, args.minibatch_size):
                    mb_inds = b_inds[start:start + args.minibatch_size]

                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    newvalue = newvalue.view(-1)
                    if args.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(newvalue - b_values[mb_inds], -args.clip_coef, args.clip_coef)
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()

                if args.target_kl is not None and approx_kl > args.target_kl:
                    break

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            sps = int(global_step / (time.time() - start_time))
            row = {"global_step": global_step, "sps": sps, "episodes": len(episode_returns),
                   "episodic_return": float(np.mean(episode_returns)) if episode_returns else None,
                   "episodic_length": float(np.mean(episode_lengths)) if episode_lengths else None,
                   "value_loss": v_loss.item(), "policy_loss": pg_loss.item(), "entropy": entropy_loss.item(),
                   "approx_kl": approx_kl.item(), "clipfrac": float(np.mean(clipfracs)), "explained_variance": float(explained_var)}
            history.append(row)
            if writer is not None:
                writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
                writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
                writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
                writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
                writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
                writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
                writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
                writer.add_scalar("losses/explained_variance", explained_var, global_step)
                writer.add_scalar("charts/SPS", sps, global_step)
            if args.verbose and (iteration % 10 == 0 or iteration == args.num_iterations):
                ret = f"{row['episodic_return']:.1f}" if row["episodic_return"] is not None else "n/a"
                print(f"[{args.env_id}] iter {iteration}/{args.num_iterations} step {global_step} "
                      f"return {ret} (n={row['episodes']}) SPS {sps}", flush=True)
    finally:
        envs.close()

    result = PPOResult(agent=agent, obs_rms=obs_rms, env_id=args.env_id, embodiment=np.asarray(embodiment),
                       eval_returns=np.zeros(0), history=history, config=args)
    if args.save_path:
        result.save(args.save_path)  # save before evaluating, so a failed evaluation never loses the agent
    if args.eval_episodes > 0:
        result.eval_returns = evaluate_policy(result.act, args.env_id, embodiment, num_episodes=args.eval_episodes,
                                              device=args.device, physics=args.physics, seed=args.seed + 10_000)
        if args.verbose:
            print(f"[{args.env_id}] deterministic eval return {result.eval_returns.mean():.1f} "
                  f"± {result.eval_returns.std():.1f} over {args.eval_episodes} episodes", flush=True)
        if writer is not None:
            writer.add_scalar("eval/episodic_return_mean", float(result.eval_returns.mean()), global_step)
    if writer is not None:
        writer.close()
    if args.save_path:
        result.save(args.save_path)
    return result


def load_result(path: str, device: str = "cpu") -> PPOResult:
    """Load a PPOResult saved with `PPOResult.save` / `save_path`."""
    data = torch.load(path, map_location=device, weights_only=False)
    agent = Agent(data["obs_dim"], data["act_dim"]).to(device)
    agent.load_state_dict(data["agent"])
    config = PPOConfig(**data["config"]) if data.get("config") else None
    return PPOResult(agent=agent, obs_rms=RunningMeanStd.from_state_dict(data["obs_rms"], device), env_id=data["env_id"],
                     embodiment=data["embodiment"], eval_returns=data["eval_returns"], config=config)


# --------------------------------------------------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------------------------------------------------


def _str2bool(s: str) -> bool:
    if s.lower() in ("1", "true", "yes", "y", "on"):
        return True
    if s.lower() in ("0", "false", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean, got {s!r}")


def _optional(cast):
    return lambda s: None if s.lower() == "none" else cast(s)


def build_parser(cls=PPOConfig, defaults: Optional[dict] = None) -> argparse.ArgumentParser:
    """An argparse parser with one --flag per config field (replaces CleanRL's tyro CLI)."""
    parser = argparse.ArgumentParser(description=cls.__doc__)
    hints = {"bool": _str2bool, "int": int, "float": float, "str": str,
             "Optional[str]": _optional(str), "Optional[float]": _optional(float), "Optional[int]": _optional(int)}
    for f in dataclasses.fields(cls):
        if f.name in ("batch_size", "minibatch_size", "num_iterations"):
            continue
        cast = hints.get(f.type if isinstance(f.type, str) else getattr(f.type, "__name__", str(f.type)), str)
        default = (defaults or {}).get(f.name, f.default)
        parser.add_argument("--" + f.name.replace("_", "-"), dest=f.name, type=cast, default=default)
    return parser


def parse_embodiment(text: Optional[str], env_id: str):
    """None/"base" -> base design; "random" -> a uniform sample; otherwise comma-separated floats."""
    if text is None or text == "base":
        return None
    if text == "random":
        env, _ = make_env(env_id, 1, device="cpu")
        try:
            return np.asarray(env.embodiment_space.sample(), dtype=np.float32).reshape(-1)
        finally:
            env.close()
    return np.array([float(v) for v in text.split(",")], dtype=np.float32)


if __name__ == "__main__":
    parser = build_parser(defaults={"log_dir": os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs")})
    parser.add_argument("--embodiment", default=None, help='"base" (default), "random", or comma-separated offsets')
    cli = vars(parser.parse_args())
    design = parse_embodiment(cli.pop("embodiment"), cli["env_id"])
    train_ppo(embodiment=design, config=PPOConfig(**cli))
