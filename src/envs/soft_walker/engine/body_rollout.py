"""Differentiable trajectories bound to the same instantiated driver as the API."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import jax
import jax.numpy as jnp

from ..contracts.status_codes import ROLLOUT_INVALID_INPUT
from .differentiation import compiled_design_valid
from .drivers import environment as drivers
from .drivers.task import TaskState, metrics
from .physics.common import StepInfo
from .physics.terrain import Terrain
from .types import BodyModel, CarryT, Embedding, ParameterT, Policy, RolloutResult


def prepare(env: drivers.BodyEnv, emb: Embedding, terrain: Terrain) -> tuple[BodyModel, TaskState]:
    """Decode, instantiate, and settle the same model used by the scalar API."""
    base = {k: v for k, v in emb.items() if k != "poisson"}
    fields = env.decoder.decode(base, env._domain.points)
    model = drivers.build_model(
        env.kind, env._domain, fields, env.channels, emb.get("poisson"), terrain, env.config
    )
    valid = compiled_design_valid(env, emb, fields)
    state = cast(TaskState, env._initialize(model))
    return model, state._replace(
        status=jnp.where(valid, state.status, ROLLOUT_INVALID_INPUT).astype(jnp.int32)
    )


def bind(
    env: drivers.BodyEnv, *, record_trajectory: bool = False
) -> Callable[[Embedding, jax.Array], RolloutResult[None]]:
    """Bind a pure open-loop rollout over actions shaped (time, channels).

    The returned function preserves device arrays and supports JIT, gradients,
    and vmapping; prevalidate custom structural constraints on the host.
    """
    cfg = env.config
    terrain = env._terrain

    def rollout(emb: Embedding, actions: jax.Array) -> RolloutResult[None]:
        """Decode and settle an embedding, then scan bounded actions on the captured device.

        Invalid action shapes or excessive rollout lengths raise ValueError before tracing
        physics.
        """
        if actions.ndim != 2 or actions.shape[1] not in (env.channels, env.max_action_space_size):
            raise ValueError("action sequence shape mismatch")
        if actions.shape[0] > cfg.horizon:
            raise ValueError("rollout exceeds horizon")
        model, state = prepare(env, emb, terrain)
        valid = jnp.all(jnp.isfinite(actions)) & jnp.all(jnp.abs(actions) <= 1)
        state = state._replace(
            status=jnp.where(valid, state.status, ROLLOUT_INVALID_INPUT).astype(jnp.int32)
        )

        def one(
            s: TaskState, a: jax.Array
        ) -> tuple[
            TaskState,
            tuple[tuple[StepInfo, jax.Array], jax.Array]
            | tuple[tuple[StepInfo, jax.Array], jax.Array, jax.Array],
        ]:
            """Advance one action and emit fixed diagnostics, observation, and optional positions."""
            new, diag = drivers.advance(env.kind, model, s, a[: env.channels], cfg, env._grid)
            obs, _ = drivers.observe(env.kind, model, new, cfg, env.max_observation_space_size)
            if record_trajectory:
                return new, (diag, obs, drivers.positions(model, new.body, env.kind))
            return new, (diag, obs)

        state, history = jax.lax.scan(jax.checkpoint(one), state, actions)
        result: RolloutResult[None] = {
            "state": state,
            "metrics": metrics(state, cfg),
            "diagnostics": history[0],
            "observations": history[1],
            "completed": (state.status == 0) & (actions.shape[0] == cfg.horizon),
        }
        if record_trajectory:
            result["positions"] = history[2]
        return result

    return rollout


def bind_policy(
    env: drivers.BodyEnv, policy: Policy[ParameterT, CarryT], steps: int | None = None
) -> Callable[[Embedding, ParameterT, CarryT], RolloutResult[CarryT]]:
    """policy(parameters, observation, carry) -> (bounded action, carry).

    The caller supplies the policy, parameterization, and recurrent carry.
    No learning method is required or constructed. Terrain is captured at bind.
    """
    cfg = env.config
    terrain = env._terrain
    steps = cfg.horizon if steps is None else steps
    if not 1 <= steps <= cfg.horizon:
        raise ValueError("invalid policy rollout length")

    def rollout(emb: Embedding, parameters: ParameterT, carry: CarryT) -> RolloutResult[CarryT]:
        """Run feedback actions from the supplied policy parameters and recurrent carry."""
        model, state = prepare(env, emb, terrain)

        def one(
            c: tuple[TaskState, CarryT], _: None
        ) -> tuple[
            tuple[TaskState, CarryT], tuple[jax.Array, jax.Array, tuple[StepInfo, jax.Array]]
        ]:
            """Observe the current state, validate the policy command, and advance one interval."""
            s, memory = c
            obs, _ = drivers.observe(env.kind, model, s, cfg, env.max_observation_space_size)
            action, memory = policy(parameters, obs, memory)
            if action.shape not in ((env.channels,), (env.max_action_space_size,)):
                raise ValueError("policy action shape mismatch")
            valid = jnp.all(jnp.isfinite(action)) & jnp.all(jnp.abs(action) <= 1)
            s = s._replace(
                status=jnp.where(valid, s.status, ROLLOUT_INVALID_INPUT).astype(jnp.int32)
            )
            n, diag = drivers.advance(env.kind, model, s, action[: env.channels], cfg, env._grid)
            return (n, memory), (action, obs, diag)

        (state, carry), hist = jax.lax.scan(jax.checkpoint(one), (state, carry), None, length=steps)
        return {
            "state": state,
            "carry": carry,
            "metrics": metrics(state, cfg),
            "actions": hist[0],
            "observations": hist[1],
            "diagnostics": hist[2],
            "completed": (state.status == 0) & (steps == cfg.horizon),
        }

    return rollout
