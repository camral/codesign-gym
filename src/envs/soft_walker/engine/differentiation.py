"""Optional pure rollouts bound to an environment obtained through the fixed API.

No alternative simulator: decode, instantiate, settling, stepping, observations,
and metrics call the same implementation used by reset/step. Hard graph/category
changes are not differentiable. Validate candidates before differentiating.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, cast

from ..contracts.fields import ACTUATOR_DIRECTIONAL, POISSON_BOUNDS
from ..contracts.status_codes import ROLLOUT_INVALID_INPUT
from .embodiments.constants import (
    FIELD_BOUND_RTOL,
    FIELD_DIRECTION_ATOL,
    PRIMITIVE_AXIS_MIN_NORM2,
)
from .embodiments.decoders import Fields
from .execution.preparation import assemble_spring
from .types import CarryT, Embedding, ParameterT, Policy, RolloutResult

if TYPE_CHECKING:
    from .drivers.spring_environment import Spring2DEnv

import jax
import jax.numpy as jnp

from .drivers.spring_environment import observation
from .physics import spring


def compiled_design_valid(env: Spring2DEnv, emb: Embedding, fields: Fields) -> jax.Array:
    """Check topology, coefficient bounds, and physical fields without host callbacks."""
    if hasattr(env.decoder, "upgrade_embedding"):
        emb = env.decoder.upgrade_embedding(emb)
    if hasattr(env, "_occupancy_shape"):
        from .topology_checks import valid as topology_valid

        valid = topology_valid(fields.occupancy, env._occupancy_shape)
    else:
        from .topology_checks import valid as topology_valid

        valid = topology_valid(fields.occupancy, env.config.grid_shape)
    for k, s in env.embodiment_space.spaces.items():
        a = emb[k]
        valid = (
            valid
            & jnp.all(jnp.isfinite(a))
            & jnp.all(a >= jnp.asarray(s.low))
            & jnp.all(a <= jnp.asarray(s.high))
        )
    # Some composite decoders define a CPPN adjacency template even when the
    # selected representation is primitives or a scalar field. Only validate
    # graph edges when this embodiment actually exposes an edge-mask parameter.
    if hasattr(env.decoder, "allowed") and "edge_mask" in env.embodiment_space.spaces:
        valid = valid & jnp.all(
            jnp.where(jnp.asarray(env.decoder.allowed), True, emb["edge_mask"] == 0)
        )
    if env.decoder.descriptor().get("name") == "primitives":
        valid = (
            valid
            & jnp.any(emb["enabled"] != 0)
            & jnp.all(jnp.sum(emb["axis"] ** 2, axis=-1) > PRIMITIVE_AXIS_MIN_NORM2)
        )
        valid = valid & jnp.all(emb["rounding"] < jnp.min(emb["half_size"], axis=-1))
    for f in fields:
        if f is not None:
            valid = valid & jnp.all(jnp.isfinite(f))
    spec = env.decoder.spec
    if getattr(fields, "poisson", None) is not None:
        valid = (
            valid
            & jnp.all(cast(jax.Array, fields.poisson) >= spec.poisson_bounds[0])
            & jnp.all(cast(jax.Array, fields.poisson) <= spec.poisson_bounds[1])
        )
    if "poisson" in emb and "poisson" not in env.decoder.space().spaces:
        nu = emb["poisson"]
        valid = (
            valid
            & (jnp.ndim(nu) == 0)
            & jnp.all(jnp.isfinite(nu))
            & jnp.all(nu >= POISSON_BOUNDS[0])
            & jnp.all(nu <= POISSON_BOUNDS[1])
        )
    for name, bounds in (
        ("occupancy", (0, 1)),
        ("young", spec.young_bounds),
        ("density", spec.density_bounds),
        ("strength", (0, 1)),
        ("actuator_type", (0, 2)),
        ("channel", (0, spec.channels - 1)),
    ):
        a = getattr(fields, name)
        valid = (
            valid
            & jnp.all(a >= bounds[0] * (1 - FIELD_BOUND_RTOL))
            & jnp.all(a <= bounds[1] * (1 + FIELD_BOUND_RTOL))
        )
    directional = fields.actuator_type == ACTUATOR_DIRECTIONAL
    valid = valid & jnp.all(
        ~directional
        | (jnp.abs(jnp.linalg.norm(fields.direction, axis=-1) - 1) <= FIELD_DIRECTION_ATOL)
    )
    if spec.design_mode == "hard":
        valid = valid & jnp.all((fields.occupancy == 0) | (fields.occupancy == 1))
    return valid


def bind_rollout(
    env: Spring2DEnv, *, record_trajectory: bool = False
) -> Callable[[Embedding, jax.Array], RolloutResult[None]]:
    """Return f(embodiment, padded_or_native_actions[T,K]) -> result dict.

    The function includes reset/settling and is suitable for jit/grad/vmap. User
    custom structural validators are host-side: prevalidate such candidates with
    env.validate_embodiment. Invalid pure results carry valid=False and NaN
    objectives, not a favorable zero score. Differentiation assumes fixed hard
    topology/category choices or the explicitly relaxed surrogate model.
    """
    if hasattr(env, "bind_rollout"):
        return env.bind_rollout(record_trajectory=record_trajectory)
    if not hasattr(env, "_topology"):
        raise TypeError("Bind to a scalar environment created by make")
    cfg, decoder, topo, channels = (env.config, env.decoder, env._topology, env.channels)
    terrain = getattr(env, "_terrain", None)

    def rollout(emb: Embedding, actions: jax.Array) -> RolloutResult[None]:
        """Instantiate an embedding and scan bounded commands through the spring driver.

        Shape and horizon violations raise ValueError; invalid numerical designs retain failure
        status.
        """
        if actions.ndim != 2 or actions.shape[1] not in (channels, env.max_action_space_size):
            raise ValueError("rollout actions must have shape (time, native_or_padded_channels)")
        if actions.shape[0] > cfg.horizon:
            raise ValueError("rollout cannot exceed the task horizon")
        model, fields = assemble_spring(decoder, emb, topo, cfg, channels, terrain)
        valid = (
            compiled_design_valid(env, emb, fields)
            & jnp.all(jnp.isfinite(actions))
            & jnp.all(jnp.abs(actions) <= 1)
        )
        state = env._initialize(model)
        state = state._replace(
            status=jnp.where(valid, state.status, jnp.asarray(ROLLOUT_INVALID_INPUT, jnp.int32))
        )

        def one(
            state: spring.State, action: jax.Array
        ) -> tuple[
            spring.State,
            tuple[spring.Diagnostics, jax.Array] | tuple[spring.Diagnostics, jax.Array, jax.Array],
        ]:
            """Advance one native command and record fixed observation and solver diagnostics."""
            new, diag = spring.advance(model, state, action[:channels], cfg)
            obs, _ = observation(model, new, cfg, env.max_observation_space_size)
            return new, (diag, obs, new.x) if record_trajectory else (diag, obs)

        state, history = jax.lax.scan(jax.checkpoint(one), state, actions)
        m = spring.metrics(model, state, cfg)
        m["objective"] = m["objective"] * jnp.where(
            m["valid"], jnp.asarray(1.0, state.x.dtype), jnp.asarray(jnp.nan, state.x.dtype)
        )
        result: RolloutResult[None] = {
            "state": state,
            "metrics": m,
            "diagnostics": history[0],
            "observations": history[1],
        }
        if record_trajectory:
            result["positions"] = history[2]
        return result

    return rollout


def bind_policy(
    env: Spring2DEnv, policy: Policy[ParameterT, CarryT], steps: int | None = None
) -> Callable[[Embedding, ParameterT, CarryT], RolloutResult[CarryT]]:
    """Trace-safe external closed-loop controller for any scalar JAX driver.

    policy(parameters, observation, carry) -> (bounded action, next carry).
    Returned observations are pre-action policy inputs; bind_rollout observations
    are post-action states. Neither wrapper constructs a learning algorithm.
    """
    if hasattr(env, "kind"):
        from .body_rollout import bind_policy as body_policy

        return body_policy(env, policy, steps)
    if not hasattr(env, "_topology"):
        raise TypeError("Bind to a scalar environment created by make")
    cfg, decoder, topo, channels = (env.config, env.decoder, env._topology, env.channels)
    terrain = getattr(env, "_terrain", None)
    steps = cfg.horizon if steps is None else steps
    if type(steps) is not int or not 1 <= steps <= cfg.horizon:
        raise ValueError("invalid policy rollout length")

    def rollout(emb: Embedding, parameters: ParameterT, carry: CarryT) -> RolloutResult[CarryT]:
        """Decode an embedding and run a differentiable policy with caller-owned recurrent carry."""
        model, fields = assemble_spring(decoder, emb, topo, cfg, channels, terrain)
        state = env._initialize(model)
        valid = compiled_design_valid(env, emb, fields)
        state = state._replace(
            status=jnp.where(valid, state.status, ROLLOUT_INVALID_INPUT).astype(jnp.int32)
        )

        def one(
            c: tuple[spring.State, CarryT], _: None
        ) -> tuple[tuple[spring.State, CarryT], tuple[jax.Array, jax.Array, spring.Diagnostics]]:
            """Observe, evaluate the policy, mark invalid commands, and advance the same physics
            kernel.
            """
            s, memory = c
            obs, _ = observation(model, s, cfg, env.max_observation_space_size)
            action, memory = policy(parameters, obs, memory)
            if action.shape not in ((channels,), (env.max_action_space_size,)):
                raise ValueError("policy action shape mismatch")
            valid = jnp.all(jnp.isfinite(action)) & jnp.all(jnp.abs(action) <= 1)
            s = s._replace(
                status=jnp.where(valid, s.status, ROLLOUT_INVALID_INPUT).astype(jnp.int32)
            )
            new, diag = spring.advance(model, s, action[:channels], cfg)
            return (new, memory), (action, obs, diag)

        (state, carry), history = jax.lax.scan(
            jax.checkpoint(one), (state, carry), None, length=steps
        )
        m = spring.metrics(model, state, cfg)
        m["objective"] = jnp.where(m["valid"], m["objective"], jnp.nan)
        m["scalar_objective"] = m["objective"][0] + cfg.energy_weight * m["objective"][1]
        return {
            "state": state,
            "carry": carry,
            "metrics": m,
            "actions": history[0],
            "observations": history[1],
            "diagnostics": history[2],
            "completed": (state.status == 0) & (steps == cfg.horizon),
        }

    return rollout
