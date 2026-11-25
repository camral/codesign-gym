"""Concrete drivers implementing the supplied, intentionally nonstandard API."""

from __future__ import annotations

import time
import warnings
from collections.abc import Callable
from copy import deepcopy
from functools import lru_cache
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike, DTypeLike

from ... import host_topology
from ...contracts.observations import (
    ACTUATOR_OBSERVATION_WIDTH,
    OBSERVATION_DENSITY_SCALE_KG_M3,
    OBSERVATION_YOUNG_SCALE_PA,
    SPRING_FIELD_WIDTH,
    SPRING_NODE_WIDTH,
    TERRAIN_PROBE_COUNT,
    TERRAIN_PROBE_FORWARD_OFFSETS,
    spring_observation_layout,
)
from ...contracts.status_codes import (
    ROLLOUT_INVALID_INPUT,
    SPRING_STATE_INVALID_GEOMETRY,
    SPRING_STATE_NONFINITE,
    SPRING_STATE_OK,
    SPRING_STATE_SOLVE_FAILED,
)
from ..action_timing import with_action_hold
from ..device import resolve_device, resolve_dtype
from ..embodiments.decoders import Decoder, FieldSpec, cast_embedding, check_embedding, make_decoder
from ..embodiments.fields import field_errors as field_errors
from ..physics import spring
from ..physics.spring_config import SpringConfig, with_integrator
from ..spaces import (
    AUTORESET_NEXT_STEP as AUTORESET_NEXT_STEP,
    Box,
    Dict as Dict,
    Env,
    ResetNeeded,
    VectorEnv,
)
from ..types import (
    AdvanceKernel,
    Array,
    DriverState,
    Embedding,
    HostArray,
    HostEmbedding,
    Info,
    InitializeKernel,
    ObserveKernel,
    PhysicsModel,
    ReportingSummary,
    RewardKernel,
    SpringSummary,
    SummaryKernel,
    TransitionDiagnostics,
)
from .vector_lifecycle import (
    VectorLifecycle,
    _stack_info as _stack_info,
    batch_space as batch_space,
)

STATUS_NAMES = {
    SPRING_STATE_OK: "OK",
    SPRING_STATE_SOLVE_FAILED: "NONLINEAR_SOLVE_FAILED",
    SPRING_STATE_INVALID_GEOMETRY: "GEOMETRY_INVALID",
    SPRING_STATE_NONFINITE: "NONFINITE_STATE",
    ROLLOUT_INVALID_INPUT: "INVALID_DESIGN",
}


def observation(
    model: spring.Model, state: spring.State, cfg: SpringConfig, max_size: int
) -> tuple[jax.Array, jax.Array]:
    """Build the padded spring observation and validity mask on device.

    Lengths and velocities use configuration scales; the final time entry is
    normalized by the policy-action horizon. Inactive nodes remain zero-masked.
    """
    center = spring.com(model, state.x)
    velocity = spring.com(model, state.v)
    global_values = jnp.concatenate(
        (
            center / cfg.length_ref,
            velocity / cfg.velocity_ref,
            jnp.reshape(state.time / (cfg.horizon * cfg.control_dt), (1,)),
        )
    )
    node_values = jnp.concatenate(
        (
            jnp.where(model.active[:, None], (state.x - model.reference) / cfg.length_ref, 0.0),
            jnp.where(model.active[:, None], state.v / cfg.velocity_ref, 0.0),
            model.active[:, None].astype(state.x.dtype),
        ),
        axis=-1,
    ).reshape(-1)
    cell_values = jnp.stack(
        (
            model.occupancy,
            model.occupancy * jnp.log(model.young / OBSERVATION_YOUNG_SCALE_PA),
            model.occupancy * model.density / OBSERVATION_DENSITY_SCALE_KG_M3,
            model.occupancy * model.strength,
        ),
        axis=-1,
    ).reshape(-1)
    channel_values = jnp.stack(
        (state.actuator, model.channel_mask.astype(state.x.dtype)), axis=-1
    ).reshape(-1)
    values = jnp.concatenate((global_values, node_values, cell_values, channel_values))
    mask = jnp.concatenate(
        (
            jnp.ones_like(global_values, dtype=bool),
            jnp.repeat(model.active, SPRING_NODE_WIDTH),
            jnp.repeat(model.occupancy > 0, SPRING_FIELD_WIDTH),
            jnp.repeat(model.channel_mask, ACTUATOR_OBSERVATION_WIDTH),
        )
    )
    if model.terrain is not None:
        from ..physics.terrain import gap_normal

        probes = center[None, :] + jnp.asarray(
            [
                [0.0, 0.0],
                [TERRAIN_PROBE_FORWARD_OFFSETS[1] * cfg.length_ref, 0.0],
                [cfg.length_ref, 0.0],
            ],
            state.x.dtype,
        )
        gap, n = gap_normal(probes, model.terrain)
        values = jnp.concatenate(
            (values, jnp.concatenate((gap[:, None] / cfg.length_ref, n), axis=-1).reshape(-1))
        )
        mask = jnp.concatenate((mask, jnp.ones(TERRAIN_PROBE_COUNT * 3, dtype=bool)))
    return jnp.pad(values, (0, max_size - values.shape[0])), jnp.pad(
        mask, (0, max_size - mask.shape[0])
    )


@lru_cache(maxsize=48)
def spring_reporting_kernels(
    cfg: SpringConfig, max_size: int
) -> tuple[ObserveKernel, SummaryKernel, RewardKernel]:
    """Share compiled view functions across identical scalar/vector lanes."""
    observe = jax.jit(lambda m, s: observation(m, s, cfg, max_size))
    summary = jax.jit(
        lambda m, s: (
            spring.metrics(m, s, cfg),
            spring.energy_budget(m, s, cfg),
            jnp.sum(m.mass),
            jnp.sum(m.occupancy),
            jnp.sum(m.active),
            jnp.sum(m.occupancy > 0),
        )
    )
    reward = jax.jit(lambda m, b, a: spring.transition_reward(m, b, a, cfg))
    return (observe, summary, reward)


def topology_errors(occupancy: ArrayLike, shape: tuple[int, int]) -> list[str]:
    """Report empty, disconnected, or vertex-only cell geometry on the host."""
    return host_topology.errors(occupancy, shape, junction_error="VERTEX_ONLY_JUNCTION")


class Spring2DEnv(Env):
    """Working M1 driver. Reset returns (obs, rejection_array, info).

    ``physics_steps_per_action`` holds each command for that many existing
    physics timesteps. The horizon counts actions; reset settling is unchanged.
    ``integrator`` overrides only the method in a supplied configuration; None
    preserves it. Without a configuration, native Kvaerno selects its required
    GMRES solver. Other solver and tolerance settings retain their defaults.
    """

    # Scalar lifecycle fields are empty until a successful reset. BodyEnv
    # uses the same lifecycle with a different physical model/state pair.
    _model: PhysicsModel | None
    _state: DriverState | None
    _initial: DriverState | None
    _terrain_seed: int | None
    _advance: AdvanceKernel
    _advance_batch: AdvanceKernel
    _initialize: InitializeKernel

    _reset_options_error = "Spring2D has no scalar reset options"
    _reset_seed_error = "seed must be a nonnegative integer"

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50, "jax": True}

    def __init__(
        self,
        *,
        is_eval_env: bool = False,
        max_action_space_size: int = 4,
        max_observation_space_size: int = 256,
        backend: str = "jax",
        dtype: DTypeLike = "float32",
        device: str | jax.Device | None = None,
        decoder: str | Decoder = "direct",
        design_mode: str = "hard",
        material_mode: str = "continuous",
        channels: int = 2,
        config: SpringConfig | None = None,
        render_mode: str | None = None,
        physics_steps_per_action: int | None = None,
        integrator: str | None = None,
    ) -> None:
        """Configure the spring driver, numerical kernels, decoder, and fixed API spaces.

        No episode is created until reset. Unsupported backends raise NotImplementedError;
        incompatible decoder precision, dimensions, or capacities raise ValueError.
        """
        if backend != "jax":
            raise NotImplementedError(
                "Spring2DEnv implements JAX; use engine.make(..., backend='warp') for native Warp"
            )
        self._initialization_config = with_integrator(config, integrator)
        self.integrator = self._initialization_config.integrator
        self.dtype = resolve_dtype(dtype)
        self.device = resolve_device(device)
        self.config = with_action_hold(self._initialization_config, physics_steps_per_action)
        self.physics_steps_per_action = self.config.substeps
        self.physics_dt = self.config.dt
        self.control_dt = self.config.control_dt
        self.is_eval_env = bool(is_eval_env)
        self.supports_transition_embodiment = False
        self.render_mode = render_mode
        if render_mode not in (None, "rgb_array"):
            raise ValueError("M1 render_mode is None or rgb_array")
        self.metadata = dict(type(self).metadata, render_fps=int(round(1 / self.config.control_dt)))
        self.decoder = (
            make_decoder(
                decoder,
                spec=FieldSpec(
                    dim=2, channels=channels, design_mode=design_mode, material_mode=material_mode
                ),
                dtype=self.dtype,
            )
            if isinstance(decoder, str)
            else decoder
        )
        if self.decoder.spec.dim != 2:
            raise ValueError("Spring2D requires a 2D decoder")
        if np.dtype(self.decoder.dtype) != self.dtype:
            raise ValueError("Decoder and environment dtypes must match")
        self.channels = self.decoder.spec.channels
        self.max_action_space_size = int(max_action_space_size)
        self.max_observation_space_size = int(max_observation_space_size)
        if self.channels > self.max_action_space_size:
            raise ValueError("ACTION_CAPACITY_EXCEEDED")
        nx, ny = self.config.grid_shape
        self.observation_size = spring_observation_layout(
            (nx + 1) * (ny + 1),
            nx * ny,
            self.channels,
            terrain=self.config.terrain is not None,
        ).width
        if self.observation_size > self.max_observation_space_size:
            raise ValueError("OBSERVATION_CAPACITY_EXCEEDED")
        self.action_space = Box(-1.0, 1.0, (self.max_action_space_size,), self.dtype)
        self.observation_space = Box(
            -np.inf, np.inf, (self.max_observation_space_size,), self.dtype
        )
        self.embodiment_space = self.decoder.space()
        self._base = deepcopy(self.decoder.base())
        self._embodiment = deepcopy(self._base)
        with jax.default_device(self.device):
            self._topology = spring.topology(self.config, self.dtype)
        self._decode = jax.jit(self.decoder.decode)
        self._instantiate = jax.jit(
            lambda f: spring.instantiate(f, self._topology, self.config, self.channels)
        )
        self._advance, self._advance_batch, self._initialize = cast(
            tuple[AdvanceKernel, AdvanceKernel, InitializeKernel], spring.kernels(self.config)
        )
        if self.config is not self._initialization_config:
            self._initialize = cast(
                InitializeKernel, spring.kernels(self._initialization_config)[2]
            )
        self._observe, self._summary, self._reward = spring_reporting_kernels(
            self.config, self.max_observation_space_size
        )
        self._terrain = None
        self._terrain_seed = None
        if self.config.terrain is not None:
            from ..physics.terrain import generate

            with jax.default_device(self.device):
                self._terrain = generate(self.config.terrain, 0, 2, self.dtype)
        self._model = None
        self._state = None
        self._initial = None
        self._steps = 0
        self._done = False
        self._closed = False
        seed_sequence = np.random.SeedSequence()
        self.np_random = np.random.default_rng(seed_sequence)
        self._np_random_seed = int(cast(int, seed_sequence.entropy))

    @property
    def unwrapped(self) -> Spring2DEnv:
        """Return this environment for Gym-compatible wrapper inspection."""
        return self

    def _prepare(
        self, embodiment: object
    ) -> tuple[Embedding | None, PhysicsModel | None, list[str]]:
        """Validate a candidate and instantiate its device model without stepping."""
        errors = check_embedding(self.decoder, embodiment)
        if errors:
            return (None, None, errors)
        with jax.default_device(self.device):
            e = jax.device_put(
                cast_embedding(self.decoder, cast(Embedding, embodiment)), self.device
            )
            fields = self._decode(e, self._topology.points)
            fields = jax.block_until_ready(fields)
            errors = field_errors(fields, self.decoder.spec, int(np.prod(self.config.grid_shape)))
            if not errors:
                errors = topology_errors(fields.occupancy, self.config.grid_shape)
            if errors:
                return (None, None, errors)
            model = jax.block_until_ready(self._instantiate(fields))
            if self._terrain is not None:
                from .environment import spawn_shift

                shift = spawn_shift(self._terrain, self.config.size) - self.config.ground_height
                model = model._replace(
                    reference=model.reference.at[:, 1].add(shift), terrain=self._terrain
                )
        if not np.isfinite(float(jnp.sum(model.mass))) or float(jnp.sum(model.mass)) <= 0:
            return None, None, ["NONPOSITIVE_MASS"]
        return e, model, []

    def validation_report(self, embodiment: object) -> tuple[str, ...]:
        """Optional detailed query. validate_embodiment itself remains bool."""
        return tuple(self._prepare(embodiment)[2])

    def validate_embodiment(self, embodiment: object) -> bool:
        """Return whether the candidate passes field and topology checks."""
        return not self.validation_report(embodiment)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
        embodiment: Embedding | None = None,
    ) -> tuple[HostArray, HostArray, Info]:
        """Validate, settle, and initialize an episode.

        Returns (observation, rejection_mask[1], info). A rejected reset invalidates
        the previous state; callers must successfully reset before stepping.
        """
        if self._closed:
            raise RuntimeError("Environment is closed")
        if options not in (None, {}):
            raise ValueError(self._reset_options_error)
        if seed is not None:
            if not isinstance(seed, (int, np.integer)) or seed < 0:
                raise ValueError(self._reset_seed_error)
            self.np_random = np.random.default_rng(int(seed))
            self._np_random_seed = int(seed)
        self._prepare_reset_context()
        candidate = self._embodiment if embodiment is None else embodiment
        # Invalidate BEFORE validation; rejection cannot expose an old live state.
        self._state = None
        self._model = None
        self._initial = None
        self._done = False
        self._steps = 0
        start = time.perf_counter()
        _, model, errors = self._prepare(candidate)
        prepared = time.perf_counter()
        if not errors:
            model = cast(PhysicsModel, model)
            state = jax.block_until_ready(self._initialize(model))
            if int(state.status) != 0:
                errors = [self._settling_error(int(state.status))]
        if errors:
            info = self._reset_failure_fields(errors)
            info["timing"] = {
                "prepare_s": prepared - start,
                "total_reset_s": time.perf_counter() - start,
            }
            return (
                np.zeros(self.observation_space.shape, self.dtype),
                np.array([True], bool),
                info,
            )
        obs, info = self._complete_reset(
            candidate, cast(PhysicsModel, model), state, start, prepared
        )
        return obs, np.array([False], bool), info

    def _prepare_reset_context(self) -> None:
        """Choose spring terrain before invalidating state; flat legacy resets draw no seed."""
        if self.config.terrain is not None:
            from ..physics.terrain import generate

            self._terrain_seed = int(self.np_random.integers(0, 2**31))
            with jax.default_device(self.device):
                self._terrain = generate(self.config.terrain, self._terrain_seed, 2, self.dtype)

    def _settling_error(self, status: int) -> str:
        """Preserve the spring API's detailed failed-settling status name."""
        return "SETTLING_" + STATUS_NAMES[status]

    def _reset_failure_fields(self, errors: list[str]) -> Info:
        """Return the spring rejection schema before common timing fields are appended."""
        return {"embodiment_errors": tuple(errors), "valid": False}

    def _complete_reset(
        self,
        candidate: Embedding,
        model: PhysicsModel,
        state: DriverState,
        started: float,
        prepared: float,
    ) -> tuple[HostArray, Info]:
        """Install an accepted spring episode and report its original observation/timing schema.

        Candidate arrays become independently owned NumPy copies. The accepted
        state also serves as the immutable autoreset snapshot.
        """
        self._embodiment = {k: np.array(v, copy=True) for k, v in candidate.items()}
        self._model, self._state, self._initial = (model, state, state)
        obs, mask = self._observe(model, state)
        info = self._info(mask=mask)
        info["embodiment_errors"] = ()
        info["timing"] = {
            "prepare_s": prepared - started,
            "settle_and_observe_s": time.perf_counter() - prepared,
            "total_reset_s": time.perf_counter() - started,
        }
        return np.asarray(obs), info

    def _info(
        self,
        diag: TransitionDiagnostics | None = None,
        mask: Array | None = None,
        *,
        summary: ReportingSummary | None = None,
    ) -> Info:
        """Materialize committed-state metrics and fixed observation/action masks."""
        model = cast(spring.Model, self._model)
        state = cast(spring.State, self._state)
        if summary is None:
            summary = jax.device_get(self._summary(model, state))
        metrics, energy, mass, occupancy, active_nodes, active_cells = cast(SpringSummary, summary)
        status = int(metrics["status"])
        result: Info = {k: np.asarray(v) for k, v in metrics.items()}
        if mask is None:
            _, mask = self._observe(model, state)
        result.update(
            {
                "status_name": STATUS_NAMES[status],
                "observation_mask": np.asarray(mask),
                "action_mask": np.pad(
                    np.asarray(model.channel_mask),
                    (0, self.max_action_space_size - self.channels),
                ),
                "active_nodes": int(active_nodes),
                "active_cells": int(active_cells),
                "mass_kg": float(mass),
                "volume_m3": float(occupancy)
                * np.prod(self.config.size)
                * self.config.thickness
                / np.prod(self.config.grid_shape),
                "energy_components_j": np.asarray(energy),
                "completed": bool(self._steps >= self.config.horizon and status == 0),
                "design_mode": self.decoder.spec.design_mode,
                "integrator": self.config.integrator,
            }
        )
        if self._terrain is not None:
            from ..physics.terrain import serialize

            result["terrain"] = serialize(self._terrain)
            result["terrain_seed"] = self._terrain_seed
        if diag is not None:
            diag = cast(spring.Diagnostics, diag)
            result["solver"] = {k: float(v) for k, v in diag._asdict().items()}
            result["solver"]["result_name"] = spring.SOLVER_CODES[int(diag.solver_code)]
        return result

    def _check_action(self, action: ArrayLike) -> jax.Array:
        """Validate the padded host command and transfer native channels to device."""
        a = np.asarray(action)
        if a.shape != (self.max_action_space_size,):
            raise ValueError(
                f"ACTION_SHAPE: expected {(self.max_action_space_size,)}, got {a.shape}"
            )
        if a.dtype.kind not in "fiu" or not np.all(np.isfinite(a)):
            raise ValueError("NONFINITE_OR_INVALID_ACTION")
        if np.any(a < -1) or np.any(a > 1):
            raise ValueError("ACTION_OUT_OF_BOUNDS: actions are bounded once, not clipped silently")
        return jax.device_put(np.asarray(a[: self.channels], dtype=self.dtype), self.device)

    def _warn_update(self, embodiment: object) -> None:
        """Report unsupported transition-time morphology updates without applying them."""
        if embodiment is not None:
            warnings.warn(
                "Transition-level embodiment changes are unsupported; supplied embodiment is ignored.",
                UserWarning,
                stacklevel=3,
            )

    def step(
        self, action: ArrayLike, embodiment: Embedding | None = None
    ) -> tuple[HostArray, float, bool, bool, Info]:
        """Advance one held-action interval and return the Gym five-value transition.

        Actions are checked before mutation. Numerical failure terminates; the
        configured action-count horizon truncates successful episodes.
        """
        if self._closed:
            raise RuntimeError("Environment is closed")
        if self._state is None:
            raise ResetNeeded("A successful reset is required; rejected resets invalidate state")
        if self._done:
            raise ResetNeeded("Episode finished; call reset before another scalar step")
        self._warn_update(embodiment)
        a = self._check_action(action)
        before = self._state
        start = time.perf_counter()
        self._state, diag = jax.block_until_ready(
            self._advance(cast(PhysicsModel, self._model), before, a)
        )
        elapsed = time.perf_counter() - start
        self._steps += 1
        terminated = bool(int(self._state.status) != 0)
        truncated = bool(self._steps >= self.config.horizon and not terminated)
        self._done = terminated or truncated
        from ..batching import reporting_kernel

        report = reporting_kernel(self._observe, self._summary, self._reward)
        obs, mask, summary, reward = jax.device_get(
            report(cast(PhysicsModel, self._model), before, self._state)
        )
        info = self._info(jax.device_get(diag), mask, summary=summary)
        info["timing"] = {"synchronized_advance_s": elapsed}
        return np.asarray(obs), float(reward), terminated, truncated, info

    def render(self) -> HostArray | None:
        """Render an RGB host image after reset, or return None when disabled."""
        if self._state is None:
            raise ResetNeeded("Render requires successful reset")
        if self.render_mode is None:
            warnings.warn(
                "render_mode was not set; no rendering performed", UserWarning, stacklevel=2
            )
            return None
        from ..render import render_rgb

        return render_rgb(
            cast(spring.Model, self._model), cast(spring.State, self._state), self.config
        )

    def close(self) -> None:
        """Release references to the active model and state; subsequent steps fail."""
        self._state = None
        self._model = None
        self._initial = None
        self._closed = True


class Spring2DVectorEnv(VectorLifecycle, VectorEnv):
    """JAX vector environment; lane coordination is shared by all JAX drivers.

    Keeps the established constructor. The lifecycle mixin
    validates complete batches before stepping and owns next-step autoresets.
    """

    def __init__(
        self,
        num_envs: int,
        _env_factory: Callable[..., Spring2DEnv] = Spring2DEnv,
        **kwargs: object,
    ) -> None:
        self._initialize_vector(num_envs, _env_factory, **kwargs)
        self.integrator = self.envs[0].integrator


def make(env_id: str = "SoftSpring2D-v0", **kwargs: object) -> tuple[Spring2DEnv, HostEmbedding]:
    """Create one JAX environment and an independent baseline embedding."""
    from .. import registered_envs

    if env_id not in registered_envs():
        raise ValueError(f"Unknown environment {env_id!r}")
    if env_id == "SoftSpring2D-v0":
        # Constructor validates forwarded factory options.
        env = Spring2DEnv(**kwargs)  # type: ignore[arg-type]
    else:
        from .environment import BodyEnv

        # Body constructor owns option validation.
        env = BodyEnv(env_id, **kwargs)  # type: ignore[arg-type]
    return env, deepcopy(env._base)


def make_vec(
    env_id: str = "SoftSpring2D-v0", num_envs: int = 1, **kwargs: object
) -> tuple[Spring2DVectorEnv, HostEmbedding]:
    """Create shared-shape JAX lanes and a stacked baseline embedding."""
    from .. import registered_envs

    if env_id not in registered_envs():
        raise ValueError(f"Unknown environment {env_id!r}")
    mode = kwargs.pop("vectorization_mode", "jax")
    if mode != "jax":
        raise ValueError("M1 provides JAX-native vectorization_mode='jax' only")
    if env_id == "SoftSpring2D-v0":
        # Legacy factory forwards optional lane factory.
        env = Spring2DVectorEnv(num_envs, **kwargs)  # type: ignore[arg-type]
    else:
        from functools import partial

        from .environment import BodyEnv

        env = Spring2DVectorEnv(num_envs, _env_factory=partial(BodyEnv, env_id), **kwargs)
    base = {k: np.stack([e._base[k] for e in env.envs]) for k in env.envs[0]._base}
    return (env, base)
