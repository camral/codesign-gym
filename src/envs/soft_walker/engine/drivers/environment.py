"""Concrete M3 JAX drivers: FEM3D, MPM2D/3D, and a beam3D lattice.

Your make/reset/step contract is the only required entry point. The numerical
functions below are internal and shared by the ordinary and differentiable paths.
Full observations expose F/C in MPM and R/omega in beams. Compact observations
are explicitly partially observed. These are development profiles, not frozen
accuracy-certified benchmark settings. General self-contact is unsupported.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from copy import deepcopy
from dataclasses import asdict
from functools import lru_cache
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import DTypeLike

from ...contracts.fields import POISSON_BOUNDS
from ...contracts.observations import body_observation_layout
from ...contracts.status_codes import BINARY_STATE_FAILED, BINARY_STATE_OK
from .. import topology_checks
from ..action_timing import with_action_hold
from ..device import resolve_device, resolve_dtype
from ..embodiments.decoders import (
    Decoder,
    Fields,
    FieldSpec,
    cast_embedding,
    check_embedding,
    make_decoder,
)
from ..embodiments.fields import field_errors
from ..numerics.selection import resolve_integrator
from ..physics import beam, fem, mpm, terrain as terrains
from ..physics.common import StepInfo, advance_retry, combine_info as combine_info
from ..spaces import Box, Dict, ResetNeeded
from ..types import (
    Array,
    BodyDomain,
    BodyKernels,
    BodyModel,
    BodyPhysicsKernels,
    BodySummary,
    DriverState,
    Embedding,
    HostArray,
    Info,
    PhysicsModel,
    ReportingSummary,
    RolloutResult,
    TransitionDiagnostics,
)
from .model_access import (
    channel_mask as channel_mask,
    module as module,
    positions as positions,
    reference as reference,
)
from .reporting import (
    _mpm_energy as _mpm_energy,
    mechanics_diagnostics as mechanics_diagnostics,
    observe as observe,
)
from .spring_environment import STATUS_NAMES as STATUS_NAMES, Spring2DEnv
from .task import BodyConfig, TaskState, mean, metrics, reward

IDS = {
    "SoftHexFEM3D-v0": "fem",
    "SoftMPM2D-v0": "mpm2",
    "SoftMPM3D-v0": "mpm3",
    "SoftBeam3D-v0": "beam",
}


def spawn_shift(terrain: terrains.Terrain, size: tuple[float, ...]) -> jax.Array:
    """Return a fixed terrain support height for the complete design envelope."""
    # Highest terrain support over the fixed design envelope, NOT a mass/shape
    # normalization. Every candidate gets the same scenario spawn transform.

    dim = len(size)
    corners = []
    for x in (0.0, size[0]):
        for z in (-0.5 * size[2], 0.5 * size[2]) if dim == 3 else (0.0,):
            corners.append(
                (
                    terrain.offset
                    - terrain.normal[0] * x
                    - (terrain.normal[2] * z if dim == 3 else 0.0)
                )
                / terrain.normal[1]
            )
    highest = jnp.max(jnp.stack(corners))
    lo = terrain.centers - terrain.half_sizes
    hi = terrain.centers + terrain.half_sizes
    overlap = (lo[:, 0] <= size[0]) & (hi[:, 0] >= 0) & terrain.active
    if dim == 3:
        overlap = overlap & (lo[:, 2] <= size[2] / 2) & (hi[:, 2] >= -size[2] / 2)
    return jnp.maximum(highest, jnp.max(jnp.where(overlap, hi[:, 1], -jnp.inf), initial=-jnp.inf))


def build_model(
    kind: str,
    domain: BodyDomain,
    fields: Fields,
    channels: int,
    poisson: jax.Array | None,
    terrain: terrains.Terrain,
    cfg: BodyConfig,
) -> BodyModel:
    """Place the reference domain on terrain and instantiate decoded material fields."""
    ref = domain.reference.at[:, 1].add(spawn_shift(terrain, cfg.size))
    domain = domain._replace(reference=ref)
    model = module(kind).instantiate(domain, fields, channels, poisson=poisson)
    return model._replace(terrain=terrain)


def trial(
    kind: str,
    model: BodyModel,
    state: TaskState,
    command: jax.Array,
    dt: jax.Array,
    cfg: BodyConfig,
    grid: mpm.Grid | None,
) -> tuple[TaskState, StepInfo]:
    """Attempt one physical substep and accumulate accepted task increments only."""
    if cfg.dynamics.actuation_model == "active_stress" and not kind.startswith("mpm"):
        raise ValueError("Active stress is currently supported by MPM drivers only")
    body = state.body._replace(status=state.status)
    if kind.startswith("mpm"):
        new, info = mpm.trial(
            cast(mpm.Model, model),
            cast(mpm.State, body),
            command,
            dt,
            cast(mpm.Grid, grid),
            cfg.dynamics,
        )
    else:
        new, info = module(kind).trial(model, body, command, dt, cfg.dynamics)
    ok = new.status == BINARY_STATE_OK
    # Accumulate mathematical integrator increments, not differences of rounded
    # absolute COM coordinates. Retry halves each accumulate their own increment.
    displacement = state.displacement + dt * mean(model, new.v)
    effort = state.effort + dt * jnp.sum((command * channel_mask(model)) ** 2)
    return TaskState(
        new,
        jnp.where(ok, displacement, state.displacement),
        jnp.where(ok, effort, state.effort),
        state.initial_com,
        new.status,
    ), info


def advance(
    kind: str,
    model: BodyModel,
    state: TaskState,
    command: jax.Array,
    cfg: BodyConfig,
    grid: mpm.Grid | None,
) -> tuple[TaskState, tuple[StepInfo, jax.Array]]:
    """Scan one control interval; any failed substep rolls the interval back."""

    def one(s: TaskState, _: None) -> tuple[TaskState, tuple[StepInfo, jax.Array]]:
        """Apply the configured retry policy to one nominal physics substep."""
        f = lambda s, a, h: trial(kind, model, s, a, h, cfg, grid)
        new, diag, trace = advance_retry(
            f, s, command, jnp.asarray(cfg.dt, s.v.dtype), cfg.dynamics
        )
        return new, (diag, trace)

    result, (diags, traces) = jax.lax.scan(jax.checkpoint(one), state, None, length=cfg.substeps)
    # A control interval is a transaction: unsuccessful intervals do not accrue
    # time, work, displacement or partial physical state.
    ok = result.status == BINARY_STATE_OK
    result = jax.tree.map(lambda n, o: jnp.where(ok, n, o), result, state)._replace(
        status=jnp.where(ok, BINARY_STATE_OK, BINARY_STATE_FAILED).astype(jnp.int32)
    )
    agg = StepInfo(
        jnp.sum(diags.accepted),
        jnp.sum(diags.rejected),
        jnp.sum(diags.nonlinear_iterations),
        jnp.sum(diags.linear_solves),
        jnp.max(diags.max_residual),
        jnp.max(diags.max_linear_residual),
        jnp.sum(diags.line_search_backtracks),
        jnp.min(diags.min_J),
        jnp.min(diags.support_valid),
        jnp.max(diags.max_penetration),
        jnp.max(diags.code),
    )
    return result, (agg, jnp.where(ok, traces, 0.0))


def initialize(kind: str, model: BodyModel, cfg: BodyConfig, grid: mpm.Grid | None) -> TaskState:
    """Settle the body, then reset objective ledgers while retaining its settled state."""
    body = module(kind).initial_state(model)
    z = jnp.asarray(0.0, body.v.dtype)
    state = TaskState(
        body,
        jnp.zeros(body.v.shape[-1], body.v.dtype),
        z,
        mean(model, positions(model, body, kind)),
        body.status,
    )

    def settle(s: TaskState, _: None) -> tuple[TaskState, None]:
        """Advance one settling interval with the model resting command."""
        return advance(kind, model, s, model.channels, cfg, grid)[0], None

    if cfg.settle_steps:
        state, _ = jax.lax.scan(settle, state, None, length=cfg.settle_steps)
    body = state.body._replace(work_signed=z, work_positive=z, dissipated=z, time=z)
    return state._replace(
        body=body,
        displacement=jnp.zeros_like(state.displacement),
        effort=z,
        initial_com=mean(model, positions(model, body, kind)),
    )


@lru_cache(maxsize=32)
def physics_kernels(kind: str, cfg: BodyConfig, grid: mpm.Grid | None) -> BodyPhysicsKernels:
    """Cache compiled transitions and reporting by immutable physics configuration."""
    return (
        jax.jit(lambda m, s, a: advance(kind, m, s, a, cfg, grid)),
        jax.jit(jax.vmap(lambda m, s, a: advance(kind, m, s, a, cfg, grid))),
        jax.jit(lambda m: initialize(kind, m, cfg, grid)),
        jax.jit(
            lambda m, s: (
                metrics(s, cfg),
                module(kind).energy_budget(m, s.body, cfg.dynamics)
                if not kind.startswith("mpm")
                else _mpm_energy(m, s.body, cfg),
                mechanics_diagnostics(kind, m, s, cfg),
            )
        ),
        jax.jit(lambda m, b, a: reward(b, a, cfg)),
    )


@lru_cache(maxsize=48)
def kernels(kind: str, cfg: BodyConfig, grid: mpm.Grid | None, max_size: int) -> BodyKernels:
    """Add fixed-size observation reporting to the shared physics kernels."""
    step, batch, init, summary, rew = physics_kernels(kind, cfg, grid)
    return step, batch, init, jax.jit(lambda m, s: observe(kind, m, s, cfg, max_size)), summary, rew


@lru_cache(maxsize=32)
def instantiation_kernel(
    kind: str, cfg: BodyConfig, channels: int
) -> Callable[[BodyDomain, Fields, jax.Array | None, terrains.Terrain], BodyModel]:
    """Cache the traceable fields-to-model assembly for one driver configuration."""
    return jax.jit(lambda domain, f, nu, t: build_model(kind, domain, f, channels, nu, t, cfg))


class BodyEnv(Spring2DEnv):
    """Scalar co-design environment for FEM, MPM, and beam task dynamics."""

    # BodyEnv retains the historical Spring2DEnv lifecycle inheritance, while
    # its immutable numerical configuration belongs to the body drivers.
    config: BodyConfig  # type: ignore[assignment]
    _initialization_config: BodyConfig  # type: ignore[assignment]
    _domain: BodyDomain
    _grid: mpm.Grid | None
    _occupancy_shape: tuple[int, ...]
    _cached_info: Info | None
    _terrain: terrains.Terrain

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50, "jax": True}

    def __init__(
        self,
        env_id: str,
        *,
        is_eval_env: bool = False,
        max_action_space_size: int = 4,
        max_observation_space_size: int = 2048,
        backend: str = "jax",
        dtype: DTypeLike = "float32",
        device: str | jax.Device | None = None,
        decoder: str | Decoder = "direct",
        design_mode: str = "hard",
        material_mode: str = "continuous",
        channels: int = 2,
        config: BodyConfig | None = None,
        render_mode: str | None = None,
        physics_steps_per_action: int | None = None,
        integrator: str | None = None,
    ) -> None:
        """Bind a body driver, decoder, and device without creating an episode.

        Invalid dimensions, precision, action capacity, or observation capacity raise
        ValueError.
        Call reset to validate an embodiment and allocate its live model/state.
        """
        if backend != "jax":
            raise NotImplementedError(
                "BodyEnv implements JAX; use engine.make(..., backend='warp') for native Warp"
            )
        self.kind = IDS[env_id]
        self.env_id = env_id
        self.integrator = resolve_integrator(integrator, backend=backend, physics=self.kind)
        dim = 2 if self.kind == "mpm2" else 3
        self.dtype = resolve_dtype(dtype)
        self.device = resolve_device(device)
        self._initialization_config = config or (
            BodyConfig(grid_shape=(2, 1), size=(0.1, 0.05)) if dim == 2 else BodyConfig()
        )
        cfg = self._initialization_config
        if not isinstance(cfg, BodyConfig) or len(cfg.grid_shape) != dim:
            raise ValueError("BodyConfig dimension mismatch")
        self.config = cfg = with_action_hold(cfg, physics_steps_per_action)
        self.physics_steps_per_action = cfg.substeps
        self.physics_dt = cfg.dt
        self.control_dt = cfg.control_dt
        if cfg.dynamics.actuation_model == "active_stress" and not self.kind.startswith("mpm"):
            raise ValueError("Active stress is currently supported by MPM drivers only")
        if self.kind.startswith("mpm") or self.kind == "beam":
            if not np.allclose(
                np.asarray(cfg.size) / cfg.grid_shape, cfg.size[0] / cfg.grid_shape[0]
            ):
                raise ValueError("MPM/beam reference cells must be uniform cubes/squares")
        self.is_eval_env = bool(is_eval_env)
        self.supports_transition_embodiment = False
        self.render_mode = render_mode
        if render_mode not in (None, "rgb_array"):
            raise ValueError("render_mode must be None or rgb_array")
        self.metadata = dict(type(self).metadata, render_fps=int(round(1 / cfg.control_dt)))
        self.decoder = (
            make_decoder(
                decoder,
                spec=FieldSpec(
                    dim=dim, channels=channels, design_mode=design_mode, material_mode=material_mode
                ),
                dtype=self.dtype,
            )
            if isinstance(decoder, str)
            else decoder
        )
        if self.decoder.spec.dim != dim or np.dtype(self.decoder.dtype) != self.dtype:
            raise ValueError("decoder dimension/dtype mismatch")
        self.channels = self.decoder.spec.channels
        self.max_action_space_size = int(max_action_space_size)
        self.max_observation_space_size = int(max_observation_space_size)
        if self.channels > self.max_action_space_size:
            raise ValueError("ACTION_CAPACITY_EXCEEDED")
        origin = (0.0, cfg.initial_clearance) + ((-0.5 * cfg.size[2],) if dim == 3 else ())
        with jax.default_device(self.device):
            if self.kind == "fem":
                self._domain = fem.structured_mesh(cfg.grid_shape, cfg.size, origin, self.dtype)
                self._grid = None
                self._occupancy_shape = cfg.grid_shape
            elif self.kind == "beam":
                self._domain = beam.topology(cfg.grid_shape, cfg.size, origin, self.dtype)
                self._grid = None
                self._occupancy_shape = cfg.grid_shape
            else:
                h = cfg.size[0] / cfg.grid_shape[0]
                self._domain = mpm.uniform_cloud(
                    cfg.grid_shape, h, origin, cfg.particles_per_axis, cfg.thickness, self.dtype
                )
                margin = cfg.grid_margin + 1
                padding = cfg.grid_padding or ((margin, margin),) * dim
                grid_shape = tuple(n + lo + hi + 1 for n, (lo, hi) in zip(cfg.grid_shape, padding))
                grid_origin = tuple(x - lo * h for x, (lo, _) in zip(origin, padding))
                self._grid = mpm.Grid(grid_shape, h, grid_origin)
                self._occupancy_shape = tuple(n * cfg.particles_per_axis for n in cfg.grid_shape)
            self._terrain = terrains.generate(cfg.terrain, 0, dim, self.dtype)
        self.action_space = Box(-1.0, 1.0, (self.max_action_space_size,), self.dtype)
        if "poisson" in self.decoder.space().spaces:
            raise ValueError("RESERVED_EMBODIMENT_KEY: poisson")
        # New designs carry local Poisson values in their decoded material
        # fields. An explicitly supplied legacy scalar remains an override.
        self.embodiment_space = Dict(dict(self.decoder.space().spaces))
        self._base = deepcopy(self.decoder.base())
        self._embodiment = deepcopy(self._base)
        self._decode = jax.jit(self.decoder.decode)
        self._instantiate = lambda f, nu, t: instantiation_kernel(self.kind, cfg, self.channels)(
            self._domain, f, nu, t
        )
        n = self._domain.reference.shape[0]
        nf = self._domain.points.shape[0]
        self.observation_size = body_observation_layout(
            self.kind, dim, n, nf, self.channels, profile=cfg.observation_profile
        ).width
        if self.observation_size > self.max_observation_space_size:
            raise ValueError(
                f"OBSERVATION_CAPACITY_EXCEEDED: {self.observation_size}>{self.max_observation_space_size}"
            )
        self.observation_space = Box(
            -np.inf, np.inf, (self.max_observation_space_size,), self.dtype
        )
        (
            self._advance,
            self._advance_batch,
            self._initialize,
            self._observe,
            self._summary,
            self._reward,
        ) = kernels(self.kind, cfg, self._grid, self.max_observation_space_size)
        if cfg is not self._initialization_config:
            self._initialize = physics_kernels(self.kind, self._initialization_config, self._grid)[
                2
            ]
        self._model = None
        self._state = None
        self._initial = None
        self._steps = 0
        self._done = False
        self._closed = False
        sequence = np.random.SeedSequence()
        self.np_random = np.random.default_rng(sequence)
        self._np_random_seed = int(cast(int, sequence.entropy))
        self._terrain_seed = 0
        self._cached_info = None

    def _prepare(self, embodiment: object) -> tuple[Embedding | None, BodyModel | None, list[str]]:
        """Validate a candidate, build its model, and reject unsupported initial grid support."""
        if not isinstance(embodiment, dict):
            return None, None, ["EMBEDDING_KEYS"]
        base = {k: v for k, v in embodiment.items() if k != "poisson"}
        errors = check_embedding(self.decoder, base)
        nu = None
        if "poisson" in embodiment:
            try:
                nu = np.asarray(embodiment["poisson"])
                nu_ok = (
                    nu.shape == ()
                    and nu.dtype.kind in "iuf"
                    and bool(np.isfinite(nu))
                    and POISSON_BOUNDS[0] <= float(nu) <= POISSON_BOUNDS[1]
                )
            except (ValueError, TypeError, OverflowError):
                nu_ok = False
            if not nu_ok:
                errors.append("POISSON_OUT_OF_BOUNDS")
        if errors:
            return (None, None, errors)
        with jax.default_device(self.device):
            e = jax.device_put(cast_embedding(self.decoder, base), self.device)
            f = jax.block_until_ready(self._decode(e, self._domain.points))
            errors = field_errors(f, self.decoder.spec, len(self._domain.points))
            if not errors:
                errors = topology_checks.errors(f.occupancy, self._occupancy_shape)
            if errors:
                return (None, None, errors)
            model = jax.block_until_ready(
                self._instantiate(
                    f, None if nu is None else jnp.asarray(nu, self.dtype), self._terrain
                )
            )
            if self.kind.startswith("mpm") and not bool(
                jnp.all(
                    mpm.stencil(
                        cast(mpm.Model, model).cloud.reference, cast(mpm.Grid, self._grid)
                    ).valid
                    | (~model.active)
                )
            ):
                return None, None, ["INITIAL_GRID_SUPPORT_INVALID"]
        if nu is not None:
            e = dict(e, poisson=jax.device_put(np.asarray(nu, self.dtype), self.device))
        return e, model, []

    _reset_options_error = "No scalar reset options are defined"
    _reset_seed_error = "seed must be nonnegative"

    def _prepare_reset_context(self) -> None:
        """Draw the body driver's terrain seed and discard cached geometry reports."""
        self._terrain_seed = int(self.np_random.integers(0, 2**31))
        with jax.default_device(self.device):
            self._terrain = terrains.generate(
                self.config.terrain, self._terrain_seed, len(self.config.size), self.dtype
            )
        self._cached_info = None

    def _settling_error(self, status: int) -> str:
        """Keep the body-driver rejection code independent of solver status detail."""
        return "SETTLING_FAILED"

    def _reset_failure_fields(self, errors: list[str]) -> Info:
        """Include the selected terrain seed in rejected body reset reports."""
        return {
            "valid": False,
            "embodiment_errors": tuple(errors),
            "terrain_seed": self._terrain_seed,
        }

    def _complete_reset(
        self,
        candidate: Embedding,
        model: PhysicsModel,
        state: DriverState,
        started: float,
        prepared: float,
    ) -> tuple[HostArray, Info]:
        """Install an accepted body episode, copying the embedding and synchronizing observations."""
        self._model, self._state, self._initial = (model, state, state)
        self._embodiment = deepcopy(candidate)
        obs, mask = jax.block_until_ready(self._observe(model, state))
        info = self._info(mask=mask)
        info["embodiment_errors"] = ()
        info["timing"] = {
            "prepare_s": prepared - started,
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
        """Convert committed metrics and mechanics diagnostics into host API reports."""
        model = cast(BodyModel, self._model)
        state = cast(TaskState, self._state)
        if summary is None:
            summary = jax.device_get(self._summary(model, state))
        m, en, mechanics = cast(BodySummary, summary)
        if mask is None:
            _, mask = self._observe(model, state)
        result: Info = {k: np.asarray(v) for k, v in m.items()}
        if self._cached_info is None:
            alpha, mass, active, action_mask = jax.device_get(
                (model.alpha, model.mass, model.active, channel_mask(model))
            )
            volume = (
                float(np.sum(alpha * np.asarray(cast(mpm.Model, model).cloud.volume)))
                if self.kind.startswith("mpm")
                else float(
                    np.sum(alpha) * np.prod(self.config.size) / np.prod(self.config.grid_shape)
                )
            )
            self._cached_info = {
                "action_mask": np.pad(action_mask, (0, self.max_action_space_size - self.channels)),
                "active_sites": int(np.sum(active)),
                "mass_kg": float(np.sum(mass)),
                "volume_m3": volume,
                "design_mode": self.decoder.spec.design_mode,
                "observation_profile": self.config.observation_profile,
                "integrator": "backward_euler",
                "terrain_seed": self._terrain_seed,
                "terrain": terrains.serialize(cast(terrains.Terrain, self._terrain)),
                "self_contact": "unsupported; only terrain contact is modeled",
                "model": self.kind,
                "metric_schema": "positive_mechanical_work_v1; endpoint increment quadrature",
                "energy_component_names": [
                    "translational_kinetic" if self.kind.startswith("mpm") else "kinetic",
                    "elastic",
                    "gravitational",
                    "contact",
                ],
                "energy_warning": "MPM kinetic diagnostic omits APIC affine moment; not total discrete APIC energy"
                if self.kind.startswith("mpm")
                else None,
                "grid": asdict(self._grid) if self._grid is not None else None,
            }
        result.update(deepcopy(self._cached_info))
        result.update(
            {
                "status_name": "OK" if int(m["status"]) == 0 else "NUMERICAL_FAILURE",
                "observation_mask": np.asarray(mask),
                "energy_components_j": np.asarray(en),
                "completed": bool(self._steps >= self.config.horizon and int(m["status"]) == 0),
                "mechanics": {k: np.asarray(v) for k, v in mechanics.items()},
                "mechanics_scope": "committed state, not rejected trial",
                "model_revision": "m3-development-0.3",
            }
        )
        if diag is not None:
            aggregate, trace = cast(tuple[StepInfo, jax.Array], diag)
            result["solver"] = {k: float(v) for k, v in aggregate._asdict().items()}
            result["accepted_dt_trace_s"] = np.asarray(trace)
        return result

    def render(self) -> HostArray | None:
        """Render the active body after a successful reset."""
        if self._state is None:
            raise ResetNeeded("Render requires successful reset")
        if self.render_mode is None:
            return super().render()
        from ..render import render_body_rgb

        return render_body_rgb(self)

    def bind_rollout(
        self, **kwargs: object
    ) -> Callable[[Embedding, jax.Array], RolloutResult[None]]:
        """Bind a pure differentiable rollout to this environment and its terrain."""
        from ..body_rollout import bind

        # Optional rollout flags are checked by the binder.
        return bind(self, **kwargs)  # type: ignore[arg-type]
