"""Fixed co-design API for native Warp models.

A resolved embodiment is a dictionary of physical model arrays. Topology and
array capacities are fixed for an environment instance; values may change at
reset and are validated without repair. The explicit ``resolved`` observation
profile is backend-native, not a claim of JAX policy-feature ordering parity.
Use the one-time exporters to preserve existing decoded robot designs.

Interactive API calls necessarily have an observation boundary. Full open-loop
experiments use ``env.episode(commands, ...)``: all internal timesteps and the
whole implicit adjoint are inside one CUDA graph, never a Python step loop.
"""

from typing import TYPE_CHECKING, cast

from ...contracts.fields import resolved_action_mask
from ...contracts.metrics import LEDGER_WORK_POSITIVE
from ...contracts.observations import resolved_observation_layout

if TYPE_CHECKING:
    from typing import Unpack

    from numpy.typing import ArrayLike, NDArray

    from .host_types import (
        EpisodeOptions,
        EpisodeResult,
        FieldArrays,
        HostArray,
        MPMState,
        NativeFactoryOptions,
        NativeInfo,
        NativeOptions,
        ScalarReset,
        ScalarTransition,
        SolidState,
    )

import warnings
from copy import deepcopy
from dataclasses import replace

import numpy as np
import warp as wp

from ..action_timing import native_action_hold
from ..numerics.selection import resolve_integrator
from ..spaces import Box, Dict, Env, ResetNeeded
from .episode import Episode
from .model import Bundle as MPMBundle, DeviceModel, uniform_body
from .reset_state import ResetState
from .solid_builders import regular_solid
from .solid_episode import SolidEpisode
from .solid_model import STATE_FIELDS as SOLID_STATE, SolidBundle, SolidDeviceModel

_IDS = {
    "SoftSpring2D-v0": "spring",
    "SoftBeam3D-v0": "beam",
    "SoftHexFEM3D-v0": "fem",
    "SoftMPM2D-v0": "mpm2",
    "SoftMPM3D-v0": "mpm3",
}


def _selected_integrator(
    env_id: str, bundle: "MPMBundle | SolidBundle | None", integrator: str | None
) -> str:
    """Validate native integration support before allocating backend resources."""
    if env_id not in _IDS:
        raise ValueError(f"Unknown environment {env_id}")
    configured = bundle.config.integrator if isinstance(bundle, SolidBundle) else "backward_euler"
    return resolve_integrator(
        integrator, backend="warp", physics=_IDS[env_id], configured=configured
    )


_RESET_ERRORS = (ValueError, TypeError, IndexError, KeyError, OverflowError)

class NativeEnv(Env):
    """Native policy actions hold their command for fixed physics timesteps.

    ``physics_steps_per_action`` aliases ``substeps``; specifying both requires
    matching values. The horizon counts actions, and settling_steps continues
    to count raw physics steps independently of the action hold.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 25}

    def __init__(
        self,
        env_id: "str",
        *,
        bundle: "MPMBundle | SolidBundle | None" = None,
        integrator: "str | None" = None,
        dtype: "str" = "float64",
        device: "str" = "cpu",
        max_action_space_size: "int | None" = None,
        max_observation_space_size: "int | None" = None,
        is_eval_env: "bool" = False,
        substeps: "int | None" = None,
        settling_steps: "int" = 0,
        horizon: "int" = 250,
        length_ref: "float" = 0.2,
        work_ref: "float" = 1.0,
        energy_weight: "float" = 0.0,
        render_mode: "str" = "rgb_array",
        observation_profile: "str" = "resolved",
        physics_steps_per_action: "int | None" = None,
    ) -> "None":
        """Declare a scalar co-design environment without allocating its device model.

        Bundle precision and physics must match the selected ID. The bundle is
        copied; callers retain ownership of their input arrays. ``integrator=None``
        keeps the bundle's method; an explicit selection replaces only its copied
        configuration. Action and
        observation capacities are fixed here. Call reset before step/episode."""
        if env_id not in _IDS:
            raise ValueError(f"Unknown environment {env_id}")
        if observation_profile != "resolved":
            raise ValueError("Select the explicit native resolved-state observation profile")
        substeps = native_action_hold(substeps, physics_steps_per_action)
        for name, value, minimum in [
            ("substeps", substeps, 1),
            ("settling_steps", settling_steps, 0),
            ("horizon", horizon, 1),
        ]:
            if type(value) is not int or value < minimum:
                raise ValueError(f"Invalid {name}")
        if min(length_ref, work_ref) <= 0 or not np.all(
            np.isfinite([length_ref, work_ref, energy_weight])
        ):
            raise ValueError("Invalid metric reference scales")
        self.kind = _IDS[env_id]
        self.integrator = _selected_integrator(env_id, bundle, integrator)
        if bundle is None:
            if self.kind.startswith("mpm"):
                bundle = uniform_body(2 if self.kind == "mpm2" else 3, precision=dtype)
            else:
                shape = (1, 2, 1) if self.kind == "beam" else None
                bundle = regular_solid(self.kind, precision=dtype, shape=shape)
        expected = MPMBundle if self.kind.startswith("mpm") else SolidBundle
        if not isinstance(bundle, expected):
            raise ValueError("Bundle physical schema does not match the environment")
        if isinstance(bundle, SolidBundle) and bundle.config.physics != self.kind:
            raise ValueError("Bundle physics differs from environment ID")
        if bundle.config.dim != (2 if self.kind in ("spring", "mpm2") else 3):
            raise ValueError("Bundle dimension differs from environment ID")
        if integrator is not None and isinstance(bundle, SolidBundle):
            bundle = replace(bundle, config=replace(bundle.config, integrator=self.integrator))
        bundle.validate()
        precision = str(bundle.fields["reference"].dtype)
        if np.dtype(dtype) != np.dtype(precision):
            raise ValueError("Requested precision and bundle dtype must match; no implicit cast")
        self.template = deepcopy(bundle)
        self.dtype = np.dtype(dtype)
        self.device_name = device
        self.is_eval_env = bool(is_eval_env)
        self.substeps, self.settling_steps, self.horizon = substeps, settling_steps, horizon
        self.physics_steps_per_action = substeps
        self.physics_dt = bundle.config.dt
        self.control_dt = self.physics_dt * substeps
        self.length_ref, self.work_ref, self.energy_weight = length_ref, work_ref, energy_weight
        self.render_mode = render_mode
        self._base = deepcopy(bundle.fields)
        self._embodiment = deepcopy(self._base)
        self.channels = bundle.channels
        self.action_size = self.channels if max_action_space_size is None else max_action_space_size
        if type(self.action_size) is not int or self.action_size < self.channels:
            raise ValueError("Action capacity is too small")
        self.action_space = Box(-1.0, 1.0, shape=(self.action_size,), dtype=self.dtype)
        self.embodiment_space = Dict(
            {
                key: Box(
                    np.iinfo(v.dtype).min if v.dtype.kind in "iu" else -np.inf,
                    np.iinfo(v.dtype).max if v.dtype.kind in "iu" else np.inf,
                    shape=v.shape,
                    dtype=v.dtype.type,
                )
                for key, v in self._base.items()
            }
        )
        n, d = len(bundle.fields["reference"]), bundle.config.dim
        self.raw_observation_size = resolved_observation_layout(
            self.kind, d, n, self.channels, self.action_size
        ).width
        self.observation_size = (
            self.raw_observation_size
            if max_observation_space_size is None
            else max_observation_space_size
        )
        if (
            type(self.observation_size) is not int
            or self.observation_size < self.raw_observation_size
        ):
            raise ValueError(f"Observation capacity is too small: need {self.raw_observation_size}")
        self.observation_space = Box(
            -np.inf, np.inf, shape=(self.observation_size,), dtype=self.dtype
        )
        self.closed = False
        self.initialized = False
        self.owner: DeviceModel | SolidDeviceModel | None = None
        self._episode: Episode | SolidEpisode | None = None
        self._reset_state: ResetState | None = None
        self._steps = 0
        self._travel = 0.0
        self._last_work = 0.0
        self._work_origin = 0.0
        self.action_mask: HostArray = np.zeros(self.action_size, bool)
        self._mass: HostArray | None = None
        self._reference: HostArray | None = None

    def _candidate(self, embodiment: "FieldArrays") -> "MPMBundle | SolidBundle":
        """Copy a candidate's resolved fields and validate the unchanged topology."""
        if not self.embodiment_space.contains(embodiment):
            raise ValueError("Embodiment shape/dtype/finiteness differs from the declared space")
        return replace(
            self.template, fields={k: np.array(v, copy=True) for k, v in embodiment.items()}
        ).validate()

    def validate_embodiment(self, embodiment: "FieldArrays") -> "bool":
        """Return whether a resolved design satisfies this environment's schema."""
        try:
            self._candidate(embodiment)
            return True
        except (ValueError, TypeError, IndexError, KeyError, OverflowError):
            return False

    def _mask(self, bundle: "MPMBundle | SolidBundle") -> "HostArray":
        """Return a padded Boolean mask of channels used by active actuators."""
        return resolved_action_mask(bundle.fields, self.kind, self.channels, self.action_size)

    def reset(
        self,
        *,
        seed: "int | None" = None,
        options: "dict[str, object] | None" = None,
        embodiment: "FieldArrays | None" = None,
    ) -> "ScalarReset":
        """Install a valid design, settle it, and return observation/rejection/info.

        Observations and the one-element Boolean rejection flag are host arrays.
        A rejected design leaves the lane uninitialized; it cannot silently reuse
        a previous model. Scalar reset options are currently unsupported.
        ``reset_lanes`` runs the same prepare -> build -> settle -> finish steps
        for many lanes, settling them as one batch."""
        self._prepare_reset(seed, options)
        try:
            candidate, settling = self._build_reset(embodiment)
            return self._finish_reset(candidate, settling, None if settling is None else settling.run())
        except _RESET_ERRORS as exc:
            return self._reject_reset(exc)

    def _prepare_reset(self, seed: "int | None", options: "dict[str, object] | None") -> "None":
        """Check the lane and options, reseed, and clear the previous episode."""
        if self.closed:
            raise RuntimeError("Environment is closed")
        if options is not None and (not isinstance(options, dict) or options):
            raise ValueError("Native scalar reset does not support reset options")
        # Keep Gymnasium's per-environment RNG metadata in step with reset.
        # The explicit offline test base has no reset method.
        base_reset = getattr(super(), "reset", None)
        if base_reset is not None:
            base_reset(seed=None if seed is None else int(seed))
        self.initialized = False
        self._episode = None
        self._reset_state = None
        if seed is not None:
            self.action_space.seed(seed)

    def _build_reset(
        self, embodiment: "FieldArrays | None"
    ) -> "tuple[MPMBundle | SolidBundle, Episode | SolidEpisode | None]":
        """Build the device model; return the candidate and the (not yet run) settling episode, or None."""
        candidate = self._candidate(self._embodiment if embodiment is None else embodiment)
        self.owner = (
            DeviceModel(cast("MPMBundle", candidate), self.device_name)
            if self.kind.startswith("mpm")
            else SolidDeviceModel(cast("SolidBundle", candidate), self.device_name)
        )
        self.action_mask = self._mask(candidate)
        fields = candidate.fields
        self._mass = (
            fields["alpha"] * fields["density"] * fields["volume"]
            if self.kind.startswith("mpm")
            else fields["mass"]
        ).copy()
        self._reference = fields["reference"].copy()
        if not self.settling_steps:
            return candidate, None
        settling = self.episode(
            np.zeros((1, self.channels), self.dtype),
            substeps=self.settling_steps,
            settling_steps=0,
        )
        return candidate, settling

    def _finish_reset(
        self,
        candidate: "MPMBundle | SolidBundle",
        settling: "Episode | SolidEpisode | None",
        result: "EpisodeResult | None",
    ) -> "ScalarReset":
        """Accept a settled candidate: copy the settled state, bind the episode and capture the autoreset state."""
        if settling is not None:
            if cast("EpisodeResult", result)["status"] != 0:
                raise ValueError(f"Initial settling failed with status {cast('EpisodeResult', result)['status']}")
            self._copy_final(settling)
        self._episode = self.episode(
            np.zeros((1, self.channels), self.dtype), substeps=self.substeps, settling_steps=0
        )
        owner = cast("DeviceModel | SolidDeviceModel", self.owner)
        self._steps, self._travel = 0, 0.0
        self._last_work = float(owner.initial.ledger.numpy()[LEDGER_WORK_POSITIVE])
        self._work_origin = self._last_work
        self.initialized = True
        observation = self._observation(owner.initial)
        self._embodiment = deepcopy(candidate.fields)
        self._reset_state = ResetState.capture(owner.initial, self._state_fields())
        return observation, np.asarray([False], dtype=bool), self._info()

    def _reject_reset(self, exc: "BaseException") -> "ScalarReset":
        """Leave the lane uninitialized and report the rejection reason."""
        self.owner = None
        return (
            np.zeros(self.observation_size, self.dtype),
            np.asarray([True], dtype=bool),
            {"valid": False, "embodiment_errors": [str(exc)]},
        )

    def _autoreset(self) -> "ScalarReset":
        """Restore the exact settled state accepted by the last explicit reset.

        Re-running CUDA settling can change last-bit reductions even for identical
        fields. A next-step autoreset reuses that accepted state, including its
        ledger, while explicit resets still rebuild and settle the requested
        design. Device model and compiled episode ownership stay unchanged.
        """
        if self.closed or self.owner is None or self._episode is None or self._reset_state is None:
            raise ResetNeeded("Autoreset requires a previously accepted initialized design")
        self._reset_state.restore(self.owner.initial)
        self._steps, self._travel = 0, 0.0
        self._last_work = float(self.owner.initial.ledger.numpy()[LEDGER_WORK_POSITIVE])
        self._work_origin = self._last_work
        self.initialized = True
        return self._observation(self.owner.initial), np.asarray([False], bool), self._info()

    def episode(
        self, commands: "ArrayLike", **kwargs: "Unpack[EpisodeOptions]"
    ) -> "Episode | SolidEpisode":
        """Bind a bounded command table to the current device model.

        Commands have shape (control_intervals, channels). The returned episode
        retains device buffers and cached graphs; creation requires a valid reset.
        Use its run method for a whole resident forward or gradient evaluation."""
        if self.owner is None:
            raise ResetNeeded("Reset a valid embodiment before binding an episode")
        options: EpisodeOptions = dict(
            substeps=self.substeps,
            settling_steps=0,
            length_ref=self.length_ref,
            work_weight=self.energy_weight / self.work_ref,
        )
        options.update(kwargs)
        if self.kind.startswith("mpm"):
            return Episode(cast("DeviceModel", self.owner), commands, **options)
        return SolidEpisode(cast("SolidDeviceModel", self.owner), commands, **options)

    def _copy_final(self, episode: "Episode | SolidEpisode") -> "None":
        """Copy the accepted episode's final buffers into this lane's initial state."""
        owner = cast("DeviceModel | SolidDeviceModel", self.owner)
        for name in self._state_fields():
            wp.copy(getattr(owner.initial, name), getattr(episode.final, name))

    def _state_fields(self) -> "tuple[str, ...]":
        """List every physical state buffer for this lane's model family."""
        return (
            ("x", "v", "F", "C", "actuator", "ledger", "status")
            if self.kind.startswith("mpm")
            else SOLID_STATE
        )

    def _command(self, action: "ArrayLike") -> "HostArray":
        """Validate without mutation so a whole environment batch is atomic."""
        if not self.initialized or self.closed:
            raise ResetNeeded("Reset a valid embodiment before stepping")
        if self._steps >= self.horizon:
            raise ResetNeeded("Episode reached its horizon")
        action = np.asarray(action)
        if (
            action.shape != (self.action_size,)
            or action.dtype.kind not in "fiu"
            or not np.all(np.isfinite(action))
            or np.any(action < -1)
            or np.any(action > 1)
        ):
            raise ValueError("Action must be finite, bounded, and match padded capacity")
        return np.where(self.action_mask[: self.channels], action[: self.channels], 0.0).astype(
            self.dtype
        )

    def step(
        self, action: "ArrayLike", embodiment: "FieldArrays | None" = None
    ) -> "ScalarTransition":
        """Hold one bounded padded action and return a five-item host transition.

        The reward uses displacement and positive-work increments. Numerical
        failure terminates the lane; reaching the action horizon truncates it.
        Design changes supplied during a transition are warned about and ignored."""
        command = self._command(action)
        if embodiment is not None:
            warnings.warn(
                "Transition-level embodiment updates are unsupported and ignored",
                UserWarning,
                stacklevel=2,
            )
        episode = cast("Episode | SolidEpisode", self._episode)
        episode.set_commands(command[None, :])
        result = episode.run()
        return self._finish_step(result)

    def _finish_step(self, result: "EpisodeResult") -> "ScalarTransition":
        """Collect a submitted solve after its GPU work has completed."""
        self._steps += 1
        episode = cast("Episode | SolidEpisode", self._episode)
        state = episode.final
        failed = result["status"] != 0
        if not failed:
            travel = episode.travel.numpy()
            mass = cast("HostArray", self._mass)
            increment = float(np.sum(mass * travel[:, 0]) / mass.sum())
            self._travel += increment
            work = float(state.ledger.numpy()[LEDGER_WORK_POSITIVE])
            reward = (
                increment / self.length_ref
                - self.energy_weight * (work - self._last_work) / self.work_ref
            )
            self._last_work = work
        else:
            reward = 0.0
            self.initialized = False
        self._copy_final(episode)
        info = self._info()
        info["numerics"] = episode.step.diagnostics()
        info["execution_seconds"] = result["execution_seconds"]
        for key in ("execution_mode", "batch_execution_seconds", "batch_size"):
            if key in result:
                info[key] = result[key]
        info["valid"] = not failed
        return (
            self._observation(state),
            float(reward),
            bool(failed),
            bool(self._steps >= self.horizon and not failed),
            info,
        )

    def _observation(self, state: "MPMState | SolidState") -> "HostArray":
        """Download resolved state, action mask and ledger into a padded flat array."""
        if self.kind.startswith("mpm"):
            particle_state = cast("MPMState", state)
            values = [
                particle_state.x.numpy() - self._reference,
                state.v.numpy(),
                particle_state.F.numpy(),
                particle_state.C.numpy(),
            ]
        else:
            solid_state = cast("SolidState", state)
            values = [
                solid_state.u.numpy(),
                state.v.numpy(),
                solid_state.rotation.numpy(),
                solid_state.omega.numpy(),
            ]
        values.extend(
            [state.actuator.numpy(), state.ledger.numpy(), self.action_mask.astype(self.dtype)]
        )
        raw = np.concatenate([v.ravel() for v in values]).astype(self.dtype)
        return np.pad(raw, (0, self.observation_size - len(raw)))

    def _info(self) -> "NativeInfo":
        """Return copied masks, cumulative physical metrics and scaled objectives."""
        return dict(
            action_mask=self.action_mask.copy(),
            observation_mask=np.arange(self.observation_size) < self.raw_observation_size,
            observation_profile="native resolved state; see docs/WARP_ALL_PHYSICS.md",
            metrics={
                "displacement": self._travel,
                "positive_work": self._last_work - self._work_origin,
            },
            objectives=np.asarray(
                [
                    -self._travel / self.length_ref,
                    (self._last_work - self._work_origin) / self.work_ref,
                ],
                self.dtype,
            ),
        )

    def render(self) -> "NDArray[np.uint8]":
        """Download the current state and render a diagnostic uint8 RGB frame."""
        if not self.initialized:
            raise ResetNeeded("A valid initialized state is required to render")
        from .solid_visualization import render_state

        owner = cast("DeviceModel | SolidDeviceModel", self.owner)
        reference = owner.model.reference.numpy()
        positions = (
            cast("MPMState", owner.initial).x.numpy()
            if self.kind.startswith("mpm")
            else reference + cast("SolidState", owner.initial).u.numpy()
        )
        edges = None if self.kind.startswith("mpm") else owner.model.entities.numpy()
        return render_state(positions, reference, edges, label=self.kind)

    def close(self) -> "None":
        """Release owned model/episode references and make this lane unsteppable."""
        self.closed = True
        self.initialized = False
        self._episode = None
        self._reset_state = None
        self.owner = None
        self._mass = None
        self._reference = None


def make(
    env_id: "str" = "SoftSpring2D-v0", **kwargs: "Unpack[NativeFactoryOptions]"
) -> "tuple[NativeEnv, FieldArrays]":
    """Return a native scalar environment and an independent resolved base design."""
    kwargs.pop("backend", None)
    env = NativeEnv(env_id, **cast("NativeOptions", kwargs))
    return env, deepcopy(env._base)
