"""Host-side vector lifecycle around one compiled JAX batch transition.

Scalar environments own their physics state. This layer coordinates validation,
seeding, lane resets, and Gymnasium info masks; array stacking stays inside the
existing compiled batching kernel.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, cast

from numpy.typing import ArrayLike

from ..spaces import Space
from ..types import DriverState, Embedding, HostArray, Info, PhysicsModel

if TYPE_CHECKING:
    from .spring_environment import Spring2DEnv
import time
from copy import deepcopy

import jax
import numpy as np

from ..spaces import AUTORESET_NEXT_STEP, Box, Dict, ResetNeeded


def _stack_info(infos: Sequence[Info]) -> Info:
    """Use Gymnasium's nested vector-info convention without requiring Gym.

    Missing keys have false masks; a present value of None still has a true
    mask. Nested dictionaries remain dictionaries even when only one lane
    supplies them, as happens to solver diagnostics on mixed autoreset calls.
    """
    out: Info = {}
    count = len(infos)

    def add(target: Info, info: Info, lane: int) -> None:
        """Insert one lane recursively, preserving missing-key masks and object-valued terminal
        observations.
        """
        for key, value in info.items():
            if key == "final_obs":
                array = target.get(key)
                if array is None:
                    array = np.full(count, None, dtype=object)
                array[lane] = value
            elif isinstance(value, dict):
                array = target.setdefault(key, {})
                add(array, value, lane)
            else:
                if key in target:
                    array = target[key]
                elif type(value) in (int, float, bool) or isinstance(value, np.number):
                    array = np.zeros(count, dtype=type(value))
                elif isinstance(value, np.ndarray):
                    array = np.zeros((count, *value.shape), dtype=value.dtype)
                else:
                    array = np.full(count, None, dtype=object)
                array[lane] = value
            mask = target.setdefault("_" + key, np.zeros(count, dtype=bool))
            mask[lane] = True
            target[key] = array

    for lane, info in enumerate(infos):
        add(out, info, lane)
    return out


def batch_space(space: Space, n: int) -> Space:
    """Prepend the lane dimension while preserving leaf bounds and dtypes."""
    if hasattr(space, "spaces"):
        return Dict({k: batch_space(s, n) for k, s in space.spaces.items()})
    return Box(
        np.broadcast_to(space.low, (n,) + space.shape).copy(),
        np.broadcast_to(space.high, (n,) + space.shape).copy(),
        dtype=space.dtype,
    )


class VectorLifecycle:
    """JAX-vmapped physics with Gymnasium next-step autoreset semantics.

    Validation/reset orchestration is per lane. Any rejected, non-reset lane
    prevents a subsequent batch step. Actions on next-step autoreset lanes are
    ignored. terminal results remain attached to the terminating transition.
    """

    envs: list[Spring2DEnv]

    def _initialize_vector(
        self, num_envs: int, _env_factory: Callable[..., Spring2DEnv], **kwargs: object
    ) -> None:
        """Construct independent lane state with shared-shape batched API spaces."""
        if type(num_envs) is not int or num_envs < 1:
            raise ValueError("num_envs must be positive")
        self.num_envs = num_envs
        self.envs = [_env_factory(**kwargs) for _ in range(num_envs)]
        e = self.envs[0]
        self.single_action_space = e.action_space
        self.single_observation_space = e.observation_space
        self.single_embodiment_space = e.embodiment_space
        self.action_space = batch_space(e.action_space, num_envs)
        self.observation_space = batch_space(e.observation_space, num_envs)
        self.embodiment_space = batch_space(e.embodiment_space, num_envs)
        self.metadata = dict(e.metadata, autoreset_mode=AUTORESET_NEXT_STEP)
        self.is_eval_env = e.is_eval_env
        self.render_mode = e.render_mode
        self.physics_steps_per_action = e.physics_steps_per_action
        self.physics_dt = e.physics_dt
        self.control_dt = e.control_dt
        self._closed = False
        self.closed = False

    @property
    def unwrapped(self) -> VectorLifecycle:
        """Return the vector environment rather than one of its scalar lanes."""
        return self

    @property
    def np_random_seed(self) -> tuple[int, ...]:
        """Return each scalar lane seed in stable lane order."""
        return tuple(getattr(e, "np_random_seed", e._np_random_seed) for e in self.envs)

    @property
    def np_random(self) -> tuple[np.random.Generator, ...]:
        """Return each scalar lane random generator without copying its state."""
        return tuple(e.np_random for e in self.envs)

    def get_attr(self, name: str) -> tuple[object, ...]:
        """Read a named attribute from every lane in order."""
        return tuple(getattr(e, name) for e in self.envs)

    def set_attr(self, name: str, values: object) -> None:
        """Broadcast one value or assign exactly one value per lane."""
        if not isinstance(values, (tuple, list)):
            values = [values] * self.num_envs
        if len(values) != self.num_envs:
            raise ValueError("Attribute batch length mismatch")
        for e, value in zip(self.envs, values):
            setattr(e, name, value)

    def call(self, name: str, *args: object, **kwargs: object) -> tuple[object, ...]:
        """Call one named lane method and return results in lane order."""
        return tuple(getattr(e, name)(*args, **kwargs) for e in self.envs)

    def _candidates(
        self, embodiment: Embedding | list[Embedding | None] | tuple[Embedding | None, ...] | None
    ) -> list[Embedding | None]:
        """Normalize shared or batched embeddings, retaining legacy upgrade behavior."""
        if embodiment is None:
            return [None] * self.num_envs
        if isinstance(embodiment, (list, tuple)):
            if len(embodiment) != self.num_envs:
                raise ValueError("Embodiment batch length mismatch")
            return list(embodiment)
        if not isinstance(embodiment, dict):
            raise TypeError("Embodiment must be dict or per-lane sequence")
        expected = set(self.single_embodiment_space.spaces)
        decoder = self.envs[0].decoder
        if hasattr(decoder, "upgrade_embedding"):
            original_keys = set(embodiment)
            present = original_keys & expected
            was_batch = bool(present) and all(
                np.shape(embodiment[k])
                == (self.num_envs,) + self.single_embodiment_space.spaces[k].shape
                for k in present
            )
            embodiment = cast(Embedding, decoder.upgrade_embedding(embodiment))
            if was_batch:
                embodiment = dict(embodiment)
                for k in (set(embodiment) - original_keys) & expected:
                    shape = self.single_embodiment_space.spaces[k].shape
                    if np.shape(embodiment[k]) == shape:
                        embodiment[k] = np.broadcast_to(
                            embodiment[k], (self.num_envs,) + shape
                        ).copy()
        legacy_override = hasattr(self.envs[0], "kind") and set(embodiment) == expected | {
            "poisson"
        }
        if set(embodiment) != expected and not legacy_override:
            return [embodiment] * self.num_envs
        is_batch = all(
            np.shape(embodiment[k]) == (self.num_envs,) + s.shape
            for k, s in self.single_embodiment_space.spaces.items()
        )
        if legacy_override:
            is_batch = is_batch and np.shape(embodiment["poisson"]) == (self.num_envs,)
        return (
            [{k: np.asarray(v)[i] for k, v in embodiment.items()} for i in range(self.num_envs)]
            if is_batch
            else [deepcopy(embodiment) for _ in range(self.num_envs)]
        )

    def validate_embodiment(
        self, embodiment: Embedding | list[Embedding | None] | tuple[Embedding | None, ...] | None
    ) -> bool:
        """Validate all candidate lanes without installing new models."""
        return all(
            e.validate_embodiment(b) for e, b in zip(self.envs, self._candidates(embodiment))
        )

    def reset(
        self,
        *,
        seed: int | Sequence[int | None] | None = None,
        options: dict[str, object] | None = None,
        embodiment: Embedding | list[Embedding | None] | tuple[Embedding | None, ...] | None = None,
    ) -> tuple[HostArray, HostArray, Info]:
        """Reset selected lanes and retain the states of unselected lanes.

        Returns observations, rejected-lane flags, and Gym-style masked info.
        The complete seed sequence is validated before any lane changes state
        or consumes randomness, including entries for unselected lanes.
        """
        if self._closed:
            raise RuntimeError("Environment is closed")
        mask: HostArray = np.ones(self.num_envs, bool)
        if options:
            if set(options) != {"reset_mask"}:
                raise ValueError("Only reset_mask is supported")
            mask = np.asarray(options["reset_mask"])
            if mask.shape != (self.num_envs,) or mask.dtype != np.bool_:
                raise ValueError("reset_mask must be a boolean lane vector")
        seeds: list[int | None]
        if seed is None:
            seeds = [None] * self.num_envs
        elif isinstance(seed, (int, np.integer)):
            seeds = [int(seed) + i for i in range(self.num_envs)]
        else:
            seeds = list(seed)
            if len(seeds) != self.num_envs:
                raise ValueError("seed batch length mismatch")
        # Match scalar reset's accepted seeds and driver-specific error text,
        # but reject the whole request before an earlier lane can be reset.
        for lane, value in zip(self.envs, seeds):
            if value is not None and (not isinstance(value, (int, np.integer)) or value < 0):
                raise ValueError(lane._reset_seed_error)
        candidates = self._candidates(embodiment)
        obs = []
        rejected = []
        infos = []
        for i, e in enumerate(self.envs):
            if mask[i]:
                o, r, info = e.reset(seed=seeds[i], embodiment=candidates[i])
            elif e._state is None:
                o = np.zeros(e.observation_space.shape, e.dtype)
                r = np.array([True])
                info = {"valid": False, "embodiment_errors": ("LANE_UNINITIALIZED",)}
            else:
                o, m = e._observe(cast(PhysicsModel, e._model), e._state)
                o = np.asarray(o)
                r = np.array([False])
                info = e._info(mask=m)
            obs.append(o)
            rejected.append(bool(r[0]))
            infos.append(info)
        return np.stack(obs), np.asarray(rejected, bool), _stack_info(infos)

    def step(
        self,
        actions: ArrayLike,
        embodiment: Embedding | list[Embedding | None] | tuple[Embedding | None, ...] | None = None,
    ) -> tuple[HostArray, HostArray, HostArray, HostArray, Info]:
        """Validate all commands, run one compiled batch, then handle lane reports.

        Done lanes reset on the next call and ignore their supplied action. No
        physics runs when every lane is an autoreset lane.
        """
        if self._closed:
            raise RuntimeError("Environment is closed")
        if any(e._state is None for e in self.envs):
            raise ResetNeeded("At least one lane has no valid reset; batch step refused")
        if embodiment is not None:
            self.envs[0]._warn_update(embodiment)

        first = self.envs[0]
        actions = np.asarray(actions)
        if actions.shape != self.action_space.shape:
            raise ValueError("BATCH_ACTION_SHAPE")
        # Validate the entire batch before changing ANY lane. One transfer,
        # rather than a separate _check_action/device_put call for each lane.
        if actions.dtype.kind not in "fiu" or not np.all(np.isfinite(actions)):
            raise ValueError("NONFINITE_OR_INVALID_ACTION")
        if np.any(actions < -1) or np.any(actions > 1):
            raise ValueError("ACTION_OUT_OF_BOUNDS: actions are not clipped silently")
        controls = jax.device_put(
            np.asarray(actions[:, : first.channels], dtype=first.dtype), first.device
        )

        auto: HostArray = np.asarray([e._done for e in self.envs], dtype=bool)
        reset_results = {}
        for i, env in enumerate(self.envs):
            if auto[i]:
                obs, rejected, info = env.reset()
                if rejected[0]:
                    raise ResetNeeded("Previously accepted embodiment failed autoreset")
                info["autoreset"] = True
                reset_results[i] = (obs, 0.0, False, False, info)

        # No dummy simulation when every lane is an autoreset lane.
        if np.all(auto):
            outputs = [reset_results[i] for i in range(self.num_envs)]
        else:
            from ..batching import vector_step_kernel

            kernel = vector_step_kernel(
                first._advance, first._observe, first._summary, first._reward
            )
            models = tuple(cast(PhysicsModel, e._model) for e in self.envs)
            states = tuple(cast(DriverState, e._state) for e in self.envs)
            start = time.perf_counter()
            next_states, device_payload = jax.block_until_ready(kernel(models, states, controls))
            elapsed = time.perf_counter() - start
            (observations, masks, summaries, rewards), diagnostics = jax.device_get(device_payload)

            outputs = []
            for i, env in enumerate(self.envs):
                if auto[i]:
                    # Mixed batches retain the original safe-dummy convention;
                    # the new state and diagnostics for this lane are discarded.
                    outputs.append(reset_results[i])
                    continue
                env._state = next_states[i]
                env._steps += 1
                summary = jax.tree.map(lambda x: x[i], summaries)
                diag = jax.tree.map(lambda x: x[i], diagnostics)
                terminated = int(summary[0]["status"]) != 0
                truncated = env._steps >= env.config.horizon and (not terminated)
                env._done = terminated or truncated
                info = env._info(diag, masks[i], summary=summary)
                info["autoreset"] = False
                # This interval includes compiled reporting, not just physics.
                # Use a new key rather than mislabeling it as advance-only time.
                info["timing"] = {"batch_synchronized_pipeline_s": elapsed}
                outputs.append((observations[i], float(rewards[i]), terminated, truncated, info))

        obs, rewards, termination_flags, truncation_flags, infos = zip(*outputs)
        return (
            np.stack(obs),
            np.asarray(rewards),
            np.asarray(termination_flags, dtype=bool),
            np.asarray(truncation_flags, dtype=bool),
            _stack_info(infos),
        )

    def render(self) -> tuple[HostArray | None, ...]:
        """Render each lane using its configured scalar renderer."""
        return tuple(e.render() for e in self.envs)

    def close(self) -> None:
        """Close every scalar lane and mark the vector environment closed."""
        for e in self.envs:
            e.close()
        self._closed = True
        self.closed = True
