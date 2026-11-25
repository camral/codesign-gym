"""Vector coordination and next-step autoreset for native Warp lanes.

Scalar lifecycle is implemented in ``env.NativeEnv``. This module owns batch
validation, lane scheduling and autoreset only; each scalar lane retains its
model, implicit solver, command buffers and captured graph.
"""

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from typing import Unpack

    from numpy.typing import ArrayLike

    from .batching import EpisodeBatchExecutor
    from .env import NativeEnv
    from .episode import Episode
    from .host_types import (
        FieldArrays,
        HostArray,
        NativeFactoryOptions,
        NativeOptions,
        NativeVectorOptions,
        ScalarReset,
        ScalarTransition,
        VectorReset,
        VectorTransition,
    )
    from .solid_episode import SolidEpisode

import warnings
from copy import deepcopy

import numpy as np
import warp as wp

from ..spaces import AUTORESET_NEXT_STEP, Box, Dict, ResetNeeded


def step_lanes(
    lanes: "Iterable[NativeEnv]", actions: "ArrayLike", *, executor: "EpisodeBatchExecutor"
) -> "list[ScalarTransition]":
    """Advance independent native lanes, submitting the whole batch first.

    The executor controls serial versus concurrent CUDA graph execution. All
    actions and lifecycle states are checked before uploading any command or
    advancing any lane. Each lane retains its own implicit solve and status.
    """
    lanes = list(lanes)
    if not lanes:
        return []
    if len({id(lane) for lane in lanes}) != len(lanes):
        raise ValueError("A batch must contain distinct environment instances")
    actions = np.asarray(actions)
    if actions.ndim != 2 or actions.shape[0] != len(lanes):
        raise ValueError("Actions must have shape (num_envs, action_capacity)")
    commands = [lane._command(action) for lane, action in zip(lanes, actions)]
    episodes = [cast("Episode | SolidEpisode", lane._episode) for lane in lanes]
    # Upload every command before launching any graph: pageable host copies
    # must not accidentally synchronize another lane's in-flight solve.
    for episode, command in zip(episodes, commands):
        episode.set_commands(command[None, :])
    results = executor.run(episodes)
    return [lane._finish_step(result) for lane, result in zip(lanes, results)]



def reset_lanes(
    lanes: "Iterable[NativeEnv]",
    embodiments: "Sequence[FieldArrays | None]",
    seeds: "Sequence[int | None]",
    *,
    executor: "EpisodeBatchExecutor",
) -> "list[ScalarReset]":
    """Reset independent native lanes, running every lane's settling episode as one batch.

    Each lane builds its model and settling episode first; the executor then settles them serially or concurrently.
    Per lane the result is exactly ``lane.reset(seed=..., embodiment=...)``: a rejected design or failed settling
    rejects only that lane.
    """
    from .env import _RESET_ERRORS

    lanes = list(lanes)
    if len({id(lane) for lane in lanes}) != len(lanes):
        raise ValueError("A batch must contain distinct environment instances")
    if not len(lanes) == len(embodiments) == len(seeds):
        raise ValueError("One embodiment and one seed are required per lane")
    for lane, seed in zip(lanes, seeds):
        lane._prepare_reset(seed, None)
    built: list = []
    for lane, embodiment in zip(lanes, embodiments):
        try:
            built.append(lane._build_reset(embodiment))
        except _RESET_ERRORS as exc:
            built.append(exc)
    settling = [i for i, item in enumerate(built) if not isinstance(item, BaseException) and item[1] is not None]
    results = dict(zip(settling, executor.run([built[i][1] for i in settling]))) if settling else {}
    out = []
    for i, (lane, item) in enumerate(zip(lanes, built)):
        if isinstance(item, BaseException):
            out.append(lane._reject_reset(item))
            continue
        try:
            out.append(lane._finish_reset(item[0], item[1], results.get(i)))
        except _RESET_ERRORS as exc:
            out.append(lane._reject_reset(exc))
    return out

class NativeVectorEnv:
    """Independent environments with concurrent CUDA graph execution.

    ``execution='auto'`` overlaps CUDA lanes on separate streams and uses the
    serial correctness path on CPU. ``serial`` provides an explicit baseline;
    ``parallel`` requires CUDA. This is not a fused multi-robot sparse solver.

    Completed lanes reset on the next call, returning zero reward on that call.
    Rejected user resets are different: they leave a lane invalid until a valid
    explicit reset and must not silently recover an earlier embodiment.
    """

    def __init__(
        self,
        env_id: "str",
        num_envs: "int",
        *,
        execution: "str" = "auto",
        **kwargs: "Unpack[NativeOptions]",
    ) -> "None":
        """Construct independent scalar lanes and one persistent batch executor.

        CUDA auto mode assigns a stream per lane; CPU uses serial execution.
        Capacities and dtypes are shared, while physical model values may differ."""
        from .env import NativeEnv, _selected_integrator

        if type(num_envs) is not int or num_envs < 1:
            raise ValueError("num_envs must be positive")
        if execution not in ("auto", "serial", "parallel"):
            raise ValueError("execution must be auto, serial, or parallel")
        self.integrator = _selected_integrator(
            env_id, kwargs.get("bundle"), kwargs.get("integrator")
        )
        device = kwargs.get("device", "cpu")
        if execution == "parallel" and not wp.get_device(device).is_cuda:
            raise ValueError("Parallel native environment execution requires CUDA")
        from .batching import EpisodeBatchExecutor

        self.execution = execution
        mode = "serial" if execution == "auto" and str(device) == "cpu" else execution
        self._executor = EpisodeBatchExecutor(execution=mode)
        self.envs = [NativeEnv(env_id, **kwargs) for _ in range(num_envs)]
        self.num_envs = num_envs
        self.metadata = dict(self.envs[0].metadata, autoreset_mode=AUTORESET_NEXT_STEP)
        self.render_mode = self.envs[0].render_mode
        self.physics_steps_per_action = self.envs[0].physics_steps_per_action
        self.physics_dt = self.envs[0].physics_dt
        self.control_dt = self.envs[0].control_dt
        self.single_action_space = self.envs[0].action_space
        self.single_observation_space = self.envs[0].observation_space
        self.action_space = Box(
            -1.0, 1.0, shape=(num_envs, self.envs[0].action_size), dtype=self.envs[0].dtype
        )
        self.observation_space = Box(
            -np.inf,
            np.inf,
            shape=(num_envs, self.envs[0].observation_size),
            dtype=self.envs[0].dtype,
        )
        self.embodiment_space = Dict(
            {
                key: Box(
                    np.stack([value.low] * num_envs),
                    np.stack([value.high] * num_envs),
                    dtype=value.dtype.type,
                )
                for key, value in self.envs[0].embodiment_space.spaces.items()
            }
        )
        self._next_reset: HostArray = np.zeros(num_envs, dtype=bool)
        self._embodiments = [deepcopy(env._base) for env in self.envs]

    def reset(
        self,
        *,
        seed: "int | Sequence[int | None] | None" = None,
        embodiment: "FieldArrays | None" = None,
        options: "dict[str, object] | None" = None,
    ) -> "VectorReset":
        """Reset selected lanes and return host observations in original lane order.

        ``options['reset_mask']`` is a Boolean array of shape ``(num_envs,)``;
        omitting it selects every lane. Unselected lanes keep their device
        owners, graphs, random generators, accepted designs and pending
        autoresets. Reading their observation does not advance physics. A lane
        without an installed state reports a rejected reset and zero observation.

        Integer seeds expand to ``seed + lane_index``, including skipped indices;
        seed sequences have one entry per lane. Only selected lanes are reseeded.
        Mask, seed and candidate indexing errors are checked before any reset.
        A rejected selected design clears its pending autoreset and requires an
        explicit valid reset before the batch can step again.
        """
        if any(env.closed for env in self.envs):
            raise RuntimeError("Environment is closed")
        if options is not None and not isinstance(options, dict):
            raise ValueError("reset options must be a dictionary or None")
        mask: HostArray = np.ones(self.num_envs, dtype=bool)
        if options:
            if set(options) != {"reset_mask"}:
                raise ValueError("Native vector reset options support only reset_mask")
            mask = np.asarray(options["reset_mask"])
            if mask.shape != (self.num_envs,) or mask.dtype != np.bool_:
                raise ValueError("reset_mask must be a boolean lane vector")
        if seed is None or isinstance(seed, (int, np.integer)):
            seeds = [None if seed is None else int(seed) + i for i in range(self.num_envs)]
        else:
            seeds = list(seed)
            if len(seeds) != self.num_envs:
                raise ValueError("Seed sequence length must match num_envs")
        if any(
            value is not None and (not isinstance(value, (int, np.integer)) or value < 0)
            for value in seeds
        ):
            raise ValueError("Seeds must be nonnegative integers or None")
        candidates = [
            None
            if embodiment is None or not mask[i]
            else {key: value[i] for key, value in embodiment.items()}
            for i in range(self.num_envs)
        ]
        results: list[ScalarReset] = []
        # Selected rejected designs must not recover an old terminal body;
        # skipped terminal lanes must still autoreset on the next step.
        self._next_reset[mask] = False
        for i, env in enumerate(self.envs):
            if mask[i]:
                result = env.reset(seed=seeds[i], embodiment=candidates[i])
                if not bool(result[1][0]):
                    self._embodiments[i] = deepcopy(env._embodiment)
            elif env.owner is None:
                result = (
                    np.zeros(env.observation_size, env.dtype),
                    np.asarray([True], dtype=bool),
                    {"valid": False, "embodiment_errors": ["LANE_UNINITIALIZED"]},
                )
            else:
                result = (
                    env._observation(env.owner.initial),
                    np.asarray([False], dtype=bool),
                    env._info(),
                )
            results.append(result)
        return (
            np.stack([r[0] for r in results]),
            np.asarray([bool(r[1][0]) for r in results], dtype=bool),
            {"per_env": [r[2] for r in results]},
        )

    def step(
        self, actions: "ArrayLike", embodiment: "FieldArrays | None" = None
    ) -> "VectorTransition":
        """Validate all actions before advancing active lanes and resetting done lanes.

        Returns stacked observations, rewards, terminated/truncated flags and
        per-lane info. A next-step autoreset produces zero reward for that lane.
        Rejected explicit resets require another valid explicit reset."""
        actions = np.asarray(actions)
        if actions.shape != self.action_space.shape:
            raise ValueError("Incorrect vector action shape")
        if (
            actions.dtype.kind not in "fiu"
            or not np.all(np.isfinite(actions))
            or np.any(actions < -1)
            or np.any(actions > 1)
        ):
            raise ValueError("Invalid vector actions")
        # Check every lane before advancing any; a numerically terminated lane
        # can autoreset, whereas a rejected explicit reset must raise.
        if any(
            e.closed
            or (not e.initialized and not self._next_reset[i])
            or (e._steps >= e.horizon and not self._next_reset[i])
            for i, e in enumerate(self.envs)
        ):
            raise ResetNeeded("Every lane requires a valid explicit reset or a pending autoreset")
        if embodiment is not None:
            warnings.warn(
                "Transition-level embodiment updates are unsupported and ignored",
                UserWarning,
                stacklevel=2,
            )
        results: list[ScalarTransition | None] = [None] * self.num_envs
        active = []
        for i, env in enumerate(self.envs):
            if self._next_reset[i]:
                observation, rejected, info = env._autoreset()
                self._next_reset[i] = False
                if bool(rejected[0]):
                    raise ResetNeeded(f"Autoreset rejected lane {i}: {info}")
                info["autoreset"] = True
                results[i] = (observation, 0.0, False, False, info)
            else:
                active.append(i)
        if active:
            advanced = step_lanes(
                [self.envs[i] for i in active], actions[active], executor=self._executor
            )
            for i, result in zip(active, advanced):
                self._next_reset[i] = result[2] or result[3]
                result[4]["autoreset"] = False
                results[i] = result
        completed = cast("list[ScalarTransition]", results)
        return (
            np.stack([r[0] for r in completed]),
            np.asarray([r[1] for r in completed]),
            np.asarray([r[2] for r in completed]),
            np.asarray([r[3] for r in completed]),
            {"per_env": [r[4] for r in completed]},
        )

    def validate_embodiment(self, embodiment: "FieldArrays") -> "bool":
        """Check a field dictionary whose leading array dimension indexes lanes."""
        try:
            return all(
                env.validate_embodiment({key: value[i] for key, value in embodiment.items()})
                for i, env in enumerate(self.envs)
            )
        except (TypeError, IndexError, KeyError):
            return False

    def close(self) -> "None":
        """Join owned CUDA streams, then release every lane's model and episode."""
        self._executor.close()
        for env in self.envs:
            env.close()


def make_vec(
    env_id: "str" = "SoftSpring2D-v0", num_envs: "int" = 1, **kwargs: "Unpack[NativeFactoryOptions]"
) -> "tuple[NativeVectorEnv, FieldArrays]":
    """Return native vector lanes and stacked independent copies of their designs."""
    kwargs.pop("backend", None)
    mode = kwargs.pop("vectorization_mode", "warp")
    if mode != "warp":
        raise ValueError("Native vectorization_mode must be warp")
    env = NativeVectorEnv(env_id, num_envs, **cast("NativeVectorOptions", kwargs))
    base = {k: np.stack([e._base[k] for e in env.envs]) for k in env.envs[0]._base}
    return env, base
