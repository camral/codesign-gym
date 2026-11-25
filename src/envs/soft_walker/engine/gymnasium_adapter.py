"""Standard Gymnasium vector contract over the existing batched backends.

The co-design factories intentionally use a three-value reset. This adapter
changes that public contract while preserving one native batched physics call;
it never constructs a SyncVectorEnv/AsyncVectorEnv around scalar simulations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from numpy.typing import ArrayLike

from .types import Embedding, HostArray, HostEmbedding, Info

if TYPE_CHECKING:
    from .drivers.spring_environment import Spring2DVectorEnv
    from .warp_native.vector_env import NativeVectorEnv

from copy import deepcopy

import numpy as np
from gymnasium.vector import AutoresetMode, VectorEnv


class GymnasiumVectorAdapter(VectorEnv):
    """Wrap a co-design vector environment for ordinary Gymnasium consumers.

    ``reset(options={'embodiment': design})`` supplies a new design. Invalid
    designs raise ValueError, leaving rejected lanes invalid until explicitly
    reset. Additional options, including reset_mask, are delegated to the
    backend, which may explicitly reject unsupported options.

    Both backends expose NumPy arrays at this boundary. JAX transformations
    belong around pure numerical kernels/rollouts, not Gymnasium step calls.
    """

    def __init__(
        self, inner: Spring2DVectorEnv | NativeVectorEnv, base_embodiment: HostEmbedding
    ) -> None:
        """Expose a co-design vector environment through the standard Gymnasium contract.

        The adapter owns no physics buffers; base_embodiment is copied for safe inspection.
        """
        self.inner = inner
        self.base_embodiment = deepcopy(base_embodiment)
        self.num_envs = inner.num_envs
        self.single_action_space = inner.single_action_space
        self.single_observation_space = inner.single_observation_space
        self.action_space = inner.action_space
        self.observation_space = inner.observation_space
        self.embodiment_space = inner.embodiment_space
        self.physics_steps_per_action = inner.physics_steps_per_action
        self.physics_dt = inner.physics_dt
        self.control_dt = inner.control_dt
        self.integrator = inner.integrator
        self.metadata = dict(
            getattr(inner, "metadata", {}), autoreset_mode=AutoresetMode.NEXT_STEP, jax=False
        )
        self.render_mode = getattr(
            inner, "render_mode", getattr(inner.envs[0], "render_mode", None)
        )
        self.closed = False

    def _info(self, info: Info) -> Info:
        """Normalize native per-lane reports into Gymnasium masked vector info."""
        if "per_env" not in info:
            return dict(info)
        # Warp's co-design API retains a list of scalar dictionaries. Convert
        # with Gymnasium's own vector info convention (including _key masks).
        result: Info = {}
        for lane, per_lane in enumerate(info["per_env"]):
            result = self._add_info(result, per_lane, lane)
        return result

    def reset(
        self,
        *,
        seed: int | list[int | None] | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[HostArray, Info]:
        """Reset lanes using standard Gymnasium arity; reject invalid designs explicitly."""
        if self.closed:
            raise RuntimeError("Environment is closed")
        if options is not None and not isinstance(options, dict):
            raise ValueError("reset options must be a dictionary or None")
        options = {} if options is None else dict(options)
        candidate = cast(
            Embedding | list[Embedding | None] | tuple[Embedding | None, ...] | None,
            options.pop("embodiment", None),
        )
        obs, rejected, info = self.inner.reset(
            seed=seed,
            options=options or None,
            # Backend validates supported shared/per-lane layouts.
            embodiment=candidate,  # type: ignore[arg-type]
        )
        rejected = np.asarray(rejected, dtype=bool)
        if np.any(rejected):
            lanes = np.flatnonzero(rejected).tolist()
            raise ValueError(f"Embodiment/reset rejected in lanes {lanes}: {info}")
        return obs, self._info(cast(Info, info))

    def step(self, actions: ArrayLike) -> tuple[HostArray, HostArray, HostArray, HostArray, Info]:
        """Delegate one batched physics transition and normalize its info layout."""
        if self.closed:
            raise RuntimeError("Environment is closed")
        obs, rewards, terminated, truncated, info = self.inner.step(actions)
        return obs, rewards, terminated, truncated, self._info(cast(Info, info))

    def get_attr(self, name: str) -> tuple[object, ...]:
        """Read scalar lane attributes in stable order."""
        if hasattr(self.inner, "get_attr"):
            return self.inner.get_attr(name)
        return tuple(getattr(env, name) for env in self.inner.envs)

    @property
    def np_random_seed(self) -> tuple[int, ...]:
        """Return the scalar environment seed for each lane."""
        return cast(tuple[int, ...], self.get_attr("np_random_seed"))

    @property
    def np_random(self) -> tuple[np.random.Generator, ...]:
        """Return each lane random generator without advancing it."""
        return cast(tuple[np.random.Generator, ...], self.get_attr("np_random"))

    def set_attr(self, name: str, values: object) -> None:
        """Broadcast an attribute or assign one value per lane."""
        if hasattr(self.inner, "set_attr"):
            return cast(None, self.inner.set_attr(name, values))
        if not isinstance(values, (tuple, list)):
            values = [values] * self.num_envs
        if len(values) != self.num_envs:
            raise ValueError("Attribute batch length mismatch")
        for env, value in zip(self.inner.envs, values):
            setattr(env, name, value)

    def call(self, name: str, *args: object, **kwargs: object) -> tuple[object, ...]:
        """Call a named method on each lane with the supplied arguments."""
        if hasattr(self.inner, "call"):
            return self.inner.call(name, *args, **kwargs)
        return tuple(getattr(env, name)(*args, **kwargs) for env in self.inner.envs)

    def render(self) -> tuple[HostArray | None, ...]:
        """Return each configured scalar lane image."""
        if hasattr(self.inner, "render"):
            return self.inner.render()
        return tuple(env.render() for env in self.inner.envs)

    def close(self, **kwargs: object) -> None:
        """Close the owned vector environment once."""
        if not getattr(self, "closed", True):
            self.inner.close()
            self.closed = True


def make_gymnasium_vec(
    env_id: str = "SoftSpring2D-v0", num_envs: int = 1, **kwargs: object
) -> GymnasiumVectorAdapter:
    """Return a standard vector env backed by JAX vmap or native Warp graphs."""
    from . import make_vec

    # Backend-specific options are validated by the selected factory.
    inner, base = make_vec(env_id, num_envs=num_envs, **kwargs)  # type: ignore[call-overload]
    try:
        return GymnasiumVectorAdapter(inner, base)
    except BaseException:
        inner.close()
        raise
