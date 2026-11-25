"""Soft co-design environments. Importing this module does not initialize JAX.

The factory and reset arities intentionally follow the supplied co-design API,
not stock Gymnasium. Precision is a process setting: set JAX_ENABLE_X64 before
using float64. The package never changes it globally on your behalf.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, overload

from .numerics.selection import available_integrators as available_integrators

if TYPE_CHECKING:
    from .drivers.spring_environment import Spring2DEnv, Spring2DVectorEnv
    from .gymnasium_adapter import GymnasiumVectorAdapter
    from .types import HostEmbedding
    from .warp_native.env import NativeEnv
    from .warp_native.vector_env import NativeVectorEnv

__version__ = "0.3.0"


@overload
def make(
    env_id: str = "SoftSpring2D-v0", *, backend: Literal["warp"], **kwargs: object
) -> tuple[NativeEnv, HostEmbedding]: ...


@overload
def make(
    env_id: str = "SoftSpring2D-v0", *, backend: Literal["jax"] = "jax", **kwargs: object
) -> tuple[Spring2DEnv, HostEmbedding]: ...


@overload
def make(
    env_id: str = "SoftSpring2D-v0", *, backend: str, **kwargs: object
) -> tuple[Spring2DEnv | NativeEnv, HostEmbedding]: ...


def make(
    env_id: str = "SoftSpring2D-v0", **kwargs: object
) -> tuple[Spring2DEnv | NativeEnv, HostEmbedding]:
    """Create a selected-backend scalar environment and its baseline embedding.

    The optional ``integrator`` keyword selects an existing compatible method;
    None preserves the supplied configuration or backend default. Query
    ``available_integrators`` for supported backend/family combinations.
    """
    if kwargs.get("backend") == "warp":
        from .warp_native.env import make as factory
    else:
        from .drivers.spring_environment import make as factory
    # Selected backend validates its own option schema.
    return factory(env_id, **kwargs)  # type: ignore[arg-type]


@overload
def make_vec(
    env_id: str = "SoftSpring2D-v0",
    num_envs: int = 1,
    *,
    backend: Literal["warp"],
    **kwargs: object,
) -> tuple[NativeVectorEnv, HostEmbedding]: ...


@overload
def make_vec(
    env_id: str = "SoftSpring2D-v0",
    num_envs: int = 1,
    *,
    backend: Literal["jax"] = "jax",
    **kwargs: object,
) -> tuple[Spring2DVectorEnv, HostEmbedding]: ...


@overload
def make_vec(
    env_id: str = "SoftSpring2D-v0", num_envs: int = 1, *, backend: str, **kwargs: object
) -> tuple[Spring2DVectorEnv | NativeVectorEnv, HostEmbedding]: ...


def make_vec(
    env_id: str = "SoftSpring2D-v0", num_envs: int = 1, **kwargs: object
) -> tuple[Spring2DVectorEnv | NativeVectorEnv, HostEmbedding]:
    """Create native backend batching and the stacked baseline embedding."""
    if kwargs.get("backend") == "warp":
        from .warp_native.vector_env import make_vec as factory
    else:
        from .drivers.spring_environment import make_vec as factory
    # Selected backend validates forwarded options.
    return factory(env_id, num_envs=num_envs, **kwargs)  # type: ignore[arg-type]


def make_gymnasium_vec(
    env_id: str = "SoftSpring2D-v0", num_envs: int = 1, **kwargs: object
) -> GymnasiumVectorAdapter:
    """Standard Gymnasium vector reset/step, preserving backend batching."""
    from .gymnasium_adapter import make_gymnasium_vec as factory

    return factory(env_id, num_envs=num_envs, **kwargs)


def registered_envs() -> dict[str, str]:
    """Describe the five low-level co-design environment identifiers."""
    return {
        "SoftSpring2D-v0": "implemented: spring 2D on JAX and native Warp",
        "SoftBeam3D-v0": "implemented: beam 3D on JAX and native Warp",
        "SoftMPM2D-v0": "implemented: MPM 2D on JAX and native Warp",
        "SoftMPM3D-v0": "implemented: MPM 3D on JAX and native Warp",
        "SoftHexFEM3D-v0": "implemented: hexahedral FEM 3D on JAX and native Warp",
    }
