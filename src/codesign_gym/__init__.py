"""
Authors: Aviraj Newatia

Codesign Gym: a gymnasium-like package for co-design problems.

The public entry points are :func:`make_vec` and :func:`make`. Every environment
parallelizes in its own way, so codesign_gym does not wrap envs in Gymnasium's
Sync/Async vector envs; instead each environment provides a ``build_vec`` that returns its
native vectorized form. ``make`` is simply ``make_vec`` with ``num_envs=1``. Both return
``(env, base_embodiment)``.
"""

from __future__ import annotations

import importlib
from contextlib import contextmanager
from typing import Any

from .registry import REGISTRY
from .single_env import SingleEnv

__version__ = "1.0.0"

__all__ = ["make", "make_vec", "SingleEnv", "REGISTRY", "list_envs"]


def list_envs() -> list[str]:
    """Return the registered environment ids."""
    return sorted(REGISTRY)


def _lazy_import(path: str):
    """Import a ``"module:function"`` string and return the function."""
    module_name, _, attr = path.partition(":")
    if not attr:
        raise ValueError(f"registry path must be 'module:function', got {path!r}")
    module = importlib.import_module(module_name)
    return getattr(module, attr)


@contextmanager
def _instantiation_status(factory: str, env_id: str, num_envs: int):
    label = f"[codesign-gym] {factory}({env_id!r}, num_envs={num_envs})"
    print(f"{label}: instantiating", flush=True)
    try:
        yield
    except BaseException as error:
        print(f"{label}: failed ({type(error).__name__})", flush=True)
        raise
    else:
        print(f"{label}: success", flush=True)


# Removed from the API: every current embodiment space is bounded and each preset's spaces are fixed at the largest
# size it allows, so there is nothing to cap. Named here so an old caller gets a clear error rather than having the
# keyword forwarded to an env's build as an unknown config key.
_REMOVED_KWARGS = ("max_action_space_size", "max_observation_space_size")


def make_vec(env_id: str, num_envs: int, *, is_eval_env: bool = False, **cfg: Any):
    """Construct a vectorized environment and return ``(env, base_embodiment)``.

    The environment is built by its own ``build_vec`` (looked up lazily in
    :data:`REGISTRY`) and is already vectorized in that env's native way — codesign_gym
    does not add a vector wrapper. ``cfg`` overrides the env's default build kwargs.
    """
    with _instantiation_status("make_vec", env_id, num_envs):
        return _build_vec(env_id, num_envs, is_eval_env=is_eval_env, **cfg)


def _build_vec(env_id: str, num_envs: int, *, is_eval_env: bool = False, **cfg: Any):
    removed = [name for name in _REMOVED_KWARGS if name in cfg]
    if removed:
        raise TypeError(f"{', '.join(removed)} {'was' if len(removed) == 1 else 'were'} removed: spaces are fixed per preset at "
                        "the largest size its embodiment space allows")
    if env_id not in REGISTRY:
        raise KeyError(f"unknown env_id {env_id!r}; choose from {list_envs()}")
    path, defaults = REGISTRY[env_id]
    build = _lazy_import(path)
    env = build(num_envs=num_envs, is_eval_env=is_eval_env, **{**defaults, **cfg})
    return env, env.base_embodiment


def make(env_id: str, **kwargs: Any):
    """Construct a single environment: ``make_vec(env_id, num_envs=1, ...)`` squeezed.

    Returns ``(env, base_embodiment)`` where ``env`` wraps the num_envs=1 vector env in
    :class:`SingleEnv`, removing the leading batch axis: ``reset``/``step`` take a single
    embodiment and return an unbatched observation and scalar reward, so single-env
    consumers need no ``[0]`` indexing.
    """
    with _instantiation_status("make", env_id, 1):
        venv, _ = _build_vec(env_id, num_envs=1, **kwargs)
        env = SingleEnv(venv)
        return env, env.base_embodiment
