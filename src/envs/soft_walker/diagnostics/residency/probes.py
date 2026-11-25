"""Small, independent GPU capture probes run before physical simulation.

These are diagnostics, not an alternative implementation of the simulator.
Each probe uses the same compiler policy and strict whole-call certificate as
production. NumPy references execute outside the device simulation boundary.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import SupportsFloat, cast

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike, DTypeLike

from envs.soft_walker.engine.numerics.krylov import triangular_backsolve
from envs.soft_walker.engine.numerics.solvers import NewtonConfig, linear_solve
from envs.soft_walker.engine.types import HostArray

from .fixtures import (
    BACKSOLVE_PROBE,
    CONTROL_PROBE,
    IMPLICIT_GRADIENT_PROBE,
    INITIAL_PROBE_INPUT,
)

PROBES = ("control", "backsolve", "implicit_gradient")


def capture_probe(
    name: str, dtype: DTypeLike
) -> tuple[
    Callable[[jax.Array], tuple[jax.Array, jax.Array]],
    Callable[[ArrayLike], tuple[HostArray, HostArray]],
    jax.Array,
]:
    """Return a pure JAX probe, independent NumPy reference, and scalar device input.

    The control probe returns a final counter and vector. Linear-solve probes
    return a scalar loss and its derivative with respect to the scalar input.
    ``dtype`` selects both sides' precision; unknown probe names raise ValueError.
    No compilation or profiling starts until the caller invokes the returned code.
    """
    dtype = np.dtype(dtype)
    initial = jnp.asarray(INITIAL_PROBE_INPUT, dtype)

    if name == "control":

        def raw(p: jax.Array) -> jax.Array | tuple[jax.Array, jax.Array]:
            """Exercise data-dependent nested device loops and a conditional branch."""
            # Data-dependent nested bounds and branch; no physics/linear solve
            # can be blamed if this tiny CUDA conditional-graph probe fails.
            size = jnp.asarray(CONTROL_PROBE.base_outer_steps, jnp.int32) + jnp.floor(
                jnp.abs(p) * CONTROL_PROBE.input_count_scale
            ).astype(jnp.int32)
            y = jnp.linspace(
                CONTROL_PROBE.vector_start,
                CONTROL_PROBE.vector_stop,
                CONTROL_PROBE.dimension,
                dtype=dtype,
            )

            def body(state: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
                i, y = state
                inner_size = CONTROL_PROBE.base_inner_steps + i % CONTROL_PROBE.inner_count_period

                def inner(carry: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
                    k, y = carry
                    y = jax.lax.cond(
                        jnp.sum(y) > p,
                        lambda v: (
                            CONTROL_PROBE.high_decay * v
                            + CONTROL_PROBE.high_sine_scale * jnp.sin(p)
                        ),
                        lambda v: (
                            CONTROL_PROBE.low_decay * v
                            - CONTROL_PROBE.low_cosine_scale * jnp.cos(p)
                        ),
                        y,
                    )
                    return k + 1, y

                _, y = jax.lax.while_loop(
                    lambda c: c[0] < inner_size,
                    inner,
                    (jnp.asarray(0, jnp.int32), y),
                )
                return i + 1, y

            return jax.lax.while_loop(
                lambda state: state[0] < size,
                body,
                (jnp.asarray(0, jnp.int32), y),
            )

        def reference(p: ArrayLike) -> tuple[HostArray, HostArray]:
            p = np.asarray(p, dtype)
            size = CONTROL_PROBE.base_outer_steps + int(
                np.floor(abs(p) * CONTROL_PROBE.input_count_scale)
            )
            y = np.linspace(
                CONTROL_PROBE.vector_start,
                CONTROL_PROBE.vector_stop,
                CONTROL_PROBE.dimension,
                dtype=dtype,
            )
            for i in range(size):
                for _ in range(
                    CONTROL_PROBE.base_inner_steps + i % CONTROL_PROBE.inner_count_period
                ):
                    if np.sum(y) > p:
                        y = CONTROL_PROBE.high_decay * y + CONTROL_PROBE.high_sine_scale * np.sin(p)
                    else:
                        y = CONTROL_PROBE.low_decay * y - CONTROL_PROBE.low_cosine_scale * np.cos(p)
            return np.asarray(size, np.int32), np.asarray(y, dtype)

    elif name == "backsolve":
        size = BACKSOLVE_PROBE.dimension
        base = np.triu(
            np.arange(size * size).reshape(size, size) / BACKSOLVE_PROBE.upper_triangle_divisor
        )
        base += np.diag(
            np.linspace(BACKSOLVE_PROBE.diagonal_start, BACKSOLVE_PROBE.diagonal_stop, size)
        )
        weights = np.arange(1, size + 1, dtype=dtype) / size

        def raw(p: jax.Array) -> jax.Array | tuple[jax.Array, jax.Array]:
            upper = jnp.asarray(base, dtype) + p * jnp.eye(size, dtype=dtype)
            rhs = jnp.cos(jnp.arange(size, dtype=dtype) + p)
            value = triangular_backsolve(upper, rhs)
            return jnp.dot(jnp.asarray(weights), value)

        def reference(p: ArrayLike) -> tuple[HostArray, HostArray]:
            p = float(cast(SupportsFloat, p))
            upper = base + p * np.eye(size)
            rhs = np.cos(np.arange(size) + p)
            x = np.linalg.solve(upper, rhs)
            dx = np.linalg.solve(upper, -np.sin(np.arange(size) + p) - x)
            return np.asarray(weights @ x, dtype), np.asarray(weights @ dx, dtype)

        raw = jax.value_and_grad(raw)

    elif name == "implicit_gradient":
        size = IMPLICIT_GRADIENT_PROBE.dimension
        base = np.diag(
            np.linspace(
                IMPLICIT_GRADIENT_PROBE.diagonal_start, IMPLICIT_GRADIENT_PROBE.diagonal_stop, size
            )
        ) + IMPLICIT_GRADIENT_PROBE.dense_coupling * np.ones((size, size))
        variation = np.diag(
            np.linspace(
                IMPLICIT_GRADIENT_PROBE.variation_start,
                IMPLICIT_GRADIENT_PROBE.variation_stop,
                size,
            )
        )
        variation += np.diag(np.full(size - 1, IMPLICIT_GRADIENT_PROBE.superdiagonal_variation), 1)
        config = NewtonConfig(
            gmres_method="fused",
            restart=size,
            max_restarts=IMPLICIT_GRADIENT_PROBE.krylov_restarts,
            linear_rtol64=IMPLICIT_GRADIENT_PROBE.linear_rtol64,
            linear_rtol32=IMPLICIT_GRADIENT_PROBE.linear_rtol32,
        )

        def loss(p: jax.Array) -> jax.Array:
            matrix = jnp.asarray(base, dtype) + p * jnp.asarray(variation, dtype)
            rhs = jnp.cos(jnp.arange(size, dtype=dtype) + p)

            def step(previous: jax.Array, _: None) -> tuple[jax.Array, None]:
                x, _ = linear_solve(
                    lambda v: matrix @ v,
                    rhs + IMPLICIT_GRADIENT_PROBE.recurrent_weight * previous,
                    jnp.diag(matrix),
                    config,
                )
                return x, None

            x, _ = jax.lax.scan(
                step, jnp.zeros(size, dtype), None, length=IMPLICIT_GRADIENT_PROBE.recurrence_steps
            )
            return 0.5 * jnp.dot(x, x)

        raw = jax.value_and_grad(loss)

        def reference(p: ArrayLike) -> tuple[HostArray, HostArray]:
            p = float(cast(SupportsFloat, p))
            matrix = base + p * variation
            rhs = np.cos(np.arange(size) + p)
            drhs = -np.sin(np.arange(size) + p)
            x = np.zeros(size)
            dx = np.zeros(size)
            for _ in range(IMPLICIT_GRADIENT_PROBE.recurrence_steps):
                x = np.linalg.solve(matrix, rhs + IMPLICIT_GRADIENT_PROBE.recurrent_weight * x)
                dx = np.linalg.solve(
                    matrix, drhs + IMPLICIT_GRADIENT_PROBE.recurrent_weight * dx - variation @ x
                )
            return np.asarray(0.5 * x @ x, dtype), np.asarray(x @ dx, dtype)
    else:
        raise ValueError("Unknown capture probe: " + str(name))

    return raw, reference, initial
