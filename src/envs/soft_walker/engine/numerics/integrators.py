"""Fixed-step native Kvaerno3 with the M2 matrix-free stage adjoint.

Coefficients: Diffrax 0.7.2 Kvaerno3 (Apache-2.0); see NOTICE.md. This is a
separate named implementation, NOT a patch of Diffrax. No adaptive controller,
FSAL reuse, or dense fallback. All control intervals end at exact boundaries.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

import jax
import jax.numpy as jnp

from .integration_tableaux import KVAERNO3_A as A, KVAERNO3_C as C, KVAERNO3_GAMMA as GAMMA
from .solvers import NewtonConfig, RootInfo, solve_root

ArgsT = TypeVar("ArgsT")


def kvaerno3_step(
    rhs: Callable[[jax.Array, jax.Array, ArgsT], jax.Array],
    t: jax.Array,
    y: jax.Array,
    dt: jax.Array,
    args: ArgsT,
    cfg: NewtonConfig = NewtonConfig(),
) -> tuple[jax.Array, RootInfo]:
    """Return one fixed Kvaerno3 step and stacked implicit-stage diagnostics.

    The RHS sees the original state shape; only nonlinear solves are flattened.
    Failed stages produce nonfinite state values for the caller to reject.
    """
    shape = y.shape
    flat = y.reshape(-1)
    stages = [dt * rhs(t, y, args).reshape(-1)]
    infos = []

    # Stage unknowns are state increments (dt * rhs), not accelerations.
    # Flatten only at the nonlinear-solver boundary; rhs sees the original shape.
    for i in range(1, 4):
        base = flat + sum(A[i][j] * stages[j] for j in range(i))
        residual = lambda k: (
            k
            - dt
            * rhs(
                t + C[i] * dt,
                (base + GAMMA * k).reshape(shape),
                args,
            ).reshape(-1)
        )
        k, info = solve_root(residual, stages[-1], cfg)
        stages.append(k)
        infos.append(info)

    value = flat + sum(A[-1][i] * stages[i] for i in range(4))
    diagnostics = jax.tree.map(lambda *a: jnp.stack(a), *infos)
    value = value * jnp.where(jnp.all(diagnostics.accepted > 0), 1.0, jnp.nan)
    return value.reshape(shape), diagnostics
