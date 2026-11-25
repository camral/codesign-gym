"""Compiled API reporting and environment batching.

Array work belongs inside jit/vmap; Python loops are retained only to maintain
per-lane API objects and to assemble human-readable info dictionaries. Temporal
recurrences and Krylov/Newton iterations are not independent batch dimensions.

The numerical step is not changed here. In particular, we do not relax solver
checks, change timesteps, or differentiate a different discretization.
"""

from __future__ import annotations

from functools import lru_cache

import jax
import jax.numpy as jnp

from .types import (
    AdvanceKernel,
    DriverState,
    ObserveKernel,
    PhysicsModel,
    ReportKernel,
    RewardKernel,
    SummaryKernel,
    TransitionDiagnostics,
    TransitionReport,
    VectorStepKernel,
)


@lru_cache(maxsize=64)
def reporting_kernel(
    observe: ObserveKernel, summary: SummaryKernel, reward: RewardKernel
) -> ReportKernel:
    """One compiled dispatch for all per-transition reporting reductions."""

    @jax.jit
    def report(model: PhysicsModel, before: DriverState, after: DriverState) -> TransitionReport:
        """Assemble device observation, mask, summary, and reward in one dispatch."""
        observation, mask = observe(model, after)
        return observation, mask, summary(model, after), reward(model, before, after)

    return report


@lru_cache(maxsize=64)
def vector_step_kernel(
    advance: AdvanceKernel, observe: ObserveKernel, summary: SummaryKernel, reward: RewardKernel
) -> VectorStepKernel:
    """Batch physics AND reporting, with stacking/slicing inside compilation.

    Input/output tuples preserve the public vector driver's scalar env objects.
    Their arrays remain on device: only the reporting payload is copied to the
    host. This avoids an eager stack/slice dispatch for each state/model leaf.
    Shapes, dtype and structure must agree; make_vec already imposes this.
    """
    report = reporting_kernel(observe, summary, reward)

    @jax.jit
    def step(
        models: tuple[PhysicsModel, ...], states: tuple[DriverState, ...], actions: jax.Array
    ) -> tuple[tuple[DriverState, ...], tuple[TransitionReport, TransitionDiagnostics]]:
        """Stack matching lane pytrees, vmap physics/reporting, and return device lane states."""
        batched_models = jax.tree.map(lambda *xs: jnp.stack(xs), *models)
        batched_states = jax.tree.map(lambda *xs: jnp.stack(xs), *states)

        next_states, diagnostics = jax.vmap(advance)(batched_models, batched_states, actions)
        payload = jax.vmap(report)(batched_models, batched_states, next_states)

        # This Python loop is traced once for a fixed batch size, not executed
        # as one device dispatch per array per lane on every environment step.
        lanes = tuple(jax.tree.map(lambda x: x[i], next_states) for i in range(len(states)))
        return lanes, (payload, diagnostics)

    return step
