"""Small external controllers: open-loop and real observed-state feedback.

All parameter vectors are flat float arrays, convenient for Adam or L-BFGS.
Observation-dependent policies use only the declared COM-velocity observation
slice, shared by every driver. No hidden state or privileged foot sensor is used.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from envs.soft_walker.engine.types import HostArray
from envs.soft_walker.presets.scenarios.config import Scenario


def leg_phases(dim: int, gait: str) -> HostArray:
    """Return the fixed phase offsets for each leg in the selected gait."""
    if dim == 2:
        return np.array([0.0, 0.0]) if gait in ("pronk", "bound") else np.array([0.0, np.pi])
    # Leg order: rear-left, rear-right, front-left, front-right.
    return np.asarray(
        {
            "walk": (0.0, np.pi, 1.5 * np.pi, 0.5 * np.pi),
            "crawl": (0.0, np.pi, 1.5 * np.pi, 0.5 * np.pi),
            "trot": (0.0, np.pi, np.pi, 0.0),
            "bound": (0.0, 0.0, np.pi, np.pi),
            "pronk": (0.0, 0.0, 0.0, 0.0),
        }[gait]
    )


def seed_parameters(case: Scenario) -> HostArray:
    """Hand-derived guesses, not optimized or gait-certified coefficients."""
    dim, count = case.dim, case.channels
    per_leg = 2 if dim == 2 else 4
    phases: HostArray = np.repeat(leg_phases(dim, case.gait), per_leg)
    signs = np.tile(np.repeat([-1.0, 1.0], per_leg // 2), 2 if dim == 2 else 4)
    if case.controller == "pulse":
        # Bias, loading/release amplitude, bending sweep, phase adjustment.
        values = np.stack(
            (np.zeros(count), np.full(count, case.lift), signs * case.amplitude, np.zeros(count)),
            axis=-1,
        )
        return values.astype(case.dtype).ravel()
    values = np.zeros((count, 5), case.dtype)
    values[:, 1] = signs * case.amplitude * np.cos(phases) - case.lift * np.sin(phases)
    values[:, 2] = signs * case.amplitude * np.sin(phases) + case.lift * np.cos(phases)
    if case.controller == "fourier":
        return values.ravel()
    if case.controller == "feedback":
        # Residual feedback around the same seed; no untested gain injected.
        return np.r_[values.ravel(), np.zeros(count * dim, case.dtype)]
    features, hidden = 2 + dim, 8
    rng = np.random.default_rng(case.seed)
    input_weights = rng.normal(scale=0.25, size=(hidden, features)).astype(case.dtype)
    # Zero output residual preserves the open-loop seed exactly. Hidden weights
    # receive gradients after the output weights move; this is documented.
    return np.r_[
        values.ravel(),
        input_weights.ravel(),
        np.zeros(hidden, case.dtype),
        np.zeros(count * hidden, case.dtype),
    ]


def parameter_count(case: Scenario) -> int:
    """Count trainable scalar coefficients for the selected controller."""
    return int(seed_parameters(case).size)


def action(
    case: Scenario, parameters: jax.Array, time: jax.Array, observation: jax.Array | None = None
) -> jax.Array:
    """Evaluate commands at interval start. Exactly one final action tanh."""
    count, dim = case.channels, case.dim
    theta = 2.0 * jnp.pi * time / case.period
    if case.controller == "pulse":
        values = parameters.reshape(count, 4)
        per_leg = 2 if dim == 2 else 4
        phase = theta + jnp.repeat(
            jnp.asarray(leg_phases(dim, case.gait), parameters.dtype), per_leg
        )
        phase = phase + values[:, 3]
        # Smooth pulse waveform: slow loading occupies most of the cycle. The
        # actuator ODE still applies; this does not create an instantaneous kick.
        waveform = jnp.tanh(4.0 * (jnp.cos(phase) - 0.25))
        logits = values[:, 0] + values[:, 1] * waveform + values[:, 2] * jnp.sin(phase)
    else:
        values = parameters[: count * 5].reshape(count, 5)
        basis = jnp.stack(
            (
                jnp.asarray(1.0, parameters.dtype),
                jnp.sin(theta),
                jnp.cos(theta),
                jnp.sin(2.0 * theta),
                jnp.cos(2.0 * theta),
            )
        )
        logits = values @ basis
        if case.controller in ("feedback", "neural"):
            if observation is None:
                raise ValueError("Feedback/neural controllers require the public observation")
            velocity = observation[dim : 2 * dim]
            if case.controller == "feedback":
                logits = logits + parameters[count * 5 :].reshape(count, dim) @ velocity
            else:
                features = jnp.concatenate((jnp.stack((jnp.sin(theta), jnp.cos(theta))), velocity))
                offset, hidden = count * 5, 8
                n = hidden * (2 + dim)
                weights = parameters[offset : offset + n].reshape(hidden, 2 + dim)
                offset += n
                bias = parameters[offset : offset + hidden]
                output = parameters[offset + hidden :].reshape(count, hidden)
                logits = logits + output @ jnp.tanh(weights @ features + bias)
    return jnp.tanh(logits)
