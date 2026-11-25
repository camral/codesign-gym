"""Small EvoGym/NEAT architecture fixture, not a NEAT optimizer or simulator.

Source architecture: EvolutionGym/examples/cppn_neat/neat.cfg. Activation semantics:
neat-python/neat/activations.py. The main CPPN decoder intentionally uses separate
physical heads and milder hidden activations; this fixture preserves the original
3-input / 1-hidden / 5-category topology and gain-5 sigmoid for comparison.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import DTypeLike

from ..types import Embedding, HostArray, HostEmbedding


def evogym_fixture(dtype: DTypeLike = "float64") -> HostEmbedding:
    """Deterministic example weights, NOT weights from a published trained robot."""
    rng = np.random.default_rng(13)
    allowed: HostArray = np.zeros((6, 9), bool)
    allowed[0, :3] = True
    allowed[1:, :4] = True
    candidates = np.flatnonzero(allowed.reshape(-1))
    selected = rng.choice(candidates, size=round(0.5 * len(candidates)), replace=False)
    mask = np.zeros_like(allowed, dtype=np.int32)
    mask.reshape(-1)[selected] = 1
    return {
        "weights": rng.normal(0, 0.25, allowed.shape).astype(dtype),
        "bias": rng.normal(0, 0.2, 6).astype(dtype),
        "edge_mask": mask,
    }


def evogym_category_scores(embedding: Embedding, points: jax.Array) -> jax.Array:
    """points[...,2] -> scores[...,5] in empty/rigid/soft/H/V order."""
    features = jnp.concatenate((points, jnp.linalg.norm(points, axis=-1, keepdims=True)), axis=-1)
    values = jnp.concatenate(
        (features, jnp.zeros(points.shape[:-1] + (6,), dtype=points.dtype)), axis=-1
    )
    weights = embedding["weights"] * embedding["edge_mask"]
    for i in range(6):
        z = jnp.sum(values * weights[i], axis=-1) + embedding["bias"][i]
        values = values.at[..., 3 + i].set(jax.nn.sigmoid(jnp.clip(5 * z, -60, 60)))
    return values[..., 4:9]
