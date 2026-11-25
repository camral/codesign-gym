"""Design+control differentiation, bound to an environment made by the fixed API.

No optimizer or neural-controller architecture is imposed on the caller.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from envs.soft_walker.engine.types import (
        Array,
    )

# Serialized reports can contain backend-specific auxiliary diagnostic trees.
Report = dict[str, Any]


def main() -> None:
    """Parse CLI settings and save relaxed design/action derivatives as NPZ and JSON.

    Select the JAX device before importing it; invalid embodiments raise rather
    than producing gradient artifacts. This example performs no optimization.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    p.add_argument("--out", default="runs/design_gradient")
    a = p.parse_args()
    os.environ["JAX_ENABLE_X64"] = "1" if a.dtype == "float64" else "0"
    os.environ["JAX_PLATFORMS"] = "cpu" if a.device == "cpu" else "cuda"
    import jax
    import jax.numpy as jnp
    import numpy as np

    from envs.soft_walker.engine import make
    from envs.soft_walker.engine.differentiation import bind_rollout
    from envs.soft_walker.engine.embodiments.decoders import cast_embedding
    from envs.soft_walker.engine.physics.spring_config import SpringConfig
    from envs.soft_walker.shared import evidence_write_json as write_json

    e, b = make(
        dtype=a.dtype,
        device=a.device,
        design_mode="relaxed",
        config=SpringConfig(grid_shape=(2, 1), settle_steps=1, substeps=4, horizon=3),
    )
    assert e.validate_embodiment(b)
    params = cast_embedding(e.decoder, b)
    actions = jnp.asarray([[0.2, -0.1], [0.15, 0.1]], dtype=a.dtype)
    rollout = bind_rollout(e)

    def loss(values: Array, actions: Array) -> Array:
        result = rollout({"values": values}, actions)
        objective = result["metrics"]["objective"]
        return objective[0] + e.config.energy_weight * objective[1]

    value, (design_grad, action_grad) = jax.block_until_ready(
        jax.jit(jax.value_and_grad(loss, argnums=(0, 1)))(params["values"], actions)
    )
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "gradients.npz", value=value, design=design_grad, actions=action_grad)
    report: Report = {
        "objective": float(value),
        "geometry_grad_norm": float(jnp.linalg.norm(design_grad[..., 0])),
        "stiffness_grad_norm": float(jnp.linalg.norm(design_grad[..., 1])),
        "density_grad_norm": float(jnp.linalg.norm(design_grad[..., 2])),
        "strength_grad_norm": float(jnp.linalg.norm(design_grad[..., 3])),
        "control_grad_norm": float(jnp.linalg.norm(action_grad)),
        "note": "Relaxed surrogate; no derivatives through hard categories/channel assignment.",
    }
    write_json(out / "summary.json", report)
    print(report)
    e.close()


if __name__ == "__main__":
    main()
