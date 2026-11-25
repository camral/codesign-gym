"""A user-side sinusoidal controller using only the fixed environment API."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from envs.soft_walker.engine.physics import spring


import argparse
import os
import time
from pathlib import Path

import numpy as np


def main() -> int:
    """Save a seeded spring rollout, design, and optional GIF from CLI settings.

    A rejected design raises RuntimeError. Return zero for a valid final state
    and one for an invalid simulation; the output retains step diagnostics.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    p.add_argument(
        "--decoder", choices=["direct", "cppn", "primitives", "bspline"], default="direct"
    )
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--out", default="runs/open_loop")
    p.add_argument("--gif", action="store_true")
    a = p.parse_args()
    if a.steps < 1:
        p.error("--steps must be positive")
    os.environ["JAX_ENABLE_X64"] = "1" if a.dtype == "float64" else "0"
    os.environ["JAX_PLATFORMS"] = "cpu" if a.device == "cpu" else "cuda"
    from envs.soft_walker.engine import make
    from envs.soft_walker.engine.embodiments.serialization import save_design
    from envs.soft_walker.engine.physics.spring_config import SpringConfig
    from envs.soft_walker.shared import evidence_write_json as write_json

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    e, b = make(
        "SoftSpring2D-v0",
        dtype=a.dtype,
        device=a.device,
        decoder=a.decoder,
        is_eval_env=True,
        config=SpringConfig(horizon=a.steps),
        render_mode="rgb_array" if a.gif else None,
    )
    obs, rejected, info = e.reset(seed=17, embodiment=b)
    if rejected[0]:
        raise RuntimeError(info)
    save_design(out / "design.json", b, e.decoder, {"seed": 17, "controller": "external sinusoid"})
    frames = []
    rows = []
    positions = []
    try:
        for k in range(a.steps):
            t = k * e.config.control_dt
            action = np.zeros(e.max_action_space_size, dtype=a.dtype)
            action[:2] = 0.35 * np.sin(2 * np.pi * 1.5 * t + np.array([0.0, np.pi / 2]))
            start = time.perf_counter()
            obs, reward, term, trunc, info = e.step(action)
            rows.append(
                {
                    "step": k,
                    "action": action,
                    "reward": reward,
                    "terminated": term,
                    "truncated": trunc,
                    "wall_s": time.perf_counter() - start,
                    **info,
                }
            )
            positions.append(np.asarray(cast("spring.State", e._state).x))  # debug snapshot
            if a.gif:
                frames.append(e.render())
            if term or trunc:
                break
        write_json(out / "trajectory.json", rows)
        np.savez_compressed(out / "positions.npz", positions=np.stack(positions))
        if frames:
            from PIL import Image

            images = [Image.fromarray(x) for x in frames]
            images[0].save(
                out / "preview.gif",
                save_all=True,
                append_images=images[1:],
                duration=round(e.config.control_dt * 1000),
                loop=0,
            )
        print(
            f"Final metrics: distance={info['distance_m']} m, work={info['work_positive_j']} J, status={info['status_name']}"
        )
        print(f"Evidence: {out.resolve()}")
        return 0 if info["valid"] else 1
    finally:
        e.close()


if __name__ == "__main__":
    raise SystemExit(main())
