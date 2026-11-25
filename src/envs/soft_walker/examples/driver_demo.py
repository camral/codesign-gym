#!/usr/bin/env python3
"""Unoptimized external open-loop or neural control, with per-step evidence.

This is a driver smoke demonstration, not a trained gait or proof of locomotion.
Precision/device are selected before importing JAX; CUDA never silently falls back.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np


def main() -> int:
    """Write an untrained driver's step evidence and optional GIF from CLI settings.

    Return one on rejected reset or physical termination, otherwise zero. Existing
    episode output is rejected; CUDA selection never falls back to the CPU.
    """
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--env",
        default="SoftHexFEM3D-v0",
        choices=[
            "SoftSpring2D-v0",
            "SoftHexFEM3D-v0",
            "SoftMPM2D-v0",
            "SoftMPM3D-v0",
            "SoftBeam3D-v0",
        ],
    )
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    p.add_argument("--terrain", choices=["flat", "slopes", "blocks"], default="flat")
    p.add_argument("--controller", choices=["open_loop", "neural"], default="open_loop")
    p.add_argument("--steps", type=int, default=25)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", required=True)
    p.add_argument("--render", action="store_true")
    a = p.parse_args()
    if a.steps < 1:
        p.error("--steps must be positive")
    import os

    os.environ["JAX_ENABLE_X64"] = "1" if a.dtype == "float64" else "0"
    os.environ["JAX_PLATFORMS"] = a.device
    os.environ.setdefault(
        "JAX_COMPILATION_CACHE_DIR",
        str(Path(__file__).resolve().parents[1] / ".jax_cache" / a.device / a.dtype),
    )
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "episode.json").exists():
        p.error("episode exists; choose a fresh directory")
    from envs.soft_walker.engine import make
    from envs.soft_walker.engine.drivers.task import BodyConfig
    from envs.soft_walker.engine.embodiments.serialization import save_design
    from envs.soft_walker.engine.physics.spring_config import SpringConfig
    from envs.soft_walker.engine.physics.terrain import TerrainConfig
    from envs.soft_walker.shared import evidence_write_json as write_json

    tc = TerrainConfig(kind=a.terrain)
    cfg: SpringConfig | BodyConfig
    if a.env == "SoftSpring2D-v0":
        cfg = SpringConfig(
            grid_shape=(4, 2),
            control_dt=0.02,
            substeps=80,
            settle_steps=5,
            horizon=a.steps,
            terrain=tc,
            linear_solver="gmres",
        )
    else:
        shape = (4, 2) if a.env == "SoftMPM2D-v0" else (4, 2, 1)
        cfg = BodyConfig(
            grid_shape=shape, size=tuple(0.05 * n for n in shape), horizon=a.steps, terrain=tc
        )
    env, base = make(
        a.env,
        device=a.device,
        dtype=a.dtype,
        config=cfg,
        max_observation_space_size=10000,
        render_mode="rgb_array" if a.render else None,
    )
    obs, rejected, info = env.reset(seed=a.seed, embodiment=base)
    save_design(out / "design.json", base, env.decoder, {"env_id": a.env})
    write_json(out / "reset.json", info)
    if rejected[0]:
        env.close()
        return 1
    rng = np.random.default_rng(a.seed)
    # External two-layer network. No optimizer/framework is constructed.
    W = rng.normal(size=(16, len(obs))) / np.sqrt(len(obs))
    V = rng.normal(size=(env.channels, 16)) / 4
    rows = []
    frames = []
    for k in range(a.steps):
        if a.controller == "open_loop":
            command = 0.5 * np.sin(
                2 * np.pi * 2 * (k * cfg.control_dt) + np.arange(env.channels) * np.pi * 0.65
            )
        else:
            command = 0.5 * np.tanh(V @ np.tanh(W @ np.tanh(obs)))
        action = np.zeros(env.max_action_space_size, dtype=a.dtype)
        action[: env.channels] = command
        start = time.perf_counter()
        obs, r, t, tr, info = env.step(action)
        wall = time.perf_counter() - start
        rows.append(
            dict(
                step=k, action=action, reward=r, terminated=t, truncated=tr, wall_s=wall, info=info
            )
        )
        write_json(
            out / "episode.json",
            {
                "args": vars(a),
                "steps": rows,
                "trained_controller": False,
                "locomotion_validated": False,
            },
        )
        if a.render:
            frames.append(env.render())
        if t or tr:
            break
    if frames:
        from PIL import Image

        imgs = [Image.fromarray(x) for x in frames]
        imgs[0].save(
            out / "preview.gif",
            save_all=True,
            append_images=imgs[1:],
            duration=round(1000 * cfg.control_dt),
            loop=0,
        )
    env.close()
    return int(any(r["terminated"] for r in rows))


if __name__ == "__main__":
    raise SystemExit(main())
