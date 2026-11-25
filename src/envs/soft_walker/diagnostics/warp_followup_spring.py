#!/usr/bin/env python3
"""Replay the returned 1,400-step spring problem using original JAX mechanics.

The objective is taken from the user's comparison helper. It is not a gait score.
This worker runs CPU JAX deliberately; native CUDA testing is a separate stage.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import asdict, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, cast

if __package__ in (None, ""):
    # Direct-file execution must select this checkout before package imports.
    _source_root = str(Path(__file__).resolve().parents[3])
    if _source_root not in sys.path:
        sys.path.insert(0, _source_root)

from envs.soft_walker.shared import (
    ENGINE_ROOT as ENGINE_ROOT,
    REPO_ROOT as REPO_ROOT,
    SCRIPTS_ROOT as SCRIPTS_ROOT,
    SOFT_ROOT as SOFT_ROOT,
    SRC_ROOT as SRC_ROOT,
    TESTS_ROOT as TESTS_ROOT,
)

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ["JAX_ENABLE_X64"] = "1"
import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

    from envs.soft_walker.engine.physics import spring
    from envs.soft_walker.engine.physics.spring_config import SpringConfig
    from envs.soft_walker.engine.types import (
        Array,
    )
    from envs.soft_walker.engine.warp_native.audit_types import ResolvedParameters
    from envs.soft_walker.engine.warp_native.solid_model import SolidBundle

# Serialized reports can contain backend-specific auxiliary diagnostic trees.
Report = dict[str, Any]


def require(value: Any, message: str) -> None:
    """Raise a descriptive validation error when a numerical or protocol prerequisite fails."""
    if not value:
        raise RuntimeError(message)


def jax_objective(
    bundle: SolidBundle, settings: Report, cfg: SpringConfig
) -> Callable[[ResolvedParameters], tuple[Array, dict[str, Array]]]:
    """Map independent resolved arrays to the original spring trial routine.

    No native residual, force, solver or gradient is used by the JAX executable.
    Unused cross-physics fields stay in the differentiated input schema and have
    zero cotangents on both sides. Geometry/support masks are fixed switches.
    """
    import jax
    import jax.numpy as j

    from envs.soft_walker.engine.physics import spring

    fixed = {name: j.asarray(bundle.fields[name]) for name in ("entities", "cells", "free")}
    require(
        bundle.terrain is None, "This bounded helper supports the original flat spring demo only"
    )
    active = fixed["free"][:, 0] > 0
    channels = j.ones(bundle.channels, dtype=bool)
    stride = min(settings["checkpoint_stride"], settings["physical_steps"])
    steps = settings["physical_steps"]
    settling = settings["settling_steps"]
    indices = j.arange(((steps + stride - 1) // stride) * stride).reshape(-1, stride)

    def objective(parameters: ResolvedParameters) -> tuple[Array, dict[str, Array]]:
        f, initial, commands = parameters["model"], parameters["initial"], parameters["commands"]
        model = spring.Model(
            reference=f["reference"],
            cells=fixed["cells"],
            edges=fixed["entities"],
            mass=f["mass"],
            active=active,
            stiffness=f["stiffness"],
            damping=f["damping"],
            length=f["length"],
            logstrain_map=f["routing"],
            contact_k=f["contact_area"][:, 0],
            occupancy=f["cell_alpha"],
            young=f["young"],
            density=j.zeros_like(f["mass"]),
            strength=f["strength"],
            channel_mask=channels,
            terrain=None,
        )
        zero = j.asarray(0.0, commands.dtype)
        ledger = initial["ledger"]
        state = spring.State(
            f["reference"] + initial["u"],
            initial["v"],
            initial["actuator"],
            ledger[1],
            ledger[0],
            zero,
            ledger[2],
            ledger[3],
            spring.com(model, f["reference"] + initial["u"]),
            j.zeros(2, commands.dtype),
            j.asarray(0, j.int32),
        )

        def one(
            carry: tuple[spring.State, Array], index: int
        ) -> tuple[tuple[spring.State, Array], None]:
            state, travel = carry

            def physical(_: None) -> tuple[spring.State, Array]:
                slot = j.maximum(index - settling, 0) // settings["substeps"]
                command = j.where(index < settling, j.zeros_like(commands[0]), commands[slot])
                accepted, diagnostics = spring.substep(model, state, command, cfg)
                increment = j.where(
                    (index >= settling) & (accepted.status == 0) & (f["mass"][:, None] > 0),
                    cfg.dt * accepted.v,
                    j.zeros_like(accepted.v),
                )
                return accepted, travel + increment

            return jax.lax.cond(index < steps, physical, lambda _: carry, None), None

        def block(
            carry: tuple[spring.State, Array], block_indices: Array
        ) -> tuple[tuple[spring.State, Array], None]:
            return jax.lax.scan(jax.checkpoint(one), carry, block_indices)[0], None

        (final, travel), _ = jax.lax.scan(
            jax.checkpoint(block), (state, j.zeros_like(initial["u"])), indices
        )
        mass = f["mass"]
        per_node = -travel[:, 0] / settings["length_ref"] + settings["velocity_weight"] * j.sum(
            final.v**2, axis=1
        )
        loss = j.sum(mass * per_node) / j.sum(mass) + settings["work_weight"] * final.work_positive
        loss = j.where(final.status == 0, loss, j.asarray(j.nan, commands.dtype))
        out = dict(
            u=final.x - f["reference"],
            v=final.v,
            rotation=j.where(
                (mass > 0)[:, None, None], initial["rotation"], j.eye(2, dtype=commands.dtype)
            ),
            omega=j.zeros_like(initial["omega"]),
            actuator=final.actuator,
            ledger=j.stack((final.work_signed, final.work_positive, final.dissipated, final.time)),
            status=j.reshape(final.status, (1,)),
            travel=travel,
        )
        return loss, out

    return objective


def main() -> int:
    """Command entry point. Replay the returned 1,400-step spring problem using original JAX mechanics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs", type=Path, default=SOFT_ROOT / "baselines/warp_followup/spring_biped"
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Reproduce the original unpreconditioned/no-refinement failure",
    )
    parser.add_argument("--gradient", action="store_true")
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    args.out.mkdir(parents=True, exist_ok=False)
    status: Report = {
        "success": False,
        "legacy": args.legacy,
        "device": "cpu",
        "precision": "float64",
    }
    start = time.perf_counter()
    try:
        import jax
        import jax.numpy as jnp

        from envs.soft_walker.engine.physics.spring_config import SpringConfig
        from envs.soft_walker.engine.warp_native.solid_model import INTEGER_FIELDS, SolidBundle

        inputs = args.inputs
        bundle = SolidBundle.load(inputs / "model.npz")
        settings = json.loads((inputs / "benchmark.json").read_text())
        commands = np.load(inputs / "commands.npy", allow_pickle=False)
        data = json.loads((inputs / "jax_spring_config.json").read_text())
        data["grid_shape"], data["size"] = tuple(data["grid_shape"]), tuple(data["size"])
        cfg = SpringConfig(**data)
        if args.legacy:
            cfg = replace(cfg, gmres_preconditioner="none", gmres_refinement_steps=0)
        parameters = {
            "model": {
                k: jnp.asarray(v)
                for k, v in bundle.fields.items()
                if k not in INTEGER_FIELDS + ("free",)
            },
            "initial": {
                k: jnp.asarray(v)
                for k, v in cast("dict[str, Array]", bundle.initial).items()
                if k != "status"
            },
            "commands": jnp.asarray(commands),
        }
        raw = jax_objective(bundle, settings, cfg)
        fn = jax.value_and_grad(raw, has_aux=True) if args.gradient else raw
        t = time.perf_counter()
        executable = jax.jit(fn).lower(parameters).compile()
        status.update(
            compilation_seconds=time.perf_counter() - t,
            physical_steps=settings["physical_steps"],
            numerical_config=asdict(cfg),
        )
        (args.out / "progress.json").write_text(json.dumps(status, indent=2))
        samples = []
        for _ in range(args.repeats):
            t = time.perf_counter()
            result = jax.block_until_ready(executable(parameters))
            samples.append(time.perf_counter() - t)
        if args.gradient:
            (loss, state), gradient = result
        else:
            loss, state = result
            gradient = None
        arrays = {
            "loss": np.asarray(loss).reshape(1),
            **{"state_" + k: np.asarray(v) for k, v in state.items()},
        }
        if gradient is not None:
            arrays["gradient_commands"] = np.asarray(gradient["commands"])
        np.savez_compressed(args.out / "parity.npz", **arrays)
        status.update(
            warm_seconds=samples, status=int(np.asarray(state["status"])[0]), loss=float(loss)
        )
        require(status["status"] == 0 and np.isfinite(loss), "Physical solve failed")
        reference = np.load(inputs / "returned_warp_reference.npz", allow_pickle=False)
        comparisons = {}
        for k, a in arrays.items():
            if k not in reference.files:
                continue
            b = reference[k]
            require(a.shape == b.shape, "Reference shape mismatch: " + k)
            rel = float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))
            passed = bool(
                np.allclose(a, b, rtol=1e-5 if k.startswith("gradient_") else 2e-8, atol=2e-11)
            )
            comparisons[k] = {
                "relative_l2": rel,
                "max_abs": float(np.max(np.abs(a - b), initial=0)),
                "passed": passed,
            }
            require(passed, "Returned Warp comparison failed: " + k)
        status.update(comparisons=comparisons, success=True)
    except BaseException:
        status["exception"] = traceback.format_exc()
        print(status["exception"], flush=True)
    finally:
        status["seconds"] = time.perf_counter() - start
        (args.out / "result.json").write_text(json.dumps(status, indent=2, default=str))
        print(json.dumps(status, indent=2, default=str), flush=True)
    return 0 if status["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
