"""Explicit test/oracle utilities, never imported by the native simulation.

These comparisons resolve one common physical model before either solver runs.
They are not a replacement for long-horizon gait refinement or CUDA auditing.
"""

from typing import TYPE_CHECKING, cast

from envs.soft_walker.contracts.metrics import DIAGNOSTIC_STATUS

if TYPE_CHECKING:
    from collections.abc import Callable

    import jax
    from numpy.typing import ArrayLike

    from envs.soft_walker.engine.physics import mpm
    from envs.soft_walker.engine.warp_native.host_types import (
        EpisodeResult,
        FieldArrays,
        HostArray,
        PathLike,
        Report,
    )
    from envs.soft_walker.engine.warp_native.model import Bundle

import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from envs.soft_walker.engine.warp_native.episode import Episode
from envs.soft_walker.engine.warp_native.model import DeviceModel, uniform_body


def contact_fixture(
    dim: "int", precision: "str", law: "str" = "active_stress", terrain: "str" = "plane"
) -> "Bundle":
    """Construct a tiny moving, actuated MPM body against a plane, slope or box."""
    bundle = uniform_body(dim, precision=precision)
    bundle.config = replace(
        bundle.config,
        dt=0.00025,
        gravity=9.81,
        contact_enabled=True,
        contact_height=0.0252,
        actuation_model=law,
        actuator_tau=0.01,
        linear_refinements=2,
    )
    n = len(bundle.fields["reference"])
    bundle.fields["alpha"].fill(0.8)
    bundle.initial = dict(
        x=bundle.fields["reference"].copy(),
        v=np.zeros((n, dim), precision),
        F=np.tile(np.eye(dim, dtype=precision), (n, 1, 1)),
        C=np.zeros((n, dim, dim), precision),
        actuator=np.full(1, 0.02, precision),
        ledger=np.zeros(4, precision),
        status=np.zeros(1, np.int32),
    )
    bundle.initial["v"][:, 0] = 0.02
    if terrain != "plane":
        centers: HostArray
        half: HostArray
        normal: HostArray = np.eye(dim, dtype=precision)[1]
        if terrain == "slope":
            normal[0] = -0.07
            normal /= np.linalg.norm(normal)
            offset = 0.0252 - 0.07 * 0.0375
            centers, half = np.empty((0, dim), precision), np.empty((0, dim), precision)
        elif terrain == "box":
            offset = 0.0
            centers = np.full((1, dim), 0.0375, precision)
            centers[0, 1] = 0.01
            half = np.full((1, dim), 0.04, precision)
            half[0, 1] = 0.0152
        else:
            raise ValueError(terrain)
        bundle.terrain = dict(
            normal=normal,
            offset=np.asarray(offset, precision),
            centers=centers,
            half_sizes=half,
            rounding=np.full(len(centers), 0.001, precision),
            active=np.ones(len(centers), np.int32),
        )
    return bundle.validate()


def parameters(bundle: "Bundle", commands: "HostArray") -> "FieldArrays":
    """Select matched differentiable command/material/state leaves for the oracle."""
    return dict(
        commands=commands,
        alpha=bundle.fields["alpha"],
        young=bundle.fields["young"],
        poisson=bundle.fields["poisson"],
        density=bundle.fields["density"],
        strength=bundle.fields["strength"],
        x=cast("FieldArrays", bundle.initial)["x"],
    )


def jax_objective(
    bundle: "Bundle", *, substeps: "int" = 1, settling_steps: "int" = 1
) -> "Callable[[dict[str, jax.Array]], tuple[jax.Array, tuple[mpm.State, jax.Array]]]":
    # Imports deliberately local: the runtime and its data files need no JAX.
    """Build the independent JAX rollout loss for the supplied resolved MPM model."""
    import jax
    import jax.numpy as j

    from envs.soft_walker.engine.physics import mpm
    from envs.soft_walker.engine.warp_native.mpm_conversion import to_jax

    model, state, grid, config = to_jax(bundle)

    def objective(p: "dict[str, jax.Array]") -> "tuple[jax.Array, tuple[mpm.State, jax.Array]]":
        """Evaluate matched loss, final state and diagnostics for parameter leaves."""
        m = model._replace(
            alpha=p["alpha"],
            young=p["young"],
            poisson=p["poisson"],
            density=p["density"],
            strength=p["strength"],
            mass=p["alpha"] * p["density"] * model.cloud.volume,
        )
        initial = state._replace(x=p["x"])
        settling = j.zeros((settling_steps, bundle.channels), model.alpha.dtype)
        actions = j.concatenate((settling, j.repeat(p["commands"], substeps, axis=0)))

        def advance(
            s: "mpm.State", a: "jax.Array"
        ) -> "tuple[mpm.State, tuple[jax.Array, jax.Array]]":
            """Take one oracle physics step and return accepted travel and diagnostics."""
            result, diag = mpm.trial(
                m, s, a, j.asarray(bundle.config.dt, m.alpha.dtype), grid, config
            )
            increment = j.where(m.active[:, None], bundle.config.dt * result.v, 0.0)
            return result, (j.stack((diag.code, diag.accepted)), increment)

        final, (diagnostics, increments) = jax.lax.scan(advance, initial, actions)
        travel = j.sum(increments, axis=0)
        loss = (
            j.sum(m.mass * (-travel[:, 0] / 0.2 + 0.1 * j.sum(final.v**2, axis=-1))) / j.sum(m.mass)
            + 0.05 * final.work_positive
        )
        return loss, (final, diagnostics)

    return objective


def compare_array(
    native: "ArrayLike", reference: "ArrayLike", *, name: "str", precision: "str"
) -> "Report":
    """Assert existing state/gradient tolerances and return quantitative errors."""
    n, r = np.asarray(native), np.asarray(reference)
    if n.shape != r.shape or not np.all(np.isfinite(n)) or not np.all(np.isfinite(r)):
        raise AssertionError(f"{name}: invalid shape or nonfinite output")
    difference = n - r
    scale = max(float(np.linalg.norm(r)), 1e-30)
    result = dict(
        relative_l2=float(np.linalg.norm(difference)) / scale,
        max_abs=float(np.max(np.abs(difference), initial=0)),
        reference_l2=float(np.linalg.norm(r)),
        shape=list(n.shape),
    )
    atol = 2e-9 if precision == "float64" else 3e-7
    rtol = 3e-7 if precision == "float64" else 3e-3
    good = (
        np.linalg.norm(difference) <= rtol * np.linalg.norm(r) + 1e-24
        if name.startswith("gradient_")
        else np.allclose(n, r, rtol=rtol, atol=atol)
    )
    if not good:
        raise AssertionError(f"{name}: {result}")
    result.update(
        rtol=rtol,
        atol=atol,
        gradient_acceptance="relative L2 plus 1e-24 absolute L2"
        if name.startswith("gradient_")
        else None,
    )
    return cast("Report", result)


def _validate_execution(result: "EpisodeResult", *, gradient: "bool", phase: "str") -> "None":
    """Reject failed forward/adjoint runs before using their values or timings."""
    if result.get("status") != 0 or not np.isfinite(result.get("loss", np.nan)):
        raise AssertionError(f"{phase}: invalid forward execution: {result}")
    if gradient and result.get("adjoint_status") != 0:
        raise AssertionError(f"{phase}: invalid adjoint execution: {result}")


def parity_case(
    dim: "int",
    precision: "str",
    law: "str",
    device: "str",
    out: "PathLike",
    *,
    terrain: "str" = "plane",
    repeats: "int" = 2,
) -> "Report":
    """Run native/JAX value-and-gradient parity and save numerical evidence.

    The output directory receives phase, timing, array and comparison files.
    Failed forwards or adjoints are rejected before timings are accepted."""
    import jax
    import jax.numpy as j

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    bundle = contact_fixture(dim, precision, law, terrain)
    commands = np.array([[0.4], [-0.2], [0.15]], precision)
    episode = Episode(
        DeviceModel(bundle, device),
        commands,
        settling_steps=1,
        checkpoint_stride=2,
        velocity_weight=0.1,
        work_weight=0.05,
    )
    (out / "phase.json").write_text(json.dumps({"phase": "native whole forward and gradient"}))
    native_result = episode.run(gradient=True)
    (out / "native_first.json").write_text(json.dumps(native_result, indent=2))
    # Archive keys are model/recording fields, never NumPy's allow_pickle option.
    np.savez_compressed(out / "native_recording.npz", **episode.recording())  # type: ignore[arg-type]
    _validate_execution(native_result, gradient=True, phase="Initial parity run")
    native_grad = episode.gradients()
    p = {k: j.asarray(v) for k, v in parameters(bundle, commands).items()}
    objective = jax_objective(bundle)
    (out / "phase.json").write_text(json.dumps({"phase": "independent JAX CPU oracle"}))
    start = time.perf_counter()
    (reference_value, (reference_state, reference_diagnostics)), reference_grad = jax.jit(
        jax.value_and_grad(objective, has_aux=True)
    )(p)
    jax.block_until_ready(reference_value)
    oracle_seconds = time.perf_counter() - start
    if np.any(np.asarray(reference_diagnostics)[:, DIAGNOSTIC_STATUS] != 0):
        raise AssertionError("JAX reference step failed")
    comparisons, arrays = {}, {}
    for name in ("x", "v", "F", "C", "actuator"):
        a, b = getattr(episode.final, name).numpy(), np.asarray(getattr(reference_state, name))
        comparisons["state_" + name] = compare_array(a, b, name=name, precision=precision)
        arrays["native_state_" + name], arrays["jax_state_" + name] = a, b
    for name, reference in reference_grad.items():
        native = (
            native_grad["commands"]
            if name == "commands"
            else native_grad["initial"]["x"]
            if name == "x"
            else native_grad["model"][name]
        )
        comparisons["gradient_" + name] = compare_array(
            native, reference, name="gradient_" + name, precision=precision
        )
        arrays["native_gradient_" + name], arrays["jax_gradient_" + name] = (
            native,
            np.asarray(reference),
        )
    comparisons["loss"] = compare_array(
        episode.loss.numpy(),
        np.asarray(reference_value).reshape(1),
        name="loss",
        precision=precision,
    )
    arrays["native_loss"], arrays["jax_loss"] = (
        episode.loss.numpy(),
        np.asarray(reference_value).reshape(1),
    )
    # Archive keys are model/recording fields, never NumPy's allow_pickle option.
    np.savez_compressed(out / "arrays.npz", **arrays)  # type: ignore[arg-type]
    # Archive keys are model/recording fields, never NumPy's allow_pickle option.
    np.savez_compressed(out / "recording.npz", **episode.recording())  # type: ignore[arg-type]
    (out / "phase.json").write_text(json.dumps({"phase": "synchronized repetitions"}))
    timings = {}
    for gradient in (False, True):
        warmup = episode.run(gradient=gradient)  # untimed complete executable
        _validate_execution(warmup, gradient=gradient, phase="Timing warmup")
        samples = []
        for repeat in range(repeats):
            sample = episode.run(gradient=gradient)
            _validate_execution(sample, gradient=gradient, phase=f"Timing repetition {repeat}")
            samples.append(sample)
        timings["value_gradient" if gradient else "forward"] = samples
    result = dict(
        dimension=dim,
        precision=precision,
        actuation_model=law,
        terrain=terrain,
        first_native=native_result,
        oracle_compile_and_execution_seconds=oracle_seconds,
        comparisons=comparisons,
        timings=timings,
        config=bundle.config.to_dict(),
        driver_diagnostics=episode.step.diagnostics(),
    )
    (out / "result.json").write_text(json.dumps(result, indent=2))
    jax.clear_caches()
    return cast("Report", result)


def directional_ladder(precision: "str", device: "str", out: "PathLike") -> "Report":
    """Compare implicit gradients with a finite-difference epsilon ladder."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    bundle = contact_fixture(2, precision)
    command = np.array([[0.4], [-0.2], [0.15]], precision)
    direction = np.array([[0.1], [-0.07], [0.08]], precision)
    episode = Episode(
        DeviceModel(bundle, device),
        command,
        settling_steps=1,
        checkpoint_stride=2,
        velocity_weight=0.1,
        work_weight=0.05,
    )
    _validate_execution(episode.run(gradient=True), gradient=True, phase="Numerical ladder center")
    gradient = episode.gradients()
    derivative = float(
        np.sum(gradient["commands"] * direction)
        + np.sum(gradient["model"]["young"] * bundle.fields["young"] * 0.8)
        + np.sum(gradient["model"]["strength"] * bundle.fields["strength"] * 0.1)
    )

    def value(theta: "float") -> "float":
        """Evaluate a perturbed resolved-model loss for the finite-difference ladder."""
        episode.set_commands(command + theta * direction)
        episode.owner.model.young.assign(
            (bundle.fields["young"] * np.exp(0.8 * theta)).astype(precision)
        )
        episode.owner.model.strength.assign(
            (bundle.fields["strength"] * (1 + 0.1 * theta)).astype(precision)
        )
        result = episode.run()
        _validate_execution(result, gradient=False, phase=f"Numerical ladder at {theta}")
        return result["loss"]

    rows = []
    for h in 0.4 * 0.5 ** np.arange(9):
        plus, minus = value(float(h)), value(float(-h))
        fd = (plus - minus) / (2 * h)
        rows.append(
            dict(
                h=float(h),
                plus=plus,
                minus=minus,
                central_difference=float(fd),
                ad=derivative,
                absolute_error=float(abs(fd - derivative)),
                relative_error=float(abs(fd - derivative) / max(abs(derivative), 1e-30)),
            )
        )
        (out / "ladder_progress.json").write_text(
            json.dumps(dict(rows=rows, completed=False), indent=2)
        )
    orders = []
    for a, b in zip(rows, rows[1:]):
        orders.append(
            float(np.log2(a["absolute_error"] / b["absolute_error"]))
            if a["absolute_error"] > 0 and b["absolute_error"] > 0
            else None
        )
    windows = [
        i
        for i in range(len(orders) - 1)
        if all(v is not None and 1.5 <= v <= 2.5 for v in orders[i : i + 2])
    ]
    best = min(row["relative_error"] for row in rows)
    passed = bool(
        best < (3e-5 if precision == "float64" else 5e-3)
        and (bool(windows) or precision == "float32")
    )
    result = dict(
        precision=precision,
        passed=passed,
        ad=derivative,
        rows=rows,
        orders=orders,
        second_order_windows=windows,
        best_relative_error=best,
        qualification="Short complete contact/settling rollout; float32 may reach a roundoff plateau.",
    )
    (out / "ladder.json").write_text(json.dumps(result, indent=2))
    if not passed:
        raise AssertionError(f"Gradient ladder failed: {result}")
    return cast("Report", result)
