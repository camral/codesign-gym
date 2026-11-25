"""Shared helpers for public tools, collected in one module during release cleanup.

This temporary common owner contains host paths, the two existing serialization
policies, periodic control and reference-region labeling, generic optimization
and checkpoints, CPPN parameterization, and optional preflight/gradient checks.
Importing it does not load JAX, Warp, Newton, or experimental/diagnostic packages.
Numerical dependencies load only when their particular helper is called.

The implementations preserve their existing schemas, defaults and numerical
expressions. ``write_json`` maps nonfinite values to null; ``evidence_write_json``
retains explicit nonfinite markers. These policies intentionally remain distinct.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import time
import zipfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, NotRequired, Protocol, TypeAlias, TypedDict, cast

import numpy as np
from numpy.typing import ArrayLike, DTypeLike

from envs.soft_walker.contracts.fields import POISSON_BOUNDS, RawFieldLayout
from envs.soft_walker.engine.types import (
    Array,
    Embedding,
    HostArray,
    HostEmbedding,
    JsonValue,
    PathLike,
    PhysicsModel,
    Shape,
)

if TYPE_CHECKING:
    import jax

    from envs.soft_walker.engine.drivers.spring_environment import Spring2DEnv
    from envs.soft_walker.engine.embodiments.decoders import CPPNDecoder
    from envs.soft_walker.presets.scenarios.config import Scenario

# Derive the same installed/check-out paths from this module's new location.
SOFT_ROOT = Path(__file__).resolve().parent
DIAGNOSTICS_ROOT = SOFT_ROOT / "diagnostics"
SCRIPTS_ROOT = SOFT_ROOT / "scripts"
ENGINE_ROOT = SOFT_ROOT / "engine"
SRC_ROOT = SOFT_ROOT.parents[1]
REPO_ROOT = SRC_ROOT.parent
TESTS_ROOT = REPO_ROOT / "tests" / "soft_walker"

# Inherited legged reference-envelope tagging, independent of current pose.
# These are display/assessment conventions rather than solver tolerances.
REGION_BOUNDARY_TOLERANCE = 1e-5  # Normalized coordinates; inclusive sampled tags.
TORSO_REGION_START = 2.0 / 3.0  # Upper third of the fixed reference envelope.
FOOT_REGION_END = 1.0 / 6.0  # Lowest sixth of the reference envelope.


# diagnostics / _paths


def require_source_checkout(root: Path) -> None:
    """Reject repository runners before they create evidence in an installed layout."""
    required = (
        root / "pyproject.toml",
        root / "src/envs/soft_walker/__init__.py",
    )
    if not all(path.is_file() for path in required) or not (root / "tests/soft_walker").is_dir():
        raise ValueError(
            "This command requires a codesign-gym source checkout with "
            "src/envs/soft_walker and tests/soft_walker; a wheel installation is insufficient. "
            "Install the checkout with 'python -m pip install -e .' and run the command there. "
            f"Expected checkout: {root}"
        )


def diagnostic_sources(root: Path) -> list[Path]:
    """List diagnostic implementations and their extracted host contracts.

    Select only the package root (which holds the registered task: config, embodiment,
    vector_env, render) and the explicit diagnostic, residency-probe, Warp-check and
    host-contract packages. Historical results
    and media are never traversed or included as source.
    """
    package = root / "src/envs/soft_walker"
    return (
        sorted((package / "diagnostics").glob("*.py"))
        + sorted(package.glob("*.py"))
        + sorted((package / "contracts").glob("*.py"))
        + sorted((package / "diagnostics/residency").glob("*.py"))
        + sorted((package / "diagnostics/warp").glob("*.py"))
    )


def source_group_files(root: Path, group: str) -> list[Path]:
    """Preserve historical source groups, including relocated host dependencies."""
    if group == "src/envs/soft_walker/diagnostics":
        return diagnostic_sources(root)
    return sorted((root / group).rglob("*.py"))


# experiments / artifacts


def json_value(value: object) -> object:
    """Normalize arrays/dataclasses for experiment JSON; nonfinite floats become null."""
    if is_dataclass(value):
        # Legacy accepts dataclasses; class objects still fail at runtime.
        return json_value(asdict(value))  # type: ignore[arg-type]
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if isinstance(value, np.ndarray) or hasattr(value, "__array__"):
        return json_value(np.asarray(value).tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: PathLike, payload: object) -> None:
    """Atomically publish one complete JSON experiment record."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(json_value(payload), indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def write_npz(path: PathLike, **arrays: ArrayLike) -> None:
    """Atomically write compressed numeric arrays without a pickle payload."""
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **{k: np.asarray(v) for k, v in arrays.items()})
    os.replace(temporary, path)


def append_event(path: PathLike, event: str, **fields: object) -> None:
    """Append and flush one timestamped experiment event."""
    with Path(path).open("a", buffering=1) as handle:
        handle.write(
            json.dumps(
                json_value({"event": event, "wall_time_unix": time.time(), **fields}),
                allow_nan=False,
            )
            + "\n"
        )


def source_hashes(root: PathLike) -> dict[str, str]:
    """Hash first-party Python sources for an experiment provenance record."""
    root = Path(root)
    files: list[Path] = []
    for folder in ("src", "tests", "scripts", "examples"):
        files.extend((root / folder).rglob("*.py"))
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(files)
        if "__pycache__" not in p.parts
    }


def environment_record(root: PathLike) -> dict[str, object]:
    """Record package versions, devices, and an allowlist of runtime settings."""
    import jax

    from envs.soft_walker.engine.spaces import USING_GYMNASIUM

    packages: dict[str, str | None] = {}
    for name in ("jax", "jaxlib", "numpy", "scipy", "gymnasium", "Pillow", "diffrax"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "source_sha256": source_hashes(root),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "devices": [str(d) for d in jax.devices()],
        "x64_enabled": jax.config.x64_enabled,
        "real_gymnasium": USING_GYMNASIUM,
        "selected_environment": {
            k: os.environ.get(k)
            for k in (
                "JAX_PLATFORMS",
                "JAX_ENABLE_X64",
                "CUDA_VISIBLE_DEVICES",
                "JAX_COMPILATION_CACHE_DIR",
                "XLA_FLAGS",
                "CODESIGN_OFFLINE_SPACES",
            )
        },
    }


def archive_run(folder: PathLike) -> Path:
    """Create the archive next to, never inside, its source folder."""
    folder = Path(folder)
    hashes = {
        str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(folder.rglob("*"))
        if p.is_file() and p.name != "SHA256SUMS" and not p.name.endswith(".tmp")
    }
    (folder / "SHA256SUMS").write_text("".join(f"{h}  {p}\n" for p, h in hashes.items()))
    target = folder.with_name(folder.name + ".zip")
    temporary = target.with_name(target.name + ".tmp")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as output:
        for path in sorted(folder.rglob("*")):
            if path.is_file() and not path.name.endswith(".tmp"):
                output.write(path, folder.name + "/" + str(path.relative_to(folder)))
    os.replace(temporary, target)
    return target


# diagnostics / evidence


def evidence_jsonable(x: object) -> JsonValue:
    """Convert evidence values to JSON while retaining explicit nonfinite markers."""
    if isinstance(x, dict):
        return {str(k): evidence_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [evidence_jsonable(v) for v in x]
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, np.ndarray):
        return evidence_jsonable(x.tolist())
    if isinstance(x, np.generic):
        return evidence_jsonable(x.item())
    if isinstance(x, float) and not np.isfinite(x):
        return {"nonfinite": str(x)}
    if hasattr(x, "__array__"):
        return evidence_jsonable(np.asarray(x))
    if x is None or isinstance(x, (str, int, float, bool)):
        return x
    return str(x)


def evidence_write_json(path: PathLike, value: object) -> None:
    """Atomically replace an evidence document after serializing its complete contents."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(evidence_jsonable(value), indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def evidence_record(event: str, **data: object) -> None:
    """Append a timestamped event only when an evidence directory is configured."""
    root = os.environ.get("CODESIGN_EVIDENCE_DIR")
    if not root:
        return
    path = Path(root) / "events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "event": event,
        "utc": datetime.now(timezone.utc).isoformat(),
        "monotonic_s": time.perf_counter(),
        **data,
    }
    with path.open("a") as f:
        f.write(json.dumps(evidence_jsonable(row), allow_nan=False) + "\n")
        f.flush()


def evidence_sha256(path: PathLike) -> str:
    """Hash a file in bounded-memory chunks for source/evidence provenance."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# experiments / controllers


def periodic_action(parameters: jax.Array, time: jax.Array, period: float) -> jax.Array:
    """(channels, 1 + 2*harmonics): bias, then sine/cosine coefficients.

    Exactly one final tanh bounds the command. Time is the start of a control
    interval; the environment holds that command for the entire interval.
    """

    import jax.numpy as jnp

    if parameters.ndim != 2 or parameters.shape[1] < 3 or parameters.shape[1] % 2 != 1:
        raise ValueError("Expected (channels, 1 + 2*harmonics) coefficients")
    harmonics = (parameters.shape[1] - 1) // 2
    phase = 2.0 * jnp.pi * time / period * jnp.arange(1, harmonics + 1)
    terms = parameters[:, 1:].reshape(parameters.shape[0], harmonics, 2)
    value = parameters[:, 0] + jnp.sum(
        terms[..., 0] * jnp.sin(phase) + terms[..., 1] * jnp.cos(phase), axis=-1
    )
    return jnp.tanh(value)


# experiments / optimization / bounded


class InvalidEvaluation(RuntimeError):
    """A physical evaluation or adjoint was invalid, so no update may use it."""

    pass


@dataclass
class OptimizationResult:
    """Best host parameter copy and accounting from a bounded optimization run.

    Evaluations includes line-search trials; iterations is the last absolute
    iteration index, including any resumed history. Completion and message
    describe why the run stopped, independently of the best loss achieved.
    """

    parameters: np.ndarray
    initial_loss: float
    best_loss: float
    evaluations: int
    iterations: int
    completed: bool
    message: str


class OptimizationCallback(Protocol):
    """Receive a defensive controller copy after an optimizer iteration."""

    def __call__(
        self, *, iteration: int, parameters: HostArray, loss: float, accepted: bool, backtracks: int
    ) -> None: ...


def optimize(
    value_and_grad: Callable[[HostArray], tuple[float | Array, Array]],
    initial: HostArray,
    out: PathLike,
    *,
    method: str = "adam",
    iterations: int = 20,
    learning_rate: float = 0.05,
    bound: float = 3.0,
    max_backtracks: int = 6,
    resume: PathLike | None = None,
    on_iteration: OptimizationCallback | None = None,
) -> OptimizationResult:
    """value_and_grad(parameters) -> (scalar, same-shaped gradient).

    The callback is responsible for compiling and synchronizing its physical
    computation. This routine additionally materializes scalar/gradient outputs
    before timing and validation. Every evaluated candidate is logged, including
    invalid line-search trials. The best checkpoint is always written first.

    Adam/GD optionally call on_iteration with keyword arguments iteration,
    parameters, loss, accepted, and backtracks after writing the iteration log
    and last checkpoint. The index is absolute when resuming; parameters is a
    defensive copy of the current controller, including on a rejected step.
    """
    if method not in ("adam", "gd", "lbfgs") or iterations < 0:
        raise ValueError("Invalid optimization method or iteration count")
    if on_iteration is not None and method == "lbfgs":
        raise ValueError("on_iteration is supported only for Adam and GD")
    if not 0 < learning_rate or not 0 < bound:
        raise ValueError("learning_rate and bound must be positive")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    parameters = np.asarray(initial).copy()
    shape, dtype = parameters.shape, parameters.dtype
    if not np.isfinite(parameters).all() or np.any(np.abs(parameters) > bound):
        raise ValueError("Initial coefficients must be finite and within bounds")
    m, v = np.zeros_like(parameters), np.zeros_like(parameters)
    start_iteration = 0
    if resume:
        with np.load(resume, allow_pickle=False) as saved:
            if saved["parameters"].shape != shape:
                raise ValueError("Resume parameter shape mismatch")
            parameters = saved["parameters"].astype(dtype)
            m, v = saved["m"].astype(dtype), saved["v"].astype(dtype)
            start_iteration = int(saved["iteration"])
    counter = 0
    best_loss = np.inf
    best = parameters.copy()
    current_iteration = start_iteration
    log = out / "optimization.jsonl"

    def evaluate(p: HostArray) -> tuple[float, HostArray]:
        """Synchronize one objective evaluation and maintain its best checkpoint."""
        nonlocal counter, best_loss, best
        t0 = time.perf_counter()
        loss, gradient = value_and_grad(np.asarray(p, dtype=dtype).reshape(shape))
        loss, gradient = float(loss), np.asarray(gradient, dtype=dtype).reshape(shape)
        counter += 1
        valid = np.isfinite(loss) and np.isfinite(gradient).all()
        append_event(
            log,
            "evaluation",
            evaluation=counter,
            iteration=current_iteration,
            loss=loss,
            gradient_l2=np.linalg.norm(gradient),
            value_and_gradient_s=time.perf_counter() - t0,
            valid=valid,
            parameters=np.asarray(p).reshape(shape),
            gradient=gradient,
        )
        if not valid:
            raise InvalidEvaluation("Nonfinite objective/adjoint; see saved trial")
        if loss < best_loss:
            best_loss, best = loss, np.asarray(p, dtype=dtype).reshape(shape).copy()
            write_npz(out / "best.npz", parameters=best, loss=best_loss)
        return loss, gradient

    initial_loss, gradient = evaluate(parameters)
    loss = initial_loss
    completed, message = True, "requested iterations completed"
    try:
        if method == "lbfgs":
            from scipy.optimize import minimize

            def scipy_fun(flat: HostArray) -> tuple[float, HostArray]:
                """Adapt flattened SciPy coordinates to the original parameter shape."""
                f, g = evaluate(flat)
                return f, g.astype(np.float64).ravel()

            def callback(flat: HostArray) -> None:
                """Record completion of a SciPy iteration."""
                nonlocal current_iteration
                current_iteration += 1
                write_npz(
                    out / "last.npz",
                    parameters=np.asarray(flat).reshape(shape),
                    m=m,
                    v=v,
                    iteration=current_iteration,
                )

            result = minimize(
                scipy_fun,
                parameters.astype(np.float64).ravel(),
                jac=True,
                method="L-BFGS-B",
                bounds=[(-bound, bound)] * parameters.size,
                callback=callback,
                options={
                    "maxiter": iterations,
                    "maxfun": max(10, iterations * 12),
                    "maxls": 10,
                    "ftol": 1e-10,
                    "gtol": 1e-7,
                },
            )
            message = str(result.message)
            completed = bool(result.success or result.nit >= iterations)
        else:
            for current_iteration in range(start_iteration + 1, start_iteration + iterations + 1):
                if method == "adam":
                    candidate_m = 0.9 * m + 0.1 * gradient
                    candidate_v = 0.999 * v + 0.001 * gradient**2
                    direction = (candidate_m / (1 - 0.9**current_iteration)) / (
                        np.sqrt(candidate_v / (1 - 0.999**current_iteration)) + 1e-8
                    )
                else:
                    candidate_m, candidate_v = m, v
                    direction = gradient
                accepted = False
                for trial in range(max_backtracks):
                    candidate = np.clip(
                        parameters - learning_rate * 0.5**trial * direction, -bound, bound
                    ).astype(dtype)
                    try:
                        new_loss, new_gradient = evaluate(candidate)
                    except InvalidEvaluation:
                        continue
                    if new_loss <= loss:
                        parameters, loss, gradient = candidate, new_loss, new_gradient
                        m, v = candidate_m, candidate_v
                        accepted = True
                        break
                append_event(
                    log,
                    "iteration",
                    iteration=current_iteration,
                    accepted=accepted,
                    loss=loss,
                    backtracks=trial,
                )
                write_npz(
                    out / "last.npz",
                    parameters=parameters,
                    m=m,
                    v=v,
                    iteration=current_iteration,
                    loss=loss,
                )
                if on_iteration is not None:
                    on_iteration(
                        iteration=current_iteration,
                        parameters=parameters.copy(),
                        loss=loss,
                        accepted=accepted,
                        backtracks=trial,
                    )
                if not accepted:
                    completed = False
                    message = "No valid improving step within bounded line search"
                    break
    except InvalidEvaluation as error:
        completed, message = False, str(error)
    report = OptimizationResult(
        best, initial_loss, best_loss, counter, current_iteration, completed, message
    )
    write_json(out / "optimization_summary.json", report)
    return report


Objective: TypeAlias = "Callable[[HostArray], tuple[float | Array, Array]]"


# experiments / optimization / checkpoint


class AdamCheckpoint(TypedDict):
    """Validated in-memory state recovered from the version-1 NPZ format."""

    parameters: HostArray
    gradient: HostArray
    m: HostArray
    v: HostArray
    best: HostArray
    iteration: int
    initial_loss: float
    best_loss: float
    loss: float
    best_iteration: int
    evaluations: int
    rejected_trials: int
    objective_seconds: float
    elapsed_before: float


class AdamCallback(Protocol):
    """Observer called after checkpointing; arrays are defensive copies."""

    def __call__(
        self,
        *,
        iteration: int,
        parameters: HostArray,
        loss: float,
        gradient: HostArray,
        timings: dict[str, float | int],
    ) -> None: ...


def load_checkpoint(
    path: PathLike, shape: Shape, dtype: np.dtype, learning_rate: float, bound: float
) -> AdamCheckpoint:
    """Validate an NPZ state and return owned arrays; reject incompatible resumes.

    Shape, precision, optimizer settings, finite moments, and monotone counters
    must match. The caller remains responsible for using the same objective.
    """
    with np.load(path, allow_pickle=False) as saved:
        if int(saved["format_version"]) != 1:
            raise ValueError("Unsupported Adam checkpoint version")
        if float(saved["learning_rate"]) != learning_rate or float(saved["bound"]) != bound:
            raise ValueError("Resume learning rate/bound mismatch")
        arrays = {}
        for key in ("parameters", "gradient", "m", "v", "best_parameters"):
            array = saved[key]
            if array.shape != shape or array.dtype != dtype or not np.isfinite(array).all():
                raise ValueError(f"Resume {key} shape, dtype, or finiteness mismatch")
            arrays[key] = array.copy()
        parameters, gradient, m, v, best = (
            arrays[k] for k in ("parameters", "gradient", "m", "v", "best_parameters")
        )
        if np.any(np.abs(parameters) > bound) or np.any(np.abs(best) > bound) or np.any(v < 0):
            raise ValueError("Resume coefficients or second moment outside valid bounds")
        iteration = int(saved["iteration"])
        if int(saved["t"]) != iteration or iteration < 0:
            raise ValueError("Resume Adam update counter mismatch")
        initial_loss, best_loss, loss = (
            float(saved[k]) for k in ("initial_loss", "best_loss", "loss")
        )
        if not np.isfinite([initial_loss, best_loss, loss]).all() or best_loss > loss:
            raise ValueError("Invalid resume objective history")
        best_iteration = int(saved["best_iteration"])
        evaluations = int(saved["evaluations"])
        rejected_trials = int(saved["rejected_trials"])
        objective_seconds = float(saved["objective_seconds"])
        elapsed_before = float(saved["elapsed_seconds"])
        if (
            not 0 <= best_iteration <= iteration
            or evaluations < iteration + 1
            or rejected_trials < 0
        ):
            raise ValueError("Invalid resume counters")

    return {
        "parameters": parameters,
        "gradient": gradient,
        "m": m,
        "v": v,
        "best": best,
        "iteration": iteration,
        "initial_loss": initial_loss,
        "best_loss": best_loss,
        "loss": loss,
        "best_iteration": best_iteration,
        "evaluations": evaluations,
        "rejected_trials": rejected_trials,
        "objective_seconds": objective_seconds,
        "elapsed_before": elapsed_before,
    }


def evaluate_objective(
    value_and_grad: Objective, candidate: HostArray, dtype: np.dtype, shape: Shape
) -> tuple[float | None, HostArray | None, str | None]:
    """Synchronize and validate one objective call without changing optimizer state.

    Solver/adjoint invalidity is returned as a retryable error. Programming
    errors such as wrong shapes propagate immediately. The input is copied.
    """
    new_loss: float | None = None
    new_gradient: HostArray | None = None
    error: str | None = None
    try:
        raw_loss, raw_gradient = value_and_grad(candidate.copy())
        loss_array = np.asarray(raw_loss)
        new_gradient = np.asarray(raw_gradient, dtype=dtype).copy()
        if loss_array.shape != () or new_gradient.shape != shape:
            raise ValueError("Objective must return a scalar and a same-shaped gradient")
        new_loss = float(loss_array)
        if not np.isfinite(new_loss) or not np.isfinite(new_gradient).all():
            error = "Nonfinite objective or adjoint"
    except (InvalidEvaluation, FloatingPointError) as exc:
        error = str(exc) or type(exc).__name__
    return new_loss, new_gradient, error


def save_checkpoint(
    out: Path,
    checkpoints: Path,
    state: Mapping[str, Array | int | float],
    iteration: int,
    best: HostArray,
    best_loss: float,
    best_iteration: int,
) -> None:
    """Write numbered/last states and a compact best replay artifact as before."""
    write_npz(checkpoints / f"iteration_{iteration:05d}.npz", **state)
    write_npz(out / "last.npz", **state)
    # best.npz is a replay artifact; resume from last.npz or a numbered state.
    write_npz(out / "best.npz", parameters=best, loss=best_loss, iteration=best_iteration)


# experiments / optimization / adam


def run_adam(
    value_and_grad: Objective,
    initial: HostArray,
    out: PathLike,
    *,
    iterations: int = 50,
    learning_rate: float = 0.03,
    bound: float = 6.0,
    on_iteration: AdamCallback | None = None,
    resume: PathLike | None = None,
) -> dict[str, object]:
    """Run ``iterations`` additional accepted Adam updates and save each state.

    ``value_and_grad(parameters)`` returns a scalar and an identically shaped
    gradient. Materializing both outputs synchronizes device computation before
    recording its duration. A valid initial evaluation plus 50 ordinary updates
    therefore needs 51 evaluations. Up to eight candidate evaluations are tried
    per update, halving that update's step only when physics/adjoints are invalid.
    Exhausting these attempts writes an incomplete summary and raises
    ``InvalidEvaluation`` without advancing Adam's moments or update counter.

    The callback receives keyword arguments ``iteration``, ``parameters``,
    ``loss``, ``gradient``, and ``timings`` after the starting checkpoint and each
    accepted update. Arrays are defensive copies. On resume the starting index
    is the saved absolute index, and its saved gradient is reused exactly.

    Every checkpoint contains Adam state, gradient, counters, best state, and
    hyperparameters. Resume requires matching shape, dtype and hyperparameters;
    the caller must also keep the objective and its physical protocol unchanged.
    The returned dictionary's ``parameters`` is the best evaluated state, while
    ``final_parameters`` is the last accepted state, which can have higher loss.
    JSON summaries contain scalars and checkpoint paths rather than large arrays.
    """
    if (
        isinstance(iterations, bool)
        or not isinstance(iterations, (int, np.integer))
        or iterations < 0
    ):
        raise ValueError("iterations must be a nonnegative integer")
    if not np.isfinite(learning_rate) or learning_rate <= 0 or not np.isfinite(bound) or bound <= 0:
        raise ValueError("learning_rate and bound must be finite and positive")
    parameters = np.asarray(initial).copy()
    if parameters.dtype.kind != "f" or parameters.size == 0:
        raise ValueError("Initial parameters must be a nonempty floating-point array")
    if not np.isfinite(parameters).all() or np.any(np.abs(parameters) > bound):
        raise ValueError("Initial coefficients must be finite and within bounds")
    shape, dtype = parameters.shape, parameters.dtype
    out = Path(out)
    if resume is None and (out / "last.npz").exists():
        raise FileExistsError("Existing Adam checkpoint: use resume or a fresh output directory")
    out.mkdir(parents=True, exist_ok=True)
    checkpoints = out / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    log = out / "optimization.jsonl"
    started = time.perf_counter()
    elapsed_before = objective_seconds = 0.0
    evaluations = rejected_trials = iteration = best_iteration = 0
    initial_loss = best_loss = loss = None
    best = parameters.copy()
    m, v = np.zeros_like(parameters), np.zeros_like(parameters)
    gradient = None

    if resume is not None:
        saved_state = load_checkpoint(resume, shape, dtype, learning_rate, bound)
        parameters = saved_state["parameters"]
        gradient = saved_state["gradient"]
        m = saved_state["m"]
        v = saved_state["v"]
        best = saved_state["best"]
        iteration = saved_state["iteration"]
        initial_loss = saved_state["initial_loss"]
        best_loss = saved_state["best_loss"]
        loss = saved_state["loss"]
        best_iteration = saved_state["best_iteration"]
        evaluations = saved_state["evaluations"]
        rejected_trials = saved_state["rejected_trials"]
        objective_seconds = saved_state["objective_seconds"]
        elapsed_before = saved_state["elapsed_before"]
    start_iteration, start_evaluations = iteration, evaluations
    target_iteration = start_iteration + int(iterations)

    def elapsed() -> float:
        """Total active elapsed seconds including resumed sessions."""
        return elapsed_before + time.perf_counter() - started

    def summary(completed: bool, message: str) -> dict[str, object]:
        """Persist scalar progress without embedding parameter arrays in JSON."""
        report = dict(
            completed=completed,
            message=message,
            initial_loss=initial_loss,
            best_loss=best_loss,
            final_loss=loss,
            best_iteration=best_iteration,
            iterations=iteration,
            accepted_updates=iteration,
            session_accepted_updates=iteration - start_iteration,
            requested_updates=int(iterations),
            start_iteration=start_iteration,
            target_iteration=target_iteration,
            evaluations=evaluations,
            session_evaluations=evaluations - start_evaluations,
            rejected_trials=rejected_trials,
            objective_seconds=objective_seconds,
            elapsed_seconds=elapsed(),
            learning_rate=learning_rate,
            bound=bound,
            best_checkpoint="best.npz" if best_loss is not None else None,
            last_checkpoint="last.npz" if loss is not None else None,
        )
        write_json(out / "optimization_summary.json", report)
        return report

    def evaluate(
        candidate: HostArray, index: int, backtracks: int
    ) -> tuple[float | None, HostArray | None, float, str | None]:
        """Record one synchronized objective evaluation and retryable failure."""
        nonlocal evaluations, rejected_trials, objective_seconds
        t0 = time.perf_counter()
        evaluations += 1
        new_loss, new_gradient, error = evaluate_objective(value_and_grad, candidate, dtype, shape)
        duration = time.perf_counter() - t0
        objective_seconds += duration
        valid = error is None
        if not valid:
            rejected_trials += 1
        with np.errstate(over="ignore", invalid="ignore"):
            gradient_l2 = (
                float(np.linalg.norm(cast(HostArray, new_gradient).astype(np.float64)))
                if valid
                else None
            )
        append_event(
            log,
            "evaluation",
            evaluation=evaluations,
            iteration=index,
            backtracks=backtracks,
            valid=valid,
            loss=new_loss,
            error=error,
            gradient_l2=gradient_l2,
            value_and_gradient_s=duration,
        )
        return new_loss, new_gradient, duration, error

    def save_state() -> None:
        """Persist the last accepted optimizer state without mutating its arrays."""
        state = dict(
            format_version=1,
            parameters=parameters,
            gradient=gradient,
            m=m,
            v=v,
            t=iteration,
            iteration=iteration,
            loss=loss,
            initial_loss=initial_loss,
            best_loss=best_loss,
            best_parameters=best,
            best_iteration=best_iteration,
            evaluations=evaluations,
            rejected_trials=rejected_trials,
            learning_rate=learning_rate,
            bound=bound,
            objective_seconds=objective_seconds,
            elapsed_seconds=elapsed(),
        )
        save_checkpoint(
            out, checkpoints, state, iteration, best, cast(float, best_loss), best_iteration
        )

    def notify(timings: dict[str, float | int]) -> None:
        """Publish defensive array copies after each accepted checkpoint."""
        if on_iteration is not None:
            on_iteration(
                iteration=iteration,
                parameters=parameters.copy(),
                loss=cast(float, loss),
                gradient=cast(HostArray, gradient).copy(),
                timings=dict(timings),
            )

    initial_seconds = 0.0
    if resume is None:
        loss, gradient, initial_seconds, error = evaluate(parameters, iteration, 0)
        if error is not None:
            # No invalid initial state is advertised as a usable checkpoint.
            loss = None
            summary(False, f"Invalid initial evaluation: {error}")
            raise InvalidEvaluation(f"Invalid initial evaluation: {error}")
        initial_loss = best_loss = loss
    # Both fresh and resumed paths have validated complete finite state here.
    gradient = cast(HostArray, gradient)
    loss = cast(float, loss)
    best_loss = cast(float, best_loss)
    save_state()
    append_event(
        log,
        "start",
        iteration=iteration,
        resumed=resume is not None,
        loss=loss,
        target_iteration=target_iteration,
    )
    summary(False, "running")
    notify(
        dict(
            value_and_gradient_s=initial_seconds,
            update_s=initial_seconds,
            backtracks=0,
            evaluations=evaluations,
            elapsed_seconds=elapsed(),
        )
    )

    for next_iteration in range(start_iteration + 1, target_iteration + 1):
        update_started = time.perf_counter()
        with np.errstate(over="ignore", invalid="ignore"):
            candidate_m = (0.9 * m + 0.1 * gradient).astype(dtype)
            candidate_v = (0.999 * v + 0.001 * gradient**2).astype(dtype)
            direction = (candidate_m / (1 - 0.9**next_iteration)) / (
                np.sqrt(candidate_v / (1 - 0.999**next_iteration)) + 1e-8
            )
        if not all(np.isfinite(a).all() for a in (candidate_m, candidate_v, direction)):
            message = f"Update {next_iteration} failed: nonfinite Adam moment or direction"
            summary(False, message)
            raise InvalidEvaluation(message)
        update_objective_seconds = 0.0
        for backtracks in range(8):
            candidate = np.clip(
                parameters - learning_rate * 0.5**backtracks * direction, -bound, bound
            ).astype(dtype)
            new_loss, new_gradient, duration, error = evaluate(
                candidate, next_iteration, backtracks
            )
            update_objective_seconds += duration
            if error is None:
                break
        else:
            message = f"Update {next_iteration} failed: eight invalid candidate evaluations; last error: {error}"
            # Include rejected evaluation counts in the last valid state.
            save_state()
            summary(False, message)
            raise InvalidEvaluation(message)
        parameters, loss, gradient = candidate, cast(float, new_loss), cast(HostArray, new_gradient)
        m, v, iteration = candidate_m, candidate_v, next_iteration
        if loss < best_loss:
            best, best_loss, best_iteration = parameters.copy(), loss, iteration
        timings = dict(
            value_and_gradient_s=update_objective_seconds,
            update_s=time.perf_counter() - update_started,
            backtracks=backtracks,
            evaluations=evaluations,
            elapsed_seconds=elapsed(),
        )
        save_state()
        append_event(
            log,
            "iteration",
            iteration=iteration,
            accepted=True,
            loss=loss,
            best_loss=best_loss,
            step_size=learning_rate * 0.5**backtracks,
            **timings,
        )
        summary(False, "running")
        notify(timings)
    report = summary(True, "requested accepted Adam updates completed")
    return dict(report, parameters=best.copy(), final_parameters=parameters.copy())


# experiments / embodiments


def _output_groups(dim: int, channels: int) -> dict[str, tuple[int, ...]]:
    """Identify continuous semantic heads while excluding discrete actuator routing."""
    layout = RawFieldLayout(dim, channels)
    return {
        "geometry": (layout.occupancy,),
        "young": (layout.young,),
        "strength": (layout.strength,),
        "direction": tuple(range(layout.direction.start, layout.direction.stop)),
        "poisson": (layout.poisson,),
    }


def _bounded_delta(seed: jax.Array, delta: jax.Array) -> jax.Array:
    """Shift within [-6, 6], preserving the seed exactly at zero.

    This is the tanh addition identity written without an inverse tanh:
    6*tanh(atanh(seed/6) + delta). The final clip only removes roundoff at
    floating-point saturation; all finite interior updates are smooth.
    An exact endpoint seed remains fixed; the safe denominator avoids 0/0.
    """

    import jax.numpy as jnp

    t = jnp.tanh(delta)
    endpoint = jnp.abs(seed) == 6.0
    denominator = jnp.where(endpoint, 1.0, 6.0 + seed * t)
    value = seed + (36.0 - seed * seed) * t / denominator
    return jnp.clip(value, -6.0, 6.0)


class CPPNParameterization:
    """Pack continuous CPPN output weights and biases into Adam coordinates.

    ``initial`` is a NumPy zero vector. ``unpack(theta)`` is a pure JAX function
    that returns a driver-compatible embedding with a trainable spatial
    Poisson output head. ``groups`` maps semantic names to Python slices;
    ``metadata`` is JSON serializable. No controller parameters are included.

    Selected output heads may read inputs and hidden nodes only. Connections
    between semantic outputs are rejected because they could change frozen
    routing or density indirectly. Disabled connections remain disabled.
    """

    def __init__(self, decoder: CPPNDecoder, embedding: Embedding) -> None:
        """Pack a valid relaxed CPPN seed into continuous Adam coordinates.

        The seed is copied. Incompatible decoders, invalid coefficients, or output-head
        dependencies
        raise ValueError before any optimization state is created.
        """

        from envs.soft_walker.engine.embodiments.decoders import (
            CPPNDecoder,
            check_embedding,
            set_poisson,
        )

        if not isinstance(decoder, CPPNDecoder):
            raise ValueError("CPPNParameterization requires a CPPNDecoder")
        if decoder.spec.design_mode != "relaxed" or decoder.spec.material_mode != "continuous":
            raise ValueError("co-design requires relaxed occupancy and continuous material")
        base = {k: np.array(v, copy=True) for k, v in embedding.items()}
        if "poisson" in base:
            # Migrate an old global override into the material output head so
            # the new trainable spatial coordinates actually affect physics.
            legacy = base.pop("poisson")
            if (
                legacy.shape != ()
                or not np.isfinite(legacy)
                or not POISSON_BOUNDS[0] <= float(legacy) <= POISSON_BOUNDS[1]
            ):
                raise ValueError("poisson must be a scalar in [0.25, 0.45]")
            base = set_poisson(decoder, base, float(legacy))
        errors = check_embedding(decoder, {k: v for k, v in base.items() if k != "poisson"})
        if errors:
            raise ValueError(f"invalid CPPN seed: {errors}")
        prefix = decoder.inputs + decoder.hidden
        if np.any(base["edge_mask"][:, prefix:]):
            raise ValueError("semantic output-to-output edges are not supported")

        self.decoder = decoder
        self.hard_decoder = CPPNDecoder(
            replace(decoder.spec, design_mode="hard"), hidden=decoder.hidden, dtype=decoder.dtype
        )
        self._base = base
        self._rows = {}
        self.groups = {}
        offset = 0
        details = {}
        for name, outputs in _output_groups(decoder.spec.dim, decoder.spec.channels).items():
            begin = offset
            rows = []
            for output in outputs:
                row = decoder.hidden + output
                columns = np.flatnonzero(base["edge_mask"][row, :prefix])
                seed_values = np.concatenate(
                    (base["weights"][row, columns], base["bias"][row : row + 1])
                )
                if np.any(np.abs(seed_values) > 6.0):
                    raise ValueError("trainable seed coordinates must be inside [-6, 6]")
                selection = slice(offset, offset + len(columns) + 1)
                self._rows[row] = (columns, selection, seed_values)
                offset = selection.stop
                rows.append(
                    {"output_index": output, "node_index": row, "weight_columns": columns.tolist()}
                )
            self.groups[name] = slice(begin, offset)
            details[name] = {"start": begin, "stop": offset, "rows": rows}
        self.initial = np.zeros(offset, dtype=decoder.dtype)
        self.metadata = {
            "representation": "cppn",
            "parameter_count": offset,
            "coordinates": "zero-centered bounded tanh shifts of CPPN output weights and biases",
            "weight_bounds": [-6.0, 6.0],
            "groups": details,
            "optimized_fields": ["occupancy", "young", "strength", "direction", "poisson"],
            "frozen_fields": ["density", "actuator_type", "channel"],
            "frozen_network": "hidden weights, hidden biases, and edge mask",
            "training_design_mode": "relaxed",
            "verification_design_mode": "hard",
            "occupancy_floor": float(decoder.spec.occupancy_floor),
            "geometry_temperature": float(decoder.spec.geometry_temperature),
            "hard_replay_required": True,
        }

    def unpack(self, theta: jax.Array) -> Embedding:
        """Map bounded optimization coordinates back to a decoder-compatible embedding."""

        import jax.numpy as jnp

        theta = jnp.asarray(theta, dtype=self.decoder.dtype)
        if theta.shape != self.initial.shape:
            raise ValueError(
                f"expected morphology coordinates {self.initial.shape}; got {theta.shape}"
            )
        weights = jnp.asarray(self._base["weights"])
        bias = jnp.asarray(self._base["bias"])
        weight_rows, bias_rows = [], []
        # Static concatenation keeps this reconstruction free of dynamic writes.
        for row in range(self.decoder.nodes):
            if row not in self._rows:
                weight_rows.append(weights[row])
                bias_rows.append(bias[row])
                continue
            columns, selection, seed_values = self._rows[row]
            values = _bounded_delta(jnp.asarray(seed_values), theta[selection])
            if np.array_equal(columns, np.arange(len(columns))):
                # The supplied random prior trains a contiguous input/hidden
                # prefix, so reconstruct it with two slices instead of hundreds
                # of scalar operations in an already large physics JAXPR.
                weight_rows.append(jnp.concatenate((values[:-1], weights[row, len(columns) :])))
            else:
                lookup = {int(column): k for k, column in enumerate(columns)}
                weight_rows.append(
                    jnp.stack(
                        [
                            values[lookup[column]] if column in lookup else weights[row, column]
                            for column in range(weights.shape[1])
                        ]
                    )
                )
            bias_rows.append(values[-1])
        result = {
            "weights": jnp.stack(weight_rows),
            "bias": jnp.stack(bias_rows),
            "edge_mask": jnp.asarray(self._base["edge_mask"]),
        }
        return result


def make_implicit_design(
    dim: int,
    seed: int = 7,
    dtype: DTypeLike = "float64",
    grid_shape: Shape | None = None,
    *,
    channels: int = 4,
    geometry_temperature: float = 0.3,
    poisson: float | None = None,
) -> tuple[CPPNDecoder, HostEmbedding, dict[str, object]]:
    """Return ``(relaxed_decoder, embedding, metadata)`` for a random CPPN body.

    Run once during CPU staging, then serialize the result for a CUDA-only
    worker. Sampling checks the *hard* body at cells and MPM's 2x particle sites.
    Added zero-weight edges expose continuous output-head training coordinates
    without changing any decoded seed field. No dynamics are run here.
    """

    from envs.soft_walker.engine.embodiments.decoders import CPPNDecoder, set_poisson
    from envs.soft_walker.engine.embodiments.morphology import generate_random_design

    grid_shape = tuple(grid_shape or ((6, 6) if dim == 2 else (4, 4, 4)))
    if not np.isfinite(geometry_temperature) or geometry_temperature <= 0:
        raise ValueError("geometry_temperature must be positive and finite")
    if poisson is not None and (not np.isfinite(poisson) or not 0.25 <= poisson <= 0.45):
        raise ValueError("poisson must be in [0.25, 0.45]")
    hard, embedding, report = generate_random_design(
        "cppn", dim=dim, seed=seed, dtype=dtype, grid_shape=grid_shape, channels=channels
    )
    hard = cast(CPPNDecoder, hard)  # The generator was explicitly requested as cppn.
    embedding = {k: np.array(v, copy=True) for k, v in embedding.items()}
    if poisson is not None:
        embedding = set_poisson(hard, embedding, poisson)
    for outputs in _output_groups(dim, channels).values():
        for output in outputs:
            embedding["edge_mask"][hard.hidden + output, : hard.inputs + hard.hidden] = 1
    decoder = CPPNDecoder(
        replace(hard.spec, design_mode="relaxed", geometry_temperature=geometry_temperature),
        hidden=hard.hidden,
        dtype=dtype,
    )
    parameterization = CPPNParameterization(decoder, embedding)
    metadata: dict[str, object] = {
        "seed_generation": report,
        "grid_shape": list(grid_shape),
        "decoder": decoder.descriptor(),
        "parameterization": parameterization.metadata,
        "note": "The random hard seed passed sampled topology checks. Relaxed training retains floor-density material outside its hard silhouette; hard final replay must be assessed separately.",
    }
    return decoder, embedding, metadata


# diagnostics / gradient_checks


def difference_ladder(
    value: Callable[[HostArray], float | Array],
    value_and_grad: Callable[[HostArray], tuple[float | Array, Array]],
    parameters: Array,
    *,
    seed: int = 7,
    steps: Sequence[float] | None = None,
) -> dict[str, object]:
    """Return all samples; do not select a flattering epsilon and discard others.

    The result separates observed convergence from best-point agreement. A
    missing O(h^2) window, failure, or roundoff plateau remains visible.
    """
    p = np.asarray(parameters)
    rng = np.random.default_rng(seed)
    direction = rng.normal(size=p.shape)
    direction /= np.linalg.norm(direction)
    direction = direction.astype(p.dtype)
    step_sizes: HostArray = np.asarray(
        steps
        if steps is not None
        else (0.2 * 0.5 ** np.arange(14) if p.dtype == np.float64 else 0.3 * 0.5 ** np.arange(10)),
        dtype=float,
    )
    start = time.perf_counter()
    f0, gradient = value_and_grad(p)
    f0, gradient = float(f0), np.asarray(gradient)
    ad = float(np.vdot(gradient, direction))
    if not np.isfinite(f0) or not np.isfinite(gradient).all():
        raise ValueError("Cannot audit a nonfinite baseline/adjoint")
    rows = []
    for h in step_sizes:
        t = time.perf_counter()
        fp = float(value((p + h * direction).astype(p.dtype)))
        fm = float(value((p - h * direction).astype(p.dtype)))
        fd = (fp - fm) / (2 * h)
        rows.append(
            {
                "h": float(h),
                "positive": fp,
                "negative": fm,
                "central_difference": fd,
                "absolute_error": abs(fd - ad),
                "relative_error": abs(fd - ad) / max(abs(ad), 1e-12),
                "forward_pair_s": time.perf_counter() - t,
                "taylor_remainder_positive": abs(fp - f0 - h * ad),
            }
        )
    for a, b in zip(rows[:-1], rows[1:]):
        a["observed_order_to_next"] = float(
            np.log(max(a["absolute_error"], 1e-300) / max(b["absolute_error"], 1e-300))
            / np.log(a["h"] / b["h"])
        )
    orders = [r.get("observed_order_to_next", np.nan) for r in rows]
    windows = [
        i for i in range(len(orders) - 1) if 1.5 <= orders[i] <= 2.5 and 1.5 <= orders[i + 1] <= 2.5
    ]
    tol = 1e-4 if p.dtype == np.float64 else 5e-3
    best = min(
        rows, key=lambda r: r["absolute_error"] if np.isfinite(r["absolute_error"]) else np.inf
    )
    return {
        "parameters": p,
        "direction": direction,
        "gradient": gradient,
        "baseline_loss": f0,
        "ad_directional": ad,
        "rows": rows,
        "second_order_windows": windows,
        "best": best,
        "agreement_passed": bool(
            best["relative_error"] <= tol or best["absolute_error"] <= tol * 1e-4
        ),
        "second_order_observed": bool(windows),
        "wall_s": time.perf_counter() - start,
        "qualification": "Branch crossings and roundoff are not automatically smooth convergence",
    }


class DifferenceSample(TypedDict):
    """One unfiltered finite-difference sample in an epsilon ladder."""

    h: float
    f_plus: float
    f_minus: float
    central: float
    autodiff: float
    abs_error: float
    relative_error: float
    observed_order: NotRequired[float | None]


class DifferenceReport(TypedDict):
    """Complete unfiltered finite-difference evidence and observed error minima."""

    name: str
    autodiff: float
    direction: HostArray
    theta: HostArray
    rows: list[DifferenceSample]
    best_relative_error: float
    best_absolute_error: float
    passed: NotRequired[bool]
    relative_threshold: NotRequired[float]
    absolute_threshold: NotRequired[float]
    second_order_window: NotRequired[bool]
    observed_second_order_window: NotRequired[bool]
    order_required: NotRequired[bool]
    roundoff_plateau: NotRequired[bool]
    cross_precision_required: NotRequired[bool]
    absolute_floor_used: NotRequired[bool]
    nontrivial_direction: NotRequired[bool]


def finite_difference_ladder(
    fn: Callable[[Array], float | Array],
    theta: Array,
    direction: Array,
    *,
    steps: Iterable[float],
    autodiff: float | Array,
    name: str,
) -> DifferenceReport:
    """Raw values retained; central error slopes use adjacent epsilon ratios.

    Tiny-epsilon roundoff is expected. Callers must check an accuracy threshold
    AND inspect a truncation-dominated window; minimizing the error alone is not
    evidence of second-order convergence. Direction is not renormalized here.
    """
    rows: list[DifferenceSample] = []
    for h in steps:
        plus = float(fn(theta + h * direction))
        minus = float(fn(theta - h * direction))
        fd = (plus - minus) / (2 * h)
        err = abs(fd - float(autodiff))
        rows.append(
            {
                "h": float(h),
                "f_plus": plus,
                "f_minus": minus,
                "central": fd,
                "autodiff": float(autodiff),
                "abs_error": err,
                "relative_error": err / max(abs(float(autodiff)), 1e-12),
            }
        )
    for i, r in enumerate(rows):
        r["observed_order"] = None
        if i and r["abs_error"] > 0 and rows[i - 1]["abs_error"] > 0:
            r["observed_order"] = float(
                np.log(rows[i - 1]["abs_error"] / r["abs_error"])
                / np.log(rows[i - 1]["h"] / r["h"])
            )
    result: DifferenceReport = {
        "name": name,
        "autodiff": float(autodiff),
        "direction": np.asarray(direction),
        "theta": np.asarray(theta),
        "rows": rows,
        "best_relative_error": min(r["relative_error"] for r in rows),
        "best_absolute_error": min(r["abs_error"] for r in rows),
    }
    evidence_record("central_difference_ladder", **result)
    return result


# diagnostics / scenario_preflight


def physical_budget(case: Scenario, *, realized_mass: float | None = None) -> dict[str, object]:
    """Linear axial-leg screening, NOT a claim of takeoff or actuator efficiency.

    U=K*delta^2/2 neglects nonlinear shape, contact, bending, damping and transfer
    losses. eta is a scenario-analysis assumption, not a model parameter. A
    strain-driven actuator does not necessarily store all of this energy.
    """
    legs = 2 if case.dim == 2 else 4
    length, width, depth, torso_height = 0.08, 0.04, 0.04, 0.04
    area = width * (0.01 if case.dim == 2 else depth)
    torso_volume = 0.16 * torso_height * (0.01 if case.dim == 2 else 0.12)
    mass = case.density * (torso_volume + legs * area * length)
    if realized_mass is not None:
        mass = float(realized_mass)
    omega = 2.0 * math.pi / case.period
    attenuation = 1.0 / math.sqrt(1.0 + (omega * case.actuator_tau) ** 2)
    strain = case.max_log_stretch * case.strength * math.tanh(abs(case.lift)) * attenuation
    stroke = length * (-math.expm1(-strain))
    stiffness = legs * case.leg_young * area / length
    energy = 0.5 * stiffness * stroke * stroke
    gravity = 9.81
    required = mass * gravity * case.target_height
    speed = math.sqrt(2.0 * gravity * case.target_height)
    return {
        "model": "linear axial-leg screening; actual takeoff remains unvalidated",
        "mass_kg": mass,
        "screening_leg_young_pa": case.leg_young,
        "effective_leg_length_m": length,
        "axial_stiffness_n_m": stiffness,
        "command_fundamental_attenuation": attenuation,
        "actuator_phase_lag_rad": math.atan(omega * case.actuator_tau),
        "screening_log_strain": strain,
        "screening_stroke_m": stroke,
        "screening_elastic_energy_j": energy,
        "target_height_m": case.target_height,
        "target_potential_energy_j": required,
        "target_takeoff_speed_m_s": speed,
        "target_ballistic_flight_s": 2.0 * speed / gravity,
        "axial_force_over_weight": stiffness * stroke / (mass * gravity),
        "energy_height_m_by_assumed_efficiency": {
            str(eta): eta * energy / (mass * gravity) for eta in (0.1, 0.2, 0.4)
        },
        "energy_margin_at_assumed_20_percent": 0.2 * energy / required,
        "pulse_warning": "fundamental attenuation is not the full pulse frequency response",
    }


def preflight(
    case: Scenario, *, design_mode: str = "hard", material_mode: str = "continuous"
) -> dict[str, object]:
    """Inspect actual sampled geometry without compiling a time integrator."""

    import jax.numpy as jnp

    from envs.soft_walker.engine import topology_checks
    from envs.soft_walker.engine.embodiments.decoders import check_embedding
    from envs.soft_walker.presets.embodiments.chambers import box_score, boxes
    from envs.soft_walker.presets.scenarios.build import decoder_and_embodiment, grid_padding

    decoder, embodiment = decoder_and_embodiment(
        case, design_mode=design_mode, material_mode=material_mode
    )
    ppa = 2 if case.driver.startswith("mpm") else 1
    shape = tuple(n * ppa for n in case.grid_shape)
    coordinates = np.stack(
        np.meshgrid(*[(np.arange(n) + 0.5) * 2.0 / n - 1.0 for n in shape], indexing="ij"), axis=-1
    ).reshape(-1, case.dim)
    fields = decoder.decode(embodiment, jnp.asarray(coordinates, case.dtype))
    occupied = np.asarray(fields.occupancy)
    volume = np.prod(case.size) * (0.01 if case.dim == 2 else 1.0) / np.prod(shape)
    mass = float(np.sum(occupied * np.asarray(fields.density)) * volume)
    ideal = np.asarray(box_score(jnp.asarray(coordinates), *map(jnp.asarray, boxes(case.dim)))) >= 0
    mask = occupied >= 0.5
    union = np.count_nonzero(mask | ideal)
    intersection = np.count_nonzero(mask & ideal)
    errors = check_embedding(decoder, embodiment) + topology_checks.errors(
        mask.astype(float), shape
    )
    active_channels = np.unique(
        np.asarray(fields.channel)[mask & (np.asarray(fields.strength) > 0)]
    )
    if len(active_channels) != case.channels:
        errors.append("MISSING_ACTUATOR_CHAMBER")
    padding = grid_padding(case)
    grid_shape = tuple(n + lo + hi + 1 for n, (lo, hi) in zip(case.grid_shape, padding))
    n = int(np.prod(shape))
    unknowns = (
        int(np.prod(grid_shape)) * case.dim
        if case.driver.startswith("mpm")
        else (
            6 * int(np.prod(case.grid_shape))
            if case.driver == "beam"
            else case.dim * int(np.prod(np.asarray(case.grid_shape) + 1))
        )
    )
    warnings = []
    if case.refinement == 1:
        warnings.append(
            "Only one coarse cell per actuator chamber; use resolved tier to test fidelity"
        )
    if case.dtype == "float32":
        warnings.append("Earlier biped float32 settling failed; no float64 fallback is allowed")
    if design_mode == "relaxed":
        warnings.append("Relaxed void has mass/stiffness; no hard-design gait certification")
    warnings += [
        "No validated general self-contact",
        "Backward Euler is dissipative; refine frozen gaits",
    ]
    if case.representation == "cppn":
        warnings.append("Compact CPPN approximates geometry; sampled mask/volume may differ")
    # The categorical palette can differ from requested continuous moduli.
    # Screen using the actually sampled actuated material, not a stale recipe value.
    actuated = mask & (np.asarray(fields.strength) > 0)
    effective_young = (
        float(np.mean(np.asarray(fields.young)[actuated])) if np.any(actuated) else case.leg_young
    )
    screen_case = replace(case, leg_young=effective_young)
    return {
        "scenario": case.to_dict(),
        "geometry_errors": errors,
        "warnings": warnings,
        "geometry_initialization_passed": not errors,
        "continuous_embedding_scalars": sum(
            v.size for v in embodiment.values() if v.dtype.kind == "f"
        ),
        "sample_shape": shape,
        "allocated_material_sites": n,
        "occupied_material_sites": int(np.count_nonzero(mask)),
        "active_channels": active_channels.tolist(),
        "mass_kg": mass,
        "mask_iou_vs_primitive_seed": intersection / max(int(union), 1),
        "relative_volume_vs_primitive_seed": float(np.sum(occupied))
        / max(int(np.count_nonzero(ideal)), 1),
        "grid_shape": grid_shape if case.driver.startswith("mpm") else None,
        "allocated_solver_unknowns": unknowns,
        "one_dense_tangent_gib_for_comparison_only": unknowns**2
        * np.dtype(case.dtype).itemsize
        / 1024**3,
        "recorded_positions_mib": (case.steps + 1)
        * n
        * case.dim
        * np.dtype(case.dtype).itemsize
        / 1024**2,
        "recording_memory_note": "estimate is particle/cell-based; nodes differ for FEM/springs; excludes solver/AD buffers",
        "physics_steps": case.steps * case.numerical_substeps,
        "physical_duration_s": case.steps * case.effective_control_dt,
        "solver_dt_s": case.effective_control_dt / case.numerical_substeps,
        "airborne_screen": physical_budget(screen_case, realized_mass=mass),
        "validation": "geometry/budget inspection only; not a dynamics pass",
        "decoder": decoder.descriptor(),
    }


# Reference-body region labels


def region_tags(env: Spring2DEnv, model: PhysicsModel) -> tuple[jax.Array, jax.Array]:
    """Torso and foot groups, fixed at reference positions and never resorted."""

    import jax.numpy as jnp

    from envs.soft_walker.engine.drivers.model_access import (
        task_reference_positions as reference_positions,
    )
    from envs.soft_walker.engine.drivers.task import BodyConfig
    from envs.soft_walker.engine.physics import beam, mpm

    ref = reference_positions(env, model)
    # Use the known spawn envelope, not the candidate's deformed bounding box.
    origin = jnp.min(ref, axis=0)
    if hasattr(env, "kind") and env.kind.startswith("mpm"):
        origin = (
            origin
            - cast(mpm.Grid, env._grid).spacing
            / cast(BodyConfig, env.config).particles_per_axis
            / 2
        )
    elif hasattr(env, "kind") and env.kind == "beam":
        origin = origin - cast(beam.Model, model).topology.h / 2
    q = (ref - origin) / jnp.asarray(env.config.size, ref.dtype)
    active = model.active
    torso = active & (q[:, 1] >= TORSO_REGION_START - REGION_BOUNDARY_TOLERANCE)
    leg_x = (q[:, 0] > 0.5).astype(jnp.int32)
    leg = leg_x if ref.shape[1] == 2 else 2 * leg_x + (q[:, 2] > 0.5).astype(jnp.int32)
    feet = jnp.stack(
        [
            active & (q[:, 1] <= FOOT_REGION_END + REGION_BOUNDARY_TOLERANCE) & (leg == k)
            for k in range(2 if ref.shape[1] == 2 else 4)
        ]
    )
    return torso, feet
