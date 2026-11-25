#!/usr/bin/env python3
"""Generate reproducible random creature designs and exercise driver swaps.

Generation uses the actual generic decoders and saves their parameters plus
sampled fields. Replay loads one unchanged design into compatible legacy
drivers. Neither operation optimizes a gait or claims equivalent dynamics.
"""

from __future__ import annotations

import argparse

# Resolve integrated imports independently of the caller's working directory.
import sys as _script_sys
from pathlib import Path, Path as _ScriptPath
from typing import TYPE_CHECKING, Any, Callable, Generic, TypedDict, TypeVar, cast

_source_root = str(_ScriptPath(__file__).resolve().parents[3])
if _source_root not in _script_sys.path:
    _script_sys.path.insert(0, _source_root)
_script_dir = str(_ScriptPath(__file__).resolve().parent)
if _script_dir not in _script_sys.path:
    _script_sys.path.insert(0, _script_dir)
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
import traceback
from dataclasses import asdict

from envs.soft_walker.shared import (
    ENGINE_ROOT as ENGINE_ROOT,
    REPO_ROOT as REPO_ROOT,
    SCRIPTS_ROOT as SCRIPTS_ROOT,
    SOFT_ROOT as SOFT_ROOT,
    SRC_ROOT as SRC_ROOT,
    TESTS_ROOT as TESTS_ROOT,
)

if TYPE_CHECKING:
    from numpy.typing import DTypeLike

    from envs.soft_walker.engine.drivers.spring_environment import Spring2DEnv
    from envs.soft_walker.engine.drivers.task import BodyConfig
    from envs.soft_walker.engine.embodiments.decoders import (
        CPPNDecoder,
        GridDecoder,
        PrimitiveDecoder,
    )
    from envs.soft_walker.engine.physics import beam, fem, mpm, spring
    from envs.soft_walker.engine.physics.spring_config import SpringConfig
    from envs.soft_walker.engine.physics.terrain import Terrain, TerrainConfig
    from envs.soft_walker.engine.types import (
        BodyModel,
        DriverState,
        JsonValue,
        PathLike,
        PhysicsModel,
    )

ROOT = REPO_ROOT
sys.path.insert(0, str(ROOT / "src"))
REPRESENTATIONS = ("primitives", "direct", "bspline", "cppn")
DRIVERS = {
    "spring": "SoftSpring2D-v0",
    "mpm2": "SoftMPM2D-v0",
    "mpm3": "SoftMPM3D-v0",
    "beam": "SoftBeam3D-v0",
    "fem": "SoftHexFEM3D-v0",
}

# Serialized reports can contain backend-specific auxiliary diagnostic trees.
Report = dict[str, Any]


ShapeT = TypeVar("ShapeT")
SizeT = TypeVar("SizeT")


class CommonConfig(TypedDict, Generic[ShapeT, SizeT]):
    """Shared constructor keywords; a spring's shape and size have exactly two axes."""

    grid_shape: ShapeT
    size: SizeT
    control_dt: float
    substeps: int
    settle_steps: int
    horizon: int
    initial_clearance: float
    length_ref: float
    energy_weight: float
    terrain: TerrainConfig


class GenerationMetadata(TypedDict):
    """Required replay metadata emitted by this command's generate subcommand."""

    grid_shape: list[int]
    size: list[float]
    seed: int


def write(path: PathLike, data: Any) -> None:
    """Serialize evidence to the requested path using this command's existing JSON encoder."""

    def encode(value: Any) -> JsonValue:
        if hasattr(value, "tolist"):
            return encode(value.tolist())
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {key: encode(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [encode(item) for item in value]
        if isinstance(value, float) and not math.isfinite(value):
            return str(value)
        return value

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(encode(data), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def fresh(path: Path) -> Path:
    """Require a new output directory so a run cannot overwrite earlier evidence."""
    path = path.expanduser().resolve()
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"Output directory must be fresh: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def digest(path: PathLike) -> str:
    """Return the SHA-256 digest of file bytes for artifact provenance."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def configure(device: str, dtype: DTypeLike) -> None:
    """Select process device/precision before importing JAX; configure cache and host thread limits."""
    os.environ["JAX_PLATFORMS"] = device
    os.environ["JAX_ENABLE_X64"] = "1" if dtype == "float64" else "0"
    os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", str(Path.home() / ".cache/codesign-jax"))
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.pop("CODESIGN_OFFLINE_SPACES", None)


def generate(args: argparse.Namespace) -> int:
    """Sample and save multiple morphology representations with their fields and validation metadata."""
    configure("cpu", args.dtype)
    import jax
    import numpy as np

    from envs.soft_walker.engine.embodiments.morphology import generate_random_design, sample_fields
    from envs.soft_walker.engine.embodiments.serialization import load_design, save_design

    grid_shape = tuple(args.grid or ((12, 10) if args.dim == 2 else (6, 4, 4)))
    if len(grid_shape) != args.dim or min(grid_shape) < 3:
        raise ValueError("--grid requires one dimension >=3 per axis")
    size = tuple(args.cell_size * n for n in grid_shape)
    sample_shape = tuple(2 * n for n in grid_shape)
    output = fresh(args.out)
    status: Report = {
        "schema": "codesign_random_creatures_v1",
        "completed": False,
        "dim": args.dim,
        "grid_shape": grid_shape,
        "sample_shape": sample_shape,
        "size": size,
        "dtype": args.dtype,
        "seed": args.seed,
        "count_per_representation": args.count,
        "candidates": [],
        "failures": [],
        "dynamics_run": False,
        "sampling": "Structured random priors and explicit rejection; saved decoder parameters determine every field",
        "display": "Hard occupancy at cell centers; sampled design cells are not solver elements",
        "source_sha256": {
            "src/envs/soft_walker/scripts/random_creatures.py": digest(__file__),
            "src/envs/soft_walker/engine/morphology.py": digest(ENGINE_ROOT / "morphology.py"),
            "src/envs/soft_walker/engine/design.py": digest(ENGINE_ROOT / "design.py"),
        },
    }
    write(output / "manifest.json", status)
    begin = time.perf_counter()
    for representation in args.representations:
        hashes = set()
        for index in range(args.count):
            seed = args.seed + index
            identifier = f"{representation}_{index:02d}_s{seed}"
            directory = output / identifier
            try:
                decoder, embodiment, report = generate_random_design(
                    representation,
                    dim=args.dim,
                    seed=seed,
                    grid_shape=grid_shape,
                    channels=args.channels,
                    dtype=args.dtype,
                    max_attempts=args.max_attempts,
                )
                directory.mkdir()
                metadata: Report = {
                    "generator": report,
                    "grid_shape": grid_shape,
                    "size": size,
                    "sample_shape": sample_shape,
                    "representation": representation,
                    "seed": seed,
                }
                save_design(directory / "design.json", embodiment, decoder, metadata)
                restored, restored_decoder, _ = load_design(directory / "design.json")
                points, fields = sample_fields(decoder, embodiment, sample_shape)
                _, roundtrip = sample_fields(restored_decoder, restored, sample_shape)
                for expected, actual in zip(fields, roundtrip):
                    np.testing.assert_array_equal(np.asarray(expected), np.asarray(actual))
                arrays = {name: np.asarray(value) for name, value in fields._asdict().items()}
                arrays.update(
                    points=np.asarray(points),
                    grid_shape=np.asarray(sample_shape, np.int32),
                    size=np.asarray(size, args.dtype),
                    raw_scores=np.asarray(
                        jax.jit(cast("GridDecoder | CPPNDecoder | PrimitiveDecoder", decoder).raw)(
                            restored, points
                        )
                    ),
                )
                # Field names are fixed; NumPy's broad kwargs include allow_pickle.
                cast(Callable[..., None], np.savez_compressed)(directory / "fields.npz", **arrays)
                signature = hashlib.sha256(
                    np.asarray(fields.occupancy).astype(np.uint8).tobytes()
                ).hexdigest()
                duplicate = signature in hashes
                hashes.add(signature)
                write(
                    directory / "generation.json",
                    dict(
                        report,
                        roundtrip_fields_identical=True,
                        sampled_occupancy_sha256=signature,
                        duplicate_of_earlier_in_family=duplicate,
                    ),
                )
                entry: Report = dict(
                    id=identifier,
                    directory=identifier,
                    representation=representation,
                    seed=seed,
                    dim=args.dim,
                    occupied_sites=int(np.count_nonzero(arrays["occupancy"])),
                    occupancy_fraction=float(np.mean(arrays["occupancy"])),
                    sampled_occupancy_sha256=signature,
                    duplicate_of_earlier_in_family=duplicate,
                    design_sha256=digest(directory / "design.json"),
                    fields_sha256=digest(directory / "fields.npz"),
                )
                status["candidates"].append(entry)
                print(
                    json.dumps(
                        {
                            "generated": identifier,
                            "occupied_sites": entry["occupied_sites"],
                            "duplicate": duplicate,
                        }
                    ),
                    flush=True,
                )
            except Exception as error:
                status["failures"].append(
                    dict(id=identifier, exception=repr(error), traceback=traceback.format_exc())
                )
                print(f"Generation failed for {identifier}: {error}", file=sys.stderr, flush=True)
            write(output / "manifest.json", status)
    status.update(
        completed=True, passed=not status["failures"], wall_seconds=time.perf_counter() - begin
    )
    write(output / "manifest.json", status)
    return 0 if status["passed"] else 1


def configuration(
    driver: str, shape: tuple[int, ...], size: tuple[float, ...], steps: int, dt: float
) -> SpringConfig | BodyConfig:
    """Build a driver-specific configuration with the requested discretization and fixed physical dt."""
    from envs.soft_walker.engine.drivers.task import BodyConfig
    from envs.soft_walker.engine.physics.spring_config import SpringConfig
    from envs.soft_walker.engine.physics.terrain import TerrainConfig

    common: CommonConfig[tuple[int, ...], tuple[float, ...]] = dict(
        grid_shape=shape,
        size=size,
        control_dt=dt,
        substeps=1,
        settle_steps=0,
        horizon=max(1, steps),
        initial_clearance=0.002,
        length_ref=size[0],
        energy_weight=0.001,
        terrain=TerrainConfig(kind="flat", blocks=0),
    )
    if driver == "spring":
        return SpringConfig(
            **cast("CommonConfig[tuple[int, int], tuple[float, float]]", common),
            linear_solver="gmres",
        )
    return BodyConfig(**common)


def observation_capacity(driver: str, shape: tuple[int, ...], channels: int) -> int:
    """Compute enough observation capacity for the selected discretization and actuator channels."""
    import numpy as np

    dim = len(shape)
    cells = int(np.prod(shape))
    if driver == "spring":
        return 5 + 5 * int(np.prod(np.asarray(shape) + 1)) + 4 * cells + 2 * channels + 9
    sites = cells * (2**dim if driver.startswith("mpm") else 1)
    nodes = int(np.prod(np.asarray(shape) + 1)) if driver == "fem" else sites
    return (
        3 * dim
        + 1
        + nodes * (3 * dim + 1)
        + sites * (8 + dim + channels)
        + 2 * channels
        + 3 * (dim + 1)
        + (12 * nodes if driver == "beam" else 0)
        + (2 * dim**2 * nodes if driver.startswith("mpm") else 0)
    )


def probe(args: argparse.Namespace) -> int:
    """Construct and validate sampled designs, recording failures independently for each candidate."""
    document = json.loads(args.design.read_text(encoding="utf-8"))
    dtype = document["decoder"]["dtype"]
    configure(args.device, dtype)
    import jax
    import numpy as np

    from envs.soft_walker.engine import make
    from envs.soft_walker.engine.drivers.model_access import (
        task_positions as positions,
        task_reference_positions as reference_positions,
    )
    from envs.soft_walker.engine.embodiments.serialization import load_design
    from envs.soft_walker.engine.physics.terrain import serialize

    jax.config.update("jax_default_matmul_precision", "highest")
    output = fresh(args.out)
    status: Report = dict(
        completed=False,
        passed=False,
        driver=args.driver,
        execution="legacy",
        design_sha256=digest(args.design),
        requested_steps=args.steps,
        scope="Short driver integration check; not a gait or backend-equivalence benchmark",
    )
    begin = time.perf_counter()
    env = None
    try:
        embodiment, decoder, saved_metadata = load_design(args.design)
        metadata = cast(GenerationMetadata, saved_metadata)
        dim = decoder.spec.dim
        if (args.driver in ("spring", "mpm2")) != (dim == 2):
            raise ValueError("Design dimension does not match driver")
        shape, size = tuple(metadata["grid_shape"]), tuple(metadata["size"])
        config = configuration(args.driver, shape, size, args.steps, args.dt)
        env, base = make(
            DRIVERS[args.driver],
            config=config,
            decoder=decoder,
            device=args.device,
            dtype=dtype,
            is_eval_env=True,
            max_action_space_size=decoder.spec.channels,
            max_observation_space_size=observation_capacity(
                args.driver, shape, decoder.spec.channels
            ),
        )
        # make defaults to JAX here; this command exposes no native-backend option.
        env = cast("Spring2DEnv", env)
        candidate = dict(embodiment)
        if args.driver == "spring":
            candidate.pop("poisson", None)  # Spring has no global Poisson parameter.
        write(output / "config.json", asdict(config))
        write(output / "embodiment.json", candidate)
        start = time.perf_counter()
        _, rejected, info = env.reset(seed=metadata.get("seed", 0), embodiment=candidate)
        status["reset_seconds_including_compile"] = time.perf_counter() - start
        write(output / "reset.json", info)
        if bool(rejected[0]):
            raise RuntimeError(f"Reset rejected: {info.get('embodiment_errors')}")
        # A successful reset establishes model/state ownership for the chosen driver.
        model = cast("PhysicsModel", env._model)
        coordinates = env._topology.points if args.driver == "spring" else env._domain.points
        fields = decoder.decode(embodiment, coordinates)
        for name in ("young", "density", "strength"):
            np.testing.assert_allclose(
                np.asarray(getattr(model, name)),
                np.asarray(getattr(fields, name)),
                rtol=2e-6 if dtype == "float32" else 1e-12,
                atol=0.0,
            )
        if args.driver != "spring" and "poisson" not in candidate:
            np.testing.assert_allclose(
                np.asarray(cast("BodyModel", model).poisson),
                np.asarray(fields.poisson),
                rtol=2e-6 if dtype == "float32" else 1e-12,
                atol=0.0,
            )
        np.testing.assert_array_equal(
            np.asarray(
                cast("spring.Model", model).occupancy
                if args.driver == "spring"
                else cast("BodyModel", model).alpha
            ),
            np.asarray(fields.occupancy),
        )
        status["instantiated_properties_match_decoder"] = True
        cast(Callable[..., None], np.savez_compressed)(
            output / "instantiated_fields.npz",
            points=np.asarray(coordinates),
            **{name: np.asarray(value) for name, value in fields._asdict().items()},
        )
        write(output / "terrain.json", serialize(cast("Terrain", model.terrain)))
        snapshots = [np.asarray(positions(env, model, cast("DriverState", env._state)))]
        times = [float(info["time_s"])]
        reports: list[Report] = []
        # Same modest external controller law across all compatible drivers.
        # No dynamics or optimization is hidden inside the decoder.
        for step in range(args.steps):
            action = 0.1 * np.sin(
                2 * np.pi * np.arange(decoder.spec.channels) / decoder.spec.channels
                + 2 * np.pi * step / max(1, args.steps)
            )
            _, reward, terminated, truncated, info = env.step(action.astype(dtype))
            reports.append(
                dict(
                    step=step,
                    action=action,
                    reward=reward,
                    terminated=terminated,
                    truncated=truncated,
                    info=info,
                )
            )
            write(output / "steps.json", reports)
            snapshots.append(np.asarray(positions(env, model, cast("DriverState", env._state))))
            times.append(float(info["time_s"]))
            if terminated or not bool(info["valid"]):
                raise RuntimeError(f"Physical step rejected: {info['status_name']}")
        objective = np.asarray(info["objective"])
        loss = float(objective[0] + config.energy_weight * objective[1])
        if not np.isfinite(loss):
            raise RuntimeError("Final loss is nonfinite")
        topology = dict(
            reference=np.asarray(reference_positions(env, model)), active=np.asarray(model.active)
        )
        if args.driver == "spring":
            topology.update(
                cells=np.asarray(cast("spring.Model", model).cells),
                edges=np.asarray(cast("spring.Model", model).edges),
            )
        elif args.driver == "fem":
            topology.update(elements=np.asarray(cast("fem.Model", model).mesh.cells))
        elif args.driver == "beam":
            topology.update(edges=np.asarray(cast("beam.Model", model).topology.edges))
        else:
            topology.update(particle_volume=np.asarray(cast("mpm.Model", model).cloud.volume))
        cast(Callable[..., None], np.savez_compressed)(output / "mechanical_model.npz", **topology)
        np.savez_compressed(
            output / "trajectory.npz", positions=np.asarray(snapshots), times=np.asarray(times)
        )
        status.update(
            completed=True,
            passed=True,
            physical_time_s=times[-1],
            loss=loss,
            loss_formula="-forward_distance/length_ref + .001*positive_work/work_ref",
            material_sites=int(len(coordinates)),
            active_mechanical_sites=int(np.count_nonzero(model.active)),
            mass_kg=float(info["mass_kg"]),
        )
    except Exception as error:
        status.update(exception=repr(error), traceback=traceback.format_exc())
        print(status["traceback"], file=sys.stderr, flush=True)
    finally:
        if env is not None:
            env.close()
        status["wall_seconds"] = time.perf_counter() - begin
        write(output / "status.json", status)
    return 0 if status["passed"] else 1


def replay(args: argparse.Namespace) -> int:
    """Execute the selected saved/native design and record its trajectory, status and timings."""
    design = args.design.expanduser().resolve()
    document = json.loads(design.read_text(encoding="utf-8"))
    dim = document["decoder"]["spec"]["dim"]
    drivers = args.drivers or (["spring", "mpm2"] if dim == 2 else ["mpm3", "beam", "fem"])
    if any((driver in ("spring", "mpm2")) != (dim == 2) for driver in drivers):
        raise ValueError("Use spring/mpm2 for 2D; mpm3/beam/fem for 3D")
    output = fresh(args.out)
    report: Report = dict(
        completed=False,
        passed=False,
        execution=args.execution,
        design=str(design),
        design_sha256=digest(design),
        device=args.device,
        stages=[],
    )
    write(output / "manifest.json", report)
    for driver in drivers:
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "_probe",
            "--design",
            str(design),
            "--driver",
            driver,
            "--device",
            args.device,
            "--steps",
            str(args.steps),
            "--dt",
            str(args.dt),
            "--out",
            str(output / driver),
        ]
        entry: Report = dict(driver=driver, command=command, timed_out=False)
        started = time.perf_counter()
        with (output / f"{driver}.log").open("w") as log:
            child = subprocess.Popen(
                command,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=os.name != "nt",
            )
            try:
                child.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                entry["timed_out"] = True
                if os.name == "nt":
                    child.terminate()
                else:
                    os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    if os.name == "nt":
                        child.kill()
                    else:
                        os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
        entry.update(returncode=child.returncode, wall_seconds=time.perf_counter() - started)
        report["stages"].append(entry)
        write(output / "manifest.json", report)
        print(json.dumps(entry), flush=True)
    report.update(
        completed=True,
        passed=all(
            stage["returncode"] == 0 and not stage["timed_out"] for stage in report["stages"]
        ),
    )
    write(output / "manifest.json", report)
    return 0 if report["passed"] else 1


def main() -> int:
    """Dispatch generation, probe, or replay with device setup performed before physics imports."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    gen = commands.add_parser(
        "generate", help="Save random designs and actual decoded fields; no dynamics"
    )
    gen.add_argument(
        "--representations", nargs="+", choices=REPRESENTATIONS, default=list(REPRESENTATIONS)
    )
    gen.add_argument("--dim", type=int, choices=(2, 3), default=2)
    gen.add_argument("--grid", type=int, nargs="+")
    gen.add_argument("--cell-size", type=float, default=0.02)
    gen.add_argument("--count", type=int, default=6)
    gen.add_argument("--seed", type=int, default=7)
    gen.add_argument("--channels", type=int, default=4)
    gen.add_argument("--dtype", choices=("float64", "float32"), default="float64")
    gen.add_argument("--max-attempts", type=int, default=100)
    gen.add_argument("--out", type=Path, required=True)
    run = commands.add_parser(
        "replay", help="Load one design into compatible legacy drivers and compute a short loss"
    )
    run.add_argument("--drivers", nargs="+", choices=tuple(DRIVERS))
    run.add_argument("--execution", choices=("legacy",), default="legacy")
    run.add_argument("--timeout", type=float, default=300.0)
    worker = commands.add_parser("_probe", help=argparse.SUPPRESS)
    worker.add_argument("--driver", choices=tuple(DRIVERS), required=True)
    for command in (run, worker):
        command.add_argument("--design", type=Path, required=True)
        command.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
        command.add_argument("--steps", type=int, default=3)
        command.add_argument("--dt", type=float, default=0.00025)
        command.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "generate":
        if args.count < 1 or args.seed < 0 or args.max_attempts < 1 or not 1 <= args.channels <= 16:
            parser.error("count/attempts must be positive, seed nonnegative, channels in [1,16]")
        if not math.isfinite(args.cell_size) or args.cell_size <= 0:
            parser.error("cell-size must be finite and positive")
        return generate(args)
    if args.steps < 0 or not math.isfinite(args.dt) or args.dt <= 0:
        parser.error("steps must be nonnegative and dt finite and positive")
    if args.command == "replay" and (not math.isfinite(args.timeout) or args.timeout <= 0):
        parser.error("timeout must be finite and positive")
    return replay(args) if args.command == "replay" else probe(args)


if __name__ == "__main__":
    raise SystemExit(main())
