# Components and Gymnasium integration

This is the current entry point for using and extending the simulator. Read the
[modularity matrix](MODULARITY_MATRIX.md) for available choices and the
[capability matrix](CAPABILITIES.md) for supported combinations and explicit
limits, and [resident execution](RESIDENT_EXECUTION.md) for executable-specific
CUDA certification. Historical milestone documents describe their original
snapshots and are not the current API reference. Repository paths in examples
are relative to the codesign-gym root.

## The pipeline

```text
embodiment dictionary + decoder
    -> sampled occupancy/material/actuator fields
    -> driver-specific mass, stiffness, particles/nodes, and contact surfaces
    -> initial state and zero-command settling
    -> controller action at each control interval
    -> substeps of physics, contact, and actuator dynamics
    -> observations, objective/reward, validity, diagnostics, optional recording
    -> gradients of eligible continuous parameters -> optimizer updates
```

The decoder describes the body; the physics driver determines how it moves.
Controllers are separate functions. Terrain and numerical settings belong to the
environment configuration. An optimizer calls a rollout or loss; it is not part
of the integrator. The native action is a vector of actuator-channel commands,
not joint torques or forces added directly to the robot's center of mass.

The coordinates are +x forward, +y up, and +z lateral in 3D. Values use SI units
unless a configuration or field explicitly defines a normalization. A control
interval lasts `control_dt`, with `substeps` physical steps of
`dt = control_dt / substeps`. JAX `settle_steps` counts original configured
control intervals; native Warp `settling_steps` counts raw physics steps.
`physics_steps_per_action` changes action cadence while preserving physical dt
and the original settling duration. See [action holding](ACTION_HOLD.md).

## Public interfaces and component boundaries

Shared host contracts live under `contracts/`: `fields.py`, `observations.py`,
`metrics.py`, `batching.py`, `status_codes.py` and `api_types.py`. They describe array/material
contracts and shared validation without importing an accelerator backend.
Backend-specific types remain with their implementation.

The registered Gym task follows the Codesign Gym layout: `config.py` (presets,
terrains, bundle builders and task setup), `embodiment.py` (the flat per-cell schema),
`vector_env.py` (the native Warp vector env) and `render.py`/`render3d.py`.
`registration.py` is the registry entry point. The registered task runs on Warp only;
the JAX backend is available through the engine factories.

The engine is the reusable simulator: it may depend on shared contracts, but not
on the registered task or its registry entry point. Generic physical assembly
belongs in the engine. `registered/builders.py` applies the registered cell-vector
schema, body presets, terrain recipes and timing to those assembly routines; that
task-specific mapping stays outside the engine. Shared contracts form a lower
dependency layer and must not import the engine or registered task implementations.

The engine groups implementation code by responsibility:

```text
engine/
  embodiments/   decoders, field interpretation, sampling priors, random bodies and persistence
  physics/       springs, beams, FEM, MPM, materials, terrain, contact and physical guards
  numerics/      integrator selection, JAX integration/solvers, shared tableaux and policies
  drivers/       scalar/vector lifecycle, task state/rewards, model access and reporting
  execution/     compiled episode preparation, capture and execution diagnostics
  warp_native/   native Warp models, kernels, episodes and adjoints
```

Application responsibilities live outside the reusable engine:

```text
experiments/     optimization/checkpoints, locomotion/scenario objectives and resident adapters
presets/         named bodies, gait seeds, scenario configurations/suites and terrain courses
terrain/         portable host-only recipe schema and validation
viewer/          saved trajectory geometry, preview pictures and interactive rendering
diagnostics/     preflight reports, qualification, benchmarks and evidence tooling
```

The [release organization guide](RELEASE_ORGANIZATION.md) maps the former
`engine.locomotion` and `engine.scenarios` imports. Generic runtime code must not
depend on a specific robot experiment. The chamber and legged constructors are
presets because they assume torso/leg regions; generic decoder contracts remain
in `engine.embodiments`. Named terrain generators retain distinct RNGs and ranges,
while recipe validation is shared. `engine.action_timing` remains a backend-neutral
control-cadence service; registered episode-horizon rounding is a separate task policy.

`physics/` contains the JAX mechanics implementations, shared physical guards,
constitutive laws and contact. `numerics/` contains generic JAX integration and
linear/nonlinear solving, plus backend-neutral coefficients and solver policies. Springs and beams retain separate
models: spring particles have translational degrees of freedom; beams include
rotations, bending, shear and torsion. Grouping them does not change their equations
or move native Warp kernels out of `warp_native/`.

`drivers/environment.py` builds and advances models, owns compiled-function caches,
and implements `BodyEnv`. `drivers/model_access.py` reads reference/current positions,
selects the physics module and exposes channel masks. `drivers/reporting.py` builds
observations and mechanics/energy reports. Keeping these responsibilities separate
allows reporting and model access to be used without duplicating environment logic.
The `drivers` namespace itself is lightweight; import functions from their owner.
`drivers/spring_environment.py` owns the spring scalar environment and common
reset lifecycle; `vector_lifecycle.py` coordinates JAX lanes and autoreset.
`drivers/task.py` owns body task state, reward and objective semantics. These
modules contain runtime behavior, not merely type declarations.

The registered wrapper's `registered/embodiment.py` owns its flat per-cell
schema and conversion boundary, distinct from latent embodiment generation.
`registered/render.py` renders that schema. General recording/viewer code and
engine rendering remain separate.
Registered presets, terrain recipes and resolved model construction are owned by
`registered/presets.py`, `registered/terrain.py` and `registered/builders.py`, respectively.

This branch intentionally removes legacy import/CLI shims and historical Python
class-module overrides. Update downstream imports using the
[breaking cleanup migration guide](BREAKING_CLEANUP_MIGRATION.md) and
[responsibility follow-up](RESPONSIBILITY_CLEANUP.md). New trusted
Python pickles use canonical defining modules. Old external pickles may require
loading with the preceding revision before export to a portable format.
Portable JSON embodiment descriptors and array-only NPZ Adam checkpoint formats
retain their schemas. The local artifact audit found no saved pickle files or
object arrays in the inspected archives; see the migration guide for its scope.

Shared host contracts have explicit owners: `contracts/fields.py` names decoder
field slices, `contracts/observations.py` names observation profiles, and
`contracts/metrics.py` names recording/diagnostic columns and native ledgers.
These names describe existing layouts; they do not reorder serialized arrays.
`engine/numerics/policy.py` records shared solver acceptance/preconditioner guards;
`engine/physics/physics_constants.py` records spring and beam/rotation safeguards,
while `contracts/status_codes.py` keeps root-solver, binary state, spring, policy and
adjoint status domains distinct. Equal integer values across domains do not imply
equal failure meanings.

`execution/preparation.py` owns pure spring decode/assembly/placement, while
rollout callers retain settling, validation and state ownership.
`experiments/scenarios/objectives.py` owns the existing scenario shaping formula used by both
ordinary and resident execution. Diagnostic-only probe settings are in
`diagnostics/residency/fixtures.py`; they are not physical model defaults.

`differentiation.py` remains the high-level pure-rollout binding interface: it
combines decoding, assembly, settling, control and reporting into differentiable
functions. Low-level implicit differentiation stays with the numerical solvers.
Finite-difference ladders and one-step descent audits live in
`shared.py`; they do not implement the training optimizer.
Warp numerical kernel implementations remain in `engine/warp_native/`.

Configuration follows its owner: `physics/spring_config.py` contains spring
settings, `drivers/task.py` body-task settings, and `numerics/solvers.py` Newton
settings. Native model configurations stay with native models.
`engine/device.py` resolves explicit dtype/device requests. Importing the helpers
does not import JAX; calling either resolver imports JAX to check its precision
policy or available devices. `engine/types.py` remains local because its annotations describe
engine models, state and kernels. Shared `contracts/` never imports those owners.

`embodiments/serialization.py` persists decoder descriptors and embedding arrays;
it is not a general object serializer. `shared.py` owns shared JSON/event/hash
writers and derivative qualification helpers, alongside optimizer/checkpoint
and controller helpers. Native audit helpers and fixtures live in `diagnostics/warp/`.
Production JAX/native converters remain in `warp_native/mpm_conversion.py` and
`solid_conversion.py`; simulation does not import diagnostic fixtures.
Resident command dispatch belongs to `scripts/_resident_commands.py`. It combines
execution with training, recording, rendering and optional gradient checks;
reusable engine execution does not depend on those command workflows.

The JAX engine accepts latent decoder dictionaries and supports pure
differentiable rollouts. The native Warp engine accepts resolved physical arrays
and supports native fixed-topology episodes/adjoints. The registered SoftWalker
Gym wrapper builds either backend from the same flat per-cell embodiment. Its
resolved observation/reward layout is shared across backends. The engine APIs
still have their own embedding and observation schemas. Export-once conversion
does not propagate decoder gradients; the explicit `EmbodimentEpisode` interface
composes decoding/assembly with the native adjoint. See the
[interface comparison](CAPABILITIES.md#three-public-interfaces).

| Component | Inputs → outputs | How to replace or extend it |
| --- | --- | --- |
| Embodiment decoder | Embedding dictionary + normalized sample points → `Fields` | Implement the `Decoder` protocol; [custom decoder example](../examples/custom_decoder.py) supplies every method |
| Spatial fields | Occupancy/material/actuator scores → bounded physical fields | Use the decoder field specification; `set_poisson(decoder, embedding, value)` demonstrates a bounded field update |
| Physics assembly | Sampled fields + discretization + static configuration → model/state | Select a factory driver; a new driver must supply assembly, initialization, transition, observation, reward, and validity contracts |
| Controller | Parameters + observation/time + optional carry → channel commands/carry | Implement a policy callable for `bind_policy`, or add a scenario controller with parameter schema and seed |
| Terrain | Plane/rounded-box recipe + coordinates → gaps/normals/contact forces | Compose an existing terrain recipe; add geometry only with force/normal and derivative tests |
| Integrator/solver | Model/state + command + dt → state and diagnostics | Configure a supported numerical path; adding one requires failure transaction and primal/adjoint checks |
| Observation/reporting | Model/state + capacities → padded arrays, masks, diagnostic dictionaries | Add a named profile; preserve existing ordering rather than inserting fields into an old profile |
| Objective | Rollout/model/state/parameters → scalar loss and auxiliary evidence | Compose a scalar differentiable loss around a pure rollout; the optimizer consumes value/gradient |
| Optimizer | Value-and-gradient callable + parameters + state → updates/checkpoint | Follow the existing optimization callback/checkpoint contract; keep physics evaluation outside persistence code |
| Rendering | Saved trajectories/fields + connectivity → images or viewer geometry | Load `Recording` and display its valid sample prefix; cosmetic surfaces never substitute for solver elements |

Public stepping performs host validation and reporting. Compiled numerical
functions use array pytrees and explicit inputs/outputs. Do not put file I/O,
mutable Python bookkeeping, or dynamic shape changes inside JIT/graph loops.
Static shape/configuration choices belong to construction; per-episode design
values belong to reset; controller commands belong to step or a pure rollout.

## Install and run a small CPU example

Use Python 3.11 or later. From the repository root, in a fresh environment:

```sh
python -m pip install -e '.[dev]'
```

The dependency declarations are in [pyproject.toml](../../../../pyproject.toml). This
installs the ordinary Python dependencies; CUDA installation and compatible
driver/toolkit selection are separate. Set `JAX_ENABLE_X64=1` before importing
JAX when using `dtype="float64"`. The following example explicitly uses CPU and
float32, with a tiny gravity-free model so it is an API example rather than a
locomotion experiment.

```python
import os
os.environ["JAX_PLATFORMS"] = "cpu"  # Set before the first JAX import.

import numpy as np
from envs.soft_walker.engine import make
from envs.soft_walker.engine.physics.spring_config import SpringConfig

cfg = SpringConfig(
    grid_shape=(2, 1), size=(0.1, 0.05),
    control_dt=0.001, substeps=1, settle_steps=0, horizon=2,
    gravity=0.0, ground=False,
)
env, embodiment = make(
    "SoftSpring2D-v0", config=cfg, dtype="float32", device="cpu",
    max_action_space_size=4, max_observation_space_size=128,
)
try:
    obs, rejected, info = env.reset(seed=7, embodiment=embodiment)
    if rejected[0]:
        raise ValueError(info["embodiment_errors"])
    assert env.observation_space.contains(obs)
    for _ in range(cfg.horizon):
        action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
        obs, reward, terminated, truncated, info = env.step(action)
        print(reward, info["valid"], terminated, truncated)
        if terminated or truncated:
            break
finally:
    env.close()
```

The public contract intentionally differs from `gymnasium.make/reset`:

| Operation | Return value / behavior |
| --- | --- |
| `envs.soft_walker.engine.make(id, ...)` | `(env, base_embodiment)`; the base is an independent dictionary |
| `env.validate_embodiment(candidate)` | Python `bool`; use `validation_report` for errors |
| `env.reset(seed=..., embodiment=...)` | `(observation, rejection_array, info)`; scalar rejection shape is `(1,)` |
| `env.step(action)` | `(observation, reward, terminated, truncated, info)` |
| `env.render()` | RGB array if created with `render_mode="rgb_array"` |
| `env.close()` | Discards state; create another environment to use it again |

`step` requires a successful reset. Rejected resets invalidate the old state.
Numerical failure terminates an episode; reaching a valid configured horizon
truncates it. Scalar environments do not automatically reset. Changing the body
through `step(action, embodiment=...)` is unsupported: the argument warns and is
ignored. Apply body changes through a new reset instead.

Actions must be finite, lie in `[-1, 1]`, and have the **padded**
`env.action_space.shape`. Values outside the bounds are rejected, not clipped.
The first `env.channels` entries supply native commands. Extra capacity has no
physical effect. `info["action_mask"]` identifies effective entries.
Observations are flat NumPy arrays with a fixed declared capacity;
`info["observation_mask"]` distinguishes meaningful entries from padding and
masked sites. Check `info["valid"]`, `status_name`, and `solver` when assessing
a trajectory. A policy should not assume all drivers share the same full layout.

## Switching physics

Backend, precision, and device are separate construction choices. The
[engine factories](../engine/__init__.py) select JAX by default, or native Warp
with `backend="warp"`. `dtype="float32"` or `"float64"` selects floating-point
precision, and `device` selects CPU/CUDA placement. JAX requires x64 enabled
before initialization for float64; it rejects silent truncation. Native Warp
requires the requested dtype to match the bundle's physical arrays and builds
precision-specialized kernels. A supplied JAX decoder must likewise match its
environment's dtype.

The [registered SoftWalker builder](../registration.py) runs on native Warp. It
defaults to float64. Without an explicit device, it selects CUDA when available.
Its defaults differ from the low-level JAX environment's float32 default.

```python
import codesign_gym as cg
import numpy as np

env, body = cg.make_vec(
    "SoftWalker-Spring2D", num_envs=4,
    dtype="float32", device="cpu", physics_steps_per_action=10,
)
try:
    obs, rejected, info = env.reset(seed=7, embodiment=body)
    assert not rejected.any(), info
    obs, reward, terminated, truncated, info = env.step(
        np.zeros(env.action_space.shape, env.action_space.dtype)
    )
finally:
    env.close()
```

Select the environment ID at construction, with the matching configuration:

| ID | Configuration | Model and state |
| --- | --- | --- |
| `SoftSpring2D-v0` | `SpringConfig` | 2D spring lattice; node positions and velocities; actuator state |
| `SoftBeam3D-v0` | `BodyConfig`, 3D shape/size | Beam lattice; translational and rotational states |
| `SoftHexFEM3D-v0` | `BodyConfig`, 3D shape/size | Hexahedral FEM with hyperelastic material |
| `SoftMPM2D-v0` | `BodyConfig`, 2D shape/size | Plane-strain MPM with declared thickness; particles, deformation `F`, affine velocity `C` |
| `SoftMPM3D-v0` | `BodyConfig`, 3D shape/size | 3D MPM with particle/grid transfers, `F`, and `C` |

All five model families also have native `backend="warp"` implementations with
backward Euler; spring additionally supports native Kvaerno3. The Warp adapter
uses resolved physical-array embodiments and the explicit
`observation_profile="resolved"`; its observation order and embodiment schema
differ from the JAX interfaces described above. Export an existing JAX embodiment
once, or supply a native bundle, rather than passing a latent embedding
directly to Warp. See [WARP_ALL_PHYSICS.md](WARP_ALL_PHYSICS.md) for the supported
integrators, native episodes, and gradient scope. Native vector environments run
independent lanes and support `reset(options={"reset_mask": boolean_mask})`.
Unselected lanes retain their state, RNG and graph ownership.

`registered_envs()` lists package IDs; these IDs are not automatically registered
with Gymnasium's global registry.

For example, this constructs a small MPM2 environment; use the same reset/step
contract above, with its own returned embodiment and action/observation spaces:

```python
from envs.soft_walker.engine import make
from envs.soft_walker.engine.drivers.task import BodyConfig

body_cfg = BodyConfig(
    grid_shape=(2, 1), size=(0.1, 0.05),
    control_dt=0.001, substeps=4, settle_steps=0, horizon=2,
)
body_env, body_embedding = make(
    "SoftMPM2D-v0", config=body_cfg, device="cpu", dtype="float32",
    max_observation_space_size=4096,
)
body_env.close()  # Construction alone does not run a trajectory.
```

### Selecting an integrator

Engine `make`, `make_vec`, and `make_gymnasium_vec` accept the same `integrator`
keyword. The registered `codesign_gym.make_vec` presets do not: they always use
backward Euler. It selects an
existing numerical method; it does not adapt one physics model to a method
implemented for another model.

| Selection | JAX engine | Native Warp engine |
| --- | --- | --- |
| `"backward_euler"` | All five physics families | All five physics families |
| `"kvaerno3_native"` | Spring 2D, GMRES | Spring 2D native stages |
| `"kvaerno3"` | Legacy spring Diffrax path, dense solves | Unavailable |
| `None` or omitted | Preserve the supplied configuration or existing default | Preserve the supplied bundle configuration or existing default |

Use the host-only registry to inspect valid choices before construction:

```python
from envs.soft_walker.engine import available_integrators, make

choices = available_integrators(backend="jax", physics="spring")
spring_env, spring_body = make(
    "SoftSpring2D-v0", integrator="kvaerno3_native", device="cpu", dtype="float32",
)
assert spring_env.integrator == "kvaerno3_native"
spring_env.close()
```

Without an explicit JAX spring configuration, selecting `"kvaerno3_native"`
constructs the compatible GMRES configuration. With an explicit `SpringConfig`,
the selector changes only its integrator: solver settings remain authoritative.
For example, selecting native Kvaerno with a dense configuration raises an error;
pass `SpringConfig(linear_solver="gmres")` to make that choice explicit. Selecting
the Diffrax path with GMRES likewise raises an error. Native factory selection
copies a supplied bundle's configuration and leaves the caller's bundle unchanged.
Unsupported names and model/backend/interface combinations raise construction
errors rather than substituting another method. `env.integrator` reports the
resolved choice.

All five families retain backward Euler as their default. Native Kvaerno uses
the already implemented three-stage spring method. The Diffrax option requires
optional `diffrax` and `optimistix` dependencies and remains an engine legacy
path; it is not exposed by registered presets. Selection does not add adaptive
timestepping, new methods, or a new residency guarantee. See the
[selection registry](../engine/numerics/selection.py),
[numerical options](NUMERICAL_BACKEND_OPTIONS.md), and
[action-holding contract](ACTION_HOLD.md).

Important compatibility rules:

- A 2D decoder cannot be passed to a 3D driver. Decoder and environment dtypes
  must match. MPM and beam reference cells must be uniform squares/cubes.
- In the JAX decoder and registered Gym embodiment pipelines, Poisson ratio is a
  spatial field bounded in `[0.25, 0.45]`. Lower-level native Warp bundle
  validators instead accept `-1 < nu < 0.5`; that wider admissible interval is
  not a claim of validation for every material. Body
  drivers no longer add a global scalar. An explicit legacy `poisson` key is
  still accepted as an override; remove it to use the local field or switch to
  spring. The spring lattice has no independent continuum Poisson control.
- Decoder coordinates are independent of physical discretization, but sampled
  topology, mass, actuator coverage, and numerical behavior can change when
  the driver or resolution changes. Validate and reset every new combination.
- Native channel counts must fit the action capacity. Large bodies need enough
  observation capacity. Shape, capacity, dtype, or static configuration changes
  require a new environment and new compiled executables.
- `BodyConfig.observation_profile="compact"` omits MPM `F/C` state and is
  explicitly partially observed. The full profile includes them. Beam
  observations include orientation and angular velocity.
- Body drivers use backward Euler and matrix-free Newton/GMRES. Their common
  factory selector accepts `integrator="backward_euler"`; `BodyConfig` does not
  gain additional methods. MPM's `explicit_trial` is a tiny-timestep comparison
  implementation, not a selectable production driver.
- Spring supports `backward_euler`, `kvaerno3`, and `kvaerno3_native` in
  `SpringConfig`. The Diffrax `kvaerno3` path requires `linear_solver="dense"`;
  the native version requires `"gmres"`. Dense spring solves have a small
  enforced capacity. A supported numerical option still needs its own CUDA
  certificate before resident production use.
- `DynamicsConfig.actuation_model="active_stress"` is currently MPM-only.
  It allows `actuator_tau=0` for direct commands. Other active-strain paths
  require a positive actuator time constant. General self-contact is absent;
  the implemented contacts are against terrain.

## Embodiments, representations, and generation

`engine/embodiments/decoders.py` defines the reusable decoders: an embedding
dictionary and sample coordinates become occupancy, material, and actuator fields.
`engine/embodiments/morphology.py` creates reproducible random initial embeddings using
those decoders, geometric priors, and rejection checks. Random generation runs
before optimization; the decoder is the part used in eligible differentiable
rollouts. `engine/embodiments/fields.py` supplies shared field interpretation and
validation. None of these modules integrates dynamics.

The separate `registered/embodiment.py` serves the registered Gym wrapper: it
packs and validates a flat vector of per-cell physical properties. It does not
implement the latent decoders. The completed naming and diagnostic organization
work is recorded in the [post-validation backlog](POST_VALIDATION_BACKLOG.md).

The common decoder output is `Fields(occupancy, young, density, strength,
direction, actuator_type, channel, poisson)`. It is evaluated at normalized material
coordinates in `[-1, 1]^dim`; geometry scores are not generally signed distances.
Implement the `Decoder` protocol in `embodiments/decoders.py` to add a representation:
`spec`, `dtype`, `space()`, `base()`, `decode(...)`, `descriptor()`, and
`structural_errors(...)`.

| Factory decoder name | Representation |
| --- | --- |
| `direct` | Bounded coefficient grid, interpolated to material sites; optional integer labels |
| `bspline` | Cubic B-spline coefficient grid |
| `cppn` | Feed-forward coordinate network with an explicit acyclic edge structure |
| `primitives` | Bounded collections of geometric primitives and field features |

Pass a name using `make(..., decoder="bspline")`, or pass a decoder object for
custom grid sizes, channel counts, temperatures, or other options. Do not swap
the dictionary of one representation into another: their schemas differ.

`design_mode="hard"` gives binary occupancy. `"relaxed"` gives a differentiable
occupancy surrogate with a positive material floor, so apparent void still has
mass/stiffness. `material_mode="continuous"` uses continuous Young's modulus;
`"categorical"` uses a three-material palette, with soft mixing under relaxed
design. Actuator type and routing use discrete choices even in the generic
relaxed decoder. Hard thresholds, integer labels, and topology changes are not
ordinary continuous design variables.

Edit a copied embodiment, then call `validate_embodiment` and `reset`. The
validation checks sampled fields and topology; it does not guarantee successful
settling or useful locomotion. Numerical tolerances, timestep, solver choice,
and grid capacity are configuration, not embodiment parameters.

`morphology.generate_random_design` produces decoder-native random bodies:

```python
from envs.soft_walker.engine.embodiments.morphology import generate_random_design
from envs.soft_walker.engine.embodiments.serialization import save_design, load_design

decoder, design, report = generate_random_design(
    "primitives", dim=2, seed=7, grid_shape=(6, 4),
    channels=4, dtype="float32", max_attempts=100,
)
save_design("my_design.json", design, decoder, metadata={"generation": report})
design, decoder, metadata = load_design("my_design.json")
```

Generation checks both cell-center and 2x MPM sample lattices, rejects unsuitable
candidates, and can raise `GenerationError` if its attempt limit is exhausted.
It runs sampling on CPU and does not simulate dynamics. To use this generated
dictionary with a body driver, keep its decoded Poisson head; to use it with
spring, retain only decoder fields. Configure
the physical envelope and sampling resolution explicitly in the new environment.

[Spatial material programming and terrain contact](SPATIAL_MATERIALS_AND_TERRAIN.md)
describes the new Poisson coordinates for each generator and legacy design migration.

`save_design/load_design` preserve numeric arrays, dtypes, shapes, and decoder
descriptors in JSON. Automatic reconstruction covers the four generic decoder
names above. For specialized/custom decoders, pass the matching instance to
`load_design(path, decoder=...)`; implementing a descriptor alone does not add
automatic reconstruction to the registry.

Other supplied starting points are `locomotion.robots.LeggedDecoder/build_demo`
for fixed chambered bipeds/quadrupeds, `scenarios.embodiments.ChamberDecoder` for
scenario representations, and `locomotion.implicit_design.make_implicit_design`
for relaxed CPPN designs. Their schemas and controllers are specific to those
builders. A seed labelled walk, trot, or pronk describes its initialization,
not demonstrated gait quality.

## Controllers and terrain

`locomotion.controllers.periodic_action(parameters, time, period)` takes
`(channels, 1 + 2*harmonics)` coefficients: bias followed by sine/cosine pairs.
It returns native-channel actions through one final `tanh`. Hold each action
for one control interval. Pad it before calling the public `env.step`; pure
resident controllers return exactly `env.channels` entries.

Scenario controllers (`experiments/scenarios/controllers.py`) provide `fourier`, `pulse`,
`feedback`, and `neural`, initialized by `seed_parameters(case)`. Feedback and
neural variants require the supplied public observation; their implemented
feedback inputs are COM velocity and phase features. They are small policy
examples, not a built-in RL training library. For another differentiable policy,
`differentiation.bind_policy` accepts
`policy(parameters, observation, carry) -> (bounded_action, next_carry)`.

`terrain.TerrainConfig` supports `flat`, `slopes`, and `blocks`, including
explicit slopes and rounded-box placements. Body drivers take it in
`BodyConfig.terrain`; spring takes it in `SpringConfig.terrain` (or uses its
older flat ground settings when this is `None`). A reset seed controls terrain
generation reproducibly, and accepted reset information saves the realized
terrain and seed. It does not seed arbitrary external controllers or random
design generation: those have their own seeds.

Scenario presets add uphill/downhill, 3D cross-slope, blocks, stairs, and hurdles;
see `presets/scenarios/config.py`, `presets/terrain/scenario_courses.py`, and
`terrain/recipes.py`. Editor preset generation is in `presets/terrain/editor_courses.py`. Cross-slope
requires 3D. Presets are built from the same terrain/contact implementation.
Changing contact stiffness, friction, material, or actuator law changes the
physical problem; changing timestep or solver tolerance changes its numerical
approximation. Neither should be silently treated as the same experiment.

## A standard Gymnasium adapter

The package inherits actual Gymnasium classes and uses its spaces, but stock
Gymnasium expects reset to return `(observation, info)`. Use a small adapter
before handing the environment to an RL library. The following class is example
application code, not an already-exported package class. Its rejected-design
policy is to raise an error; it never substitutes another body or an old state.
The method signatures follow the [Gymnasium Env API](https://gymnasium.farama.org/api/env/).

```python
from copy import deepcopy
import gymnasium as gym
from envs.soft_walker.engine import make

class GymnasiumAdapter(gym.Env):
    def __init__(self, env_id="SoftSpring2D-v0", **kwargs):
        self.inner, self.embodiment = make(env_id, **kwargs)
        self.action_space = self.inner.action_space
        self.observation_space = self.inner.observation_space
        self.metadata = dict(self.inner.metadata)
        self.metadata["jax"] = False  # This adapter exposes NumPy arrays.
        self.render_mode = self.inner.render_mode

    @staticmethod
    def _info(info):
        info = dict(info)
        # Wall-clock diagnostics are nondeterministic; keep physics diagnostics.
        info.pop("timing", None)
        info.pop("execution_seconds", None)
        return info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = {} if options is None else dict(options)
        candidate = deepcopy(options.pop("embodiment", self.embodiment))
        if options:
            raise ValueError(f"Unsupported reset options: {tuple(options)}")
        obs, rejected, info = self.inner.reset(seed=seed, embodiment=candidate)
        if bool(rejected[0]):
            raise ValueError(f"Embodiment/reset rejected: {info['embodiment_errors']}")
        self.embodiment = candidate
        return obs, self._info(info)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.inner.step(action)
        return obs, reward, terminated, truncated, self._info(info)

    def render(self):
        return self.inner.render()

    def close(self):
        self.inner.close()
```

For example, after the small CPU example's imports and `cfg` definition:

```python
adapter = GymnasiumAdapter(
    config=cfg, device="cpu", dtype="float32",
    max_action_space_size=4, max_observation_space_size=128,
)
try:
    obs, info = adapter.reset(seed=7)
    action = np.zeros(adapter.action_space.shape, adapter.action_space.dtype)
    obs, reward, terminated, truncated, info = adapter.step(action)
    candidate = deepcopy(adapter.embodiment)
    obs, info = adapter.reset(seed=8, options={"embodiment": candidate})
finally:
    adapter.close()
```

For `gym.make`, place the adapter in an importable application module and
register a distinct ID, e.g. `gym.register("CodesignSpring2D-v0",
entry_point="my_adapter:GymnasiumAdapter")`. Supply `config`, `device`, and
capacities through `gym.make`. Registration and custom-environment conventions
are described in [Gymnasium's environment guide](https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/).
The underlying driver already truncates at `config.horizon`; use additional
time-limit wrappers only when that extra episode policy is intentional.

The co-design `make_vec` returns `(vector_env, batched_embodiment)` and retains
the three-value reset contract. JAX uses `vmap`; native Warp uses independent
CUDA graphs on concurrent streams when `execution="auto"` or `"parallel"`.
Both preserve next-step autoreset and support masked resets. Invalid selected
embodiments invalidate only those lanes; subsequent batch stepping requires every
non-autoresetting lane to have a valid reset.

Use the package's dedicated vector adapter with ordinary Gymnasium consumers:

```python
from envs.soft_walker.engine import make_gymnasium_vec

env = make_gymnasium_vec(
    "SoftSpring2D-v0", num_envs=4, backend="jax", device="cpu",
    dtype="float32", config=cfg,
)
try:
    observation, info = env.reset(seed=7)
    action = np.zeros(env.action_space.shape, env.action_space.dtype)
    observation, reward, terminated, truncated, info = env.step(action)
finally:
    env.close()
```

The adapter preserves the backend's batched step rather than constructing one
scalar wrapper per lane. New designs go in `reset(options={"embodiment": design})`;
rejected designs raise and retain the backend's invalid-lane state. For Warp,
select `backend="warp", device="cuda", execution="parallel"` and a native bundle
instead of a JAX configuration. [Parallel environments](PARALLEL_ENVIRONMENTS.md)
describes execution modes, constraints, examples and throughput measurement.

## Differentiable rollouts versus public stepping

Native Warp now exposes [resident policy/loss callbacks](NATIVE_CALLBACKS.md)
and an [embodiment-to-native gradient bridge](EMBODIMENT_NATIVE_GRADIENTS.md).
The callback accepts Warp kernels; the bridge composes JAX assembly derivatives
with native adjoints at the episode boundary. Neither differentiates the NumPy
Gym step interface.

The Gymnasium adapter is for observation/action interaction. `env.step` stages
actions, synchronizes physics, and returns host NumPy observations and metrics.
Do not differentiate through that Python/Gym boundary or mistake a loop of
GPU-backed Gym steps for one resident episode.

For JAX differentiation, use `differentiation.bind_rollout(env)` or
`bind_policy(env, policy)`. These return pure functions over embeddings and
actions/parameters, including preparation and settling. Apply `jax.jit`,
`jax.grad`/`value_and_grad`, and `vmap` to these functions as appropriate.
Their ordinary JIT execution is not by itself proof of CUDA residency.

For certified whole-episode execution, use `execution.bind_episode`,
`bind_demo_episode`, or `bind_scenario_episode` to create an `EpisodeProgram`.
The custom binder takes separate action, measurement, recording, running-cost,
and terminal-loss functions. Its measurement vector must put physical time in
entry zero: the completion check compares it with `steps * control_dt`.

`EpisodeProgram.loss/forward/record` take `(embedding, controller_parameters)`
and include settling. `value_and_grad()` defaults to controller derivatives;
use `argnums=(0, 1)` for appropriate continuous design and controller inputs.
Generic embeddings can contain integer/discrete leaves, so a joint optimizer
usually needs an explicit continuous parameterization rather than blindly
differentiating the whole dictionary. `fixed_*` functions accept a prepared
`(model, initial_state, parameters)` when reuse of a settled state is intended.

The required sequence is: bind a **raw whole function**, transform it for the
desired gradient/population, `compile_program(..., policy=ResidentPolicy(...))`,
stage device inputs, then `compiled.certify(...)`. Production calls are enabled
only after certification. CPU certificates check numerics and explicitly make
no GPU-residency claim. CUDA certification inspects actual host submissions;
one graph can still contain many GPU kernels, nonlinear iterations, and adjoint
solves. A component/configuration change needs fresh validation and certification.

The demo/scenario CLIs distinguish `--execution resident` from
`--execution legacy`; see [RESIDENT_COMMANDS.md](RESIDENT_COMMANDS.md). Legacy
paths remain useful diagnostic/reference executions. Resident paths reuse the
same physics through a different compilation boundary. Neither option swaps
an implicit model for an explicit model or makes a successful gait automatic.
The host may still run the optimizer, save outputs, and render between complete
simulation calls. The active `run_forward_experiments.py` worker uses resident
compiled record and value/gradient boundaries and a host Adam loop.

## Small extension examples

The contracts above are independently replaceable within the compatibility
limits in the matrix. The following examples show where changes belong.

**Representation and spatial materials.** A decoder owns its parameter schema;
common field interpretation lives in `embodiments/fields.py`. This example uses the
same field contract with B-splines and a bounded Poisson field. `set_poisson`
accepts a scalar for a uniform field; spatial variation requires editing the
appended Poisson coefficients in the decoder's embedding schema. `Fields`
records sampled values; assembly decides which properties affect its mechanics.

```python
from envs.soft_walker.engine.embodiments.decoders import FieldSpec, make_decoder, set_poisson

material_decoder = make_decoder(
    "bspline", spec=FieldSpec(dim=2), dtype="float32",
)
material_design = set_poisson(material_decoder, material_decoder.base(), 0.30)
assert not material_decoder.structural_errors(material_design)
```

**Controllers and terrain.** A policy may keep its own recurrent carry, but it
returns bounded native-channel commands without modifying the environment.
The example is a stateless two-channel bias policy; replace its expression with
an observation-dependent network for feedback. A different terrain is a
configuration value, independent of that policy.

```python
import jax.numpy as jnp
from envs.soft_walker.engine.physics.terrain import TerrainConfig

def bias_policy(parameters, observation, carry):
    """Return native-channel commands; this example does not use sensors."""
    return jnp.tanh(parameters), carry

slope = TerrainConfig(kind="slopes", fixed_slopes=(0.08, 0.0))
```

Pass `slope` to a new `SpringConfig` or `BodyConfig`. For a scalar JAX environment,
`bind_policy(env, bias_policy)` returns a pure callable accepting
`(embedding, parameters, carry)`. Use two parameters only when `env.channels == 2`.
The physics and observation construction remain inside that binder.

**Integration and linear solvers.** Select an existing method with the factory's
`integrator` keyword as described above. Configure a JAX spring solver with
`SpringConfig(linear_solver="gmres")`, or replace
`BodyConfig.dynamics.newton.gmres_method` with `"incremental"`, `"batched"`, or
`"fused"`. These alternatives have different solver work and rounding behavior;
their convergence gates still apply. A new integrator must return the existing
state/diagnostics transaction, including failure rollback. See `physics/spring.py`
and `numerics/integrators.py`, then register only the implemented model/backend
combinations in `numerics/selection.py`. Adding a registry entry alone does not
implement a numerical method.
The optional spring Diffrax Kvaerno3 path also needs `diffrax` and `optimistix`
(`python -m pip install diffrax optimistix`); the integrated base installation
does not declare those optional packages.

**Losses, gradients, and optimization.** A custom loss returns one scalar from
the chosen pure rollout. Apply `jax.value_and_grad(loss)` before handing it to an
optimizer. `locomotion.adam_experiment.run_adam` consumes a callable returning
`(loss, gradient)` and host floating parameters; it saves checkpoints outside
the traced computation. To prototype an optimizer against that contract without
compiling physics, use this analytic objective:

```python
def quadratic_value_and_grad(parameters):
    """A deterministic optimizer-contract fixture, not a locomotion objective."""
    return float(np.sum(parameters**2)), 2 * parameters
```

`run_adam(quadratic_value_and_grad, np.array([0.2], dtype=np.float32), output,
iterations=1)` writes one accepted update plus the initial checkpoint. To resume,
pass the saved `last.npz` and the same objective/hyperparameters; `iterations`
counts additional updates. `adam_state.py` handles evaluation validation and
checkpoint I/O; `adam_experiment.py` retains the update and retry rules. Do not
replace its objective silently when switching solvers, bodies, or backends.

**Observation, serialization, and rendering.** Add a named observation profile
instead of inserting entries into an existing flat layout. Extend a custom
decoder's `descriptor()` and supply its instance to `load_design`; automatic
reconstruction requires an explicit registry implementation. A custom display
can use `Recording(directory).world[frame]` for saved Z-up positions and
`DetailGeometry(recording).edges` for recorded connectivity. `Skin` is cosmetic.
The viewer's `recording.py` needs only NumPy; optional Newton allocation and
interaction live in `view_newton.py`. `host_terrain.py` supplies the shared
display-height calculation; the viewer and scenario wrappers keep their own
input/default checks. It does not replace physical contact calculations.

Batch coordination, partial resets and action holding are interface contracts,
not alternate constitutive laws. Extend them above the scalar transition, as in
engine `drivers/vector_lifecycle.py`, native `warp_native/vector_env.py`, and the
registered `vector_env.py`; keep lane state, graph buffers, masks and
autoreset independent.

The registered wrapper takes its host setup from `config.py` and its input
validation from `contracts/batching.py`; device selection, settling, model ownership
and transitions stay in the native engine. Low-level JAX scalar reset
uses `Spring2DEnv.reset` with body-driver hooks for terrain preparation, rejection
reporting and accepted-state installation. The hooks preserve each family's
error strings, RNG draws, copy behavior and timing keys.

Native MPM and solid episodes share host checkpoint orchestration through
`warp_native/_episode_host.py`. Concrete episode classes still allocate their
own buffers and implement physical forward/reverse steps; their existing Warp
kernel factories stay in place. `_bundle_io.py` shares archive transport while
each bundle class owns its schema and validation. `_terrain.py` shares native
terrain validation/upload without sharing device storage between owners.

## Source map and verification scope

| Topic | Source and corresponding checks |
| --- | --- |
| Factory, reset/step, spaces, vector behavior | [spring_environment.py](../engine/drivers/spring_environment.py), `drivers/vector_lifecycle.py`, `spaces.py`; [contract tests](../../../../tests/soft_walker/test_contract.py) |
| Body construction, observations, driver dispatch | [environment.py](../engine/drivers/environment.py), [model_access.py](../engine/drivers/model_access.py), [reporting.py](../engine/drivers/reporting.py), `drivers/task.py`; [driver tests](../../../../tests/soft_walker/test_m3_drivers.py) |
| Numerical models and differentiation | `physics/{spring,beam,spring_tangent,beam_tangent,fem,mpm,material,contact}.py`; `numerics/{integrators,solvers,krylov}.py` |
| Generic decoders and embodiment files | `embodiments/{decoders,fields,constants,serialization}.py`; [decoder tests](../../../../tests/soft_walker/test_decoders.py), [reporting tests](../../../../tests/soft_walker/test_m3_reporting.py) |
| Random embodiment generation | `embodiments/{morphology,priors,precedents}.py`; [RANDOM_CREATURES.md](RANDOM_CREATURES.md) |
| Fixed robots and scenario components | `presets/locomotion.py`, `presets/gaits.py`, `presets/scenarios/{config,suites,build}.py`, `presets/embodiments/chambers.py`, `experiments/scenarios/controllers.py`, `presets/terrain/` |
| Pure rollouts and resident programs | `differentiation.py`, `body_rollout.py`, `execution/{program,compiled,policy}.py` |
| Shared host topology and display terrain | [host_topology.py](../host_topology.py), [host_terrain.py](../host_terrain.py); [host-helper checks](../../../../tests/soft_walker/test_shared_host_helpers.py) |
| Registered-task setup and batch inputs | [config.py](../config.py), [contracts/batching.py](../contracts/batching.py); the vector env keeps state and transitions |
| Native episode host lifecycle and bundle transport | `warp_native/{_episode_host,_bundle_io,_terrain}.py`; [ownership/schema checks](../../../../tests/soft_walker/test_native_host_consolidation.py) |
| Offline evidence and diagnostic commands | [diagnostics package](../diagnostics/README.md); [diagnostic regression checks](../../../../tests/soft_walker/test_diagnostic_consolidation.py) |

The [consolidation validation record](CONSOLIDATION_VALIDATION.md) documents the
2026-10-14 extraction checks against commit `2ec3bd1`, separately from the
historical verification below.

The guide's examples are deliberately small. Successful construction, a smoke
step, or sampled morphology validation does not establish accuracy, stable long
rollouts, gradient agreement, performance, or CUDA residency for another setup.
Use the saved physical/numerical protocol and the relevant validation documents
when extending an experiment.

Historical documentation verification on 2026-09-22 used the installed Ubuntu environment,
Gymnasium 1.2.3, and `JAX_PLATFORMS=cpu`: the spring reset/two-step example, MPM2
construction, and adapter reset/step/design-reset examples ran successfully.
`gymnasium.utils.env_checker.check_env(adapter, skip_render_check=True)` passed
for the small spring configuration, with the expected warnings about unbounded
observation-space limits. All five Python blocks compiled. The random-generation
and design-file block was checked against source but was not executed in this
documentation check. No GPU execution or RL training was performed.

The [cleanup validation report](CLEANUP_VALIDATION.md) records the original
readability checks. The [breaking cleanup report](BREAKING_CLEANUP_VALIDATION.md)
and [responsibility consolidation](RESPONSIBILITY_CLEANUP.md) record subsequent
package changes and their verification separately.
