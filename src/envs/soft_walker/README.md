# Co-design: source and visual results

This folder is ready to share. Open **[START_HERE.html](START_HERE.html)** to browse
the terrains, random creatures, Newton views, and optimization animations.
Keep the directory structure intact. The galleries need no Python installation.
If your browser restricts local pages, run `python -m http.server 8000` here and
open `http://localhost:8000/START_HERE.html`.

Those saved galleries belong to the full source checkout or shared results bundle.
The installable wheel and source distribution contain the simulator, examples,
small reference inputs and written guides; they omit `runs/`, `handoff/`,
`START_HERE.html` and the local viewer launchers. The source distribution also
includes the test sources and typing configuration. See
[distribution contents](docs/DOCUMENTATION.md#distribution-contents) before using
checkout-relative gallery, evidence or developer commands from an installed package.

## Current architecture and feature support

Start with [components and Gym integration](docs/COMPONENTS_AND_GYMNASIUM.md) for
the data flow, replaceable interfaces, and examples. The
[modularity matrix](docs/MODULARITY_MATRIX.md) lists configurable components,
their choices and interface-specific restrictions. The
[capability matrix](docs/CAPABILITIES.md) distinguishes the JAX engine, native
Warp engine, and registered SoftWalker wrapper (Warp only), including partial implementations,
unsupported features, and combinations without matching validation.

The [common integrator selector](docs/INTEGRATOR_SELECTION.md) explains selecting
existing methods consistently across the JAX engine, native Warp, and the Gym wrapper.

The [release organization guide](docs/RELEASE_ORGANIZATION.md) maps experiment,
preset and rendering imports to their current owners. The
[shared-helper migration](docs/SHARED_HELPER_MIGRATION.md) lists the utilities
collected in `shared.py` and their previous names. [Code conventions](docs/CODE_CONVENTIONS.md)
describes typing, formatting and compiled-kernel boundaries. The
[documentation guide](docs/DOCUMENTATION.md) explains where the engine API reference lives
(the Soft Walker section of the Codesign Gym docs) and how to build it.

The [release cleanup validation report](docs/RELEASE_CLEANUP_VALIDATION.md) records
numerical preservation, regression coverage, runtime comparisons and limitations.
The [pre-release sweep](docs/PRE_RELEASE_SWEEP.md) records the subsequent reset
fix, distribution checks and actual Newton/OpenGL viewer validation.

[Historical records](docs/HISTORY.md) collect earlier migrations, validation,
benchmarks and experiment reports separately from current usage guidance.
Dated sections and saved galleries below retain their historical provenance.

## Parallel environments

Native Warp vectors now submit each environment's CUDA graph on an independent
stream. All five physics drivers support `execution="auto"` (concurrent on CUDA,
serial on CPU), `"parallel"` (CUDA required), or `"serial"` for comparison. Each
environment keeps its own model, state and Newton/Krylov convergence criteria.
Registered SoftWalker presets preserve per-cell embodiments, partial resets and
rendering while using the same concurrent executor. The registered task runs on
Warp only; use the engine factories directly for JAX.

JAX continues to batch physics through `jit(vmap(...))`. For the standard
Gymnasium vector contract and wrappers, use
`envs.soft_walker.engine.make_gymnasium_vec`; it retains the backend's batched
step. See [parallel environments](docs/PARALLEL_ENVIRONMENTS.md) for both APIs,
lifecycle behavior, limits and a reproducible throughput benchmark.

Both engines accept `physics_steps_per_action=10` to hold a command across ten
resident integration steps while preserving the physics timestep and settling.
The registered SoftWalker presets also accept it and preserve their target
episode duration. Their existing default remains 40 steps per action.
See [action holding](docs/ACTION_HOLD.md) for timing semantics and matched-duration
benchmarks.

## Spatial materials and terrain update

The September 24 source update adds a spatial Poisson field to every built-in
design generator. MPM, FEM and beam assembly consume the local values; native
Warp beam construction now accepts nonuniform Poisson fields. The registered
native wrapper keeps its existing flat per-cell embodiment API, which already
includes Poisson values. Springs carry that field without claiming an independent
continuum Poisson control.

Spring terrain presets are now available as `SoftWalker-Spring2DStairs` and
`SoftWalker-Spring2DGaps`, using the shared terrain contact law. Slopes and rounded
steps have focused force/gradient and short-trajectory checks; accepting a terrain
does not guarantee long-horizon convergence on arbitrary obstacles.

See [spatial material programming](docs/SPATIAL_MATERIALS_AND_TERRAIN.md),
[nonflat contact](docs/NONFLAT_TERRAIN.md), and
[this integration's changes and checks](docs/SPATIAL_MATERIAL_INTEGRATION.md).
The four new `create_poisson_gallery.py`, `create_poisson_mechanics_demo.py`,
`create_spring_terrain_gallery.py`, and `build_material_terrain_gallery.py` tools
live in `src/envs/soft_walker/scripts/` and generate a fresh offline visual gallery.
Existing saved visual results below retain their original provenance.

## Contents

- `engine/`: reusable simulation, numerical methods and generic body decoders,
  importable as `envs.soft_walker.engine`. Its tests live in the codesign-gym
  repository at `tests/soft_walker/`. The `scripts/` command-line tools are included
  in this folder and use the integrated engine namespace.
- `shared.py`: shared optimization/checkpointing, artifact, controller, body-parameterization,
  and validation helpers, temporarily collected in one module.
- `experiments/`: task-specific rollout, objective, and resident execution workflows.
- `presets/`: named legged bodies, gait seeds, terrain courses and scenario suites.
- `terrain/`: portable terrain recipe validation; `examples/` and `baselines/`
  contain example inputs and recorded comparison references.
- `docs/`: model documentation and guides for using and extending the project.
- `diagnostics/`: offline audits, benchmarks and evidence tooling. Run their
  canonical modules or files; the former `scripts/` forwarding files were removed.
  See the [diagnostics guide](diagnostics/README.md) for command and import migration.
- `viewer/`: Newton rendering, the terrain editor, and launchers with an optional
  `-Python` override for your own Newton environment.
- `runs/`: all terrain render sets and robot galleries from the shared snapshot,
  optimization GIFs, saved designs and fields, trajectories, controller
  checkpoints, assessments, and selected GPU residency evidence.
- `handoff/`: dependency versions, captured experiment status, and file hashes.

The folder contains no Git repository, Git handoff patches, Codex/agent metadata,
editor configuration, virtual environment, or build cache. The handoff evidence
includes machine-local reproduction launchers; they document validation and are
not runtime dependencies or portable installation commands. The current source
includes the native Warp backends, subsequent solver
and batching updates, backend parity additions, and the readability cleanup
documented above. The visual
results below remain the historical September 22 snapshot.
Historical result metadata can contain original machine paths; these are
provenance, not required installation paths. Some older documents describe the
original workstation or omitted historical helper scripts. Use the setup and
entry points below for this release.

## Start with these guides

- [Components and Gymnasium integration](docs/COMPONENTS_AND_GYMNASIUM.md): the
  pipeline, swapping physics/embodiment generators/controllers, compatibility
  constraints, pure JAX rollouts, and a standard Gymnasium adapter example.
- [Random creature generation](docs/RANDOM_CREATURES.md): four representations
  in 2D and 3D, reusable designs, and gallery generation.
- [Terrain demo](docs/TERRAIN_DEMO.md) and [Newton viewer](viewer/README.md).
- [Performance handoff](docs/PERFORMANCE_HANDOFF.md): measured quadruped GPU
  timings, residency evidence, numerical method, and optimization candidates.

## Install the simulator

The simulator is part of the codesign-gym base install (Python 3.11 or later).
From the codesign-gym repository root:

```bash
python -m pip install -e .          # add '.[dev]' for pytest
pytest tests/soft_walker            # this simulator's test suite
python -m envs.soft_walker.diagnostics.run_warp_followup --help
```

For NVIDIA GPUs, use codesign-gym's `codesign-gym-install-jax-gpu` command (see the
repository README). `requirements-tested.txt` records the versions the included runs
were produced with (Python 3.13, jax 0.9.0.1); it is a provenance record, not an
install step.

The [current dependency upgrade record](../../../docs/newton-dependency-upgrade.md)
documents the Newton development / Warp 1.19 / JAX 0.10 checks. The older
`requirements-tested.txt` remains the provenance of the included historical runs.

The native Warp backend supports MPM in 2D/3D, springs, Hex8 FEM, and beams.
See [the integration update](docs/WARP_INTEGRATION_UPDATE.md) for commands,
the source layout, and numerical validation scope. The latest gradient repairs
and deterministic JAX option are described in [gradient accuracy](docs/NUMERICAL_ACCURACY.md). The existing
public API is `envs.soft_walker.engine.make` / `make_vec`; select the native
backend with `backend="warp"` and a resolved model bundle.

The optional interactive Newton renderer uses a separate environment. Its
observed versions are in [ENVIRONMENT.json](handoff/ENVIRONMENT.json). The
pre-rendered galleries do not need Newton, JAX, or a GPU. With Newton installed,
enter the package directory from the repository root before using the viewer
and saved-run paths below:

```bash
cd src/envs/soft_walker
python viewer/view_newton.py --run runs/forward_adam50/biped/best
python viewer/terrain_editor.py --help
```

## Rendering the registered presets (`render_mode="rgb_array"`)

`SoftWalker*` envs made with `render_mode="rgb_array"` return 3D frames (1280x800 RGB) drawn live by
`viewer/view_newton.py` through `render3d.py`: blue checkered floor, black pits under gap platforms, grey stairs,
a camera that tracks the body, particles for MPM and white-topped links for Beam3D. Newton (0.2.0) and pyglet
are base dependencies; the interactive viewer above additionally needs `imgui-bundle`.

Newton needs an OpenGL context. Without a display (e.g. a headless GPU box) it uses EGL. If no context can
be opened, `render()` warns once and returns the 2D Pillow side view (`render.py`, 720x420) instead.
To enable 3D frames on a headless Ubuntu NVIDIA box, install EGL matching the driver (check the flavour
with `dpkg -l | grep libnvidia-compute`):

```bash
sudo apt-get install -y libegl1 libnvidia-gl-595-server   # -server driver 595; use libnvidia-gl-<major> for desktop drivers
python -c "import pyglet; pyglet.options['headless'] = True; pyglet.window.Window(visible=False)"  # opens = EGL works
```

## What these results demonstrate

The results are the frozen September 22, 2026 snapshot from the previous ZIP:
the original biped and implicit 2D runs completed 50 Adam updates; the quadruped
has two recorded updates; implicit 3D passed its short initial GPU simulation
and gradient check; the quieter biped is paused at 11 updates. The implicit 2D
body moves forward by **tumbling**, both before and after hardening, rather than
walking upright. These experiments have different objectives and numerical
settings; inspect their saved protocols and assessments before comparing them.
Local background training may have advanced since this snapshot.

The simulator source includes later integrated updates and the current cleanup;
the animations above retain their original experimental provenance. Source and artifact hashes are in
[MANIFEST.json](handoff/MANIFEST.json) and `SHA256SUMS`; these describe this cleaned
folder. Historical per-run manifests describe their original evidence sets and
can refer to compiler dumps or logs that are not included here. The original
source-and-visuals ZIP remains available separately and was not modified.
