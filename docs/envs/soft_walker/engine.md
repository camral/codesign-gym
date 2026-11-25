# The soft-body engine

The `SoftWalker-*` ids run on the native Warp backend of Andrew Spielberg's soft-body engine (`envs.soft_walker.engine`).
You can also use the engine directly, without `codesign_gym`. It has two backends: a differentiable JAX backend and a
native Warp backend. This page tells you which interface to use and what each part of the engine does.

## The pipeline

The engine goes through these steps in sequence:

**embedding → sampled fields → physical assembly → controllers → integration and contact → observations, rewards, losses and gradients**

- The **decoder** describes the body, including spatial materials and actuator fields.
- The **physics family** sets the nodes or particles, the state and the mechanics.
- The **controllers** make actuator-channel commands. The optimizers use the loss gradients where they are available.

When you select a physics family, the discretization and the observation layout change with it. They are not
interchangeable between families.

## Choose an interface

| Interface | Body input | Lifecycle and execution |
|---|---|---|
| `codesign_gym.make_vec("SoftWalker-…")` | A flat, bounded per-cell design vector | Native Warp lanes, partial resets, next-step autoreset ([Spaces](spaces.md)) |
| JAX engine (`engine.make(..., backend="jax")`) | A decoder's embedding dictionary | Scalar or `jit(vmap)` vector transitions, and pure differentiable rollouts |
| Native Warp engine (`engine.make(..., backend="warp")`) | Resolved physical arrays in a model bundle | Fixed-topology native episodes and adjoints, with one CUDA graph for each vector lane |

The JAX backend is available only through the engine. The registered `SoftWalker-*` ids always use Warp.

The interfaces return different things, so do not mix them:

- The engine factories return `(environment, base_embodiment)`. Their reset returns `(observation, rejected, info)`.
- The engine's `make_gymnasium_vec` adapter returns the standard Gymnasium `(observation, info)` from reset.
- `codesign_gym.make_vec` follows the [Codesign Gym API](../../concepts.md).

A Gym step returns NumPy values on the host. A step inside an autodiff transform is therefore not a differentiable
rollout. For gradients, use the engine's pure JAX bindings or the native episode and adjoint interfaces. To
differentiate through a native latent body, use the `EmbodimentEpisode` bridge, which goes through decoding and
assembly.

## What each module does

All paths are below `envs.soft_walker`.

| Module | What it does |
|---|---|
| `registration`, `vector_env`, `config`, `embodiment`, `render`, `render3d` | The registered `SoftWalker-*` task: the per-cell design, the presets and terrains, the Warp vector env and rendering |
| `contracts` | Array schemas, material bounds, metric columns and status codes. These do not depend on a backend. |
| `engine.embodiments` | Decoders, sampled fields, body generation and persistence |
| `engine.physics`, `engine.numerics` | Mechanics and contact; numerical integration and solvers |
| `engine.drivers`, `engine.execution`, `engine.warp_native` | The runtime lifecycle, compiled execution, and native models and kernels |
| `experiments` | Optimization and checkpoints, rollout objectives, assessments and resident adapters |
| `presets` | Named chamber bodies, gait seeds, scenario suites and terrain courses |
| `terrain` | Host-only terrain recipes: validation and conversion |
| `viewer` | Saved recordings, preview images and the optional interactive viewer |
| `diagnostics` | Offline qualification, benchmarks and evidence collection |

The experiments and presets use the engine. The engine does not import them, and it does not import the registered task.

`engine.action_timing` sets how often a command changes. When you change the number of physics steps for each action,
the physical timestep stays the same. The registered task also rounds its action horizon up so that the episode keeps
its target duration.

## Supported choices and limits

| Component | Choices and limits |
|---|---|
| Physics | Spring 2D, MPM 2D and 3D, beam 3D, Hex8 FEM 3D. Both backends have all five families. |
| Integration | Backward Euler for all families. Native Kvaerno3 stages for the spring family on both backends (`integrator="kvaerno3_native"`, engine factories only; the `SoftWalker-*` ids always use backward Euler). The older Diffrax Kvaerno3 is JAX-only and spring-only, and needs optional packages and dense solves. |
| Precision and device | float32 or float64, CPU or CUDA. Test coverage differs for each fixture. For float64 on JAX, enable JAX x64 before you import JAX. Codesign Gym never sets it process-wide. |
| Embodiment | Direct grid, B-spline, CPPN, primitives and special decoders. Native episodes need a fixed topology and routing. |
| Materials | Spatial occupancy, density, elasticity, Poisson's ratio and actuator fields. The spring family stores Poisson's ratio but has no independent continuum Poisson control. |
| Terrain | Planes, slopes, rounded boxes and combinations of them. A course preset does not guarantee that a trajectory converges. |
| Control | Open-loop, feedback and recurrent interfaces, each with its own backend contract. The resident JAX action callback has no separate input for a recurrent carry. |
| Batching | JAX transformed batches or independent native graphs and streams. Both have selective vector resets and action holding. |
| Rendering | Host previews and saved recordings. Newton is installed with the engine; without OpenGL, rendering falls back to the 2D view. The display surfaces are not the solver geometry. |

"Supported" means that an implementation exists within its documented limits. It does not mean that every combination
was validated. The [capability matrix](../../../src/envs/soft_walker/docs/CAPABILITIES.md)
tells you which combinations are supported, partial, not implemented, or not yet validated. General self-contact and
imported collision meshes are not implemented. A GPU implementation or a short unit test does not guarantee a speed.

For extension examples, array shapes, file formats and reset variants, read the [authored guides](guides.md).
