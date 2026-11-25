# Soft-robot capabilities and support boundaries

The [modularity matrix](MODULARITY_MATRIX.md) lists selectable components and
their alternatives. This page records support levels and validation limits.

This is the current implementation map, read together with the
[component guide](COMPONENTS_AND_GYMNASIUM.md). A feature being implemented does
not certify every body, timestep, precision, device, or optimization objective.
The linked tests describe concrete coverage; saved galleries and milestone
reports describe the experiments that produced those artifacts.

## Reading the status labels

- **Supported within stated constraints:** an exposed implementation with tests for the
  stated contract. Read the restrictions before composing components.
- **Experimental or partial:** a prototype, restricted implementation, or
  diagnostic path whose limitations affect how it can be used.
- **Not implemented:** no supported execution path provides this capability.
- **Validation not established:** a plausible combination lacks matching test or
  experiment evidence. This is different from a missing implementation.

CPU/CUDA and float32/float64 are selectable by both engines. Coverage is recorded
per test and run; an implementation-level check is not a blanket numerical or
performance guarantee. The registered presets default to native Warp, float64,
and CUDA when available. Standalone viewer dependencies are optional.

## Three public interfaces

| Interface | Body input | Execution and reset | Important boundary |
| --- | --- | --- | --- |
| JAX engine `make` / `make_vec` | A decoder's embedding dictionary | Pure JAX transitions; vector `jit(vmap)`; selective vector reset | Fields and observations depend on the selected driver |
| Native Warp engine `make` / `make_vec` | Resolved physical arrays in a native bundle | Conditional CUDA graphs; serial CPU reference execution; selective vector reset | Fixed topology; `EmbodimentEpisode` additionally composes decoder/assembly gradients with native adjoints |
| Registered `codesign_gym.make_vec("SoftWalker…")` | Flat bounded per-cell embodiment vector | Select `backend="warp"` (default) or `"jax"`; both supply selective reset, rendering, and co-design lifecycle | Shared resolved observation/reward layout; embodiment layout differs from engine embeddings |

Engine factories return `(environment, base_embodiment)`. Co-design reset returns
`(observation, rejected, info)`. `engine.make_gymnasium_vec` supplies standard
Gymnasium vector reset `(observation, info)` while retaining backend batching.
The engine IDs and registered SoftWalker IDs are separate namespaces.

Both interfaces accept `integrator`. `"backward_euler"` is available for all five
families on both backends; `"kvaerno3_native"` is spring-only on both backends.
The legacy Diffrax `"kvaerno3"` option is limited to the JAX engine, with dense
solves and optional dependencies; registered Gym factories reject it. Omitting
the selector, or passing `None`, preserves the existing configuration/default.
`env.integrator` exposes the resolved choice. The
[factory selection contract](COMPONENTS_AND_GYMNASIUM.md#selecting-an-integrator)
explains configuration precedence and the host-only availability query.
The [selector validation record](INTEGRATOR_SELECTION.md) distinguishes this
factory change from the earlier numerical-method evidence.

Native CUDA execution requires a runtime supporting conditional graphs; it does
not silently fall back to the CPU loop when that capability is unavailable.
Parallel vectors own independent graphs and solver buffers per lane.

Evidence: [factories](../engine/__init__.py), [registered builder](../registration.py),
[wrapper](../vector_env.py), [JAX vector tests](../../../../tests/soft_walker/test_parallel_gymnasium_jax.py),
[native vector tests](../../../../tests/soft_walker/test_warp_native_parallel_env.py),
[registered wrapper tests](../../../../tests/test_soft_walker_vector.py).

## JAX and Warp differences in both directions

Both engines implement spring 2D, beam 3D, hexahedral FEM 3D, and MPM 2D/3D,
including terrain contact, backward Euler, fixed-topology gradients, batching,
and action holding. The important differences are the surrounding interfaces
and the numerical implementation, not a missing physics family.

| Capability | JAX engine | Native Warp engine |
| --- | --- | --- |
| Latent embodiment decoders and gradients through decoding/assembly | Direct grid, B-spline, CPPN, primitives, and specialized decoders; eligible continuous parameters can participate in rollout gradients | `EmbodimentEpisode` composes the JAX decoder/assembly VJP with native physics adjoints; fixed topology/routing, host optimizer boundary |
| Closed-loop differentiable controllers | `bind_policy` supports observations and recurrent carry; eligible resident scenario programs support feedback/neural controllers | `NativePolicy` captures a Warp kernel with parameters and recurrent carry; evaluated once per held action and differentiated through checkpoint replay |
| Running and terminal loss callbacks | Resident episode builders accept custom callbacks | `NativeObjective` captures differentiable running/terminal Warp kernels and returns objective-parameter gradients |
| Spring Kvaerno3 integration | Diffrax/dense and JAX-native/GMRES variants | `integrator="kvaerno3_native"`: three native implicit stages, including state/material/control adjoints; spring only |
| Selective low-level vector reset | `reset_mask` implemented | `reset_mask` implemented, preserving untouched lane state, RNG and graph ownership |
| MPM active-support solver storage | Optional fixed-capacity compact stencil storage via `mpm_compact_nodes`; overflow fails explicitly | Compact active-support buffers, with an explicit capacity option and overflow rejection |
| MPM node-block preconditioner and linear-work reuse options | Optional exact node blocks with predictor or per-Newton refresh; true transpose preconditioning; fused-GMRES checked residual reuse | Optional 2x2/3x3 node blocks and checked reuse of linear-solve work |
| Local tangent caches | Optional MPM constitutive blocks and solid element caches; retain material/control derivatives | Cached local element blocks; analytic spring/FEM tangent option |
| Registered SoftWalker task presets | Not exposed: the registered task runs on Warp only; use the engine factories for JAX | Native graphs, the only registered backend |

The Warp storage/solver entries describe implemented performance mechanisms,
not a guarantee of better timing. JAX transformation-based populations and
Warp independent CUDA graphs/streams are different ways to batch environments;
neither is evidence that the other backend lacks batching or device residency.
The generic JAX recurrent-policy API and resident scenario API also have
different contracts: arbitrary recurrent carry is not a separate input of
`EpisodeProgram`'s action callback.

Warp gradients are not limited to controls: native adjoints include continuous
resolved model fields and initial states. Decoder gradients additionally require
the chain rule upstream of those arrays. For example, the NumPy solid builder
converts cell density into
nodal masses and beam Poisson/material values into extension/bending coefficients.
Export-once native differentiation does not propagate through that conversion;
the explicit `EmbodimentEpisode` bridge supplies this missing chain rule.
The beam force uses the resolved extension/bending coefficients; changing only
saved Young's-modulus/Poisson metadata is not equivalent to differentiating a
rebuilt beam. See [solid assembly](../engine/warp_native/solid_builders.py),
[resolved field schema](../engine/warp_native/solid_model.py), and
[beam force](../engine/warp_native/solid_kernels.py).

Source references: [differentiable JAX policies](../engine/differentiation.py),
[resident JAX programs](../engine/execution/program.py),
[native episode/adjoint scope](WARP_ALL_PHYSICS.md),
[native MPM solver](../engine/warp_native/step.py),
[native solid solver](../engine/warp_native/solid_step.py), and
[JAX MPM transfers](../engine/physics/mpm.py).

The [native callback contracts](NATIVE_CALLBACKS.md),
[embodiment gradient bridge](EMBODIMENT_NATIVE_GRADIENTS.md), and
[numerical options guide](NUMERICAL_BACKEND_OPTIONS.md) give concrete usage and
restrictions. [Parity validation](BACKEND_PARITY_WORK.md) records this subsequent
work separately from the completed readability cleanup evidence below.

## Physics, integration, and contact

| Capability | Status and implementations | Restrictions / evidence |
| --- | --- | --- |
| Spring 2D | Supported: JAX and Warp | Lattice springs; no independently programmable continuum Poisson response. [Mechanics](../../../../tests/soft_walker/test_mechanics.py) |
| Beam 3D | Supported: JAX and Warp | Translational and rotational state on the beam discretization. [Beam checks](../../../../tests/soft_walker/test_m3_terrain_beam.py) |
| Hexahedral FEM 3D | Supported: JAX and Warp | Structured hexahedral assembly, not an arbitrary tetrahedral mesh importer. [Driver checks](../../../../tests/soft_walker/test_m3_drivers.py) |
| MPM 2D and 3D | Supported: JAX and Warp | 2D is plane strain with thickness; grid extent and interpolation support must contain the motion. [Transfers](../../../../tests/soft_walker/test_m2_transfer.py), [native edge cases](../../../../tests/soft_walker/test_warp_all_mpm_edges.py) |
| Backward Euler | Supported across the five families | Engine factories accept `integrator="backward_euler"` on JAX and Warp; it is the only method of the registered `SoftWalker-*` presets. Newton/Krylov convergence is checked; an implicit step can still fail. [Solver checks](../../../../tests/soft_walker/test_m2_solvers.py) |
| Kvaerno3 spring integration | Supported with constraints in both engines; not exposed by the registered presets | Common selector `integrator="kvaerno3_native"` uses existing spring stages. JAX requires GMRES; when an explicit configuration is supplied, incompatible solver settings raise an error. Warp uses `SolidEpisode`; direct backward-Euler `SolidStep` still rejects this selection. Legacy `"kvaerno3"` requires dense solves and is JAX-engine-only. Other physics families reject both Kvaerno selections. [Registry](../engine/numerics/selection.py), [JAX](../../../../tests/soft_walker/test_m2_spring_paths.py), [Warp](../../../../tests/soft_walker/test_warp_spring_kvaerno.py) |
| Explicit MPM trial | Experimental/diagnostic | Tiny-step comparison code, not a selectable production environment backend |
| Terrain contact | Supported: planes, slopes, rounded boxes, compositions | Smooth contact/friction; all model families have terrain paths. [JAX](../../../../tests/soft_walker/test_nonflat_terrain.py), [Warp](../../../../tests/soft_walker/test_warp_native_nonflat_terrain.py) |
| Registered terrain presets | Supported: flat, stairs, gaps | Preset recipes do not guarantee convergence on arbitrary designs. [Wrapper terrain checks](../../../../tests/test_soft_walker_vector.py) |
| General self-contact, arbitrary collision meshes, hard contact, imported height fields | Not implemented | These are not implied by terrain rendering or a box composition |
| Arbitrary large-deformation/contact trajectory | Validation not established by a short unit test | Check solver status, geometry validity, grid support, and trajectory-specific evidence |

Changing physics may require a new discretization, observation capacity,
configuration, and compilation. Similar factory signatures do not make model
state, observation order, or gradients interchangeable.

The optional Diffrax variant requires separately installed `diffrax` and
`optimistix`; they are not part of the integrated base dependency list.

## Precision and device evidence

The subsequent [parity validation](BACKEND_PARITY_WORK.md) adds CPU and CUDA checks
in both precisions for the new numerical options, native callbacks, spring stages,
embodiment gradients and registered JAX environments. Final deduplicated results
are 1,326 CPU float64, 253 CPU float32, 191 CUDA float64 and 138 CUDA float32 passes,
with 13 documented skips and no unresolved failures. JAX numerical fixtures use
`jax.default_matmul_precision("highest")`. This remains fixture-specific evidence,
not certification of every decoder/terrain/controller/solver combination.

The earlier cleanup's historical test scope is recorded below; its outcomes and timings are in
the [validation report](CLEANUP_VALIDATION.md). These fixtures do not establish
every cross-product of decoder, terrain, controller, solver, and body size.

| Engine family | Dimensions | CPU float64 | CPU float32 | CUDA float64 | CUDA float32 |
| --- | --- | --- | --- | --- | --- |
| JAX spring | 2D | Engine suite | Selected numerical/contract checks | Forward, gradients, state/reward comparison | Not revalidated in this cleanup |
| JAX beam | 3D | Engine suite | Forward/action-hold checks | Forward, gradients, state/reward comparison | Not revalidated in this cleanup |
| JAX FEM | 3D | Engine suite | Forward/action-hold checks | Forward, gradients, state/reward comparison | Not revalidated in this cleanup |
| JAX MPM | 2D and 3D | Engine suite | Forward/action-hold checks | Forward, gradients, state/reward comparison | Not revalidated in this cleanup |
| Native Warp spring | 2D | Engine suite | Forward/action-hold/vector checks | Forward/adjoints, state/reward comparison, warm batch benchmark | Not revalidated in this cleanup |
| Native Warp beam | 3D | Engine suite | Forward/action-hold/vector checks | Forward/adjoints, state/reward comparison, warm batch benchmark | Not revalidated in this cleanup |
| Native Warp FEM | 3D | Engine suite | Forward/action-hold/vector checks | Forward/adjoints, state/reward comparison, warm batch benchmark | Not revalidated in this cleanup |
| Native Warp MPM | 2D and 3D | Engine suite | Forward/action-hold/vector checks | Forward/adjoints, state/reward comparison, warm batch benchmark | Not revalidated in this cleanup |
| Registered Gym wrapper | Spring/MPM2 2D; beam/FEM/MPM3 3D | Construction, non-slow contracts, rendering; focused action-hold/parallel tests | Float32 construction check; full numerical coverage not established | All-family action-hold/parallel checks; complete wrapper suite (long episodes use spring, terrain settling also uses MPM2) | Not revalidated in this cleanup |

Primary fixture references are [action holding](../../../../tests/soft_walker/test_action_hold.py),
[JAX driver/policy gradients](../../../../tests/soft_walker/test_m3_drivers.py),
[native forward/adjoint parity](../../../../tests/soft_walker/test_warp_additions.py),
[native vector execution](../../../../tests/soft_walker/test_warp_native_parallel_env.py),
and [registered wrapper contracts](../../../../tests/test_soft_walker_vector.py).
Long registered-wrapper CPU episodes are not part of the completed validation
scope; the tests explicitly recommend CUDA for those full episodes. Headless
image/geometry checks do not establish interactive Newton UI behavior.

## Embodiments and spatial properties

| Capability | Status | Boundary |
| --- | --- | --- |
| Direct grid, cubic B-spline, CPPN, geometric primitives | Supported in the JAX decoder pipeline in 2D/3D | Each decoder has its own embedding schema; dimensions must match the driver |
| Random creature generation | Supported | Produces sampled geometry/material/actuator fields and validation reports; a valid sample need not locomote |
| Chambered walkers and relaxed implicit designs | Supported specialized generators | They have specialized parameterizations and are not aliases for every generic decoder |
| Spatial occupancy, Young's modulus, density, actuator strength/direction/type/channel, Poisson ratio | Supported semantic fields | Which fields affect mechanics depends on the driver; spring has no independent continuum Poisson control |
| Continuous versus categorical materials | Supported | Relaxed material mixing is a surrogate; categorical choices are discrete in hard mode |
| Hard versus relaxed geometry | Supported with constraints | Relaxed void retains a positive material floor; hard thresholds/topology changes are not ordinary differentiable variables |
| Latent embodiment optimized with native Warp | Supported with constraints | `EmbodimentEpisode` decodes/assembles at the whole-episode boundary and maps native gradients upstream; integer topology/routing remain fixed. This host optimizer interface is not a `jax.jit` simulation callback |
| Arbitrary changes of dimension/topology during a step | Not implemented | Construct/reset an appropriate environment; do not reinterpret existing arrays |

See [decoder contracts](../engine/embodiments/decoders.py), [random creatures](RANDOM_CREATURES.md),
[spatial properties](SPATIAL_MATERIALS_AND_TERRAIN.md),
[decoder tests](../../../../tests/soft_walker/test_decoders.py), and
[Poisson generator checks](../../../../tests/soft_walker/test_spatial_poisson_generators.py).

## Controllers, losses, and optimization

| Capability | Status | Boundary |
| --- | --- | --- |
| Bounded actuator-channel commands | Supported by all engines | Commands are filtered/applied by the actuation model; they are not arbitrary external body forces |
| Fourier, pulse, feedback, neural scenario controllers | Supported in JAX scenario paths | Parameter shapes, sensor selection, and resident eligibility depend on the controller |
| Active strain | Supported model-dependent mechanics | Time constants and direction conventions belong to the selected model |
| Active stress | Supported for MPM | Not a general spring/beam/FEM actuation switch |
| Travel, hop, low-bounce, custom scalar objectives | Supported in the corresponding rollout builders | Objective values are not proof of flight or useful gait; inspect assessment outputs |
| Adam, gradient descent, L-BFGS-B and population scouting | Supported workflow options | Availability/resume semantics differ by command; preserve the task when resuming checkpoints |
| JAX gradients through continuous decoder fields and policies | Supported with constraints | Fixed structure and differentiable variables only; validity/convergence gates still apply |
| Native Warp physical-array/control adjoints | Supported with constraints | Supported fields and fixed topology are documented in [native physics](WARP_ALL_PHYSICS.md) |
| Native Warp differentiation through a JAX latent decoder | Supported with constraints | Explicit `EmbodimentEpisode` assembly pullback; JAX does not execute the native physics or differentiate a hidden JAX fallback. [Chain-rule checks](../../../../tests/soft_walker/test_warp_embodiment_bridge.py) |
| Terrain-parameter optimization | Validation not established in public optimization workflows | Normal JAX rollout builders bind the terrain; native episode adjoints omit terrain fields. Contact support alone does not establish terrain gradients |
| Gradients through registered Gym step/design assembly | Not implemented | The wrapper exposes NumPy interaction; use separate engine rollout/physical-array adjoint interfaces for derivatives |
| Native Warp whole-episode feedback/recurrent policy fusion | Supported with constraints | A `NativePolicy` Warp kernel runs inside the captured episode, including policy-parameter and initial-carry gradients. It does not directly accept a Python/JAX policy callable. [Callback checks](../../../../tests/soft_walker/test_warp_episode_callbacks.py) |
| Guaranteed walking/jumping from a gait name | Not implemented as a guarantee | Gait names specify seeds; successful locomotion requires measured trajectory evidence |

The provided implicit co-optimization workflow does not optimize every available
field: density, discrete type/channel routing, and hidden network structure are
fixed there. A field being programmable does not mean every optimizer exposes it.

Evidence: [scenario controllers](../experiments/scenarios/controllers.py),
[rollouts](../engine/differentiation.py),
[implicit design tests](../../../../tests/soft_walker/test_implicit_design.py),
[Adam continuation tests](../../../../tests/soft_walker/test_adam_experiment.py),
[Warp spring gradients](../../../../tests/soft_walker/test_warp_spring_full_gradient.py),
[MPM adjoints](../../../../tests/soft_walker/test_warp_mpm_adjoint_accuracy.py).

## Execution, interchange, and visualization

| Capability | Status | Boundary |
| --- | --- | --- |
| Batched JAX environments | Supported | Jitted `vmap`; shared static shapes/configuration with lane-specific values |
| Concurrent Warp CUDA environments | Supported | Separate lane graphs/streams and solver states, not one fused multi-robot sparse solve |
| Holding actions over physics substeps | Supported in both engines and registered presets | Keeps integration dt and settling; registered default remains 40 steps/action. [Timing semantics](ACTION_HOLD.md) |
| Device-resident JAX episodes | Supported for eligible programs | Residency certification belongs to the actual executable/device, not merely a `device="cuda"` flag |
| NumPy public step interface | Supported | Observations/rewards/reporting cross a host boundary; batching does not remove every host transfer |
| Standard Gymnasium vector adapter | Supported | Standard reset arity differs from the co-design API. [Batching guide](PARALLEL_ENVIRONMENTS.md) |
| Built-in decoder design JSON | Supported | Versioned formats migrate historical material schemas; custom decoders may need an explicit instance |
| Checkpoint resume | Supported by designated optimization commands | Requires matching task/design/objective and optimizer state; not arbitrary cross-backend resume |
| Diagnostic plots/GIFs and Newton playback | Supported | Newton is optional and playback follows saved samples; display skin does not add physics |
| Particle/element toggles | Supported where recording data exists | MPM displays saved particle centers; FEM boundary faces are not evidence of recorded interior elements |
| Registered cross-backend observation/reward schema | Not exposed | The registered task runs on Warp only; the engine factories share the resolved profile between backends. |
| Arbitrary cross-backend checkpoint interchange | Not implemented | Use documented engine schemas and explicit resolved-bundle conversion; optimizer/runtime state is not interchangeable by renaming its backend |

See [resident execution](RESIDENT_EXECUTION.md),
[serialization](../engine/embodiments/serialization.py), [viewer guide](../viewer/README.md),
[action-hold tests](../../../../tests/soft_walker/test_action_hold.py), and
[native batching checks](../../../../tests/soft_walker/test_warp_native_batch_executor.py).

## Updating this reference

When adding a feature, update its component contract, supported combinations,
limits, and test links together. Record device/precision and execution path with
new evidence. Do not promote a historical experiment or an untested combination
to a current support guarantee. [Cleanup validation](CLEANUP_VALIDATION.md) records
the checks performed during this readability refactor.
