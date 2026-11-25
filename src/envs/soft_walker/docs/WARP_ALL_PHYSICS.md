# Native Warp: all five physical models

## What is actually implemented

This release adds Spring2D, Beam3D and HexFEM3D to the existing MPM2D/MPM3D
runtime. They are independent Warp implementations, not JAX wrappers. The
optional `solid_oracle` module and `export-demo` command intentionally use JAX
outside simulation to export a resolved existing design or check its equations.

| Model | Dynamics and spatial model | Current native integrator |
|---|---|---|
| Spring2D | Shared nodal masses, cross-braced springs, axial damping, preferred lengths | Backward Euler |
| Beam3D | Corotational midpoint Timoshenko energy; translation, rotation, inertia, axial/shear/bending/torsion | Backward Euler |
| HexFEM3D | Total-Lagrangian Hex8, eight Gauss points, compressible neo-Hookean energy, lumped masses | Backward Euler |
| MPM2D/3D | Existing uniform-reference APIC/MPM, active strain or active stress, compact grid solve | Backward Euler |

The new Lagrangian models match the equations in the JAX implementations. In
particular the beam is **not** replaced by an axial-spring approximation and the
Hex8 element is **not** reduced to one-point quadrature. The moderate-compressibility,
relative-rotation and unsupported self-contact limitations of those models remain.

Kvaerno integration is **not** included in the new Warp ports. The exporter
rejects a mismatching spring integrator rather than silently substituting Euler.
The native Lagrangian configurations explicitly require their implemented
cached-block/adaptive-GMRES path; MPM-only options are rejected there.

## Execution and derivatives

`SolidEpisode` captures a whole forward program or a whole value-and-gradient
program. Its outer timestep loops, nonlinear iterations, Krylov iterations,
checkpoint replay and reverse steps use device conditional loops on CUDA.
Python constructs the graph; replay submits one graph. CPU execution is a
correctness path and uses Warp's CPU interpretation of these loops.

An episode is **not one literal persistent CUDA kernel**. One complete interaction
is evaluated per thread: a spring, a beam, or a complete eight-point Hex8 element.
Local contact, global force accumulation, linear-system reductions and final node
updates have real dependency boundaries. No Python convergence poll is inserted
in a captured CUDA execution.

Newton linearization is stored as local sparse blocks: 4x4 per spring, 12x12 per
beam, 24x24 per hexahedron. Warp-generated local adjoints form rows of those
blocks in parallel from the analytic force. Krylov products gather/multiply/scatter
those blocks; the transpose uses the actual nonsymmetric transpose. There is no
global N-by-N dense matrix and no finite-difference force or Jacobian.

This first local-block formulation prioritizes exact equations and derivatives.
Local row assembly repeats some element work; its runtime and memory are reported
and should be profiled before claiming it is the fastest possible Hex8 assembly.
Each timestep's converged residual is differentiated implicitly. Newton's search
history is not differentiated. Both primal and transpose solves retain explicit
true-residual checks. Failed solves roll back state; failed gradients are not
returned as valid zeros.

For the zero-work derivative convention, separate solid adjoint targets, and
float32 MPM refinement behavior, see [gradient accuracy](NUMERICAL_ACCURACY.md).

The whole-episode path accepts **open-loop command tables**. Neural/feedback
policies can use ordinary API interaction, but that interaction is not advertised
as a single resident episode. A fused native feedback-policy API and a fused
multi-robot sparse solve are still separate work.

## Entry points and representation scope

```python
from envs.soft_walker.engine import make
from envs.soft_walker.engine.warp_native.solid_builders import regular_solid

bundle = regular_solid("beam", shape=(1, 2, 1), precision="float64")
env, embodiment = make(
    "SoftBeam3D-v0",
    backend="warp",
    device="cpu",                 # "cuda" is explicit; no fallback
    bundle=bundle,
    observation_profile="resolved",
    max_action_space_size=16,
    max_observation_space_size=4096,
)
obs, rejected, info = env.reset(embodiment=embodiment)
```

The existing factory/reset/step arities, masks, invalid-reset behavior and
warning-and-ignore transition embodiment behavior are retained. The Warp factory
routes before importing the JAX implementation.

The initial native observation profile is explicitly **resolved full state**.
It is not the JAX driver's policy-feature ordering. Its observation mask is
included. Existing policies need an observation adapter rather than an
undocumented claim of drop-in feature equivalence.

The native embodiment dictionary contains resolved continuous physical arrays
with fixed connectivity. This allows optimization of reference positions,
masses, spring/beam coefficients, FEM materials, actuator routing and initial
conditions. It is not an automatic derivative through the NumPy regular-grid
builder or a user CPPN/primitive/B-spline decoder. The original JAX decoder can be
used once to export a resolved body; differentiating that decoder natively is
outside this patch. Discrete topology/constraints remain discrete.

`make_vec(..., backend="warp", vectorization_mode="warp")` uses independent
environments with next-step autoreset. `execution="auto"` submits their CUDA
graphs on concurrent streams, then collects the batch; CPU uses serial execution.
`execution="parallel"` requires CUDA, and `"serial"` provides a comparison path.
Each lane retains independent Newton/Krylov tolerances and status. It is **not**
a fused multi-robot sparse solver. Measure workload-specific throughput with
`python -m envs.soft_walker.diagnostics.benchmark_parallel_envs`; large lanes may already saturate the GPU.
Use an environment's `episode(commands, ...)` to construct a whole
resident evaluation rather than calling Python `step` repeatedly for an optimizer.

## New MPM performance controls

- `compact_nodes=N` allocates at most N compact solver nodes. All particle support
  is still discovered. An exceeded capacity marks the step invalid on device,
  keeps every access in bounds, and rolls back. It never truncates a valid body or
  resizes through a host callback. Zero preserves the original capacity rule.
- `linear_work_reuse=True` explicitly starts the correction from zero, uses b
  instead of evaluating b-A0, and retains the independent residual when no
  refinement changed the solution. Any attempted refinement triggers a fresh
  check of the accepted solution, including when its candidate was rejected.
- `block_preconditioner=True` (requires `cache_tangent=True`) assembles the current
  node-diagonal 2x2/3x3 blocks. The adjoint uses their inverse transpose. A singular
  preconditioner block falls back to the previous scalar preconditioner and is
  counted. This changes only the preconditioner, not the equation or acceptance.

These options remain explicit in MPM defaults. The speed audit adds `linear_reuse`,
`compact`, `compact_reuse`, `block`, and `all` variants. Use the same timestep,
precision, command table and tolerances for comparison. A 128-node capacity is a
reasonable **experiment** for the recorded biped with at most 90 live nodes; it
is not a universal guarantee for subsequently optimized bodies.

BSR global assembly, mixed precision and a control-only adjoint were not silently
introduced. They remain measured alternatives after these conservative changes.

## Readable implementation map

- `solid_builders.py`: pure NumPy regular geometry and physical-array construction.
- `solid_model.py`: validated portable bundles and native device allocation.
- `solid_kernels.py`: explicit spring/beam/Hex8/contact equations and local sparse blocks.
- `solid_step.py`: safeguarded nonlinear solves and implicit VJPs.
- `solid_episode*.py`: whole forward/reverse scheduling and block checkpoints.
- `warp_native/env.py`: fixed-API wrapper and explicitly scoped observation profile.
- `solid_oracle.py`: optional JAX equations and one-time export; never a native callback.
- `warp_solids.py`: build, export, replay, recording and evidence archive.
- `warp_solid_worker.py`: equation parity, transpose checks, implicit gradients and FD ladders.
- `run_warp_ports_tests.py`: selected subprocess tests, timeouts, hashes and ZIP collection.

New code has expanded statements and line breaks; an isolated formatting pass
was checked for AST and comment-token equality. Numerical changes are listed
separately and are not mislabeled as formatting-only.

## Validation interpretation

The new tests compare original JAX forces/residuals at a native root, the native
commit function, Jacobian and transpose actions, and implicit control gradients.
Whole native episodes include settling and multiple controls; central differences
mix command and physical-material directions. These are small deterministic
physical problems, not newly validated walking gaits.

Float32 derivative probes explicitly tighten the primal absolute residual to
3e-8. The default 3e-6 residual can obscure a very small short-horizon derivative.
This is a **test configuration**, not a hidden default change; the worker records
it. Float32 ladders include a coarse truncation window and the fine roundoff
plateau. Do not infer long-horizon gradient accuracy from one good epsilon.

The GPU audit is a local acceptance check. CPU success does not establish CUDA
correctness, capture compatibility or speedup. The audit is run separately from
warm timing, without another CUPTI profiling session active.

## Recording and rendering

Native replay writes `trajectory.npz`, `model.npz`, `commands.npy`, `protocol.json`,
metrics, gradients when requested, and hashes. Frames are actual simulated states
at physical timestamps. `--animation` produces a small diagnostic GIF after the
complete simulation returns. It does not interpolate robot motion or claim a
walking/airborne qualification. Large production rendering belongs in the user's
renderer. Use checkpoint stride and memory budget explicitly; there is no hidden
host-streaming fallback during a resident run.
