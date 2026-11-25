# Optional local-work numerical paths

JAX and native Warp solve the supported physical equations through different
layouts and solver implementations. These options control repeated linear work;
they do not change material laws, timesteps, or acceptance tolerances. Their
usefulness depends on body size, occupied support, precision, contact, and Krylov
iterations. Existing defaults remain unchanged.

Integrator selection is separate from these solver-work options. The common
factory keyword `integrator="backward_euler"` selects the existing method for all
five families on JAX or Warp; `"kvaerno3_native"` selects existing spring stages
on either backend, including registered Gym factories. Legacy `"kvaerno3"`
remains a JAX-engine-only Diffrax option with dense solves. `None` preserves the
configuration/default. Explicit JAX spring solver settings are preserved and
must be compatible; only an omitted configuration lets the native selection
construct the GMRES default. Unsupported combinations fail explicitly. See the
[selection contract](COMPONENTS_AND_GYMNASIUM.md#selecting-an-integrator) and
[host-only availability registry](../engine/numerics/selection.py).

| JAX setting | Owner / default | Contract |
|---|---|---|
| `mpm_compact_nodes` | `DynamicsConfig`, `0` | Zero retains full-grid storage. A positive integer bounds occupied stencil support. Overflow returns `StepInfo.code=13` and rejects the step; it never truncates a physical solve. |
| `mpm_cache_tangent` | `DynamicsConfig`, `False` | Cache exact per-particle constitutive `dP/dF` blocks at each linearization point. |
| `mpm_block_preconditioner` | `DynamicsConfig`, `False` | Use exact node-diagonal 2×2/3×3 momentum blocks, including nonsymmetric contact, as a preconditioner; the residual still includes all couplings. |
| `mpm_preconditioner_update` | `DynamicsConfig`, `"predictor"` | Freeze at the predictor, or select `"newton"` to refresh each Newton iterate and the final root for the adjoint. |
| `fem_cache_tangent` | `DynamicsConfig`, `False` | Cache constitutive blocks at element quadrature points. |
| `beam_cache_tangent` | `DynamicsConfig`, `False` | Cache 12×24 endpoint-load derivatives; retain the exact SO(3) update derivative. |
| `spring_cache_tangent` | `SpringConfig`, `False` | Cache 2×4 edge-force position/velocity derivatives. |
| `linear_work_reuse` | `NewtonConfig`, `False` | Requires fused GMRES. Reuse its independently checked physical residual; refresh after refinement changes the iterate. |
| `gmres_linear_work_reuse` | `SpringConfig`, `False` | Forward that residual-reuse option to spring GMRES. |

These are local caches, not dense global Jacobians. Continuous material, actuator,
geometry, and initial-state derivatives remain in the chain rule. Preconditioner
construction is frozen for implicit differentiation; its transpose solve applies
the actual inverse-block transpose. Singular/nonfinite blocks fail the existing
true-residual gate without substituting a different solver.

```python
from dataclasses import replace
from envs.soft_walker.engine.physics.common import DynamicsConfig

physical = DynamicsConfig()
numerical = replace(
    physical,
    mpm_compact_nodes=256,  # Capacity chosen for this body's occupied stencil support.
    newton=replace(physical.newton, gmres_method="fused", linear_work_reuse=True),
)
```

Capacity is problem-specific. A moving body can need more support than its initial
shape. Inspect status rather than assuming initial occupancy proves rollout coverage.

Native `MPMConfig` exposes `compact_nodes`, `cache_tangent`, `block_preconditioner`,
and `linear_work_reuse`; `SolidConfig` has native local-tangent/Krylov settings.
MPM's node-block preconditioner requires `cache_tangent=True`. Native solids
require `cache_tangent=True` and `krylov_method="adaptive_dynamic"`; compact-grid,
node-block and mass-weighted options apply only to MPM. Native beam tangents use
autodiff; analytic tangent assembly is available for spring and FEM.
Cache layouts and capacity accounting are backend-specific. Native spring
`integrator="kvaerno3_native"` uses three implicit stage solves through `SolidEpisode`.
Direct backward-Euler `SolidStep` rejects that selection. Both engines use the
unchanged coefficients in the host-only `engine/numerics/integration_tableaux.py`.

Native export is a fixed-step boundary: enabled continuum bisection is explicitly
rejected. Active stress is an MPM law, not a spring/beam/FEM actuation switch.
Backend solver mechanics are not interchangeable configuration objects; select
their settings explicitly and check diagnostics.

Run a matched warm comparison with:

```sh
python -m envs.soft_walker.diagnostics.benchmark_jax_local_work \
  --device cuda --dtype float64 --steps 8 --repeats 7 \
  --out runs/jax_local_work.json
```

This holds body, contact, commands, materials, timestep and convergence settings
fixed. It checks accepted states and command/material gradients across full-grid,
compact, tangent-cache, residual-reuse and both block-refresh modes. Compilation
is separate from synchronized warm forward and value-and-gradient time. The short
deformed-biped fixture measures solver work; it is not a walking-success claim.
See the [work record](BACKEND_PARITY_WORK.md) for validation outcomes.
