# Common integrator selection

Engine factories accept `integrator` to select their existing integration
methods. The registered `SoftWalker-*` presets do not take it; they always use
backward Euler. This adds construction-time selection and
capability discovery. It does not add a numerical method, change timestep or
solver tolerances, or adapt a spring method to another physics family.

The implementation starts from local source checkpoint `e75d10d`. The focused
validation groups completed with 354 passes, no failures and no skips. This is
a grouped execution count; some contracts run in more than one precision/device.
The earlier [responsibility cleanup](RESPONSIBILITY_CLEANUP.md) remains a
separate, completed stage with its original results and performance limitations.

## Available methods

| Method | JAX engine | Native Warp engine |
| --- | --- | --- |
| `backward_euler` | Spring, beam, FEM, MPM2, MPM3 | Spring, beam, FEM, MPM2, MPM3 |
| `kvaerno3_native` | Spring, with GMRES | Spring native implicit stages |
| `kvaerno3` | Legacy spring Diffrax path, with dense solves | Unsupported |

`None` or omission preserves the existing configuration or default. Backward
Euler is each engine's default configuration and the only method of the registered presets. Unsupported selections raise `ValueError`; no method is silently
substituted. `env.integrator` exposes the resolved method on scalar/vector
environments and their Gymnasium adapter.

The host-only registry can be queried without initializing JAX, Warp or Diffrax:

```python
from envs.soft_walker.engine import available_integrators

assert available_integrators(backend="warp", physics="spring") == (
    "backward_euler", "kvaerno3_native",
)
```

The query describes implemented combinations. It does not inspect installed
optional packages, solver compatibility, or convergence for a particular body.
The Diffrax option requires separately installed `diffrax` and `optimistix`.

## Configuration and lifecycle

This selects the existing JAX native spring method with a compatible default:

```python
from envs.soft_walker.engine import make

env, body = make(
    "SoftSpring2D-v0", integrator="kvaerno3_native", device="cpu", dtype="float32",
)
assert env.integrator == "kvaerno3_native"
assert env.config.linear_solver == "gmres"
env.close()
```

The factory chooses GMRES automatically only when no spring configuration was
provided. For an explicit `SpringConfig`, every solver setting remains
authoritative: `integrator="kvaerno3_native"` with a dense configuration raises
an error, as does `integrator="kvaerno3"` with GMRES. Set
`SpringConfig(linear_solver="gmres")` explicitly when supplying other settings
for native Kvaerno. Reusing an already compatible configuration preserves its
identity. Changing the selector creates a replacement configuration.

For a supplied native solid bundle, an explicit selector replaces only the
copied configuration; the caller's configuration and physical arrays are
unchanged. Subsequent embodiment resets keep that selection. MPM remains
backward-Euler-only and gains no new serialized configuration field. Native
`SolidEpisode` dispatches the existing spring stages; low-level `SolidStep`
remains a backward-Euler primitive and rejects Kvaerno configuration.

The method's implicit stages are internal to one physics step.
`physics_steps_per_action` still counts complete physics steps sharing one
action, and `control_dt = physics_steps_per_action * physics_dt`. Selecting an
integrator changes neither that timing relation nor observation/reward schemas.
No additional adaptive-step, differentiation or residency guarantees follow
from selecting a method. See [action holding](ACTION_HOLD.md) and the
[component guide](COMPONENTS_AND_GYMNASIUM.md#selecting-an-integrator).

## Validation record

Commands, environment settings, logs and JUnit XML are recorded under
`workspace/runs/integrator_selection`. The compact release evidence directory is
`handoff/integrator_selection_validation`; the source inventory records the
source commit separately from the evidence snapshot. `source_freeze.json`
identifies the final production/test Python and build/typing configuration bytes.

| Group | Result | Evidence |
| --- | --- | --- |
| Native factory selection, copied-bundle ownership, defaults and rejection | 19 passed | `native_selection_host.xml` |
| Registered factory/layout/builder selection and package/helper contracts | 19 passed | `registered_selection_host.xml` |
| Existing CPU float64 methods, gradients, rollback, action holding and resets | 57 passed; no skips | `existing_cpu64.xml` |
| Existing CPU float32 regression selection | 49 passed; no skips | `existing_cpu32.xml` |
| New selector and registered numerical contracts, CPU float64 | 48 passed; no skips | `selection_cpu64.xml` |
| New selector and registered numerical contracts, CPU float32 | 48 passed; no skips | `selection_cpu32.xml` |
| New selector and registered numerical contracts, CUDA float64 | 46 passed; no skips | `selection_cuda64.xml` |
| New selector and registered numerical contracts, CUDA float32 | 48 passed; no skips | `selection_cuda32.xml` |
| Existing CUDA float64 native stages, action holding and reset selection | 20 passed; no skips | `existing_cuda64.xml` |
| Scoped formatting/lint and host typing | Ruff: 351 files; mypy: 227 source files passed | `scoped_ruff.json`, `scoped_ruff.log`, `mypy.log` |
| Documentation links and added examples | 250 local links and 17 Python snippets checked; all four new query/construction examples executed on CPU float32 | `documentation_checks.json`, `documentation_examples.json` |

The native source review compares all 42 native Python modules with `e75d10d`,
allowing only declared host selection/configuration/schema changes. Numerical
bodies and graph machinery match the checkpoint
(`native_selection_scope.json`). New tests compare selector-based JAX state and
action-gradient results with the original explicit-configuration route. Existing
tests retain their tolerances and solver settings.

The optional Diffrax checks executed successfully rather than being skipped.
There are no unexplained failures in these groups. CPU and CUDA float32/float64
selector checks include registered forward parity and spring action gradients;
the additional full native-stage/adjoint regression group was run on CUDA
float64. These selected checks do not constitute a fresh full-repository suite.

The broader source review covers 82 non-Warp modules: 77 match exactly and five
contain only the declared construction changes. Registered resolved JAX physics
and its six supporting modules are unchanged. Scoped Ruff checks pass for 351
files, and host mypy passes for 227 source files; the final evidence records
their checked source versions.

The native Kvaerno fixture explicitly places its JAX reference on CPU. Its CUDA
run therefore validates Warp CUDA against JAX CPU; separate new JAX selector
tests cover the JAX CUDA route. Passing a configuration check does not establish
numerical accuracy for every morphology, contact condition or timestep. No new
performance benchmark or speedup claim is part of this selector change.

The first CUDA float64 selector run collected before two CPU-only Gymnasium
adapter construction checks were added. Subsequent selector stages include
those two cases; this explains their different collection sizes.

Implementation and representative checks:

- [Host-only registry](../engine/numerics/selection.py) and
  [spring configuration selection](../engine/physics/spring_config.py).
- [JAX selector tests](../../../../tests/soft_walker/test_integrator_selection.py),
  and [native selection tests](../../../../tests/soft_walker/test_native_integrator_selection.py).
- [Native spring stages/adjoints](../../../../tests/soft_walker/test_warp_spring_kvaerno.py),
  and [action-hold equivalence](../../../../tests/soft_walker/test_action_hold.py).
