# Embodiment gradients with native Warp physics

`EmbodimentEpisode` closes the chain rule from an embedding through field decoding
and physical assembly to the native episode loss. It uses the existing JAX decoder
and assembly, then Warp for every physical step, checkpoint replay, and implicit
adjoint. JAX applies the assembly VJP to the returned physical-array gradients.

This is an explicit host optimizer interface. It is not a JAX-transformable Warp
simulation callback. Host/device transfers occur at a whole-episode boundary;
they do not insert Python into the integrator or Newton/Krylov loop.

## Small complete example

From the repository root, with its `src` package installed or on `PYTHONPATH`:

```python
import numpy as np
from envs.soft_walker.engine import make
from envs.soft_walker.engine.physics.spring_config import SpringConfig
from envs.soft_walker.engine.warp_native.embodiment_bridge import EmbodimentEpisode

cfg = SpringConfig(
    grid_shape=(2, 1), size=(0.1, 0.05), control_dt=0.00025,
    substeps=1, settle_steps=0, horizon=4, ground=False,
)
env, body = make(
    "SoftSpring2D-v0", config=cfg, decoder="direct",
    design_mode="relaxed", dtype="float32", device="cpu",
)
try:
    commands = np.full((4, env.channels), 0.15, np.float32)
    episode = EmbodimentEpisode(
        env, body, commands, device="cpu", velocity_weight=0.7,
    )
    report, body_gradient, command_gradient = episode.evaluate(
        body, commands, gradient=True,
    )
    assert report["status"] == report["adjoint_status"] == 0
    assert body_gradient is not None and command_gradient is not None
    print(report["loss"], {key: value.shape for key, value in body_gradient.items()})
finally:
    env.close()
```

Change the native episode's `device` to `"cuda"` to move its physics to CUDA.
The environment's own `device` controls decoder/assembly placement. There is no
requirement to use the same device for both, but cross-device transfer has a cost.
For float64, enable JAX x64 before constructing the environment.

## Contract and restrictions

- The supplied environment describes a decoder, discretization, material laws,
  terrain, and solver configuration. The bridge uses its assembly routines; it
  does not call that environment's physics step. Keep it alive and unchanged while
  using the bridge: do not regenerate terrain through an external reset or replace
  its decoder, configuration or discretization. Construct a new bridge for those
  changes; `evaluate` is the supported body/command update boundary.
- `evaluate(body, commands, gradient=True)` returns the native result, a dictionary
  of cotangents for floating embedding leaves, and the command-table cotangent.
  With a `NativePolicy`, that last value is sensitivity to its generated held
  actions; the policy overwrites the supplied command table, so it is not a
  derivative with respect to the supplied values. Forward-only calls return
  `None` in the two gradient positions.
- Embedding keys, shapes, and dtypes stay fixed. Integer labels, connectivity,
  active masks, and discrete actuator routing stay fixed. A change is rejected
  before native storage updates. Relaxed geometry is appropriate for continuous
  geometry optimization; crossing a hard threshold is not an ordinary derivative.
- The pullback includes actual assembled coefficients. For example, beam Poisson
  and Young fields affect extension/bending matrices. Differentiating only saved
  material metadata would miss those terms; this bridge differentiates assembly.
- Invalid primal/adjoint solutions do not produce usable gradients. Existing
  native convergence, support-capacity, geometry, and finiteness gates still apply.
- Native export constraints remain explicit. Continuum exports require fixed steps
  (`DynamicsConfig(max_halvings=0)`). MPM additionally requires
  `newton=NewtonConfig(preconditioner_side="right")`; a different MPM side is
  rejected. Solid export preserves physical coefficients, tolerances and iteration
  budgets, while using native GMRES mechanics rather than reproducing a selected
  JAX dense/Krylov implementation. Non-MPM solids require active strain.
  Spring supports backward Euler and `kvaerno3_native`; the Diffrax-specific variant
  is not a native Warp integrator.
- Native storage and graph identities persist between valid parameter updates.
  The caller's body/command arrays are not mutated. The optimizer remains outside
  simulation; its checkpoint must retain parameters, moments, update count, and the
  same task/decoder/objective configuration.
- Native owners contain shared model/adjoint buffers. Do not overlap evaluations
  or parameter updates between episodes sharing an owner. Use independent owners
  for concurrent lanes.
- Custom native controllers and objectives use the separate
  [resident callback interface](NATIVE_CALLBACKS.md). Their parameter gradients
  are available through `episode.episode.gradients()` alongside resolved-array
  gradients. They are distinct parameter groups from the embedding dictionary.

The assembly/episode tests cover all five physics families, plus direct grid,
B-spline, CPPN, and primitive decoder pullbacks. Specialized decoders implementing
the same protocol can be used, but a test of one decoder/family does not establish
every possible combination. See
[the checks](../../../../tests/soft_walker/test_warp_embodiment_bridge.py) and
[the support matrix](CAPABILITIES.md).
