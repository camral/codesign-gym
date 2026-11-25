# Native Warp policies, objectives, and embodiment gradients

`engine.warp_native.episode.Episode` (MPM2/MPM3) and
`engine.warp_native.solid_episode.SolidEpisode` (spring/beam/FEM) accept optional
`policy=NativePolicy(...)` and `objective=NativeObjective(...)` arguments. These
callbacks are **Warp kernels**. Python constructs the graph; callback evaluation,
physics, convergence checks, checkpoint replay, and implicit reverse solves stay
on the CUDA device during a captured evaluation. A Python/JAX policy cannot be
passed in place of a Warp kernel.

The existing bounded command table and built-in travel/velocity/work objective
remain the defaults. Enabling either callback does not require enabling the other.
State positions and velocities use SI units. Spring, beam, and FEM state stores
displacement `u`; MPM stores absolute particle position `x`. Beam additionally
stores rotation matrices and angular velocity. Actuator commands are dimensionless
and must be finite in `[-1, 1]`.

## Kernel contracts

Create kernels using the selected owner's evaluated `owner.k.real`, `Model`, and
`State` types. They are precision/physics-specific Warp types. Do not enable
postponed Python annotations in a file that defines these kernels.

| Callback | Arguments in order | Output contract |
|---|---|---|
| Policy | `model, state, parameters, carry, index, command, next_carry` | Write every action channel and every carry entry. |
| Running cost | `model, initial, previous, next_state, command, parameters, clock, cost` | Write/add one scalar cost per physics step. |
| Terminal objective | `model, initial, final, parameters, accumulated_cost, loss` | Write/add the final scalar objective. |

Parameters and recurrent carry are finite, one-dimensional arrays in the owner's
precision. `index` is a one-element int32 array containing the zero-based control
interval. `clock` is a one-element int32 physics-step index, including settling.
Scalar cost/loss outputs are one-element arrays and are cleared before each launch;
parallel kernels can use `wp.atomic_add`. Treat model/state/parameter inputs as
borrowed and read-only. Callback output arrays belong to the episode.

`initial` is the owner's initial state **before settling**. Excluding settling
from the running-cost sum does not subtract it from the final state's work/time
ledgers. Define that reference explicitly when writing a terminal objective.

Launch dimensions accept a positive integer, a tuple of one to four positive
integers, `"channels"`, or `"nodes"`. The default is one thread. Empty recurrent
carry is supported. User kernels are responsible for writing all declared output
entries and using operations supported by Warp's reverse-mode differentiation.

## Action holding and reverse differentiation

The policy runs once at the start of each control interval, after settling.
`substeps=k` holds that command for `k` physics steps. Settling uses zero commands.
The device command table records actual policy outputs and can be downloaded after
evaluation for replay. Recurrent carry has a checkpoint at each control boundary;
reverse replay restores the physical state and matching carry before evaluating
the policy VJP. Physics steps still use converged-root adjoints, rather than
differentiating the Newton search history.

Running costs exclude settling by default. Pass `include_settling=True` when
constructing `NativeObjective` with its terminal kernel to include it.
Costs are **summed without an automatic `dt` factor**; multiply by
the timestep in the callback if the desired loss is a time integral. The terminal
kernel decides how accumulated cost contributes to the final loss.

After `episode.run(gradient=True)`, `episode.gradients()` retains existing
`commands`, `model`, and `initial` entries and adds applicable
`policy_parameters`, `policy_initial_carry`, and `objective_parameters` entries.
The command gradient is the local sensitivity to held actions; policy parameter
gradients also include state-feedback and recurrence paths. Direct running-cost
dependence on model, prior/next state, and initial state participates in the same
chain rule.

Invalid policy commands/carry mark the physical rollout failed. Failed physical
rollouts or nonfinite custom loss produce a nonfinite loss and cannot expose
successful gradients. Invalid kernel types, launch dimensions, parameter shapes,
and nonfinite parameters raise at setup/update. `set_callback_parameters` updates
values in existing buffers; topology, shapes, precision, and graph ownership stay
fixed. It invalidates access to old result gradients until another evaluation.
Episodes sharing a native owner also share its model/adjoint buffers; do not
overlap their evaluations or updates. Concurrent lanes need independent owners.

## Runnable resident-controller example

Save this as a Python file and run it with the project installed. Warp needs source
files for kernel inspection. It defaults to CPU; set `WARP_TEST_DEVICE=cuda:0` for
the conditional CUDA graph path. Initial compilation is separate from warm replay.

```python
import os
from dataclasses import replace

import numpy as np
import warp as wp

from envs.soft_walker.engine.warp_native.episode_callbacks import NativeObjective, NativePolicy
from envs.soft_walker.engine.warp_native.solid_builders import regular_solid
from envs.soft_walker.engine.warp_native.solid_episode import SolidEpisode
from envs.soft_walker.engine.warp_native.solid_model import SolidDeviceModel

bundle = regular_solid("spring", shape=(2, 1), precision="float64")
bundle.config = replace(bundle.config, contact_enabled=False)
owner = SolidDeviceModel(bundle, os.environ.get("WARP_TEST_DEVICE", "cpu"))
real, Model, State = owner.k.real, owner.k.Model, owner.k.State


@wp.kernel
def policy(model: Model, state: State, parameters: wp.array(dtype=real),
           carry: wp.array(dtype=real), index: wp.array(dtype=int),
           command: wp.array(dtype=real), next_carry: wp.array(dtype=real)):
    value = wp.tanh(parameters[0] + parameters[1] * state.v[0][0] + carry[0])
    for channel in range(command.shape[0]):
        command[channel] = value
    next_carry[0] = real(0.8) * carry[0] + real(0.02) * value


@wp.kernel
def objective(model: Model, initial: State, final: State,
              parameters: wp.array(dtype=real), accumulated_cost: wp.array(dtype=real),
              loss: wp.array(dtype=real)):
    loss[0] = -final.v[0][0] + parameters[0] * final.ledger[1]


episode = SolidEpisode(
    owner,
    np.zeros((3, owner.channels)),
    substeps=2,
    checkpoint_stride=2,
    policy=NativePolicy(policy, [0.15, -0.2], [0.0]),
    objective=NativeObjective(objective, [0.001]),
)
result = episode.run(gradient=True)
print(result["status"], result["loss"])
print(episode.gradients()["policy_parameters"])
```

## Decoder/assembly gradients around native physics

`engine.warp_native.embodiment_bridge.EmbodimentEpisode` supports a fixed-topology
JAX decoder/assembly boundary around native Warp physics. It works with the five
physics families and the existing direct, B-spline, CPPN, and primitive decoders.
JAX evaluates sampled fields and assembly, Warp evaluates the complete physical
episode, and a JAX assembly pullback maps resolved parameter cotangents to the
embedding. No JAX physics step is used by this bridge.

```python
import jax
import numpy as np

from envs.soft_walker.engine import make
from envs.soft_walker.engine.physics.spring_config import SpringConfig
from envs.soft_walker.engine.warp_native.embodiment_bridge import EmbodimentEpisode

jax.config.update("jax_enable_x64", True)
env, embedding = make(
    "SoftSpring2D-v0", dtype="float64", device="cpu", decoder="direct",
    design_mode="relaxed",
    config=SpringConfig(grid_shape=(2, 1), size=(0.1, 0.05),
                        control_dt=0.00025, substeps=1, settle_steps=0,
                        ground=False, gravity=3.0),
)
try:
    commands = np.full((2, env.channels), 0.15, dtype=np.float64)
    bridge = EmbodimentEpisode(env, embedding, commands, device="cpu")
    result, embedding_gradient, command_gradient = bridge.evaluate(
        embedding, commands, gradient=True,
    )
    print(result["loss"], sorted(embedding_gradient))
finally:
    env.close()
```

For other physics, use `SoftBeam3D-v0`, `SoftHexFEM3D-v0`, `SoftMPM2D-v0`, or
`SoftMPM3D-v0` with `BodyConfig`. Native export requires compatible fixed-step
settings, including continuum `max_halvings=0` and right-preconditioned MPM solves.
Spring supports explicitly selected backward Euler or `kvaerno3_native`; the
Diffrax `kvaerno3` path is not silently substituted.

This bridge is a host optimizer API: each complete evaluation validates inputs,
copies assembled arrays to native storage, downloads native gradients, and runs
the assembly pullback. It is not itself JAX-transformable or entirely transfer-free.
Shapes, masks, discrete actuator routes, and topology must remain fixed. Continuous
updates that change an active mask or discrete routing label are rejected. There
are no gradients through those discrete choices. `evaluate` returns
`(result, embedding_gradient, command_gradient)`; the last two are `None` for
forward-only evaluation. The JAX assembly environment must outlive the bridge.

## Native Kvaerno and selective reset

For a spring `SolidBundle`, set `integrator="kvaerno3_native"` explicitly via
`dataclasses.replace(bundle.config, ...)`. `SolidEpisode` and native environments
select the three-stage `SpringKvaernoStep`; direct `SolidStep` accepts backward
Euler only and rejects a mismatched integrator. Stage roots, adjoints, and actual
tableau displacement remain on device. Cached-root replay checks each restored
stage root. Failure in any stage rolls back the complete step. Native intermediate
stage geometry checks remain enabled, so its conservative accepted domain need not
match JAX exactly at inadmissible intermediate states.

`NativeVectorEnv.reset(options={"reset_mask": boolean_mask}, ...)` resets selected
lanes only. An integer seed expands using original lane indices; unselected RNG,
state, pending autoreset, and device ownership are preserved. Uninitialized,
unselected lanes stay unallocated. The mask must have shape `(num_envs,)` and
Boolean dtype. Scalar native reset does not accept this vector-only option.

The separate `engine.resolved_bundle.to_jax` boundary copies resolved coefficients
into a `ResolvedJaxBody` for the registered JAX wrapper. It preserves physical
arrays, work/time/status, and configured acceptance tolerances; it does not
reconstruct missing voxel material metadata. Spring/beam coordinate locks that
their JAX models cannot represent are rejected. Float64 conversion requires JAX
x64 enabled, and source NumPy buffers are never borrowed.

Validation references: `tests/soft_walker/test_warp_episode_callbacks.py`,
`test_warp_embodiment_bridge.py`, `test_warp_spring_kvaerno.py`,
`test_warp_native_partial_reset.py`, and `test_resolved_bundle_conversion.py`.
These tests establish their small fixtures, precisions, and devices; they are
not claims that arbitrary learned policies or physical timesteps converge.
