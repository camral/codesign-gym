# Holding actions over physics substeps

`physics_steps_per_action=k` holds one command for `k` integration steps, then
returns one observation, accumulated reward and termination result. Warp executes
the substeps inside its existing captured GPU graph; JAX uses its compiled scan.
Actuator filtering and physics still advance at every integration step.

```python
from envs.soft_walker.engine import make_gymnasium_vec

env = make_gymnasium_vec(
    'SoftSpring2D-v0', num_envs=4, backend='warp', device='cuda:0',
    dtype='float64', integrator='backward_euler', physics_steps_per_action=10,
)
print(env.physics_dt, env.control_dt, env.physics_steps_per_action)
observations, info = env.reset(seed=7)
observations, rewards, terminated, truncated, info = env.step(env.action_space.sample())
env.close()
```

Omit `backend='warp'` for JAX. The keyword is also accepted by `make` and
`make_vec`. Scalar, vector and standard Gymnasium adapters expose the three
timing attributes in the example. The ordinary co-design reset still returns
three values; `make_gymnasium_vec` uses the standard two-value reset.

`integrator` selects the method used by each physics step; action holding sets
how many such steps share one command. The common selector accepts
`"backward_euler"` for all five families on both backends and
`"kvaerno3_native"` for spring on both backends. The legacy Diffrax `"kvaerno3"`
choice is limited to the JAX spring engine. `None` or an omitted selector keeps
the existing configuration/default, and `env.integrator` reports the resolved
choice. See the [configuration rules](COMPONENTS_AND_GYMNASIUM.md#selecting-an-integrator)
for solver compatibility; holding a command does not relax those rules.

## Timing semantics

The physical integration timestep is preserved:

```
control_dt = physics_steps_per_action * physics_dt
```

For native Warp, `physics_dt` comes from the resolved bundle's configuration.
The existing `substeps` keyword remains available; use one action-hold keyword
at a time. The default is unchanged when neither is supplied.

For JAX, the original configuration defines `physics_dt = control_dt / substeps`.
The new keyword updates both the control interval and the number of substeps to
keep that physical timestep. Changing `config.substeps` alone continues to refine
integration at a fixed configured control interval. Original reset-settling
duration is preserved, including in pure differentiable rollout initialization.

At the engine API, `horizon` remains a number of policy actions. Set it explicitly
when comparing equal physical episode durations. Solver tolerances, materials,
contact and actuator time constants are unaffected.

## Registered codesign-gym presets

The integrated SoftWalker registration also accepts the option:

```python
import codesign_gym as cg

env, design = cg.make_vec(
    'SoftWalker-MPM2D', num_envs=4, device='cuda',
    physics_steps_per_action=10,
)
observations, rejected, info = env.reset(embodiment=design)
env.close()
```

The registered presets keep their physical integration timestep (normally
0.00025 seconds), settling duration and target episode duration (normally
2.4 seconds). The action horizon is rounded up to cover that target duration;
the possible excess is less than one action interval. Thus `k=10` gives a
0.0025-second action interval and 960 actions per default episode. Omitting the
option retains the existing `k=40`, 0.01-second interval and 240 actions.

The default preset already amortizes reporting across 40 physics steps. Setting
it to 10 increases controller feedback frequency; it is not an additional
tenfold repeat of its existing control interval.

The registered `SoftWalker-*` presets always integrate with backward Euler. In the
engine factories, choosing native Kvaerno preserves `physics_dt` and `control_dt`:
its internal implicit stages belong to each physics step and do not count as
additional held actions or physics substeps.

## Measuring overhead reduction

Calling `env.step` ten times from Python incurs ten observation/reporting
boundaries. One call with `physics_steps_per_action=10` incurs one boundary for
the same ten physics steps. The benchmark below holds identical commands in
both paths and checks states and accumulated rewards at each action boundary.
It compares equal simulated durations, excluding construction, capture and
warmup. Throughput is reported in physics steps/second and simulated time per
wall-clock second, not just API calls/second.

```bash
export WARP_CACHE_PATH="$PWD/runs/warp_native_cache"
python -m envs.soft_walker.diagnostics.benchmark_action_hold \
  --device cuda:0 --dtype float64 --physics all --preset larger \
  --batch-sizes 1 4 16 --hold 10 --periods 5 --repeats 5 \
  --trace --out runs/action_hold_larger
```

The benchmark's contact-free model presets isolate API overhead. They do not
establish locomotion quality or a universal speedup for production robots.
