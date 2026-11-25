# Parallel environments

`make_vec` batches independent environments for spring 2D, beam 3D, FEM 3D,
MPM 2D, and MPM 3D. Each lane has its own state, material values, commands,
solver status, reward, and episode clock. The vector interface uses fixed
capacities: choose one physics type, precision, device, model array shapes,
and action/observation capacity for each vector environment. Material and
actuator values can differ between lanes. Build separate vectors for different
topologies or capacities.

Use `physics_steps_per_action=10` to keep a command constant across ten resident
physics steps before returning observations and rewards. This preserves the
integration timestep. See [action holding](ACTION_HOLD.md) for episode timing,
reset settling, and a benchmark that compares equal physical durations.

## Native Warp

```python
import numpy as np
from envs.soft_walker.engine import make_vec

env, designs = make_vec(
    'SoftSpring2D-v0', backend='warp', num_envs=16,
    device='cuda:0', dtype='float64', execution='auto',
    substeps=4, horizon=250,
)
observations, rejected, info = env.reset(seed=42, embodiment=designs)
if rejected.any():
    raise ValueError(f'Rejected lanes: {np.flatnonzero(rejected)}')
actions = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
observations, rewards, terminated, truncated, info = env.step(actions)
env.close()
```

`designs` is a dictionary whose arrays have leading dimension `num_envs`.
Modify those arrays before reset to assign distinct embodiments to lanes.
`observations` and `actions` also have a leading environment dimension.
Rewards, termination flags, and truncation flags have shape `(num_envs,)`.
Per-lane information lives in `info['per_env']`.

Execution modes:

| Mode | CUDA | CPU |
| --- | --- | --- |
| `auto` (default) | Concurrent lane graphs | Serial lane execution |
| `parallel` | Concurrent lane graphs | Raises `ValueError` |
| `serial` | One lane at a time | One lane at a time |

CUDA parallel execution submits each lane's captured complete physics program
on an independent stream before waiting for the batch. Newton/Krylov work
and all internal substeps remain on the GPU. This permits overlap across
independent lanes; it does **not** combine their sparse systems into one
fused kernel. CUDA decides whether kernels actually overlap based on available
resources. A larger batch can improve device utilization, but performance
depends on body size, solver work, and the observation boundary. Measure it
on the intended task rather than assuming a speedup.

Device memory scales approximately with the number of lanes: each lane owns
its model, state, solver workspace, and captured graph. Start with a small
batch when working with large 3D bodies. Lane resets allocate new owners;
the batch executor must bind the new episodes rather than replay old buffers.

The public API still accepts/returns NumPy arrays. This is appropriate for
Gymnasium-style interaction but involves host transfers at every `step`.
For open-loop experiments, `env.envs[i].episode(commands, ...)` provides the
existing resident whole-episode execution and adjoint interface. The parallel
episode executor also supports independent whole-episode jobs. This feature
does not introduce a differentiable tensor vector API or optimize a batch
loss automatically.

## Reset, failure, and padding rules

- A completed or numerically failed lane resets on the **next** vector step.
  That call returns its initial observation, zero reward, false done flags,
  and `info['per_env'][i]['autoreset'] == True`. Its action on that call is
  ignored. Other lanes continue normally.
- Invalid actions are checked for the entire batch before any active lane
  advances. They must have the declared shape, be finite, and lie in `[-1, 1]`.
- Unused actuator channels and padded action entries do not drive physics.
- A rejected explicit reset remains invalid until the user explicitly resets
  it with an accepted design. It cannot silently recover an older design on
  the next step.
- Both engine vectors accept `options={'reset_mask': boolean_lane_vector}`.
  Unselected lanes retain their state and RNG; native lanes also retain their
  graph owners. An uninitialized unselected lane remains invalid.

## JAX

The JAX vector implementation already uses a jitted `jax.vmap` over the
independent physics transitions for all five drivers. Use the same factory
without `backend='warp'` (and omit the Warp-only `execution` keyword):

```python
from envs.soft_walker.engine import make_vec

env, designs = make_vec('SoftSpring2D-v0', num_envs=16, dtype='float32')
observations, rejected, info = env.reset(embodiment=designs)
observations, rewards, terminated, truncated, info = env.step(
    env.action_space.sample()
)
env.close()
```

JAX accepts `device='cpu'` or `device='cuda:0'`; its default is the process's
selected device. Set `JAX_PLATFORMS` and CUDA visibility before importing it
when limiting the available backends. Float64 requires
`JAX_ENABLE_X64=1` before import. JAX batches must also share static shapes and
configuration. The JAX interface supports explicit partial reset through
`options={'reset_mask': boolean_lane_vector}`. Both backends preserve each
lane's own termination and autoreset lifecycle.

The engine reset result includes a rejected-design vector, so it has three
items rather than stock Gymnasium's two. Use `make_gymnasium_vec` for the
standard contract and Gymnasium vector wrappers:

```python
from envs.soft_walker.engine import make_gymnasium_vec
from gymnasium.wrappers.vector import RecordEpisodeStatistics

env = RecordEpisodeStatistics(make_gymnasium_vec(
    'SoftSpring2D-v0', num_envs=16, dtype='float32',
    # Add backend='warp', device='cuda:0' for the native backend.
))
observations, info = env.reset(seed=7)
observations, rewards, terminated, truncated, info = env.step(env.action_space.sample())
env.close()
```

This adapter preserves the backend's batched execution. To provide designs,
pass `options={'embodiment': designs}` to reset. A rejected design raises
`ValueError` and still requires an explicit valid reset before stepping.
Both backends advertise Gymnasium `AutoresetMode.NEXT_STEP`; native `per_env`
information is converted to Gymnasium's dictionary-of-arrays format with
presence masks. The adapter lives in `src/envs/soft_walker/engine/gymnasium_adapter.py`.

The registered `codesign_gym` SoftWalker presets run on native Warp, with a flat
per-cell embodiment, resolved observations and fixed reward scales. They accept
`execution='auto'`, `'serial'`, or `'parallel'` (CUDA required); serial is an explicit
reference mode. They support partial reset and rendering.
Use the engine's `make_gymnasium_vec` above when a consumer requires standard
two-value Gymnasium reset. The registry's co-design reset still returns three values.

## Correctness and performance checks

```bash
WARP_TEST_DEVICE=cpu WARP_TEST_DTYPE=float64 python -m pytest -q \
  tests/soft_walker/test_warp_native_parallel_env.py
WARP_TEST_DEVICE=cuda:0 WARP_TEST_DTYPE=float64 python -m pytest -q \
  tests/soft_walker/test_warp_native_parallel_env.py
python -m envs.soft_walker.diagnostics.benchmark_parallel_envs \
  --device cuda:0 --dtype float64 \
  --physics spring beam fem mpm2 mpm3 \
  --batch-sizes 1 4 16 --steps 5 --repeats 5 \
  --out runs/parallel_envs_cuda
```

The native tests compare vector transitions with independent scalar runs,
using different materials and commands, and cover masks, invalid-action
atomicity, reset/replay, numerical-failure isolation, and next-step autoreset.
The benchmark compares identical serial and concurrent trajectories and
rejects failures or mismatches. Its reported latency includes NumPy
observations and metrics; reset, graph capture, compilation, and warmup are
reported separately and excluded. The default bodies are small and have
contact disabled, so these timings characterize batching overhead rather than
walking speed or difficult terrain convergence. CPU benchmarking compares
`serial` against `auto`, both serial on CPU, as a control.
