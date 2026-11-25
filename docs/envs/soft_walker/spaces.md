# Observations, actions and rewards

## Observation

The observation is the simulator's full resolved state, as one flat float64 `Box` (unbounded). Its size is fixed per model and does not change with the design: empty cells keep their nodes or particles, which simply carry no mass.
The state arrays are flattened one after another (each one in its own C order), followed by three short blocks:

| Block | MPM models | Solid models (`Spring2D`, `Beam3D`, `HexFEM3D`) |
|---|---|---|
| 1 | particle position minus its rest position, `n·d` | node displacement `u`, `n·d` |
| 2 | particle velocity, `n·d` | node velocity, `n·d` |
| 3 | deformation gradient `F`, `n·d·d` | node rotation matrix, `n·d·d` |
| 4 | affine velocity field `C` (APIC), `n·d·d` | node angular velocity, `n·d` |
| 5 | actuator state, one per channel | actuator state, one per channel |
| 6 | energy ledger: signed work, positive work, dissipated energy, time | same |
| 7 | action mask, one per channel (1 = channel drives something) | same |

Here `n` is the number of particles or nodes and `d` the dimension:

| Model | `n` | Per node or particle | Channels | Observation size |
|---|---|---|---|---|
| `Spring2D` | 63 nodes (9 x 7 cell corners) | 10 | 4 | 642 |
| `MPM2D` | 192 particles (2 per cell per axis) | 12 | 4 | 2316 |
| `MPM3D` | 2304 particles | 24 | 16 | 55332 |
| `Beam3D` | 288 nodes (one per cell) | 18 | 16 | 5220 |
| `HexFEM3D` | 441 nodes (9 x 7 x 7 cell corners) | 18 | 16 | 7974 |

`info["observation_mask"]` marks the valid entries. Every entry is valid in these presets, so the mask is all `True`.
The function below splits an observation into named blocks:

```python
import numpy as np
from codesign_gym import make


def split_observation(obs, env):
    """Flat observation -> dict of named arrays (n = particles or nodes, d = dimension, c = channels)."""
    d, c = len(env.embodiment_space.grid), env.action_space.shape[-1]
    mpm = env.vector_env.driver.startswith("mpm")
    widths = [d, d, d * d, d * d] if mpm else [d, d, d * d, d]
    names = ["displacement", "velocity", "deformation_gradient", "affine_velocity"] if mpm else ["displacement", "velocity", "rotation", "angular_velocity"]
    n = (obs.shape[-1] - 2 * c - 4) // sum(widths)
    out, start = {}, 0
    for name, w in zip(names, widths):
        block = obs[start:start + n * w]
        out[name] = block.reshape(n, d) if w == d else block.reshape(n, d, d)
        start += n * w
    out["actuator"], out["ledger"], out["action_mask"] = obs[start:start + c], obs[start + c:start + c + 4], obs[start + c + 4:]
    return out


env, base = make("SoftWalker-HexFEM3D")
parts = split_observation(np.zeros(env.observation_space.shape), env)
for name, value in parts.items():
    print(name, value.shape)
env.close()
```

```text
displacement (441, 3)
velocity (441, 3)
rotation (441, 3, 3)
angular_velocity (441, 3)
actuator (16,)
ledger (4,)
action_mask (16,)
```

## Action

A `Box(-1, 1)` float64 vector with one command per actuation channel: 4 in 2D, 16 in 3D. Each cell follows the channel in its `channel` field, through a first-order actuator (time constant 0.02 s).
Non-finite actions or actions outside `[-1, 1]` raise `ValueError`; they are not clipped. Commands for channels the design does not use are zeroed. The action mask lists the used channels; it is in `info["action_mask"]` after reset and step, and in the last block of the observation.
In `Beam3D` the mask counts only channels routed on a link between two solid cells, since links to empty cells have no stiffness.

Each action is held for 40 physics steps of 0.25 ms (a 0.01 s control step).
`make(..., physics_steps_per_action=k)` changes the hold to `k` physics steps at the same physics step size; the number of control steps is then rounded up so the episode still lasts at least 2.4 s.

## Reward

Each step pays forward travel per body length minus positive actuator work per work scale:

```text
reward = Δx / L  -  ΔW⁺ / W_ref
```

- `Δx` is the step's mass-weighted mean travel of the body along `x`,
- `L = 0.16 m` is the body length (8 cells),
- `ΔW⁺` is the positive work the actuators did during the step,
- `W_ref` is 0.4 J in 2D and 4 J in 3D.

The return is the total travel in body lengths minus the total positive work over `W_ref`. Its range is not yet characterised.
`info["metrics"]` holds the running totals (`displacement` in metres, `positive_work` in joules), and `info["objectives"]` holds `[-travel / L, work / W_ref]`, so the return so far is `-sum(objectives)`.

## Episode

Each explicit reset rebuilds the body from the design and lets it settle under gravity for 0.2 s with zero command. Then the episode runs for 240 control steps (2.4 s).

- **Truncation:** after step 240.
- **Termination:** a physics step fails (the solver diverges or, in MPM, the body leaves the background grid). That step pays 0 and sets `info["valid"]` to `False`.

The copy autoresets on its next `step` with its last accepted design, and that call returns reward 0 and `info["autoreset"] = True`.
The autoreset does not settle the body again: it restores the settled state from the last explicit reset, so the first observation is the same as that reset's.
`reset(options={"reset_mask": m})` resets only the copies where `m` is `True`; a copy waiting for its autoreset keeps it.

The physics is deterministic: `seed` only seeds the action space.

## Info

Keys after `reset`:

| Key | Content |
|---|---|
| `valid` | `True` if the copy reset |
| `embodiment_errors` | Why the design was rejected (only when one was) |
| `action_mask` | Channels the design uses |
| `observation_mask` | Valid observation entries |
| `observation_profile` | A label for the observation format |
| `metrics` | `displacement` (m) and `positive_work` (J) so far |
| `objectives` | `[-travel / L, work / W_ref]` so far |

`step` adds:

| Key | Content |
|---|---|
| `autoreset` | This call reset the copy instead of stepping it |
| `out_of_domain` | MPM: the failed step had the body within one cell of the background grid's edge (walked off the grid rather than diverged) |
| `numerics` | Solver diagnostics (residuals, iterations, ...) |
| `execution_seconds`, `batch_execution_seconds`, `batch_size`, `execution_mode` | Timing of the step |

With `make_vec`, every key is stacked across copies, as in the other environments.
