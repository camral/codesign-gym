# Concepts

## Design and control

A co-design problem has two parts:

- the **design** (the API calls it the *embodiment*): what the agent is. A body shape, a warehouse layout, a Pokémon team, a microgrid's battery size.
- the **control policy**: how the agent acts, one step at a time, as in any Gymnasium environment.

The reward depends on both. A good design for one policy can be a bad design for another, so you cannot optimize them separately.
Codesign Gym gives the control half the normal Gymnasium interface and adds the design half on top. You can use any optimizer for either half.

## The design space

Every environment has an `embodiment_space`, a Gymnasium space like `action_space`. Its type depends on the environment:

| Environment | `embodiment_space` |
|---|---|
| LQ Structure | `Dict` of masses, stiffnesses, damping (`Box`) and actuator/sensor placement (`MultiBinary`) |
| MuJoCo bodies (Ball Catcher, Gym ports, ...) | `Box` of link offsets in [-1, 1] |
| Soft Walker | `Box` with a fixed layout of per-cell values |
| NeroGrasp | custom mesh space (fingertip triangle meshes) |
| Pokémon | custom team space (species, moves, natures, ...) |

Each environment's **Design space** page lists the layout, the bounds and how to build a design by hand.

```python
from codesign_gym import make

env, base = make("LQStructure-LQG")
print(env.embodiment_space)
sample = env.embodiment_space.sample()
print(env.embodiment_space.contains(sample))
env.close()
```

`make` and `make_vec` also return the **base design**: a known, reasonable design (for MuJoCo bodies, the stock model).

## Rejection

`validate_embodiment` rejects only **impossible** designs: values out of bounds, the wrong shape, or a design that breaks a hard rule of the environment (for example, an LQ chain with no actuator).

```python
import numpy as np
from codesign_gym import make

env, base = make("LQStructure-LQG")
no_actuators = {**base, "actuators": np.zeros(6, dtype=np.int8)}
print(env.validate_embodiment(no_actuators))          # False

obs, rejected, info = env.reset(seed=0, embodiment=no_actuators)
print(rejected)                                       # True
obs, rejected, info = env.reset(seed=0, embodiment=base)
print(rejected)                                       # False: reset again before you step
env.close()
```

After a rejected reset, call `reset` again with a valid design before you call `step`. Every environment raises until then.
How a design *outside the space* is reported (flag or exception) differs between environments; each environment page says which.

## Vector environments

`make_vec(env_id, num_envs)` runs `num_envs` copies. Each environment vectorizes in its own native way (JAX `vmap`, batched MuJoCo Warp worlds, worker processes), so there are no Gymnasium Sync/Async wrappers.

A vector environment takes designs in three forms:

```python
import numpy as np
from codesign_gym import make_vec

envs, base = make_vec("LQStructure-LQR", 3)

envs.reset(seed=0, embodiment=base)                                      # one design for every copy
heavy = {**base, "masses": np.full(6, 1.9)}
envs.reset(seed=0, embodiment=[base, heavy, None])                       # one design per copy; None = base design
obs, rejected, info = envs.reset(seed=0)                                 # no design: base design
print(rejected.shape)                                                    # (3,): one flag per copy
envs.close()
```

Environments whose design is a single array also accept a `(num_envs, n)` batch. LQ Structure, Microgrid and Network also accept a stacked dict, each value with a leading `num_envs` axis.
In `step`, a `None` entry keeps that copy's current design.

`validate_embodiment` on a vector environment checks each copy's design and returns one bool per copy.

## Design changes during an episode

Some environments let the design change between steps: `step(action, embodiment=...)`.

| Supported | Not supported (ignored with a warning) |
|---|---|
| LQ Structure, Microgrid, Racing, Shape-shifting Hand | All others |

An evaluation environment (`make(..., is_eval_env=True)`) always ignores a design passed to `step`, so a fixed design cannot change during evaluation.
See the [mid-episode tutorial](tutorials/mid-episode-designs.md).

## Seeds

`reset(seed=...)` takes an int, or one int per copy. The same seed gives the same episode.
An int seed gives every parallel environment copy its own independent stream, derived from that seed, and runs seeded `s` and `s + 1` share no stream. A list gives copy `i` the seed in position `i`, so `seed=[s] * num_envs` starts every copy of LQ Structure, Microgrid, Network, Soft Walker or Pokémon the same way.

## Partial reset and autoreset

`reset(options={"reset_mask": mask})` resets only the copies where `mask` is `True`. The others continue their episodes.

When a copy's episode ends, it resets on the **next** `step` and keeps its last design. The step that ended the episode returns its final observation.

All environments except the two below follow Gymnasium's `NEXT_STEP`: the next step only resets the copy, ignores its action, and returns the first observation of the new episode, with reward 0 and `terminated` and `truncated` both `False`.

Two environments differ:

- **Pokémon** resets in the **same** step (`SAME_STEP`); the finished battle's last observation is in `info["final_obs"]`. It does not support partial resets.
- **NeroGrasp** runs all copies on one clock: they finish together, there is no autoreset, and partial resets raise.

## Fixed-size spaces

Observation and action spaces have a fixed size for each environment id, sized for the largest design the space allows. A smaller design (fewer base stations, actuators or cells) is padded, and a mask in the observation or `info` marks what is live.

## Options for `make` and `make_vec`

| Option | Use |
|---|---|
| `is_eval_env=True` | Evaluation env: `step` ignores design changes. |
| `device="cpu"`, `"cuda"`, `"cuda:N"` | MuJoCo environments. The default is CUDA if available, else CPU. |
| `render_mode="rgb_array"` | `render()` returns an image (`make`) or one image per copy (`make_vec`). MuJoCo, Racing, Warehouse and Soft Walker also accept `render(env_idx=i)`. |
| `render_mode="human"` | Opens a window (needs a display). |
| `render_mode="html"` | Pokémon only: `render()` returns a battle replay as HTML. |
| `render_mode="web"` | Racing only: `save_dashboard()` saves a recorded run. |
| `verbose=True` | Shows backend logs during construction. |

Any other keyword is passed to the environment's builder; each environment page lists the useful ones.
Which environment supports which render mode is listed in [Rendering](tutorials/rendering.md#which-environment-supports-what).
