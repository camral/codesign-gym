# Design space

The design is a flat float32 vector with three entries per robot: 6 for `TruckUnload-Single` and 9 for
`TruckUnload-Multi`. Each entry is the length change of one offsettable link of that robot and accept positive
and negative offsets. Everything at the far end (the next joint, the gripper) moves with it. Every robot
has its own entries - nothing is shared between robots.

An entry `o` changes its link's length by `o · max_offset` metres, with `max_offset = 1.0` m. The base design, all zeros,
is the stock robots.

## Slots

| Slots | Robot | Link | Body | Length at 0 (m) | Bounds |
|---|---|---|---|---|---|
| 0 | Loader | Column | `link0_offsettable` | 0.383 | `[-0.343, 1]` |
| 1 | Loader | First arm link | `link1_offsettable` | 0.125 | `[-0.085, 1]` |
| 2 | Loader | Second arm link | `link2_offsettable` | 0.125 | `[-0.085, 1]` |
| 3 | Stacker 1 (red) | Column | `stacker0_link0_offsettable` | 0.383 | `[-0.343, 1]` |
| 4 | Stacker 1 | First arm link | `stacker0_link1_offsettable` | 0.1 | `[-0.06, 1]` |
| 5 | Stacker 1 | Second arm link | `stacker0_link2_offsettable` | 0.1 | `[-0.06, 1]` |
| 6 | Stacker 2 (blue, Multi only) | Column | `stacker1_link0_offsettable` | 0.383 | `[-0.343, 1]` |
| 7 | Stacker 2 | First arm link | `stacker1_link1_offsettable` | 0.1 | `[-0.06, 1]` |
| 8 | Stacker 2 | Second arm link | `stacker1_link2_offsettable` | 0.1 | `[-0.06, 1]` |

The column is the vertical post the carriage slides up. Its top end is a stop for the carriage (a tendon limit), so the
column length sets how high the arm can lift. The two arm links set the arm's horizontal reach. At `1` a
stacker arm link grows from 0.1 m to 1.1 m.

## Length-safe lower bounds

Links can shrink as well as grow, but never down to nothing. Each lower bound is raised so that the shortest possible
link is `min_link_length + min_length_pad = 0.02 + 0.02 = 0.04` m:

```text
lower bound = (0.04 − stock length) / max_offset
```

For the stacker arm links that is `(0.04 − 0.1) / 1.0 = −0.06`. The bounds depend on `max_offset`: after
`envs.change_max_offset(x)` the env recomputes them and replaces `embodiment_space`.

## What follows from the shape

- **Mass and inertia** of each link are recomputed from its new box geometry at the stock density.
- **The robots' fields do not follow the design.** Each robot has a fixed box-shaped field, and its arm tip leaving
  the field ends the episode. The stacker fields reach 1.2 m past the belt, so every design in bounds spawns inside its
  field.
- **The arm cannot reach the ground.** The lowest the tip gets is about 0.51 m for every design, which is why the floor
  target is a raised pad.
- **The design is not observed.** The policy has to infer each robot's body from how it moves.

## Base design

```python
import numpy as np
from codesign_gym import make_vec

envs, base = make_vec("TruckUnload-Single", 2, device="cpu")
print(base)                                               # [0. 0. 0. 0. 0. 0.]
print(envs.embodiment_space.low)                          # [-0.343 -0.085 -0.085 -0.343 -0.06  -0.06 ]
print(envs.embodiment_space.high)                         # [1. 1. 1. 1. 1. 1.]
```

## Build a design

Give the stacker a longer arm and a shorter column, and leave the loader as it is:

```python
design = base.copy()
design[[4, 5]] = 0.3        # stacker arm links 0.1 m -> 0.4 m
design[3] = -0.1            # stacker column 0.1 m shorter
print(envs.validate_embodiment([base, design]))           # [ True  True]

obs, rejected, info = envs.reset(embodiment=[base, design])
print(np.round(envs.env_offsets[1], 3))                   # metres: [ 0.   0.   0.  -0.1  0.3  0.3]
```

One design is broadcast to every copy; a `(num_envs, n)` array or a list gives one design per copy, and a `None` entry
means the base design.

## What is rejected

`validate_embodiment` returns one boolean per copy. It is `False` for a design with the wrong shape, a non-finite entry,
or an entry outside its bounds:

```python
too_short = base.copy()
too_short[4] = -0.1         # would make a 0.0 m link; the bound is -0.06
print(envs.validate_embodiment(too_short))                # [False False]
envs.close()
```

`reset` raises `ValueError` if any copy's design is invalid. In-bounds designs that cannot do the job, such as an arm
too short to reach the shelf, are valid; they just earn a low return. The `embodiment_rejected` flags that `reset`
returns are `True` only where a copy's physics went non-finite while it was being set up.

## Build options

`make` and `make_vec` forward keyword arguments to the env, on top of the preset that the id selects:

| Argument | Default | Meaning |
|---|---|---|
| `device` | CUDA if visible, else `"cpu"` | Where the batched simulation runs |
| `render_mode` | `None` | `"rgb_array"` or `"human"` |
| `max_offset` | 1.0 | Metres per unit of a design entry |
| `min_link_length`, `min_length_pad` | 0.02, 0.02 | Together set the shortest allowed link |
| `time_limit_s`, `inactivity_limit_s` | 120 (Single) / 300 (Multi), 15 | Episode clocks in seconds |
| `truncate_at_time_limit` | `False` | Report the time limit as truncation instead of termination |
| `belt_velocity` | 0.2 | Centre of the belt speed each respawn draws, in m/s |
| `belt_velocity_spread` | 0.2 | Relative range of the drawn belt speed (`0` for a fixed speed); must be in `[0, 1)` |
| `box_jitter` | `0.45 · box_clearance` (9 mm) | Largest `x`/`y` nudge of a container box at respawn, in m; `ValueError` unless below `box_clearance / 2` |

Any other preset key in `envs.truck_unload.config` can be overridden the same way.
