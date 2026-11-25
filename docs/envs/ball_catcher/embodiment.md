# Design space

The design is a flat float32 vector of 45 entries. 15 links are offsettable: each is a capsule or
cylinder that runs from its body origin to a far end, and the design moves that far end (and everything attached to it).
Each link takes three entries, one per body-frame axis `x`, `y`, `z`, so its far end can move anywhere in a box
(the link gets longer, turns, or both).

An entry of `1` moves the far end `max_offset` metres (1.0 m by default) along its axis, in the direction that points
away from the joint. Every link is a `_pos` link, so each entry is bounded to `[0, 1]`: a link can never shorten.
The base design is all zeros.

## Slots

Each link's three slots are its body-frame `x`, `y` and `z` axes, in that order. The sign says which way `1` moves the
far end in the link's own frame. The axis the link already runs along lengthens it; the other two turn it.

| Slots | Body | Part | Length at 0 (m) | `+1` moves the far end along | Runs along |
|---|---|---|---|---|---|
| 0–2 | `front_left_leg_offsettable_xyz_pos` | Front left thigh | 0.574 | `+x`, `+y`, `−z` | `y` and `−z` |
| 3–5 | `fl_knee_offsettable_xyz_pos` | Front left shin | 0.71 | `+x`, `+y`, `−z` | `−z` |
| 6–8 | `front_right_leg_offsettable_xyz_pos` | Front right thigh | 0.574 | `+x`, `−y`, `−z` | `−y` and `−z` |
| 9–11 | `fr_knee_offsettable_xyz_pos` | Front right shin | 0.71 | `+x`, `+y`, `−z` | `−z` |
| 12–14 | `back_leg_offsettable_xyz_pos` | Back thigh | 0.574 | `−x`, `+y`, `−z` | `−x` and `−z` |
| 15–17 | `back_knee_offsettable_xyz_pos` | Back shin | 0.71 | `+x`, `+y`, `−z` | `−z` |
| 18–20 | `torso_arm_offsettable_xyz_pos` | Arm mount (cylinder on the base) | 0.15 | `+x`, `+y`, `+z` | `+z` |
| 21–23 | `link2_offsettable_xyz_pos` | Left arm, upper link | 0.5 | `+x`, `+y`, `−z` | `−z` |
| 24–26 | `link3_offsettable_xyz_pos` | Left arm, lower link | 0.5 | `+x`, `+y`, `−z` | `−z` |
| 27–29 | `link4_offsettable_xyz_pos` | Left arm, short link (cylinder) | 0.1 | `+x`, `+y`, `−z` | `−z` |
| 30–32 | `link6_offsettable_xyz_pos` | Left arm, end link (cylinder); carries the glove | 0.1 | `+x`, `+y`, `−z` | `−z` |
| 33–35 | `link2r_offsettable_xyz_pos` | Right arm, upper link | 0.5 | `+x`, `+y`, `−z` | `−z` |
| 36–38 | `link3r_offsettable_xyz_pos` | Right arm, lower link | 0.5 | `+x`, `+y`, `−z` | `−z` |
| 39–41 | `link4r_offsettable_xyz_pos` | Right arm, short link (cylinder) | 0.1 | `+x`, `+y`, `−z` | `−z` |
| 42–44 | `link6r_offsettable_xyz_pos` | Right arm, end link (cylinder); ends in a sphere | 0.1 | `+x`, `+y`, `−z` | `−z` |

So the third slot of every arm link and shin lengthens it, and slots 0–17 together set the stance. The two arms are
independent: nothing is mirrored for you.

## What follows from the shape

- **Mass and inertia** of every changed link are recomputed from its new geometry, at the density of the stock link. A
  longer arm reaches further but is heavier: the left arm's upper link weighs 0.049 kg at the base design and 0.120 kg with
  its third slot at `1`. The sphere at the end of the right hand is massed with that link.
- **Spawn height** adapts to the design. The base is raised or lowered so that its lowest floor-touching point sits
  where the base design's does, so long legs start standing rather than buried.
- **The design is not observed.** The policy only sees joint states, the ball and the glove, so it has to infer the
  body it is in from how it moves.

## Base design

```python
import numpy as np
from codesign_gym import make_vec

envs, base = make_vec("BallCatcher-Pitch", 2, device="cpu")
print(base.shape, base.dtype, base.min(), base.max())     # (45,) float32 0.0 0.0
print(envs.embodiment_space)                              # Box(0.0, 1.0, (45,), float32)
```

## Build a design

Start from the base design and set the slots you want. Here the left arm's upper link grows by 0.4 m and its lower link
swings 0.2 m sideways along its `y` axis:

```python
design = base.copy()
design[23] = 0.4        # link2, along its own length (-z)
design[25] = 0.2        # link3, sideways (+y)
print(envs.validate_embodiment([base, design]))           # [ True  True]

obs, rejected, info = envs.reset(seed=0, embodiment=[base, design])
print(envs.env_offsets[1, 21:27])                         # metres: [0.  0.  0.4 0.  0.2 0. ]
```

`env_offsets` reports each copy's applied design in metres, and `info["embodiment"]` returns it in design units. One
design is broadcast to every copy; a `(num_envs, 45)` array or a list gives one design per copy, and a `None` entry
means the base design.

## What is rejected

`validate_embodiment` returns one boolean per copy. It is `False` for a design with the wrong shape, a non-finite entry,
or an entry outside `[0, 1]`:

```python
too_long = base.copy()
too_long[23] = 1.5
print(envs.validate_embodiment(too_long))                 # [False False]
print(envs.validate_embodiment(np.zeros(44)))             # [False False]
```

`reset` raises `ValueError` on such a design, and nothing is changed. Odd but in-bounds designs, such as a huge arm
that cannot lift itself, are valid; they just earn a low return. The `embodiment_rejected` flags that `reset` returns
are `True` only where a copy's physics went non-finite while it was being set up.

## Build options

`make` and `make_vec` forward these keyword arguments to the env:

| Argument | Default | Meaning |
|---|---|---|
| `device` | CUDA if visible, else `"cpu"` | Where the batched simulation runs |
| `render_mode` | `None` | `"rgb_array"` or `"human"` |
| `is_eval_env` | `False` | Marks an evaluation env |
| `max_offset` | `1.0` | Metres that a design entry of `1` moves a far end |
| `max_episode_steps` | `4000` | Episode limit in physics steps (5 per control step, so 800 control steps) |
| `terminate_out_of_field` | `True` | End the episode when the ball or the glove leaves the 20 m field |
| `seed` | `1234` | Seed of the batch's random stream when `reset` gets no seed |

`envs.change_max_offset(new_offset)` changes `max_offset` on a live env; it applies from each copy's next reset. The
design space stays `[0, 1]`; only the distance a full entry covers changes.

```python
envs.close()
```
