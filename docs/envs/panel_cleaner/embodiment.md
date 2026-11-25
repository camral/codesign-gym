# Design space

The design is a flat float32 vector of 48 entries. 16 links are offsettable: each is a capsule or
cylinder that runs from its body origin to a far end, and the design moves that far end (and everything attached to it).
Each link takes three entries, one per body-frame axis `x`, `y`, `z`, so its far end can move anywhere in a box
(the link gets longer, turns, or both).

An entry of `1` moves the far end `max_offset` metres along its axis, in the direction that points away from the joint.
`max_offset` is 0.15 m in `SolarCleaner-Reach` and 0.1 m in the other two. Every link is a `_pos` link, so each entry
is bounded to `[0, 1]`: a link can never shorten. The base design is all zeros.

## Slots

Each link's three slots are its body-frame `x`, `y` and `z` axes, in that order. The axis the link runs along lengthens
it; the other two turn it. Every leg is designed separately, so the robot need not stay symmetric.

| Slots | Body | Part | Length at 0 (m) | `+1` moves the far end along | Runs along |
|---|---|---|---|---|---|
| 0–2 | `front_left_leg_offsettable_xyz_pos` | Front left thigh | 0.34 | `+x`, `+y`, `−z` | `−z` |
| 3–5 | `fl_knee_offsettable_xyz_pos` | Front left shin | 0.38 | `+x`, `+y`, `−z` | `−z` |
| 6–8 | `front_right_leg_offsettable_xyz_pos` | Front right thigh | 0.34 | `+x`, `+y`, `−z` | `−z` |
| 9–11 | `fr_knee_offsettable_xyz_pos` | Front right shin | 0.38 | `+x`, `+y`, `−z` | `−z` |
| 12–14 | `back_left_leg_offsettable_xyz_pos` | Back left thigh | 0.34 | `+x`, `+y`, `−z` | `−z` |
| 15–17 | `bl_knee_offsettable_xyz_pos` | Back left shin | 0.38 | `+x`, `+y`, `−z` | `−z` |
| 18–20 | `back_right_leg_offsettable_xyz_pos` | Back right thigh | 0.34 | `+x`, `+y`, `−z` | `−z` |
| 21–23 | `br_knee_offsettable_xyz_pos` | Back right shin | 0.38 | `+x`, `+y`, `−z` | `−z` |
| 24–26 | `torso_arm_offsettable_xyz_pos` | Arm post on the back | 0.4 | `+x`, `+y`, `+z` | `+z` |
| 27–29 | `link0_offsettable_xyz_pos` | Upper post, after the yaw joint | 0.4 | `+x`, `+y`, `+z` | `+z` |
| 30–32 | `torso_arm2_offsettable_xyz_pos` | Forward boom | 0.2 | `+x`, `+y`, `+z` | `+x` |
| 33–35 | `link1_offsettable_xyz_pos` | Side link | 0.2 | `+x`, `+y`, `+z` | `+y` |
| 36–38 | `link2_offsettable_xyz_pos` | Upper arm | 0.3 | `+x`, `+y`, `−z` | `−z` |
| 39–41 | `link3_offsettable_xyz_pos` | Forearm | 0.3 | `+x`, `+y`, `−z` | `−z` |
| 42–44 | `link4_offsettable_xyz_pos` | Wrist link (cylinder) | 0.1 | `+x`, `+y`, `−z` | `−z` |
| 45–47 | `link5_offsettable_xyz_pos` | End link (cylinder); carries the roller | 0.1 | `+x`, `+y`, `−z` | `−z` |

Slots 0–23 shape the legs and slots 24–47 the arm. The slot that lengthens a link is the third one for every link
except the forward boom (slot 30) and the side link (slot 34).

## What follows from the shape

- **Mass and inertia** of every changed link are recomputed from its new geometry at the stock link's density, so
  longer legs are heavier. (The arm links are very light, a few grams each, so arm changes barely move the mass.)
- **Spawn height** adapts to the design: the base is raised or lowered so that its lowest floor-touching point sits
  where the base design's does. On the `Stability` terrain each copy is then seated on the hills under it.
- **The design is not observed.** The policy has to infer the body it is in from how it moves.

## Base design

```python
import numpy as np
from codesign_gym import make_vec

envs, base = make_vec("SolarCleaner-Reach", 2, device="cpu")
print(base.shape, base.dtype, base.max())                 # (48,) float32 0.0
print(envs.embodiment_space)                              # Box(0.0, 1.0, (48,), float32)
print(envs.max_offset)                                    # 0.15
```

## Build a design

Lengthen the arm's upper link and forearm by the full 0.15 m, and swing the front left thigh outward:

```python
design = base.copy()
design[[38, 41]] = 1.0      # upper arm and forearm, along their length (-z)
design[1] = 0.5             # front left thigh, sideways (+y, away from the body)
print(envs.validate_embodiment([base, design]))           # [ True  True]

obs, rejected, info = envs.reset(seed=0, embodiment=[base, design])
print(np.round(envs.env_offsets[1, 36:42], 3))            # metres: [0.   0.   0.15 0.   0.   0.15]
```

Note that the bounds are per axis of the link's own frame, not per side of the robot: a right leg cannot swing further
right (its `−y`) because that would need a negative entry. `env_offsets` reports each copy's applied design in metres.
One design is broadcast to every copy; a `(num_envs, 48)` array or a list gives one design per copy, and a `None` entry
means the base design.

## What is rejected

`validate_embodiment` returns one boolean per copy. It is `False` for a design with the wrong shape, a non-finite entry,
or an entry outside `[0, 1]`:

```python
print(envs.validate_embodiment(np.full(48, -0.1)))        # [False False]
print(envs.validate_embodiment([base, np.full(48, 2.0)])) # [ True False]
```

`reset` raises `ValueError` if any copy's design is invalid. Odd but in-bounds designs, such as an arm too short to
reach the high cells, are valid; they just earn a low return. The `embodiment_rejected` flags that `reset` returns are
`True` only where a copy's physics went non-finite while it was being set up.

## Build options

`make` and `make_vec` forward keyword arguments to the env, on top of the preset that the id selects:

| Argument | Default | Meaning |
|---|---|---|
| `device` | CUDA if visible, else `"cpu"` | Where the batched simulation runs |
| `render_mode` | `None` | `"rgb_array"` or `"human"` |
| `is_eval_env` | `False` | Also switches the render camera to follow the robot |
| `camera_mode` | `"fixed"` (`"track"` for an eval env) | Overrides the camera choice |
| `max_offset` | 0.15 / 0.1 / 0.1 | Metres that a design entry of `1` moves a far end |
| `min_link_length` | 0.02 | Shortest a link may become (never reached, since links cannot shorten) |
| `min_base_height_frac` | 0.75 | End the episode when the base's height above the ground drops below this fraction of the design's standing height at spawn; `0` turns it off |
| `truncate_at_time_limit` | `False` | Report a goal's budget running out as truncation instead of termination |
| `seed` | 1234 | Seed of the batch's random stream when `reset` gets no seed |

Any other preset key (panel layout, budgets, terrain, disturbances) can be overridden the same way, for example
`make_vec("SolarCleaner-Traverse", 8, rand_pos_range=0.0)`. `envs.change_max_offset(new_offset)` changes `max_offset`
on a live env; it applies from each copy's next reset.

```python
envs.close()
```
