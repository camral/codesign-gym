# How the design works

All seven ports share one mechanism. Some links of the robot are *offsettable*: each is a capsule that runs from its
joint (the body origin) to a far end, and the design moves that far end. Everything attached at the far end — the next
link, a foot, a hand — moves with it.

## Slots and axes

The design is a flat float32 vector with one entry (a *slot*) per axis per link, each in `[-1, 1]`. A link takes one,
two or three slots, depending on which body-frame axes its far end may move along:

| Link kind | Slots | Effect |
|---|---|---|
| One axis (`x` or `z`) | 1 | Moves the far end along the link, so it only changes length |
| Two axes (`xz` or `xy`) | 2 | Moves the far end in the robot's plane: length and in-plane angle |
| Three axes (`xyz`) | 3 | Moves the far end anywhere in a box: length and 3D direction |

An entry of `+1` moves the far end `max_offset` metres along its axis, in the direction that points *away* from the
joint, and `-1` moves it the same distance back. So on an axis the link already runs along, a positive value always
lengthens the link and a negative one shortens it; on an axis across the link, it turns the link. Each port page lists
every slot, its link, its axis and the direction `+1` moves the far end.

Slots are numbered in the order the links appear in the model, and links on the left and right of the body are
independent: nothing is mirrored for you.

The Humanoid ports add four more slots that scale sphere radii instead of moving an end: the feet and the hands. There an
entry `o` sets the radius to `r0 · (1 + 0.5 · o)`, so `-1` halves the radius and `+1` makes it 1.5 times larger.

## Step size and length floor

`max_offset` is set per port to half the median link length. A link cannot be shortened past its joint or below
`min_link_length` (twice the largest link radius); offsets beyond that point all build the same, shortest link. When
the env is built, it prints one warning line per link whose range reaches that floor, for example
`[hopper] WARNING: foot_heel_offsettable_x: 0.130 m undeformed, 0.015 m at its shortest, floored at 0.120 m`.

| Id | `max_offset` (m) | `min_link_length` (m) |
|---|---|---|
| `HalfCheetah-v4` | 0.148 | 0.092 |
| `Swimmer-v4` | 0.5 | 0.2 |
| `Hopper-v4` | 0.115 | 0.12 |
| `Walker2d-v4` | 0.163 | 0.12 |
| `Ant-v4` | 0.141 | 0.16 |
| `Humanoid-v4` | 0.144 | 0.12 |
| `HumanoidStandup-v4` | 0.144 | 0.12 |

Both can be overridden through `make` / `make_vec`, for example `make("Hopper-v4", max_offset=0.2)`. The design space
stays `[-1, 1]`; only the distance a full step covers changes.

## What follows from the shape

- **Mass and inertia** are recomputed for every changed link from its new capsule, at the density the stock model gives
  that link. A longer limb reaches further but weighs more. Where the port splits one stock capsule into two links (the
  Hopper and Walker2d torso, the Hopper foot), the pair is massed as one capsule, so the base design keeps the stock
  mass. A Humanoid forearm is massed together with its hand.
- **Spawn height** adapts to the design. After the shape is built, the root is raised or lowered so that the lowest
  point that can touch the floor sits where it does for the base design. Long legs therefore start standing on the
  floor rather than buried in it, and the base design spawns exactly as stock.
- **Termination thresholds do not.** The healthy ranges (Hopper, Walker2d, Ant, Humanoid) are the stock ones whatever
  the design. Shorter legs start the torso lower and closer to the height limit, which makes them harder to keep
  healthy. That is a valid design with a lower return, not an error.

Designs where limbs overlap each other at the extremes of their range are valid too; they simulate, just badly.

## Base design

The base design is all zeros: every link where the stock model has it.

```python
import numpy as np
from codesign_gym import make

env, base = make("Hopper-v4", device="cpu")
print(base)                     # [0. 0. 0. 0. 0. 0. 0. 0.]
print(env.embodiment_space)     # Box(-1.0, 1.0, (8,), float32)
env.close()
```

## Build a design

Start from the base design and set the slots you want. In the Hopper, slot 3 moves the knee straight down (a longer
thigh) and slot 5 moves the ankle down (a longer leg):

```python
env, base = make("Hopper-v4", device="cpu")
design = base.copy()
design[3] = 1.0                 # thigh 0.115 m longer
design[5] = -0.5                # leg 0.0575 m shorter
print(env.validate_embodiment(design))      # True
obs, rejected, info = env.reset(seed=0, embodiment=design)
print(rejected, info["embodiment"].tolist())    # False [0.0, 0.0, 0.0, 1.0, 0.0, -0.5, 0.0, 0.0]
env.close()
```

`info["embodiment"]` is the design in use. A design outside `[-1, 1]` is not clipped:

```python
env, base = make("Hopper-v4", device="cpu")
too_long = np.full(8, 1.5, dtype=np.float32)
print(env.validate_embodiment(too_long))    # False
try:
    env.reset(seed=0, embodiment=too_long)
except ValueError as error:
    print("ValueError")                     # ValueError
env.close()
```
