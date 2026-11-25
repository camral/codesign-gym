# Design space

The hand's pads and links are the ones from [Rotate](../rotate_in_hand/embodiment.md) (the hammers' hand adds only the wrist slides), so the design is the same `Box(248,)` float32 vector:

| Entries | Bounds | Meaning |
|---|---|---|
| `[0:245]` | 0 – 1 | Height of each cell of the 16 pads, as a share of the pad's elevation range (2 cm for the palm and fingertip-end pads, 1 cm for the pads along the finger links) |
| `[245]` | -1 – 1 | Finger spacing: moves the index and ring fingers away from (+) or towards (−) the middle finger, ±15 mm |
| `[246]` | -1 – 1 | Width of every index, middle and ring finger link, ±15 mm |
| `[247]` | -1 – 1 | Width of every thumb link, ±15 mm |

The pad order is in the [Rotate pad table](../rotate_in_hand/embodiment.md#pad-order). The base design is all zeros. As in Rotate, the pads along the finger links taper to flat at both ends (`relief_length = 2`).

The hand hangs palm-down here, so the palm pad (`[0:80]`) and the fingertip pads (`rf_tip_x`, `mf_tip_x`, `ff_tip_x`, `thumb_x`: `[131:140]`, `[167:176]`, `[203:212]`, `[227:245]`) face the table.
For the Cube, the table is placed for the hand with every pad fully raised; the hammers use a fixed 145 mm gap. In every preset the env refuses to build if a hand with every pad fully raised would start inside the table or the object.

## Build a design

```python
# notest: needs the morphhand extra
import numpy as np
from codesign_gym import make

env, base = make("HandPickUp-Cube")
design = base.copy()
design[np.r_[131:140, 167:176, 203:212, 227:245]] = 1.0   # raise every fingertip pad by 1 cm
design[246] = 0.4                      # fingers 6 mm wider
print(env.validate_embodiment(design))  # True
```

## What is rejected

Only out-of-range designs: the wrong shape, a non-finite entry, a pad cell outside `[0, 1]` or a link offset outside `[-1, 1]`.
`validate_embodiment` returns `False` for these, and `reset` and `step` raise `ValueError`.

## Changing the design mid-episode

As in Rotate, `step(action, embodiment)` reshapes the hand from that step on, a `None` entry in a per-copy list keeps that copy's design, a copy that autoresets keeps its latest design, and evaluation envs ignore the change with a warning.
