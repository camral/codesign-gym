# Design space

The hand has a wrist, three fingers and a thumb, built from boxes; each finger's distal link and tip are one segment. The finger flexion joints bend only towards the palm; the finger base joints also move sideways. Its palm and fingers carry 16 heightfield pads of different grids.
The design is one flat `Box(248,)` float32 vector: the height of every pad cell, then three link offsets.

| Entries | Bounds | Meaning |
|---|---|---|
| `[0:245]` | 0 – 1 | Height of each pad cell, as a share of the pad's elevation range: 2 cm for the palm and fingertip-end pads, 1 cm for the pads along the finger links (0 = flat, 1 = fully raised) |
| `[245]` | -1 – 1 | Finger spacing: stretches the hidden links from the palm to the index and ring finger bases, moving those fingers away from (+) or towards (−) the middle finger |
| `[246]` | -1 – 1 | Width of every index, middle and ring finger link |
| `[247]` | -1 – 1 | Width of every thumb link |

A link offset of ±1 is ±15 mm (`max_link_offset = 0.015`): the spacing links get 15 mm longer or shorter, and each finger box gets 15 mm wider or narrower, centred, with its pads widening with it.
No link gets narrower or shorter than 2 mm (`min_link_length`). Link masses and inertias follow the new geometry.
Narrow spacing with wide fingers makes neighbouring fingers overlap: with the spacing entry `s` and the finger-width entry `w`, the gap between neighbouring finger links is `16 + 15 s − 15 w` mm, so designs with `w − s > 1.07` start with the fingers pressed into each other.

Each cell of the design is one cell of the pad's grid in the hand's MJCF. The physics uses a finer heightfield (`grid_subd = 2`): every cell is a 2 × 2 patch of samples from a quadratic spline through the pad's cells, so the pads are smooth surfaces rather than steps.

The pads along the finger links (`*_proximal_x`, `*_medial_x`, `*_tip_x`, `th_medial_x`, `thumb_x`) taper to flat at both ends, so curled links do not press into each other (`relief_length = 2`). Along a pad, the sample `d` samples from the nearer end is scaled by `min(1, d / 2)`: the end samples are always flat, the next ones are at half height, and the rest follow the design. The design itself is not changed, and `info["embodiment"]` reports it untapered.

## Pad order

Pads follow the order of the hand's `<hfield>` assets. Within a pad with `ncol` columns, cell `(r, c)` is entry `pad_start + ncol r + c`.
The names use `ff` for the index (first) finger, `mf` for the middle, `rf` for the ring finger and `th` for the thumb.

| Entries | Pads (rows × columns, first entry) |
|---|---|
| `[0:80]` | `palm` (10 × 8, 0) |
| `[80:104]` | Fingertip ends: `rf_tip_z` (3 × 2, 80), `mf_tip_z` (3 × 2, 86), `ff_tip_z` (3 × 2, 92), `thumb_z` (3 × 2, 98) |
| `[104:140]` | Ring finger: `rf_proximal_x` (3 × 5, 104), `rf_medial_x` (3 × 4, 119), `rf_tip_x` (3 × 3, 131) |
| `[140:176]` | Middle finger: `mf_proximal_x` (3 × 5, 140), `mf_medial_x` (3 × 4, 155), `mf_tip_x` (3 × 3, 167) |
| `[176:212]` | Index finger: `ff_proximal_x` (3 × 5, 176), `ff_medial_x` (3 × 4, 191), `ff_tip_x` (3 × 3, 203) |
| `[212:245]` | Thumb: `th_medial_x` (3 × 5, 212), `thumb_x` (3 × 6, 227) |

The `_x` pads face the palm side of each link (`*_tip_x` and `thumb_x` cover the merged distal-and-tip segment); the `_z` pads cap the end of each fingertip.

## Base design

The base design is all zeros: every pad flat and every link as built.

```python
# notest: needs the morphhand extra
from codesign_gym import make

env, base = make("RotateInHand-Cube")
print(env.embodiment_space.shape, base.shape)      # (248,) (248,)
print(base.min(), base.max())                      # 0.0 0.0
```

## Build a design

Start from the base design and set the entries you want:

```python
# notest: needs the morphhand extra
import numpy as np

PAD = {"palm": (0, 10, 8), "ff_tip_x": (203, 3, 3)}  # first entry, rows, columns, from the table above

design = base.copy()
start, rows, cols = PAD["palm"]
design[start:start + rows * cols] = 0.5            # raise the whole palm pad by 1 cm
start, rows, cols = PAD["ff_tip_x"]
cells = design[start:start + rows * cols].reshape(rows, cols)
cells[1, :] = 1.0                                  # a 1 cm ridge across the middle row of the index fingertip pad
design[245] = 0.3                                  # spread the index and ring fingers 4.5 mm each
design[247] = -0.5                                 # thumb links 7.5 mm narrower
print(env.validate_embodiment(design))             # True
```

## What is rejected

Only out-of-range designs: `validate_embodiment` returns `False` for a vector of the wrong shape, a non-finite entry, a pad cell outside `[0, 1]` or a link offset outside `[-1, 1]`.
Any design inside the bounds is valid, even one whose raised pads stop the fingers from closing.
`reset` and `step` raise `ValueError` on an invalid design instead of returning a rejection.

## Changing the design mid-episode

`step(action, embodiment)` reshapes the hand from that step on, without a reset.
The new design applies to every copy it names: pass one design for all copies, a `(num_envs, 248)` batch, or a per-copy list where `None` keeps that copy's design.
A copy that ends and autoresets keeps its latest design. Evaluation envs (`is_eval_env=True`) ignore a design passed to `step`, with a warning.
See [Examples](examples.md).
