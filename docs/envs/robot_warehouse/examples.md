# Examples

## Draw layouts and render them

Build two custom layouts from text, run them next to the base layout with `make_vec`, and save one image with all three. `render(env_idx=i)` draws copy `i` only:

```python
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
from codesign_gym import make_vec

H, W = 10, 10                                                    # Warehouse-Small
GOALS = [(H - 1, W // 2 - 1), (H - 1, W // 2)]
CELLS = [(r, c) for r in range(H) for c in range(W) if (r, c) not in GOALS]   # bit i <-> CELLS[i]


def from_picture(picture):
    rows = picture.split()
    return np.array([rows[r][c] == "#" for r, c in CELLS], dtype=np.int8)


ring = from_picture("""
..........
..........
..........
.##....##.
.##....##.
.##.##.##.
.##.##.##.
.##....##.
.##....##.
....GG....
""")
packed = from_picture("""
..........
..........
..........
..........
..........
..........
##########
##########
##########
....GG....
""")

envs, base = make_vec("Warehouse-Small", 3, render_mode="rgb_array")
designs = np.stack([base, ring, packed])
print(designs.sum(axis=1), envs.validate_embodiment(designs))   # [25 28 30] [ True  True  True]
obs, rejected, info = envs.reset(seed=0, embodiment=designs)
frames = [envs.render(env_idx=i) for i in range(3)]
print(frames[0].shape)                                          # (396, 302, 3)
path = Path(tempfile.mkdtemp()) / "layouts.png"
Image.fromarray(np.hstack(frames)).save(path)
envs.close()
```

In `packed`, an unloaded robot can still reach every shelf by driving under the rows, but a loaded robot can only leave the shelf rows from the bottom one, next to the goals. Both layouts are valid; which one scores better is for the optimizer to find out.

## Read the observation

Each robot's 66 values start with its own state and end with the shelves in its 3×3 window:

```python
import numpy as np
from codesign_gym import make

DIRECTIONS = ["up", "right", "down", "left"]
env, base = make("Warehouse-Small")
obs, rejected, info = env.reset(seed=0)
for k, view in enumerate(obs.reshape(2, 66)):
    row, col, carrying = view[:3].astype(int)
    facing = DIRECTIONS[int(np.argmax(view[3:7]))]
    shelves = view[48:66].reshape(3, 3, 2)[..., 0].astype(int)  # shelf present in each window cell
    print(f"robot {k}: cell ({row}, {col}), facing {facing}, carrying {bool(carrying)}")
    print(shelves)
env.close()
```

```text
robot 0: cell (2, 0), facing down, carrying False
[[0 0 1]
 [0 0 1]
 [0 0 0]]
robot 1: cell (0, 7), facing down, carrying False
[[0 0 0]
 [0 0 0]
 [0 1 1]]
```

Robot 0 is on the left edge, so the window's left column lies outside the grid and reads as empty; the right column is the base layout's shelves at `(1, 1)` and `(2, 1)`.

## Change one copy's layout

The layout can only change at a reset. A partial reset changes it for some copies and leaves the others running:

```python
envs, base = make_vec("Warehouse-Small", 2)
obs, rejected, info = envs.reset(seed=0)
for _ in range(10):
    obs, reward, terminated, truncated, info = envs.step(envs.action_space.sample())
obs, rejected, info = envs.reset(options={"reset_mask": np.array([False, True])}, embodiment=[None, ring])
print(info["step_count"], rejected)                             # [10  0] [False False]
envs.close()
```

A reset copy with no design (`embodiment=None`, or a `None` entry) gets the base layout, not the one it had before.
