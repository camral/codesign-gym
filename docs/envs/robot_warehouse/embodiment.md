# Design space

The design is the shelf layout: a `MultiBinary(n)` vector with one bit per floor cell. Bit `i` set to 1 puts a shelf on cell `i`.

## Cell indexing

The floor is an `H × W` grid with row 0 at the top. The two goal cells sit in the middle of the bottom row, at `(H−1, W//2−1)` and `(H−1, W//2)`. They are reserved for deliveries, so they have no bit.
Every other cell has one, in row-major order: bit `i` is the `i`-th cell when you read the grid row by row, left to right, skipping the goals. For any cell above the bottom row, that is simply bit `r · W + c`.

| Id | Grid `H × W` | Bits `n` | Robots | Minimum shelves | Maximum shelves | Requested at once |
|---|---|---|---|---|---|---|
| `Warehouse-Small` | 10 × 10 | 98 | 2 | 25 | 98 | 12 |
| `Warehouse-Medium` | 16 × 16 | 254 | 4 | 115 | 252 | 24 |
| `Warehouse-Congested` | 20 × 22 | 438 | 8 | 264 | 432 | 32 |

The minimum shelf count is the design pressure: a layout must hold at least that many shelves. The maximum is `H · W − robots`, so the robots always have room to stand. Any count in between is allowed.

There are no fixed aisles. Robots that are not carrying anything drive under shelves, but a robot carrying a shelf cannot enter a cell that holds another shelf. So the layout decides where loaded robots can go.

## Base design

The base layout is rack pairs two columns wide with an aisle every third column, an open row above the goals, and clear cells in the two columns above the goals. Every base shelf has a route to a goal.
The helper below draws a design. The later blocks on this page reuse it:

```python
import numpy as np
from codesign_gym import make

H, W = 10, 10                                                    # Warehouse-Small
GOALS = [(H - 1, W // 2 - 1), (H - 1, W // 2)]
CELLS = [(r, c) for r in range(H) for c in range(W) if (r, c) not in GOALS]   # bit i <-> CELLS[i]


def show(design):
    """Draw a design: '#' shelf, 'G' goal, '.' free floor."""
    grid = np.full((H, W), ".")
    for (r, c), bit in zip(CELLS, design):
        if bit:
            grid[r, c] = "#"
    for r, c in GOALS:
        grid[r, c] = "G"
    print("\n".join("".join(row) for row in grid))


env, base = make("Warehouse-Small")
print(base.shape, base.dtype, base.sum())
show(base)
env.close()
```

```text
(98,) int8 25
..........
.##....##.
.##....##.
..#....##.
.##....##.
.#.....##.
.##....##.
.##.....#.
..........
....GG....
```

## Build a design

Draw the layout as text and convert it with the same cell order:

```python
PICTURE = """
..........
.##.##.##.
.##.##.##.
.##.##.##.
.##.##.##.
..........
.#......#.
.#......#.
..........
....GG....
"""


def from_picture(picture):
    rows = picture.split()
    return np.array([rows[r][c] == "#" for r, c in CELLS], dtype=np.int8)


env, base = make("Warehouse-Small")
design = from_picture(PICTURE)
print(design.sum(), env.validate_embodiment(design))        # 28 True
env.close()
```

Use an integer dtype such as `int8`. A boolean or float array is rejected, even with 0/1 values.

## What is rejected

`validate_embodiment` returns `False`, and `reset` returns `rejected=True`, for:

- a vector of the wrong length,
- a non-integer dtype (including `bool`), or a value other than 0 and 1,
- fewer shelves than the minimum, or more than `H · W − robots`.

A layout that traps robots or blocks every route to the goals is still valid. It just earns little reward.

```python
env, base = make("Warehouse-Small")
too_few = base.copy()
too_few[np.flatnonzero(too_few)[0]] = 0                         # 24 shelves, below the minimum of 25
print(env.validate_embodiment(too_few))                         # False
print(env.validate_embodiment(base.astype(bool)))               # False: bool is not an integer dtype
obs, rejected, info = env.reset(seed=0, embodiment=too_few)
print(rejected)                                                 # True
env.close()
```

A rejected reset does not raise, but `step` raises `RuntimeError` until that copy is reset with a valid layout.
Shelves are fixed when an episode starts, so `step` ignores any design with a `UserWarning` ("Transition-level embodiment changes are not permitted"), valid or not. Autoreset reinstalls the copy's current layout.

With `make_vec`, pass one design (used by every copy), an `(num_envs, n)` array, or a list with a design or `None` per copy. `None` means the base layout.

## Build options

`make` forwards keyword arguments to the preset's config (`envs.robot_warehouse.config.RobotWarehouseConfig`):

| Kwarg | Effect |
|---|---|
| `num_agents` | Number of robots (changes the action and observation size) |
| `num_shelves` | Minimum shelf count |
| `request_queue_size` | Shelves requested at once (at most `num_shelves`) |
| `time_limit` | Steps per episode (default 500) |
| `sensor_range` | Radius of each robot's view (default 1, a 3×3 view) |
| `shelf_rows`, `shelf_columns`, `column_height` | Set the grid: `H = (column_height + 1) · shelf_rows + 2`, `W = 3 · shelf_columns + 1`; `shelf_columns` must be odd |

The base layout is rebuilt for the new settings. If the minimum shelf count does not fit the base pattern, `make` raises `ValueError`.

```python
env, base = make("Warehouse-Small", num_agents=3, time_limit=1000)
print(env.action_space, env.observation_space.shape)            # MultiDiscrete([5 5 5]) (198,)
env.close()
```
