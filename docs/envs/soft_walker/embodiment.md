# Design space

The body is a grid of 2 cm cells: 8 x 6 in the 2D models (`Spring2D`, `MPM2D`) and 8 x 6 x 6 in the 3D models (`MPM3D`, `Beam3D`, `HexFEM3D`).
Grid axis 0 is `x`, the walking direction; axis 1 is `y`, up; axis 2 (3D only) is `z`, across the body.

The design is one flat float64 vector. Each cell contributes `7 + dim` values, and the cells follow each other in C order over the grid (the last axis changes fastest).
So reshaping the vector to `(*grid, 7 + dim)` gives one row of fields per cell.

## Per-cell fields

| Field | 2D index | 3D index | Bounds | Meaning |
|---|---|---|---|---|
| `occupancy` | 0 | 0 | 0 – 1 | The cell is solid when `occupancy >= 0.5`. It does not scale mass, stiffness or actuation. |
| `young` | 1 | 1 | 5 000 – 100 000 | Young's modulus (Pa) |
| `density` | 2 | 2 | 800 – 1 250 | Density (kg/m³) |
| `strength` | 3 | 3 | 0 – 1 | Actuator strength |
| `direction` | 4–5 | 4–6 | -1 – 1 each | Fibre direction of a directional actuator (`x, y[, z]`) |
| `kind` | 6 | 7 | 0 – 2, integer | Actuator kind: 0 none, 1 directional, 2 isotropic |
| `channel` | 7 | 8 | 0 – 3 (2D), 0 – 15 (3D), integer | Action channel that drives the cell |
| `poisson` | 8 | 9 | 0.25 – 0.45 | Poisson's ratio (the spring lattice stores it but has no continuum Poisson control) |

| Model | Grid | Cells | Values per cell | Design length | Channels |
|---|---|---|---|---|---|
| `Spring2D`, `MPM2D` | 8 x 6 | 48 | 9 | 432 | 4 |
| `MPM3D`, `Beam3D`, `HexFEM3D` | 8 x 6 x 6 | 288 | 10 | 2880 | 16 |

The terrain suffix (`Stairs`, `Gaps`) does not change the design space.
`env.embodiment_space` is a `Box` that prints this layout and carries the grid as `embodiment_space.grid`:

```python
from codesign_gym import make

env, base = make("SoftWalker-MPM2D")
print(env.embodiment_space)
print(env.embodiment_space.grid, base.shape)    # (8, 6) (432,)
env.close()
```

```text
CellEmbodimentSpace(grid=(8, 6), shape=(432,), dtype=float64, per_cell={occupancy=[0, 1], young=[5000, 100000], density=[800, 1250], strength=[0, 1], direction[2]=[-1, 1], kind=[0, 2], channel=[0, 3], poisson=[0.25, 0.45]})
(8, 6) (432,)
```

## Base design

The base design is a solid block, with every cell a directional actuator and a Poisson's ratio of 0.35. The other fields follow Warp's own builder defaults, so they differ by model:

| Model | `young` | `density` | `strength` | `direction` | `channel` |
|---|---|---|---|---|---|
| `Spring2D`, `HexFEM3D` | 20 000 | 1 000 | 0.8 | `+y` | flat cell index mod channels |
| `Beam3D` | 20 000 | 1 000 | 0.8 | `+x` | flat cell index mod channels |
| `MPM2D`, `MPM3D` | 12 000 | 1 000 | 0.95 | `+y` | 0 for every cell |

The MPM base body cannot walk: every cell is on channel 0 with a vertical fibre, so any gait is symmetric and moves it nowhere. Finding a body that walks is part of the problem.

## Reading and writing a design

Reshape the flat vector to one row per cell, edit it, and flatten it back. The helpers below name the fields so the indices don't have to be remembered:

```python
import numpy as np
from codesign_gym import make


def field_slices(dim):
    """Field name -> slice into one cell's row (direction takes dim entries)."""
    names = ["occupancy", "young", "density", "strength", "direction", "kind", "channel", "poisson"]
    out, start = {}, 0
    for name in names:
        width = dim if name == "direction" else 1
        out[name] = slice(start, start + width)
        start += width
    return out


def to_grid(vector, grid):
    """Flat design -> array of shape (*grid, 7 + dim), one row per cell."""
    return np.asarray(vector, dtype=float).reshape(*grid, 7 + len(grid)).copy()


def to_vector(cells):
    """Array of shape (*grid, 7 + dim) -> flat design."""
    return cells.reshape(-1)


env, base = make("SoftWalker-Spring2D")
grid = env.embodiment_space.grid
F = field_slices(len(grid))
cells = to_grid(base, grid)
print(cells.shape)                               # (8, 6, 9)
print(cells[..., F["channel"]][..., 0])          # channel of every cell, x down the rows
assert np.array_equal(to_vector(cells), base)
env.close()
```

```text
(8, 6, 9)
[[0. 1. 2. 3. 0. 1.]
 [2. 3. 0. 1. 2. 3.]
 [0. 1. 2. 3. 0. 1.]
 [2. 3. 0. 1. 2. 3.]
 [0. 1. 2. 3. 0. 1.]
 [2. 3. 0. 1. 2. 3.]
 [0. 1. 2. 3. 0. 1.]
 [2. 3. 0. 1. 2. 3.]]
```

`envs.soft_walker.embodiment` has the same conversion as `to_cells(vector, dim)` (a dict of per-cell arrays) and `from_cells(props, dim, dtype)`.

## Build a design

This MPM2D body has two legs. The front leg and back leg sit on channels 0 and 1 with vertical fibres, and the torso sits on channel 2 with horizontal fibres, so the two legs can be driven out of phase:

```python
env, base = make("SoftWalker-MPM2D")
grid = env.embodiment_space.grid                 # (8, 6): x, y
F = field_slices(2)
cells = to_grid(base, grid)

occupancy = np.ones(grid)
occupancy[2:6, 0:2] = 0                          # cut the gap between the legs
cells[..., F["occupancy"]] = occupancy[..., None]
cells[0:2, 0:2, F["channel"]] = 0                # back leg
cells[6:8, 0:2, F["channel"]] = 1                # front leg
cells[:, 2:6, F["channel"]] = 2                  # torso
cells[:, 2:6, F["direction"]] = [1.0, 0.0]       # torso fibres along x
cells[:, 4:6, F["young"]] = 40000.0              # stiffer top

design = to_vector(cells)
print(env.validate_embodiment(design))           # True
env.close()
```

Fields of an empty cell are ignored by the physics, but they must still be in bounds and have integer `kind` and `channel`. Whether an empty cell may hold a directional actuator without a fibre is up to the Warp builder: MPM2D and MPM3D accept it, while Spring2D, Beam3D and HexFEM3D reject it.

## What is rejected

`validate_embodiment` returns `False`, and `reset` returns `rejected=True` for that copy, when the design:

- has the wrong shape, a non-finite value, or a value outside its bounds,
- has a non-integer `kind` or `channel`,
- has a directional actuator (`kind == 1`) with a zero fibre direction in a solid cell (`DIRECTIONAL_ACTUATOR_WITHOUT_FIBER`),
- has no solid cell, or solid cells that are not all face-connected (`DISCONNECTED_GEOMETRY`),
- has cells that touch only along an edge or at a corner (`NONMANIFOLD_JUNCTION`),
- fails the Warp builder's own checks when the body is built.

`validate_embodiment` builds the body, so it agrees with `reset`. A rejected `reset` does not raise: it puts the reasons in `info["embodiment_errors"]` and stepping the vector env raises (`ResetNeeded`) while any copy is uninitialised, until that copy is reset with a valid design.
`envs.soft_walker.embodiment.structural_errors` returns the same reasons, without the build step:

```python
from envs.soft_walker.embodiment import structural_errors

env, base = make("SoftWalker-Spring2D")
grid = env.embodiment_space.grid
cells = to_grid(base, grid)
cells[3, :, F["occupancy"]] = 0                  # cut the body in two
print(env.validate_embodiment(to_vector(cells)), structural_errors(to_vector(cells), grid, 4))

cells = to_grid(base, grid)
cells[..., F["occupancy"]] = 0
cells[0, 0, F["occupancy"]] = cells[1, 1, F["occupancy"]] = 1    # two cells touching at a corner
print(structural_errors(to_vector(cells), grid, 4))

cells = to_grid(base, grid)
cells[0, 0, F["kind"]] = 1.5
print(structural_errors(to_vector(cells), grid, 4))
env.close()
```

```text
False ['DISCONNECTED_GEOMETRY']
['DISCONNECTED_GEOMETRY', 'NONMANIFOLD_JUNCTION']
['INTEGER_REQUIRED: kind/channel']
```

A design that passes can still be rejected at reset: the body first settles under gravity for 0.2 s with zero command, and if that settling fails numerically, `reset` returns `rejected=True` with `Initial settling failed ...` in `info["embodiment_errors"]`.

## Changing the design

The body is rebuilt at every reset, so a new design goes in through `reset(embodiment=...)`.
A design passed to `step` is dropped with a `UserWarning`. On autoreset, a copy keeps its last accepted design.
