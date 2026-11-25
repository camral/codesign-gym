# Design space

The design is a `dict` of five arrays, one entry per mass in a chain of `n` masses (default `n = 6`).
Spring and damper `i` connect mass `i` to mass `i - 1` (mass 0 connects to a wall).

| Key | Space | Bounds | Meaning |
|---|---|---|---|
| `masses` | `Box(n,)` float64 | 0.5 – 2.0 | Mass of each element |
| `stiffness` | `Box(n,)` float64 | 0.5 – 5.0 | Stiffness of each spring |
| `damping` | `Box(n,)` float64 | 0.05 – 1.0 | Damping of each damper |
| `actuators` | `MultiBinary(n)` | 0 / 1 | Mass `i` has an actuator |
| `sensors` | `MultiBinary(n)` | 0 / 1 | Mass `i` is measured (LQG only) |

Which keys are designed depends on the id. A key that is not designed is fixed to its base value: its space is a length-`n` `Box(1, 1)` (int8), which admits only that value:

| Id | `actuators` | `sensors` | Extra constraints |
|---|---|---|---|
| `LQStructure-LQR` | fixed: all 1 | fixed: all 1 | — |
| `LQStructure-LQG` | designed | designed | at least one sensor |
| `LQStructure-QP` | designed | fixed: all 1 | total mass ≤ 8.25, at most 3 actuators |

Every setup needs at least one actuator.

## Base design

The base design is the midpoint of each bound, fully sensed and, for LQR and LQG, fully actuated. The QP base actuates only the first three masses (`[1, 1, 1, 0, 0, 0]`), so that it respects the three-actuator budget:

```python
from codesign_gym import make

env, base = make("LQStructure-LQR")
for key, value in base.items():
    print(key, value)
env.close()
```

```text
masses [1.25 1.25 1.25 1.25 1.25 1.25]
stiffness [2.75 2.75 2.75 2.75 2.75 2.75]
damping [0.525 0.525 0.525 0.525 0.525 0.525]
actuators [1 1 1 1 1 1]
sensors [1 1 1 1 1 1]
```

## Build a design

Start from the base design and change the keys you want:

```python
import numpy as np
from codesign_gym import make

env, base = make("LQStructure-LQG")
design = {
    **base,
    "masses": np.linspace(0.6, 1.8, 6),                     # light near the wall, heavy at the tip
    "actuators": np.array([0, 0, 1, 0, 0, 1], dtype=np.int8),
    "sensors": np.array([1, 0, 0, 0, 0, 1], dtype=np.int8),
}
print(env.validate_embodiment(design))                      # True
env.close()
```

## What is rejected

`validate_embodiment` returns `False`, and `reset` returns `rejected=True`, for:

- a value outside its bounds, or a non-binary placement,
- a change to a fixed key (for example, `actuators` in `LQStructure-LQR`),
- no actuator, or (LQG) no sensor,
- (QP) a total mass above the budget, or more actuators than allowed.

A rejected reset does not raise; `step` raises `RuntimeError` until a reset succeeds. An invalid design passed to `step` raises `ValueError`.

With `make_vec`, `validate_embodiment` returns a `(num_envs,)` bool array and never raises. It accepts a single dict (applied to every copy), a list of dicts (with `None` for "no design"), or a stacked dict whose values have a leading `num_envs` axis (a leading axis of 1 is applied to every copy). Wrong keys or a batch of the wrong size give `False`.

## Changing the size

`n` sets the chain length, and other `LQConfig` fields can be overridden through `make`:

```python
from codesign_gym import make

env, base = make("LQStructure-LQR", n=10, horizon=400)
print(env.observation_space.shape, base["masses"].shape)    # (40,) (10,)
env.close()
```
