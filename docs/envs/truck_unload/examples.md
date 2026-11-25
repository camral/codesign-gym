# Examples

All examples run on the CPU in under a minute. On a GPU, drop `device="cpu"` and raise the number of copies.

## Sweep the stacker's arm length

Set both stacker arm links (slots 4 and 5) to a different length in each copy and read how far the stacker's tip starts
from its target box. The stacker's block starts at offset 57, and the box-minus-tip vector sits at `[46:49]` in it,
scaled by 1.75 m:

```python
import numpy as np
import torch
from codesign_gym import make_vec

values = np.array([-0.06, 0.0, 0.25, 0.5, 1.0])
envs, base = make_vec("TruckUnload-Single", len(values), device="cpu")
designs = np.repeat(base[None], len(values), axis=0)
designs[:, [4, 5]] = values[:, None]

obs, rejected, info = envs.reset(embodiment=designs)
stacker = obs[:, 57:114]
gap = torch.linalg.norm(stacker[:, 46:49], dim=1) * 1.75
for v, g, has_target in zip(values, gap, stacker[:, 56]):
    link = 0.1 + v * envs.max_offset
    print(f"stacker links {link:.2f} m: tip {g:.2f} m from its target box (has target: {bool(has_target)})")
envs.close()
```

## Roll out and split the reward by robot

`info["reward_per_robot"]` splits every step's reward between the loader and the stackers, so a centralised policy can
be scored per robot. Here three designs run 100 control steps (4 s) of random actions:

```python
envs, base = make_vec("TruckUnload-Multi", 3, device="cpu")
long_arms = base.copy()
long_arms[[1, 2, 4, 5, 7, 8]] = 0.3                     # every arm link 0.3 m longer
short_arms = envs.embodiment_space.low.copy()           # every link at its shortest
short_arms[[0, 3, 6]] = 0.0                             # stock columns
designs = {"base": base, "long arms": long_arms, "short arms": short_arms}

obs, rejected, info = envs.reset(embodiment=list(designs.values()))
rng = np.random.default_rng(0)
per_robot = torch.zeros(3, 3)
running = torch.ones(3, dtype=torch.bool)
for _ in range(100):
    action = rng.uniform(-1.0, 1.0, size=(3, 24)).astype(np.float32)
    obs, reward, terminated, truncated, info = envs.step(action)
    per_robot += info["reward_per_robot"] * running[:, None]
    running &= ~(terminated | truncated)
for name, r, alive in zip(designs, per_robot, running):
    print(f"{name:10s} loader {r[0]:+.3f}  red {r[1]:+.3f}  blue {r[2]:+.3f}  still running: {bool(alive)}")
envs.close()
```

Random actions usually drive a tip out of its field within a few seconds, which ends the episode; the `running` mask
stops counting a copy once that happens.

## Render one copy

With `render_mode="rgb_array"`, `render(env_idx=i)` returns copy `i` as a `(480, 480, 3)` uint8 array, drawn with that
copy's design:

```python
import os
import tempfile

from PIL import Image

envs, base = make_vec("TruckUnload-Single", 2, device="cpu", render_mode="rgb_array")
long_stacker = base.copy()
long_stacker[[4, 5]] = 0.5
obs, rejected, info = envs.reset(embodiment=[base, long_stacker])
for _ in range(5):
    envs.step(np.zeros((2, 16), dtype=np.float32))
frame = envs.render(env_idx=1)
print(frame.shape, frame.dtype)                         # (480, 480, 3) uint8
path = os.path.join(tempfile.gettempdir(), "truck_unload_long_stacker.png")
Image.fromarray(frame).save(path)
print("saved", path)
envs.close()
```

Rendering uses MuJoCo's offscreen renderer, so a headless Linux machine needs an EGL or OSMesa context
(`MUJOCO_GL=egl`).
