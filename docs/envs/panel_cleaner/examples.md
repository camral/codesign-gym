# Examples

All examples run on the CPU in about a minute. On a GPU, drop `device="cpu"` and raise the number of copies.

## How far each design starts from the first cell

In `SolarCleaner-Reach` the cells sit high and well away from the spawn. Give each copy a different body and read the
roller-to-goal vector (`obs[65:68]`, scaled by 1.75 m) straight after the reset, which already reflects the design:

```python
import numpy as np
import torch
from codesign_gym import make_vec

ARM_LENGTH = [26, 29, 30, 34, 38, 41, 44, 47]              # the slot of each arm link that lengthens it
envs, base = make_vec("SolarCleaner-Reach", 3, device="cpu")
long_arm = base.copy()
long_arm[ARM_LENGTH] = 1.0                                  # every arm link 0.15 m longer
long_legs = base.copy()
long_legs[[2, 5, 8, 11, 14, 17, 20, 23]] = 1.0              # every leg link 0.15 m longer
designs = {"base": base, "long arm": long_arm, "long legs": long_legs}

obs, rejected, info = envs.reset(seed=0, embodiment=list(designs.values()))
gap = torch.linalg.norm(obs[:, 65:68], dim=1) * 1.75
for name, g in zip(designs, gap):
    print(f"{name:10s} roller starts {g:.2f} m from the first cell")
```

## Roll out random actions and count cleaned goals

The same env, stepped for 100 control steps (4 s) with random actions. `goals_cleaned` counts goals reached in the
current episode, and `info["final_return"]` holds the return of any copy that ended:

```python
rng = np.random.default_rng(0)
returns = torch.zeros(3)
for _ in range(100):
    action = rng.uniform(-1.0, 1.0, size=(3, 18)).astype(np.float32)
    obs, reward, terminated, truncated, info = envs.step(action)
    returns += reward
    done = terminated | truncated
    if done.any():
        print("ended:", [n for n, d in zip(designs, done) if d], "returns", info["final_return"][done])
for name, r, g in zip(designs, returns, envs.goals_cleaned):
    print(f"{name:10s} reward so far {r:+.3f}, goals reached {int(g)}")
```

## Swap the design of one copy

A partial reset gives one copy a new design while the others keep running. Here copy 2 switches to the long arm; copies
0 and 1 keep their episode and their design:

```python
mask = np.array([False, False, True])
obs, rejected, info = envs.reset(embodiment=[None, None, long_arm], options={"reset_mask": mask})
print(np.round(envs.env_offsets[:, 38], 2))                 # metres: [0.   0.15 0.15]
envs.close()
```

The `None` entries are ignored here because those copies are not reset.

## Render the tracking camera

An evaluation env renders from a camera that follows the robot. On `SolarCleaner-Stability` the ground and the panel
heights change every episode:

```python
import os
import tempfile

from PIL import Image

envs, base = make_vec("SolarCleaner-Stability", 1, device="cpu", is_eval_env=True, render_mode="rgb_array")
obs, rejected, info = envs.reset(seed=3)
for _ in range(20):
    envs.step(np.zeros((1, 18), dtype=np.float32))
frame = envs.render(env_idx=0)
print(frame.shape, frame.dtype)                             # (480, 480, 3) uint8
path = os.path.join(tempfile.gettempdir(), "solar_cleaner_stability.png")
Image.fromarray(frame).save(path)
print("saved", path)
envs.close()
```

Rendering uses MuJoCo's offscreen renderer, so a headless Linux machine needs an EGL or OSMesa context
(`MUJOCO_GL=egl`).
