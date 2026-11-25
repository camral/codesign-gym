# Examples

All examples run on the CPU in well under a minute. On a GPU, drop `device="cpu"` and raise the number of copies.

## Evaluate several designs at once

Each copy of a `make_vec` env can hold its own design. Here four bodies face the same kind of throw while holding still
(an all-zero action holds every joint), and each one's closest approach of ball to glove is recorded. A copy stops being
counted once its first episode ends.

```python
import numpy as np
import torch
from codesign_gym import make_vec

envs, base = make_vec("BallCatcher-Pitch", 4, device="cpu")

long_arm = base.copy()
long_arm[[23, 26]] = 0.4              # left arm: upper and lower link 0.4 m longer
long_legs = base.copy()
long_legs[[5, 11, 17]] = 0.3          # all three shins 0.3 m longer
wide = base.copy()
wide[[1, 7, 12]] = 0.3                # thighs swung 0.3 m outward (front left +y, front right -y, back -x)
designs = {"base": base, "long arm": long_arm, "long legs": long_legs, "wide stance": wide}

obs, rejected, info = envs.reset(seed=0, embodiment=list(designs.values()))
returns = torch.zeros(4)
closest = torch.full((4,), float("inf"))
running = torch.ones(4, dtype=torch.bool)
for _ in range(30):                   # 1.5 s; the ball arrives after about 0.4 s
    obs, reward, terminated, truncated, info = envs.step(np.zeros((4, 19), dtype=np.float32))
    returns += reward * running
    gap = torch.linalg.norm(obs[:, 67:70], dim=1) * 10.0     # ball minus glove, back in metres
    closest = torch.where(running, torch.minimum(closest, gap), closest)
    running &= ~(terminated | truncated)
for i, name in enumerate(designs):
    print(f"{name:12s} return {returns[i]:.3f}  closest {closest[i]:.2f} m")
envs.close()
```

A body that cannot move will rarely catch anything, so the returns here come from the small progress term. The point
is the pattern: one `reset` with one design per copy, then ordinary batched stepping.

## Sweep one slot

Sweep the length of the left arm's upper link (slot 23) over its whole range, one value per copy, and compare where the
glove ends up:

```python
import numpy as np
from codesign_gym import make_vec

values = np.linspace(0.0, 1.0, 5)
envs, base = make_vec("BallCatcher-Pitch", len(values), device="cpu")
designs = np.repeat(base[None], len(values), axis=0)
designs[:, 23] = values

obs, rejected, info = envs.reset(seed=0, embodiment=designs)
for _ in range(5):
    obs, reward, terminated, truncated, info = envs.step(np.zeros((len(values), 19), dtype=np.float32))
glove = obs[:, 57:60] * 10.0                                # goal site position, metres
for v, extra, p in zip(values, envs.env_offsets[:, 23], glove):
    print(f"slot 23 = {v:.2f} (+{extra:.2f} m)  glove at x={p[0]:.2f} y={p[1]:.2f} z={p[2]:.2f}")
envs.close()
```

## Render one copy

With `render_mode="rgb_array"`, `render()` returns one frame per copy and `render(env_idx=i)` returns copy `i` alone, as
a `(480, 480, 3)` uint8 array. The frame shows that copy's own design.

```python
import os
import tempfile

import numpy as np
from PIL import Image
from codesign_gym import make_vec

envs, base = make_vec("BallCatcher-Arc", 2, device="cpu", render_mode="rgb_array")
long_arm = base.copy()
long_arm[[23, 26, 35, 38]] = 0.5                            # both arms, upper and lower links
obs, rejected, info = envs.reset(seed=1, embodiment=[base, long_arm])
for _ in range(10):
    envs.step(np.zeros((2, 19), dtype=np.float32))

frame = envs.render(env_idx=1)
print(frame.shape, frame.dtype)                             # (480, 480, 3) uint8
path = os.path.join(tempfile.gettempdir(), "ball_catcher_long_arm.png")
Image.fromarray(frame).save(path)
print("saved", path)
envs.close()
```

Rendering uses MuJoCo's offscreen renderer, so a headless Linux machine needs an EGL or OSMesa context
(`MUJOCO_GL=egl`).
