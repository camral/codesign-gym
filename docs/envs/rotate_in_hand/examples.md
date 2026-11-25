# Examples

These examples need the `morphhand` extra and are not run when the docs are built.

## Reshape the hand during an episode

`step(action, embodiment)` changes the design from that step on, with no reset.
Here every copy starts flat and raises its fingertip pads a little more every 25 steps:

```python
# notest: needs the morphhand extra
import numpy as np
import torch
from codesign_gym import make_vec

envs, base = make_vec("RotateInHand-Cube", 4)
TIPS = np.r_[131:140, 167:176, 203:212, 227:245]    # the fingertip pads (ring, middle, index, thumb)

obs, rejected, info = envs.reset(seed=0, embodiment=base)
design = base.copy()
for t in range(200):
    action = torch.zeros((4, 19))                   # replace with a policy
    if t % 25 == 0 and t > 0:
        design[TIPS] = min(1.0, design[TIPS].max() + 0.2)
        obs, reward, terminated, truncated, info = envs.step(action, embodiment=design)
    else:
        obs, reward, terminated, truncated, info = envs.step(action)
print(info["final_return"], envs.goals_reached)
envs.close()
```

One design is broadcast to every copy. A copy that ends keeps its latest design when it autoresets.

## Change only some copies

A per-copy list sets the copies that have a design and leaves `None` copies as they are.
A partial reset restarts chosen copies, with their own designs:

```python
# notest: needs the morphhand extra
import numpy as np
import torch
from codesign_gym import make_vec

envs, base = make_vec("RotateInHand-Cube", 4)
obs, rejected, info = envs.reset(seed=0)            # every copy on the base design

wide = base.copy()
wide[246] = 1.0                                     # fingers 15 mm wider
obs, reward, terminated, truncated, info = envs.step(torch.zeros((4, 19)), embodiment=[wide, None, None, wide])

mask = np.array([False, True, False, False])
spread = base.copy()
spread[245] = 1.0
obs, rejected, info = envs.reset(options={"reset_mask": mask}, embodiment=[None, spread, None, None])
print(info["embodiment"][:, 245:])                  # copies 0 and 3 wide, copy 1 spread, copy 2 base
envs.close()
```

At reset, a `None` entry means the base design; at step, it means "keep the current one".

## Freeze the design for evaluation

An evaluation env ignores designs passed to `step`, so a fixed design cannot change mid-episode:

```python
# notest: needs the morphhand extra
import warnings
import torch
from codesign_gym import make_vec

envs, base = make_vec("RotateInHand-Cube", 2, is_eval_env=True)
design = base.copy()
design[0:80] = 1.0                                   # the palm pad raised 2 cm
obs, rejected, info = envs.reset(seed=0, embodiment=design)
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    envs.step(torch.zeros((2, 19)), embodiment=base)     # ignored
print(caught[0].message)                             # Transition-level embodiment changes are not permitted
envs.close()
```
