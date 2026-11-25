# Examples

These examples need the `morphhand` extra and are not run when the docs are built.

## Grow the pads after the grasp

A design can follow the task's phases. Here each copy keeps flat pads until it first holds the object clear of the table, then raises its fingertip pads for the turning. A copy that ends keeps the raised pads when it respawns:

```python
# notest: needs the morphhand extra
import numpy as np
import torch
from codesign_gym import make_vec

envs, base = make_vec("HandPickUp-Cube", 4)
grippy = base.copy()
grippy[np.r_[131:140, 167:176, 203:212, 227:245]] = 1.0   # fingertip pads raised 1 cm

obs, rejected, info = envs.reset(seed=0, embodiment=base)
switched = torch.zeros(4, dtype=torch.bool, device=obs.device)
for t in range(375):
    action = torch.zeros((4, 19))                     # replace with a policy
    held = (obs[:, 56] == 1) & (obs[:, 57] == 0)      # touching the hand, off the table
    new = held & ~switched
    designs = [grippy if bool(n) else None for n in new]   # None keeps a copy's design
    obs, reward, terminated, truncated, info = envs.step(action, embodiment=designs if new.any() else None)
    switched |= new
print(info["final_return"])
envs.close()
```

## Compare designs on one object

Different designs can run side by side, one per copy:

```python
# notest: needs the morphhand extra
import numpy as np
import torch
from codesign_gym import make_vec

envs, base = make_vec("HandPickUp-Hammer", 3)
designs = np.stack([base, base, base])
designs[1, 245] = 1.0                                 # fingers spread 15 mm
designs[2, 246] = 1.0                                 # fingers 15 mm wider
obs, rejected, info = envs.reset(seed=0, embodiment=designs)
returns = torch.zeros(3, device=obs.device)
done = torch.zeros(3, dtype=torch.bool, device=obs.device)
for t in range(175):                                  # the hammers' 7 s episode
    obs, reward, terminated, truncated, info = envs.step(torch.zeros((3, 22)))   # the hammers' hand has 22 actuators
    returns += reward * ~done
    done |= terminated | truncated
print(returns)
envs.close()
```
