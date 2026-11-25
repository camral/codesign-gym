# Changing the design during an episode

In LQ Structure, Microgrid, Racing and the Shape-shifting Hand, the design can change between two steps.
This lets you study co-design optimization which makes use of non-stationary embodiments. In an evaluation environment, mid-episode design changes are disabled even for these environments.
We believe that there might exist interesting co-design algorithms which make use of mid-episode embodiment changes to evaluate new designs!

## One environment

Pass the new design to `step`. It applies from that step on, and later steps keep it:

```python
import numpy as np
from codesign_gym import make

env, base = make("LQStructure-LQR")
env.reset(seed=0, embodiment=base)

stiffer = {**base, "stiffness": np.full(6, 4.5)}
obs, reward, terminated, truncated, info = env.step(np.zeros(6), embodiment=stiffer)
print(info["embodiment"]["stiffness"])      # [4.5 ...]: changed from this step

obs, reward, terminated, truncated, info = env.step(np.zeros(6))
print(info["embodiment"]["stiffness"])      # still [4.5 ...]
env.close()
```

## A vector environment

Give one entry per copy. `None` keeps that copy's current design:

```python
import numpy as np
from codesign_gym import make_vec

envs, base = make_vec("LQStructure-LQR", 2)
envs.reset(seed=0)
stiffer = {**base, "stiffness": np.full(6, 4.5)}
obs, reward, terminated, truncated, info = envs.step(np.zeros((2, 6)), embodiment=[stiffer, None])
print([d["stiffness"][0] for d in info["embodiment"]])   # [4.5, 2.75]: only copy 0 changed
envs.close()
```

## Evaluation environments

An evaluation environment keeps the design fixed. It ignores a design passed to `step` and warns:

```python
import numpy as np
from codesign_gym import make

env, base = make("LQStructure-LQR", is_eval_env=True)
env.reset(seed=0, embodiment=base)
obs, reward, terminated, truncated, info = env.step(np.zeros(6), embodiment={**base, "stiffness": np.full(6, 4.5)})
print(info["embodiment"]["stiffness"][0])    # 2.75: unchanged (with a warning)
env.close()
```

Use an evaluation environment to score a final design, so a policy cannot reshape it.

## Environments without mid-episode changes

The other environments fix the design for the whole episode (a MuJoCo body is rebuilt at reset, a warehouse layout is fixed). They ignore a design passed to `step` and warn. Change their design at `reset`.
