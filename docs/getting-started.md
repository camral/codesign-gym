# Getting started

## Install

⚠️🚫🐍 **Instructions may not work with Conda, so do not use it.**

You need Python 3.11 or later.

```bash
git clone https://github.com/camral/codesign-gym
pip install -e ./codesign-gym
```

Most environments run on this base install. A few need an extra step:

| Environments | Extra step |
|---|---|
| Pokémon (`Pokenv-*`) | Node.js 22 or later, then `pip install -e './codesign-gym[showdown]'` and `codesign-gym-install-showdown`. See [Pokémon](envs/pokenv/index.md). |
| Shape-shifting Hand (`RotateInHand-*`, `HandPickUp-*`) | `pip install -e './codesign-gym[morphhand]'` (with uv, add `--no-sources`). |
| Racing (`Racing-*`) | `pip install --no-deps jax-pf@git+https://github.com/hzheng40/jax_pf f1tenth_gym_jax@git+https://github.com/f1tenth/f1tenth_gym_jax` |
| JAX on an NVIDIA GPU (Racing, Warehouse) | `codesign-gym-install-jax-gpu` |
| Tests | `pip install -e './codesign-gym[dev]'`, then `pytest` |

The `morphhand` extra replaces `mujoco-warp` with a fork for every environment in that Python environment.

Check the install:

```python
from codesign_gym import list_envs

print(len(list_envs()), "environments")
print(list_envs()[:5])
```

## Your first episode

Every environment is created with {func}`codesign_gym.make` (one copy) or {func}`codesign_gym.make_vec` (many copies).
Both return the environment and its **base design**, the default design the environment starts from.

```python
from codesign_gym import make

env, base = make("LQStructure-LQR")
print(env.embodiment_space)        # the space of all designs
print(base["masses"])              # the base design is a member of it

obs, rejected, info = env.reset(seed=0, embodiment=base)
episode_return = 0.0
terminated = truncated = False
while not (terminated or truncated):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    episode_return += reward
print("return", episode_return)
env.close()
```

Two things differ from plain Gymnasium:

- `reset` takes an `embodiment` and returns **three** values: `(obs, rejected, info)`. `rejected` is `True` when the design is impossible (see [Rejection](concepts.md#rejection)).
- `make` returns `(env, base)`, not just `env`.

## Try a design

A design is a member of `env.embodiment_space`. Here, the LQ chain gets heavier masses and stiffer springs:

```python
import numpy as np
from codesign_gym import make

env, base = make("LQStructure-LQR")
design = {**base, "masses": np.full(6, 1.8), "stiffness": np.full(6, 4.0)}
print(env.validate_embodiment(design))      # True: inside the bounds

obs, rejected, info = env.reset(seed=0, embodiment=design)
print(rejected)                             # False
env.close()
```

An impossible design, such as a mass outside its bounds, is rejected:

```python
import numpy as np
from codesign_gym import make

env, base = make("LQStructure-LQR")
too_heavy = {**base, "masses": np.full(6, 10.0)}
print(env.validate_embodiment(too_heavy))   # False
env.close()
```

## Next steps

- [Concepts](concepts.md) explains designs, rejection, vector environments, seeds and autoreset.
- [Tutorials](tutorials/index.md) build a full co-design loop.
- [Environments](envs/index.md) lists every environment with its design space and examples.
