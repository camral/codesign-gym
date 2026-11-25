# Adding an environment

An environment is a subclass of {class}`core.codesign_vec_env.CoDesignVecEnv` plus one row in the registry.

## 1. Write the vector environment

The base class supplies the public `reset`, `step` and `validate_embodiment`. You implement three hooks with your own vectorization (NumPy, JAX `vmap`, batched simulator worlds, processes):

- `_reset(seed, options, embodiment)` returns `(obs, embodiment_rejected, info)`,
- `_step(actions, embodiment)` returns `(obs, rewards, terminations, truncations, info)`,
- `_validate_embodiment(embodiment)` returns one bool per copy.

This toy environment pushes a point mass to the origin. The design is the mass:

```python
import numpy as np
from gymnasium import spaces
from core.codesign_vec_env import CoDesignVecEnv


class PointMassVectorEnv(CoDesignVecEnv):
    metadata = {**CoDesignVecEnv.metadata, "supports_transition_embodiment": False}

    def __init__(self, num_envs, *, is_eval_env=False):
        super().__init__(
            num_envs,
            single_observation_space=spaces.Box(-np.inf, np.inf, (2,), np.float64),
            single_action_space=spaces.Box(-1.0, 1.0, (1,), np.float64),
            embodiment_space=spaces.Box(0.1, 10.0, (1,), np.float64),
            base_embodiment=np.array([1.0]),
            is_eval_env=is_eval_env,
        )
        self.mass = np.ones(num_envs)
        self.state = np.zeros((num_envs, 2))
        self.t = np.zeros(num_envs, dtype=int)

    def _validate_embodiment(self, embodiment):
        designs = self._designs(embodiment)
        return np.array([self.embodiment_space.contains(np.asarray(d, np.float64)) for d in designs])

    def _designs(self, embodiment):
        if embodiment is None:
            return [self.base_embodiment] * self.num_envs
        if isinstance(embodiment, (list, tuple)):
            return [self.base_embodiment if d is None else d for d in embodiment]
        return [embodiment] * self.num_envs

    def _reset(self, *, seed=None, options=None, embodiment=None):
        rng = np.random.default_rng(seed)
        rejected = ~self._validate_embodiment(embodiment)
        self.mass = np.array([float(np.asarray(d)[0]) for d in self._designs(embodiment)])
        self.state = np.c_[rng.uniform(-1, 1, self.num_envs), np.zeros(self.num_envs)]
        self.t[:] = 0
        return self.state.copy(), rejected, {}

    def _step(self, actions, embodiment=None):
        self.state[:, 1] += 0.1 * actions[:, 0] / self.mass      # velocity
        self.state[:, 0] += 0.1 * self.state[:, 1]               # position
        self.t += 1
        reward = -self.state[:, 0] ** 2 - 0.01 * self.mass        # heavier is slower but costs more
        truncated = self.t >= 100
        self.t[truncated] = 0                                     # (a real env autoresets here)
        return self.state.copy(), reward, np.zeros(self.num_envs, bool), truncated, {}


envs = PointMassVectorEnv(4)
obs, rejected, info = envs.reset(seed=0, embodiment=[np.array([0.5]), None, np.array([20.0]), None])
print(rejected)                                   # [False False  True False]: 20 kg is out of bounds
```

Rules to follow:

- **Validate structure only.** Reject impossible designs, never bad ones. Low reward is the signal for a bad design.
- **Keep spaces fixed.** Size observation and action spaces for the largest design, and pad and mask smaller ones.
- **Do not observe the design** unless the task needs it; most environments here keep it out of the observation.
- **Set `supports_transition_embodiment`** to `False` if the design cannot change mid-episode. The base class then ignores a design passed to `step` and warns.

## 2. Add a builder

Create `src/envs/<name>/registration.py` with a `build_vec` function. It returns the vector environment, which must expose `base_embodiment`:

```python
# src/envs/point_mass/registration.py  # notest
from .vector_env import PointMassVectorEnv


def build_vec(num_envs, *, is_eval_env=False, **cfg):
    return PointMassVectorEnv(num_envs, is_eval_env=is_eval_env, **cfg)
```

## 3. Register the id

Add a row to `REGISTRY` in `src/codesign_gym/registry.py`. The builder is imported only when the id is requested, so heavy dependencies load lazily:

```python
"PointMass-v0": ("envs.point_mass.registration:build_vec", {}),  # notest
```

Several ids can share one builder with a discriminator keyword, for example `{"setup": "hard"}`.

## 4. Test and document

- Add tests under `tests/` (see `tests/test_lq_vector.py` for the contract checks).
- Write `src/envs/<name>/README.md`, then add a section under `docs/envs/<name>/` that includes it (copy `docs/envs/lq_structure/`), and add the folder to `ENV_SECTIONS` in `docs/conf.py`.
