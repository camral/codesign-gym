# A co-design loop

This tutorial optimizes the design of the LQ chain together with a simple policy.
It runs on a CPU in a few seconds.

The pattern works for every environment:

1. an **outer loop** proposes a batch of designs,
2. a **vector environment** evaluates them in parallel, one design per copy,
3. the outer loop keeps the best design, and the policy is updated from the same rollouts.

## Evaluate a batch of designs

`make_vec` gives one copy per candidate. `reset` takes a list with one design per copy, and `rejected` says which candidates are impossible.

```python
import numpy as np
from codesign_gym import make_vec

POP = 8
envs, base = make_vec("LQStructure-LQR", POP)


def rollout(envs, designs, gain, seed):
    """Run one episode per copy with the policy u = -gain * velocity. Returns the return of each copy."""
    obs, rejected, info = envs.reset(seed=seed, embodiment=designs)
    returns = np.zeros(len(designs))
    done = np.zeros(len(designs), dtype=bool)
    while not done.all():
        velocities = obs[:, 6:12]                       # obs = [positions(6), velocities(6), actuator mask(6), sensor mask(6)]
        obs, reward, terminated, truncated, info = envs.step(-gain * velocities)
        returns += np.where(done, 0.0, reward)
        done |= terminated | truncated
    return np.where(rejected, -np.inf, returns)


rng = np.random.default_rng(0)
designs = [envs.embodiment_space.sample() for _ in range(POP)]
print(rollout(envs, designs, gain=1.0, seed=0))
```

## Optimize design and policy together

The outer loop below is a minimal evolution strategy:
it keeps the best design, perturbs it to make the next batch, and tries a few policy gains on the way.
Replace it with CMA-ES, Bayesian optimization or an RL algorithm for the policy; the environment calls stay the same.

```python
def perturb(design, rng, scale=0.15):
    new = dict(design)
    for key, lo, hi in (("masses", 0.5, 2.0), ("stiffness", 0.5, 5.0), ("damping", 0.05, 1.0)):
        new[key] = np.clip(design[key] * np.exp(scale * rng.standard_normal(design[key].shape)), lo, hi)
    return new


best_design, best_gain, best_return = base, 1.0, -np.inf
for generation in range(5):
    designs = [best_design] + [perturb(best_design, rng) for _ in range(POP - 1)]
    for gain in (0.5, 1.0, 2.0):
        returns = rollout(envs, designs, gain, seed=0)        # same seed: candidates face the same start states
        i = int(np.argmax(returns))
        if returns[i] > best_return:
            best_design, best_gain, best_return = designs[i], gain, returns[i]
    print(f"generation {generation}: best return {best_return:.3f} with gain {best_gain}")

envs.close()
```

## What to change for other environments

- **The design proposal.** Read the environment's *Design space* page. A `Box` design can be perturbed like this; a mesh or team design needs its own operators.
- **The policy.** Here it is a fixed linear rule. In practice, train an RL policy across the batch; the rollouts above already give you `(obs, action, reward)` for every copy.
- **Rejected designs.** `rollout` scores them `-inf`. Choose your own penalty; the environment never repairs a design.
- **Scale.** GPU environments (MuJoCo, Soft Walker) run hundreds or thousands of copies in one `make_vec`.

The LQ chain also has an exact optimal controller for each fixed design, so you can check what your loop reaches:

```bash
python -m envs.lq_structure.ground_truth_demo --setup lqr --render off
```
