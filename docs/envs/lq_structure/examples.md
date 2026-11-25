# Examples

## A feedback policy

In `LQStructure-LQR` every mass is actuated, so action entry `i` is the force on mass `i`. Damp the chain with velocity feedback:

```python
import numpy as np
from codesign_gym import make

env, base = make("LQStructure-LQR")
obs, rejected, info = env.reset(seed=0, embodiment=base)
total = 0.0
for _ in range(200):
    velocities = obs[6:12]
    obs, reward, terminated, truncated, info = env.step(-2.0 * velocities)
    total += reward
print(f"return {total:.3f}")
env.close()
```

## Compare actuator placements

In `LQStructure-LQG`, actuator and sensor placement is part of the design. Evaluate three placements in parallel:

```python
import numpy as np
from codesign_gym import make_vec

envs, base = make_vec("LQStructure-LQG", 3)
placements = {
    "wall end": [1, 1, 0, 0, 0, 0],
    "tip end": [0, 0, 0, 0, 1, 1],
    "spread": [1, 0, 0, 1, 0, 1],
}
designs = [{**base, "actuators": np.array(p, dtype=np.int8)} for p in placements.values()]


def policy(obs):
    """Velocity feedback on the actuated masses, in actuator order (see Observations, actions and rewards)."""
    actions = np.zeros((len(obs), 6))
    for i, o in enumerate(obs):
        damping_force = -2.0 * o[6:12][o[12:18] == 1]      # velocities of actuated masses
        actions[i, : damping_force.size] = damping_force
    return actions


obs, rejected, info = envs.reset(seed=0, embodiment=designs)
returns = np.zeros(3)
for _ in range(200):
    obs, reward, terminated, truncated, info = envs.step(policy(obs))
    returns += reward
for name, value in zip(placements, returns):
    print(f"{name:9s} {value:.3f}")
envs.close()
```

## The optimal controller

For a fixed design, the optimal controller is computable exactly. The ground-truth script solves it and rolls it out in the environment:

```bash
python -m envs.lq_structure.ground_truth_demo --setup lqr --render off
python -m envs.lq_structure.ground_truth_demo --setup lqg --episodes 200 --render off
```

Use it to check how close a co-design method gets to the best policy for its final design.
