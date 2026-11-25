# Examples

## A toggle-aware greedy policy

Because an action toggles a connection, a policy must check the current connections before acting. This one connects each user to its strongest built station and then sends the no-op while that connection holds.
Compare it with sending the strongest station every step, which disconnects every other step, and with doing nothing:

```python
import numpy as np
from codesign_gym import make

U, M = 15, 8                                             # Network-Urban: users, max_bs


def greedy(obs, mask):
    """Connect each user to its strongest built station; no-op once connected."""
    blocks = obs.reshape(-1, U, 2 * M + 1)               # works for one copy or many
    snr = np.where(mask[..., 1:], blocks[..., M:2 * M], -1.0)
    best = snr.argmax(-1)                                # station index per user
    connected = np.take_along_axis(blocks[..., :M], best[..., None], -1)[..., 0] == 1
    return np.where(connected, 0, best + 1).reshape(np.shape(mask)[:-1])


def always_best(obs, mask):
    blocks = obs.reshape(U, 2 * M + 1)
    return 1 + np.where(mask[:, 1:], blocks[:, M:2 * M], -1.0).argmax(-1)


env, base = make("Network-Urban")
for name, policy in [("no-op", lambda o, m: np.zeros(U, dtype=int)), ("always best", always_best), ("greedy", greedy)]:
    obs, rejected, info = env.reset(seed=0)
    total = 0.0
    for _ in range(100):
        obs, reward, terminated, truncated, info = env.step(policy(obs, info["action_mask"]))
        total += reward
    print(f"{name:12s} {total:7.2f}")
env.close()
```

```text
no-op        -100.00
always best   -76.31
greedy        -57.12
```

## Compare deployments in parallel

Give every copy the same seed so that the users move identically, then evaluate a different deployment in each copy:

```python
import numpy as np
from codesign_gym import make_vec

envs, base = make_vec("Network-Urban", 4)


def deployment(positions, tx):
    stations = base["stations"].copy()
    stations[: len(positions), :2] = positions
    stations[: len(positions), 2] = tx
    return {"num_bs": len(positions), "stations": stations}


grid6 = [(x, y) for x in (25, 75, 125) for y in (40, 110)]
ring8 = [(75 + 55 * np.cos(a), 75 + 55 * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 8, endpoint=False)]
designs = {
    "base (3 @ 30 dBm)": base,
    "3 @ 33 dBm": deployment(base["stations"][:3, :2], 33.0),
    "grid (6 @ 33 dBm)": deployment(grid6, 33.0),
    "ring (8 @ 30 dBm)": deployment(ring8, 30.0),
}
print(envs.validate_embodiment(list(designs.values())))

obs, rejected, info = envs.reset(seed=[0] * 4, embodiment=list(designs.values()))
returns = np.zeros(4)
for _ in range(100):
    obs, reward, terminated, truncated, info = envs.step(greedy(obs, info["action_mask"]))
    returns += reward
for name, value in zip(designs, returns):
    print(f"{name:18s} {value:7.2f}")
envs.close()
```

```text
[ True  True  True  True]
base (3 @ 30 dBm)   -44.20
3 @ 33 dBm          -27.18
grid (6 @ 33 dBm)     5.67
ring (8 @ 30 dBm)     2.21
```

With the same greedy policy, spreading more stations across the map beats raising the power of the base three, and all four designs stay within the 12.77 W budget.

## Change one copy's design

Stations cannot move during an episode, so a new design goes in through `reset`. A partial reset installs it in one copy and leaves the others running:

```python
import numpy as np
from codesign_gym import make_vec

envs, base = make_vec("Network-Urban", 2)
obs, rejected, info = envs.reset(seed=0)
for _ in range(10):
    obs, reward, terminated, truncated, info = envs.step(greedy(obs, info["action_mask"]))

obs, rejected, info = envs.reset(
    embodiment=[deployment(grid6, 33.0), None],
    options={"reset_mask": np.array([True, False])},
)
print(rejected, info["num_bs"])
envs.close()
```

```text
[False False] [6 3]
```
