# Examples

Printed distances below depend slightly on the device: on CPU, JAX's numerics give values within about 1 m of those shown (crash counts are unchanged).

## A lidar driver

This controller steers towards the longest lidar beam while holding the centre line, and sets its target speed from the free distance straight ahead.
It works on a whole batch of observations, so every example below runs it with `make_vec`. The action is a steering *velocity*, so the controller turns a target steering angle into a rate.

A single run says little about a driver this simple: whether it survives depends on where on the track it starts. So each example runs several start points at once (one copy per seed) and reports the mean distance and the number of crashes.

```python
import numpy as np
from codesign_gym import make_vec

BEAM_ANGLES = np.linspace(-2.35, 2.35, 32)          # right to left, radians from the heading
SEEDS = list(range(8))                               # one start point per seed


def drive(obs):
    """Follow the widest gap in the lidar, hold the centre line, and slow down for walls ahead."""
    obs = np.atleast_2d(obs)
    ey, epsi, steer_angle, speed, scan = obs[:, 1], obs[:, 2], obs[:, 3], obs[:, 4], obs[:, 7:]
    gap = BEAM_ANGLES[np.argmax(scan, axis=1)]
    target_angle = np.clip(-ey - epsi + 0.3 * gap, -0.42, 0.42)
    target_speed = np.minimum(10.0, 2.0 * scan[:, 14:18].min(axis=1))
    action = np.stack([10.0 * (target_angle - steer_angle), 5.0 * (target_speed - speed)], axis=1)
    return action.astype(np.float32)


def run(envs, seeds, designs, change=None):
    """One episode per copy. ``change(t)`` may return designs to pass to step t. Returns (distance, crashed) per copy."""
    obs, rejected, info = envs.reset(seed=seeds, embodiment=designs)
    returns, done, crashed = np.zeros(envs.num_envs), np.zeros(envs.num_envs, bool), np.zeros(envs.num_envs, bool)
    t = 0
    while not done.all():
        obs, reward, terminated, truncated, info = envs.step(drive(obs), embodiment=change(t) if change else None)
        returns += np.where(done, 0.0, reward)          # ignore copies that already finished
        crashed |= terminated & ~done
        done |= terminated | truncated
        t += 1
    return returns, crashed


envs, base = make_vec("Racing-Spielberg", len(SEEDS))
distance, crashed = run(envs, SEEDS, base)
print(np.round(distance, 1), f"mean {distance.mean():.1f} m, crashes {crashed.sum()}/{len(SEEDS)}")
envs.close()
```

```text
[95.6 91.2 23.8 95.7 28.7 15.6 94.2 91.9] mean 67.1 m, crashes 3/8
```

From five start points the stock car drives the full 1500 steps (about 92–96 m). From the other three it hits a wall early.

## Compare setups

Run four setups from the same eight start points. Copy `4·s + k` runs setup `k` from seed `s`, so within a seed only the design differs:

```python
designs = np.array([
    base,                                   # stock car
    [0.5, 4.0, 4.0, 35.57, 0.5192],         # low grip, soft tyres
    [1.1, 6.0, 6.0, 35.57, 0.5192],         # high grip, stiff tyres (heavier)
    [1.1, 6.0, 6.0, 55.0, 0.45],            # everything maxed, weight forward
], dtype=np.float32)
envs, _ = make_vec("Racing-Spielberg", 4 * len(SEEDS))
distance, crashed = run(envs, [s for s in SEEDS for _ in designs], np.tile(designs, (len(SEEDS), 1)))
print("mean distance", np.round(distance.reshape(-1, 4).mean(axis=0), 1))
print("crashes      ", crashed.reshape(-1, 4).sum(axis=0))
envs.close()
```

```text
mean distance [66.5 12.7 58.3 62.6]
crashes       [4 8 3 4]
```

With this driver the stock setup goes furthest on average, and the fully upgraded car is second. Its upgrades add 1.2 kg, so its effective grip (1.065) is below its tyres' 1.1. The low-grip car crashes from every start point.
(The stock car crashes here from one start point more than in the first example. Each step draws its noise from a key split across the whole batch, so the same seed can play out slightly differently in a batch of a different size.)

## Change the setup mid-episode

Racing accepts a new design at any `step`, and the car keeps its position and speed. Here every start point runs twice: a dry copy, and a copy whose track turns wet after 3 s (grip drops to 0.5). The driver does not adapt:

```python
wet = base.copy()
wet[0] = 0.5
envs, _ = make_vec("Racing-Spielberg", 2 * len(SEEDS))
to_wet = [None, wet] * len(SEEDS)                    # None keeps a copy's current design
distance, crashed = run(envs, [s for s in SEEDS for _ in range(2)], base, change=lambda t: to_wet if t == 300 else None)
print("mean distance (dry, wet)", np.round(distance.reshape(-1, 2).mean(axis=0), 1))
print("crashes       (dry, wet)", crashed.reshape(-1, 2).sum(axis=0))
envs.close()
```

```text
mean distance (dry, wet) [66.5 46.5]
crashes       (dry, wet) [3 7]
```

The wet track crashes the car from 7 of the 8 start points, against 3 when it stays dry.

An evaluation env (`is_eval_env=True`) ignores a design passed to `step`, with a warning.

## Watch a run

With `render_mode="rgb_array"`, `render()` returns a frame per copy with the design and the derived mass, grip and `a_max` written on it. With `render_mode="web"`, the env records the run and `save_dashboard` writes f1tenth's HTML viewer:

```python
import tempfile
from pathlib import Path
from codesign_gym import make

env, base = make("Racing-Levine", render_mode="rgb_array")
obs, rejected, info = env.reset(seed=0)
for _ in range(200):
    obs, reward, terminated, truncated, info = env.step(drive(obs)[0])
frame = env.render()
print(frame.shape, frame.dtype)                      # (420, 420, 3) uint8
env.close()

env, base = make("Racing-Levine", render_mode="web")
obs, rejected, info = env.reset(seed=0)
for _ in range(200):
    obs, reward, terminated, truncated, info = env.step(drive(obs)[0])
path = env.save_dashboard(str(Path(tempfile.mkdtemp()) / "levine.html"))
env.close()
```
