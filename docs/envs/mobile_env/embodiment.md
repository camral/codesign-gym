# Design space

The design is the base-station deployment: how many stations to build, where, and how strongly each one transmits. It is a `dict` with two keys:

| Key | Space | Meaning |
|---|---|---|
| `num_bs` | `Discrete(max_bs + 1)`, int | Number of stations built; must be in `[min_bs, max_bs]` |
| `stations` | `Box((max_bs, 3), float32)` | Row `i` is station `i`: `(x, y, tx)` |

Only the first `num_bs` rows are built. The remaining rows are padding and are ignored, but keep them inside the box (the base design fills them with `(0, 0, tx_min)`).
Positions are in mobile-env's map units and `tx` is the transmit power in dBm.

The ids differ in map, users and budget:

| Id | Map | Users | Speed | Carrier | `num_bs` | `tx` (dBm) | Power budget |
|---|---|---|---|---|---|---|---|
| `Network-Urban` | 150 × 150 | 15 | 1.5 m/s | 3.5 GHz, 20 MHz | 3 – 8 | 23 – 33 | 12.77 W |
| `Network-Suburban` | 200 × 300 | 20 | 8 m/s | 2.1 GHz, 20 MHz | 3 – 7 | 33 – 43 | 111.73 W |
| `Network-Motorway` | 400 × 400 | 10 | 33 m/s | 700 MHz, 10 MHz | 1 – 4 | 40 – 46 | 127.39 W |

The power budget caps the summed transmit power of the built stations, in watts (`W = 10^(dBm/10) / 1000`). It is 80% of `max_bs` stations at full power, so the whole fleet can never run at `tx_max` together.
For example, in `Network-Urban` six stations at 33 dBm (2.0 W each) fit, but seven do not.

## Base design

The base design is a reference deployment taken from mobile-env's own scenarios: 3 stations at 30 dBm in Urban, 7 at 38 dBm in Suburban and 3 at 43 dBm in Motorway.

```python
from codesign_gym import make

env, base = make("Network-Urban")
print(base["num_bs"])
print(base["stations"])
env.close()
```

```text
3
[[110. 130.  30.]
 [ 65.  80.  30.]
 [120.  30.  30.]
 [  0.   0.  23.]
 [  0.   0.  23.]
 [  0.   0.  23.]
 [  0.   0.  23.]
 [  0.   0.  23.]]
```

## Build a design

Fill a `(max_bs, 3)` array, starting from the base design so that the padding rows stay valid:

```python
import numpy as np
from codesign_gym import make

env, base = make("Network-Urban")
stations = base["stations"].copy()
grid = [(x, y) for x in (25, 75, 125) for y in (40, 110)]   # 6 stations on a 3 x 2 grid
stations[:6, :2] = grid
stations[:6, 2] = 33.0                                     # full power
print(env.validate_embodiment({"num_bs": 6, "stations": stations}))     # True: 6 × 2.0 W ≤ 12.77 W

stations[6] = (75, 75, 33.0)
print(env.validate_embodiment({"num_bs": 7, "stations": stations}))     # False: 7 × 2.0 W > 12.77 W

stations[:7, 2] = 31.0
print(env.validate_embodiment({"num_bs": 7, "stations": stations}))     # True: 7 × 1.26 W
env.close()
```

## What is rejected

`validate_embodiment` returns `False`, and `reset` returns `rejected=True`, for a design with:

- keys other than `num_bs` and `stations`,
- `num_bs` outside `[min_bs, max_bs]`,
- `stations` not of shape `(max_bs, 3)`, or containing NaN or infinity,
- a built station outside the map or outside `[tx_min, tx_max]`,
- a total transmit power above the budget,
- two built stations closer than `min_separation` (off by default).

Both also check the design against the `embodiment_space`, so they also reject a non-integer `num_bs` (such as 3.7) and padding rows outside the box.

A rejected reset does not raise. Its observation is all zeros and its `info` is empty, and `step` raises `RuntimeError` until a reset succeeds.
A design passed to `step` is never applied: stations are fixed for the episode, so it is dropped with a `UserWarning` ("Transition-level embodiment changes are not permitted"), even if it is invalid. A new design takes effect at the next `reset`.

With `make_vec`, `validate_embodiment` returns a `(num_envs,)` bool array and never raises. It takes a single dict (applied to every copy), a list of dicts (with `None` for "no design"), or a stacked dict whose values have a leading `num_envs` axis (a leading axis of 1 is applied to every copy). Wrong keys or a batch of the wrong size give `False`.

## Build options

`make` and `make_vec` forward keyword arguments as overrides of the preset's `MobileEnvConfig`. Useful ones:

| Kwarg | Effect |
|---|---|
| `num_ues` | Number of users (changes the observation and action sizes) |
| `min_bs`, `max_bs` | Station budget (`max_bs` changes the observation, action and design sizes) |
| `tx_min`, `tx_max` | Transmit power range, dBm |
| `tx_budget_fraction` | Power budget as a fraction of `max_bs` stations at `tx_max` (default 0.8; 0 disables it) |
| `min_separation` | Minimum distance between built stations (default 0, off) |
| `velocity` | User speed, m/s |
| `width`, `height` | Map size |
| `ep_max_time` | Episode length (default 100) |
| `seed_positions`, `seed_tx` | The base design's station positions and power |

```python
from codesign_gym import make

env, base = make("Network-Urban", num_ues=5, min_separation=20.0)
print(env.observation_space.shape, env.action_space)
env.close()
```

```text
(85,) MultiDiscrete([9 9 9 9 9])
```
