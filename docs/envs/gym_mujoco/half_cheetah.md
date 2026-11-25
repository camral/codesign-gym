# HalfCheetah

```{include} ../../_generated/readmes/half_cheetah.md
```

## Design slots

16 slots, `max_offset = 0.148` m. Axes are in each link's body frame, which at the rest pose lines up with the world:
`x` points forward and `z` up. The torso is split at its centre into a rear and a front half, each a one-axis link, so
slots 2 and 9 move the hips backward and forward.

| Slot | Link | Body | Axis | `+1` moves the far end | Far end at 0 (m) |
|---|---|---|---|---|---|
| 0 | Head | `head_offsettable_xz` | x | forward | (0.229, 0, 0.193) |
| 1 | Head | `head_offsettable_xz` | z | up | |
| 2 | Rear torso half | `torso_rear_offsettable_x` | x | backward (longer) | (−0.5, 0, 0) |
| 3 | Back thigh | `bthigh_offsettable_xz` | x | forward | (0.16, 0, −0.25) |
| 4 | Back thigh | `bthigh_offsettable_xz` | z | down | |
| 5 | Back shin | `bshin_offsettable_xz` | x | backward | (−0.28, 0, −0.14) |
| 6 | Back shin | `bshin_offsettable_xz` | z | down | |
| 7 | Back foot | `bfoot_offsettable_xz` | x | forward | (0.05, 0, −0.181) |
| 8 | Back foot | `bfoot_offsettable_xz` | z | down | |
| 9 | Front torso half | `torso_front_offsettable_x` | x | forward (longer) | (0.5, 0, 0) |
| 10 | Front thigh | `fthigh_offsettable_xz` | x | backward | (−0.14, 0, −0.24) |
| 11 | Front thigh | `fthigh_offsettable_xz` | z | down | |
| 12 | Front shin | `fshin_offsettable_xz` | x | forward | (0.13, 0, −0.18) |
| 13 | Front shin | `fshin_offsettable_xz` | z | down | |
| 14 | Front foot | `ffoot_offsettable_xz` | x | forward | (0.079, 0, −0.116) |
| 15 | Front foot | `ffoot_offsettable_xz` | z | down | |

The far end is given relative to the link's joint. The back foot, front shin and front foot can reach the length floor
(0.092 m).

## Observations and actions

The observation is `Box(17,)`, the v4 layout: joint positions without the torso's x position, then all velocities.

| Index | Content |
|---|---|
| 0 | Torso height (`rootz`) |
| 1 | Torso pitch angle (`rooty`) |
| 2–7 | Joint angles: `bthigh`, `bshin`, `bfoot`, `fthigh`, `fshin`, `ffoot` |
| 8–10 | Torso x velocity, z velocity, pitch rate |
| 11–16 | Joint angular velocities, same order as 2–7 |

The action is `Box(6,)` in `[-1, 1]`: one torque per joint in the order above, with gears 120, 90, 60, 120, 60 and 30.

## Reward and termination

```text
reward = forward_velocity − 0.1 · ‖action‖²
```

`forward_velocity` is the torso's x displacement over the control step (0.05 s), divided by that step. There is no
healthy termination: an episode ends only by truncation at 1000 steps, or if the copy's state stops being finite (then
`terminated` is `True` and the reward 0). `info` holds the v4 keys `x_position`, `x_velocity`, `reward_run` and
`reward_ctrl`.

## Example: longer thighs

Run the base cheetah and one with both thighs 0.148 m longer side by side, under the same open-loop gait:

```python
import numpy as np
from codesign_gym import make_vec

envs, base = make_vec("HalfCheetah-v4", 2, device="cpu")
long_thighs = base.copy()
long_thighs[[4, 11]] = 1.0                  # back and front thigh, far end straight down
obs, rejected, info = envs.reset(seed=0, embodiment=np.stack([base, long_thighs]))

phase = np.array([0.0, 1.0, 2.0, np.pi, np.pi + 1.0, np.pi + 2.0])
for t in range(100):
    action = np.sin(0.5 * t + phase).astype(np.float32)
    obs, reward, terminated, truncated, info = envs.step(np.stack([action, action]))
for name, x in zip(["base", "long thighs"], info["x_position"].tolist()):
    print(f"{name:12s} x = {x:+.2f} m")
envs.close()
```
