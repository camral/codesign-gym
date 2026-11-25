# Hopper

```{include} ../../_generated/readmes/hopper.md
```

## Design slots

8 slots, `max_offset = 0.115` m. Axes are in each link's body frame, which at the rest pose lines up with the world:
`x` points forward and `z` up. The stock torso capsule is split at its centre into an upper and a lower half, and the
stock foot at the ankle into a toe and a heel; each of these four is a one-axis link that only changes length.

| Slot | Link | Body | Axis | `+1` moves the far end | Far end at 0 (m) |
|---|---|---|---|---|---|
| 0 | Upper torso | `torso_upper_offsettable_z` | z | up (longer) | (0, 0, 0.2) |
| 1 | Lower torso | `torso_lower_offsettable_z` | z | down (longer, lowers the hip) | (0, 0, −0.2) |
| 2 | Thigh | `thigh_offsettable_xz` | x | forward | (0, 0, −0.45) |
| 3 | Thigh | `thigh_offsettable_xz` | z | down (longer) | |
| 4 | Leg (shin) | `leg_offsettable_xz` | x | forward | (0, 0, −0.5) |
| 5 | Leg (shin) | `leg_offsettable_xz` | z | down (longer) | |
| 6 | Toe | `foot_toe_offsettable_x` | x | forward (longer) | (0.26, 0, 0) |
| 7 | Heel | `foot_heel_offsettable_x` | x | backward (longer) | (−0.13, 0, 0) |

The torso halves and the heel can reach the length floor (0.12 m).

## Observations and actions

The observation is `Box(11,)`, the v4 layout: positions without the torso's x position, then velocities clipped to
`[-10, 10]`.

| Index | Content |
|---|---|
| 0 | Torso height (`rootz`) |
| 1 | Torso pitch angle (`rooty`) |
| 2–4 | Joint angles: thigh, leg, foot |
| 5–7 | Torso x velocity, z velocity, pitch rate |
| 8–10 | Joint angular velocities: thigh, leg, foot |

The action is `Box(3,)` in `[-1, 1]`: the thigh, leg and foot torques, gear 200 each.

## Reward and termination

```text
reward = forward_velocity + 1 − 1e-3 · ‖action‖²
```

`forward_velocity` is the torso's x displacement over the control step (0.008 s), divided by that step. The healthy
bonus of 1 is paid on every step, including the one that terminates, as in v4.

The episode terminates when the hopper is no longer healthy: torso height at most 0.7 m, torso angle at least 0.2 rad
in magnitude, or any entry of the state after the x and z positions reaching 100 in magnitude. These limits are the
stock ones for every design. Otherwise it is truncated at 1000 steps. `info` holds `x_position` and `x_velocity`.

## Example: longer legs

Let the base hopper and one with a 0.115 m longer thigh and shin stand passively, and count the steps until each falls:

```python
import numpy as np
from codesign_gym import make_vec

envs, base = make_vec("Hopper-v4", 2, device="cpu")
long_legs = base.copy()
long_legs[[3, 5]] = 1.0                     # thigh and leg, far ends straight down
obs, rejected, info = envs.reset(seed=0, embodiment=np.stack([base, long_legs]))
print("start height", obs[:, 0].tolist())   # the longer legs start the torso higher

fell_at = [None, None]
for t in range(1, 401):
    obs, reward, terminated, truncated, info = envs.step(np.zeros((2, 3), dtype=np.float32))
    for i in range(2):
        if fell_at[i] is None and bool(terminated[i]):
            fell_at[i] = t
    if None not in fell_at:
        break
print("fell after", fell_at)
envs.close()
```
