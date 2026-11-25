# Walker2d

```{include} ../../_generated/readmes/walker2d.md
```

## Design slots

12 slots, `max_offset = 0.163` m. Axes are in each link's body frame, which at the rest pose lines up with the world:
`x` points forward and `z` up. The stock torso capsule is split at its centre into an upper and a lower half, each a
one-axis link, and each foot is a one-axis link running forward from the ankle. The right leg (stock names `thigh`,
`leg`, `foot`) and the left leg (`*_left`) are independent.

| Slot | Link | Body | Axis | `+1` moves the far end | Far end at 0 (m) |
|---|---|---|---|---|---|
| 0 | Upper torso | `torso_upper_offsettable_z` | z | up (longer) | (0, 0, 0.2) |
| 1 | Lower torso | `torso_lower_offsettable_z` | z | down (longer, lowers the hips) | (0, 0, −0.2) |
| 2 | Right thigh | `thigh_offsettable_xz` | x | forward | (0, 0, −0.45) |
| 3 | Right thigh | `thigh_offsettable_xz` | z | down (longer) | |
| 4 | Right leg (shin) | `leg_offsettable_xz` | x | forward | (0, 0, −0.5) |
| 5 | Right leg (shin) | `leg_offsettable_xz` | z | down (longer) | |
| 6 | Right foot | `foot_toe_offsettable_x` | x | forward (longer) | (0.2, 0, 0) |
| 7 | Left thigh | `thigh_left_offsettable_xz` | x | forward | (0, 0, −0.45) |
| 8 | Left thigh | `thigh_left_offsettable_xz` | z | down (longer) | |
| 9 | Left leg (shin) | `leg_left_offsettable_xz` | x | forward | (0, 0, −0.5) |
| 10 | Left leg (shin) | `leg_left_offsettable_xz` | z | down (longer) | |
| 11 | Left foot | `foot_left_toe_offsettable_x` | x | forward (longer) | (0.2, 0, 0) |

The torso halves and both feet can reach the length floor (0.12 m).

## Observations and actions

The observation is `Box(17,)`, the v4 layout: positions without the torso's x position, then velocities clipped to
`[-10, 10]`.

| Index | Content |
|---|---|
| 0 | Torso height (`rootz`) |
| 1 | Torso pitch angle (`rooty`) |
| 2–4 | Right joint angles: thigh, leg, foot |
| 5–7 | Left joint angles: thigh, leg, foot |
| 8–10 | Torso x velocity, z velocity, pitch rate |
| 11–16 | Joint angular velocities, same order as 2–7 |

The action is `Box(6,)` in `[-1, 1]`: the torques on the right thigh, leg and foot, then the left ones, gear 100 each.

## Reward and termination

```text
reward = forward_velocity + 1 − 1e-3 · ‖action‖²
```

`forward_velocity` is the torso's x displacement over the control step (0.008 s), divided by that step. The healthy
bonus of 1 is paid on every step, including the one that terminates, as in v4.

The episode terminates when the torso height leaves the open range 0.8–2.0 m or the torso angle reaches 1 rad in
magnitude. These limits are the stock ones for every design. Otherwise it is truncated at 1000 steps. `info` holds
`x_position` and `x_velocity`.

## Faster CPU physics

On the CPU, `make_vec("Walker2d-v4", n, device="cpu", physics="native")` steps the physics with C MuJoCo
(`mujoco.rollout`, one thread per core) instead of MuJoCo Warp's CPU backend. Resets still run the Warp kernels, so
reset observations and designs are identical to the default path. Steps run in float64 C MuJoCo rather than float32
Warp: they agree with the default path to about 1e-3 after one step and then drift apart as any two engines do, while
episodes end on the same step and earn the same return (`tests/test_walker2d_native.py`). `physics="native"` needs
`device="cpu"`; the default is `physics="warp"`.

On an Apple M5 (10 cores), random actions with a respawn on almost every step, the full env ran at:

| copies | `physics="warp"` (steps/s) | `physics="native"` (steps/s) |
|---|---|---|
| 1 | 20 | 2,424 |
| 64 | 815 | 10,413 |
| 1024 | 3,044 | 22,193 |
| 4096 | 3,466 | 28,008 |

## Example: try a new design in one copy

Start two copies on the base design, then give copy 1 a longer right thigh and a shorter left one with a partial reset.
Copy 0 keeps its episode:

```python
import numpy as np
from codesign_gym import make_vec

envs, base = make_vec("Walker2d-v4", 2, device="cpu")
obs, rejected, info = envs.reset(seed=0)
for _ in range(20):
    obs, reward, terminated, truncated, info = envs.step(np.zeros((2, 6), dtype=np.float32))

uneven = base.copy()
uneven[3] = 1.0                             # right thigh 0.163 m longer
uneven[8] = -1.0                            # left thigh 0.163 m shorter
designs = np.stack([base, uneven])
obs, rejected, info = envs.reset(embodiment=designs, options={"reset_mask": np.array([False, True])})
print(info["embodiment"][:, [3, 8]].tolist())   # [[0.0, 0.0], [1.0, -1.0]]

for _ in range(20):
    obs, reward, terminated, truncated, info = envs.step(np.zeros((2, 6), dtype=np.float32))
print("torso height", [round(z, 3) for z in obs[:, 0].tolist()])
envs.close()
```
