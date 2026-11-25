# Ant

```{include} ../../_generated/readmes/ant.md
```

## Design slots

36 slots, `max_offset = 0.141` m. Each leg has three links — a hip link fixed to the torso, an upper leg after the hip
joint and a lower leg after the ankle joint — and each link takes three slots, one per body-frame axis. At the rest
pose every body frame lines up with the torso's: `x` points forward (the rewarded direction), `y` to the left and `z`
up. Legs are named by the stock body names and placed by the corner they point to.

| Leg | Link | Body | Far end at 0 (m) | x slot | y slot | z slot |
|---|---|---|---|---|---|---|
| 1 (+x, +y) | Hip | `front_left_leg_offsettable_xyz` | (0.2, 0.2, 0) | 0 (+x) | 1 (+y) | 2 (+z) |
| 1 (+x, +y) | Upper leg (`hip_1`) | `aux_1_offsettable_xyz` | (0.2, 0.2, 0) | 3 (+x) | 4 (+y) | 5 (+z) |
| 1 (+x, +y) | Lower leg (`ankle_1`) | `left_ankle_offsettable_xyz` | (0.4, 0.4, 0) | 6 (+x) | 7 (+y) | 8 (+z) |
| 2 (−x, +y) | Hip | `front_right_leg_offsettable_xyz` | (−0.2, 0.2, 0) | 9 (−x) | 10 (+y) | 11 (+z) |
| 2 (−x, +y) | Upper leg (`hip_2`) | `aux_2_offsettable_xyz` | (−0.2, 0.2, 0) | 12 (−x) | 13 (+y) | 14 (+z) |
| 2 (−x, +y) | Lower leg (`ankle_2`) | `right_ankle_offsettable_xyz` | (−0.4, 0.4, 0) | 15 (−x) | 16 (+y) | 17 (+z) |
| 3 (−x, −y) | Hip | `back_leg_offsettable_xyz` | (−0.2, −0.2, 0) | 18 (−x) | 19 (−y) | 20 (+z) |
| 3 (−x, −y) | Upper leg (`hip_3`) | `aux_3_offsettable_xyz` | (−0.2, −0.2, 0) | 21 (−x) | 22 (−y) | 23 (+z) |
| 3 (−x, −y) | Lower leg (`ankle_3`) | `third_ankle_offsettable_xyz` | (−0.4, −0.4, 0) | 24 (−x) | 25 (−y) | 26 (+z) |
| 4 (+x, −y) | Hip | `right_back_leg_offsettable_xyz` | (0.2, −0.2, 0) | 27 (+x) | 28 (−y) | 29 (+z) |
| 4 (+x, −y) | Upper leg (`hip_4`) | `aux_4_offsettable_xyz` | (0.2, −0.2, 0) | 30 (+x) | 31 (−y) | 32 (+z) |
| 4 (+x, −y) | Lower leg (`ankle_4`) | `fourth_ankle_offsettable_xyz` | (0.4, −0.4, 0) | 33 (+x) | 34 (−y) | 35 (+z) |

The sign in brackets is the direction `+1` moves the far end along that axis: in `x` and `y` it always points away from
the torso, so `+1` on both lengthens the link along its diagonal, while `z` moves the far end up. The hip links and upper
legs can reach the length floor (0.16 m).

## Observations and actions

The observation is `Box(27,)`, the v4 default layout (`use_contact_forces=False`): positions without the torso's x and
y, then all velocities.

| Index | Content |
|---|---|
| 0 | Torso height |
| 1–4 | Torso orientation quaternion (w, x, y, z) |
| 5–12 | Joint angles: `hip_1`, `ankle_1`, `hip_2`, `ankle_2`, `hip_3`, `ankle_3`, `hip_4`, `ankle_4` |
| 13–15 | Torso linear velocity (x, y, z) |
| 16–18 | Torso angular velocity |
| 19–26 | Joint angular velocities, same order as 5–12 |

The action is `Box(8,)` in `[-1, 1]`, in the stock actuator order `hip_4`, `ankle_4`, `hip_1`, `ankle_1`, `hip_2`,
`ankle_2`, `hip_3`, `ankle_3` (gear 150 each). Note that this differs from the joint order in the observation.

## Reward and termination

```text
reward = forward_velocity + 1 − 0.5 · ‖action‖²
```

`forward_velocity` is the torso's x displacement over the control step (0.05 s), divided by that step. There is no
contact cost, as in v4's default. The healthy bonus of 1 is paid on every step, including the one that terminates.

The episode terminates when the torso height leaves 0.2–1.0 m (the stock range, for every design), and is otherwise
truncated at 1000 steps. `info` holds the v4 keys `reward_forward`, `reward_ctrl`, `reward_survive`, `x_position`,
`y_position`, `distance_from_origin`, `x_velocity`, `y_velocity` and `forward_reward`.

## Example: longer lower legs

Lengthen all four lower legs along their diagonal and compare the torso height and return with the base ant under the
same random actions:

```python
import numpy as np
from codesign_gym import make_vec

envs, base = make_vec("Ant-v4", 2, device="cpu")
long_shins = base.copy()
long_shins[[6, 7, 15, 16, 24, 25, 33, 34]] = 1.0     # x and y slot of every lower leg
obs, rejected, info = envs.reset(seed=0, embodiment=np.stack([base, long_shins]))

rng = np.random.default_rng(0)
returns = np.zeros(2)
for _ in range(50):
    action = rng.uniform(-1, 1, size=8).astype(np.float32)
    obs, reward, terminated, truncated, info = envs.step(np.stack([action, action]))
    returns += reward.cpu().numpy()
print("torso height", [round(z, 3) for z in obs[:, 0].tolist()])
print("return", returns.round(1).tolist())
envs.close()
```
