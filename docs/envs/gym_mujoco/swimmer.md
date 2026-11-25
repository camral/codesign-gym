# Swimmer

```{include} ../../_generated/readmes/swimmer.md
```

## Design slots

6 slots, `max_offset = 0.5` m. The swimmer lies in the horizontal plane: in each link's body frame `x` runs along the
body (positive towards the nose at the rest pose) and `y` sideways within the swimming plane.

| Slot | Link | Body | Axis | `+1` moves the far end | Far end at 0 (m) |
|---|---|---|---|---|---|
| 0 | Front (nose) | `front_offsettable_xy` | x | forward (longer) | (1, 0, 0) |
| 1 | Front (nose) | `front_offsettable_xy` | y | sideways (+y) | |
| 2 | Middle | `mid_offsettable_xy` | x | backward (longer) | (−1, 0, 0) |
| 3 | Middle | `mid_offsettable_xy` | y | sideways (+y) | |
| 4 | Back (tail) | `back_offsettable_xy` | x | backward (longer) | (−1, 0, 0) |
| 5 | Back (tail) | `back_offsettable_xy` | y | sideways (+y) | |

All three links are 1 m long at the base design, and the stretch reaches 0.5–1.5 m along the body. The first joint
sits where the front and middle links meet; moving the middle link's far end moves the second joint and the tail with it.

## Observations and actions

The observation is `Box(8,)`, the v4 layout: positions without the x and y of the front tip, then all velocities.

| Index | Content |
|---|---|
| 0 | Heading of the front link (`free_body_rot`) |
| 1–2 | Joint angles (`motor1_rot`, `motor2_rot`) |
| 3–4 | x and y velocity |
| 5 | Heading rate |
| 6–7 | Joint angular velocities |

The action is `Box(2,)` in `[-1, 1]`: the torques on the two joints, gear 150.

## Reward and termination

```text
reward = forward_velocity − 1e-4 · ‖action‖²
```

`forward_velocity` is the x displacement over the control step (0.04 s), divided by that step. There is no early
termination: episodes are truncated at 1000 steps (or terminated, with reward 0, if the state stops being finite).
`info` holds the v4 keys `reward_fwd`, `reward_ctrl`, `x_position`, `y_position`, `distance_from_origin`,
`x_velocity`, `y_velocity` and `forward_reward`.

## Example: a longer tail

Swim the base body and one with a 1.5 m tail with the same travelling wave:

```python
import numpy as np
from codesign_gym import make_vec

envs, base = make_vec("Swimmer-v4", 2, device="cpu")
long_tail = base.copy()
long_tail[4] = 1.0                          # tail 0.5 m longer
obs, rejected, info = envs.reset(seed=0, embodiment=np.stack([base, long_tail]))

for t in range(200):
    wave = np.array([np.sin(0.3 * t), np.sin(0.3 * t + 1.5)], dtype=np.float32)
    obs, reward, terminated, truncated, info = envs.step(np.stack([wave, wave]))
for name, x in zip(["base", "long tail"], info["x_position"].tolist()):
    print(f"{name:10s} x = {x:+.3f} m")
envs.close()
```
