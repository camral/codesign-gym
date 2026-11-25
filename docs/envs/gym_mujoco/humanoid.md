# Humanoid

```{include} ../../_generated/readmes/humanoid.md
```

## Design slots

28 slots, `max_offset = 0.144` m. Slots 0–23 move the far ends of the eight limb links, three axes each; slots 24–27
scale the radius of the feet and hands. At the rest pose every body frame lines up with the torso's: `x` points
forward, `y` to the humanoid's left and `z` up. The torso, waist and pelvis are fixed.

| Link | Body | Far end at 0 (m) | x slot | y slot | z slot |
|---|---|---|---|---|---|
| Right thigh | `right_thigh_offsettable_xyz` | (0, 0.01, −0.34) | 0 (+x) | 1 (+y) | 2 (−z, longer) |
| Right shin | `right_shin_offsettable_xyz` | (0, 0, −0.3) | 3 (+x) | 4 (+y) | 5 (−z, longer) |
| Left thigh | `left_thigh_offsettable_xyz` | (0, −0.01, −0.34) | 6 (+x) | 7 (−y) | 8 (−z, longer) |
| Left shin | `left_shin_offsettable_xyz` | (0, 0, −0.3) | 9 (+x) | 10 (+y) | 11 (−z, longer) |
| Right upper arm | `right_upper_arm_offsettable_xyz` | (0.16, −0.16, −0.16) | 12 (+x) | 13 (−y) | 14 (−z) |
| Right forearm | `right_lower_arm_offsettable_xyz` | (0.16, 0.16, 0.16) | 15 (+x) | 16 (+y) | 17 (+z) |
| Left upper arm | `left_upper_arm_offsettable_xyz` | (0.16, 0.16, −0.16) | 18 (+x) | 19 (+y) | 20 (−z) |
| Left forearm | `left_lower_arm_offsettable_xyz` | (0.16, −0.16, 0.16) | 21 (+x) | 22 (−y) | 23 (+z) |

The sign in brackets is the direction `+1` moves the far end along that axis. On the legs, the `z` slot is the length
and `x` swings the knee or ankle forward. On the arms all three axes point along the link's diagonal, so `+1` on all
three lengthens it without turning it. A forearm's hand rides its far end, so a longer forearm moves the hand outward.
The arm links can reach the length floor (0.12 m).

| Slot | Sphere | Radius at 0 (m) |
|---|---|---|
| 24 | Right foot | 0.075 |
| 25 | Left foot | 0.075 |
| 26 | Right hand | 0.04 |
| 27 | Left hand | 0.04 |

A radius slot `o` sets the radius to `r0 · (1 + 0.5 · o)`, from half to 1.5 times the stock size; the mass follows.

## Observations and actions

The observation is `Box(376,)`, the v4 layout. `nbody = 14` counts the world body.

| Index | Content |
|---|---|
| 0 | Torso height |
| 1–4 | Torso orientation quaternion (w, x, y, z) |
| 5–21 | 17 joint angles, in model order: `abdomen_z`, `abdomen_y`, `abdomen_x`, right hip x/z/y, `right_knee`, left hip x/z/y, `left_knee`, `right_shoulder1`, `right_shoulder2`, `right_elbow`, then the left arm |
| 22–44 | `qvel`: torso linear and angular velocity, then the 17 joint velocities |
| 45–184 | `cinert`: body inertias, 14 × 10 |
| 185–268 | `cvel`: body velocities, 14 × 6 |
| 269–291 | `qfrc_actuator`: actuator forces per degree of freedom, 23 |
| 292–375 | `cfrc_ext`: external contact forces on the bodies, 14 × 6 |

The `cinert` block changes with the design, since it holds the bodies' mass and inertia.

The action is `Box(17,)` in `[-1, 1]`, in the stock actuator order (`abdomen_y`, `abdomen_z`, `abdomen_x`, right hip x/z/y,
`right_knee`, left hip x/z/y, `left_knee`, right shoulders and elbow, left shoulders and elbow). It is mapped linearly
onto the stock control range `[-0.4, 0.4]`, so `ctrl = 0.4 · action`. Stock Gymnasium takes `ctrl` directly as the action;
to replay a stock action here, divide it by 0.4.

## Reward and termination

```text
reward = 1.25 · com_forward_velocity + 5 − 0.1 · ‖ctrl‖²
```

`com_forward_velocity` is the x displacement of the centre of mass over the control step (0.015 s), divided by that
step. The control cost is on the applied `ctrl`, as in v4. The healthy bonus of 5 is paid on every step, including the
one that terminates.

The episode terminates when the torso height leaves the open range 1.0–2.0 m (the stock range, for every design), and is
otherwise truncated at 1000 steps. `info` holds the v4 keys `reward_linvel`, `reward_quadctrl`, `reward_alive`,
`x_position`, `y_position`, `distance_from_origin`, `x_velocity`, `y_velocity` and `forward_reward`; the positions and
velocities are those of the centre of mass.

## Example: longer shins and bigger feet

Let the base humanoid and one with 0.144 m longer shins and 1.5× larger feet stand passively, and count the steps until
each falls:

```python
import numpy as np
from codesign_gym import make_vec

envs, base = make_vec("Humanoid-v4", 2, device="cpu")
sturdy = base.copy()
sturdy[[5, 11]] = 1.0                       # both shins longer
sturdy[[24, 25]] = 1.0                      # both feet 1.5x the radius
obs, rejected, info = envs.reset(seed=0, embodiment=np.stack([base, sturdy]))
print("start height", [round(z, 3) for z in obs[:, 0].tolist()])

fell_at = [None, None]
for t in range(1, 301):
    obs, reward, terminated, truncated, info = envs.step(np.zeros((2, 17), dtype=np.float32))
    for i in range(2):
        if fell_at[i] is None and bool(terminated[i]):
            fell_at[i] = t
    if None not in fell_at:
        break
print("fell after", fell_at)
envs.close()
```
