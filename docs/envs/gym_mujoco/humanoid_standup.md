# HumanoidStandup

```{include} ../../_generated/readmes/humanoid_standup.md
```

## Design slots

28 slots, `max_offset = 0.144` m, on the same links and spheres as [Humanoid](humanoid.md): slots 0–23 move the far
ends of the eight limb links, slots 24–27 scale the feet and hands. The stock model, however, is built lying on its back,
so the leg frames differ from the Humanoid's. In the leg links' body frames at the rest pose, `x` runs along the body
from head to feet, `y` to the humanoid's left and `z` up, out of its chest. The arm links are built as in the Humanoid.

| Link | Body | Far end at 0 (m) | x slot | y slot | z slot |
|---|---|---|---|---|---|
| Right thigh | `right_thigh_offsettable_xyz` | (0.34, 0.01, 0) | 0 (+x, longer) | 1 (+y) | 2 (+z) |
| Right shin | `right_shin_offsettable_xyz` | (0.3, 0, 0) | 3 (+x, longer) | 4 (+y) | 5 (+z) |
| Left thigh | `left_thigh_offsettable_xyz` | (0.34, −0.01, 0) | 6 (+x, longer) | 7 (−y) | 8 (+z) |
| Left shin | `left_shin_offsettable_xyz` | (0.3, 0, 0) | 9 (+x, longer) | 10 (+y) | 11 (+z) |
| Right upper arm | `right_upper_arm_offsettable_xyz` | (0.16, −0.16, −0.16) | 12 (+x) | 13 (−y) | 14 (−z) |
| Right forearm | `right_lower_arm_offsettable_xyz` | (0.16, 0.16, 0.16) | 15 (+x) | 16 (+y) | 17 (+z) |
| Left upper arm | `left_upper_arm_offsettable_xyz` | (0.16, 0.16, −0.16) | 18 (+x) | 19 (+y) | 20 (−z) |
| Left forearm | `left_lower_arm_offsettable_xyz` | (0.16, −0.16, 0.16) | 21 (+x) | 22 (−y) | 23 (+z) |

So on the legs the **`x`** slot is the length here, where on the Humanoid it is the `z` slot. A design vector copied
from one id to the other builds a different body: for example, slot 2 lengthens the right thigh on the Humanoid but
swings its knee on HumanoidStandup.

| Slot | Sphere | Radius at 0 (m) |
|---|---|---|
| 24 | Right foot | 0.075 |
| 25 | Left foot | 0.075 |
| 26 | Right hand | 0.04 |
| 27 | Left hand | 0.04 |

A radius slot `o` sets the radius to `r0 · (1 + 0.5 · o)`. The arm links can reach the length floor (0.12 m).

## Observations and actions

The observation is `Box(376,)` with the same layout as [Humanoid](humanoid.md#observations-and-actions): torso height,
orientation, joint angles, `qvel`, then `cinert`, `cvel`, `qfrc_actuator` and `cfrc_ext`. The action is `Box(17,)` in
`[-1, 1]`, in the same actuator order, mapped onto the stock control range `[-0.4, 0.4]` (`ctrl = 0.4 · action`).

## Reward and termination

```text
reward = torso_height / 0.003 − 0.1 · ‖ctrl‖² − min(0.5e-6 · ‖cfrc_ext‖², 10) + 1
```

The height term divides by the physics timestep (0.003 s), not the control step, as in v4. The impact cost is on the
external contact forces and capped at 10. There is no early termination: every episode runs 1000 steps (or terminates,
with reward 0, if the state stops being finite). `info` holds `reward_linup`, `reward_quadctrl` and `reward_impact`.

## Example: longer arms

Lengthen both arms (upper arm and forearm, along their diagonal) and compare the first steps' return with the base body,
both lying still:

```python
import numpy as np
from codesign_gym import make_vec

envs, base = make_vec("HumanoidStandup-v4", 2, device="cpu")
long_arms = base.copy()
long_arms[12:24] = 1.0                      # every arm slot: each arm link about 0.25 m longer
obs, rejected, info = envs.reset(seed=0, embodiment=np.stack([base, long_arms]))

returns = np.zeros(2)
for _ in range(20):
    obs, reward, terminated, truncated, info = envs.step(np.zeros((2, 17), dtype=np.float32))
    returns += reward.cpu().numpy()
print("torso height", [round(z, 3) for z in obs[:, 0].tolist()])
print("return", returns.round(1).tolist())
envs.close()
```
