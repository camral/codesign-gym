# Observations, actions and rewards

A Nero arm carries a LEAP hand down to an object on a table, closes, and lifts.
The arm follows a fixed script; the policy only moves the hand.
Only the index finger and the thumb take part: the middle and ring fingers are held curled.

## Observation

A `Dict` with an `actor` and a `critic` entry, both float32 torch tensors on the simulation device.
The critic sees everything the actor sees, plus the object's true state:

| Slice | Content |
|---|---|
| `actor[0]` | Episode progress, `step / 100` |
| `actor[1:8]` | Arm joint positions (7, rad), set by the script |
| `actor[8:24]` | Hand joint positions (16, rad): index `mcp, rot, pip, dip`, then middle, ring, and thumb `cmc, axl, mcp, ipl` |
| `actor[24:27]` | Thumb tip position (world, m) |
| `actor[27:30]` | Index tip position (world, m) |
| `critic[0:30]` | The actor observation |
| `critic[30:33]` | Object position (world, m) |
| `critic[33:37]` | Object orientation, quaternion `w, x, y, z` |
| `critic[37]` | Object mass × 10 (kg) |
| `critic[38]` | 1 while both fingertips touch the object |
| `critic[39]` | Articulation angle (always 0 for these objects) |

The design is not observed. Two config fields change the layout:

- `asymmetric_observations=False` gives the actor the full 40-value critic observation.
- `include_object_one_hot=True` adds a one-hot object id (one column per object of the id): after the fingertip positions in the actor when observations are symmetric, otherwise after the mass in the critic.

```python
import torch
from codesign_gym import make_vec

env, base = make_vec("NeroGrasp-SetScrew", 1, use_cpu=True)
obs, rejected, info = env.reset(seed=0, embodiment=base)
actor, critic = obs["actor"][0], obs["critic"][0]
print(actor.shape, critic.shape)                 # torch.Size([30]) torch.Size([40])
print(critic[30:33])                             # tensor([-0.3900, -0.0400,  0.0075]): the set screw on the table
```

## Action

A `Box(16,)` in `[-1, 1]`, one entry per hand joint in the order above.
Each step, entry `k` moves joint `k`'s position target to its current position plus `action[k] / 10` rad.
The target is clipped to the joint's control range, narrowed to `[-1, 1]` rad where the range is wider.
Entries 4–11 (middle and ring fingers) are ignored, because those fingers are held curled.

A control step is 0.1 s (20 physics steps of 5 ms).

## Reward

A copy is "in contact" when both fingertips touch the object. Each step pays:

| Term | Value |
|---|---|
| Height | Object height in metres, while in contact |
| Contact | 0.02, while in contact |
| Distance | `0.003 / (1 + (d / 0.05)²)`, where `d` is the larger of the two fingertip-to-object distances (0 for a touching fingertip); paid only before step 60, when the arm starts to lift |
| Success | +1 on the last step if the copy is in contact and the object is more than 3 cm high |

A copy that has terminated earns 0 for the rest of the episode.

The return is at least 0. The code does not bound how high a held object can rise, so there is no strict maximum short of
the arm's reach. If the object stays at or below 0.2 m, which is below the highest palm position of the arm script
(0.234 m), the return is at most 100 × (0.2 + 0.02) + 59 × 0.003 + 1 ≈ 23.2. A lifted grasp realistically earns about 7.

## Episode

100 steps (10 s), the length of the arm script. All copies share this clock.

A copy terminates when one of its safeguards fires:

- the object's acceleration implies a contact force above 1000 N, for 2 steps in a row,
- a fingertip's bounding box overlaps the object's by more than 2 cm on every axis (tunnelling), for 2 steps,
- the object is more than 1.5 cm above the table without both fingertips on it, for 2 steps,
- part of the robot other than the fingertips touches the table or sinks 2 cm below it,
- the object is more than 1 m from the fingertips, or the physics returns NaN.

`terminated` and `truncated` are `True` only on the step a copy ends. After that it earns 0, and `info["finished"]` keeps it marked.
Once every copy has ended, `step` raises `RuntimeError` until `reset()`.
There is no autoreset, and `reset(options={"reset_mask": m})` with any `False` entry raises `NotImplementedError`.

Spawn position and rotation noise and action noise are off, so an episode depends only on the design, the actions and, for the washer ids (`-Thin`, `-All`), the washer size drawn at each reset from the seeded generator.

## Info

All per-copy values are torch tensors on the simulation device.

| Key | When | Content |
|---|---|---|
| `episode/step`, `episode/return` | always | Step count and return so far |
| `embodiment/id` | always | Index of the design each copy runs |
| `task/object` | always | Object name of each copy (a tuple of strings) |
| `contacts/fingertip_object` | always | `(num_envs, 2)` bool, `[thumb, index]` touching the object |
| `safeguards/force_count`, `safeguards/tunneling_count`, `safeguards/in_air_count` | always | Consecutive steps each safeguard has been active |
| `rewards/total`, `rewards/contact`, `rewards/distance`, `rewards/f2f_penalty` | step | This step's reward terms (`f2f_penalty` is always 0) |
| `episode/terminated`, `episode/truncated`, `finished` | step | End flags; `finished` stays `True` once a copy has ended |
| `embodiment/value` | reset | The supplied meshes of every copy, as `vertices` / `faces` tensors |
| `embodiment/rejection_reasons` | reset | Per copy, why its design was rejected (empty string if not) |
| `mesh_quality/<metric>` | reset | `(num_envs, 2)` mesh quality metrics, `[thumb, index]` |
| `episode/final_return`, `episode/completed` | last step | Return of every copy |
| `object_return/<object>` | last step | Mean return over the copies holding that object |
| `all_reward/<term>`, `<object>_reward/<term>` | last step | Mean per-episode sum of each reward term |
| `quality/<metric>_mean`, `quality/<metric>_std` | last step | Mesh quality summaries |

```python
obs, reward, terminated, truncated, info = env.step(torch.zeros((1, 16)))
print(info["task/object"], info["contacts/fingertip_object"])   # ('set_screw',) tensor([[False, False]])
env.close()
```
