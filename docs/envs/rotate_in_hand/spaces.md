# Observations, actions and rewards

The hand is fixed palm-up at its base; its wrist can roll, pitch and yaw. An 8 cm cube (0.1 kg) is dropped onto the palm at every reset.
The task is to turn the cube through a stream of goal orientations, each 1 rad from where the cube was when the last goal was met.
Everything about the object is measured in the palm's frame, and so are the goals (`goal_frame="palm"`), so turning the wrist alone does not count as turning the cube.
With `goal_frame="world"` the goals are world orientations instead, which stay put when the wrist turns; they are still observed in the palm's frame.

## Observation

A `Box(57,)` float32 torch tensor on the simulation device:

| Slice | Content | Bounds |
|---|---|---|
| `[0:19]` | Joint positions, each divided by the largest magnitude of its range: wrist `roll, pitch, yaw`, then ring, middle, index and thumb joints `0–3` | ±inf |
| `[19:38]` | Joint velocities (rad/s), same order | ±inf |
| `[38:41]` | Object position in the palm frame, in units of 10 cm | ±inf |
| `[41:45]` | Object orientation in the palm frame, quaternion `w, x, y, z` | [-1, 1] |
| `[45:48]` | Object linear velocity relative to the palm, palm axes (m/s) | ±inf |
| `[48:51]` | Object angular velocity relative to the palm, palm axes (rad/s) | ±inf |
| `[51:55]` | Current goal orientation in the palm frame, quaternion `w, x, y, z` | [-1, 1] |
| `[55]` | Angle to the goal, divided by π | [0, 1] |
| `[56]` | 1 if the hand touched the object in either of the last 2 control steps | [0, 1] |

The design is not observed.

## Action

A `Box(19,)` in `[-1, 1]`, one entry per velocity actuator. Entry `a` sets that joint's target velocity to `2 a` rad/s.
The actuator order differs from the observation's joint order:

| Entries | Actuators |
|---|---|
| `[0:3]` | Wrist `pitch, yaw, roll` |
| `[3:7]` | Index finger joints 0–3 |
| `[7:11]` | Middle finger joints 0–3 |
| `[11:15]` | Ring finger joints 0–3 |
| `[15:19]` | Thumb joints 0–3 |

A control step is 0.04 s (8 physics steps of 5 ms).

## Reward

Gains are given only while the hand touches the object (the flag in `obs[56]`), and a step at the goal only counts while it is 1.

| Term | Value |
|---|---|
| Shaping | Potential-based: each step, how far the angle to the goal fell during it, as a share of the goal's starting angle, times 3. A full turn to the goal gives 3. A gain is multiplied by the touch flag; a loss (turning away) is charged in full, touching or not, so turning away and back nets at most 0 and dropping the cube costs. |
| Goal | 0.7 for each step the cube spends within 0.1 rad of the goal; the goal is met after 10 such steps (not necessarily in a row), so it gives 7 |

A goal met is worth 10, and the next goal is drawn at once from the cube's current orientation.
The return can be negative: a cube that ends further from its goal than it started has lost more shaping than it gained.

## Episode

| Ends with | When |
|---|---|
| `terminated` | The cube falls 0.5 m below the palm |
| `terminated` | 7.5 s pass without meeting the current goal |
| `terminated` | The physics diverges (that copy's observation is zeroed and its reward is 0) |
| `truncated` | 15 s (375 steps) pass |

A copy that ends is respawned on the following `step`, with its latest design. This is Gymnasium's `NEXT_STEP` mode: that step ignores the copy's action and returns its first observation, with reward 0 and `terminated` and `truncated` both `False`. The shaping counts from the respawn pose.
`reset(options={"reset_mask": m})` resets only the copies where `m` is `True` (a boolean array of shape `(num_envs,)`), leaving the others untouched. `step` raises `RuntimeError` until every copy has had a successful reset, so the first resets may be partial, and a copy that a reset rejected can be reset on its own.
The goal stream depends on the seed: `reset(seed=s)` replays it.

## Reset and info

`reset` returns `(obs, rejected, info)`. `rejected` is a `(num_envs,)` bool tensor, `True` only for a copy whose physics diverged at the respawn; an invalid design raises `ValueError` instead.

| Key | When | Content |
|---|---|---|
| `embodiment` | reset | `(num_envs, 248)` design in force in each copy |
| `final_obs` | step | Observation of every copy; for a copy that ended on this step, its last observation |
| `final_return` | step | Return so far, or the episode's return for a copy that ended on this step |
| `final_steps` | step | Control steps so far, or the episode's length for a copy that ended |

The env also exposes per-copy tensors after each step: `goals_reached` (goals met this episode), `is_touching` (this step's reward multiplier) and `last_success` (met a goal on this step).
`env_offsets` gives the design in force in metres, and `diverged_worlds` counts copies whose physics diverged.
