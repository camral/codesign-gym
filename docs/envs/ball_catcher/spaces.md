# Observations, actions and rewards

Observations, rewards and flags come back as torch tensors on the env's device (CPU tensors with `device="cpu"`).
Actions can be NumPy arrays or tensors.

## Observation

A `Box(70,)` float32 vector. Positions are divided by the half-size of the field (10 m, centred on the origin), so they
read in metres / 10; velocities are in raw SI units. Only the quaternion entries are bounded (`[-1, 1]`); every other
entry is declared `(-inf, inf)`.

| Slice | Content |
|---|---|
| `[0:3]` | Base position / 10 |
| `[3:7]` | Base orientation quaternion (w, x, y, z) |
| `[7:26]` | 19 joint angles, each divided by the largest magnitude in its range (order below) |
| `[26:32]` | Base linear and angular velocity |
| `[32:51]` | 19 joint velocities (rad/s) |
| `[51:54]` | Ball position / 10 |
| `[54:57]` | Ball velocity (m/s) |
| `[57:60]` | Goal site (inside the glove) position / 10 |
| `[60:64]` | Goal site orientation quaternion, sign chosen so `w ≥ 0` |
| `[64:67]` | Goal site linear velocity (m/s) |
| `[67:70]` | Ball minus goal site, / 10 |

The joint order in `[7:26]` and `[32:51]` is the model's: `hip_fl`, `ankle_fl`, `hip_fr`, `ankle_fr`, `hip_b`,
`ankle_b`, `joint_l0`, `joint_l2`, `joint_l1`, `joint_l3`, `joint_l4`, `joint_l5`, `joint_l6`, `joint_l2r`, `joint_l1r`,
`joint_l3r`, `joint_l4r`, `joint_l5r`, `joint_l6r`.

The design is not part of the observation.

## Action

A `Box(19,)` vector in `[-1, 1]`. Every actuator is a velocity servo: entry `i` is mapped linearly onto that actuator's
target joint velocity range, and the servo drives the joint toward it within a ±60 N·m force limit. An all-zero action
holds every joint still.

| Entries | Joints | Velocity range (rad/s) |
|---|---|---|
| `0–5` | `hip_fl`, `ankle_fl`, `hip_fr`, `ankle_fr`, `hip_b`, `ankle_b` | ±6 |
| `6` | `joint_l0` (arm yaw on the base) | ±3 |
| `7–12` | `joint_l1`, `joint_l2`, `joint_l3`, `joint_l4`, `joint_l5`, `joint_l6` (left arm) | ±3 |
| `13–18` | `joint_l1r`, `joint_l2r`, `joint_l3r`, `joint_l4r`, `joint_l5r`, `joint_l6r` (right arm) | ±3 |

The action order lists `joint_l1` before `joint_l2`, the reverse of the observation order.

## Reward

The return lies between 0 and 10 and is made of three terms. Let `d` be the distance from the ball to the goal site
inside the glove, and `d0` its value at the start of the episode.

| Term | Gives | Total |
|---|---|---|
| Progress | `0.1 · (best d so far − d) / d0`, only when `d` beats its best | up to 0.1 |
| At goal | 0.495 on each step with `d < 0.1` m, for the first 10 such steps | up to 4.95 |
| Grounded catch | 4.95 once, when the ball is caught and all three feet touch the floor | 4.95 |

The ball counts as *caught* once it has spent 10 steps in total within 0.1 m of the goal site and is still within
0.2 m. No term is negative.

## Episode end

| Ends with | When |
|---|---|
| `terminated` | Grounded catch (success) |
| `terminated` | Dropped: the 10 steps at the goal are done but the ball is more than 0.2 m away |
| `terminated` | The ball or the goal site leaves the field, a 20 m cube centred on the origin |
| `terminated` | The copy's physics went non-finite (its observation row is zeroed and its reward is 0) |
| `truncated` | 800 control steps (40 s) |

A control step is 5 physics steps of 0.01 s. A caught ball that is not on all three feet does not end the episode.

## Autoreset and partial reset

A copy that ends is reset on the next `step`, with the same design and a fresh throw.
This is Gymnasium's `NEXT_STEP` mode: that `step` ignores the copy's action and returns its first observation, with
reward 0 and `terminated` and `truncated` both `False`. The terminal observation of the old episode is the one returned by
the step that ended it.

`reset(options={"reset_mask": m})` resets only the copies where the boolean array `m` is `True`. `step` raises
`RuntimeError` until every copy has had a successful reset, so the first resets may be partial, and a copy that a reset
rejected can be reset on its own. Seeds: an integer seeds the whole batch, and a sequence of one integer per copy gives each copy its own stream.

## Info

`reset` returns `{"embodiment": t}`, where `t` is the `(num_envs, 45)` tensor of applied designs.

Every `step` returns, for every copy:

| Key | Content |
|---|---|
| `final_obs` | The observation of that step, before any reset |
| `final_return` | The episode return so far |
| `final_steps` | Episode length in control steps |

Read them only for the copies with `terminated | truncated`. After each step the env also exposes boolean tensors that
say why a copy ended: `last_held_at_goal` (grounded catch), `last_caught`, `last_dropped`, `last_grounded`,
`last_at_goal`, `last_left_field`, `last_time_limit` and `last_diverged`.
