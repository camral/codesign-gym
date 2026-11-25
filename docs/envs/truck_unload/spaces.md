# Observations, actions and rewards

Observations, rewards and flags come back as torch tensors on the env's device (CPU tensors with `device="cpu"`).
Actions can be NumPy arrays or tensors.

## Observation

One block of 57 values per robot, loader first, then the episode clock: `57 · robots + 1` values (115 for `Single`,
172 for `Multi`). Block `r` starts at `57 · r`. Positions are relative to that robot's field: `(p − field centre) /
field half-size`. Velocities are divided by 10 and relative vectors by 1.75 m. Only the quaternions, the two flags and
the clock are bounded; every other entry is declared `(-inf, inf)`.

| Offset in the block | Content |
|---|---|
| `[0:10]` | The robot's joint positions, each divided by the largest magnitude in its range (order below) |
| `[10:20]` | The robot's joint velocities / 10 |
| `[20:23]` | Arm tip position, field-normalised |
| `[23:27]` | Arm tip orientation quaternion |
| `[27:30]` | Arm tip linear velocity / 10 |
| `[30:33]` | Arm tip angular velocity / 10 |
| `[33:36]` | Target box position, field-normalised |
| `[36:40]` | Target box orientation quaternion |
| `[40:43]` | Target box linear velocity / 10 |
| `[43:46]` | Target box angular velocity / 10 |
| `[46:49]` | Target box minus tip, / 1.75 |
| `[49:52]` | Target box minus its goal, / 1.75 |
| `[52:55]` | Tip minus the target's goal, / 1.75 |
| `[55]` | 1 while the gripper holds a box |
| `[56]` | 1 if the robot has a target box |

The last entry of the observation is the fraction of the time limit still ahead: 1 after a reset, 0 on the final step.

The ten joints of each robot are the base's `joint_x`, `joint_y`, `joint_z` (slides), `joint_pitch`, `joint_roll`,
`joint_yaw` (hinges), then the arm's `j1` (carriage slide on the column) and `j2`, `j3`, `j4` (hinges). Stacker joints
carry the prefix `stacker0_`, `stacker1_`. The base's `z`, pitch and roll are passive.

The target is the box the robot is working on: for the loader the next box in the container, for a stacker the next
box of its colour in its field. A robot without a target reads zeros in `[33:52]` and `0` in `[56]`. The design is not
part of the observation.

## Action

Eight entries per robot, loader first, each in `[-1, 1]`:

| Entry in the robot's block | Actuator | Range |
|---|---|---|
| `0` | Base `x` velocity | ±1 m/s |
| `1` | Base `y` velocity | ±1 m/s |
| `2` | Base yaw rate | ±1 rad/s |
| `3` | Carriage `j1` velocity | ±0.5 m/s |
| `4–6` | Arm hinges `j2`, `j3`, `j4` velocity | ±3 rad/s |
| `7` | Gripper adhesion | `≤ 0` off; `(0, 1]` maps onto the full adhesion range |

Entries 0–6 are velocity servos: the value is mapped linearly onto the target velocity range, so an all-zero action
holds the robot still. Robot `r`'s entries start at `8 · r`.

## Reward

Each box is worth a fixed share of its robot's budget: the loader's boxes share 5 and the stackers' boxes share the
other 5, so a fully unloaded and stacked line earns at most 10 (the reach and carry terms stay slightly below their share in practice).

| Id | Per loader box | Per stacker box |
|---|---|---|
| `TruckUnload-Single` | 1.25 (4 boxes) | 1.0 (5 boxes) |
| `TruckUnload-Multi` | 0.625 (8 boxes) | 0.5 (10 boxes, 5 per stacker) |

A box's share is given in four quarters, one phase at a time:

| Phase | Gives a quarter of the share for |
|---|---|
| Reach | Moving the tip toward the box, in proportion to the distance closed (negative when moving away) |
| Carry | New records of the box's distance to its goal while holding it |
| Release | Letting go once the box is at its goal (in proportion to how far the grip drops) |
| Goal | Spread over 10 steps with the box at its goal and the tip clear of it (loader: across the belt within 5 cm, past its start; floor stacker: inside its 3×3 region; shelf stacker: within 5 cm) |

A box counts as delivered after those 10 steps (cumulative). If a delivered box later leaves its goal region, everything
given for it is taken back on that step, and it can be delivered again. The loader's goal is the belt; the stackers'
are a shelf level or the floor area. `info["reward_per_robot"]` splits each step's reward by robot.

## Episode end

| Ends with | When |
|---|---|
| `terminated` | The loader delivers nothing for 15 s, or no stacker delivers anything for 15 s (375 steps) |
| `terminated` | A box is lost (falls out of play) |
| `terminated` | Any robot's arm tip leaves its field |
| `terminated` | The time limit (120 s / 300 s; `truncated` instead with `truncate_at_time_limit=True`) |
| `terminated` | The copy's physics went non-finite (its observation row is zeroed and its reward is 0) |

A control step is 8 physics steps of 0.005 s. With the default settings `truncated` is always `False`. Delivering every
box sets `info["success"]` but does not end the episode.

## Autoreset and partial reset

A copy that ends is reset on the next `step` with the same design and a newly drawn scene (see below).
This is Gymnasium's `NEXT_STEP` mode: that `step` ignores the copy's action and returns its first observation, with
reward 0 and `terminated` and `truncated` both `False`. The terminal observation of the old episode is the one returned by
the step that ended it.

`reset(options={"reset_mask": m})` resets only the copies where the boolean array `m` is `True`; the others keep their
state and design. The first step after a reset or respawn rewards only that step's own progress.

## Scene variation

Every respawn (from `reset` or autoreset) draws a slightly different scene for that copy:

- the belt speed, uniform in `belt_velocity · (1 ± belt_velocity_spread)` (0.16 to 0.24 m/s by default), one speed for
  the belt and both rollers;
- an `x` and `y` nudge of each container box, uniform in `±box_jitter` (9 mm by default). The nudge must stay below half
  the clearance between boxes, so neighbours never overlap.

`reset(seed=...)` replays these draws: an integer seeds every copy (each still draws its own values), and a sequence of one
integer per copy gives each copy its own seed. Set `belt_velocity_spread=0` and `box_jitter=0` for a fixed scene.
`envs.set_belt_velocity(v)` sets the belt speed of every copy at once and the centre of later draws.

## Info

`reset` returns `{"embodiment": t}`, where `t` is the `(num_envs, n)` tensor of applied designs (`n` is 6 in
`TruckUnload-Single` and 9 in `TruckUnload-Multi`). Every `step` returns:

| Key | Shape | Content |
|---|---|---|
| `reward_per_robot` | `(num_envs, robots)` | This step's reward split by robot, loader first; sums to the reward |
| `boxes_delivered` | `(num_envs, robots)` | Boxes delivered so far this episode, net of any that left their goal |
| `boxes_undelivered` | `(num_envs, robots)` | Delivered boxes that left their goal region on this step |
| `success` | `(num_envs,)` | Every robot has delivered all its boxes |
| `final_obs` | `(num_envs, obs)` | This step's observation, before any reset |
| `final_return` | `(num_envs,)` | Episode return so far |
| `final_steps` | `(num_envs,)` | Episode length in control steps |

The `final_*` keys are present on every step; read them only for the copies with `terminated | truncated`. The env
also exposes per-copy tensors that say why a copy ended: `last_idle`, `last_dropped`, `last_left_field`,
`last_time_limit` and `last_diverged`.
