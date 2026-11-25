# Observations, actions and rewards

Observations, rewards and flags come back as torch tensors on the env's device (CPU tensors with `device="cpu"`).
Actions can be NumPy arrays or tensors.

## Observation

A `Box(69,)` float32 vector. Positions are given relative to the preset's *field*, a box around the spawn and the
array: `(p − field centre) / field half-size`, so they read about `[-1, 1]` inside it. Velocities are divided by 10.
Only the quaternions, the goal normal and the clock are bounded; every other entry is declared `(-inf, inf)`.

| Slice | Content |
|---|---|
| `[0:3]` | Base position, field-normalised |
| `[3:7]` | Base orientation quaternion (w, x, y, z) |
| `[7:25]` | 18 joint angles, each divided by the largest magnitude in its range (order below) |
| `[25:31]` | Base linear and angular velocity / 10 |
| `[31:49]` | 18 joint velocities / 10 |
| `[49:52]` | Roller position, field-normalised |
| `[52:56]` | Roller orientation quaternion |
| `[56:59]` | Roller linear velocity / 10 |
| `[59:62]` | Current goal point, field-normalised |
| `[62:65]` | Current goal's surface normal (unit vector) |
| `[65:68]` | Roller minus goal point, / 1.75 m |
| `[68]` | Time left on the current goal's budget: 1 when it is handed out, 0 on its last step |

The joint order is `fl_hip2`, `fl_hip`, `fl_knee`, `fr_hip2`, `fr_hip`, `fr_knee`, `bl_hip2`, `bl_hip`, `bl_knee`,
`br_hip2`, `br_hip`, `br_knee`, then the arm's `joint_l0` to `joint_l5`. On the step a cell is cleaned, the goal block
`[59:69]` already describes the next goal.

The field for each preset (centre, half-size, in metres):

| Id | Centre | Half-size |
|---|---|---|
| `SolarCleaner-Reach` | (1.65, 0, 1.2) | (3.75, 3.8, 2.6) |
| `SolarCleaner-Stability` | (1.85, 0, 2.05) | (4.75, 3.4, 3.75) |
| `SolarCleaner-Traverse` | (2.15, 0, 0.95) | (4.45, 3.7, 2.0) |

The design is not part of the observation.

## Action

A `Box(18,)` vector in `[-1, 1]`, in the same joint order as the observation. Every actuator is a velocity servo: entry
`i` is mapped linearly onto that joint's target velocity range, and the servo drives the joint toward it within its force
limit. An all-zero action holds every joint still.

| Entries | Joints | Velocity range (rad/s) | Force limit (N·m) |
|---|---|---|---|
| `0–11` | Hip (side), hip, knee of each leg: front left, front right, back left, back right | ±6 | ±120 |
| `12–17` | Arm `joint_l0` to `joint_l5` | ±3 | ±60 |

## Reward

The goals are visited in a fixed order: the cells of one panel in a back-and-forth sweep, then a midpoint, then the
next panel. A cell is cleaned once the roller has spent enough steps on it, counted cumulatively (1 step in `Reach` and
`Traverse`, 15 in `Stability`). The roller is *on* a cell when its centre is within the cell's footprint and resting
on the surface (within 2 cm); a midpoint is reached by entering a box of half-extent 0.3 m around it.

A full sweep gives at most 10 (the approach term stays slightly below its share in practice), split into two terms:

| Term | Gives | Total |
|---|---|---|
| Approach | `(2.5 / goals) · (previous distance − distance) / d0` each step, where `d0` is the distance when the goal was handed out | 2.5 over the sweep |
| Cleaning | `7.5 / (cells · steps per cell)` on each step the roller is on its current cell | 7.5 over the sweep |

| Id | Approach per goal | Cleaning per step on a cell |
|---|---|---|
| `SolarCleaner-Reach` | 0.0338 | 0.104 |
| `SolarCleaner-Stability` | 0.0301 | 0.00625 |
| `SolarCleaner-Traverse` | 0.0123 | 0.0391 |

The approach term is a difference of distances, so moving away from the goal costs exactly what moving toward it gave:
single-step rewards and short returns can be slightly negative. A copy that ends early keeps what it has earned.

## Episode end

There is no fixed episode length. Each goal gets a step budget when it is handed out: the walk budget for the first
cell, for each midpoint and for the first cell after one, and the shorter cell budget otherwise (see the table in the
overview). A control step is 8 physics steps of 0.005 s.

| Ends with | When |
|---|---|
| `terminated` | Every goal reached (the sweep is done) |
| `terminated` | The current goal's budget runs out (reported as `truncated` instead with `truncate_at_time_limit=True`) |
| `terminated` | The roller leaves the field |
| `terminated` | `Stability`, `Traverse`: a robot part other than the roller presses 5 mm or more into a panel |
| `terminated` | `Stability`: the base walks off the terrain patch |
| `terminated` | The base sinks below 75% of the design's standing height, measured above the ground under it (`min_base_height_frac`) |
| `terminated` | The copy's physics went non-finite (its observation row is zeroed and its reward is 0) |

With the default settings `truncated` is always `False`.

## Autoreset and partial reset

A copy that ends is reset on the next `step` with the same design; on `Stability` it also gets new terrain.
This is Gymnasium's `NEXT_STEP` mode: that `step` ignores the copy's action and returns its first observation, with
reward 0 and `terminated` and `truncated` both `False`. The terminal observation of the old episode is the one returned by
the step that ended it.

`reset(options={"reset_mask": m})` resets only the copies where the boolean array `m` is `True`, with the design given
for them; the other copies carry on and keep their designs. Seeds: an integer seeds the batch, and a sequence of one
integer per copy gives each copy its own stream, terrain included.

## Info

`reset` returns `{"embodiment": t}`, where `t` is the `(num_envs, 48)` tensor of applied designs.

Every `step` returns, for every copy:

| Key | Content |
|---|---|
| `final_obs` | The observation of that step, before any reset |
| `final_return` | The episode return so far |
| `final_steps` | Episode length in control steps |

Read them only for the copies with `terminated | truncated`. After each step the env also exposes per-copy tensors that
say what happened: `goals_cleaned` (goals reached this episode), `last_at_goal`, `last_cleaned`, `last_all_clean`,
`last_left_field`, `last_time_limit`, `last_body_contact`, `last_off_green`, `last_too_low` and `last_diverged`.
