# Observations, actions and rewards

## Observation

A `Box(66 · robots,)` float32 vector: the robots' views one after another, 66 integer values each (with the default `sensor_range=1`). Robot `k`'s view is `obs[66k : 66k + 66]`:

| Slice | Content |
|---|---|
| `[0:2]` | The robot's cell `(row, col)` |
| `[2]` | 1 if it is carrying a shelf |
| `[3:7]` | Its direction, one-hot: up, right, down, left |
| `[7]` | 1 if it stands on a goal cell |
| `[8:48]` | The 8 neighbouring cells (the 3×3 window around the robot in row-major order, minus its own cell), 5 values each: 1 if another robot is there, then that robot's direction one-hot |
| `[48:66]` | All 9 cells of the window in row-major order, 2 values each: 1 if a shelf is there, then 1 if that shelf is requested |

Cells outside the grid read as empty. A robot sees requested shelves only inside its window; the request queue itself is not observed.

## Action

`MultiDiscrete([5] * robots)`: one action per robot, applied in robot order within a step.

| Action | Effect |
|---|---|
| 0 | Nothing |
| 1 | Move one cell forward (a robot at the edge stays put) |
| 2 | Turn left (90° counter-clockwise) |
| 3 | Turn right (90° clockwise) |
| 4 | Toggle load: pick up the shelf in this cell if not carrying, or put the carried shelf down if not on a goal |

An unloaded robot drives under shelves. Moving forward while carrying into a cell that holds a shelf is invalid. The per-robot `info["action_mask"]` (shape `(robots, 5)`) marks it `False`, and an invalid action is replaced by action 0.

Shelves can be put down on any cell except a goal, including cells that were empty in the design, so the layout changes during an episode as robots move shelves around. Each reset restores the design.

## Reward

The reward is shared by all robots:

```text
reward = number of requested shelves delivered this step
```

A delivery happens as soon as a requested shelf is on a goal cell, carried by a robot; no toggle is needed. The check runs every step, so a robot that stays on a goal is credited again if its shelf is requested again. The shelf then stops being requested, and a new request is drawn at random from the shelves on the floor that are not already requested. The robot must leave the goal and put the shelf down elsewhere before it can fetch another.
At most two deliveries can happen per step (one per goal), so 1000 is an upper bound on the return over 500 steps.

## Episode

500 steps (`time_limit`). Robots and their directions are placed at random on free cells at reset, and the initial requests are drawn at random from the layout's shelves.

- `terminated=True` when two robots end up in the same cell (on the final step, a collision reports `truncated=True` instead).
- `truncated=True` at the step limit.

## Info

| Key | Content |
|---|---|
| `action_mask` | `(robots, 5)` bool per copy: which actions are valid now |
| `step_count` | Step counter of each copy |
| `embodiment_rejected` | `step` only; always `False` |

## Rendering

With `render_mode="rgb_array"`, `render()` returns a tuple with one frame per copy, and `render(env_idx=i)` returns copy `i`'s frame alone. A frame draws the grid (30 px per cell), green goals, grey shelves (orange when requested) and blue robots (purple when carrying) with a white tick for their direction, above a text panel with the step and the episode's deliveries. Frame sizes are even, so they can go straight into an H.264 video.
`render_mode="human"` shows one copy (copy 0 by default) in a Jumanji matplotlib window.
