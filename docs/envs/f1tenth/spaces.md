# Observations, actions and rewards

## Observation

A `Box(39,)` float32 vector (with the default 32 lidar beams):

| Index | Content | Unit |
|---|---|---|
| 0 | `s`: progress along the centre line as a fraction of the lap, measured from the track's start line | fraction in [0, 1] |
| 1 | `ey`: lateral offset from the centre line, positive to the left | m |
| 2 | `epsi`: heading relative to the centre line | rad |
| 3 | Steering angle | rad |
| 4 | Longitudinal speed | m/s |
| 5 | Yaw rate | rad/s |
| 6 | Slip angle at the centre of gravity | rad |
| 7–38 | Lidar ranges, from 2.35 rad to the right to 2.35 rad to the left (4.7 rad field of view, evenly spaced) | m |

The lidar range is capped at 10 m before Gaussian noise with std 0.01 m is added, so readings can slightly exceed 10 m. `s` is an absolute track position, so it jumps back to 0 when the car crosses the start line.

## Action

A `Box(2,)` float32 vector `[steering velocity, acceleration]`:

- **Steering velocity** (rad/s), bounded to `[-3.2, 3.2]`. It is the rate of change of the steering angle, not the angle itself. The angle stops at ±0.4189 rad.
- **Acceleration** (m/s²). The box is `[-17.62, 17.62]`: the largest `a_max = F_drive / m` that any design in the space can reach, so the box is the same for every design. The dynamics then clip the command to the current car's own `a_max` (9.51 for the base car), for braking as well. Above 7.319 m/s the forward limit shrinks to `a_max · 7.319 / v`, and speed is kept within `[-5, 20]` m/s.

A command outside a car's limits is clipped silently, so a policy can use the same action scale for every design.

## Reward

Each step pays the distance the car travelled along the centre line in that step, in metres:

```text
reward = s(t+1) − s(t)          (wrapped across the start line)
```

Driving backwards earns negative reward. There is no penalty for crashing beyond losing the rest of the episode. One lap of the track is worth its length: 343.3 m on Spielberg and 63.2 m on Levine.

## Episode

Each step is 0.01 s of simulation (one RK4 step of the single-track model with tyre slip). At reset the car starts at rest at a uniformly random point on the centre line, with a lateral offset drawn from `[-0.45, 0.45]` m, facing along the track.

The episode ends with

- `terminated=True` when the car touches a wall (a crashed car is frozen in place), or when the simulation produces non-finite values (`info["diverged"]`);
- `truncated=True` after the step limit (1500 steps on Spielberg, 1000 on Levine, 10000 for an evaluation env), or when the car completes one lap (`info["success"]`).

The step limit can be changed with `time_limit=` (see [Design space](embodiment.md)).

With `reset(seed=s)` all copies draw their starts from one random stream, so they start at different points. Pass one seed per copy (for example `seed=[0] * num_envs`) to start every copy at the same point.

## Info

`reset` returns:

| Key | Content |
|---|---|
| `design` | `(num_envs, 5)` designs in use (the base design for a rejected copy, which cannot step until it is reset with a valid design) |
| `step_count` | Step counter of each copy |

`step` returns:

| Key | Content |
|---|---|
| `embodiment_rejected` | Always `False` (an invalid design at `step` raises instead) |
| `design` | `(num_envs, 5)` designs in use after this step |
| `step_count` | Step counter of each copy |
| `success` | The copy finished a full lap on this step |
| `diverged` | The simulation produced non-finite values; the observation and reward are zeroed and the copy terminates |

## Rendering

`render_mode="rgb_array"` makes `render()` return a tuple with one 420×420 frame per copy (one frame for `make`): the track, the car, its recent trail, and a text panel with the design, mass, effective grip, `a_max` and lap progress.
`render(env_idx=i)` returns copy `i`'s frame alone. `"human"` shows copy 0 (or copy `i` when you call `render(env_idx=i)`) in a live window. `"web"` records the whole run instead; call `env.save_dashboard(path)` to write f1tenth's interactive HTML dashboard, with one rollout per copy.
