# Ball Catcher

Ids: `BallCatcher-Pitch`, `BallCatcher-Arc`, `BallCatcher-Fielding`. Engine: MuJoCo Warp.

A robot learns to catch a thrown ball.

- **Design:** the length and direction of each link; link masses follow from the shape.
- **Control:** the robot joints.
- **Reward:** 0 to 10 per episode.
- **Design changes in `step`:** not supported. The policy does not see the design.

| Id | Throw | Return [max / min] |
|---|---|---|
| `BallCatcher-Pitch` | Close and low, aimed across a spread of lateral positions. | +10 / 0 |
| `BallCatcher-Arc` | Farther and lofted, with moderate spread. | +10 / 0 |
| `BallCatcher-Fielding` | Distant and high, with a wide spread in launch speed. | +10 / 0 |

## Throw Parameters

Every throw starts from `pitch_from`. A new throw is drawn for each copy at every reset, and all spreads are uniform. A throw uses one of two modes:

- **Arrival** (`Pitch`): the ball passes through a point in the box `pitch_to ± pitch_target_half_widths` after a flight time of `pitch_arrival_time ± pitch_arrival_spread`. The launch velocity is solved from these, as `(target - start) / t - gravity * t / 2`. `pitch_arrival_time` must be larger than its spread.
- **Speed** (`Arc`, `Fielding`): the ball is launched at `pitch_vel ± pitch_vel_spread` m/s straight at a point within `pitch_to_radius` of `pitch_to`. Gravity bends it down, so it arrives below that point.

Setting `pitch_arrival_time` selects arrival mode. Leaving it as `None` selects speed mode. Each mode ignores the other mode's parameters. The target is where the ball would pass if nothing stopped it, and the robot can catch the ball anywhere along its flight.

| Parameter | `Pitch` | `Arc` | `Fielding` |
|---|---|---|---|
| `pitch_from` (m) | (8, 2, 1) | (8, 2, 1) | (8, 2, 1) |
| `pitch_to` (m) | (0.4648899, 0.2, 1.5) | (0.4648899, 0.2, 5.5) | (0, 0, 4.5) |
| `pitch_target_half_widths` (m) | (0, 1.5, 0.15) | | |
| `pitch_arrival_time` ± `pitch_arrival_spread` (s) | 0.4 ± 0.025 | | |
| `pitch_to_radius` (m) | | 0.5 | 1.0 |
| `pitch_vel` ± `pitch_vel_spread` (m/s) | | 14 ± 0.5 | 15 ± 2 |

`env.set_pitch(...)` changes the throw during a run, for example for a curriculum. In arrival mode it only accepts `pitch_from` and `pitch_to`.
