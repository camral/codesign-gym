# Ball Catcher

```{include} ../../_generated/readmes/ball_catcher.md
```

## At a glance

| | |
|---|---|
| Ids | `BallCatcher-Pitch`, `BallCatcher-Arc`, `BallCatcher-Fielding` |
| Design | `Box(45,)` float32 in `[0, 1]`: 15 links × 3 body-frame axes |
| Observation | `Box(70,)` float32 (torch tensor on the env's device) |
| Action | `Box(19,)` float32 in `[-1, 1]`: joint velocity targets |
| Episode | Up to 800 steps of 0.05 s (40 s) |
| Mid-episode design changes | No (a design passed to `step` is dropped with a warning) |
| Autoreset | Next step, keeping each copy's design; partial reset supported |
| Render modes | `rgb_array`, `human` |
| Extra install | None |
| Device | CUDA if visible, otherwise CPU; `device="cpu"` forces CPU |
| Source | [`src/envs/ball_catcher`](../../../src/envs/ball_catcher) |

The robot is a three-legged base carrying two arms. The left arm ends in a glove with a goal site inside it, and the
right arm ends in a sphere. The three ids share the robot and differ only in the throw:

| Id | Thrown from | Aimed at | Speed |
|---|---|---|---|
| `BallCatcher-Pitch` | `(8, 2, 1)` m | A point on the plane `x = 0.465` m with `y = 0.2 ± 1.5` m and `z = 1.5 ± 0.15` m | Whatever reaches that point in `0.4 ± 0.025` s |
| `BallCatcher-Arc` | `(8, 2, 1)` m | A point within 0.5 m of `(0.465, 0.2, 5.5)` | `14 ± 0.5` m/s |
| `BallCatcher-Fielding` | `(8, 2, 1)` m | A point within 1.0 m of `(0, 0, 4.5)` | `15 ± 2` m/s |

All spreads are uniform and drawn per copy at every reset. In `Pitch` the launch velocity is solved so that the ball,
under gravity, passes through the sampled point at the sampled time. In `Arc` and `Fielding` the ball is launched
straight at the sampled point and gravity bends it down, so it arrives well below the aim point. The robot can intercept
the ball anywhere along its flight.

```{toctree}
:maxdepth: 1

embodiment
spaces
examples
```
