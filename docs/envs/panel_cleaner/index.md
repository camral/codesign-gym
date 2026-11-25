# Solar Panel Cleaner

```{include} ../../_generated/readmes/panel_cleaner.md
```

## At a glance

| | |
|---|---|
| Ids | `SolarCleaner-Reach`, `SolarCleaner-Stability`, `SolarCleaner-Traverse` |
| Design | `Box(48,)` float32 in `[0, 1]`: 16 links × 3 body-frame axes |
| Observation | `Box(69,)` float32 (torch tensor on the env's device) |
| Action | `Box(18,)` float32 in `[-1, 1]`: joint velocity targets |
| Episode | No fixed length: a step budget per goal (0.04 s steps) |
| Mid-episode design changes | No (a design passed to `step` is dropped with a warning) |
| Autoreset | Next step, keeping each copy's design; partial reset supported |
| Render modes | `rgb_array`, `human` |
| Extra install | None |
| Device | CUDA if visible, otherwise CPU; `device="cpu"` forces CPU |
| Source | [`src/envs/panel_cleaner`](../../../src/envs/panel_cleaner) |

The robot is a quadruped with a six-joint arm on its back and a roller (0.25 m wide, 0.05 m radius) at the arm's tip.
It spawns 1.5 m behind its model's default position (3.3–3.9 m from the nearest panel) and has to visit every cell of every panel in a fixed order. The three ids share the
robot and the design space; the scene sets which part of the body matters most.

| | `Reach` | `Stability` | `Traverse` |
|---|---|---|---|
| Panels (rows × columns) | 1 × 3 | 2 × 2, spread out | 3 × 4, tightly packed |
| Panel length × width, centre height, tilt | 1.3 × 1.9 m, 2.0 m, 20° | 1.2 × 1.6 m, 1.4 m, 30° | 1.1 × 1.3 m, 1.4 m, 20° |
| Cells per panel | 4 × 6 | 4 × 5 | 4 × 4 |
| Cells / goals in total | 72 / 74 | 80 / 83 | 192 / 203 |
| Steps on a cell to clean it | 1 | 15 | 1 |
| Step budget: walk to a panel / next cell | 700 / 150 | 700 / 200 | 600 / 120 |
| `max_offset` | 0.15 m | 0.1 m | 0.1 m |
| What the panels collide with | The roller only | The whole robot | The whole robot |
| A body part penetrating a panel (by 5 mm or more) | Passes through it | Ends the episode | Ends the episode |
| Ground | Flat | Smooth hills 0.3–1.2 m high, new each episode | Flat |
| Disturbances | None | 320 N shoves and random leg forces | None |
| Spawn jitter | None | None | ±0.15 m, ±10° |

The goals are the cells plus one *midpoint* in the gap before each new panel, which splits the walk between two panels
into two shorter approaches. With rendering on, `is_eval_env=True` switches the camera from a fixed view of the whole
scene to one that follows the robot from behind.

```{toctree}
:maxdepth: 1

embodiment
spaces
examples
```
