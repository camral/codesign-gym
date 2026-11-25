# Truck Unloading Line

```{include} ../../_generated/readmes/truck_unload.md
```

## At a glance

| | |
|---|---|
| Ids | `TruckUnload-Single`, `TruckUnload-Multi` |
| Design | `Box(6,)` (Single) / `Box(9,)` (Multi) float32: 3 link lengths per robot |
| Observation | `Box(115,)` (Single) / `Box(172,)` (Multi) float32 (torch tensor on the env's device) |
| Action | `Box(16,)` (Single) / `Box(24,)` (Multi) float32 in `[-1, 1]`: 8 per robot |
| Episode | 3000 steps (120 s) / 7500 steps (300 s) of 0.04 s, ended earlier by 15 s of inactivity |
| Mid-episode design changes | No (a design passed to `step` is dropped with a warning) |
| Autoreset | Next step, keeping each copy's design; partial reset supported |
| Render modes | `rgb_array`, `human` |
| Extra install | None |
| Device | CUDA if visible, otherwise CPU; `device="cpu"` forces CPU |
| Source | [`src/envs/truck_unload`](../../../src/envs/truck_unload) |

Every robot is a SCARA arm on a mobile base that drives in `x`, `y` and yaw. The arm has a vertical column with a
carriage sliding up it, two horizontal links on hinge joints, a wrist hinge and a suction (adhesion) gripper. Robot 0
is always the loader; the stackers follow in order.

| | `Single` | `Multi` |
|---|---|---|
| Robots | Loader + 1 stacker | Loader + 2 stackers |
| Container | 2 × 2 × 1 boxes (4) | 2 × 2 × 2 boxes (8) |
| Boxes already on the belt | 1 | 2 |
| Stacker 1 | Red boxes onto a shelf (5 boxes) | Red boxes onto a shelf (5 boxes) |
| Stacker 2 | — | Blue boxes anywhere on a flat 3 × 3 area (5 boxes) |
| Time limit | 120 s | 300 s |

Boxes are 0.1 m cubes of 0.5 kg. The belt carries them at about 0.2 m/s from the loader toward the stackers. Each respawn
draws a slightly different scene, seeded by `reset(seed=...)`: the belt speed varies by up to ±20% and each container box
is nudged by up to 9 mm (see [Scene variation](spaces.md#scene-variation)).

```{toctree}
:maxdepth: 1

embodiment
spaces
examples
```
