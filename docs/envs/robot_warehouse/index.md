# Warehouse

```{include} ../../_generated/readmes/robot_warehouse.md
```

## At a glance

| | |
|---|---|
| Ids | `Warehouse-Small`, `Warehouse-Medium`, `Warehouse-Congested` |
| Design | `MultiBinary(n)` int8, one bit per floor cell except the two goals (n = 98 / 254 / 438) |
| Observation | `Box(66 · robots,)` float32: each robot's local 3×3 view (132 / 264 / 528) |
| Action | `MultiDiscrete([5] * robots)`: one move per robot (2 / 4 / 8 robots) |
| Episode | 500 steps; ends early if two robots collide |
| Mid-episode design changes | No (a design passed to `step` is ignored with a warning) |
| Autoreset | Next step; partial reset supported |
| Render modes | `rgb_array`, `human` |
| Extra install | None |
| Source | [`src/envs/robot_warehouse`](../../../src/envs/robot_warehouse) |

```{toctree}
:maxdepth: 1

embodiment
spaces
examples
```
