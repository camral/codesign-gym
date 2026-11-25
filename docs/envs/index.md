# Environments

Every environment has its own section: an overview, its **design space** (layout, bounds, how to build a design), its **observations, actions and rewards**, and runnable **examples**.

## Native

Made for co-design.

| Environment | Ids | Design | Control |
|---|---|---|---|
| [Ball Catcher](ball_catcher/index.md) | `BallCatcher-Pitch`, `-Arc`, `-Fielding` | Link geometry | Catch a thrown ball |
| [Solar Panel Cleaner](panel_cleaner/index.md) | `SolarCleaner-Reach`, `-Stability`, `-Traverse` | Leg and arm link geometry | Walk to panels and clean them |
| [Truck Unloading Line](truck_unload/index.md) | `TruckUnload-Single`, `-Multi` | Column and arm lengths of each robot | Unload and stack boxes |
| [NeroGrasp](nerograsp/index.md) | `NeroGrasp-<Object>` (13 ids) | Fingertip meshes | Grasp and lift small objects |
| [Shape-shifting Hand: Rotate](rotate_in_hand/index.md) | `RotateInHand-Cube` | Pad heights, finger spacing and width | Turn an object to goal angles |
| [Shape-shifting Hand: Pick Up](hand_pick_up/index.md) | `HandPickUp-Cube`, `-Hammer`, `-SmallHammer` | Pad heights, finger spacing and width | Lift and turn an object |
| [Soft Walker](soft_walker/index.md) | `SoftWalker-<Model>[Stairs\|Gaps]` (15 ids) | Material and actuation of each cell | Walk forward |

## Reframed

Known RL tasks with a hidden design problem.

| Environment | Ids | Design | Control |
|---|---|---|---|
| [LQ Structure](lq_structure/index.md) | `LQStructure-LQR`, `-LQG`, `-QP` | Masses, springs, dampers, sensors, actuators | Regulate the structure |
| [Microgrid](microgrid/index.md) | `Microgrid-Random`, `-Forecast`, `-Campus`, `-OffGridSingle`, `-OffGridCampus` | Solar, battery and generator size | Dispatch energy |
| [Wireless Network](mobile_env/index.md) | `Network-Urban`, `-Suburban`, `-Motorway` | Base station count, position and power | Connect users to stations |
| [Racing](f1tenth/index.md) | `Racing-Spielberg`, `-Levine` | Tyres, power, weight distribution | Drive a lap |
| [Warehouse](robot_warehouse/index.md) | `Warehouse-Small`, `-Medium`, `-Congested` | Shelf positions | Move robots to deliver shelves |
| [Pokémon](pokenv/index.md) | `Pokenv-Random`, `-MaxPower`, `-Heuristic`, `-League*` (23 ids) | Team | Battle |

## Extended

Classic MuJoCo tasks with a body you can reshape.

| Environment | Ids | Design | Control |
|---|---|---|---|
| [Gym MuJoCo ports](gym_mujoco/index.md) | `HalfCheetah-v4`, `Swimmer-v4`, `Hopper-v4`, `Walker2d-v4`, `Ant-v4`, `Humanoid-v4`, `HumanoidStandup-v4` | Limb lengths and angles | Same as Gymnasium v4 |

```{toctree}
:hidden:
:caption: Native

ball_catcher/index
panel_cleaner/index
truck_unload/index
nerograsp/index
rotate_in_hand/index
hand_pick_up/index
soft_walker/index
```

```{toctree}
:hidden:
:caption: Reframed

lq_structure/index
microgrid/index
mobile_env/index
f1tenth/index
robot_warehouse/index
pokenv/index
```

```{toctree}
:hidden:
:caption: Extended

gym_mujoco/index
```
