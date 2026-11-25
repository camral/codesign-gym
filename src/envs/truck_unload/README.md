# Truck Unloading Line

Ids: `TruckUnload-Single`, `TruckUnload-Multi`. Engine: MuJoCo Warp.

A loader robot empties a container down a ramp onto a conveyor, and stacker robots pick boxes of their own colour off the belt. Every robot is a SCARA arm on a mobile base.

- **Design:** the column length and two arm link lengths of each robot, set independently. Links can grow by up to `max_offset` (1.0 m) or shrink down to a minimum length.
- **Control:** one central policy drives all arms, bases and suction grippers.
- **Reward:** each box gives reward for being reached, carried, released and placed at its goal. A fully unloaded and stacked line earns at most 10 (5 for the loader's boxes, 5 shared by the stackers); the progress terms can go negative. `info["reward_per_robot"]` splits the reward per robot, loader first. A box that leaves its goal after delivery loses all of its reward.
- **Design changes in `step`:** not supported. The policy does not see the design.

| Id | Task |
|---|---|
| `TruckUnload-Single` | One stacker shelves the red boxes from a 2x2x1 container. |
| `TruckUnload-Multi` | From a 2x2x2 container, a red stacker fills a shelf and a blue stacker places its boxes anywhere on a flat 3x3 area. |
