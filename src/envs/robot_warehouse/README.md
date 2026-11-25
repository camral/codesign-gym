# Warehouse

Ids: `Warehouse-Small`, `Warehouse-Medium`, `Warehouse-Congested`. Uses: [Jumanji RobotWarehouse](https://instadeepai.github.io/jumanji/environments/robot_warehouse/) (JAX).

A fleet of robots fetches requested shelves and delivers them to goal cells.

- **Design:** where each shelf sits on a fixed grid.
- **Control:** one central policy moves every robot.
- **Reward:** +1 per requested shelf delivered, over 500 steps.
- **Design changes in `step`:** not supported.

Unlike stock RobotWarehouse, there are no fixed corridors. Only the goal cells are reserved, and aisles are simply wherever shelves aren't. Since a loaded robot can't pass through a shelf cell, the layout decides where robots can go: shelves near the goals shorten routes, spread-out shelves ease congestion, and a layout that is too dense traps robots.

Validation only checks the layout's form: a binary integer grid of the right size (goal cells have no bit, so shelves can never be placed on them), at least the preset's minimum number of shelves, and at most as many shelves as there are cells left free for the robots. An unnavigable layout is still valid; it just scores poorly.

| Id | Floor | Robots | Minimum shelves (fill) | Return [max / min] |
|---|---|---|---|---|
| `Warehouse-Small` | 10x10 | 2 | 25 (~25%) | +1000 / 0 |
| `Warehouse-Medium` | 16x16 | 4 | 115 (~45%) | +1000 / 0 |
| `Warehouse-Congested` | 20x22 | 8 | 264 (~60%) | +1000 / 0 |

The maximum is a ceiling no policy reaches, since every delivery takes travel time.

The base layouts use two-column rack blocks with vertical aisles, an open delivery row and clear cells above the goals, so every base shelf has a clear route to a goal. Your designs don't have to follow this.

`render(env_idx=i)` draws a single copy.
