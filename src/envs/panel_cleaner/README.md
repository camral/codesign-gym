# Solar Panel Cleaner

Ids: `SolarCleaner-Reach`, `SolarCleaner-Stability`, `SolarCleaner-Traverse`. Engine: MuJoCo Warp.

A four-legged robot with a roller on its arm walks up to an array of solar panels and rolls over each cell to clean it.

- **Design:** the shape of every leg and arm link, each set independently.
- **Control:** walking to the panels and sweeping the roller over them.
- **Reward:** a cell counts as clean once the roller has stayed on it long enough. A full sweep earns at most 10, split over the cells. The approach term can go slightly negative when the robot moves away from its goal. If the robot falls, its body sinks below 75% of its standing height, or it fouls a panel, the episode ends and it keeps what it has earned.
- **Design changes in `step`:** not supported. The policy does not see the design.

| Id | What makes it hard | Return [max / min] |
|---|---|---|
| `SolarCleaner-Reach` | The arm: one row of tall, distant panels that only a long, well-angled arm can reach. No disturbances. | +10 / ≈0 |
| `SolarCleaner-Stability` | The legs: panels on rolling terrain, with small pushes every step and occasional side shoves while the roller cleans. | +10 / ≈0 |
| `SolarCleaner-Traverse` | Careful walking: tightly packed, solid panels, and the episode ends if anything but the roller touches one. | +10 / ≈0 |

The evaluation env renders with a camera that follows the robot.
