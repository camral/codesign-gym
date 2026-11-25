# Shape-shifting Hand: Rotate

Id: `RotateInHand-Cube`. Engine: MuJoCo Warp; the `morphhand` extra (`pip install -e '.[morphhand]'`) installs a fork with the heightfield collision fixes this env needs.

A humanoid hand turns an object through a series of goal orientations. Its palm and fingers carry 16 heightfield pads.

- **Design:** the height of each pad cell (245 values in [0, 1]), plus finger spacing, finger width and thumb width (3 values).
- **Control:** the 19 hand actuators.
- **Reward:** potential-based shaping for closing in on the goal (turning away costs what turning back gives), given only while the hand touches the object, plus a bonus for holding a goal.
- **Design changes in `step`:** supported. `step(action, embodiment)` reshapes the hand from that step on, and an autoreset keeps the latest design. Evaluation envs ignore the change with a warning. The policy does not see the design.

| Id | Task | Return per goal [max] |
|---|---|---|
| `RotateInHand-Cube` | Turn a box through goals 1 rad apart, each starting where the last was met. 7.5 s per goal, 15 s per episode. | +10 (negative if the cube ends further from the goal than it started) |

Without the `morphhand` extra the env runs on upstream mujoco-warp, whose heightfield collision drops pad contacts and reports bogus deep ones for the box.
