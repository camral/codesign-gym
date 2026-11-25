# Shape-shifting Hand: Pick Up

Ids: `HandPickUp-Cube`, `HandPickUp-Hammer`, `HandPickUp-SmallHammer`. Engine: MuJoCo Warp; the `morphhand` extra (`pip install -e '.[morphhand]'`) installs a fork with the heightfield collision fixes these envs need.

The hand from [Rotate](../rotate_in_hand/README.md), hung palm-down over a table. The object starts on the table, and the hand has to lift it before turning it is rewarded.

- **Design:** the same pad heights, finger spacing and finger widths as Rotate.
- **Control:** the 19 hand actuators; the hammer presets' hand also slides its wrist in x, y and z (22 actuators).
- **Reward:** a goal counts only while the object is held clear of the table, and the shaping starts at the first lift. A fingertip term rewards closing in on the object. Both shaping terms are potential-based, so losing ground costs what regaining it gives. The object is lost once it is 0.4 m from the palm horizontally.
- **Design changes in `step`:** supported, as in Rotate. The policy does not see the design.

| Id | Task | Return [max] |
|---|---|---|
| `HandPickUp-Cube` | Lift a box, then turn it through palm-frame goals 0.5 rad apart. 6 s per goal, 15 s per episode. | +9 per goal, +1 fingertip |
| `HandPickUp-Hammer` | Lift a hammer and hold it at one world-fixed orientation and place while liftoff-gated random 3D torques act on it; meeting the goal ends the episode. 9 s per episode (5 s dwell). | +10 |
| `HandPickUp-SmallHammer` | The same with a slimmer hammer. | +10 |

Returns can be negative: losing ground on the goal or the object costs reward.
Without the `morphhand` extra the envs run on upstream mujoco-warp, whose heightfield collision drops pad contacts and reports bogus deep ones for box objects.
