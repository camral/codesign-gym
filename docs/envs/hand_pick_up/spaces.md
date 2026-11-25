# Observations, actions and rewards

The hand hangs palm-down over a table, with its wrist free to roll, pitch and yaw; for the hammers it also slides in x, y and z (`hand.xml`, against the Cube's `hand_no_xyz.xml`). The object rests on the table under the middle of the palm at every reset, and the hand must lift it before turning it is rewarded.
The Cube's goals are in the palm's frame, as in [Rotate](../rotate_in_hand/spaces.md). The hammers' goal is fixed in the world: an orientation and a place, which stay put whatever the hand does.

| Id | Object | Table top below the palm origin | Goal |
|---|---|---|---|
| `HandPickUp-Cube` | 8 cm cube, 0.1 kg | derived from the cube's size (174 mm, logged at build at INFO level) | A stream of palm-frame orientations, each 0.5 rad from the cube's orientation when the previous goal was met |
| `HandPickUp-Hammer` | Hammer: 28 cm × 4 cm handle, 12 cm × 6 cm head, 0.45 kg | 145 mm | One world orientation (the hammer tilted 15° about y) **and** place: its origin 7 cm above where the palm origin starts, within 5 cm |
| `HandPickUp-SmallHammer` | Slimmer hammer: 28 cm × 2–3 cm handle, 10.5 cm × 2.5 cm head, 0.43 kg | 145 mm | As the Hammer |

The hammers' target place is given as `rel_pos = (0, 0, -0.07)` m in the palm's frame and fixed in the world where it lies at the palm's reset pose (`goal_frame="world"`).
A random horizontal push of 3 N acts on the hammers for 5 of every 20 control steps.

## Observation

A torch float32 tensor: the [Rotate observation](../rotate_in_hand/spaces.md#observation), then 4 more values. With `d` the number of hand joints (19, or 22 for the hammers), the layout is:

| Slice | Content |
|---|---|
| `[0:d]` | Normalised joint positions (hammers: `x, y, z` first; then wrist `roll, pitch, yaw`, then ring, middle, index, thumb) |
| `[d:2d]` | Joint velocities (rad/s, m/s for the slides) |
| `[2d:2d+3]` | Object position in the palm frame, in units of 10 cm |
| `[2d+3:2d+7]` | Object orientation in the palm frame, quaternion `w, x, y, z` |
| `[2d+7:2d+13]` | Object linear and angular velocity relative to the palm |
| `[2d+13:2d+17]` | Goal orientation in the palm frame |
| `[2d+17]` | Angle to the goal, divided by π |
| `[2d+18]` | 1 if the hand touched the object in either of the last 2 control steps |
| `[2d+19]` | 1 if the table touched the object in either of the last 2 control steps |
| `[2d+20:2d+23]` | Target position in the palm frame, in units of 10 cm (constant `rel_pos` for a palm-frame goal) |

That is `Box(61,)` for the Cube and `Box(67,)` for the hammers. The target position is there in every preset, so the observation's size does not depend on the goal settings. The design is not observed.

## Action

As in Rotate: one entry per velocity actuator in `[-1, 1]`, in the order wrist `pitch, yaw, roll`, index, middle, ring, thumb, each a joint velocity target of `2 a` rad/s.
The hammers' hand has three more entries first, `x, y, z`, each a slide velocity target of `0.15 a` m/s, so its action is `Box(22,)`. A control step is 0.04 s.

## Reward

The grip multiplier is 1 when the hand touches the object and the table does not, 0.5 when both do, and 0 when the hand does not.

| Term | Value |
|---|---|
| Fingertip | Potential-based: each step, how far the summed fingertip-to-object distance fell, scaled so closing all of it gives 1 per episode; moving away costs the same. Given from the respawn, touching or not |
| Shaping | As in Rotate (potential-based, 3 per goal, gains times the grip multiplier, losses in full), but only from the first time the hand holds the object clear of the table |
| Goal | Cube: 6 per goal, given as 0.3 for each of 20 steps (not necessarily in a row) spent within 0.05 rad of the goal while gripped off the table. Hammers: 6, given as 0.08 for each of 75 such steps, within 0.05 rad and 5 cm |

For the hammers the shaping is split: 1.5 for the orientation and 1.5 for the distance to the target place.
A goal met is worth 9 (Cube, plus at most 1 per episode from the fingertip term) or 10 in total (hammers). Losing ground costs reward, so returns can be negative.

## Episode

| Ends with | When |
|---|---|
| `terminated` | The object is more than 0.4 m from the palm, horizontally |
| `terminated` | 6 s pass without meeting the current goal (Cube) |
| `terminated` | The goal is met (hammers) |
| `terminated` | The physics diverges |
| `truncated` | 15 s (375 steps) pass for the Cube, 7 s (175 steps) for the hammers |

Autoreset, partial reset, seeding, the reset rejection flag and the `info` keys are as in [Rotate](../rotate_in_hand/spaces.md#episode): a copy that ends is respawned on the next `step`, which ignores its action and returns its first observation with reward 0 (Gymnasium's `NEXT_STEP`).
