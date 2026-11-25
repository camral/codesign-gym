# Walker2d

Id: `Walker2d-v4`. Engine: MuJoCo Warp.

A planar two-legged walker. This is Gymnasium's `Walker2d-v4` with a body you can reshape.

- **Design:** 12 values: the length and in-plane angle of the thigh and leg of each leg, and the length of each foot and of the upper and lower torso (left and right independent). Mass and inertia follow from the shape, so a longer limb reaches further but weighs more. Every design in range is valid.
- **Control:** all six leg joints.
- **Reward:** forward velocity + healthy bonus − 1e-3·‖action‖². The episode ends when the torso leaves 0.8–2.0 m height or ±1 rad, whatever the design.
- **Design changes in `step`:** not supported; the body is rebuilt at reset.

Returns are unbounded: a passive policy falls after about 100 steps, and a trained walker reaches the thousands.

Observations, reward and episode length match Gymnasium v4, and the base design (`base_embodiment = 0`) reproduces the stock model. The policy does not see the design. `tests/test_gym_fidelity.py` checks each port against stock Gymnasium.

Other ports: [HalfCheetah-v4](../half_cheetah/README.md), [Swimmer-v4](../swimmer/README.md), [Hopper-v4](../hopper/README.md), [Ant-v4](../ant/README.md), [Humanoid-v4](../humanoid/README.md), [HumanoidStandup-v4](../humanoid_standup/README.md).
