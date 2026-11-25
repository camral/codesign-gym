# HalfCheetah

Id: `HalfCheetah-v4`. Engine: MuJoCo Warp.

A planar two-legged runner. This is Gymnasium's `HalfCheetah-v4` with a body you can reshape.

- **Design:** 16 values: the length and in-plane angle of the six leg segments and the head, and the length of the front and rear torso halves. Mass and inertia follow from the shape, so a longer limb reaches further but weighs more. Every design in range is valid.
- **Control:** the six leg joints.
- **Reward:** forward velocity − 0.1·‖action‖² over 1000 steps.
- **Design changes in `step`:** not supported; the body is rebuilt at reset.

Returns are unbounded: an untrained policy scores about 0, a random one about −300 to −450, and a trained runner reaches the thousands.

Observations, reward and episode length match Gymnasium v4, and the base design (`base_embodiment = 0`) approximates the stock model (see the fidelity note in the Gym MuJoCo docs). The policy does not see the design. `tests/test_gym_fidelity.py` checks each port against stock Gymnasium.

Other ports: [Swimmer-v4](../swimmer/README.md), [Hopper-v4](../hopper/README.md), [Walker2d-v4](../walker2d/README.md), [Ant-v4](../ant/README.md), [Humanoid-v4](../humanoid/README.md), [HumanoidStandup-v4](../humanoid_standup/README.md).
