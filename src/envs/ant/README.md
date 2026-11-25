# Ant

Id: `Ant-v4`. Engine: MuJoCo Warp.

A 3D quadruped. This is Gymnasium's `Ant-v4` with a body you can reshape.

- **Design:** 36 values: the 3D length and direction of all twelve leg links (hip, thigh and ankle on each leg, all independent). Mass and inertia follow from the shape, so a longer limb reaches further but weighs more. Every design in range is valid.
- **Control:** the eight leg joints.
- **Reward:** forward velocity + healthy bonus − 0.5·‖action‖², with no contact-force term (as in v4's default). The episode ends when the torso height leaves 0.2–1.0 m.
- **Design changes in `step`:** not supported; the body is rebuilt at reset.

Returns are unbounded: a passive ant stays standing for about 1000, flailing can go negative, and a trained ant reaches the thousands.

Observations, reward and episode length match Gymnasium v4, and the base design (`base_embodiment = 0`) reproduces the stock model. The policy does not see the design. `tests/test_gym_fidelity.py` checks each port against stock Gymnasium.

Other ports: [HalfCheetah-v4](../half_cheetah/README.md), [Swimmer-v4](../swimmer/README.md), [Hopper-v4](../hopper/README.md), [Walker2d-v4](../walker2d/README.md), [Humanoid-v4](../humanoid/README.md), [HumanoidStandup-v4](../humanoid_standup/README.md).
