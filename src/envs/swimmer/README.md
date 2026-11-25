# Swimmer

Id: `Swimmer-v4`. Engine: MuJoCo Warp.

A three-link swimmer pushing through fluid. This is Gymnasium's `Swimmer-v4` with a body you can reshape.

- **Design:** 6 values: the length and in-plane angle of all three links. Mass and inertia follow from the shape, so a longer limb reaches further but weighs more. Every design in range is valid.
- **Control:** the two joints between the links.
- **Reward:** forward velocity − 1e-4·‖action‖² over 1000 steps.
- **Design changes in `step`:** not supported; the body is rebuilt at reset.

Returns are unbounded: an untrained policy scores about −20 to +20, a simple sine gait about +25 to +30, and a trained swimmer reaches the hundreds.

Observations, reward and episode length match Gymnasium v4, and the base design (`base_embodiment = 0`) reproduces the stock model. The policy does not see the design. `tests/test_gym_fidelity.py` checks each port against stock Gymnasium.

Other ports: [HalfCheetah-v4](../half_cheetah/README.md), [Hopper-v4](../hopper/README.md), [Walker2d-v4](../walker2d/README.md), [Ant-v4](../ant/README.md), [Humanoid-v4](../humanoid/README.md), [HumanoidStandup-v4](../humanoid_standup/README.md).
