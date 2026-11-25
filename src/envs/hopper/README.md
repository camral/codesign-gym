# Hopper

Id: `Hopper-v4`. Engine: MuJoCo Warp.

A planar one-legged hopper. This is Gymnasium's `Hopper-v4` with a body you can reshape.

- **Design:** 8 values: the length and in-plane angle of the thigh and leg, and the length of the upper and lower torso, toe and heel. Mass and inertia follow from the shape, so a longer limb reaches further but weighs more. Every design in range is valid.
- **Control:** the thigh, leg and foot joints.
- **Reward:** forward velocity + 1 per healthy step − 1e-3·‖action‖². The episode ends when the torso leaves its healthy height or angle range, which stays fixed whatever the design.
- **Design changes in `step`:** not supported; the body is rebuilt at reset.

Returns are unbounded: a passive policy falls after about 150 steps (return ≈ 150), and a trained hopper reaches the thousands.

Observations, reward and episode length match Gymnasium v4, and the base design (`base_embodiment = 0`) reproduces the stock model. The policy does not see the design. `tests/test_gym_fidelity.py` checks each port against stock Gymnasium.

Other ports: [HalfCheetah-v4](../half_cheetah/README.md), [Swimmer-v4](../swimmer/README.md), [Walker2d-v4](../walker2d/README.md), [Ant-v4](../ant/README.md), [Humanoid-v4](../humanoid/README.md), [HumanoidStandup-v4](../humanoid_standup/README.md).
