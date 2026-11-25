# Humanoid

Id: `Humanoid-v4`. Engine: MuJoCo Warp.

A 3D biped. This is Gymnasium's `Humanoid-v4` with a body you can reshape.

- **Design:** 28 values: the 3D length and direction of the thighs, shins, upper arms and forearms, plus the foot and hand radii, left and right independent. The torso and waist stay fixed. Mass and inertia follow from the shape, so a longer limb reaches further but weighs more. Every design in range is valid.
- **Control:** the 17 joints.
- **Reward:** COM forward velocity × 1.25 + 5 per healthy step − 0.1·‖ctrl‖². The episode ends when the torso height leaves 1.0–2.0 m.
- **Design changes in `step`:** not supported; the body is rebuilt at reset.

Returns are unbounded: a passive humanoid falls after about 40 steps (as in stock Humanoid-v4), and a trained one reaches the thousands.

Observations, reward and episode length match Gymnasium v4, and the base design (`base_embodiment = 0`) approximates the stock model (see the fidelity note in the Gym MuJoCo docs). The policy does not see the design. `tests/test_gym_fidelity.py` checks each port against stock Gymnasium.

**Solver note.** Stock Humanoid uses the PGS solver, which MuJoCo Warp lacks, so this port uses Newton. Observations, reward and termination match v4, but contact dynamics differ.

Other ports: [HalfCheetah-v4](../half_cheetah/README.md), [Swimmer-v4](../swimmer/README.md), [Hopper-v4](../hopper/README.md), [Walker2d-v4](../walker2d/README.md), [Ant-v4](../ant/README.md), [HumanoidStandup-v4](../humanoid_standup/README.md).
