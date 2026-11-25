# HumanoidStandup

Id: `HumanoidStandup-v4`. Engine: MuJoCo Warp.

The Humanoid body, starting flat on its back, has to stand up. This is Gymnasium's `HumanoidStandup-v4` with a body you can reshape.

- **Design:** the same 28 links as [Humanoid](../humanoid/README.md), but the body is built lying down, so the leg-length slots are on a different axis. A Humanoid design gives a different body here. Mass and inertia follow from the shape, so a longer limb reaches further but weighs more. Every design in range is valid.
- **Control:** the 17 joints.
- **Reward:** torso height ÷ timestep − 0.1·‖ctrl‖² − a capped contact penalty + 1. There is no early termination; every episode runs 1000 steps.
- **Design changes in `step`:** not supported; the body is rebuilt at reset.

Returns are unbounded: a passive policy scores about 30,000, and one that stands up reaches the hundreds of thousands.

Observations, reward and episode length match Gymnasium v4, and the base design (`base_embodiment = 0`) approximates the stock model (see the fidelity note in the Gym MuJoCo docs). The policy does not see the design. `tests/test_gym_fidelity.py` checks each port against stock Gymnasium.

**Solver note.** The body starts in full ground contact, and Newton (MuJoCo Warp has no PGS) resolves those contacts differently from stock. State and reward track stock closely, but the contact-force part of the observation differs.

Other ports: [HalfCheetah-v4](../half_cheetah/README.md), [Swimmer-v4](../swimmer/README.md), [Hopper-v4](../hopper/README.md), [Walker2d-v4](../walker2d/README.md), [Ant-v4](../ant/README.md), [Humanoid-v4](../humanoid/README.md).
