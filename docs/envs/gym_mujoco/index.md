# Gym MuJoCo ports

The seven classic Gymnasium MuJoCo tasks, with a body you can reshape. Each port keeps the v4 observation, reward,
termination and episode length, and adds a design: a vector that moves the far ends of the robot's links, making them
longer, shorter or pointed in a new direction. Mass and inertia follow from the new shape.

The physics runs on [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp), batched over all copies on one
device. The v4 observation, reward and termination code is reimplemented in batched PyTorch, so observations and
rewards come back as `torch.Tensor`s on that device.

## At a glance

| Id | Design | Observation | Action | Control step | Early termination | Solver |
|---|---|---|---|---|---|---|
| `HalfCheetah-v4` | 16 | 17 | 6 | 0.05 s | No | Newton |
| `Swimmer-v4` | 6 | 8 | 2 | 0.04 s | No | Newton |
| `Hopper-v4` | 8 | 11 | 3 | 0.008 s | Yes | Newton |
| `Walker2d-v4` | 12 | 17 | 6 | 0.008 s | Yes | Newton |
| `Ant-v4` | 36 | 27 | 8 | 0.05 s | Yes | Newton |
| `Humanoid-v4` | 28 | 376 | 17 | 0.015 s | Yes | Newton (stock: PGS) |
| `HumanoidStandup-v4` | 28 | 376 | 17 | 0.015 s | No | Newton (stock: PGS) |

Every space is a float32 `Box`. Designs and actions are bounded to `[-1, 1]`; observations are unbounded. Every episode
is truncated at 1000 control steps.

## Shared behaviour

- **Design changes only at reset.** The body is rebuilt when a copy resets. A design passed to `step()` is dropped with a
  `UserWarning` ("Transition-level embodiment changes are not permitted").
- **The policy does not see the design.** The observation is exactly the v4 one, with no design block.
- **Every design in range is valid.** `validate_embodiment` checks shape, finiteness and bounds only. A design outside
  `[-1, 1]` makes `reset()` raise `ValueError`.
- **Rejected resets.** `reset()` returns `rejected` per copy. It is `True` only when a copy's state is not finite after
  the reset, and stepping then raises `RuntimeError` until a full reset succeeds.
- **Autoreset.** Gymnasium's `NEXT_STEP` mode: a copy that ends (`terminated` or `truncated`) is respawned with its last
  design on the next `step()` call, which ignores its action and returns its first observation, with reward 0 and
  `terminated` and `truncated` both `False`. When any copy ends, `info` also holds
  `final_obs`, `final_return` and `final_steps` (in control steps) for every copy; read them where
  `terminated | truncated` is `True`.
- **Partial reset.** `reset(options={"reset_mask": mask})` resets only the copies where `mask` is `True`, after one
  full reset. A copy waiting to autoreset that is not in the mask still respawns on the next step.
- **Rendering.** `render_mode="rgb_array"` returns 480 × 480 RGB frames: `render()` gives a tuple with one frame per
  copy, `render(env_idx=i)` one frame. Each frame draws that copy's own design. `render_mode="human"` opens MuJoCo's
  passive viewer and needs a display.
- **Device.** By default the env runs on a CUDA device if one is visible, otherwise on the CPU. Pass `device="cpu"` or
  `device="cuda:N"` to choose; an explicit GPU that is not available is an error. The first build on a machine compiles
  Warp kernels, which takes a while; later builds use the cache. Warp's CPU backend is slow; Walker2d also takes
  `physics="native"`, which steps the CPU physics with C MuJoCo instead (see [Walker2d](walker2d.md)).
- **Fidelity.** With the base design, Swimmer, Ant, Hopper and Walker2d reproduce the stock Gymnasium model closely;
  HalfCheetah re-authors its leg geometry and Humanoid uses the Newton solver and re-anchored forearms, so they match
  within looser tolerances. `tests/test_gym_fidelity.py` steps six ports (all but HumanoidStandup) next to stock
  Gymnasium with the same controls and checks the observations against recorded per-env tolerances; rewards are
  checked in separate targeted tests.

A minimal loop:

```python
import numpy as np
from codesign_gym import make

env, base = make("Hopper-v4", device="cpu")
obs, rejected, info = env.reset(seed=0, embodiment=base)
total = 0.0
for _ in range(50):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    total += reward
    if terminated or truncated:
        break
print(obs.shape, f"return {total:.1f}")    # torch.Size([11]) return ...
env.close()
```

```{toctree}
:maxdepth: 1

embodiment
examples
half_cheetah
swimmer
hopper
walker2d
ant
humanoid
humanoid_standup
```
