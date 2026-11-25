# Examples

These examples run on the CPU with a few copies and short rollouts. On a GPU, drop `device="cpu"` and raise `num_envs`;
the physics of all copies runs in one batch.

## Evaluate several designs at once

Each copy of a `make_vec` env can carry its own design. Pass one row per copy to `reset()`, run until every copy has
finished its first episode, and read each copy's return from `info["final_return"]` on the step it ends:

```python
import numpy as np
from codesign_gym import make_vec

envs, base = make_vec("Walker2d-v4", 4, device="cpu")
rng = np.random.default_rng(0)
designs = rng.uniform(-1, 1, size=(4, base.size)).astype(np.float32)
designs[0] = base                                   # keep the stock walker as a reference
obs, rejected, info = envs.reset(seed=0, embodiment=designs)

episode_return = [None] * 4
while None in episode_return:
    obs, reward, terminated, truncated, info = envs.step(np.zeros((4, 6), dtype=np.float32))
    done = (terminated | truncated).tolist()
    for i in range(4):
        if done[i] and episode_return[i] is None:
            episode_return[i] = float(info["final_return"][i])
for i, value in enumerate(episode_return):
    print(f"design {i}: return {value:.1f}")
envs.close()
```

A copy that ends is respawned with the same design on the next step, so the loop can keep stepping all four while it
waits for the slowest one. `final_return` holds a value for every copy whenever any copy ends; only the entries where
`terminated | truncated` is `True` are finished episodes.

## Sweep one slot

Hold every slot at its base value except one, and spread that one over its range, one value per copy. Here the
front thigh of the HalfCheetah (slot 11, which moves its far end straight down, so it changes both its length and its angle) runs the same open-loop gait at five lengths:

```python
import numpy as np
from codesign_gym import make_vec

values = np.linspace(-1.0, 1.0, 5)
envs, base = make_vec("HalfCheetah-v4", len(values), device="cpu")
designs = np.tile(base, (len(values), 1))
designs[:, 11] = values
obs, rejected, info = envs.reset(seed=0, embodiment=designs)

phase = np.array([0.0, 1.0, 2.0, np.pi, np.pi + 1.0, np.pi + 2.0])
returns = np.zeros(len(values))
for t in range(100):
    action = np.sin(0.5 * t + phase).astype(np.float32)
    obs, reward, terminated, truncated, info = envs.step(np.tile(action, (len(values), 1)))
    returns += reward.cpu().numpy()
for value, ret in zip(values, returns):
    print(f"slot 11 = {value:+.1f}: return {ret:7.1f}")
envs.close()
```

The same pattern gives a grid over two slots (one copy per grid point), or a population for an evolutionary design
search: each generation is one `reset()` with a new `(num_envs, n)` design array.

## Render one copy

With `render_mode="rgb_array"`, `render(env_idx=i)` returns copy `i` as a 480 × 480 RGB array, drawn with that copy's
own design; `render()` without an index returns a tuple with one frame per copy. Rendering needs an OpenGL context: on a
headless Linux machine the env selects EGL by itself, and `MUJOCO_GL` overrides the choice.

```python
import numpy as np
from codesign_gym import make_vec

envs, base = make_vec("Hopper-v4", 2, device="cpu", render_mode="rgb_array")
long_legs = base.copy()
long_legs[[3, 5]] = 1.0
obs, rejected, info = envs.reset(seed=0, embodiment=np.stack([base, long_legs]))

frames = []
for _ in range(30):
    obs, reward, terminated, truncated, info = envs.step(np.zeros((2, 3), dtype=np.float32))
    frames.append(envs.render(env_idx=1))
print(len(frames), frames[0].shape, frames[0].dtype)    # 30 (480, 480, 3) uint8
envs.close()
```

To keep a video, write `frames` with any video library, for example `imageio.mimsave("hopper.mp4", frames, fps=30)`.
