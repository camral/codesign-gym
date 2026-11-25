# Examples

The examples that reset or step are marked to run on a GPU. On a CPU, a reset takes several minutes (the 0.2 s settling is 800 implicit physics steps) and a step takes seconds.
Building the environment, reading its spaces and validating designs is fast anywhere.

## Compare the models

All five models share the task and the per-cell design layout; they differ in grid dimension, channel count and the size of the state they expose:

```python
from codesign_gym import make

for model in ["Spring2D", "MPM2D", "MPM3D", "Beam3D", "HexFEM3D"]:
    env, base = make(f"SoftWalker-{model}")
    print(f"{model:9s} grid={env.embodiment_space.grid!s:10s} design={base.size:5d} "
          f"obs={env.observation_space.shape[0]:6d} channels={env.action_space.shape[0]:2d} base valid={env.validate_embodiment(base)}")
    env.close()
```

```text
Spring2D  grid=(8, 6)     design=  432 obs=   642 channels= 4 base valid=True
MPM2D     grid=(8, 6)     design=  432 obs=  2316 channels= 4 base valid=True
MPM3D     grid=(8, 6, 6)  design= 2880 obs= 55332 channels=16 base valid=True
Beam3D    grid=(8, 6, 6)  design= 2880 obs=  5220 channels=16 base valid=True
HexFEM3D  grid=(8, 6, 6)  design= 2880 obs=  7974 channels=16 base valid=True
```

## Screen a batch of designs

A design optimizer usually proposes many bodies at once. With `make_vec`, `validate_embodiment` takes one design per copy and returns one flag per copy, so impossible bodies can be dropped before any simulation runs.
Here three random occupancy patterns are checked alongside a two-legged body (the same body as in [Design space](embodiment.md)):

```python
import numpy as np
from codesign_gym import make_vec

F = {"occupancy": 0, "young": 1, "direction": slice(4, 6), "channel": 7}    # 2D field indices


def two_legged(base, grid):
    """Two legs on channels 0 and 1, a torso on channel 2 with fibres along x."""
    cells = base.reshape(*grid, -1).copy()
    cells[2:6, 0:2, F["occupancy"]] = 0
    cells[0:2, 0:2, F["channel"]] = 0
    cells[6:8, 0:2, F["channel"]] = 1
    cells[:, 2:6, F["channel"]] = 2
    cells[:, 2:6, F["direction"]] = [1.0, 0.0]
    return cells.reshape(-1)


envs, base = make_vec("SoftWalker-MPM2D", 4)
grid = envs.embodiment_space.grid
rng = np.random.default_rng(0)
designs = [two_legged(base, grid)]
for _ in range(3):
    cells = base.reshape(*grid, -1).copy()
    cells[..., F["occupancy"]] = rng.random(grid) < 0.7                     # keep about 70% of the cells
    designs.append(cells.reshape(-1))
print(envs.validate_embodiment(np.stack(designs)))
envs.close()
```

```text
[ True False False False]
```

Random carving almost always leaves islands or cells joined only at a corner, so all three random bodies are flagged `False`. See [What is rejected](embodiment.md#what-is-rejected) for the full list.

## Walk with an open-loop gait

Drive the two-legged body with sine waves, the legs half a cycle apart, and compare it with the base block. The 2.5 Hz gait gives six cycles per episode.
The reward already accounts for energy, so the return is the travel in body lengths minus the work penalty:

```python
# notest: each reset settles for 800 physics steps; run on a GPU
envs, base = make_vec("SoftWalker-MPM2D", 2)
designs = np.stack([base, two_legged(base, envs.embodiment_space.grid)])
obs, rejected, info = envs.reset(seed=0, embodiment=designs)
assert not rejected.any(), info.get("embodiment_errors")

phase = np.array([0.0, np.pi, 0.5 * np.pi, 0.0])                            # channels 0-3
returns = np.zeros(2)
for t in range(240):
    action = np.sin(2 * np.pi * 2.5 * t * 0.01 + phase)
    obs, reward, terminated, truncated, info = envs.step(np.stack([action, action]))
    returns += reward
    if (terminated | truncated).any():
        break
for name, ret, travel in zip(["base block", "two legs"], returns, info["metrics"]["displacement"]):
    print(f"{name:10s} return {ret:+.4f}  travel {100 * travel:+.2f} cm")
envs.close()
```

Output on an RTX PRO 6000:

```text
base block return -1.1511  travel -0.00 cm
two legs   return -1.6634  travel +1.48 cm
```

Both copies reach step 240 together, so the loop stops on truncation. The base MPM block travels nowhere by symmetry; the two-legged body can.
Channels a design does not use (channel 3 here, and 1–3 for the base block) are zeroed by the environment, so the same action can be sent to every copy.

## Record a video

With `render_mode="rgb_array"`, `render()` returns an 800 x 1280 RGB frame that follows the body. Without an OpenGL context it warns once and returns a 420 x 720 side view instead.
These frames are saved as a GIF with Pillow:

```python
# notest: needs a full episode; run on a GPU
from PIL import Image
from codesign_gym import make

env, base = make("SoftWalker-MPM2DStairs", render_mode="rgb_array")
env.reset(seed=0, embodiment=two_legged(base, env.embodiment_space.grid))
frames = [env.render()]
for t in range(240):
    action = np.sin(2 * np.pi * 2.5 * t * 0.01 + phase)
    obs, reward, terminated, truncated, info = env.step(action)
    if t % 4 == 0:
        frames.append(env.render())
    if terminated or truncated:
        break
images = [Image.fromarray(f) for f in frames]
images[0].save("walker.gif", save_all=True, append_images=images[1:], duration=40, loop=0)
env.close()
```

With `make_vec`, `render()` returns one frame per copy, and `render(env_idx=i)` returns copy `i`'s frame.
`render_mode="human"` opens a live viewer instead.
