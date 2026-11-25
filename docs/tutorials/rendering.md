# Rendering

Choose a render mode when you create the environment, as in Gymnasium: `make(env_id, render_mode=...)`.
Without a render mode, `render()` returns `None`.

## Which environment supports what

| Environments | `rgb_array` | `human` | Other | `render(env_idx=i)` |
|---|---|---|---|---|
| MuJoCo: Ball Catcher, Solar Panel Cleaner, Truck Unloading Line, NeroGrasp, Shape-shifting Hand (Rotate, Pick Up), all seven Gym ports | Yes | MuJoCo's interactive viewer, showing one copy | — | Yes |
| Soft Walker | Yes (3D view; 2D side view without OpenGL) | Newton's live viewer, showing one copy | — | Yes |
| Warehouse | Yes | Jumanji's viewer, showing one copy | — | Yes |
| Racing | Yes | A matplotlib window showing one copy | `web` | Yes |
| LQ Structure, Microgrid, Wireless Network | Yes | A window for each copy | — | No |
| Pokémon | No | No | `html` | No |

Notes:

- **Which copy `human` shows.** MuJoCo environments show copy `viz_env_id` (default 0; set it with `make_vec(..., viz_env_id=i)`) or the copy passed as `render(env_idx=i)`. Soft Walker, Warehouse and Racing show copy 0 or `render(env_idx=i)`.
- **When `human` redraws.** MuJoCo, Soft Walker, Warehouse and Racing refresh the window on every `reset` and `step`. LQ Structure, Microgrid and Wireless Network redraw when you call `render()`.
- **Before the first frame.** `render()` returns `None` before the first `reset`; Microgrid returns `None` until the first `step`.
- **Headless machines.** `human` needs a display. For `rgb_array`, MuJoCo needs `MUJOCO_GL=egl`, and Soft Walker's 3D view needs EGL (it warns once and falls back to the 2D side view without it).

## Images

With `render_mode="rgb_array"`, `render()` returns an RGB image as a NumPy array:

```python
from codesign_gym import make

env, base = make("LQStructure-LQR", render_mode="rgb_array")
env.reset(seed=0, embodiment=base)
frame = env.render()
print(frame.shape, frame.dtype)        # (height, width, 3) uint8
env.close()
```

A vector environment returns a tuple with one image per copy. Environments with `env_idx` support (see the table) also take `render(env_idx=i)` to draw one copy only, which is much cheaper with many copies.

To save a video, collect frames and write them with any video library, for example `imageio`:

```python
import imageio.v2 as imageio                # pip install imageio  # notest
from codesign_gym import make

env, base = make("LQStructure-LQR", render_mode="rgb_array")
env.reset(seed=0, embodiment=base)
frames = [env.render()]
for _ in range(100):
    env.step(env.action_space.sample())
    frames.append(env.render())
imageio.mimsave("lq.gif", frames, duration=0.05)
env.close()
```

## One copy of a vector environment

```python
from codesign_gym import make_vec

envs, base = make_vec("Hopper-v4", 2, device="cpu", render_mode="rgb_array")
envs.reset(seed=0)
frames = envs.render()                 # one frame per copy
frame = envs.render(env_idx=1)         # copy 1 only
print(len(frames), frame.shape)        # 2 (480, 480, 3)
envs.close()
```

## Pokémon replays (`html`)

Pokémon battles are turn-based, so there are no image frames. With `render_mode="html"`, `render()` returns the replay of the last finished battle as an HTML string (one string or `None` per copy with `make_vec`). It stays available until a newer battle finishes or you call `reset()`. Save it to a `.html` file and open it in a browser. See the [Pokémon examples](../envs/pokenv/examples.md).

## Racing dashboard (`web`)

With `render_mode="web"`, Racing records the whole run instead of drawing frames, and `render()` returns `None`. Call `save_dashboard()` to write an interactive dashboard of the recorded run. See the [Racing examples](../envs/f1tenth/examples.md).
