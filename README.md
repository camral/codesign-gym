# Codesign Gym

Codesign Gym is a set of co-design benchmarks with a Gymnasium-like API.
In co-design, you optimize two things together: the **design** of the agent (its body, layout or team) and its **control policy**.
The API calls the design the *embodiment*.

Contributors: see [CONTRIBUTORS.md](CONTRIBUTORS.md).

## Basic Install

### Setup uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
mkdir ~/code/cgym/
cd ~/code/cgym/
uv venv .venv --python 3.12
source .venv/bin/activate
```

### Clone Repo and Install

```bash
git clone https://github.com/camral/codesign-gym
# for SSH: git clone git@github.com:camral/codesign-gym.git
cd codesign-gym
uv pip install -e .
```

The base install uses MuJoCo 4.0.1 and a pinned Newton development revision.
See the [dependency upgrade record](docs/newton-dependency-upgrade.md) for tested
versions, regression commands, and the separate legacy notebook/fork setup.

⚠️🚫🐍 **Instructions may not work with Conda, so do not use it.**

See [additional-install-steps](https://github.com/camral/codesign-gym/blob/main/README.md#additional-install-steps) below for per-environment installation information.

## Quick-start Notebook Tutorials

You can find an end-to-end notebook at [`notebooks/rl-setup.ipynb`](notebooks/rl-setup.ipynb), a notebook covering setting up codesign-baselines and the MorphHands at [`notebooks/morphhand_tutorial.ipynb`](notebooks/morphhand_tutorial.ipynb), and a notebook on co-designed fingertips for tool use (NeroGrasp-Tools) at [`notebooks/nerotools_tutorial.ipynb`](notebooks/nerotools_tutorial.ipynb).

## Documentation

The full documentation is in [`docs/`](docs/index.md):

- [Getting started](docs/getting-started.md): install and first steps.
- [Environments](docs/envs/index.md): every environment family, with its design space (layout, bounds, validity rules and how to build a design), observations, actions, rewards, episode rules and runnable examples.
- [API](docs/api.md) and [concepts](docs/concepts.md): the co-design interface.
- [Adding an environment](docs/adding-an-environment.md).

## Environments

**Native** (made for co-design)

| Environment | Ids | Design | Control |
|---|---|---|---|
| [Ball Catcher](src/envs/ball_catcher/README.md) | `BallCatcher-Pitch`, `-Arc`, `-Fielding` | Arm link geometry | Catch a thrown ball |
| [Solar Panel Cleaner](src/envs/panel_cleaner/README.md) | `SolarCleaner-Reach`, `-Stability`, `-Traverse` | Leg and arm link geometry | Walk to panels and clean them |
| [Truck Unloading Line](src/envs/truck_unload/README.md) | `TruckUnload-Single`, `-Multi` | Column and arm lengths of each robot | Unload and stack boxes |
| [Grasping Small Objects](src/envs/nerograsp/README.md) | `NeroGrasp-<Object>` (13 ids) | Fingertip meshes | Grasp and lift small objects |
| [Shape-shifting Hand: Rotate](src/envs/rotate_in_hand/README.md) | `RotateInHand-Cube` | Pad heights, finger spacing and width | Turn an object to goal angles |
| [Shape-shifting Hand: Pick Up](src/envs/hand_pick_up/README.md) | `HandPickUp-Cube`, `-Hammer`, `-SmallHammer` | Pad heights, finger spacing and width | Lift and turn an object |
| [Soft Walker](src/envs/soft_walker/ENVIRONMENT.md) | `SoftWalker-<Model>[Stairs\|Gaps]` (15 ids) | Material and actuation of each cell | Walk forward |

**Reframed** (known RL tasks with a hidden design problem)

| Environment | Ids | Design | Control |
|---|---|---|---|
| [LQ Structure](src/envs/lq_structure/README.md) | `LQStructure-LQR`, `-LQG`, `-QP` | Masses, springs, dampers, sensors, actuators | Regulate the structure |
| [Microgrid](src/envs/microgrid/README.md) | `Microgrid-Random`, `-Forecast`, `-Campus`, `-OffGridSingle`, `-OffGridCampus` | Solar, battery and generator size | Dispatch energy |
| [Wireless Network](src/envs/mobile_env/README.md) | `Network-Urban`, `-Suburban`, `-Motorway` | Base station count, position and power | Connect users to stations |
| [Racing](src/envs/f1tenth/README.md) | `Racing-Spielberg`, `-Levine` | Tyres, power, weight distribution | Drive a lap |
| [Warehouse](src/envs/robot_warehouse/README.md) | `Warehouse-Small`, `-Medium`, `-Congested` | Shelf positions | Move robots to deliver shelves |
| [Pokémon](src/envs/pokenv/README.md) | `Pokenv-Random`, `-MaxPower`, `-Heuristic`, `-League*` (23 ids) | Team | Battle |

**Extended** (classic MuJoCo tasks with a body you can change)

| Environment | Ids | Design | Control |
|---|---|---|---|
| [Gym MuJoCo ports](src/envs/half_cheetah/README.md) | `HalfCheetah-v4`, `Swimmer-v4`, `Hopper-v4`, `Walker2d-v4`, `Ant-v4`, `Humanoid-v4`, `HumanoidStandup-v4` | Limb lengths and angles | Same as Gymnasium v4 |

## API

Basic usage example:

```python
from codesign_gym import make, make_vec, list_envs

print(list_envs())                                   # all environment ids

env, base = make("LQStructure-LQR")                  # base = the default design
design = env.embodiment_space.sample()
obs, rejected, info = env.reset(seed=0, embodiment=design)
if not rejected:
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
env.close()

envs, base = make_vec("LQStructure-LQR", 4)          # 4 parallel copies
obs, rejected, info = envs.reset(seed=0, embodiment=base)   # rejected.shape == (4,)
envs.close()
```

The API is Gymnasium with these changes:

| Item | Change |
|---|---|
| `make(env_id, ...)`, `make_vec(env_id, num_envs, ...)` | Return `(env, base_embodiment)`, where `base_embodiment` is the default design. |
| `env.embodiment_space` | The space of all designs, like `action_space`. |
| `env.reset(embodiment=..., seed=..., options=...)` | Returns `(obs, embodiment_rejected, info)`. |
| `env.step(action, embodiment=None)` | Changes the design mid-episode, where the environment supports it. |
| `env.validate_embodiment(design)` | Returns whether the design is possible (one bool per copy for a vector env). |
| `make(..., is_eval_env=True)` | Makes an evaluation env, where `step` ignores design changes. |

Rules:

- **Designs.** A vector env takes one design for all copies, a batch with one row per copy, or a list. A `None` in the list means the base design at reset and the current design at step.
- **Rejection.** Only impossible designs (out of bounds, wrong shape) are rejected. A bad design is still valid; it just earns a low reward.
- **After a rejection**, reset that copy with a valid design before you call `step`.
- **Design changes in `step`** work in LQ Structure, Microgrid, Racing and the Shape-shifting Hand. Other environments ignore them with a warning.
- **Seeds.** Pass an int or one int per copy. An int gives each copy its own independent stream, so any two different seeds give independent runs.
- **Partial reset.** `reset(options={"reset_mask": mask})` resets only the copies where `mask` is `True` (not in Pokémon or NeroGrasp).
- **Autoreset.** A finished copy resets on the next `step` and keeps its design. Pokémon and NeroGrasp differ; see their READMEs.
- **Spaces** have a fixed size for each environment id. Smaller designs are padded and masked.

Options for `make` and `make_vec`:

| Option | Use |
|---|---|
| `device="cpu"`, `"cuda"`, `"cuda:N"` | MuJoCo environments. The default is CUDA if available, else CPU. |
| `render_mode="rgb_array"` | `render()` returns an image (`make`) or one image for each copy (`make_vec`). MuJoCo, Racing, Warehouse and Soft Walker also accept `render(env_idx=i)` for one copy. |
| `render_mode="human"` | Opens a window (needs a display). |
| `render_mode="html"` | Pokémon only: `render()` returns a battle replay as HTML. |
| `render_mode="web"` | Racing only: `save_dashboard()` saves a recorded run. |
| `verbose=True` | Shows backend logs during construction. |

## Additional Install Steps

Most environments run on the base install. A few need an extra step:

| Environments | Extra step |
|---|---|
| Shape-shifting Hand (`RotateInHand-*`, `HandPickUp-*`) | Included in the base install with upstream MuJoCo Warp. The optional collision-fix fork currently conflicts with MuJoCo 4; see the [dependency record](docs/newton-dependency-upgrade.md). |
| Racing (`Racing-*`) | `pip install --no-deps jax-pf@git+https://github.com/hzheng40/jax_pf f1tenth_gym_jax@git+https://github.com/f1tenth/f1tenth_gym_jax` |
| JAX on an NVIDIA GPU (Racing, Warehouse) | `codesign-gym-install-jax-gpu` |
| Pokémon (`Pokenv-*`) | Install Node.js 22 or later, then do the steps in [Pokémon](#pokémon). |
| Tests | `pip install -e './codesign-gym[dev]'`, then `pytest` |
| RL tutorial (`notebooks/`) | `pip install -e './codesign-gym[tutorial]'` |

The `morphhand` extra replaces `mujoco-warp` with a fork for every environment in that Python environment.
`codesign-gym-install-jax-gpu` picks the CUDA version for your GPU and driver (`--dry-run` shows the command first).

#### Custom Installation (e.g. for Pokémon)

```bash
pip install -e './codesign-gym[showdown]'
codesign-gym-install-showdown
```

The second command builds the Pokémon Showdown server. The environment starts and stops the server for you.
See the [Pokémon README](src/envs/pokenv/README.md) for usage.

To develop the Pokémon package at the same time, install a local copy:

```bash
git clone --recurse-submodules https://github.com/camral/pokemon-codesign-env
pip install -e ./pokemon-codesign-env
codesign-gym-install-showdown
```

