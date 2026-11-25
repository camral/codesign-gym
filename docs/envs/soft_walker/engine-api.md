# Engine API

These are the engine's host-facing entry points. To use the registered ids, use `codesign_gym.make_vec` (see
[Examples](examples.md)); you do not need this page.

The factories are lazy: when you import them, no accelerator starts. When you call a factory, it loads the backend
you selected. The signatures do not show the options that go through `**kwargs`, so the table gives them.

## Factory options

| Option | Contract |
|---|---|
| `env_id` | One of the five ids that `registered_envs()` returns (for example `SoftSpring2D-v0`). These are not the `SoftWalker-*` ids of Codesign Gym. |
| `backend` | `"jax"` (the engine default) or `"warp"`. The `SoftWalker-*` ids always use Warp. |
| `dtype`, `device` | The precision and the device. On JAX, float64 needs the process's JAX x64 setting. The native factories check their own device and precision. |
| `integrator` | `None` keeps the configured method or the backend default. Use `available_integrators` before you construct an env. An unsupported combination raises; the engine does not silently use a different method. |
| `physics_steps_per_action` | A positive Python `int`. It holds each command for that many physics steps. The physical timestep and the settling duration do not change; the low-level action horizon does not change. |
| `num_envs` | Vector factory only. The number of independent environments. Vector observations and commands have a leading environment axis. |
| Other keyword arguments | Backend- and family-specific settings: decoder or configuration, a resolved native bundle, spaces and vector execution. They are not interchangeable between backends. |

## Entry points

```{eval-rst}
.. autofunction:: envs.soft_walker.engine.make

.. autofunction:: envs.soft_walker.engine.make_vec

.. autofunction:: envs.soft_walker.engine.make_gymnasium_vec

.. autofunction:: envs.soft_walker.engine.registered_envs

.. autofunction:: envs.soft_walker.engine.numerics.selection.available_integrators
```

## Integrator metadata

```{eval-rst}
.. autoclass:: envs.soft_walker.engine.numerics.selection.IntegratorSpec
   :members:
   :exclude-members: __init__
```

## Spring configuration

Lengths are in metres, durations in seconds, density in kg/m³ and Young's modulus in Pa. Command amplitudes and the
normalized objective terms have no units. The configuration cannot be changed after you make it. This class is for the
spring family only; it is not a general native model configuration.

```{eval-rst}
.. autoclass:: envs.soft_walker.engine.physics.spring_config.SpringConfig
   :members: dt
```

## Saved recordings

The saved coordinates and connectivity are the record of the simulated samples. Visual extrusions, particle radii and
skins are only for display. Loading a recording reads files on the host, and it can reject a damaged or unsupported
file format.

```{eval-rst}
.. autoclass:: envs.soft_walker.viewer.recording.Recording
   :members: frame_indices, verdict

.. autofunction:: envs.soft_walker.viewer.recording.world_coordinates
```
