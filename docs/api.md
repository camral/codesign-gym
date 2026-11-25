# API reference

## Entry points

```{eval-rst}
.. autofunction:: codesign_gym.make
.. autofunction:: codesign_gym.make_vec
.. autofunction:: codesign_gym.list_envs
```

`codesign_gym.REGISTRY` maps each environment id to its builder (`"module:function"`) and default keyword arguments.

## Environment classes

`make_vec` returns a subclass of {class}`core.codesign_vec_env.CoDesignVecEnv`. `make` wraps it in {class}`codesign_gym.SingleEnv`, which removes the batch axis.

```{eval-rst}
.. autoclass:: codesign_gym.SingleEnv
   :members: reset, step, validate_embodiment, render, close

.. autoclass:: core.codesign_vec_env.CoDesignVecEnv
   :members: reset, step, validate_embodiment

.. autoclass:: core.codesign_env.CoDesignEnv
   :members: reset, step, validate_embodiment
```

## Return values

| Call | Returns |
|---|---|
| `make(env_id, **kwargs)` | `(env, base_embodiment)` |
| `make_vec(env_id, num_envs, **kwargs)` | `(env, base_embodiment)` |
| `env.reset(seed=None, options=None, embodiment=None)` | `(obs, embodiment_rejected, info)` |
| `env.step(action, embodiment=None)` | `(obs, reward, terminated, truncated, info)` |
| `env.validate_embodiment(design)` | `bool` (`make`) or one bool per copy (`make_vec`) |
