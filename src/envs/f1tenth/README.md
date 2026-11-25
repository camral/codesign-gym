# Racing

Ids: `Racing-Spielberg`, `Racing-Levine`. Uses: [f1tenth_gym_jax](https://github.com/f1tenth/f1tenth_gym_jax) (JAX).

A scaled race car runs a time trial.

- **Design:** tyre grip, cornering stiffness, drive power and weight distribution.
- **Control:** steering and throttle/brake.
- **Reward:** progress along the centre line, in metres.
- **Design changes in `step`:** supported, with no state correction needed.

Mass isn't designed directly: grippier, stiffer tyres and more power each add mass, and tyre load sensitivity then eats into the gain. You can't max out everything, but every design in range is valid.

| Id | Track | Lap length |
|---|---|---|
| `Racing-Spielberg` | Spielberg | 343.3 m |
| `Racing-Levine` | Levine | 63.2 m |

The return is the distance driven along the centre line, negative when driving backwards, and a completed lap ends the episode. Within the training step limit (top speed 20 m/s, reverse at most 5 m/s), the return is at most about 300 m on Spielberg, where a lap cannot be completed in time, and one lap (63.2 m) on Levine, and at least about −75 m and −50 m. Evaluation episodes are longer (10,000 steps).

## Install

```bash
pip install --no-deps jax-pf@git+https://github.com/hzheng40/jax_pf f1tenth_gym_jax@git+https://github.com/f1tenth/f1tenth_gym_jax
codesign-gym-install-jax-gpu        # optional, for an NVIDIA GPU
```

`render_mode="web"` records a run, and `save_dashboard()` saves it.
