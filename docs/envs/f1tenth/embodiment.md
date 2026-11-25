# Design space

The design is the car's setup: a `Box(5,)` float32 vector. Both ids share the same space; only the track differs.

| Index | Name | Bounds | Base | Meaning |
|---|---|---|---|---|
| 0 | `mu0` | 0.5 – 1.1 | 1.0489 | Tyre friction coefficient before load sensitivity |
| 1 | `C_Sf` | 4.0 – 6.0 | 4.718 | Front cornering-stiffness coefficient |
| 2 | `C_Sr` | 4.0 – 6.0 | 5.4562 | Rear cornering-stiffness coefficient |
| 3 | `F_drive` | 25 – 55 N | 35.57 | Drive force |
| 4 | `lr_frac` | 0.45 – 0.60 | 0.5192 | Share of the wheelbase between the centre of gravity and the rear axle |

The base design is the stock F1TENTH car. The design is not part of the observation.

## How the car's physics follow from the design

Mass is not designed. Each upgrade adds weight, and the extra weight then costs grip. The environment turns the five design values into eight parameters of the single-track vehicle model (`envs.f1tenth.embodiment.design_to_param_vector`):

```text
m      = 3.74 + 1.0·(mu0 − 1.0489) + 0.3·((C_Sf + C_Sr) − 10.1742) + 0.03·(F_drive − 35.57)    [kg]
mu     = mu0 · (1 − 0.10 · (m − 3.74) / 3.74)                                                  tyre load sensitivity
lr     = lr_frac · 0.3302,   lf = 0.3302 − lr                                                   [m], fixed wheelbase
I      = 0.4629 · m · lf · lr                                                                   yaw inertia [kg m²]
a_max  = F_drive / m                                                                            [m/s²]
C_Sf, C_Sr passed through unchanged
```

A light, low-grip car (`[0.5, 4, 4, 25, 0.45]`) weighs 2.22 kg; a fully upgraded one (`[1.1, 6, 6, 55, 0.6]`) weighs 4.92 kg, and its effective grip is only 1.065, not 1.1.
The other vehicle parameters (steering limits, top speed 20 m/s, centre-of-gravity height, car size) are fixed at the f1tenth defaults.

Print the base design and the parameters it produces:

```python
from codesign_gym import make
from envs.f1tenth.embodiment import design_to_param_vector, PARAM_FIELDS

env, base = make("Racing-Spielberg")
print(base)
for name, value in zip(PARAM_FIELDS, design_to_param_vector(base, env.cfg)[0]):
    print(f"{name:6s} {value:.4f}")
env.close()
```

```text
[ 1.0489  4.718   5.4562 35.57    0.5192]
m      3.7400
mu     1.0489
I      0.0471
lf     0.1588
lr     0.1714
a_max  9.5107
C_Sf   4.7180
C_Sr   5.4562
```

The base car's mass, grip, `a_max` and yaw inertia (0.04712 kg m²) match the stock f1tenth car.

## Build a design

Copy the base design and change the entries you want:

```python
import numpy as np
from codesign_gym import make
from envs.f1tenth.embodiment import design_to_param_vector

env, base = make("Racing-Spielberg")
design = base.copy()
design[0] = 1.1      # grippier tyres
design[3] = 45.0     # more drive force
print(env.validate_embodiment(design))                  # True
p = design_to_param_vector(design, env.cfg)[0]
print(f"mass {p[0]:.2f} kg, effective grip {p[1]:.3f}, a_max {p[5]:.2f} m/s^2")
env.close()
```

```text
True
mass 4.07 kg, effective grip 1.090, a_max 11.05 m/s^2
```

## What is rejected

Every design inside the bounds is valid. `validate_embodiment` returns `False` only for a vector that is not shape `(5,)`, contains a non-finite value, or has an entry outside its bounds.

At `reset`, a rejected copy gets `rejected=True`; `reset` does not raise, but `step` raises `RuntimeError` until that copy is reset with a valid design.
At `reset`, `embodiment=None` (or a `None` entry in a per-copy list) gives the base design, in a partial reset too.
At `step`, an invalid design raises `ValueError` and no copy's design changes. A `None` entry in a per-copy list keeps that copy's current design.

With `make_vec`, pass one design (used by every copy), an `(num_envs, 5)` array, or a list with one design or `None` per copy.

## Build options

`make` forwards keyword arguments to the preset's config (`envs.f1tenth.config.F1TenthConfig`). Useful ones:

| Kwarg | Default | Effect |
|---|---|---|
| `time_limit` | 1500 / 1000 | Steps per training episode (an evaluation env always uses 10000) |
| `num_beams` | 32 | Lidar beams, so the observation has `7 + num_beams` values |
| `mu_min`, `mu_max`, `csf_min`, … `lr_frac_max` | see table above | Design bounds |
| `kappa_mu`, `kappa_c`, `kappa_p` | 1.0, 0.3, 0.03 | Mass added per unit of grip, cornering stiffness and drive force |
| `k_mu` | 0.10 | Load sensitivity: grip lost per unit of relative extra mass |
| `dynamic_index` | 0.4629 | Scale of the yaw inertia (the default gives the stock car's 0.04712 kg m²) |

Changing the design bounds also changes the action space, because the acceleration bound is the largest `a_max` any design can reach.

```python
from codesign_gym import make

env, base = make("Racing-Levine", num_beams=64, time_limit=3000)
print(env.observation_space.shape)                      # (71,)
env.close()
```
