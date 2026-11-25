# Racing

```{include} ../../_generated/readmes/f1tenth.md
```

## At a glance

| | |
|---|---|
| Ids | `Racing-Spielberg`, `Racing-Levine` |
| Design | `Box(5,)` float32: grip, front and rear cornering stiffness, drive force, weight split |
| Observation | `Box(39,)` float32: 7 track-relative car states + 32 lidar ranges |
| Action | `Box(2,)` float32: steering velocity in `[-3.2, 3.2]` rad/s, acceleration in `[-17.62, 17.62]` m/s² |
| Episode | 0.01 s steps; 1500 (Spielberg) or 1000 (Levine) steps, 10000 for an evaluation env; ends early on a crash or a full lap |
| Mid-episode design changes | Yes |
| Autoreset | Next step; partial reset supported |
| Render modes | `rgb_array`, `human`, `web` |
| Extra install | `f1tenth_gym_jax` and `jax_pf` from git, with `--no-deps` (see Install above) |
| Source | [`src/envs/f1tenth`](../../../src/envs/f1tenth) |

```{toctree}
:maxdepth: 1

embodiment
spaces
examples
```
