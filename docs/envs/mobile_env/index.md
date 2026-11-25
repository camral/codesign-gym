# Wireless Network

```{include} ../../_generated/readmes/mobile_env.md
```

## At a glance

| | |
|---|---|
| Ids | `Network-Urban`, `Network-Suburban`, `Network-Motorway` |
| Design | `Dict`: `num_bs` (int) and a padded `(max_bs, 3)` array of `(x, y, tx)` rows |
| Observation | `Box(num_ues · (2 · max_bs + 1),)` float32 in `[-1, 1]` (255 / 300 / 90 values) |
| Action | `MultiDiscrete([max_bs + 1] * num_ues)`: one station choice per user |
| Episode | 100 steps |
| Mid-episode design changes | No (ignored with a warning) |
| Autoreset | Next step; partial reset supported |
| Render modes | `rgb_array`, `human` |
| Extra install | None |
| Source | [`src/envs/mobile_env`](../../../src/envs/mobile_env) |

```{toctree}
:maxdepth: 1

embodiment
spaces
examples
```
