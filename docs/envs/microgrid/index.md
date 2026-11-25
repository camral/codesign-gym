# Microgrid

```{include} ../../_generated/readmes/microgrid.md
```

## At a glance

| | |
|---|---|
| Ids | `Microgrid-Random`, `Microgrid-Forecast`, `Microgrid-Campus`, `Microgrid-OffGridSingle`, `Microgrid-OffGridCampus` |
| Design | `Dict` of 3 scalars: solar kW, battery kWh, generator kW |
| Observation | `Box(5,)` float32 (`Box(29,)` in Forecast) |
| Action | `Box(3,)` float32 in `[-1, 1]`: battery, generator, grid |
| Episode | 24 one-hour steps (72 in the off-grid ids) |
| Mid-episode design changes | Yes (training only) |
| Autoreset | Next step; partial reset supported |
| Render modes | `rgb_array`, `human` |
| Extra install | None |
| Source | [`src/envs/microgrid`](../../../src/envs/microgrid) |

```{toctree}
:maxdepth: 1

embodiment
spaces
examples
```
