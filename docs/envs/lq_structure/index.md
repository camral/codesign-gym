# LQ Structure

```{include} ../../_generated/readmes/lq_structure.md
```

## At a glance

| | |
|---|---|
| Ids | `LQStructure-LQR`, `LQStructure-LQG`, `LQStructure-QP` |
| Design | `Dict` of 5 arrays of length `n` (default 6) |
| Observation | `Box(24,)` float64 |
| Action | `Box(6,)` float64 (`[-1, 1]` in QP, unbounded otherwise) |
| Episode | 200 steps of 0.05 s |
| Mid-episode design changes | Yes |
| Autoreset | Next step; partial reset supported |
| Render modes | `rgb_array`, `human` |
| Extra install | None |
| Source | [`src/envs/lq_structure`](../../../src/envs/lq_structure) |

```{toctree}
:maxdepth: 1

embodiment
spaces
examples
```
