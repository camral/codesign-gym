# Soft Walker

```{include} ../../_generated/readmes/soft_walker.md
```

## At a glance

| | |
|---|---|
| Ids | `SoftWalker-<Model>`, `SoftWalker-<Model>Stairs`, `SoftWalker-<Model>Gaps` for `Spring2D`, `MPM2D`, `MPM3D`, `Beam3D`, `HexFEM3D` (15 ids) |
| Design | Flat `Box` float64, `7 + dim` values per cell: 432 (2D, 8 x 6 cells) or 2880 (3D, 8 x 6 x 6 cells) |
| Observation | `Box` float64, full simulator state: 642 (`Spring2D`), 2316 (`MPM2D`), 55332 (`MPM3D`), 5220 (`Beam3D`), 7974 (`HexFEM3D`) |
| Action | `Box(-1, 1)` float64, one per channel: 4 (2D) or 16 (3D) |
| Episode | 0.2 s settling, then 240 steps of 0.01 s |
| Mid-episode design changes | No (the body is rebuilt at reset) |
| Autoreset | Next step, with the last accepted design; partial reset supported |
| Render modes | `rgb_array`, `human` |
| Extra install | None (EGL for headless 3D frames) |
| Device / dtype | CUDA when Warp finds a GPU, else CPU; float64 by default (`device=`, `dtype=`) |
| Integrator | Backward Euler |
| Source | [`src/envs/soft_walker`](../../../src/envs/soft_walker) |

```{toctree}
:maxdepth: 1

embodiment
spaces
examples
engine
engine-api
guides
```
