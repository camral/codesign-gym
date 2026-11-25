# NeroGrasp

```{include} ../../_generated/readmes/nerograsp.md
```

## At a glance

| | |
|---|---|
| Ids | `NeroGrasp-All`, `-Small`, `-Tiny`, `-Thin`, and one per object: `-SetScrew`, `-TNut`, `-LockNut`, `-Screw`, `-Scissors`, `-Tweezers`, `-Spatula`, `-Stick`, `-ScrewLong` |
| Design | `{"index": Mesh, "thumb": Mesh}`: two watertight triangle meshes (98 vertices, 192 faces in the base design) |
| Observation | `Dict(actor=Box(30,), critic=Box(40,))` float32 |
| Action | `Box(16,)` float32 in `[-1, 1]` (joint-target changes for the 16 LEAP hand joints) |
| Episode | 100 steps of 0.1 s; the arm follows a fixed script |
| Mid-episode design changes | No (`step` ignores a design with a warning) |
| Autoreset | None: all copies share one clock and end together; no partial reset |
| Render modes | `rgb_array`, `human` |
| Extra install | None (VHACD is in the base install; VisACD is optional) |
| Device | CUDA when available; `use_cpu=True` or `device="cpu"` runs on the CPU |
| Source | [`src/envs/nerograsp`](../../../src/envs/nerograsp) |

```{toctree}
:maxdepth: 1

embodiment
spaces
examples
```
