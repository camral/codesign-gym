# NeroGrasp & NeroTools

Native MuJoCo Warp co-design environments for a 7-DoF Nero arm with full-mesh fingertip co-design.

## Registered IDs
- **Small-object grasping (`NeroGrasp-<Object>`)** — 13 IDs: `All`, `Small`, `Tiny`, `Thin`, `SetScrew`, `TNut`, `LockNut`, `Screw`, `Scissors`, `Tweezers`, `Spatula`, `Stick`, `ScrewLong`.
  - LEAP hand (16-D hand control, scripted descend-hold-lift arm).
  - Embodiment: `{"index": Mesh, "thumb": Mesh}` (25 immutable mounting vertices at indices `0..24`).
- **Articulated tool manipulation (`NeroGrasp-Tools-<Tool>`)** — 2 IDs: `NeroGrasp-Tools-Carabiner`, `NeroGrasp-Tools-WireCutters`.
  - 2-jaw parallel gripper (8-D arm + gripper control, or `trajectory="demo"` residual control).
  - Embodiment: `{"left": Mesh, "right": Mesh}` (12 immutable mounting vertices at indices `0..11`).

## Convex Decomposition Backends
- **`cuacd`** *(default on CUDA)*: Persistent multi-worker GPU pool (`K=12`) with automatic single-mesh VHACD fallback and pool recycling on kernel faults. Automatically falls back to `vhacd` on CPU or when `cuacd` is not installed.
  - Installing `cuacd` from source on Blackwell (`sm_120`, e.g., RTX 5090 / RTX PRO 6000) requires setting `CUACD_GPU_ARCHS` (upstream defaults to `80;86;89;90`):
    ```bash
    CUACD_GPU_ARCHS="80;86;89;90;120" pip install git+https://github.com/eliphatfs/cuacd.git
    ```
- **`vhacd`**: CPU fallback via `trimesh`.
- **`visacd`**: Optional batched OptiX/CUDA backend (`use_visacd=True` or `convex_decomposition_backend="visacd"`).

## Quick Example
```python
import codesign_gym as cg
from envs.nerograsp import Mesh, MeshSpace, NeroGraspConfig, NeroToolsConfig

env, base = cg.make_vec("NeroGrasp-SetScrew", num_envs=8)
obs, rejected, info = env.reset(seed=0, embodiment=base)
```

## Embodiment Reparameterization (CBD-SA)
Because NeroGrasp's native embodiment is a raw `dict[str, Mesh]`, black-box and RL optimizers typically wrap it with a low-dimensional cage deformation. See [`examples/nerograsp_cbdsa_reparam.py`](../../../examples/nerograsp_cbdsa_reparam.py) for a minimal standalone example using [Structural Affinity](https://camral.github.io/structural-affinity/) (`structural-affinity`) to map a flat vector `theta` to valid fingertip meshes while keeping mounting vertices pinned:

```bash
pip install git+https://github.com/camral/structural-affinity.git
python examples/nerograsp_cbdsa_reparam.py
```
