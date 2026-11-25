# Examples

These run on the CPU (`use_cpu=True`). Mesh decomposition uses a `spawn` process pool, so when you run a snippet as a script, put it under `if __name__ == "__main__":`. The first build compiles the simulation kernels and takes a while; later builds reuse the cache.

## Compare two fingertip designs across objects

`NeroGrasp-Tiny` has four objects, so 8 copies hold 2 designs, each tested on all four objects.
This example compares the base fingertips with an index tip stretched 20% longer, under a policy that keeps the hand still while the arm runs its script:

```python
import numpy as np
import torch
from codesign_gym import make_vec


def stretch_tip(mesh, factor, mount_y=0.004):
    """Scale the replaceable vertices (index 25 on) along y, away from the mounting plane."""
    out = mesh.copy()
    out.vertices[25:, 1] = mount_y + (out.vertices[25:, 1] - mount_y) * factor
    return out


envs, base = make_vec("NeroGrasp-Tiny", 8, use_cpu=True)
print(len(base))                                     # 2: one design per group of four objects
designs = [base[0], {"index": stretch_tip(base[1]["index"], 1.2), "thumb": base[1]["thumb"].copy()}]

obs, rejected, info = envs.reset(seed=0, embodiment=designs)
assert not rejected.any()
hold = torch.zeros((8, 16))
while True:
    obs, reward, terminated, truncated, info = envs.step(hold)
    if info["finished"].all():
        break

returns = info["episode/final_return"].cpu().numpy()
for design, obj, ret in zip(info["embodiment/id"].tolist(), info["task/object"], returns):
    print(f"design {design}  {obj:10s} {ret:.3f}")
```

```text
2
design 0  set_screw  0.091
design 0  t_nut      0.091
design 0  lock_nut   0.089
design 0  screw      0.083
design 1  set_screw  0.089
design 1  t_nut      0.089
design 1  lock_nut   0.087
design 1  screw      0.081
```

With the hand held still nothing is lifted, so these returns are only the small distance reward; a trained policy is what separates designs.

Every copy ends together, so the loop runs until `info["finished"]` is all `True`; stepping again would raise until the next `reset()`.

## Check sampled designs before a reset

One rejected design blocks the whole batch, so screen candidates with `validate_embodiment` first.
It returns one flag per copy, and the copies of one design share a flag:

```python
rng = np.random.default_rng(0)
candidates = [envs.embodiment_space.sample(rng) for _ in range(2)]
candidates[1]["thumb"] = stretch_tip(base[1]["thumb"], 0.6)    # too small: below 90% of the base volume
print(envs.validate_embodiment(candidates))

obs, rejected, info = envs.reset(embodiment=candidates)
print(rejected.cpu().numpy())
print(info["embodiment/rejection_reasons"][4].splitlines()[1])
envs.close()
```

```text
[ True  True  True  True False False False False]
[False False False False  True  True  True  True]
embodiment row 1, fingertip leap_th_tip: leap_th_tip volume 1.209564e-06 < 1.814347e-06
```

## Reparameterize the mesh space with CBD-SA

Because the native design is a raw `dict[str, Mesh]`, black-box and RL optimizers typically wrap it with a low-dimensional cage deformation.
Using [Structural Affinity](https://camral.github.io/structural-affinity/) (`structural-affinity`), you can map a flat continuous vector `theta` to valid fingertip meshes while keeping the mounting patch (`0..24`) fixed:

```python
import numpy as np
from structural_affinity import affinity_matrix, mvc_3d, propagation_operator, sparsify
from codesign_gym import make
from envs.nerograsp import Mesh

# See examples/nerograsp_cbdsa_reparam.py for the full runnable script (including make_ring_cage).
env, base = make("NeroGrasp-Thin", use_cpu=True, snap_plane_clamp=True)
verts = np.asarray(base["index"].vertices, dtype=np.float64)
inner_v, inner_f = make_ring_cage(verts, n_rings=10, n_sides=12, clearance=0.0010)
outer_v, outer_f = make_ring_cage(inner_v, n_rings=6, n_sides=8, clearance=0.0030)

W_mesh = mvc_3d(verts, inner_v, inner_f)
W_lu = np.maximum(mvc_3d(inner_v, outer_v, outer_f), 0.0)
W_lu /= np.maximum(W_lu.sum(axis=1, keepdims=True), 1e-12)
C_aff = sparsify(affinity_matrix(W_lu), density=0.35)
A_op = propagation_operator(C_aff, alpha=0.5, normalize=True)
J = W_mesh[25:] @ A_op.toarray()  # (N_deform, N_inner)

# Optional SVD reduction: compress N_inner handles -> top-4 dominant modes
_, _, vt = np.linalg.svd(J, full_matrices=False)
V_r = vt[:4].T / np.maximum(np.max(np.abs(vt[:4].T), axis=0, keepdims=True), 1e-8)
J_svd = J @ V_r  # (N_deform, 4)

# Apply mode coefficients D (4, 3) to deformable vertices only:
deformed = verts.copy()
deformed[25:] += J_svd @ D
design = {"index": Mesh(deformed, base["index"].faces), "thumb": base["thumb"].copy()}
```

Run the complete standalone script with:

```bash
pip install git+https://github.com/camral/structural-affinity.git
python examples/nerograsp_cbdsa_reparam.py
```
