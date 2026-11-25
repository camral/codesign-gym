# Design space

The design is the shape of the two fingertips that touch the object: the index tip and the thumb tip of the LEAP hand.
Each is a full triangle mesh, so a design can change the shape freely, including the number of vertices and faces.

| Part | Content |
|---|---|
| `Mesh.vertices` | `(N, 3)` float array, absolute XYZ in metres in the fingertip's frame |
| `Mesh.faces` | `(F, 3)` integer array of triangle indices into `vertices` |
| `design["index"]` | Index fingertip mesh |
| `design["thumb"]` | Thumb fingertip mesh |

`Mesh` and `MeshSpace` are importable from `envs.nerograsp`. `Mesh.copy()` returns an independent copy.

The design space is a `MeshSpace` (a Gymnasium `Space`). The full format and its rules are in
[MESH_EMBODIMENT_API.md](../../../src/envs/nerograsp/MESH_EMBODIMENT_API.md).

## The mounting patch

Every fingertip is bolted to the finger through a fixed mounting patch: vertices `0`–`24` and the 32 faces that use only those vertices.
They must stay exactly as they are in the base design.
Everything after vertex 24 is the replaceable tip, and it may be moved, refined or remeshed, as long as the result:

- is watertight (every edge is shared by exactly two faces),
- attaches along every boundary edge of the mounting patch,
- is one connected piece, with no unused vertices, duplicate triangles or degenerate (repeated-index) triangles.

## Designs and objects in `make_vec`

Multi-object ids evaluate each design on every one of their objects.
`num_envs` must be a multiple of the id's object count, and the vector design is a list of `num_envs / n_objects` dictionaries:
design `d` drives copies `d * n_objects` to `d * n_objects + n_objects - 1`, and copy `i` holds object `i % n_objects`.

| Id | Objects, in copy order | `n_objects` |
|---|---|---|
| `NeroGrasp-Tiny` | set screw, T-nut, lock nut, screw | 4 |
| `NeroGrasp-Small` | scissors, tweezers, spatula, stick, long screw | 5 |
| `NeroGrasp-Thin` | washer | 1 |
| `NeroGrasp-All` | the Small objects, then the Tiny objects, then the washer | 10 |
| `NeroGrasp-<Object>` | that object | 1 |

So `make_vec("NeroGrasp-Tiny", 8)` takes a list of 2 designs, and copies 0–3 test the first one on the four Tiny objects.
`info["embodiment/id"]` and `info["task/object"]` name the design and object of every copy.
With `make`, the design is a single dictionary.

## Base design

Mesh decomposition uses a `spawn` process pool, so when you run these snippets as a script, put them under `if __name__ == "__main__":`.

The base design is the `angle_76` fingertip, the same mesh for both fingers: 98 vertices and 192 faces, about 16 × 14 × 16 mm.

```python
import numpy as np
from codesign_gym import make

env, base = make("NeroGrasp-SetScrew", use_cpu=True)
print(env.embodiment_space)
for name, mesh in base.items():
    print(name, mesh.vertices.shape, mesh.faces.shape)
```

```text
MeshSpace({'index': (base_vertices=98, base_faces=192, mounting_vertices=25), 'thumb': (base_vertices=98, base_faces=192, mounting_vertices=25)}, sample_perturbation_m=0.0001)
index (98, 3) (192, 3)
thumb (98, 3) (192, 3)
```

The mounting patch sits in the plane `y = 4 mm`, and the tip extends from there to `y = 18 mm`.

## Build a design

Copy the base meshes and move the replaceable vertices.
Here the index tip is stretched 20% further from the mounting plane, keeping the mounting patch unchanged:

```python
def stretch_tip(mesh, factor, mount_y=0.004):
    """Scale the replaceable vertices along y, away from the mounting plane."""
    out = mesh.copy()
    out.vertices[25:, 1] = mount_y + (out.vertices[25:, 1] - mount_y) * factor
    return out


longer = {"index": stretch_tip(base["index"], 1.2), "thumb": base["thumb"].copy()}
print(env.validate_embodiment(longer))          # True
```

`embodiment_space.sample()` returns a valid-format design with every replaceable vertex moved by up to ±0.1 mm (`sample_perturbation_m`).
It is a starting point for search, not a guaranteed-valid design: samples still go through the checks below.

## What is rejected

Before checking, the env shrinks the replaceable vertices back toward the mounting seam if the tip exceeds the largest allowed bounding box, so a design that is too large is scaled down, not rejected.
`validate_embodiment` then returns `False` (it does not raise), and `reset` returns `rejected=True`, for:

- a broken format: a moved mounting vertex or face, an open or disconnected mesh, unused vertices, duplicate or degenerate triangles,
- a tip whose volume is under 90% of the base tip's (`min_volume_ratio`),
- a tip thinner than 0.3 mm (`min_thickness_mm`), or smaller than the minimum bounding box (`enforce_minimum_bbox`),
- a degenerate mesh (`reject_degenerate_meshes`),
- with `snap_plane_clamp=True` only, any change to the vertex or face count.

The thresholds are fields of `MeshRejectorConfig`; pass `rejectors=MeshRejectorConfig(...)` to `make` or `make_vec` to change them (`None` turns one off).
A rejected reset does not raise. Its reasons are in `info["embodiment/rejection_reasons"]`, and `step` raises `RuntimeError` until a reset succeeds:

```python
shorter = {"index": stretch_tip(base["index"], 0.6), "thumb": base["thumb"].copy()}
print(env.validate_embodiment(shorter))         # False

obs, rejected, info = env.reset(embodiment=shorter)
print(rejected)                                  # True
print(info["embodiment/rejection_reasons"])

obs, rejected, info = env.reset(embodiment=base)  # recover with a valid design
print(rejected)                                  # False
env.close()
```

```text
False
True
Invalid embodiment:
embodiment row 0, fingertip leap_if_tip: leap_if_tip volume 1.209564e-06 < 1.814347e-06
embodiment row 0, fingertip leap_if_tip: leap_if_tip min bbox max [8.0, 12.4, 8.0] mm < [3.0, 14.0, 3.0] mm
False
```

In a vector env, one rejected design rejects the whole batch: every copy must reset successfully before stepping.
