# Spatial Poisson ratio and terrain contact

Poisson's ratio is now a decoded material field, alongside Young's modulus,
density and actuator properties. Its default is 0.35 and supported design bounds
are 0.25–0.45. Changing its embedding coordinates is differentiable with fixed
topology and actuator labels.

## Programming each representation

| Generator | Poisson coordinates in the embedding |
|---|---|
| Direct grid | `values[..., decoder.spec.poisson_index]` |
| B-spline grid | Same index, evaluated with cubic B-spline interpolation |
| General CPPN | Final output row of `weights` and final `bias` |
| General primitives | `features[..., decoder.spec.poisson_index - 1]` |
| Stock biped/quadruped | `poisson_values`, a physical-value coefficient grid with three samples per axis |
| Scenario chamber generators, all four geometry representations | The same independent `poisson_values` coefficient grid |

The general decoders use bounded scores for the new head. It is appended after
all old channel logits, so existing actuator indices are unchanged. The stock
walker/scenario grids contain physical Poisson values directly and use linear
interpolation in material coordinates. Their new coefficients default to 0.35.
The random-creature generators produce spatially varying Poisson fields in 2D
and 3D. The implicit CPPN co-design parameterization includes a separate trainable
`poisson` group; density and discrete actuator choices remain frozen there.
For gradient optimization, initialize Poisson scores inside their bounds:
exactly saturated endpoints can have zero gradients. The default 0.35 seed
is in the interior.

Example for a direct grid:

```python
import numpy as np
from envs.soft_walker.engine import make
from envs.soft_walker.engine.embodiments.decoders import set_poisson

env, embedding = make("SoftMPM2D-v0", decoder="direct", dtype="float32")
embedding = set_poisson(env.decoder, embedding, 0.35)
head = env.decoder.spec.poisson_index
shape = embedding["values"].shape[:-1]
embedding["values"][..., head] = np.linspace(-1, 1, shape[0], dtype=np.float32)[:, None]
observation, rejected, info = env.reset(embodiment=embedding)
assert not rejected[0], info
env.close()
```

For a stock walker, change `embedding["poisson_values"]` instead. Its shape is
`(3, 3, 1)` in 2D or `(3, 3, 3, 1)` in 3D, with entries between 0.25 and 0.45.
These values affect material properties at the actual particle/element sites;
they do not relocate simulation particles.

## Physics meaning

- MPM uses local Poisson values per material particle; FEM uses them per element.
- Beams use local voxel values to form shear modulus `G = E / (2*(1+nu))`.
  Each link uses harmonic endpoint averages of both `E` and `G`, representing
  equal-length material segments in series. The stored link bending/torsion and
  extension/shear matrices therefore change when local Poisson values change.
- The 2D central-force spring lattice has no independent continuum Poisson
  parameter. Its decoder can carry the field when switching representations,
  but the spring mechanics do not claim to implement it.
- Native Warp MPM/FEM preserve local values. Native beam bundles store assembled
  stiffness matrices plus effective edge material metadata. Rebuild/export after
  changing the design; editing metadata alone does not rebuild beam mechanics.

## Existing designs

New environments do not inject a global `poisson` key. New design JSON files use
version 3. Loading an old built-in v1/v2 design pads only the new material head;
existing geometry, stiffness, actuator and routing coordinates are preserved.
Old stock walker/scenario dictionaries gain a uniform 0.35 coefficient grid.

An explicitly supplied legacy scalar `embedding["poisson"]` remains a global
override for body environments, including per-lane vector inputs. It supersedes
the spatial field. Remove it before optimizing local Poisson coordinates;
`set_poisson` rejects ambiguous use with that override. Old implicit CPPN designs
are converted to a uniform material head before constructing new optimizer
coordinates. Optimizer coordinate counts have increased, so old Adam state
vectors need a new parameterization rather than direct reuse.

## Nonflat terrain

Springs, MPM, FEM and beams use the shared terrain gap and surface-normal
description for slopes and rounded boxes. Steps, blocks and rough terrain are
compositions of those primitives. Friction acts in the local tangent plane.
Contact remains a regularized penalty law; small penetration is expected.

A rounded-box tie previously selected one face for the contact force while
JAX's energy derivative averaged tied faces. The shared gap now follows the
same deterministic face convention as the normal and native Warp path. Contact
virtual-work tests cover spring/FEM forces, beam force and torque, and MPM marker
transfer in 2D/3D. Exact ties use that selected derivative; the terrain remains
piecewise smooth, rather than differentiable across every face switch.

Focused tests live in `tests/soft_walker/test_spatial_poisson_*.py`,
`tests/soft_walker/test_nonflat_terrain.py`, `tests/soft_walker/test_warp_native_nonflat_terrain.py`,
and `tests/soft_walker/test_warp_native_nonflat_rollout.py`. The rollout tests compare
complete spring states and command gradients after two accepted contact steps
on slopes and rounded steps. See [contact details](NONFLAT_TERRAIN.md) for the
supported terrain primitives and validation scope.
