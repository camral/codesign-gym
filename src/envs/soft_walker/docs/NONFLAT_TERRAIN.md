# Nonflat terrain contact

Spring 2D, MPM 2D/3D, Hex8 FEM and beam models accept the shared analytical
terrain: a plane, a slope, or rounded axis-aligned boxes joined to a ground
half-space. `TerrainConfig.fixed_slopes` selects deterministic slopes;
`fixed_boxes` specifies each box as `(cx, cy, cz, hx, hy, hz, rounding)`.
The second coordinate is vertical. In 2D, the lateral slope must be zero and
the z components of boxes are omitted by terrain generation.

The shared contact law uses signed gap and the local outward normal. The
normal penalty is the derivative of a softplus penetration energy. Friction
acts in the local tangent plane and dissipates energy. Contact remains a
regularized penalty law, with finite penetration; these primitives do not
implement hard contact, imported height fields or self-contact.

MPM evaluates contact at reference-face markers carried by each particle's
deformation and transfers forces through the same marker interpolation used
by its energy. Beam contact includes marker rotation and the corresponding
torque. Spring and FEM distribute the exposed reference area to nodes. The
native Warp exporter preserves the resolved terrain arrays. A spring model
with explicit terrain uses the shared multidimensional contact law rather
than the legacy flat-only spring friction formula.

Box medial axes and switches between solid primitives are nonsmooth. The
declared derivative selects the first tied box face, the first tied box, and
the plane when its gap equals the nearest box gap. The JAX distance now uses
the same selected face as its reported normal and the native Warp routine;
previously `max` averaged tied face derivatives. Gap and force values are
unchanged by this correction. A central finite difference at a nonsmooth tie
does not define that selected one-sided derivative.

Focused regressions are in [test_nonflat_terrain.py](../../../../tests/soft_walker/test_nonflat_terrain.py),
[test_warp_native_nonflat_terrain.py](../../../../tests/soft_walker/test_warp_native_nonflat_terrain.py),
and [test_warp_native_nonflat_rollout.py](../../../../tests/soft_walker/test_warp_native_nonflat_rollout.py).
They cover energy/force virtual work for every physics family, native/JAX
contact values and derivatives on slopes and rounded corners, solid contact
routing, and two accepted spring steps with command gradients on a slope and
a rounded step. This is local and short-trajectory qualification, not a claim
that every long robot trajectory converges on every terrain.

For CPU validation from a source checkout:

```sh
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
  CODESIGN_TEST_DTYPE=float64 WARP_TEST_DTYPE=float64 WARP_TEST_DEVICE=cpu \
  PYTHONPATH=src python -m pytest -q tests/soft_walker/test_nonflat_terrain.py \
  tests/soft_walker/test_warp_native_nonflat_terrain.py tests/soft_walker/test_warp_native_nonflat_rollout.py
```

For float32, set both dtype variables to `float32` and `JAX_ENABLE_X64=0`.
The native tests also accept `WARP_TEST_DEVICE=cuda:0`; keep JAX on CPU for the
independent oracle and expose the intended GPU instead of hiding CUDA. CUDA
episode tests require native conditional graph support.
