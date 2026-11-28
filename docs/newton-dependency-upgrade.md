# Newton and shared simulator dependencies

This upgrade keeps MuJoCo 4 and uses Newton development revision
[`b3528c6`](https://github.com/newton-physics/newton/commit/b3528c6cc5d8fc261f59963890084abebce86cfc)
(`1.9.0.dev0`). That revision declares MuJoCo Core and MuJoCo Warp `~=4.0.1`
in its simulator extra. Newton 1.8 instead declares MuJoCo 3.17. The source
revision is fixed in `pyproject.toml`; installation requires Git and access to
GitHub. Replace it with a compatible released Newton version after validation.

The base package requires Warp >=1.19, JAX >=0.10, MuJoCo/MuJoCo Warp ~=4.0.1,
and Newton's OpenGL/UI dependencies. Jumanji imports `tree` directly, so
`dm-tree` is declared explicitly instead of relying on older Chex versions to
install it. The development extra includes Gymnasium's MuJoCo dependencies for
the stock-environment comparison tests.

MuJoCo Warp now computes each world's solver inertia scale from its current
mass matrix. Its model no longer exposes `stat.meaninertia`. The twelve
MuJoCo environment backends only expand that field when an older fork provides
it. Walker2d's native-model mirror follows the same rule. Per-world masses,
inertias, constraint constants, and design-reset behavior remain covered by the
existing environment tests.

The live Soft Walker recording adapter now supplies the metric-column metadata
used by the interactive UI and updates its displacement readout from the live
geometry. Without that metadata the viewer raised `AttributeError` during reset
when the UI was active.

## Reproduce the checked environment

Validation uses Linux x86_64, Python 3.12.13, an RTX 5090, and NVIDIA driver CUDA
support reported as 13.2. The key package versions are recorded in
[`constraints-newton-tested.txt`](../constraints-newton-tested.txt): Newton at
the revision above, Warp 1.19.0, JAX/JAXlib 0.10.0, MuJoCo/MuJoCo Warp 4.0.1,
Torch 2.13.0, NumPy 2.3.5, SciPy 1.17.0, and Gymnasium 1.2.3. This is a
constraint file for the numerical stack, not a complete transitive lock.

```bash
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -e '.[dev]' -c constraints-newton-tested.txt
# Also install the historical JAX diagnostic stack used in this test environment.
uv pip install diffrax -c constraints-newton-tested.txt
uv pip check
```

The final base/dev dependency set also resolves for Python 3.11; that is a
resolution check, not a second runtime test platform. Base plus `dev`,
`tutorial`, `showdown`, and Newton's `sim` extra resolve together. The Showdown
Node server and JAX CUDA plugins are outside this verification run.

When switching an existing environment back and forth between MuJoCo 3.x and
4.x, uninstalling the monolithic 3.x distribution can remove files belonging
to 4.x's split core distribution. A fresh environment avoids that problem.
If necessary, reinstall the matching `mujoco`, `mujoco-core`, and
`mujoco-studio` packages together.

## Regression checks

- All twelve MuJoCo families pass their 48 CPU factory/reset/step cases and
  24 CUDA cases (single and vector interfaces). The CPU selection also passes
  14 device/forwarding checks. CUDA runs were isolated by family; hand cases
  initially exhausted the shared GPU and passed after releasing test memory.
- Newton's own MuJoCo solver passes free-body dynamics checks on CPU and CUDA.
- Seven real OpenGL checks pass: all five Soft Walker geometries, the terrain
  editor, and material/actuator morphology rendering. These instantiate Newton
  directly, so a Pillow fallback cannot make them pass.
- The interactive viewer also passes a reset/render/step check under Xvfb;
  that check and the recording-contract tests pass together (10 tests).
- The focused Warp MPM, JAX spring, and viewer-contract selection passes 46
  tests. CPU partial-reset and JAX continuum/beam mechanics checks pass 51
  tests, including all five native Soft Walker physics families.
- The registered Soft Walker vector suite passes 83 tests with one interactive
  display skip, using `WARP_TEST_DEVICE=cuda:0` for its physics cases.
- Warehouse and the shared seeding/reset selection pass 68 tests, with two
  optional-environment skips, after adding `dm-tree`.
- The broad existing environment suite produced 676 passes and 26 skips.
  Two Warehouse import failures were repaired and verified as above. Eleven
  other failures concern existing documentation links and omitted legacy
  asset files. All eleven reproduce on an untouched snapshot of main `f093af8`
  (11 failures, 27 passes in that baseline selection).

Commands used, with `JAX_PLATFORMS=cpu` and `MUJOCO_GL=egl` where applicable:

```bash
pytest tests/test_mujoco_device.py -k 'not cuda'
pytest tests/test_mujoco_device.py -k test_factory_cuda_reset_and_step
pytest tests/test_newton_solver.py
CODESIGN_TEST_NEWTON_RENDER=1 PYGLET_HEADLESS=1 LIBGL_ALWAYS_SOFTWARE=1 \
  pytest tests/test_soft_walker_newton_render.py
WARP_TEST_DEVICE=cuda:0 LIBGL_ALWAYS_SOFTWARE=1 xvfb-run -a pytest \
  tests/test_soft_walker_vector.py::test_human_newton_viewer_refreshes_during_rollout \
  tests/soft_walker/test_viewer_recording_contracts.py
WARP_TEST_DEVICE=cuda:0 pytest tests/soft_walker/test_warp_native.py \
  tests/soft_walker/test_m2_spring_paths.py \
  tests/soft_walker/test_viewer_recording_contracts.py
WARP_TEST_DEVICE=cpu pytest tests/soft_walker/test_warp_native_partial_reset.py \
  tests/soft_walker/test_m2_continuum.py tests/soft_walker/test_m3_terrain_beam.py
WARP_TEST_DEVICE=cuda:0 pytest tests/test_soft_walker_vector.py
pytest tests/test_robot_warehouse.py tests/test_robot_warehouse_vector.py \
  tests/test_seeding.py tests/test_step_before_reset.py
pytest tests --ignore=tests/soft_walker --ignore=tests/test_soft_walker_vector.py \
  --ignore=tests/test_mujoco_device.py
```

FEM's first CUDA compilation is unusually expensive; the existing
[Warp test guide](../src/envs/soft_walker/docs/WARP_PORTS_COMMANDS.md) documents
builds exceeding 900 seconds. Keep compiler optimization enabled and allow
time for that build before measuring warm execution.

## Existing limitations and follow-up work

The broad suite's eleven unrelated failures comprise ten Soft Walker
documentation checks referencing files removed from main, and one packaging
check for two legacy `deformgrip/ftip_deform/old/*.obj` files omitted from the
package-data patterns. The dependency upgrade does not claim a clean full suite.
The older Soft Walker `requirements-tested.txt` remains historical provenance;
its original JAX version has not been relabeled as this validation environment.

The `morphhand`/`all` extra currently cannot resolve: the collision-fix fork
reports a MuJoCo Warp version below 4.0, conflicting with the base requirement.
The base hand environments run with upstream MuJoCo Warp, whose contact
behavior still differs from that fork. The legacy `notebook-requirements.txt`
uses MuJoCo 3.14 and the fork with `--no-deps`; it is a separate historical setup
and has not been revalidated here. Updating that fork is separate work.

The racing extra also cannot resolve normally: its `jax-pf` dependency asks
for JAX >=0.6.1,<0.7, incompatible with the base package. Its existing
`--no-deps` installation instructions do not establish compatibility with the
new stack. Racing and the external Showdown server were not exercised.

Newton Cosserat currently declares Newton >=1.8,<1.9. Before adding the
continuum environment, update and validate its dependency declaration against
this pinned development revision. This PR verifies existing Co-Design Gym
environments and Newton's rigid solver; it does not validate a Cosserat rollout.
