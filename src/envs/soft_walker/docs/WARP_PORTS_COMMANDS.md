# Warp ports and solver-optimization commands

For this integrated checkout, install from the codesign-gym repository root
with `python -m pip install -e '.[dev]'`. Run the commands below from
`src/envs/soft_walker` so their `scripts/` and bundled example paths resolve.
Offline audit commands use the canonical `envs.soft_walker.diagnostics` modules.
The patch is already integrated. The patch-application section below records
the original standalone delivery and is not an installation step here.

## Baseline and installation

Apply `codesign_warp_all_patch.zip` to the source snapshot uploaded as
`35d0ff1f-1fbb-4c69-9340-7525051ac1ce.zip` (the repaired MPM/speed checkout).
Do not apply it on top of a hypothetical earlier all-physics port: this patch
contains those ports. It preserves unrelated local files and refuses conflicting
changes to any touched file. Keep the working Warp 1.16.0 / JAX 0.9.0.1 installation.
No dependency upgrade or reinstall is prescribed for an editable checkout.

```bash
unzip codesign_warp_all_patch.zip
cd /path/to/your/current/checkout
python /path/to/codesign_warp_all_patch/apply.py --root . --check
python /path/to/codesign_warp_all_patch/apply.py --root .
python -m pip check
```

`--check` does not modify the checkout. Application logs are saved beside the
helper. `--reverse` restores the original touched files only when their target
hashes still match; it is not a force-overwrite command.

## Focused correctness collection

Run CPU and CUDA sequentially, not concurrently. These are new and affected
checks, not the historical full suite, a large gait search, or a CUDA timing claim.
The JAX oracle runs on CPU by design: this is numerical parity, not a same-device
performance benchmark. Each stage runs in its own process and retains failures,
source identity, stdout, JUnit, numerical arrays, ladders, and elapsed time.

```bash
export JAX_COMPILATION_CACHE_DIR="$HOME/.cache/codesign-jax"

python -m envs.soft_walker.diagnostics.run_warp_ports_tests \
  --device cpu --dtypes float64 float32 \
  --physics spring beam fem \
  --out runs/warp_ports_cpu

CUDA_VISIBLE_DEVICES=0 python -m envs.soft_walker.diagnostics.run_warp_ports_tests \
  --device cuda --dtypes float64 float32 \
  --physics spring beam fem --audit \
  --timeout 3600 \
  --out runs/warp_ports_cuda
```

Do **not** enable `--offline-spaces` on the local installation. The sandbox uses
that explicitly labeled adapter only because Gymnasium is unavailable there.
`--audit` requires the existing CUPTI installation and must not overlap another
Nsight/XProf/CUPTI profiling session. It separately checks native whole-forward
and whole-value-and-gradient submissions. Missing CUDA never falls back to CPU.
A timeout kills the stage and preserves the archive. Use a new output directory
for a repeat; do not mix evidence from different source revisions.

The first FEM CUDA build can exceed the runner's default 900-second stage
timeout. The command above allows 3600 seconds per stage for that cold build;
cached execution and warm benchmark timings are measured separately. Preserve
the Warp and CUDA driver caches when repeating runs.

FEM explicitly uses compiler optimization level 2. Do not lower it to level 0
to shorten compilation: on the local sm120 GPU, unoptimized float64 force
adjoints required over 700 KB of local storage per thread and CUDA graph
instantiation failed. The optimized kernels used about 13 KB per thread and
passed the float64 GPU correctness and parity checks. Equations, automatic
differentiation, unrolling, and solver tolerances remain unchanged.

## Build and replay all three native smoke bodies

These are small animated mechanics checks, **not qualified walking gaits**.
The native constructors, replay and adjoints do not import JAX.

```bash
for model in spring beam fem; do
  python scripts/warp_solids.py make-smoke \
    --physics "$model" --dtype float64 --steps 16 \
    --out "runs/${model}_smoke_setup"

  python scripts/warp_solids.py replay \
    --physics "$model" --device cpu --dtype float64 \
    --bundle "runs/${model}_smoke_setup/model.npz" \
    --gradient --animation --repeats 3 \
    --out "runs/${model}_smoke_cpu"
done
```

After correctness passes, repeat `replay` with `--device cuda --audit` and fresh
output directories. `--gradient` records and evaluates the complete loss and
implicit adjoint; omit it for a forward-only timing. `--animation` renders only
after the complete execution returns. `trajectory.npz` contains actual sampled
states and physical timestamps; the GIF does not synthesize intermediate motion.

The default smoke timestep is 0.5 ms. Sixteen control samples with one substep
are only 8 ms, deliberately a startup check rather than evidence of a gait.
The recorder labels such output as unqualified.

## Export your existing robot/controller to a native solid bundle

Export runs JAX only once to resolve the existing design, material arrays,
terrain and controller commands. The native simulation never calls JAX.
The Lagrangian ports implement backward Euler and active strain. Incompatible
integrator/actuation choices are rejected rather than silently substituted.

```bash
python scripts/warp_solids.py export-demo \
  --physics spring --gait alternating --cycles 1 --dtype float64 \
  --out runs/spring_biped_export

python scripts/warp_solids.py export-demo \
  --physics beam --gait trot --cycles 1 --dtype float64 \
  --out runs/beam_quadruped_export

python scripts/warp_solids.py export-demo \
  --physics fem --gait trot --cycles 1 --dtype float64 \
  --out runs/fem_quadruped_export
```

Use `--demo-config path/to/config.json` to supply the existing `DemoConfig`
schema, and `--controller path/to/checkpoint.npz` for coefficients under the
`controller` or `parameters` key (or a plain NPY). This exporter is not a universal
translator for arbitrary custom scenario/controller classes. The driver and
precision in the JSON must match the explicit command. Its own cycles value
is authoritative when JSON is supplied.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/warp_solids.py replay \
  --physics beam --device cuda --dtype float64 \
  --bundle runs/beam_quadruped_export/model.npz \
  --checkpoint-stride 16 --memory-mib 512 \
  --gradient --animation --audit --repeats 3 \
  --out runs/beam_quadruped_warp
```

Substitute `spring`/`fem` and the corresponding bundle. Commands and protocol are
read beside the bundle by default. The full command length and settling are
retained; the replay does not silently truncate the gait to a smoke case.
Pass `--commands` only to select a different bounded command table explicitly.
No new sustained walking qualification or large-scale CUDA performance result
is claimed by this delivery.

## Fixed API examples

```python
import numpy as np
from envs.soft_walker.engine import make

env, embodiment = make(
    "SoftBeam3D-v0", backend="warp", device="cuda", dtype="float64",
    max_action_space_size=8, observation_profile="resolved",
)
observation, rejected, info = env.reset(embodiment=embodiment)
assert not rejected
commands = np.zeros((8, env.channels), dtype=np.float64)
episode = env.episode(commands, checkpoint_stride=4)
result = episode.run(gradient=True)       # one whole CUDA graph replay
assert result["status"] == 0 and result["adjoint_status"] == 0
gradients = episode.gradients()          # reporting boundary, after completion
env.close()
```

This exposes the resolved physical-array embodiment. It does not claim identical
JAX observation ordering or native gradients through an arbitrary latent decoder.
Use `bundle=SolidBundle.load(...)` to select an exported robot. Interactive
`env.step` returns observations and therefore has an API boundary; use an episode
for resident open-loop execution. `make_vec(..., backend="warp",
vectorization_mode="warp")` supplies correct independent lanes with next-step
autoreset, not a new fused multi-robot sparse solver.

## MPM optimization A/B on the existing biped

The baseline `optimized` method retains the options already measured by the
user. New variants change one implementation aspect at a time:

| Variant | Linear-work reuse | Explicit compact allocation | Node-block preconditioner |
|---|---|---|---|
| optimized | no | no | no |
| linear_reuse | yes | no | no |
| compact_reuse | yes | requested capacity | no |
| block | yes | no | yes |
| all | yes | requested capacity | yes |

The recorded biped had at most 90 live nodes. A 128-node experiment is justified
for that frozen body, not guaranteed for every new design. Capacity overflow is
an explicit transactional failure, never silent node truncation.

```bash
CUDA_VISIBLE_DEVICES=0 python -m envs.soft_walker.diagnostics.run_warp_speed_audit \
  --device cuda --dtypes float64 --dims 2 \
  --bundle runs/YOUR_BIPED_EXPORT/model.npz \
  --commands runs/YOUR_BIPED_EXPORT/commands.npy \
  --checkpoint-stride 32 \
  --variants optimized linear_reuse compact_reuse block all \
  --compact-nodes 128 --gradient --compare-jax --repeats 3 \
  --skip-tests --skip-ladder --timeout 3600 \
  --out runs/mpm_new_ab
```

The timeout is a watchdog, not a promised execution time. This benchmark retains
its own whole-graph audit, fixed model, full command table, settling, timestep,
precision, objective, tolerances, raw timings and work counts. Do not run it
inside another CUPTI profiling session.

The corresponding direct MPM replay flags are:

```text
--krylov-method adaptive --cache-tangent --reuse-replay-roots
--compact-nodes 128 --linear-work-reuse --block-preconditioner
```

These MPM settings are not silently applied to the new Lagrangian models, which
have permanent mesh-node degrees of freedom and their own cached local blocks.
Global BSR assembly, mixed precision and control-only adjoints are not enabled
by this patch; they remain separately testable alternatives.

## Return evidence

```bash
zip -j runs/warp_all_return.zip \
  runs/warp_ports_cpu.zip \
  runs/warp_ports_cuda.zip \
  runs/mpm_new_ab.zip
```

Include the requested native robot replay ZIPs as available, including failed
runs. Each focused native worker already compares its result with the JAX
reference and saves the arrays, so no extra same-machine benchmark is required
to make its correctness result meaningful. CPU/CUDA runs must use the same
source and configuration; their archives retain that identity for comparison.
