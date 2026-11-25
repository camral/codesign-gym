# Whole-episode GPU execution — strict, runtime-certified candidate

This change addresses CPU-driven time/solver loops, not merely individual kernel
fusion. The previous fused-GMRES trace still contained 3,730 host graph launches,
3,453 host WHILE checks, and 3,688 GPU device-to-host copies for one 20 ms interval.
The previous trace summary missed graph calls because their names were decorated.

## One boundary, not one persistent CUDA kernel

The intended boundary is one CUDA graph submission containing all device work:

```
CPU: validate schema/topology, allocate, stage candidate, compile/capture
GPU: decode design -> instantiate -> initialize/settle
     -> all control intervals and implicit substeps -> objective
     -> all reverse/checkpoint/implicit-transpose work (gradient executable)
CPU: read final outputs -> external optimizer / evidence / renderer
```

A graph contains multiple kernels, with device-controlled WHILE/conditional nodes.
There is no claim that a complete FEM/MPM solver is one physical CUDA kernel.
All spatial and candidate-level parallelism remains; dependent timesteps stay
sequential ON THE GPU rather than becoming invalid independent vmap samples.

The public co-design factory/reset/step API is unchanged. Bulk execution is bound
from an environment made through that API. An external policy calling ordinary
Python env.step remains an interactive reference, not a whole-episode GPU path.

## Code organization

- `execution/program.py`: uncompiled pure complete forward/loss/record programs.
  Numeric preparation and settling remain differentiable. Time uses device scans;
  recording uses preallocated device buffers; population candidates use vmap.
- `execution/policy.py`: per-executable CUDA graph/compiler policy, not global
  physics or precision changes. CPU is explicit, never an automatic fallback.
- `execution/compiled.py`: outer compilation, device/memory checks, numerical
  warmup, runtime certification, timing, and failure evidence.
- `execution/cupti.py`: lightweight process-wide CUDA DRIVER_API observer. It
  counts every entry but does not retain millions of child-kernel activities.
- `execution/trace.py`: optional detailed trace verification and robust decorated
  CUDA API name handling. Missing data is never evidence of zero activity.
- `execution/runner.py`: existing demo/scenario CLI integration. The optimizer and
  renderer remain host-side; no recording chunk returns to Python during a run.

`jax.value_and_grad(raw_loss)` is transformed BEFORE outer JIT/compilation.
Likewise `vmap` is applied before compilation, including independent gradients.
Differentiating an already compiled forward executable is not this interface.
A fixed-model/settled-state variant exists for explicit reuse; the default takes
embedding and controller values and includes all initialization/settling.

## Compiler/runtime hazards handled

The pinned XLA host WHILE thunk copies a predicate to CPU and blocks on every
iteration. Its known-trip-count path can also use a host C++ for-loop without a
"While" trace event. Counting only WHILE markers is therefore insufficient.

The strict policy requests WHILE and CONDITIONAL CUDA graphs, keeps loops rolled,
and excludes dynamic-slice fusion. The pinned DynamicSliceFusionCmd can evaluate
slice offsets on the host or copy them from the device even while recording a
graph. Disabling that fusion retains ordinary device gather/slice semantics.
Graph scheduling begins with SERIALIZE dependency ordering; this is not a request
to serialize spatial threads. No numerical tolerance, iteration budget, integrator,
material, contact model, timestep, or preconditioner is changed by this patch.

The previous WHILE-capture experiments failed settling. The new policy differs
by excluding dynamic-slice fusion, controlling graph scheduling/unrolling, and
comparing numerical results. This is a targeted candidate remedy, NOT proof of
the cause of the previous failure or a claim that CUDA has passed here.

## Certification and scope

A newly compiled CUDA executable cannot be used for production until its complete
invocation passes its own certificate. Forward loss, full record, gradient, and
population executables are separately certified. Shape/dtype changes compile new
executables and require new evidence.

The mandatory lightweight certificate requires one driver cuGraphLaunch, zero
standalone host kernel submissions, zero host memcpy/memset operations, no graph
host-function nodes, and finite/repeatable numerical output. Inputs/outputs must
remain on the selected GPU and Python transfer guards disallow copies inside the
call. The JAX program is audited for effects/callbacks. No short trace certificate
is substituted for a long executable's certificate.

CUPTI callback counts observe host CUDA API entries, NOT GPU child activities. The
optional detailed short-probe trace independently checks graph provenance, GPU
H2D/D2H activity, and host control-flow events. The default simulation has no host
memory offload; callbacks cannot establish a universal property of arbitrary
third-party kernels or custom FFI implementations. Only supported native JAX
programs are within this contract.

CUPTI is used sequentially: it cannot overlap Nsight/XProf or another subscriber.
An empty observer, unavailable library, invalid result, or unsupported graph
structure fails closed and leaves evidence. Counters never truncate; sample API
names are capped for log size. Certification is untimed, so Python observer
callback overhead is not included in production measurements.

Compilation/library initialization/graph capture are setup. One final blocking
wait and one output read are the completion boundary, not timestep feedback.
Changing candidate buffers may update graph arguments before launch. Numerical
certification checks an alternate candidate/buffer as well as repeated execution.
One measured invocation is evidence for that executable/run, not a formal proof
about every future compiler or third-party backend.

## Memory and recording

The recorder allocates complete metric/action history and selected position frames
on the device. `--record-stride` changes only snapshot density, not physics.
`--recording-mib` rejects excessive output allocation; it does not include compiler
workspace, which is recorded separately. An out-of-budget run fails rather than
silently streaming to CPU. Partial chunk evidence is intentionally no longer
available during a resident episode; metadata/errors and the last completed
candidate remain saved. Host GIF/NPZ export starts after device completion.

## Numerical and scientific scope

No new walking or flight claim is made. Existing failed/dragging gaits remain so.
The local checks compare whole model preparation, settling, trajectory endpoints,
recordings, and design/control gradients to the existing implementations. They
also retain central-difference ladders and actual stage statuses. CPU passes do
not certify CUDA graphs, CUPTI compatibility, or speedup. The current runtime
contract does not promise forward-mode or higher derivatives through every solver.

The only non-execution bug fix is the compiled design validator's CPPN-edge test:
it now checks whether edge_mask is actually exposed in the embodiment schema.
Composite scenario decoders have a CPPN template even when another representation
is selected, so checking for a template attribute alone was incorrect.

## Primary implementation references

- Pinned XLA while runtime:
  https://raw.githubusercontent.com/openxla/xla/bb760b047bdbfeff962f0366ad5cc782c98657e0/xla/backends/gpu/runtime/while_thunk.cc
- Pinned XLA command buffer/dynamic slice:
  https://raw.githubusercontent.com/openxla/xla/bb760b047bdbfeff962f0366ad5cc782c98657e0/xla/backends/gpu/runtime/command_buffer_cmd.cc
- NVIDIA device conditional graphs:
  https://developer.nvidia.com/blog/dynamic-control-flow-in-cuda-graphs-with-conditional-nodes/
- CUPTI callback API and stable data prefix:
  https://docs.nvidia.com/cupti/api/group__CUPTI__CALLBACK__API.html
  https://docs.nvidia.com/cupti/api/structCUpti__CallbackData.html
- JAX transfer guards:
  https://docs.jax.dev/en/latest/transfer_guard.html

See RESIDENT_COMMANDS.md for focused local commands. The full historical suite
remains deferred to M5; selected affected/new tests are used in this iteration.
