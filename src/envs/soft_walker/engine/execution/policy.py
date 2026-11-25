"""Per-executable CUDA graph options, not process-wide numerical switches.

A JIT is not a promise of device-side control flow. XLA normally executes some
WHILE thunks on the host. Strict execution requests conditional CUDA graphs and
is usable only after an observed whole-executable residency certificate passes.
"""

from __future__ import annotations

from dataclasses import dataclass


class ResidencyError(RuntimeError):
    """Requested device-resident execution was not demonstrated; no fallback."""


@dataclass(frozen=True)
class ResidentPolicy:
    """Configuration for one compiled, single-device execution boundary.

    cpu is an explicitly labelled numerical reference, never a CUDA fallback.
    No policy changes physics, timesteps, solver tolerances, or solver budgets.
    """

    device: str = "cuda"
    max_recording_bytes: int = 512 * 1024**2
    deterministic_reductions: bool = False

    def __post_init__(self) -> None:
        """Reject unknown devices, nonpositive byte budgets, or nonboolean reduction policy."""
        if self.device not in ("cpu", "cuda"):
            raise ValueError("device must be cpu or cuda")
        if type(self.max_recording_bytes) is not int or self.max_recording_bytes < 1:
            raise ValueError("max_recording_bytes must be a positive integer")
        if type(self.deterministic_reductions) is not bool:
            raise ValueError("deterministic_reductions must be a boolean")

    def compiler_options(self) -> dict[str, str | bool | int]:
        """Return per-executable XLA capture settings without changing physical numerics."""
        if self.device == "cpu":
            return {}
        options: dict[str, str | bool | int] = {
            # Explicit allowlist: dynamic-slice graph commands can read offsets
            # on the CPU in the pinned XLA version. Exclude that command AND the
            # fusion that would create it. Ordinary device gather/slice remains.
            "xla_gpu_enable_command_buffer": (
                "FUSION,CUBLAS,CUBLASLT,CUDNN,CUSTOM_CALL,WHILE,CONDITIONAL"
            ),
            "xla_gpu_enable_dynamic_slice_fusion": False,
            # The separate memcpy rewriter also recognizes AD-generated scan
            # storage updates. Keep their ordinary fusion kernels so they can
            # remain inside the graph instead of emitting host copy thunks.
            "xla_disable_hlo_passes": "fusion-dynamic-memcpy-rewriter",
            "xla_gpu_command_buffer_scheduling_mode": "SERIALIZE",
            "xla_gpu_command_buffer_unroll_loops": False,
            "xla_gpu_graph_min_graph_size": 1,
            # Otherwise XLA can silently run the old host-thunk path under a
            # profiler. The certificate compares profiled/unprofiled values.
            "xla_enable_command_buffers_during_profiling": True,
        }
        if self.deterministic_reductions:
            # Duplicate-index scatter adds (including gather transposes) may
            # vary between CUDA executions. Request deterministic lowering;
            # the optimized expander avoids the much slower serial expansion.
            # This is per executable and still requires the runtime certificate.
            options["xla_gpu_deterministic_ops"] = True
            options["xla_gpu_enable_scatter_determinism_expander"] = True
        return options

    def descriptor(self) -> dict[str, object]:
        """Describe required evidence and compiler options in the persisted policy schema."""
        return {
            "device_requested": self.device,
            "deterministic_reductions": self.deterministic_reductions,
            "compiler_options": self.compiler_options(),
            "required_boundary": "one whole-episode executable invocation",
            "cuda_acceptance": {
                "host_graph_submissions": 1,
                "host_kernel_launches": 0,
                "host_loop_or_branch_events": 0,
                "device_to_host_copies_inside_invocation": 0,
                "host_to_device_copies_inside_invocation": 0,
                "simulation_callbacks": 0,
                "post_graph_notifications": (
                    "allowed only with recorded same-stream/event ordering after "
                    "the complete graph, no subsequent simulation work, and no "
                    "JAXPR effects or host-offloaded values"
                ),
            },
            "single_cuda_kernel": False,
            "execution_model": (
                "one device-controlled CUDA graph; multiple device kernels are "
                "allowed inside it, not CPU submissions between timesteps"
            ),
            "cpu_is_residency_evidence": False,
            "runtime_certificate_required": self.device == "cuda",
            "max_recording_bytes": self.max_recording_bytes,
        }
