"""Whole-episode native execution with block checkpointing for reverse mode.

One CUDA graph encloses all timesteps and nested nonlinear/Krylov loops. A
second graph can enclose the complete value-and-gradient computation. Python
constructs graphs; it does not inspect convergence or launch one timestep at a
time during CUDA replay. Graph correctness/residency still requires GPU testing.

The objective is explicitly a development loss, not a substituted benchmark:
negative mass-weighted accumulated integration displacement / length_ref, plus declared squared-speed
and positive-work terms. Controllers are supplied as bounded command tables.
"""

from typing import TYPE_CHECKING

from ...contracts.metrics import LEDGER_WIDTH, NATIVE_DIAGNOSTIC_WIDTH

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from .episode_callbacks import NativeObjective, NativePolicy
    from .host_types import FieldArrays
    from .model import DeviceModel

import math

import numpy as np
import warp as wp

from ._episode_host import EpisodeHost
from .episode_kernels import episode_kernels
from .host_types import MPMState
from .model import MODEL_FIELDS, STATE_FIELDS
from .step import ImplicitStep


class Episode(EpisodeHost[MPMState]):
    """Own one fixed-shape MPM command rollout and its checkpointed adjoint.

    Device buffers survive repeated runs. Commands are dimensionless values
    in [-1, 1] with shape (control_intervals, channels). Model topology,
    checkpoint capacities and integration timestep remain fixed after capture."""

    _state_fields = STATE_FIELDS
    _model_fields = MODEL_FIELDS
    _constant_model_fields = ("kind", "channel")
    _fixed_model_fields = (
        "box_centers",
        "box_half",
        "box_rounding",
        "box_active",
        "plane_normal",
        "plane_offset",
    )
    owner: "DeviceModel"

    def __init__(
        self,
        model: "DeviceModel",
        commands: "ArrayLike",
        *,
        substeps: "int" = 1,
        settling_steps: "int" = 0,
        checkpoint_stride: "int" = 16,
        memory_mib: "float" = 512,
        direction: "ArrayLike | None" = None,
        length_ref: "float" = 0.2,
        velocity_weight: "float" = 0.0,
        work_weight: "float" = 0.0,
        reuse_replay_roots: "bool" = False,
        policy: "NativePolicy | None" = None,
        objective: "NativeObjective | None" = None,
    ) -> "None":
        """Allocate command, state, checkpoint and adjoint buffers before graph capture.

        The checkpoint memory budget is MiB. Capacity overflow raises MemoryError;
        there is no host-streaming fallback. The owner and its initial state must
        remain alive and keep the same array storage throughout graph replay."""
        commands = np.asarray(commands, dtype=model.precision)
        if commands.ndim != 2 or commands.shape[1] != model.channels or commands.shape[0] < 1:
            raise ValueError("Commands must have shape (control_intervals, channels)")
        if not np.all(np.isfinite(commands)) or np.max(np.abs(commands)) > 1:
            raise ValueError("Commands must be finite and bounded in [-1, 1]")
        for name, value, minimum in [
            ("substeps", substeps, 1),
            ("settling_steps", settling_steps, 0),
            ("checkpoint_stride", checkpoint_stride, 1),
        ]:
            if type(value) is not int or value < minimum:
                raise ValueError(f"Invalid {name}")
        if length_ref <= 0 or not all(
            np.isfinite(x) for x in (length_ref, velocity_weight, work_weight, memory_mib)
        ):
            raise ValueError("Invalid objective or memory scale")
        if type(reuse_replay_roots) is not bool:
            raise ValueError("reuse_replay_roots must be a boolean")
        self.reuse_replay_roots = reuse_replay_roots
        self.owner, self.device = model, model.device
        self.k, self.e = model.k, episode_kernels(model.config.dim, model.precision)
        self.substeps, self.settling = substeps, settling_steps
        self.steps = settling_steps + len(commands) * substeps
        self.stride = min(checkpoint_stride, self.steps)
        self.blocks = math.ceil(self.steps / self.stride)
        n, d, c = model.n, model.config.dim, model.channels
        bytes_per_state = (n * (2 * d + 2 * d * d) + c + LEDGER_WIDTH) * np.dtype(
            model.precision
        ).itemsize + 4
        self.history_bytes = (self.blocks + self.stride) * bytes_per_state
        if self.history_bytes > memory_mib * 1024**2:
            raise MemoryError(
                f"Checkpoint storage {self.history_bytes} exceeds configured budget; no host streaming fallback"
            )
        self.commands = wp.array(commands, dtype=self.k.real, device=self.device)
        self.command_gradient = wp.zeros_like(self.commands)
        self.working = model.empty_state()
        self.step = ImplicitStep(model, state=self.working)
        self.root_cache_bytes = 0
        if self.reuse_replay_roots:
            self.root_cache_bytes = self.stride * (
                self.step.capacity * d * np.dtype(model.precision).itemsize + 4
            )
            self.history_bytes += self.root_cache_bytes
            if self.history_bytes > memory_mib * 1024**2:
                raise MemoryError(
                    "State and root checkpoints exceed the memory budget; no host fallback"
                )
            self.replay_roots = wp.zeros(
                (self.stride, self.step.capacity), dtype=self.k.vec, device=self.device
            )
            self.replay_codes = wp.zeros(self.stride, dtype=int, device=self.device)
        self.checkpoints = self.history(self.blocks)
        self.scratch = self.history(self.stride)
        self.rolling = model.empty_state()
        self.initial_gradient = model.empty_state()
        self.final = model.empty_state()
        self.final_seed = model.empty_state()
        self.model_gradient = self.gradient_model(allocate=True)
        self.model_adjoint = self.gradient_model(allocate=False)
        self.state_adjoint = self.gradient_state(self.working)
        self.final_adjoint = self.gradient_state(self.final)
        self.input_adjoint = self.gradient_state(model.initial)
        self.launch_size = max(n, c, LEDGER_WIDTH, 2 * d)
        for name in (
            "clock",
            "block",
            "condition",
            "inner_condition",
            "block_condition",
            "adjoint_status",
        ):
            setattr(self, name, wp.zeros(1, dtype=int, device=self.device))
        self.diags = wp.full(
            (self.steps, NATIVE_DIAGNOSTIC_WIDTH), -1.0, dtype=self.k.real, device=self.device
        )
        self.travel = wp.zeros(n, dtype=self.k.vec, device=self.device, requires_grad=True)
        self.loss = wp.zeros(1, dtype=self.k.real, device=self.device, requires_grad=True)
        self.loss_sums = wp.zeros(2, dtype=self.k.real, device=self.device, requires_grad=True)
        direction = np.eye(d)[0] if direction is None else np.asarray(direction)
        if direction.shape != (d,) or not np.all(np.isfinite(direction)):
            raise ValueError("Invalid objective direction")
        self.direction = self.k.vec(*direction)
        self.length_ref, self.velocity_weight, self.work_weight = (
            length_ref,
            velocity_weight,
            work_weight,
        )
        self.graphs: dict[str, wp.Graph] = {}
        self.compile_seconds: dict[str, float] = {}
        self.completed = False
        self.last_was_gradient = False

        from .episode_callbacks import EpisodeCallbacks

        self.callbacks = (
            EpisodeCallbacks(self, policy, objective, memory_mib)
            if policy is not None or objective is not None
            else None
        )

    def forward_body(self) -> "None":
        """Perform one resident physics step and store the scheduled diagnostics/state."""
        self.launch(
            self.e.store,
            self.launch_size,
            self.checkpoints,
            self.working,
            self.clock,
            self.stride,
            0,
            1,
        )
        if self.callbacks is not None:
            self.callbacks.policy_forward()
        self.load_command()
        self.step.forward()
        if self.callbacks is not None:
            self.callbacks.running_forward()
        self.launch(
            self.e.accumulate_travel,
            self.owner.n,
            self.owner.model,
            self.step.output,
            self.k.real(self.owner.config.dt),
            self.travel,
        )
        self.launch(
            self.e.record_diag,
            1,
            self.clock,
            self.step.code,
            self.step.counts,
            self.step.residual_norm,
            self.step.max_linear,
            self.step.st.live,
            self.diags,
        )
        self.launch(self.e.copy_state, self.launch_size, self.step.output, self.working)
        self.launch(self.e.advance_clock, 1, self.clock, self.steps, self.working, self.condition)

    def reverse_step(self) -> "None":
        """Propagate one converged-root implicit adjoint and accumulate command gradients."""
        self.launch(
            self.e.restore, self.launch_size, self.scratch, self.working, self.clock, self.stride
        )
        self.load_command()

        self.step.forward(
            record_derivatives=True,
            solved_root_loader=self.load_saved_root if self.reuse_replay_roots else None,
        )
        self.launch(
            self.e.seed_travel_velocity,
            self.owner.n,
            self.owner.model,
            self.travel.grad,
            self.k.real(self.owner.config.dt),
            self.rolling,
        )
        seeds = {
            getattr(self.step.output, name): getattr(self.rolling, name)
            for name in STATE_FIELDS
            if name != "status"
        }
        cost_seeded = self.callbacks.running_backward() if self.callbacks is not None else False
        self.step.backward(seeds, clear=not cost_seeded)
        self.launch(
            self.e.check_reverse_step,
            1,
            self.step.sums,
            self.step.code,
            self.step.cfg.linear_limit,
            self.adjoint_status,
        )
        self.launch(
            self.e.add_command_gradient,
            self.owner.channels,
            self.step.command.grad,
            self.clock,
            self.settling,
            self.substeps,
            self.command_gradient,
        )
        if self.callbacks is not None:
            self.callbacks.policy_backward()
        self.launch(self.e.copy_state, self.launch_size, self.state_adjoint, self.rolling)
        self.launch(self.e.add_model, self.launch_size, self.model_adjoint, self.model_gradient)
        self.launch(
            self.e.reverse_clock,
            1,
            self.clock,
            self.block,
            self.stride,
            self.inner_condition,
            self.adjoint_status,
        )

    def recording(self) -> "FieldArrays":
        """Stored block-boundary states plus the final state, without interpolation."""
        if not self.completed:
            raise RuntimeError("Run the episode before downloading its recording")
        completed = int(np.count_nonzero(self.diags.numpy()[:, 0] == 0))
        rows = np.arange(0, completed, self.stride)
        result = {
            name: np.concatenate(
                (
                    getattr(self.checkpoints, name).numpy()[: len(rows)],
                    getattr(self.final, name).numpy()[None, ...],
                ),
                axis=0,
            )
            for name in ("x", "v", "F", "C", "actuator", "ledger")
        }
        result["physical_time"] = np.r_[rows, completed] * self.owner.config.dt
        result["accumulated_displacement"] = self.travel.numpy()
        result["diagnostics"] = self.diags.numpy()
        return result
