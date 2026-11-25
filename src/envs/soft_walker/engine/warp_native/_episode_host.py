"""Shared host orchestration for independently owned native episodes.

Concrete constructors retain all model, state, solver and graph storage. This
class composes their existing kernel namespaces; no Warp DSL types or kernels
are constructed here, and physical step bodies remain on the concrete classes.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from types import SimpleNamespace
from typing import TYPE_CHECKING, Generic, TypeVar

import numpy as np
import warp as wp

from .host_types import MPMState, SolidState

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from .episode_callbacks import EpisodeCallbacks
    from .host_types import EpisodeGradients, EpisodeResult, Shape
    from .model import DeviceModel
    from .solid_model import SolidDeviceModel
    from .step import ImplicitStep

_State = TypeVar("_State", MPMState, SolidState)


class EpisodeHost(Generic[_State], ABC):
    """Checkpoint lifecycle over a concrete, fixed state schema.

    Subclasses allocate every attribute below per instance before capture.
    Field tuples describe existing schemas; they never create Warp structs or
    share buffers between owners. Concrete forward/reverse bodies preserve the
    physical integrator and displacement semantics.
    """

    _state_fields: tuple[str, ...]
    _model_fields: tuple[str, ...]
    _constant_model_fields: tuple[str, ...]
    _fixed_model_fields: tuple[str, ...]
    owner: DeviceModel | SolidDeviceModel
    step: ImplicitStep
    device: wp.Device
    k: SimpleNamespace
    e: SimpleNamespace
    callbacks: EpisodeCallbacks | None
    working: _State
    checkpoints: _State
    scratch: _State
    rolling: _State
    final: _State
    final_adjoint: _State
    input_adjoint: _State
    initial_gradient: _State
    model_gradient: SimpleNamespace
    model_adjoint: SimpleNamespace
    commands: wp.array
    command_gradient: wp.array
    clock: wp.array
    block: wp.array
    condition: wp.array
    inner_condition: wp.array
    block_condition: wp.array
    adjoint_status: wp.array
    replay_roots: wp.array
    replay_codes: wp.array
    diags: wp.array
    travel: wp.array
    loss: wp.array
    loss_sums: wp.array
    loss_tape: wp.Tape
    direction: object
    launch_size: int
    settling: int
    substeps: int
    steps: int
    stride: int
    history_bytes: int
    root_cache_bytes: int
    reuse_replay_roots: bool
    length_ref: float
    velocity_weight: float
    work_weight: float
    graphs: dict[str, wp.Graph]
    compile_seconds: dict[str, float]
    completed: bool
    last_was_gradient: bool

    @abstractmethod
    def forward_body(self) -> None:
        """Advance concrete physics and accumulate its displacement objective."""

    @abstractmethod
    def reverse_step(self) -> None:
        """Differentiate the concrete step and its displacement contribution."""

    def _step_kernel_namespaces(self) -> tuple[SimpleNamespace | None, ...]:
        """Select only the concrete step's modules needed before graph capture."""
        return self.step.it, getattr(self.step, "analytic", None)

    def launch(self, kernel: "wp.Kernel", count: "Shape", *args: "object") -> "None":
        """Launch on the episode device, honoring the kernel's backward-recording flag."""
        wp.launch(
            kernel,
            dim=count,
            inputs=list(args),
            device=self.device,
            record_tape=kernel.options.get("enable_backward") is not False,
        )

    def history(self, count: "int") -> "_State":
        """Allocate count checkpoint slots using this model's fixed state layout."""
        history = self.e.History()
        for name in self._state_fields:
            value = getattr(self.owner.initial, name)
            shape = count if name == "status" else (count, value.shape[0])
            setattr(history, name, wp.zeros(shape, dtype=value.dtype, device=self.device))
        return history

    def gradient_state(self, state: "_State") -> "_State":
        """Build a state struct referencing existing gradient buffers where present."""
        result = self.k.State()
        for name in self._state_fields:
            array = getattr(state, name)
            setattr(result, name, array.grad if array.grad is not None else array)
        return result

    def gradient_model(self, allocate: "bool") -> "SimpleNamespace":
        """Build a resolved-model cotangent struct, allocating only when requested."""
        result = self.k.Model()
        for name in self._model_fields:
            array = getattr(self.owner.model, name)
            value = wp.zeros_like(array, requires_grad=False) if allocate else array.grad
            setattr(result, name, value if value is not None else array)
        # Terrain is fixed during this slice; no terrain-parameter gradient claim.
        for name in self._fixed_model_fields:
            setattr(result, name, getattr(self.owner.model, name))
        return result

    def load_command(self) -> "None":
        """Load the held command at the current device clock, including settling zeros."""
        self.launch(
            self.e.load_command,
            self.owner.channels,
            self.commands,
            self.clock,
            self.settling,
            self.substeps,
            self.step.command,
        )

    def load_saved_root(self) -> "None":
        """Restore the saved converged root before device-side replay validation."""
        self.launch(
            self.e.load_root,
            self.step.capacity,
            self.replay_roots,
            self.replay_codes,
            self.clock,
            self.step.q,
            self.step.code,
        )

    def forward_program(self) -> "None":
        """Initialize and execute the complete forward rollout inside the outer graph."""
        self.step.st.work_counts.zero_()
        self.launch(self.e.copy_state, self.launch_size, self.owner.initial, self.working)
        self.clock.zero_()
        self.condition.fill_(1)
        self.diags.fill_(-1.0)
        self.travel.zero_()
        if self.callbacks is not None:
            self.callbacks.begin_forward()
        wp.capture_while(self.condition, self.forward_body)
        self.launch(self.e.copy_state, self.launch_size, self.working, self.final)
        self.loss_sums.zero_()
        if self.callbacks is not None and self.callbacks.objective is not None:
            with wp.Tape() as tape:
                self.callbacks.terminal()
            self.loss_tape = tape
            return
        with wp.Tape() as tape:
            self.launch(
                self.e.objective_particles,
                self.owner.n,
                self.owner.model,
                self.owner.initial,
                self.final,
                self.travel,
                self.direction,
                self.k.real(self.length_ref),
                self.k.real(self.velocity_weight),
                self.loss_sums,
            )
            self.launch(
                self.e.objective_finish,
                1,
                self.loss_sums,
                self.final,
                self.k.real(self.work_weight),
                self.loss,
            )
        self.loss_tape = tape

    def replay_body(self) -> "None":
        """Recompute one checkpoint-block step into device scratch for reverse mode."""
        self.launch(
            self.e.store,
            self.launch_size,
            self.scratch,
            self.working,
            self.clock,
            1,
            self.stride,
            0,
        )
        self.load_command()
        self.step.forward()
        if self.reuse_replay_roots:
            self.launch(
                self.e.save_root,
                self.step.capacity,
                self.step.q,
                self.step.code,
                self.clock,
                self.replay_roots,
                self.replay_codes,
            )
        self.launch(self.e.copy_state, self.launch_size, self.step.output, self.working)
        self.launch(
            self.e.advance_replay,
            1,
            self.clock,
            self.block,
            self.stride,
            self.steps,
            self.inner_condition,
            self.working,
        )

    def reverse_block(self) -> "None":
        """Restore, replay and differentiate one checkpoint block on device."""
        self.launch(self.e.restore, self.launch_size, self.checkpoints, self.working, self.block, 0)
        self.launch(
            self.e.begin_replay, 1, self.clock, self.block, self.stride, self.inner_condition
        )
        wp.capture_while(self.inner_condition, self.replay_body)
        self.launch(
            self.e.begin_block_reverse,
            1,
            self.clock,
            self.block,
            self.stride,
            self.steps,
            self.inner_condition,
            self.working,
            self.adjoint_status,
        )
        wp.capture_while(self.inner_condition, self.reverse_step)
        self.launch(self.e.reverse_block, 1, self.block, self.block_condition, self.adjoint_status)

    def value_gradient_program(self) -> "None":
        """Run the complete forward rollout followed by checkpointed implicit reverse mode."""
        self.forward_program()
        self.step.clear_gradients()
        for state in (self.final, self.owner.initial):
            for name in self._state_fields:
                value = getattr(state, name)
                if value.grad is not None:
                    value.grad.zero_()
        self.loss.grad.zero_()
        self.loss_sums.grad.zero_()
        self.travel.grad.zero_()
        if self.callbacks is not None:
            self.callbacks.begin_reverse()
        self.loss_tape.backward(self.loss)
        self.launch(self.e.copy_state, self.launch_size, self.final_adjoint, self.rolling)
        self.launch(self.e.copy_state, self.launch_size, self.input_adjoint, self.initial_gradient)
        for name in self._model_fields:
            value = getattr(self.model_gradient, name)
            if name not in self._constant_model_fields:
                value.zero_()
        self.launch(self.e.add_model, self.launch_size, self.model_adjoint, self.model_gradient)
        self.command_gradient.zero_()
        self.launch(
            self.e.begin_reverse,
            1,
            self.block,
            self.steps,
            self.stride,
            self.block_condition,
            self.final,
            self.adjoint_status,
        )
        wp.capture_while(self.block_condition, self.reverse_block)
        self.launch(self.e.add_state, self.launch_size, self.rolling, self.initial_gradient)
        self.launch(
            self.e.poison_gradient,
            max(self.launch_size, self.commands.size),
            self.initial_gradient,
            self.adjoint_status,
            self.command_gradient,
        )

    def compile(self, gradient: "bool" = False) -> "None":
        """CUDA compilation/capture is a setup boundary, outside measured replay."""
        if not self.device.is_cuda:
            return
        key = "gradient" if gradient else "forward"
        if key in self.graphs:
            return
        if not wp.is_conditional_graph_supported():
            raise RuntimeError("Whole-episode execution requires CUDA conditional graph support")
        start = time.perf_counter()
        program = self.value_gradient_program if gradient else self.forward_program
        from .loading import preload_kernels

        self.loaded_modules = preload_kernels(
            self.device,
            self.k,
            self.e,
            *self._step_kernel_namespaces(),
            *(self.callbacks.kernels if self.callbacks is not None else ()),
        )
        with wp.ScopedCapture(device=self.device, force_module_load=False) as capture:
            program()
        self.graphs[key] = capture.graph
        self.compile_seconds[key] = time.perf_counter() - start

    def run(self, gradient: "bool" = False) -> "EpisodeResult":
        """One whole graph submission on CUDA; no per-step Python dispatch."""
        self.compile(gradient)
        start = time.perf_counter()
        if self.device.is_cuda:
            wp.capture_launch(self.graphs["gradient" if gradient else "forward"])
        elif gradient:
            self.value_gradient_program()
        else:
            self.forward_program()
        wp.synchronize_device(self.device)
        seconds = time.perf_counter() - start
        self.completed = True
        self.last_was_gradient = bool(gradient)
        return collect_result(self, gradient, seconds)

    def gradients(self) -> "EpisodeGradients":
        """Download command/model/initial-state gradients after a successful adjoint run.

        Reject absent, failed or nonfinite derivatives. Returned host arrays have
        the same shapes as their corresponding differentiable inputs."""
        if not self.completed or not self.last_was_gradient:
            raise RuntimeError("Run a whole value-and-gradient evaluation first")
        if int(self.final.status.numpy()[0]) != 0:
            raise RuntimeError("No valid derivatives are available for a failed episode")
        if int(self.adjoint_status.numpy()[0]) != 0:
            raise RuntimeError(
                "Checkpoint replay or implicit adjoint failed; derivatives are unavailable"
            )
        result: EpisodeGradients = dict(
            commands=self.command_gradient.numpy(),
            model={
                name: getattr(self.model_gradient, name).numpy()
                for name in self._model_fields
                if name not in self._constant_model_fields
            },
            initial={
                name: getattr(self.initial_gradient, name).numpy()
                for name in self._state_fields
                if name != "status"
            },
        )
        arrays = [result["commands"], *result["model"].values(), *result["initial"].values()]
        if any(not np.all(np.isfinite(array)) for array in arrays):
            raise RuntimeError("Implicit adjoint failed; nonfinite derivatives are not usable")
        if self.callbacks is not None:
            result.update(self.callbacks.gradients())
        return result

    def set_commands(self, commands: "ArrayLike") -> "None":
        """Update inputs between complete runs without reallocating graph buffers."""
        commands = np.asarray(commands, dtype=self.owner.precision)
        if commands.shape != self.commands.shape or not np.all(np.isfinite(commands)):
            raise ValueError("Updated command table must preserve shape and be finite")
        if np.max(np.abs(commands)) > 1:
            raise ValueError("Commands are outside [-1, 1]")
        self.commands.assign(commands)
        self.completed = False
        self.last_was_gradient = False

    def set_callback_parameters(
        self,
        *,
        policy: "ArrayLike | None" = None,
        objective: "ArrayLike | None" = None,
        initial_carry: "ArrayLike | None" = None,
    ) -> "None":
        """Update fixed-shape policy/objective inputs without replacing captured buffers."""
        if self.callbacks is None:
            raise ValueError("This episode has no native callbacks")
        self.callbacks.set_parameters(
            policy=policy, objective=objective, initial_carry=initial_carry
        )


def collect_result(
    episode: EpisodeHost[MPMState] | EpisodeHost[SolidState], gradient: bool, seconds: float
) -> EpisodeResult:
    """Download completed results using the caller's scalar or batch timing.

    Synchronization, submission and completion flags remain the caller's
    responsibility so batch collection cannot serialize in-flight lanes.
    """
    return dict(
        loss=float(episode.loss.numpy()[0]),
        status=int(episode.final.status.numpy()[0]),
        adjoint_status=int(episode.adjoint_status.numpy()[0]) if gradient else None,
        execution_seconds=seconds,
        checkpoint_bytes=episode.history_bytes,
        solver_capacity=episode.step.capacity,
        steps=episode.steps,
        root_cache_bytes=episode.root_cache_bytes,
        reuse_replay_roots=episode.reuse_replay_roots,
        compile_seconds=dict(episode.compile_seconds),
    )
