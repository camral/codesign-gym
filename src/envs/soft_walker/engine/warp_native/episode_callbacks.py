"""Resident native policy/objective extension contracts and exact replay adjoints.

User callbacks are Warp kernels specialized for the owner's Model/State types.
Python selects and captures launches once; no user Python runs inside CUDA replay.
Callback arrays must follow the documented signatures and fixed dimensions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import warp as wp
from numpy.typing import ArrayLike

from .callback_kernels import callback_kernels

if TYPE_CHECKING:
    from .episode import Episode
    from .host_types import CallbackGradients, Shape
    from .solid_episode import SolidEpisode


@dataclass(frozen=True)
class NativePolicy:
    """A differentiable resident controller evaluated once per control interval.

    Kernel inputs: ``(model, state, parameters, carry, index, command, next_carry)``.
    ``index`` is a one-element int32 control index; all other arrays use the owner
    precision. Write every command and carry output, without mutating inputs.
    Commands must lie in [-1,1]. State/parameters/carry derivatives use Warp AD.
    Launch dimension may be ``"channels"``, ``"nodes"``, or an explicit shape.
    """

    kernel: wp.Kernel
    parameters: ArrayLike
    initial_carry: ArrayLike = ()
    launch_dim: Shape | str = 1


@dataclass(frozen=True)
class NativeObjective:
    """Differentiable native terminal loss and optional per-physics-step cost.

    Terminal signature: ``(model, initial, final, parameters, accumulated_cost, loss)``.
    Running signature: ``(model, initial, previous, next, command, parameters, clock, cost)``.
    Each scalar output is cleared before launch; parallel kernels may atomic-add.
    Running costs are summed, not automatically multiplied by dt. The terminal
    kernel determines how the sum contributes. Costs exclude settling by default.
    Failure/adjoint checks remain those of the enclosing physics episode.
    """

    terminal: wp.Kernel
    parameters: ArrayLike = ()
    running: wp.Kernel | None = None
    terminal_dim: Shape | str = 1
    running_dim: Shape | str = 1
    include_settling: bool = False


class EpisodeCallbacks:
    """Own callback buffers and controller history alongside one native episode."""

    def __init__(
        self,
        episode: Episode | SolidEpisode,
        policy: NativePolicy | None,
        objective: NativeObjective | None,
        memory_mib: float,
    ) -> None:
        """Allocate fixed device buffers; reject nonfinite or non-vector parameters."""
        # Both generated physics families expose the same launch/state contracts;
        # their runtime-generated Warp struct classes cannot form a Python union.
        self.episode: Any = episode
        self.policy, self.objective = policy, objective
        self.k = callback_kernels(episode.owner.precision)
        self.kernels = [self.k]
        self.cost_total = self.array(np.zeros(1))
        self.cost = self.array(np.zeros(1))
        self.cost_seed = self.array(np.zeros(1))
        if objective is not None:
            if not isinstance(objective.include_settling, bool):
                raise ValueError("include_settling must be a boolean")
            if not isinstance(objective.terminal, wp.Kernel) or (
                objective.running is not None and not isinstance(objective.running, wp.Kernel)
            ):
                raise TypeError("Objective callbacks must be Warp kernels")
            self.dimension(objective.terminal_dim)
            self.dimension(objective.running_dim)
            self.objective_parameters = self.array(objective.parameters)
            self.kernels += [objective.terminal, objective.running]
        if policy is not None:
            if not isinstance(policy.kernel, wp.Kernel):
                raise TypeError("Policy callback must be a Warp kernel")
            self.dimension(policy.launch_dim)
            self.policy_parameters = self.array(policy.parameters)
            self.initial_memory = self.array(policy.initial_carry)
            self.carry = self.array(policy.initial_carry)
            self.next_carry = self.array(policy.initial_carry)
            self.carry_seed = self.array(np.zeros(self.carry.shape))
            self.command = self.array(np.zeros(episode.owner.channels))
            self.command_seed = self.array(np.zeros(episode.owner.channels))
            self.condition = wp.zeros(1, dtype=int, device=episode.device)
            self.index = wp.zeros(1, dtype=int, device=episode.device)
            self.history = wp.zeros(
                (episode.commands.shape[0] + 1, self.carry.size),
                dtype=self.k.real,
                device=episode.device,
            )
            episode.history_bytes += self.history.size * np.dtype(episode.owner.precision).itemsize
            if episode.history_bytes > memory_mib * 1024**2:
                raise MemoryError("Controller checkpoints exceed the episode memory budget")
            self.kernels += [policy.kernel]

    def array(self, value: ArrayLike) -> wp.array:
        """Copy a finite one-dimensional host vector into differentiable device storage."""
        data = np.asarray(value, dtype=self.episode.owner.precision)
        if data.ndim != 1 or not np.all(np.isfinite(data)):
            raise ValueError("Callback parameters/carry must be finite one-dimensional arrays")
        return wp.array(data, dtype=self.k.real, device=self.episode.device, requires_grad=True)

    def dimension(self, value: Shape | str) -> Shape:
        """Resolve a symbolic launch dimension at capture setup, never during replay."""
        if isinstance(value, str):
            if value == "channels":
                return self.episode.owner.channels
            if value == "nodes":
                return self.episode.owner.n
            raise ValueError("Callback dimension must be channels, nodes, or an explicit shape")
        shape = (value,) if type(value) is int else value
        if (
            not isinstance(shape, tuple)
            or not shape
            or len(shape) > 4
            or any(type(size) is not int or size < 1 for size in shape)
        ):
            raise ValueError("Callback launch dimensions must be positive integer sizes")
        return value

    def begin_forward(self) -> None:
        """Reset running cost and controller carry at the complete-episode boundary."""
        self.cost_total.zero_()
        if self.policy is not None:
            self.history.zero_()
            if self.carry.size:
                self.episode.launch(
                    self.k.initial_carry, self.carry.size, self.initial_memory, self.history
                )

    def begin_reverse(self) -> None:
        """Reset callback cotangents before the terminal loss's reverse launch."""
        self.cost_total.grad.zero_()
        if self.objective is not None:
            self.objective_parameters.grad.zero_()
        if self.policy is not None:
            self.policy_parameters.grad.zero_()
            self.carry_seed.zero_()

    def _boundary(self) -> None:
        """Compute whether the current physics step starts a held control interval."""
        e = self.episode
        e.launch(self.k.boundary, 1, e.clock, e.settling, e.substeps, self.condition, self.index)

    def _policy_launch(self) -> None:
        """Evaluate the user policy from the restored physical state and saved carry."""
        e = self.episode
        assert self.policy is not None
        if self.carry.size:
            e.launch(self.k.load_carry, self.carry.size, self.history, self.index, self.carry)
        e.launch(
            self.policy.kernel,
            self.dimension(self.policy.launch_dim),
            e.owner.model,
            e.working,
            self.policy_parameters,
            self.carry,
            self.index,
            self.command,
            self.next_carry,
        )

    def policy_forward(self) -> None:
        """Write held command/carry tables on device at control boundaries only."""
        if self.policy is None:
            return
        e = self.episode
        self._boundary()

        def evaluate() -> None:
            """Capture one policy evaluation and validate its output on device."""
            self._policy_launch()
            e.launch(
                self.k.save_policy,
                max(e.owner.channels, self.carry.size),
                self.command,
                self.next_carry,
                self.index,
                e.commands,
                self.history,
                e.working.status,
            )

        wp.capture_if(self.condition, evaluate)

    def policy_backward(self) -> None:
        """Differentiate each held command once after all its physics-step adjoints."""
        if self.policy is None:
            return
        e = self.episode
        self._boundary()

        def reverse() -> None:
            """Seed command and recurrent-carry outputs; add state/model/parameter adjoints."""
            self.carry.grad.zero_()
            self.next_carry.grad.zero_()
            self.command.grad.zero_()
            with wp.Tape() as tape:
                self._policy_launch()
            e.launch(
                self.k.load_command_seed,
                e.owner.channels,
                e.command_gradient,
                self.index,
                self.command_seed,
            )
            tape.backward(grads={self.command: self.command_seed, self.next_carry: self.carry_seed})
            wp.copy(self.carry_seed, self.carry.grad)

        wp.capture_if(self.condition, reverse)

    def _running_launch(self) -> None:
        """Evaluate a per-step scalar without accumulating it or altering physics inputs."""
        assert self.objective is not None and self.objective.running is not None
        e = self.episode
        self.cost.zero_()
        e.launch(
            self.objective.running,
            self.dimension(self.objective.running_dim),
            e.owner.model,
            e.owner.initial,
            e.working,
            e.step.output,
            e.step.command,
            self.objective_parameters,
            e.clock,
            self.cost,
        )

    def running_forward(self) -> None:
        """Accumulate declared costs, leaving settling inclusion explicit."""
        if self.objective is None or self.objective.running is None:
            return
        e = self.episode
        self._running_launch()
        e.launch(
            self.k.accumulate_cost,
            1,
            self.cost,
            e.clock,
            e.settling,
            int(self.objective.include_settling),
            self.cost_total,
        )

    def terminal(self) -> None:
        """Write the custom terminal scalar inside the enclosing loss tape."""
        assert self.objective is not None
        e = self.episode
        e.loss.zero_()
        e.launch(
            self.objective.terminal,
            self.dimension(self.objective.terminal_dim),
            e.owner.model,
            e.owner.initial,
            e.final,
            self.objective_parameters,
            self.cost_total,
            e.loss,
        )
        e.launch(self.k.guard_loss, 1, e.final.status, e.loss)

    def running_backward(self) -> bool:
        """Add running-cost seeds before the implicit physics VJP; return whether initialized."""
        if self.objective is None or self.objective.running is None:
            return False
        e = self.episode
        e.step.clear_gradients()
        for value in vars(e.owner.initial).values():
            if isinstance(value, wp.array) and value.grad is not None:
                value.grad.zero_()
        self.cost.grad.zero_()
        with wp.Tape() as tape:
            self._running_launch()
        e.launch(
            self.k.cost_seed,
            1,
            self.cost_total.grad,
            e.clock,
            e.settling,
            int(self.objective.include_settling),
            self.cost_seed,
        )
        tape.backward(grads={self.cost: self.cost_seed})
        e.launch(e.e.add_state, e.launch_size, e.gradient_state(e.step.output), e.rolling)
        e.launch(e.e.add_state, e.launch_size, e.input_adjoint, e.initial_gradient)
        return True

    def set_parameters(
        self,
        *,
        policy: ArrayLike | None = None,
        objective: ArrayLike | None = None,
        initial_carry: ArrayLike | None = None,
    ) -> None:
        """Validate all updates before assigning into existing captured device buffers."""
        updates = []
        for name, value, attribute in (
            ("policy", policy, "policy_parameters"),
            ("objective", objective, "objective_parameters"),
            ("initial_carry", initial_carry, "initial_memory"),
        ):
            if value is None:
                continue
            target = getattr(self, attribute, None)
            if target is None:
                raise ValueError(f"No {name} callback input was configured")
            data = np.asarray(value, dtype=self.episode.owner.precision)
            if data.shape != target.shape or not np.all(np.isfinite(data)):
                raise ValueError(f"{name} parameters must preserve shape and remain finite")
            updates.append((target, data))
        for target, data in updates:
            target.assign(data)
        self.episode.completed = False
        self.episode.last_was_gradient = False

    def gradients(self) -> CallbackGradients:
        """Download callback gradients after the enclosing episode validates its adjoint."""
        result: CallbackGradients = {}
        if self.objective is not None:
            result["objective_parameters"] = self.objective_parameters.grad.numpy()
        if self.policy is not None:
            result["policy_parameters"] = self.policy_parameters.grad.numpy()
            result["policy_initial_carry"] = self.carry_seed.numpy()
        if any(not np.all(np.isfinite(value)) for value in result.values()):
            raise RuntimeError("Native callback produced nonfinite gradients")
        return result
