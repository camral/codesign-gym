"""Pure whole-episode programs, including settling and differentiable design.

No Python loop drives simulation time. The factory binds static topology and
controller semantics; embedding/model/state values remain device arguments.
Recording selects samples *on the device*. There is no streaming/offload path.
An oversized recording is rejected instead of silently splitting the episode.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, cast

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import DTypeLike

from ...contracts.status_codes import ROLLOUT_INVALID_INPUT
from ..differentiation import compiled_design_valid
from ..drivers import environment as drivers
from ..physics import spring
from ..types import (
    AdvanceKernel,
    BodyModel,
    DriverState,
    Embedding,
    PhysicsModel,
    TransitionDiagnostics,
)
from .contracts import (
    Action,
    EpisodeCarry,
    EpisodeFunction,
    EpisodeOutput,
    FixedEpisodeFunction,
    FixedLossFunction,
    Initializer,
    LossFunction,
    Measure,
    RecordCarry,
    RunningCost,
    Sample,
    Samples,
    StepRows,
    TerminalLoss,
)
from .policy import ResidencyError
from .preparation import assemble_spring

if TYPE_CHECKING:
    from ..drivers.environment import BodyConfig
    from ..drivers.spring_environment import Spring2DEnv
    from ..drivers.task import TaskState
    from ..physics.common import StepInfo
    from ..physics.terrain import Terrain, TerrainConfig


def prepare(env: Spring2DEnv, embedding: Embedding) -> tuple[PhysicsModel, DriverState]:
    """Same design decode, spawn transform, and settling as the public reset.

    Schema/topology validation belongs to setup; its JAX validity check remains
    here so an invalid differentiated candidate cannot get a favorable score.
    """
    if hasattr(env, "kind"):
        from ..body_rollout import prepare as body_prepare

        return body_prepare(cast(drivers.BodyEnv, env), embedding, cast("Terrain", env._terrain))

    model, fields = assemble_spring(
        env.decoder,
        embedding,
        env._topology,
        env.config,
        env.channels,
        getattr(env, "_terrain", None),
    )
    state = env._initialize(model)
    valid = compiled_design_valid(env, embedding, fields)
    return model, state._replace(
        status=jnp.where(valid, state.status, ROLLOUT_INVALID_INPUT).astype(jnp.int32)
    )


def stage_embedding(
    env: Spring2DEnv, embedding: Embedding, *, seed: int = 0
) -> tuple[Embedding, PhysicsModel]:
    """Validate static input and set the exact reset terrain, without simulating.

    Called outside the execution boundary. It does not mark env.reset completed.
    The complete numerical initialization/settling is inside EpisodeProgram.
    """
    from ..physics import terrain

    if not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be nonnegative")
    env.np_random = np.random.default_rng(int(seed))
    env._np_random_seed = int(seed)
    env._terrain_seed = int(env.np_random.integers(0, 2**31))
    with jax.default_device(env.device):
        env._terrain = terrain.generate(
            cast("TerrainConfig", env.config.terrain),
            cast(int, env._terrain_seed),
            len(env.config.size),
            env.dtype,
        )
    values, model, errors = env._prepare(embedding)
    if errors:
        raise ValueError("Embodiment rejected: " + ", ".join(errors))
    return cast(Embedding, values), cast(PhysicsModel, model)


@dataclass(frozen=True)
class EpisodeProgram:
    """Uncompiled functions; transform first, then compile once at the boundary.

    All functions take (embedding, controller_parameters). State preparation,
    settling, controller evaluation, substeps, objective, and reverse solves are
    included. `fixed_*` alternatives take (model, initial, parameters) for users
    who deliberately reuse a device-resident settled initial condition.
    """

    forward: EpisodeFunction
    loss: LossFunction
    record: EpisodeFunction
    fixed_forward: FixedEpisodeFunction
    fixed_loss: FixedLossFunction
    fixed_record: FixedEpisodeFunction
    initialize: Initializer
    steps: int
    frame_indices: tuple[int, ...]
    description: dict[str, object]

    def value_and_grad(self, *, argnums: int | tuple[int, ...] = 1) -> Callable[..., object]:
        """Transform the raw loss before compilation; argnums selects the design or controller.

        Embedding gradients preserve decoder parameter keys. Controller gradients have
        the same shape and selected device as the controller input."""
        return jax.value_and_grad(self.loss, argnums=argnums)

    def population(
        self, *, gradients: bool = False, batched_design: bool = False
    ) -> Callable[..., object]:
        # Different candidates are independent; simulation time is not. Designs
        # can also be batched when they share topology/capacity and dtype shapes.
        """Map independent controller candidates, optionally with matching candidate designs.

        Candidate axes are leading axes; topology and per-candidate array shapes must
        match. The resulting function still needs compilation and certification."""
        function = self.value_and_grad() if gradients else self.loss
        return jax.vmap(function, in_axes=(0 if batched_design else None, 0))


def _sample_indices(steps: int, stride: int) -> tuple[int, ...]:
    """Validate positive intervals and include both reset and final recording samples."""
    if type(steps) is not int or steps < 1:
        raise ValueError("steps must be a positive integer")
    if type(stride) is not int or stride < 1:
        raise ValueError("record_stride must be a positive integer")
    return tuple(sorted(set(range(0, steps + 1, stride)) | {steps}))


def bind_episode(
    env: Spring2DEnv,
    *,
    steps: int,
    action: Action,
    measure: Measure,
    sample: Sample,
    terminal_loss: TerminalLoss,
    running_cost: RunningCost | None = None,
    record_stride: int = 1,
    checkpoint_block: int = 10,
    description: dict[str, object] | None = None,
) -> EpisodeProgram:
    """Bind arbitrary controller/task semantics to unchanged driver stepping.

    action(model, state, parameters, index) -> bounded native action
    measure(model, state) -> a small floating diagnostic vector
    sample(model, state) -> a recording pytree (positions, optional diagnostics)
    running_cost(model, initial, previous, next, row, index) -> scalar
    terminal_loss(model, initial, final, row, accumulated_cost) -> scalar
    """
    if checkpoint_block < 1:
        raise ValueError("checkpoint_block must be positive")
    indices = _sample_indices(steps, record_stride)
    if steps > env.config.horizon:
        raise ValueError("Episode exceeds configured task horizon")
    advance: AdvanceKernel
    if hasattr(env, "kind"):
        if cast("BodyConfig", env.config).dynamics.newton.debug_linear:
            raise ResidencyError("debug_linear uses a host callback; disable it")
        advance = lambda m, s, a: drivers.advance(
            env.kind,
            cast(BodyModel, m),
            cast("TaskState", s),
            a,
            cast("BodyConfig", env.config),
            env._grid,
        )
    else:
        advance = lambda m, s, a: spring.advance(
            cast(spring.Model, m), cast(spring.State, s), a, env.config
        )

    def solver_row(information: TransitionDiagnostics, dtype: DTypeLike) -> jax.Array:
        """Normalize spring/body solver diagnostics into the existing seven-column device row."""
        if hasattr(env, "kind"):
            info = cast("tuple[StepInfo, jax.Array]", information)[0]
            return jnp.stack(
                (
                    info.code,
                    info.max_residual,
                    info.max_linear_residual,
                    info.min_J,
                    info.support_valid,
                    info.max_penetration,
                    info.nonlinear_iterations,
                )
            )
        information = cast(spring.Diagnostics, information)
        return jnp.stack(
            (
                information.solver_code,
                information.max_residual,
                jnp.asarray(-1.0, dtype),
                information.min_area_ratio,
                jnp.asarray(1.0, dtype),
                information.max_penetration,
                information.nonlinear_iterations,
            )
        )

    def one(
        model: PhysicsModel,
        initial: DriverState,
        parameters: jax.Array,
        carry: EpisodeCarry,
        index: jax.Array,
    ) -> tuple[EpisodeCarry, StepRows]:
        """Advance one valid policy interval and accumulate its scalar running objective.

        Invalid commands mark the state rejected; stopped lanes retain state and emit
        zero actions and sentinel solver rows, preserving static scan shapes."""
        state, cost = carry

        def run(_: None) -> tuple[DriverState, jax.Array, jax.Array, jax.Array, jax.Array]:
            command = action(model, state, parameters, index)
            if command.shape != (env.channels,):
                raise ValueError("Controller must return native-channel actions")
            good = jnp.all(jnp.isfinite(command)) & jnp.all(jnp.abs(command) <= 1.0)
            trial_state = state._replace(
                status=jnp.where(good, state.status, ROLLOUT_INVALID_INPUT).astype(jnp.int32)
            )
            new, info = advance(model, trial_state, command)
            row = measure(model, new)
            increment = (
                jnp.asarray(0.0, parameters.dtype)
                if running_cost is None
                else running_cost(model, initial, state, new, row, index)
            )
            return new, cost + increment, row, command, solver_row(info, parameters.dtype)

        def stopped(_: None) -> tuple[DriverState, jax.Array, jax.Array, jax.Array, jax.Array]:
            return (
                state,
                cost,
                measure(model, state),
                jnp.zeros(env.channels, parameters.dtype),
                -jnp.ones(7, parameters.dtype),
            )

        # This branch must become a device conditional inside the CUDA graph.
        # The certificate rejects any remaining host Conditional/While thunk.
        new, cost, row, command, solver = jax.lax.cond(
            (index < steps) & (state.status == 0), run, stopped, None
        )
        return (new, cost), (row, command, solver)

    def finish(
        model: PhysicsModel, initial: DriverState, final: DriverState, cost: jax.Array
    ) -> EpisodeOutput:
        """Form final metrics and reject incomplete, failed, or nonfinite physical rollouts."""
        row = measure(model, final)
        value = terminal_loss(model, initial, final, row, cost)
        expected = jnp.asarray(steps * env.config.control_dt, row.dtype)
        valid = (
            (final.status == 0)
            & jnp.isfinite(value)
            & jnp.isclose(row[0], expected, rtol=5e-4, atol=1e-6)
        )
        return {
            "state": final,
            "metrics": row,
            "cost": cost,
            "loss": jnp.where(valid, value, jnp.nan),
            "valid": valid,
        }

    def fixed_forward(
        model: PhysicsModel, initial: DriverState, parameters: jax.Array
    ) -> EpisodeOutput:
        """Run the fixed settled model entirely in checkpointed device scans.

        The scalar objective is returned with the final state; no per-step history is
        stored or downloaded by this path."""

        def block(carry: EpisodeCarry, block_index: jax.Array) -> tuple[EpisodeCarry, None]:
            """Checkpoint one fixed-size block; padded indices preserve the finished state."""

            def substep(c: EpisodeCarry, offset: jax.Array) -> tuple[EpisodeCarry, None]:
                index = block_index * checkpoint_block + offset
                c, _ = one(model, initial, parameters, c, index)
                return c, None

            # Both scans are compiled into the same entrypoint. Checkpointing
            # stores/recomputes GPU data; it never uses pinned-host offload.
            return jax.lax.scan(jax.checkpoint(substep), carry, jnp.arange(checkpoint_block))

        blocks = (steps + checkpoint_block - 1) // checkpoint_block
        (state, cost), _ = jax.lax.scan(
            jax.checkpoint(block),
            (initial, jnp.asarray(0.0, parameters.dtype)),
            jnp.arange(blocks),
        )
        return finish(model, initial, state, cost)

    def fixed_record(
        model: PhysicsModel, initial: DriverState, parameters: jax.Array
    ) -> EpisodeOutput:
        # Full-horizon metric/action history is small. Position snapshots are
        # written only at selected intervals, into a preallocated GPU buffer.
        """Record device metrics and selected position samples into fixed-capacity buffers.

        Metrics include the initialized sample; commands and solver diagnostics refer
        to completed control intervals. No Python timestep loop or host transfer runs."""
        first_sample = sample(model, initial)
        buffers = jax.tree.map(
            lambda x: jnp.zeros((len(indices),) + x.shape, x.dtype).at[0].set(x),
            first_sample,
        )
        targets = jnp.asarray(indices, jnp.int32)

        def record_one(carry: RecordCarry, index: jax.Array) -> tuple[RecordCarry, StepRows]:
            """Advance once and conditionally write the next requested frame on device."""
            state, cost, memory, next_slot = carry
            (new, cost), outputs = one(model, initial, parameters, (state, cost), index)
            save = (next_slot < len(indices)) & (
                index + 1 == targets[jnp.minimum(next_slot, len(indices) - 1)]
            )

            def write(old: Samples) -> Samples:
                values = sample(model, new)
                return jax.tree.map(lambda b, v: b.at[next_slot].set(v), old, values)

            memory = jax.lax.cond(save, write, lambda old: old, memory)
            return (new, cost, memory, next_slot + save.astype(jnp.int32)), outputs

        (final, cost, memory, written), history = jax.lax.scan(
            record_one,
            (initial, jnp.asarray(0.0, parameters.dtype), buffers, jnp.asarray(1, jnp.int32)),
            jnp.arange(steps),
        )
        result = finish(model, initial, final, cost)
        return cast(
            EpisodeOutput,
            dict(
                result,
                history={
                    "metrics": jnp.concatenate((measure(model, initial)[None], history[0])),
                    "actions": history[1],
                    "solver": history[2],
                    "samples": memory,
                    "frame_indices": targets,
                    "frames_written": written,
                },
            ),
        )

    def initialize(embedding: Embedding) -> tuple[PhysicsModel, DriverState]:
        """Decode and settle this candidate without changing the host environment lifecycle."""
        return prepare(env, embedding)

    def forward(embedding: Embedding, parameters: jax.Array) -> EpisodeOutput:
        """Evaluate a candidate embedding and controller with device-side initialization."""
        return fixed_forward(*initialize(embedding), parameters)

    def record(embedding: Embedding, parameters: jax.Array) -> EpisodeOutput:
        """Evaluate and record a candidate using the same initialization and objective."""
        return fixed_record(*initialize(embedding), parameters)

    return EpisodeProgram(
        forward,
        lambda e, p: forward(e, p)["loss"],
        record,
        fixed_forward,
        lambda m, s, p: fixed_forward(m, s, p)["loss"],
        fixed_record,
        initialize,
        steps,
        indices,
        dict(
            description or {},
            steps=steps,
            settle_steps=env.config.settle_steps,
            physical_substeps=steps * env.config.substeps,
            record_stride=record_stride,
            checkpoint_block=checkpoint_block,
            host_offload=False,
            host_timestepping=False,
        ),
    )
