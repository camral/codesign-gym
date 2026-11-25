"""Shared host-side contracts without importing an accelerator runtime.

An embedding maps stable decoder parameter names to arrays. Shapes and dtypes
belong to the decoder's space; moving an embedding between host and device does
not change its keys. Array aliases deliberately accept both NumPy and JAX.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, Protocol, TypeAlias, TypedDict, TypeVar

from numpy.typing import NDArray

from ..contracts.fields import HostArray as HostArray

if TYPE_CHECKING:
    import jax

    from .drivers.task import TaskState
    from .physics import beam, fem, mpm, spring
    from .physics.common import StepInfo

# A design mixes floating fields and integer/bool topology labels. The dtype is
# supplied by its space/config, so Any applies only to the NumPy scalar dtype.
Array: TypeAlias = "jax.Array | NDArray[Any]"
Embedding: TypeAlias = dict[str, Array]
HostEmbedding: TypeAlias = dict[str, HostArray]
JsonValue: TypeAlias = "None | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]"
PathLike: TypeAlias = str | Path
Shape: TypeAlias = tuple[int, ...]

# Gym info is a heterogeneous report tree; observations remain separate arrays.
InfoValue: TypeAlias = "Array | JsonValue | tuple[InfoValue, ...] | dict[str, InfoValue]"
# Gymnasium allows backend/plugin-defined info keys and object-array lane
# masks; typed observations and rollout/state records stay outside this boundary.
Info: TypeAlias = dict[str, Any]

PhysicsModel: TypeAlias = "spring.Model | beam.Model | fem.Model | mpm.Model"
PhysicsState: TypeAlias = "spring.State | beam.State | fem.State | mpm.State"
DriverState: TypeAlias = "spring.State | TaskState"

BodyModel: TypeAlias = "beam.Model | fem.Model | mpm.Model"
BodyState: TypeAlias = "beam.State | fem.State | mpm.State"
BodyDomain: TypeAlias = "beam.Topology | fem.Mesh | mpm.Cloud"
TransitionDiagnostics: TypeAlias = "spring.Diagnostics | tuple[StepInfo, jax.Array]"
SpringSummary: TypeAlias = "tuple[dict[str, Array], Array, Array, Array, Array, Array]"
BodySummary: TypeAlias = "tuple[dict[str, Array], Array, dict[str, Array]]"
ReportingSummary: TypeAlias = "SpringSummary | BodySummary"

ObserveKernel: TypeAlias = "Callable[[PhysicsModel, DriverState], tuple[jax.Array, jax.Array]]"
SummaryKernel: TypeAlias = "Callable[[PhysicsModel, DriverState], ReportingSummary]"
RewardKernel: TypeAlias = "Callable[[PhysicsModel, DriverState, DriverState], jax.Array]"
AdvanceKernel: TypeAlias = (
    "Callable[[PhysicsModel, DriverState, jax.Array], tuple[DriverState, TransitionDiagnostics]]"
)
InitializeKernel: TypeAlias = "Callable[[PhysicsModel], DriverState]"
BodyPhysicsKernels: TypeAlias = (
    "tuple[AdvanceKernel, AdvanceKernel, InitializeKernel, SummaryKernel, RewardKernel]"
)
BodyKernels: TypeAlias = "tuple[AdvanceKernel, AdvanceKernel, InitializeKernel, ObserveKernel, SummaryKernel, RewardKernel]"

ParameterT = TypeVar("ParameterT", contravariant=True)
CarryT = TypeVar("CarryT")


class Policy(Protocol[ParameterT, CarryT]):
    """Traceable feedback controller; action channels are native or padded."""

    def __call__(
        self, parameters: ParameterT, observation: jax.Array, carry: CarryT
    ) -> tuple[jax.Array, CarryT]: ...


class RolloutResult(TypedDict, Generic[CarryT], total=False):
    """Device rollout output; optional trajectory/policy fields depend on binding."""

    state: DriverState
    metrics: dict[str, jax.Array]
    diagnostics: TransitionDiagnostics
    observations: jax.Array
    positions: jax.Array
    actions: jax.Array
    carry: CarryT
    completed: jax.Array


TransitionReport: TypeAlias = "tuple[jax.Array, jax.Array, ReportingSummary, jax.Array]"
ReportKernel: TypeAlias = "Callable[[PhysicsModel, DriverState, DriverState], TransitionReport]"
VectorStepKernel: TypeAlias = "Callable[[tuple[PhysicsModel, ...], tuple[DriverState, ...], jax.Array], tuple[tuple[DriverState, ...], tuple[TransitionReport, TransitionDiagnostics]]]"


class ExperimentResult(TypedDict, total=False):
    """Bound locomotion result; detailed history is present only when recording."""

    loss: jax.Array
    valid: jax.Array
    final_metrics: jax.Array
    state: DriverState
    metrics: jax.Array
    actions: jax.Array
    positions: jax.Array
    solver_diagnostics: jax.Array
    flight_metrics: jax.Array


ExperimentHistory: TypeAlias = "tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array] | None"
ScenarioStepData: TypeAlias = "tuple[jax.Array, jax.Array, jax.Array, jax.Array]"
ScenarioStep: TypeAlias = "Callable[[PhysicsModel, DriverState, jax.Array, jax.Array], tuple[DriverState, ScenarioStepData]]"
