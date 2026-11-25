"""Typed device pytrees and host evidence at the whole-episode boundary.

Episode arrays stay on the selected device until an explicit evidence download.
Metrics include time in seconds in column zero; position samples have shape
``(frames, points, dimension)`` in metres. The remaining metric columns follow
the bound scenario or locomotion protocol, rather than a shared implicit order.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, NotRequired, TypeAlias, TypedDict

from ..types import DriverState, Embedding, HostArray, PhysicsModel

if TYPE_CHECKING:
    import jax

    from .audit_types import SubmissionReport, TraceReport

Samples: TypeAlias = "dict[str, jax.Array]"
EpisodeCarry: TypeAlias = "tuple[DriverState, jax.Array]"
StepRows: TypeAlias = "tuple[jax.Array, jax.Array, jax.Array]"
RecordCarry: TypeAlias = "tuple[DriverState, jax.Array, Samples, jax.Array]"
Action: TypeAlias = "Callable[[PhysicsModel, DriverState, jax.Array, jax.Array], jax.Array]"
Measure: TypeAlias = "Callable[[PhysicsModel, DriverState], jax.Array]"
Sample: TypeAlias = "Callable[[PhysicsModel, DriverState], Samples]"
RunningCost: TypeAlias = "Callable[[PhysicsModel, DriverState, DriverState, DriverState, jax.Array, jax.Array], jax.Array]"
TerminalLoss: TypeAlias = (
    "Callable[[PhysicsModel, DriverState, DriverState, jax.Array, jax.Array], jax.Array]"
)


class EpisodeHistory(TypedDict):
    """Device recording; metrics include reset, actions/solver rows do not."""

    metrics: jax.Array
    actions: jax.Array
    solver: jax.Array
    samples: Samples
    frame_indices: jax.Array
    frames_written: jax.Array


class EpisodeOutput(TypedDict):
    """Whole-episode result; a failed physical rollout has a NaN loss."""

    state: DriverState
    metrics: jax.Array
    cost: jax.Array
    loss: jax.Array
    valid: jax.Array
    history: NotRequired[EpisodeHistory]


EpisodeFunction: TypeAlias = "Callable[[Embedding, jax.Array], EpisodeOutput]"
LossFunction: TypeAlias = "Callable[[Embedding, jax.Array], jax.Array]"
FixedEpisodeFunction: TypeAlias = "Callable[[PhysicsModel, DriverState, jax.Array], EpisodeOutput]"
FixedLossFunction: TypeAlias = "Callable[[PhysicsModel, DriverState, jax.Array], jax.Array]"
Initializer: TypeAlias = "Callable[[Embedding], tuple[PhysicsModel, DriverState]]"


class EquivalenceReport(TypedDict):
    """Host comparison after synchronization, with unchanged numerical tolerances."""

    passed: bool
    errors: list[str]
    max_absolute_difference: NotRequired[float]


class JaxprAudit(TypedDict):
    """Recursive primitive inventory and forbidden host effects."""

    primitive_counts: dict[str, int]
    forbidden: list[str]
    passed: bool


class ShapeRecord(TypedDict):
    """One flattened output leaf's allocation estimate in bytes."""

    shape: list[int]
    dtype: str
    bytes: int


class OutputSummary(TypedDict):
    """Downloaded leaf diagnostics used to reject invalid certification outputs."""

    leaves: list[dict[str, object]]
    self_check: EquivalenceReport
    external_reference: NotRequired[EquivalenceReport]


class Certificate(TypedDict, total=False):
    """Incrementally persisted certificate, including evidence on early failure."""

    passed: bool
    device: str
    phases: list[dict[str, object]]
    phase: str
    external_reference_supplied: bool
    unprofiled_repeat: EquivalenceReport
    gpu_residency_demonstrated: bool
    status: str
    alternate_buffers_checked: bool
    submission_audit: SubmissionReport
    observed_comparison: EquivalenceReport
    evidence_scope: dict[str, object]
    detailed_trace: TraceReport
    detailed_numerical_comparison: EquivalenceReport
    exception: str
    seconds_including_warmup_and_audit: float


class IterationArguments(TypedDict):
    """Optimizer callback payload; parameters are a defensive host copy."""

    iteration: int
    parameters: HostArray
    loss: float
    accepted: bool
    backtracks: int
