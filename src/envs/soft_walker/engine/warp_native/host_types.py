"""Host-side contracts for native Warp execution, without importing JAX.

Field dictionaries contain independently owned NumPy arrays at import/reset
boundaries. Device states expose Warp arrays; reading them with ``numpy()`` is
an explicit synchronization boundary. Shapes depend on fixed model capacities.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, NotRequired, Protocol, TypeAlias, TypedDict

from numpy.typing import NDArray

if TYPE_CHECKING:
    import warp as wp

    from .episode_callbacks import NativeObjective, NativePolicy
    from .model import Bundle
    from .solid_model import SolidBundle


HostArray: TypeAlias = NDArray[Any]  # Physical and integer field dtypes vary by schema.
FieldArrays: TypeAlias = dict[str, HostArray]
PathLike: TypeAlias = str | Path
Shape: TypeAlias = int | tuple[int, ...]
ReportValue: TypeAlias = (
    None | bool | int | float | str | list["ReportValue"] | dict[str, "ReportValue"]
)
Report: TypeAlias = dict[str, ReportValue]


class EpisodeResult(TypedDict):
    """Downloaded scalar results; timings are seconds, capacities are bytes/counts."""

    loss: float
    status: int
    adjoint_status: int | None
    execution_seconds: float
    checkpoint_bytes: int
    solver_capacity: int
    steps: int
    root_cache_bytes: int
    reuse_replay_roots: bool
    compile_seconds: dict[str, float]
    execution_mode: NotRequired[str]
    batch_execution_seconds: NotRequired[float]
    batch_size: NotRequired[int]


class EpisodeGradients(TypedDict):
    """Host cotangents matching command, resolved-model and initial-state shapes."""

    commands: HostArray
    model: FieldArrays
    initial: FieldArrays
    policy_parameters: NotRequired[HostArray]
    policy_initial_carry: NotRequired[HostArray]
    objective_parameters: NotRequired[HostArray]


class CallbackGradients(TypedDict, total=False):
    """Optional native callback inputs, differentiated at fixed array shape."""

    policy_parameters: HostArray
    policy_initial_carry: HostArray
    objective_parameters: HostArray


class MPMState(Protocol):
    """Device particle state: vectors ``(n, d)``, matrices ``(n, d, d)``.

    Actuator state has one entry per channel. Ledger entries are signed work,
    positive work, dissipated energy (joules), and physical time (seconds).
    Status has one int32 entry and zero means success.
    """

    x: wp.array
    v: wp.array
    F: wp.array
    C: wp.array
    actuator: wp.array
    ledger: wp.array
    status: wp.array


class SolidState(Protocol):
    """Device nodal displacement/velocity, rotations, actuator state and ledger.

    ``u``, ``v`` and ``omega`` have shape ``(nodes, dim)``; rotations have shape
    ``(nodes, dim, dim)``. Ledger/status use the same convention as ``MPMState``.
    """

    u: wp.array
    v: wp.array
    rotation: wp.array
    omega: wp.array
    actuator: wp.array
    ledger: wp.array
    status: wp.array


class MetricValues(TypedDict):
    """Accumulated displacement in metres and positive actuator work in joules."""

    displacement: float
    positive_work: float


class NativeInfo(TypedDict, total=False):
    """Per-lane public metadata; rejected resets report errors instead of metrics."""

    action_mask: HostArray
    observation_mask: HostArray
    observation_profile: str
    metrics: MetricValues
    objectives: HostArray
    valid: bool
    embodiment_errors: list[str]
    numerics: Report
    execution_seconds: float
    execution_mode: str
    batch_execution_seconds: float
    batch_size: int
    autoreset: bool


class VectorInfo(TypedDict):
    """Metadata remains per lane in the native co-design vector API."""

    per_env: list[NativeInfo]


class NativeOptions(TypedDict, total=False):
    """Keyword options forwarded unchanged to ``NativeEnv`` by factories."""

    bundle: Bundle | SolidBundle | None
    integrator: str | None
    dtype: str
    device: str
    max_action_space_size: int | None
    max_observation_space_size: int | None
    is_eval_env: bool
    substeps: int | None
    settling_steps: int
    horizon: int
    length_ref: float
    work_ref: float
    energy_weight: float
    render_mode: str
    observation_profile: str
    physics_steps_per_action: int | None


class NativeVectorOptions(NativeOptions, total=False):
    """Scalar options plus the independent-lane execution policy."""

    execution: str


class NativeFactoryOptions(NativeVectorOptions, total=False):
    """Engine factory keywords, consumed before constructing native lanes."""

    backend: str
    vectorization_mode: str


class EpisodeOptions(TypedDict, total=False):
    """Fixed graph shape, checkpoint budget and objective configuration."""

    substeps: int
    settling_steps: int
    checkpoint_stride: int
    memory_mib: float
    direction: HostArray | None
    length_ref: float
    velocity_weight: float
    work_weight: float
    reuse_replay_roots: bool
    policy: NativePolicy | None
    objective: NativeObjective | None


ScalarReset: TypeAlias = tuple[HostArray, HostArray, NativeInfo]
ScalarTransition: TypeAlias = tuple[HostArray, float, bool, bool, NativeInfo]
VectorReset: TypeAlias = tuple[HostArray, HostArray, VectorInfo]
VectorTransition: TypeAlias = tuple[HostArray, HostArray, HostArray, HostArray, VectorInfo]
RootLoader: TypeAlias = Callable[[], None]


class SolidExportOptions(TypedDict, total=False):
    """Numerical settings copied from matched JAX solid configuration."""

    actuator_tau: float
    integrator: str
    armijo: float
    atol32: float
    atol64: float
    cache_tangent: bool
    contact_enabled: bool
    contact_height: float
    contact_stiffness: float
    contact_width: float
    dt: float
    friction: float
    friction_width: float
    gravity: float
    grid_shape: tuple[int, ...]
    krylov_method: str
    linear_refinements: int
    linear_rtol32: float
    linear_rtol64: float
    max_backtracks: int
    max_log_stretch: float
    max_newton: int
    max_restarts: int
    min_J: float
    min_area_ratio: float
    min_length_ratio: float
    origin: tuple[float, ...]
    physics: str
    restart: int
    velocity_ref: float
    vertical_axis: int
