"""Portable NumPy model bundles and native device allocation.



A bundle contains resolved physics, not a JAX object or a serialized executable.

The exporter is separate; importing or executing this module never imports JAX.

"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .host_types import FieldArrays, HostArray, MPMState, PathLike


from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
import warp as wp

from ...contracts.fields import POISSON_DEFAULT
from ...contracts.metrics import LEDGER_WIDTH
from ..numerics.policy import LINEAR_ACCEPTANCE_EPS_FACTOR, LINEAR_ACCEPTANCE_RTOL_FACTOR
from ..physics.construction_defaults import (
    BODY_DENSITY_KG_M3,
    PLANAR_THICKNESS_M,
    UNIFORM_MPM_STRENGTH,
    UNIFORM_MPM_YOUNG_PA,
)
from ._bundle_io import BundleMetadata, bundle_arrays, read_archive, write_archive
from ._solver_limits import mpm_support_capacity
from ._terrain import populate_terrain, validate_terrain
from .config import MPMConfig
from .kernels import kernels

MODEL_FIELDS = (
    "reference",
    "volume",
    "alpha",
    "young",
    "poisson",
    "density",
    "strength",
    "direction",
    "kind",
    "channel",
    "face_offset",
    "contact_area",
)

STATE_FIELDS = ("x", "v", "F", "C", "actuator", "ledger", "status")

# The original small, gravity-free MPM check body uses a fixed padded grid. These
# dimensions are fixture choices, not a travel-capacity policy for locomotion.
_UNIFORM_CELL_SPACING_M = 0.025
_UNIFORM_ORIGIN_M = 0.025
_UNIFORM_GRID_CELLS_PER_AXIS = 8
_UNIFORM_GRID_ORIGIN_M = -0.05


def uniform_body(
    dim: "int" = 2,
    cells: "tuple[int, ...] | None" = None,
    spacing: "float" = _UNIFORM_CELL_SPACING_M,
    particles_per_axis: "int" = 2,
    origin: "tuple[float, ...] | None" = None,
    thickness: "float" = PLANAR_THICKNESS_M,
    precision: "str" = "float64",
) -> "Bundle":
    """Build a small uniform MPM check body as owned NumPy arrays, without a device.

    ``cells`` counts cells per axis and ``particles_per_axis`` samples each cell
    axis. ``spacing``, ``origin`` and planar ``thickness`` are metres. The initial
    fields use the shared uniform-MPM construction profile: Young modulus in Pa,
    density in kg/m^3 and dimensionless strength/Poisson ratio. Fibres point +y;
    all particles use directional actuator channel zero. The returned configuration
    is gravity-free with a fixed padded grid; this is not a locomotion preset.

    Unsupported dimensions/precision, inconsistent shapes or nonpositive sampling
    scales raise ``ValueError``. This helper performs no device initialization.
    """

    if dim not in (2, 3):
        raise ValueError("Only dimensions 2 and 3 are supported")

    cells = (1,) * dim if cells is None else tuple(cells)

    origin = (_UNIFORM_ORIGIN_M,) * dim if origin is None else tuple(origin)

    if len(cells) != dim or len(origin) != dim or min(cells) < 1:
        raise ValueError("Invalid body shape")

    if (
        type(particles_per_axis) is not int
        or particles_per_axis < 1
        or spacing <= 0
        or thickness <= 0
    ):
        raise ValueError("Invalid particle spacing or thickness")

    if any(type(v) is not int for v in cells):
        raise ValueError("Cell counts must be integers")

    if precision not in ("float32", "float64"):
        raise ValueError("Unsupported precision")

    sh = tuple(v * particles_per_axis for v in cells)

    index = np.array(list(product(*(range(v) for v in sh))))

    hp = spacing / particles_per_axis

    x = (index + 0.5) * hp + origin

    n = len(x)

    factor = thickness if dim == 2 else 1.0

    offset: list[HostArray] = []

    area = np.zeros((n, 2 * dim))

    for axis in range(dim):
        for side in (-1, 1):
            face = len(offset)

            r = np.zeros(dim)

            r[axis] = side * hp / 2

            offset.append(r)

            exposed = index[:, axis] == (0 if side == -1 else sh[axis] - 1)

            area[exposed, face] = hp ** (dim - 1) * factor

    direction = np.zeros((n, dim))

    direction[:, 1] = 1.0

    fields = dict(
        reference=x,
        volume=np.full(n, hp**dim * factor),
        alpha=np.ones(n),
        young=np.full(n, UNIFORM_MPM_YOUNG_PA),
        poisson=np.full(n, POISSON_DEFAULT),
        density=np.full(n, BODY_DENSITY_KG_M3),
        strength=np.full(n, UNIFORM_MPM_STRENGTH),
        direction=direction,
        kind=np.ones(n, np.int32),
        channel=np.zeros(n, np.int32),
        face_offset=np.array(offset),
        contact_area=area,
    )

    fields = {
        key: np.asarray(value, dtype=np.int32 if key in ("kind", "channel") else precision)
        for key, value in fields.items()
    }

    return Bundle(
        fields,
        MPMConfig(
            grid_shape=(_UNIFORM_GRID_CELLS_PER_AXIS,) * dim,
            origin=(_UNIFORM_GRID_ORIGIN_M,) * dim,
            spacing=spacing,
            gravity=0,
            mass_weighted=True,
        ),
        1,
    )


@dataclass
class Bundle:
    """Portable resolved particle model and optional initial state.



    Fields use fixed particle/channel capacities and one floating precision;

    kind/channel indices are int32. Reference positions are metres, volume

    is m³, Young's modulus is pascals, and density is kg/m³. Bundles own host

    arrays; DeviceModel uploads independent device storage after validation."""

    fields: "FieldArrays"

    config: MPMConfig

    channels: int

    terrain: "FieldArrays | None" = None

    initial: "FieldArrays | None" = None

    def validate(self) -> "Bundle":
        """Validate field shapes, precision, material bounds and state consistency.



        Return this bundle for fluent construction. Raise ValueError for an

        incompatible schema, empty occupancy, invalid material, terrain or state;

        validation never repairs or clips supplied arrays."""

        if set(self.fields) != set(MODEL_FIELDS):
            raise ValueError("Resolved model fields do not match the schema")

        reference = np.asarray(self.fields["reference"])

        if reference.ndim != 2:
            raise ValueError("reference must be a particle-by-coordinate array")

        n, dim = reference.shape

        if dim != self.config.dim or n < 1 or type(self.channels) is not int or self.channels < 1:
            raise ValueError("Invalid particle dimension/count/channel count")

        _, capacity = mpm_support_capacity(n, self.config.grid_shape, self.config.compact_nodes)
        self.config.validate_solver_capacity(capacity * dim)

        dtype = np.asarray(self.fields["reference"]).dtype

        if dtype not in (np.dtype("float32"), np.dtype("float64")):
            raise ValueError("Use float32 or float64 consistently")

        shapes: dict[str, tuple[int, ...]] = {name: (n,) for name in MODEL_FIELDS}

        shapes.update(
            reference=(n, dim),
            direction=(n, dim),
            face_offset=(2 * dim, dim),
            contact_area=(n, 2 * dim),
        )

        for name, shape in shapes.items():
            value = np.asarray(self.fields[name])

            if value.shape != shape or not np.all(np.isfinite(value)):
                raise ValueError(f"Invalid {name} shape or nonfinite values")

            expected = np.dtype("int32") if name in ("kind", "channel") else dtype

            if value.dtype != expected:
                raise ValueError(f"{name} must have dtype {expected}")

        f = self.fields

        if np.any((f["alpha"] < 0) | (f["alpha"] > 1)) or np.sum(f["alpha"]) <= 0:
            raise ValueError("Invalid or empty occupancy")

        if min(np.min(f["young"]), np.min(f["density"]), np.min(f["volume"])) <= 0:
            raise ValueError("Material stiffness, density and volume must be positive")

        if np.any((f["poisson"] <= -1) | (f["poisson"] >= 0.5)):
            raise ValueError("Poisson ratio is outside compressible elasticity")

        if np.any((f["strength"] < 0) | (f["strength"] > 1)) or np.any(f["contact_area"] < 0):
            raise ValueError("Invalid actuator strength or contact area")

        if np.any((f["kind"] < 0) | (f["kind"] > 2)):
            raise ValueError("Unknown actuator type")

        if np.any((f["channel"] < 0) | (f["channel"] >= self.channels)):
            raise ValueError("Actuator channel out of range")

        if self.terrain is not None:
            validate_terrain(self.terrain, dim, normal_first=True)

        if self.initial is not None:
            if set(self.initial) != set(STATE_FIELDS):
                raise ValueError("Invalid initial-state schema")

            shapes = dict(
                x=(n, dim),
                v=(n, dim),
                F=(n, dim, dim),
                C=(n, dim, dim),
                actuator=(self.channels,),
                ledger=(LEDGER_WIDTH,),
                status=(1,),
            )

            for key, shape in shapes.items():
                value = np.asarray(self.initial[key])

                if value.shape != shape or not np.all(np.isfinite(value)):
                    raise ValueError(f"Invalid initial {key} shape or values")

            if int(self.initial["status"][0]) != 0:
                raise ValueError("Initial state must be valid, not a previous failed state")

            if np.any(np.abs(self.initial["actuator"]) > 1):
                raise ValueError("Initial actuator state must be bounded")

        return self

    def save(self, path: "PathLike") -> "None":
        """Write validated arrays and schema metadata to a pickle-free NPZ archive."""

        self.validate()

        path = Path(path)

        path.parent.mkdir(parents=True, exist_ok=True)

        metadata: BundleMetadata = {
            "schema": 1,
            "channels": self.channels,
            "config": self.config.to_dict(),
        }

        arrays = bundle_arrays(self.fields, self.terrain, self.initial)

        write_archive(path, metadata, arrays)

    @classmethod
    def load(cls, path: "PathLike") -> "Bundle":
        """Read a schema-1 NPZ archive, copy its arrays and validate the result."""

        meta, groups = read_archive(path, schema=1, schema_error="Unsupported model-bundle version")

        return cls(
            groups["model"],
            MPMConfig(**meta["config"]),
            meta["channels"],
            groups["terrain"] or None,
            groups["state"] or None,
        ).validate()


class DeviceModel:
    """Device arrays are authoritative after construction; no NumPy stepping."""

    def __init__(self, bundle: Bundle, device: "str | wp.Device" = "cpu") -> "None":
        """Validate a bundle and allocate independent model/state buffers on device."""

        bundle.validate()

        wp.init()

        self.device = wp.get_device(device)

        if str(device).startswith(("cuda", "gpu")) and not self.device.is_cuda:
            raise RuntimeError("Requested CUDA device is unavailable")

        self.config = bundle.config

        self.precision = str(bundle.fields["reference"].dtype)

        self.k = kernels(self.config.dim, self.precision)

        k = self.k

        self.model = k.Model()

        self.n = len(bundle.fields["reference"])

        self.channels = bundle.channels

        self.parameters = k.Parameters()

        cfg, p = self.config, self.parameters

        p.shape, p.origin = k.ivec(*cfg.grid_shape), k.vec(*cfg.origin)

        p.h, p.dt, p.gravity, p.axis = cfg.spacing, cfg.dt, cfg.gravity, cfg.vertical_axis

        p.velocity_ref, p.tau = cfg.velocity_ref, cfg.actuator_tau

        p.log_stretch, p.stress_ratio = cfg.max_log_stretch, cfg.active_stress_ratio

        p.stress_mode, p.weighted = (
            int(cfg.actuation_model == "active_stress"),
            int(cfg.mass_weighted),
        )

        p.min_J, p.contact, p.stiffness = cfg.min_J, int(cfg.contact_enabled), cfg.contact_stiffness

        p.width, p.friction, p.velocity_width = cfg.contact_width, cfg.friction, cfg.friction_width

        p.atol = cfg.atol64 if self.precision == "float64" else cfg.atol32

        rt = cfg.linear_rtol64 if self.precision == "float64" else cfg.linear_rtol32

        p.linear_limit = (
            LINEAR_ACCEPTANCE_RTOL_FACTOR * rt
            + LINEAR_ACCEPTANCE_EPS_FACTOR * np.finfo(self.precision).eps
        )

        p.max_newton, p.max_backtracks, p.armijo = cfg.max_newton, cfg.max_backtracks, cfg.armijo

        for name, value in bundle.fields.items():
            dtype = (
                int
                if name in ("kind", "channel")
                else (k.vec if name in ("reference", "direction", "face_offset") else k.real)
            )

            setattr(
                self.model,
                name,
                wp.array(value, dtype=dtype, device=self.device, requires_grad=(dtype is not int)),
            )

        populate_terrain(self.model, bundle.terrain, cfg, k, self.device)

        self.initial = self.state(bundle.initial)

    def state(self, values: "FieldArrays | None" = None) -> "MPMState":
        """Allocate particle state from host arrays or the undeformed rest state.



        Host arrays are copied to this model's device. Vector fields have shape

        (particles, dim), matrices (particles, dim, dim); no stepping occurs."""

        k, n, d = self.k, self.n, self.config.dim

        if values is None:
            values = dict(
                x=self.model.reference.numpy(),
                v=np.zeros((n, d)),
                F=np.tile(np.eye(d), (n, 1, 1)),
                C=np.zeros((n, d, d)),
                actuator=np.zeros(self.channels),
                ledger=np.zeros(LEDGER_WIDTH),
                status=np.zeros(1, np.int32),
            )

        state = k.State()

        for name in STATE_FIELDS:
            dtype = (
                int
                if name == "status"
                else (k.vec if name in ("x", "v") else k.mat if name in ("F", "C") else k.real)
            )

            value = np.asarray(values[name]).reshape(1) if name == "status" else values[name]

            setattr(
                state,
                name,
                wp.array(value, dtype=dtype, device=self.device, requires_grad=(dtype is not int)),
            )

        return state

    def empty_state(self) -> "MPMState":
        """Allocate zeroed device buffers matching the initial state and gradients."""

        state = self.k.State()

        for name in STATE_FIELDS:
            reference = getattr(self.initial, name)

            setattr(state, name, wp.zeros_like(reference, requires_grad=reference.requires_grad))

        return state

    @staticmethod
    def download(state: "MPMState") -> "FieldArrays":
        """Synchronize and copy every particle-state field into host NumPy arrays."""

        return {name: getattr(state, name).numpy() for name in STATE_FIELDS}
