"""Portable resolved Lagrangian bundles and native allocation (no JAX import)."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import SimpleNamespace

    from .host_types import FieldArrays, HostArray, PathLike, SolidState


from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import warp as wp

from ...contracts.metrics import LEDGER_WIDTH
from ..numerics.policy import LINEAR_ACCEPTANCE_EPS_FACTOR, LINEAR_ACCEPTANCE_RTOL_FACTOR
from ._bundle_io import BundleMetadata, bundle_arrays, read_archive, write_archive
from ._terrain import populate_terrain, validate_terrain
from .config import MPMConfig
from .solid_kernels import solid_kernels

STATE_FIELDS = ("u", "v", "rotation", "omega", "actuator", "ledger", "status")

ARRAY_FIELDS = (
    "reference",
    "mass",
    "inertia",
    "free",
    "entities",
    "cells",
    "cell_alpha",
    "stiffness",
    "damping",
    "length",
    "routing",
    "extension",
    "bending",
    "axis",
    "young",
    "poisson",
    "strength",
    "direction",
    "kind",
    "channel",
    "alpha",
    "gradients",
    "weights",
    "contact_area",
    "offsets",
)

INTEGER_FIELDS = ("entities", "cells", "kind", "channel")


@dataclass(frozen=True)
class SolidConfig(MPMConfig):
    """Fixed-step matched solid numerics. Inherited grid fields encode dimension

    only; these models allocate mesh DOFs, not a background MPM grid.

    """

    physics: str = "spring"

    parameter_adjoint: str = "velocity"  # solids use their own nodal residual

    krylov_method: str = "adaptive_dynamic"

    cache_tangent: bool = True

    integrator: str = "backward_euler"

    tangent_assembly: str = "auto"  # analytic springs/Hex8, AD for beams

    min_length_ratio: float = 0.1

    min_area_ratio: float = 0.03

    legacy_spring_contact: bool = True

    # Resolved parameter gradients can subtract much larger force sensitivities.

    # Use a stricter float64 transpose target without tightening forward Newton.

    adjoint_rtol64: float = 2e-11

    adjoint_rtol32: float = 2e-5

    def __post_init__(self) -> "None":
        """Reject unsupported solid integrators, tangents and actuation settings."""

        super().__post_init__()

        if self.adjoint_rtol64 <= 0 or self.adjoint_rtol32 <= 0:
            raise ValueError("Adjoint relative tolerances must be positive")

        if self.tangent_assembly not in ("auto", "analytic", "autodiff"):
            raise ValueError("tangent_assembly must be auto, analytic, or autodiff")

        if self.physics == "beam" and self.tangent_assembly == "analytic":
            raise ValueError("The beam tangent currently requires autodiff")

        if self.parameter_adjoint != "velocity":
            raise ValueError("Mass-form grid parameter adjoints apply only to MPM")

        if self.physics not in ("spring", "beam", "fem"):
            raise ValueError("physics must be spring, beam, or fem")

        if self.dim != (2 if self.physics == "spring" else 3):
            raise ValueError("Wrong physical dimension")

        if self.krylov_method != "adaptive_dynamic" or not self.cache_tangent:
            raise ValueError("Lagrangian ports use cached local blocks and adaptive_dynamic GMRES")

        if self.compact_nodes or self.block_preconditioner or self.mass_weighted:
            raise ValueError("Compact-grid, mass-weighted and node-block options apply to MPM only")

        if self.integrator not in ("backward_euler", "kvaerno3_native") or (
            self.integrator == "kvaerno3_native" and self.physics != "spring"
        ):
            raise ValueError(
                "Select backward_euler, or kvaerno3_native for spring; integrators are never substituted"
            )

        if self.actuation_model != "active_strain" or self.actuator_tau <= 0:
            raise ValueError("The matched Lagrangian laws require active strain and positive tau")

        if not 0 < self.min_length_ratio < 1 or not 0 < self.min_area_ratio < 1:
            raise ValueError("Invalid geometric admissibility limits")


@dataclass
class SolidBundle:
    """Resolved nodal/element arrays and matched solid numerical configuration.



    Reference lengths and h are metres, area_ref is m², masses are kg, and

    material moduli are pascals. Integer connectivity uses int32 indices.

    Topology and capacities remain fixed after uploading a DeviceModel."""

    fields: "FieldArrays"

    config: SolidConfig

    channels: int

    h: float

    area_ref: float

    terrain: "FieldArrays | None" = None

    initial: "FieldArrays | None" = None

    @property
    def precision(self) -> "str":
        """Floating precision shared by all physical arrays in this bundle."""

        return str(np.asarray(self.fields["reference"]).dtype)

    def validate(self) -> "SolidBundle":
        """Validate all native-indexed storage before any device allocation.



        Resolved coefficients are independent physical inputs: this does not

        reconstruct masses/contact/routing from a high-level design, or impose

        connectivity/occupancy coupling absent from the resolved API.

        """

        if set(self.fields) != set(ARRAY_FIELDS):
            raise ValueError(
                f"Resolved solid schema mismatch: {set(self.fields) ^ set(ARRAY_FIELDS)}"
            )

        f = {key: np.asarray(value) for key, value in self.fields.items()}

        reference, entities, cells = f["reference"], f["entities"], f["cells"]

        d, physics = self.config.dim, self.config.physics

        if reference.ndim != 2 or reference.shape[1] != d or len(reference) < 1:
            raise ValueError("reference must be a nonempty node-by-coordinate array")

        dtype = reference.dtype

        if dtype not in (np.dtype("float32"), np.dtype("float64")):
            raise ValueError("Use float32 or float64 consistently")

        if type(self.channels) is not int or self.channels < 1:
            raise ValueError("Invalid channel capacity")

        n = len(reference)
        scalar_dofs = n * (6 if physics == "beam" else d)
        self.config.validate_solver_capacity(scalar_dofs)

        if entities.ndim != 2 or entities.shape[1] != (8 if physics == "fem" else 2):
            raise ValueError("Invalid interaction connectivity shape")

        ne = len(entities)

        if physics == "fem" and ne == 0:
            raise ValueError("FEM requires at least one element")

        if cells.ndim != 2:
            raise ValueError("cells must be a two-dimensional connectivity array")

        if physics == "spring":
            if cells.shape[1] != 4 and cells.shape != (0, 0):
                raise ValueError("Spring admissibility cells require four vertices")

        elif cells.shape != (0, 0):
            raise ValueError("Non-spring cells use the empty (0, 0) placeholder")

        expected: dict[str, tuple[int, ...]] = {name: (ne,) for name in ARRAY_FIELDS}

        expected.update(
            reference=(n, d),
            mass=(n,),
            inertia=(n,),
            free=(n, d),
            entities=(ne, 8 if physics == "fem" else 2),
            cells=cells.shape,
            cell_alpha=(len(cells),),
            routing=(ne, self.channels),
            extension=(ne, d, d),
            bending=(ne, d, d),
            axis=(ne, d),
            direction=(ne, d),
            gradients=(8, 8, d) if physics == "fem" else (0, 0, d),
            weights=(ne, 8) if physics == "fem" else (ne, 0),
            contact_area=(n, 6 if physics == "beam" else 1),
            offsets=(6, d) if physics == "beam" else (0, d),
        )

        for key, value in f.items():
            expected_dtype = np.dtype("int32") if key in INTEGER_FIELDS else dtype

            if (
                value.shape != expected[key]
                or value.dtype != expected_dtype
                or not np.all(np.isfinite(value))
            ):
                raise ValueError(
                    f"Invalid {key}: expected {expected[key]} with finite {expected_dtype} values"
                )

        for name in ("entities", "cells"):
            if f[name].size and (f[name].min() < 0 or f[name].max() >= n):
                raise ValueError(f"Out-of-range {name} connectivity")

        mass_sum: HostArray = np.sum(f["mass"], dtype=np.float64)

        if (
            np.any(f["mass"] < 0)
            or not np.isfinite(mass_sum)
            or mass_sum <= 0
            or np.any(f["contact_area"] < 0)
        ):
            raise ValueError("Invalid mass/contact measure or empty body")

        if np.any((f["free"] != 0) & (f["free"] != 1)):
            raise ValueError("free must be a zero/one constraint mask")

        if physics == "spring":
            if np.any(f["stiffness"] < 0) or np.any(f["damping"] < 0) or np.any(f["length"] <= 0):
                raise ValueError(
                    "Spring stiffness/damping must be nonnegative and reference lengths positive"
                )

            if np.any((f["cell_alpha"] < 0) | (f["cell_alpha"] > 1)):
                raise ValueError("Invalid spring cell occupancy")

        if physics == "beam":
            if np.any((f["mass"] > 0) & (f["inertia"] <= 0)):
                raise ValueError("Active beam nodes require positive inertia")

            if ne and not np.allclose(np.linalg.norm(f["axis"], axis=1), 1.0, rtol=1e-6, atol=1e-7):
                raise ValueError("Beam reference axes must have unit length")

            # Forces are the derivative of the stated quadratic beam energy

            # only for symmetric elastic matrices. Zero-stiffness links remain

            # supported, as do roundoff-size eigenvalues near the PSD boundary.

            for name in ("extension", "bending"):
                matrices = f[name].astype(np.float64)

                scales = np.max(np.abs(matrices), axis=(1, 2), initial=0.0)

                tolerance = 64 * np.finfo(dtype).eps * scales

                skew = np.max(
                    np.abs(matrices - matrices.transpose(0, 2, 1)), axis=(1, 2), initial=0.0
                )

                if np.any(skew > tolerance):
                    raise ValueError(f"Beam {name} matrices must be symmetric")

                if np.any(np.linalg.eigvalsh(matrices) < -tolerance[:, None]):
                    raise ValueError(f"Beam {name} matrices must be positive semidefinite")

        if physics == "fem":
            if (
                np.any(f["young"] <= 0)
                or np.any(f["weights"] <= 0)
                or np.any((f["poisson"] <= -1) | (f["poisson"] >= 0.5))
            ):
                raise ValueError("Invalid material/quadrature")

            if np.any((f["alpha"] < 0) | (f["alpha"] > 1)):
                raise ValueError("Invalid element occupancy")

            if np.any((f["strength"] < 0) | (f["strength"] > 1)):
                raise ValueError("Actuator strength must be in [0,1]")

            if np.any((f["kind"] < 0) | (f["kind"] > 2)):
                raise ValueError("Unknown actuator type")

            if np.any((f["channel"] < 0) | (f["channel"] >= self.channels)):
                raise ValueError("Channel outside capacity")

        if (
            not np.isfinite(self.h)
            or self.h <= 0
            or not np.isfinite(self.area_ref)
            or self.area_ref <= 0
        ):
            raise ValueError("Invalid reference scale")

        if self.terrain is not None:
            validate_terrain(self.terrain, d, normal_first=False)

        if self.initial is not None:
            shapes = dict(
                u=(n, d),
                v=(n, d),
                rotation=(n, d, d),
                omega=(n, d),
                actuator=(self.channels,),
                ledger=(LEDGER_WIDTH,),
                status=(1,),
            )

            if set(self.initial) != set(shapes):
                raise ValueError("State schema mismatch")

            for key, shape in shapes.items():
                value = np.asarray(self.initial[key])

                expected_dtype = np.dtype("int32") if key == "status" else dtype

                if (
                    value.shape != shape
                    or value.dtype != expected_dtype
                    or not np.all(np.isfinite(value))
                ):
                    raise ValueError(f"Invalid initial {key}")

            if int(self.initial["status"][0]) != 0 or np.any(np.abs(self.initial["actuator"]) > 1):
                raise ValueError("Failed initial state or unbounded actuator")

        return self

    def save(self, path: "PathLike") -> "None":
        """Write validated solid fields, state and metadata to a pickle-free NPZ."""

        self.validate()

        metadata: BundleMetadata = dict(
            schema="warp-solid-v1",
            config=asdict(self.config),
            channels=self.channels,
            h=self.h,
            area_ref=self.area_ref,
        )

        arrays = bundle_arrays(self.fields, self.terrain, self.initial)

        path = Path(path)

        path.parent.mkdir(parents=True, exist_ok=True)

        write_archive(path, metadata, arrays)

    @classmethod
    def load(cls, path: "PathLike") -> "SolidBundle":
        """Read an archived solid schema into independent arrays and validate it."""

        meta, groups = read_archive(
            path, schema="warp-solid-v1", schema_error="Not a native Lagrangian bundle"
        )

        return cls(
            groups["model"],
            SolidConfig(**meta["config"]),
            meta["channels"],
            meta["h"],
            meta["area_ref"],
            groups["terrain"] or None,
            groups["state"] or None,
        ).validate()


def terrain_struct(
    bundle: "SolidBundle", k: "SimpleNamespace", device: "str | wp.Device"
) -> "SimpleNamespace":
    """Allocate this bundle's fixed plane/box terrain on the selected device."""

    t = k.Terrain()

    populate_terrain(t, bundle.terrain, bundle.config, k, device)

    for name, member in k.Terrain.vars.items():
        if name.startswith(("_", "box_", "plane_")):
            continue

        typ = member.type

        if hasattr(typ, "dtype"):
            setattr(
                t, name, wp.zeros((0,) * getattr(typ, "ndim", 1), dtype=typ.dtype, device=device)
            )

    return t


class SolidDeviceModel:
    """Device-owned solid model and initial state with fixed buffer capacities."""

    def __init__(self, bundle: "SolidBundle", device: "str | wp.Device" = "cpu") -> "None":
        """Validate and upload a resolved spring, beam or FEM model."""

        bundle.validate()

        wp.init()

        self.bundle, self.config = bundle, bundle.config

        self.device = wp.get_device(device)

        self.precision = bundle.precision

        self.k = k = solid_kernels(bundle.config.physics, self.precision)

        self.n, self.ne, self.channels = (
            len(bundle.fields["reference"]),
            len(bundle.fields["entities"]),
            bundle.channels,
        )

        self.model = k.Model()

        for name, value in bundle.fields.items():
            dtype = k.real

            if name in INTEGER_FIELDS:
                dtype = int

            elif name in ("reference", "free", "axis", "direction", "gradients", "offsets"):
                dtype = k.vec

            elif name in ("extension", "bending"):
                dtype = k.mat

            setattr(
                self.model,
                name,
                wp.array(
                    value,
                    dtype=dtype,
                    device=self.device,
                    requires_grad=name not in INTEGER_FIELDS + ("free",),
                ),
            )

        self.model.h, self.model.area_ref = bundle.h, bundle.area_ref

        self.model.min_length_ratio, self.model.min_area_ratio = (
            self.config.min_length_ratio,
            self.config.min_area_ratio,
        )

        self.model.legacy_contact = int(
            self.config.physics == "spring"
            and self.config.legacy_spring_contact
            and bundle.terrain is None
        )

        self.model.terrain = terrain_struct(bundle, k, self.device)

        p = self.parameters = k.Parameters()

        cfg = self.config

        p.shape, p.origin = k.base.ivec(*cfg.grid_shape), k.vec(*cfg.origin)

        p.h, p.dt, p.gravity, p.axis = bundle.h, cfg.dt, cfg.gravity, cfg.vertical_axis

        p.velocity_ref, p.tau, p.log_stretch = (
            cfg.velocity_ref,
            cfg.actuator_tau,
            cfg.max_log_stretch,
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

        self.initial = self.state(bundle.initial)

    def state(self, values: "FieldArrays | None" = None) -> "SolidState":
        """Allocate validated nodal state, defaulting to zero displacement/velocity.



        Rotations default to identity. Input arrays are copied, and invalid

        shapes, nonfinite entries or a failed initial status raise ValueError."""

        k, n, d = self.k, self.n, self.config.dim

        if values is None:
            values = dict(
                u=np.zeros((n, d), self.precision),
                v=np.zeros((n, d), self.precision),
                rotation=np.tile(np.eye(d, dtype=self.precision), (n, 1, 1)),
                omega=np.zeros((n, d), self.precision),
                actuator=np.zeros(self.channels, self.precision),
                ledger=np.zeros(LEDGER_WIDTH, self.precision),
                status=np.zeros(1, np.int32),
            )

        expected = dict(
            u=(n, d),
            v=(n, d),
            rotation=(n, d, d),
            omega=(n, d),
            actuator=(self.channels,),
            ledger=(LEDGER_WIDTH,),
            status=(1,),
        )

        if set(values) != set(expected):
            raise ValueError("State schema mismatch")

        out = k.State()

        for name in STATE_FIELDS:
            a = np.asarray(values[name])

            if a.shape != expected[name] or not np.all(np.isfinite(a)):
                raise ValueError(f"Invalid initial {name}")

            dtype = (
                int
                if name == "status"
                else k.mat
                if name == "rotation"
                else k.vec
                if name in ("u", "v", "omega")
                else k.real
            )

            setattr(
                out,
                name,
                wp.array(a, dtype=dtype, device=self.device, requires_grad=name != "status"),
            )

        if int(np.asarray(values["status"])[0]) != 0 or np.any(np.abs(values["actuator"]) > 1):
            raise ValueError("Failed initial state or unbounded actuator")

        return out

    def empty_state(self) -> "SolidState":
        """Allocate zeroed arrays matching the initial nodal state."""

        out = self.k.State()

        for name in STATE_FIELDS:
            setattr(out, name, wp.zeros_like(getattr(self.initial, name)))

        return out

    @staticmethod
    def numpy_state(state: "SolidState") -> "FieldArrays":
        """Synchronize and copy nodal state arrays back to host memory."""

        return {name: getattr(state, name).numpy() for name in STATE_FIELDS}
