"""Hard biped/quadruped fixtures with resolution-independent chamber semantics.

+x is travel, +y is vertical, and +z is lateral. All initial particles use the
existing uniform reference lattice. The decoder only masks that lattice; it
neither moves particles nor introduces fictitious external actuation forces.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import DTypeLike

from envs.soft_walker.engine import make
from envs.soft_walker.engine.drivers.spring_environment import Spring2DEnv
from envs.soft_walker.engine.drivers.task import BodyConfig
from envs.soft_walker.engine.embodiments.decoders import Fields, FieldSpec, interpolate
from envs.soft_walker.engine.numerics.solvers import NewtonConfig
from envs.soft_walker.engine.physics.common import DynamicsConfig
from envs.soft_walker.engine.physics.contact import PlaneContact
from envs.soft_walker.engine.physics.spring_config import SpringConfig
from envs.soft_walker.engine.spaces import Box, Dict
from envs.soft_walker.engine.types import Embedding, HostEmbedding

ENV_IDS = {
    "spring": "SoftSpring2D-v0",
    "mpm2": "SoftMPM2D-v0",
    "mpm3": "SoftMPM3D-v0",
    "beam": "SoftBeam3D-v0",
    "fem": "SoftHexFEM3D-v0",
}


@dataclass(frozen=True)
class DemoConfig:
    """Serializable physical/numerical protocol, independent of the optimizer.

    The refinement factor subdivides the same reference envelope and chambers.
    Numerical settings are recorded, never optimized implicitly. A longer replay
    may require a larger fixed MPM grid margin; exhausted support is a failure,
    not a wall or an invitation to teleport/recenter the robot.
    """

    driver: str = "mpm2"
    dtype: str = "float64"
    cell_size: float = 0.025
    refinement: int = 1
    period: float = 0.5
    control_dt: float = 0.02
    substeps: int = 40
    cycles: int = 5
    settle_seconds: float = 0.2
    gravity: float = 9.81
    friction: float = 0.8
    friction_velocity: float = 0.003
    contact_stiffness: float = 1.0e6
    contact_width: float = 0.0005
    actuator_tau: float = 0.03
    max_log_stretch: float = 0.3
    actuation_model: str = "active_strain"
    active_stress_ratio: float = 0.4
    grid_margin: int = 5
    travel_body_lengths: float = 3.0
    seed: int = 0
    gmres_method: str = "incremental"
    spring_solver: str = "dense"

    def __post_init__(self) -> None:
        """Validate a fixed locomotion physical protocol before environment construction."""
        if self.driver not in ENV_IDS:
            raise ValueError(f"Unknown driver: {self.driver}")
        if self.dtype not in ("float32", "float64"):
            raise ValueError("dtype must be float32 or float64")
        for name in (
            "cell_size",
            "period",
            "control_dt",
            "contact_width",
            "friction_velocity",
            "max_log_stretch",
        ):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.actuation_model not in ("active_strain", "active_stress"):
            raise ValueError("Unknown actuation model")
        if self.actuation_model == "active_stress" and self.driver not in ("mpm2", "mpm3"):
            raise ValueError("Active stress is currently supported by MPM drivers only")
        if not math.isfinite(self.active_stress_ratio) or not 0 <= self.active_stress_ratio < 1:
            raise ValueError("active_stress_ratio must be finite and in [0, 1)")
        if (
            not math.isfinite(self.actuator_tau)
            or self.actuator_tau < 0
            or (self.actuator_tau == 0 and self.actuation_model != "active_stress")
        ):
            raise ValueError(
                "Zero actuator_tau requires active_stress; otherwise it must be positive"
            )
        for name in ("refinement", "substeps", "cycles", "grid_margin"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "gravity",
            "friction",
            "contact_stiffness",
            "settle_seconds",
            "travel_body_lengths",
        ):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not math.isclose(
            self.period / self.control_dt, round(self.period / self.control_dt), abs_tol=1e-8
        ):
            raise ValueError("period must be an integer number of control intervals")
        if not math.isclose(
            self.settle_seconds / self.control_dt,
            round(self.settle_seconds / self.control_dt),
            abs_tol=1e-8,
        ):
            raise ValueError("settle_seconds must be an integer number of control intervals")
        if self.spring_solver not in ("dense", "gmres"):
            raise ValueError("Unknown spring linear solver")
        if self.gmres_method not in ("incremental", "batched", "fused"):
            raise ValueError("Unknown GMRES method")

    @property
    def dim(self) -> int:
        """Spatial dimension implied by the selected physics driver."""
        return 2 if self.driver in ("spring", "mpm2") else 3

    @property
    def channels(self) -> int:
        """Number of actuator chambers in the fixed biped or quadruped."""
        return 4 if self.dim == 2 else 16

    @property
    def grid_shape(self) -> tuple[int, ...]:
        """Reference lattice shape after applying uniform refinement."""
        return (6 * self.refinement,) * self.dim

    @property
    def size(self) -> tuple[float, ...]:
        """Physical reference envelope dimensions in metres."""
        return (6 * self.cell_size,) * self.dim

    @property
    def steps_per_cycle(self) -> int:
        """Number of policy intervals in one exact controller period."""
        return round(self.period / self.control_dt)

    @property
    def steps(self) -> int:
        """Total policy-action intervals in the configured number of cycles."""
        return self.cycles * self.steps_per_cycle


class LeggedDecoder:
    """Torso plus 2/4 legs; two/four vertical chambers in each leg.

    Controller channels are stable in reference coordinates. Each 2D leg has
    inner/outer x chambers. Each 3D leg has the Cartesian product of x/z halves.
    A passive torso occupies the upper third of the envelope. A two-cell-wide
    central gap separates legs at the default six-cell resolution.

    Only bounded physical material parameters are currently design variables.
    Geometry is deliberately fixed while testing whether control can work.
    """

    def __init__(self, dim: int = 2, dtype: DTypeLike = "float64") -> None:
        self.dtype = np.dtype(dtype)
        self.spec = FieldSpec(
            dim=dim, channels=4 if dim == 2 else 16, design_mode="hard", material_mode="continuous"
        )

    def space(self) -> Dict:
        """Return bounded chamber material/design coefficient spaces."""
        return Dict(
            {
                "young": Box(5000.0, 100000.0, (2,), self.dtype),
                "density": Box(800.0, 1250.0, (), self.dtype),
                "strength": Box(0.0, 1.0, (), self.dtype),
                "poisson_values": Box(0.25, 0.45, (3,) * self.spec.dim + (1,), self.dtype),
            }
        )

    def base(self) -> HostEmbedding:
        """Return a fresh baseline chamber embedding on the host."""
        return {
            "young": np.asarray([12000.0, 60000.0], self.dtype),
            "density": np.asarray(1000.0, self.dtype),
            "strength": np.asarray(0.95, self.dtype),
            "poisson_values": np.full((3,) * self.spec.dim + (1,), 0.35, self.dtype),
        }

    def upgrade_embedding(self, embodiment: Embedding) -> Embedding:
        """Fill the appended Poisson field in an exact legacy chamber layout."""
        if isinstance(embodiment, dict) and "poisson_values" not in embodiment:
            return dict(embodiment, poisson_values=self.base()["poisson_values"])
        return embodiment

    def structural_errors(self, embodiment: Embedding) -> list[str]:
        """Validate chamber-specific structural constraints."""
        return []

    def descriptor(self) -> dict[str, object]:
        """Return portable chamber geometry and field settings."""
        return {
            "name": "legged_chambers_v1",
            "spec": asdict(self.spec),
            "dtype": self.dtype.name,
            "geometry": "fixed hard reference fields",
        }

    def regions(self, points: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """Return occupancy, torso, leg ID, channel ID in material coordinates."""
        q = (points + 1.0) / 2.0
        torso = q[:, 1] >= 2.0 / 3.0
        end_x = (q[:, 0] < 1.0 / 3.0) | (q[:, 0] > 2.0 / 3.0)
        leg_x = (q[:, 0] > 0.5).astype(jnp.int32)
        local_x = q[:, 0] - leg_x * (2.0 / 3.0)
        chamber_x = (local_x >= 1.0 / 6.0).astype(jnp.int32)
        if self.spec.dim == 2:
            leg = leg_x
            channel = 2 * leg + chamber_x
            occupied = torso | end_x
        else:
            end_z = (q[:, 2] < 1.0 / 3.0) | (q[:, 2] > 2.0 / 3.0)
            leg_z = (q[:, 2] > 0.5).astype(jnp.int32)
            local_z = q[:, 2] - leg_z * (2.0 / 3.0)
            chamber_z = (local_z >= 1.0 / 6.0).astype(jnp.int32)
            leg = 2 * leg_x + leg_z
            channel = 4 * leg + 2 * chamber_x + chamber_z
            occupied = torso | (end_x & end_z)
        return occupied, torso, leg, channel

    def decode(self, embodiment: Embedding, points: jax.Array) -> Fields:
        """Sample geometry and spatial material/actuator fields at normalized points."""
        embodiment = self.upgrade_embedding(embodiment)
        occupied, torso, _, channel = self.regions(points)
        active = occupied & ~torso
        direction = jnp.broadcast_to(jnp.eye(self.spec.dim, dtype=points.dtype)[1], points.shape)
        return Fields(
            occupied.astype(points.dtype),
            jnp.where(torso, embodiment["young"][1], embodiment["young"][0]),
            jnp.broadcast_to(embodiment["density"], occupied.shape),
            jnp.where(active, embodiment["strength"], 0.0),
            direction,
            active.astype(jnp.int32),
            channel,
            jnp.clip(interpolate(embodiment["poisson_values"], points)[:, 0], 0.25, 0.45),
        )


def build_demo(
    config: DemoConfig, *, device: str = "cpu", render: bool = False
) -> tuple[Spring2DEnv, HostEmbedding]:
    """Construct a normal scalar environment through the fixed public factory.

    This function does not reset it or inject state. The returned embodiment
    remains editable through the normal design dictionary and reset API.
    """
    decoder = LeggedDecoder(config.dim, config.dtype)
    settle_steps = round(config.settle_seconds / config.control_dt)
    physics: SpringConfig | BodyConfig
    common = dict(
        grid_shape=config.grid_shape,
        size=config.size,
        control_dt=config.control_dt,
        substeps=config.substeps,
        settle_steps=settle_steps,
        horizon=config.steps,
        initial_clearance=0.001,
        length_ref=config.size[0],
        work_ref=4.0 if config.dim == 3 else 0.4,
        energy_weight=0.0,
    )
    if config.driver == "spring":
        physics = SpringConfig(
            # The selected config validates dimension-specific common options.
            **common,  # type: ignore[arg-type]
            gravity=config.gravity,
            friction=config.friction,
            friction_velocity=config.friction_velocity,
            contact_modulus=config.contact_stiffness * config.size[0],
            contact_width=config.contact_width,
            actuator_tau=config.actuator_tau,
            max_log_stretch=config.max_log_stretch,
            linear_solver=config.spring_solver,
            gmres_method=config.gmres_method,
        )
    else:
        contact = PlaneContact(
            enabled=True,
            stiffness=config.contact_stiffness,
            width=config.contact_width,
            friction=config.friction,
            velocity_width=config.friction_velocity,
        )
        dynamics = DynamicsConfig(
            gravity=config.gravity,
            contact=contact,
            actuator_tau=config.actuator_tau,
            max_log_stretch=config.max_log_stretch,
            max_halvings=0,
            actuation_model=config.actuation_model,
            active_stress_ratio=config.active_stress_ratio,
            mpm_mass_weighted=True,
            newton=NewtonConfig(
                gmres_method=config.gmres_method,
                preconditioner_side="right",
                linear_refinements=2,
                restart=40,
                max_restarts=20,
            ),
        )
        margin = (config.grid_margin + 1) * config.refinement
        forward = max(margin, math.ceil(config.travel_body_lengths * 6 * config.refinement) + 2)
        padding = ((margin, forward),) + ((margin, margin),) * (config.dim - 1)
        physics = BodyConfig(
            # The selected config validates dimension-specific common options.
            **common,  # type: ignore[arg-type]
            dynamics=dynamics,
            grid_margin=config.grid_margin,
            grid_padding=padding,
            observation_profile="full",
        )
    return make(
        ENV_IDS[config.driver],
        config=physics,
        decoder=decoder,
        channels=config.channels,
        max_action_space_size=config.channels,
        max_observation_space_size=250000 if config.dim == 3 else 25000,
        device=device,
        dtype=config.dtype,
        is_eval_env=True,
        render_mode="rgb_array" if render else None,
    )
