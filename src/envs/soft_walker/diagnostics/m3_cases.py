"""New driver fixtures, independent of policy/optimizer. No claimed gait quality."""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING, Any, Callable, cast

import jax
import jax.numpy as jnp

from envs.soft_walker.engine import make
from envs.soft_walker.engine.differentiation import bind_rollout
from envs.soft_walker.engine.drivers.task import BodyConfig, TaskState
from envs.soft_walker.engine.numerics.solvers import NewtonConfig
from envs.soft_walker.engine.physics import beam, spring
from envs.soft_walker.engine.physics.common import DynamicsConfig
from envs.soft_walker.engine.physics.contact import PlaneContact
from envs.soft_walker.engine.physics.terrain import TerrainConfig
from envs.soft_walker.engine.types import RolloutResult
from envs.soft_walker.shared import (
    ENGINE_ROOT as ENGINE_ROOT,
    REPO_ROOT as REPO_ROOT,
    SCRIPTS_ROOT as SCRIPTS_ROOT,
    SOFT_ROOT as SOFT_ROOT,
    SRC_ROOT as SRC_ROOT,
    TESTS_ROOT as TESTS_ROOT,
)

if TYPE_CHECKING:
    from numpy.typing import DTypeLike

    from envs.soft_walker.engine.drivers.spring_environment import Spring2DEnv
    from envs.soft_walker.engine.types import (
        Array,
        Embedding,
        PhysicsState,
    )

IDS = {
    "fem": "SoftHexFEM3D-v0",
    "mpm2": "SoftMPM2D-v0",
    "mpm3": "SoftMPM3D-v0",
    "beam": "SoftBeam3D-v0",
}
PARAMETERS = [
    "geometry",
    "stiffness",
    "density",
    "actuator_strength",
    "poisson",
    "control0",
    "control1",
]

# Serialized reports can contain backend-specific auxiliary diagnostic trees.
Report = dict[str, Any]


def diagnostic_value(state: PhysicsState, kind: str) -> Array:
    """Explicit test-only feature scaling; never used by rewards/objectives.

    Beam rotation is a necessary observable for bending/shear parameters.
    Its 0.0316 rad/s reference keeps that contribution resolved in float32.
    Near-pure axial work is physically almost independent of Poisson ratio.
    """
    value = 1.0e5 * state.work_signed + 20 * jnp.sum(state.v**2)
    if kind == "beam":
        value = value + 1.0e3 * jnp.sum(cast(beam.State, state).omega ** 2)
    return value


def driver_case(
    kind: str,
    dtype: DTypeLike,
    *,
    steps: int = 3,
    terrain_kind: str = "slopes",
    gmres_method: str = "incremental",
) -> tuple[
    Callable[..., Array],
    Callable[..., RolloutResult[None]],
    Array,
    Report,
    Spring2DEnv,
    Callable[..., Embedding],
]:
    """Construct a body-driver audit with explicit embedding, rollout, and scalar diagnostic loss."""
    dim = 2 if kind == "mpm2" else 3
    shape = (2, 1) if dim == 2 else ((2, 2, 1) if kind == "beam" else (2, 1, 1))
    size = tuple(0.05 * n for n in shape)
    nc = NewtonConfig(
        atol64=3e-13,
        atol32=1e-6,
        linear_rtol64=2e-11,
        linear_rtol32=1e-6,
        max_steps=20,
        gmres_method=gmres_method,
    )
    cfg = BodyConfig(
        grid_shape=shape,
        size=size,
        control_dt=0.001,
        substeps=1,
        settle_steps=1,
        horizon=steps,
        initial_clearance=0.0003,
        grid_margin=1,
        terrain=TerrainConfig(kind=terrain_kind, blocks=1, max_slope=0.03),
        dynamics=DynamicsConfig(newton=nc, contact=PlaneContact(enabled=True), max_halvings=0),
    )
    env, base = make(
        IDS[kind], dtype=dtype, config=cfg, design_mode="relaxed", max_observation_space_size=4096
    )
    # Choose reproducible terrain without doing a separate, unused reset solve.
    from envs.soft_walker.engine.physics import terrain

    env._terrain = terrain.generate(cfg.terrain, 17, dim, dtype)
    env._terrain_seed = 17
    base = jax.tree.map(lambda x: jnp.asarray(x, dtype), base)
    if kind == "beam":
        # Bilayer routing excites bending/shear. A nearly axial two-channel
        # fixture made Poisson sensitivity ~1e-10: physically unobservable,
        # not a useful finite-difference audit. No tolerance is loosened.
        from envs.soft_walker.engine.embodiments.decoders import grid_points

        y = jnp.asarray(grid_points(base["values"].shape[:-1])[..., 1], dtype)
        v = base["values"].at[..., 4].set(0.8).at[..., 5].set(0.6)
        v = v.at[..., 13].set(-2 * y).at[..., 14].set(2 * y)
        base = dict(base, values=v)
    rollout = bind_rollout(env)

    def embedding(theta: Array) -> Report:
        values = (
            base["values"]
            .at[..., 0]
            .add(theta[0])
            .at[..., 1]
            .add(theta[1])
            .at[..., 2]
            .add(theta[2])
            .at[..., 3]
            .add(theta[3])
        )
        return dict(values=values, poisson=jnp.asarray(0.35, dtype) + 0.08 * jnp.tanh(theta[4]))

    def run(theta: Array) -> RolloutResult[None]:
        i = jnp.arange(steps, dtype=dtype)
        actions = jnp.stack((theta[5] * (1 - 0.1 * i), theta[6] * (1 - 0.08 * i)), -1)
        return rollout(embedding(theta), actions)

    def loss(theta: Array) -> Array:
        r = run(theta)
        s = cast(TaskState, r["state"]).body
        value = diagnostic_value(s, kind)
        return jnp.where(r["metrics"]["valid"], value, jnp.nan)

    theta = jnp.asarray([0.13, 0.07, -0.05, 0.09, 0.11, 0.24, -0.17], dtype)
    meta = {
        "kind": kind,
        "env_id": IDS[kind],
        "config": asdict(cfg),
        "parameters": PARAMETERS,
        "decoder": env.decoder.descriptor(),
        "terrain_seed": 17,
        "terrain": terrain.serialize(env._terrain),
        "loss": "1e5*work_signed_J+20*sum(v_m_s^2)"
        + ("+1e3*sum(omega_rad_s^2)" if kind == "beam" else ""),
        "scope": "smooth fixed-topology diagnostic loss, not the locomotion objective",
        "derivatives": "decode -> instantiate -> settle -> regularized terrain contact -> control trajectory",
    }
    return loss, run, theta, meta, env, embedding


def spring_case(
    dtype: DTypeLike, steps: int = 3, *, gmres_method: str = "incremental"
) -> tuple[
    Callable[..., Array],
    Callable[..., RolloutResult[None]],
    Array,
    Report,
    Spring2DEnv,
    Callable[..., Embedding],
]:
    """Construct the spring audit with explicit controller coefficients and decoder embedding."""
    from envs.soft_walker.engine.physics import terrain
    from envs.soft_walker.engine.physics.spring_config import SpringConfig

    cfg = SpringConfig(
        grid_shape=(2, 1),
        size=(0.1, 0.05),
        control_dt=0.001,
        substeps=1,
        settle_steps=1,
        horizon=steps,
        linear_solver="gmres",
        gmres_method=gmres_method,
        initial_clearance=0.0003,
        root_tol64=3e-13,
        root_tol32=2e-7,
        terrain=TerrainConfig(kind="slopes", blocks=1, max_slope=0.03),
    )
    env, base = make(dtype=dtype, config=cfg, design_mode="relaxed")
    env._terrain = terrain.generate(cast(TerrainConfig, cfg.terrain), 17, 2, dtype)
    env._terrain_seed = 17
    base = jax.tree.map(lambda x: jnp.asarray(x, dtype), base)
    rollout = bind_rollout(env)

    def embedding(theta: Array) -> Report:
        v = (
            base["values"]
            .at[..., 0]
            .add(theta[0])
            .at[..., 1]
            .add(theta[1])
            .at[..., 2]
            .add(theta[2])
            .at[..., 3]
            .add(theta[3])
        )
        return {"values": v}

    def run(theta: Array) -> RolloutResult[None]:
        i = jnp.arange(steps, dtype=dtype)
        return rollout(
            embedding(theta), jnp.stack((theta[4] * (1 - 0.1 * i), theta[5] * (1 - 0.08 * i)), -1)
        )

    def loss(theta: Array) -> Array:
        r = run(theta)
        s = r["state"]
        return jnp.where(
            r["metrics"]["valid"], diagnostic_value(cast(spring.State, s), "spring"), jnp.nan
        )

    theta = jnp.asarray([0.13, 0.07, -0.05, 0.09, 0.24, -0.17], dtype)
    meta = {
        "kind": "spring",
        "env_id": "SoftSpring2D-v0",
        "config": asdict(cfg),
        "parameters": [
            "geometry",
            "stiffness",
            "density",
            "actuator_strength",
            "control0",
            "control1",
        ],
        "decoder": env.decoder.descriptor(),
        "terrain_seed": 17,
        "terrain": terrain.serialize(env._terrain),
        "loss": "1e5*work_signed_J+20*sum(v_m_s^2)",
        "scope": "changed relaxed contact area and terrain path",
    }
    return loss, run, theta, meta, env, embedding
