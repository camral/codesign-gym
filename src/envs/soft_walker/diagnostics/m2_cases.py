"""Deterministic, serialized-in-reports M2 numerical fixtures; no optimizer."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, cast

import jax
import jax.numpy as jnp

from envs.soft_walker.engine.embodiments.decoders import (
    CPPNDecoder,
    FieldSpec,
    cast_embedding,
    make_decoder,
)
from envs.soft_walker.engine.numerics.solvers import NewtonConfig
from envs.soft_walker.engine.physics import contact, fem, mpm
from envs.soft_walker.engine.physics.common import DynamicsConfig
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

    from envs.soft_walker.engine.physics.common import StepInfo
    from envs.soft_walker.engine.types import (
        Array,
        Embedding,
        PhysicsModel,
        PhysicsState,
    )

# Serialized reports can contain backend-specific auxiliary diagnostic trees.
Report = dict[str, Any]


def design_case(
    name: str,
    dtype: DTypeLike,
    decoder_name: str = "direct",
    contact_enabled: bool = True,
    steps: int = 4,
    debug_linear: bool = False,
) -> tuple[
    Callable[..., Array],
    Callable[..., tuple[PhysicsState, StepInfo, StepInfo]],
    Array,
    Report,
    Callable[..., PhysicsModel],
    Callable[..., tuple[PhysicsState, StepInfo]],
]:
    """Construct a differentiable continuum test case and return its loss, rollout, parameters, and model builders."""
    dim = 2 if name == "mpm" else 3
    decoder = make_decoder(
        decoder_name, spec=FieldSpec(dim=dim, design_mode="relaxed"), dtype=dtype
    )
    base = cast_embedding(decoder, decoder.base())
    if decoder_name == "primitives":
        base["kind"] = base["kind"].at[0].set(0)
    domain: mpm.Cloud | fem.Mesh
    initial: Callable[[PhysicsModel], PhysicsState]
    if name == "mpm":
        domain = mpm.uniform_cloud(cell_shape=(2, 1), origin=(0.025, 0.0003), dtype=dtype)
        grid = mpm.Grid(shape=(8, 7))
        initial = cast("Callable[[PhysicsModel], PhysicsState]", mpm.initial_state)
    else:
        domain = fem.structured_mesh(
            shape=(2, 1, 1), size=(0.1, 0.05, 0.05), origin=(0.0, 0.0003, 0.0), dtype=dtype
        )
        grid = None
        initial = cast("Callable[[PhysicsModel], PhysicsState]", fem.initial_state)
    # A tight audit profile, not the default production tolerance.
    nc = NewtonConfig(
        atol64=3e-13,
        atol32=1e-6,
        linear_rtol64=2e-11,
        linear_rtol32=1e-6,
        max_steps=20,
        debug_linear=debug_linear,
    )
    cfg = DynamicsConfig(
        gravity=9.81 if contact_enabled else 0.0,
        newton=nc,
        contact=contact.PlaneContact(enabled=contact_enabled),
        max_halvings=0,
    )
    dt = 0.001

    def embedding(theta: Array) -> Embedding:
        e = dict(base)
        if decoder_name in ("direct", "bspline"):
            e["values"] = (
                e["values"]
                .at[..., 0]
                .add(theta[0])
                .at[..., 1]
                .add(theta[1])
                .at[..., 2]
                .add(theta[2])
                .at[..., 3]
                .add(theta[3])
            )
        elif decoder_name == "cppn":
            e["bias"] = (
                e["bias"]
                .at[cast(CPPNDecoder, decoder).hidden : cast(CPPNDecoder, decoder).hidden + 4]
                .add(theta[:4])
            )
        else:
            e["half_size"] = e["half_size"].at[0, 0].add(theta[0] * 0.15)
            e["features"] = (
                e["features"].at[:, 0].add(theta[1]).at[:, 1].add(theta[2]).at[:, 2].add(theta[3])
            )
        return e

    def model_from_theta(theta: Array) -> PhysicsModel:
        field = decoder.decode(embedding(theta), domain.points)
        nu = 0.35 + 0.08 * jnp.tanh(theta[4])
        model = cast(
            "Callable[..., PhysicsModel]", mpm.instantiate if name == "mpm" else fem.instantiate
        )(domain, field, poisson=nu, surface_smoothing=0.002)
        return model

    def trial(
        model: PhysicsModel, s: PhysicsState, a: Array, h: float
    ) -> tuple[PhysicsState, StepInfo]:
        if name == "mpm":
            return mpm.trial(
                cast(mpm.Model, model), cast(mpm.State, s), a, h, cast(mpm.Grid, grid), cfg
            )
        return fem.trial(cast(fem.Model, model), cast(fem.State, s), a, h, cfg)

    def rollout(theta: Array) -> tuple[PhysicsState, StepInfo, StepInfo]:
        model = model_from_theta(theta)
        s = initial(model)
        # Fixed one-step settling is differentiated, then work counters reset.
        s, settle_info = trial(model, s, jnp.zeros(2, dtype), dt)
        z = jnp.asarray(0.0, dtype)
        s = s._replace(work_signed=z, work_positive=z, dissipated=z, time=z)

        def body(s: PhysicsState, i: int) -> tuple[PhysicsState, StepInfo]:
            a = jnp.stack((theta[5] * (1 - 0.1 * i), theta[6] * (1 - 0.08 * i)))
            return trial(model, s, a, dt)

        state, infos = jax.lax.scan(jax.checkpoint(body), s, jnp.arange(steps))
        return state, infos, settle_info

    def loss(theta: Array) -> Array:
        state, info, settle = rollout(theta)
        # Smooth diagnostic loss in declared units, not a locomotion reward.
        # Includes design-sensitive work and terminal motion; no absolute-value
        # switch at zero or positive-work cusp in this gradient audit objective.
        loss = 1.0e5 * state.work_signed + 20 * jnp.sum(state.v**2)
        ok = (state.status == 0) & jnp.all(info.accepted > 0) & (settle.accepted > 0)
        return loss * jnp.where(ok, 1.0, jnp.nan)

    theta = jnp.asarray([0.13, 0.07, -0.05, 0.09, 0.11, 0.24, -0.17], dtype)
    meta = {
        "kind": name,
        "decoder": decoder.descriptor(),
        "steps": steps,
        "dt": dt,
        "settle_steps": 1,
        "newton": nc.__dict__,
        "contact": cfg.contact.__dict__,
        "dimension": dim,
        "loss": "1e5*work_signed_J + 20*sum(v_m_per_s^2)",
        "parameters": [
            "geometry",
            "stiffness",
            "density",
            "actuator_strength",
            "poisson",
            "control0",
            "control1",
        ],
        "grid": grid.__dict__ if grid else None,
    }
    return loss, rollout, theta, meta, model_from_theta, trial
