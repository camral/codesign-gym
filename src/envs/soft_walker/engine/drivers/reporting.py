"""Traceable observations, energy budgets, and committed-state diagnostics.

Functions assemble fixed-layout JAX arrays without stepping the simulator or
copying data to the host. Reporting preserves the same material/node ordering
and padding masks as the original driver API.
"""

from __future__ import annotations

from typing import cast

import jax
import jax.numpy as jnp

from ...contracts.fields import ACTUATOR_TYPE_COUNT
from ...contracts.observations import (
    ACTUATOR_OBSERVATION_WIDTH,
    BEAM_MECHANICS_WIDTH,
    BEAM_ROTATION_WIDTH,
    OBSERVATION_DENSITY_SCALE_KG_M3,
    OBSERVATION_YOUNG_SCALE_PA,
    TERRAIN_PROBE_COUNT,
    TERRAIN_PROBE_FORWARD_OFFSETS,
)
from ..physics import beam, mpm, terrain as terrains
from ..types import BodyModel
from .model_access import channel_mask, positions, reference
from .task import BodyConfig, TaskState, mean


def observe(
    kind: str, model: BodyModel, state: TaskState, cfg: BodyConfig, max_size: int
) -> tuple[jax.Array, jax.Array]:
    """Build the fixed observation layout and equally shaped validity mask.

    Node/material indices retain reference ordering. Full MPM observations
    include deformation and affine velocity; compact ones are partially observed.
    """
    s = state.body
    dim = s.v.shape[-1]
    p = positions(model, s, kind)
    ref = reference(model, kind)
    center = mean(model, p)
    vel = mean(model, s.v)
    globals = jnp.concatenate(
        (
            center / cfg.length_ref,
            vel / cfg.velocity_ref,
            state.displacement / cfg.length_ref,
            jnp.reshape(s.time / (cfg.horizon * cfg.control_dt), (1,)),
        )
    )
    values = [globals]
    masks = [jnp.ones_like(globals, dtype=bool)]
    nodes = jnp.concatenate(
        (
            (ref - state.initial_com) / cfg.length_ref,
            (p - ref) / cfg.length_ref,
            s.v / cfg.velocity_ref,
            model.active[:, None].astype(p.dtype),
        ),
        axis=-1,
    )
    nodes = jnp.where(model.active[:, None], nodes, 0.0)
    values.append(nodes.reshape(-1))
    masks.append(jnp.repeat(model.active, nodes.shape[-1]))
    # A fixed flattened block preserves material indices rather than sorting by
    # current geometry or changing channel order during deformation.
    fields = jnp.concatenate(
        (
            jnp.stack(
                (
                    model.alpha,
                    jnp.log(model.young / OBSERVATION_YOUNG_SCALE_PA),
                    model.density / OBSERVATION_DENSITY_SCALE_KG_M3,
                    model.strength,
                    model.poisson,
                ),
                -1,
            ),
            model.direction,
            jax.nn.one_hot(model.kind, ACTUATOR_TYPE_COUNT, dtype=p.dtype),
            jax.nn.one_hot(model.channel, len(model.channels), dtype=p.dtype),
        ),
        axis=-1,
    )
    fa = model.alpha > 0
    fields = jnp.where(fa[:, None], fields, 0.0)
    values.append(fields.reshape(-1))
    masks.append(jnp.repeat(fa, fields.shape[-1]))
    if kind == "beam":
        s = cast(beam.State, s)
        model = cast(beam.Model, model)
        extra = jnp.concatenate(
            (
                (s.rotation - jnp.eye(3, dtype=p.dtype)).reshape(len(p), BEAM_ROTATION_WIDTH),
                s.omega * cfg.length_ref / cfg.velocity_ref,
            ),
            -1,
        )
        values.append(jnp.where(model.active[:, None], extra, 0.0).reshape(-1))
        masks.append(jnp.repeat(model.active, BEAM_MECHANICS_WIDTH))
    if kind.startswith("mpm") and cfg.observation_profile == "full":
        s = cast(mpm.State, s)
        extra = jnp.concatenate(
            (
                (s.F - jnp.eye(dim, dtype=p.dtype)).reshape(len(p), -1),
                s.C.reshape(len(p), -1) * cfg.length_ref / cfg.velocity_ref,
            ),
            -1,
        )
        values.append(jnp.where(model.active[:, None], extra, 0.0).reshape(-1))
        masks.append(jnp.repeat(model.active, 2 * dim * dim))
    cm = channel_mask(model)
    act = jnp.stack((jnp.where(cm, s.actuator, 0.0), cm.astype(p.dtype)), -1)
    values.append(act.reshape(-1))
    masks.append(jnp.repeat(cm, ACTUATOR_OBSERVATION_WIDTH))
    offsets = (
        jnp.zeros((TERRAIN_PROBE_COUNT, dim), p.dtype)
        .at[:, 0]
        .set(jnp.asarray(TERRAIN_PROBE_FORWARD_OFFSETS, p.dtype) * cfg.length_ref)
    )
    gap, normal = terrains.gap_normal(
        center[None, :] + offsets, cast(terrains.Terrain, model.terrain)
    )
    probes = jnp.concatenate((gap[:, None] / cfg.length_ref, normal), axis=-1).reshape(-1)
    values.append(probes)
    masks.append(jnp.ones_like(probes, dtype=bool))
    v = jnp.concatenate(values)
    mask = jnp.concatenate(masks)
    return jnp.pad(v, (0, max_size - v.size)), jnp.pad(mask, (0, max_size - mask.size))


def mechanics_diagnostics(
    kind: str, model: BodyModel, state: TaskState, cfg: BodyConfig
) -> dict[str, jax.Array]:
    """Committed-state diagnostics; rejected trial residuals remain in StepInfo."""
    s = state.body
    out = {
        "max_speed_m_s": jnp.max(jnp.linalg.norm(s.v, axis=-1)),
        "finite_state": jnp.all(jnp.isfinite(s.v)),
    }
    if kind == "beam":
        s = cast(beam.State, s)
        model = cast(beam.Model, model)
        R = s.rotation
        eye = jnp.eye(3, dtype=R.dtype)
        i, j = model.topology.edges.T
        out.update(
            rotation_orthogonality_max=jnp.max(jnp.abs(jnp.swapaxes(R, -1, -2) @ R - eye)),
            rotation_determinant_error_max=jnp.max(jnp.abs(jnp.linalg.det(R) - 1)),
            max_omega_rad_s=jnp.max(jnp.linalg.norm(s.omega, axis=-1)),
        )
        cosine = (jnp.trace(jnp.swapaxes(R[i], -1, -2) @ R[j], axis1=-2, axis2=-1) - 1) / 2
        links = model.active[i] & model.active[j]
        out["relative_link_angle_max_rad"] = jnp.max(
            jnp.where(links, jnp.arccos(jnp.clip(cosine, -1.0, 1.0)), 0.0), initial=0.0
        )
    elif kind.startswith("mpm"):
        s = cast(mpm.State, s)
        out.update(
            min_particle_J=jnp.min(jnp.where(model.active, jnp.linalg.det(s.F), jnp.inf)),
            max_affine_rate_s_inv=jnp.max(jnp.linalg.norm(s.C, axis=(-2, -1))),
        )
    return out


def _mpm_energy(model: mpm.Model, state: mpm.State, cfg: BodyConfig) -> jax.Array:
    """Return kinetic, elastic, gravity, and contact energies in joules."""
    from ..physics import contact

    r = jnp.einsum("pij,fj->pfi", state.F, model.cloud.face_offset)
    points = state.x[:, None, :] + r
    vel = state.v[:, None, :] + jnp.einsum("pij,pfj->pfi", state.C, r)
    _, uc, _ = contact.response(
        points, vel, model.contact_area, cfg.dynamics.contact, model.terrain
    )
    kinetic = 0.5 * jnp.sum(model.mass[:, None] * state.v**2)
    return jnp.stack(
        (
            kinetic,
            mpm.elastic_energy(model, state.F, state.actuator, cfg.dynamics),
            jnp.sum(model.mass * cfg.dynamics.gravity * state.x[:, 1]),
            jnp.sum(uc),
        )
    )
