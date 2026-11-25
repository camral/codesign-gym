"""Focused follow-up regressions, not the historical all-physics suite."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable


import os
from dataclasses import replace

import numpy as np
import pytest
import warp as wp

PRECISION = os.environ.get("WARP_TEST_DTYPE", "float64")
DEVICE = os.environ.get("WARP_TEST_DEVICE", "cpu")


def execute(function: Callable[[], None], device: wp.context.Device) -> None:
    if device.is_cuda:
        # Conditional graphs already require a runtime supporting lazy loads.
        with wp.ScopedCapture(device=device, force_module_load=False) as captured:
            function()
        wp.capture_launch(captured.graph)
    else:
        function()
    wp.synchronize_device(device)


@pytest.mark.parametrize("physics", ["spring", "fem"])
def test_analytic_rows_match_autodiff_and_transpose(physics: str) -> None:
    from envs.soft_walker.diagnostics.warp.solid_fixtures import fixture
    from envs.soft_walker.engine.warp_native.solid_model import SolidDeviceModel
    from envs.soft_walker.engine.warp_native.solid_step import SolidStep

    bundle, *_ = fixture(physics, PRECISION, True)
    rng = np.random.default_rng(20260923)
    # Nonzero incoming velocities exercise the often-missed spring viscous
    # derivative of the changing unit direction. Only free columns contribute.
    bundle.initial["v"] = rng.normal(0, 0.015, bundle.initial["v"].shape).astype(PRECISION)
    bundle.fields["free"][0, 0] = 0
    if physics == "spring":
        bundle.initial["u"][:, 0] += np.linspace(0, 0.001, len(bundle.fields["mass"])).astype(
            PRECISION
        )
    steps = []
    data = []
    for assembly in ("autodiff", "analytic"):
        candidate = replace(bundle, config=replace(bundle.config, tangent_assembly=assembly))
        step = SolidStep(SolidDeviceModel(candidate, DEVICE))
        step.command.assign(np.asarray([0.31, -0.22], PRECISION))
        step.q.assign(
            rng.normal(0, 0.005, step.q.numpy().shape).astype(PRECISION) if not data else data[0]
        )
        if not data:
            data.append(step.q.numpy().copy())

        def build() -> None:
            step.prepare()
            step.refresh_linearization()

        execute(build, step.device)
        steps.append(step)
    tol = 2e-5 if PRECISION == "float32" else 2e-11
    np.testing.assert_allclose(
        steps[1].rows.grad.numpy(), steps[0].rows.grad.numpy(), rtol=tol, atol=tol * 1e-2
    )
    assert int(steps[1].st.work_counts.numpy()[2]) == 1
    direction = rng.normal(size=steps[0].q.numpy().shape).astype(PRECISION)
    outputs = []
    for step in steps:
        x = wp.array(direction, dtype=step.k.qvec, device=step.device)
        out = wp.zeros_like(x)
        pair = []
        for transpose in (False, True):
            execute(lambda: step.apply_tangent(x, out, out, 1.0, 0.0, transpose), step.device)
            pair.append(out.numpy().copy())
        outputs.append(pair)
        counts = step.st.work_counts.numpy()
        assert counts[0] == counts[1] == 1
    np.testing.assert_allclose(outputs[1], outputs[0], rtol=tol, atol=tol * 1e-2)


@pytest.mark.parametrize("dim", [2, 3])
def test_mass_parameter_adjoint_against_velocity_form(dim: int) -> None:
    from envs.soft_walker.diagnostics.warp.checks import contact_fixture
    from envs.soft_walker.engine.warp_native.episode import Episode
    from envs.soft_walker.engine.warp_native.model import DeviceModel

    bundle = contact_fixture(dim, PRECISION, "active_strain", "slope")
    bundle.config = replace(
        bundle.config, mass_weighted=True, krylov_method="adaptive", cache_tangent=True
    )
    commands = np.asarray([[0.3], [-0.18]], PRECISION)
    results = []
    for formulation in ("velocity", "mass"):
        candidate = replace(bundle, config=replace(bundle.config, parameter_adjoint=formulation))
        episode = Episode(
            DeviceModel(candidate, DEVICE),
            commands,
            settling_steps=1,
            checkpoint_stride=2,
            velocity_weight=0.13,
            work_weight=0.07,
            reuse_replay_roots=True,
        )
        report = episode.run(True)
        assert report["status"] == report["adjoint_status"] == 0, report
        gradients = episode.gradients()
        state = {
            name: getattr(episode.final, name).numpy().copy()
            for name in ("x", "v", "F", "C", "actuator", "ledger")
        }
        results.append(
            (
                report,
                {
                    f"{group}/{name}": value.copy()
                    for group in ("model", "initial")
                    for name, value in gradients[group].items()
                },
                gradients["commands"].copy(),
                state,
            )
        )
    # The selector changes only the parameter VJP. Independent CUDA reductions
    # may differ by an ULP; check the complete forward state as well as the loss
    # at a dtype-scaled roundoff budget before comparing any gradients.
    eps = np.finfo(PRECISION).eps
    forward = [(name, results[0][3][name], results[1][3][name]) for name in results[0][3]]
    forward.append(("loss", np.asarray(results[0][0]["loss"]), np.asarray(results[1][0]["loss"])))
    for name, old, new in forward:
        scale = float(np.max(np.abs(old), initial=0))
        np.testing.assert_allclose(new, old, rtol=8 * eps, atol=8 * eps * scale, err_msg=name)
    for name, old in results[0][1].items():
        new = results[1][1][name]
        # Near-zero analytic cancellations get only a small absolute allowance.
        tolerance = 3e-3 if PRECISION == "float32" else 2e-7
        error = np.linalg.norm(new.astype("float64") - old)
        assert error <= tolerance * np.linalg.norm(old) + (
            1e-7 if PRECISION == "float32" else 1e-12
        ), (name, error)
    np.testing.assert_allclose(
        results[1][2], results[0][2], rtol=2e-4 if PRECISION == "float32" else 1e-8, atol=1e-9
    )


def test_solid_configuration_rejects_inapplicable_adjoint_and_tangent() -> None:
    from envs.soft_walker.engine.warp_native.config import MPMConfig
    from envs.soft_walker.engine.warp_native.solid_model import SolidConfig

    with pytest.raises(ValueError, match="parameter_adjoint"):
        MPMConfig(parameter_adjoint="invalid")
    with pytest.raises(ValueError, match="MPM"):
        SolidConfig(parameter_adjoint="mass")
    with pytest.raises(ValueError, match="beam tangent"):
        SolidConfig(
            physics="beam",
            grid_shape=(4, 4, 4),
            origin=(0.0, 0.0, 0.0),
            tangent_assembly="analytic",
        )
    for name in ("adjoint_rtol64", "adjoint_rtol32"):
        for value in (0.0, -1.0, float("nan"), float("inf")):
            with pytest.raises(ValueError):
                SolidConfig(**{name: value})


def test_spring_solver_configuration_and_positive_diagonal() -> None:
    from envs.soft_walker.diagnostics.warp.solid_fixtures import fixture
    from envs.soft_walker.engine.physics.spring import spring_diagonal
    from envs.soft_walker.engine.physics.spring_config import SpringConfig

    _, model, _, cfg, *_ = fixture("spring", PRECISION, True)
    diagonal = np.asarray(spring_diagonal(model, cfg))
    assert diagonal.shape == (model.reference.size,)
    assert np.all(np.isfinite(diagonal)) and np.all(diagonal >= 1)
    assert diagonal.dtype == np.dtype(PRECISION)
    for value in [-1, 4, True, 1.5]:
        with pytest.raises(ValueError):
            SpringConfig(gmres_refinement_steps=value)
    with pytest.raises(ValueError):
        SpringConfig(gmres_preconditioner="invalid")


def test_float32_spring_with_x64_enabled() -> None:
    import jax
    import jax.numpy as jnp

    from envs.soft_walker.diagnostics.warp.solid_fixtures import fixture
    from envs.soft_walker.engine.physics import spring

    with jax.enable_x64(True):
        _, model, state, cfg, *_ = fixture("spring", "float32", True)
        cfg = replace(cfg, linear_solver="gmres", gmres_method="fused")
        new, diagnostics = jax.jit(lambda s, a: spring.substep(model, s, a, cfg))(
            state, jnp.asarray([0.2, -0.1], jnp.float32)
        )
        assert new.x.dtype == new.v.dtype == jnp.float32
        assert int(new.status) == 0
        assert np.isfinite(np.asarray(new.x)).all()


def test_preload_is_scoped_to_supplied_kernel_namespaces(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    from envs.soft_walker.engine.warp_native.loading import preload_kernels
    from envs.soft_walker.engine.warp_native.solid_kernels import solid_kernels

    k = solid_kernels("spring", PRECISION)
    calls = []
    monkeypatch.setattr(
        wp, "load_module", lambda module, device: calls.append((module.name, device))
    )
    loaded = preload_kernels(
        "cpu", SimpleNamespace(first=k.force_kernel, duplicate=k.node_residual), None
    )
    assert loaded == (k.force_kernel.module.name,)
    assert calls == [(k.force_kernel.module.name, "cpu")]


def test_new_modes_are_serialized_and_old_configs_load() -> None:
    from dataclasses import asdict

    from envs.soft_walker.engine.warp_native.config import MPMConfig
    from envs.soft_walker.engine.warp_native.solid_model import SolidConfig

    original = MPMConfig()
    assert MPMConfig(**original.to_dict()) == original
    assert original.parameter_adjoint == "mass"
    solid = SolidConfig()
    assert SolidConfig(**asdict(solid)) == solid
    assert solid.parameter_adjoint == "velocity"
