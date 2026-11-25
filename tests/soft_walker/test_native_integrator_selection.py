"""Native integrator options are host-validated and preserve per-lane ownership."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from envs.soft_walker.engine.warp_native import env as native_env
from envs.soft_walker.engine.warp_native.solid_builders import regular_solid
from envs.soft_walker.engine.warp_native.vector_env import NativeVectorEnv


@pytest.mark.parametrize("selection", (None, "backward_euler", "kvaerno3_native"))
@pytest.mark.parametrize("precision", ("float32", "float64"))
def test_native_selection_copies_only_config_and_preserves_reset_template(
    selection: str | None, precision: str
) -> None:
    bundle = regular_solid("spring", shape=(1, 1), precision=precision)
    bundle.config = replace(bundle.config, integrator="kvaerno3_native")
    original_config = bundle.config
    original_fields = {name: value.copy() for name, value in bundle.fields.items()}
    expected = "kvaerno3_native" if selection is None else selection
    environment = native_env.NativeEnv(
        "SoftSpring2D-v0", bundle=bundle, integrator=selection, dtype=precision
    )
    assert environment.integrator == expected
    assert environment.template.config == replace(original_config, integrator=expected)
    assert bundle.config is original_config
    assert bundle.config.integrator == "kvaerno3_native"
    for name, values in bundle.fields.items():
        np.testing.assert_array_equal(values, original_fields[name])
        assert not np.shares_memory(environment.template.fields[name], values)
    candidate = environment._candidate(bundle.fields)
    assert candidate.config == environment.template.config
    assert candidate.config.integrator == expected
    assert environment.owner is None
    assert environment._episode is None


@pytest.mark.parametrize(
    "env_id",
    ("SoftSpring2D-v0", "SoftBeam3D-v0", "SoftHexFEM3D-v0", "SoftMPM2D-v0", "SoftMPM3D-v0"),
)
def test_native_backward_euler_remains_default_for_every_family(env_id: str) -> None:
    default = native_env.NativeEnv(env_id)
    selected = native_env.NativeEnv(env_id, integrator="backward_euler")
    assert default.integrator == selected.integrator == "backward_euler"
    assert default.template.config == selected.template.config
    for name, values in default.template.fields.items():
        np.testing.assert_array_equal(values, selected.template.fields[name])
    assert default.owner is selected.owner is None


@pytest.mark.parametrize(
    ("env_id", "selection"),
    (
        ("SoftSpring2D-v0", "kvaerno3"),
        ("SoftSpring2D-v0", "unknown"),
        ("SoftBeam3D-v0", "kvaerno3_native"),
        ("SoftHexFEM3D-v0", "kvaerno3_native"),
        ("SoftMPM2D-v0", "kvaerno3_native"),
        ("SoftMPM3D-v0", "kvaerno3_native"),
    ),
)
def test_unsupported_native_integrators_fail_before_builders_or_device_lookup(
    env_id: str, selection: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("Unsupported selection reached resource construction")

    monkeypatch.setattr(native_env, "regular_solid", forbidden)
    monkeypatch.setattr(native_env, "uniform_body", forbidden)
    monkeypatch.setattr(native_env.wp, "get_device", forbidden)
    with pytest.raises(ValueError):
        native_env.NativeEnv(env_id, integrator=selection, device="cuda")
    with pytest.raises(ValueError):
        NativeVectorEnv(env_id, 2, integrator=selection, device="cuda", execution="parallel")


def test_native_vector_lanes_inherit_fixed_selection_without_sharing_templates() -> None:
    bundle = regular_solid("spring", shape=(1, 1))
    original_config = bundle.config
    vector = NativeVectorEnv(
        "SoftSpring2D-v0", 2, bundle=bundle, integrator="kvaerno3_native", device="cpu"
    )
    assert vector.integrator == "kvaerno3_native"
    assert bundle.config is original_config
    assert bundle.config.integrator == "backward_euler"
    for lane in vector.envs:
        assert lane.integrator == lane.template.config.integrator == "kvaerno3_native"
        assert lane._candidate(bundle.fields).config.integrator == "kvaerno3_native"
        assert lane.owner is None
    for name, values in vector.envs[0].template.fields.items():
        assert not np.shares_memory(values, vector.envs[1].template.fields[name])


def test_native_factory_preserves_existing_bundle_integrator_when_option_is_omitted() -> None:
    bundle = regular_solid("spring", shape=(1, 1))
    bundle.config = replace(bundle.config, integrator="kvaerno3_native")
    environment, base = native_env.make(bundle=bundle)
    assert environment.integrator == environment.template.config.integrator == "kvaerno3_native"
    for name, values in base.items():
        assert not np.shares_memory(values, bundle.fields[name])
        assert not np.shares_memory(values, environment.template.fields[name])
