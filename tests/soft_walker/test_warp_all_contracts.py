"""Resolved-port validation, API lifecycle, and solid cache lifetime checks.

Device owners/episodes are replaced by test doubles for API checks. Their small
CPU Warp state buffers retain real snapshot copying and restoration. Schema tests
exercise validation before wp.init. The final cache tests use native execution
on WARP_TEST_DEVICE with WARP_TEST_DTYPE, capturing complete CUDA programs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from envs.soft_walker.engine.warp_native.model import Bundle


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from envs.soft_walker.engine.warp_native.solid_model import SolidBundle


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType


import os
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from soft_paths import SOFT_ROOT

wp = pytest.importorskip("warp")

from envs.soft_walker.engine.warp_native import vector_env as native_vector
from envs.soft_walker.engine.warp_native.solid_builders import regular_solid
from envs.soft_walker.engine.warp_native.solid_model import SolidDeviceModel

PRECISION = os.environ.get("WARP_TEST_DTYPE", "float64")


@pytest.mark.parametrize("kind", ["spring", "beam", "fem"])
def test_resolved_schema_checks_every_array_shape_before_device_init(
    kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = regular_solid(kind, precision=PRECISION)
    assert bundle.validate() is bundle
    attempted_init = []
    monkeypatch.setattr(wp, "init", lambda: attempted_init.append(True))
    for key, value in bundle.fields.items():
        malformed = deepcopy(bundle)
        # Every field, including unused placeholders, has a declared rank.
        malformed.fields[key] = value[..., None]
        with pytest.raises(ValueError, match="shape|Invalid|require|expected|reference|cells"):
            SolidDeviceModel(malformed, "cpu")
    assert not attempted_init


@pytest.mark.parametrize(
    "kind,key,value",
    [
        ("spring", "stiffness", -1.0),
        ("spring", "damping", -1.0),
        ("spring", "length", 0.0),
        ("spring", "cell_alpha", -0.1),
        ("spring", "cell_alpha", 1.1),
        ("spring", "mass", -1.0),
        ("beam", "inertia", 0.0),
        ("beam", "axis", 0.0),
        ("fem", "young", 0.0),
        ("fem", "poisson", 0.5),
        ("fem", "poisson", -1.0),
        ("fem", "alpha", -0.1),
        ("fem", "alpha", 1.1),
        ("fem", "strength", -0.1),
        ("fem", "strength", 1.1),
        ("fem", "kind", 3),
        ("fem", "channel", 2),
        ("fem", "weights", 0.0),
        ("fem", "contact_area", -1.0),
        ("fem", "free", 0.5),
    ],
)
def test_invalid_resolved_physical_values_rejected(kind: str, key: str, value: float | int) -> None:
    bundle = regular_solid(kind, precision=PRECISION)
    bundle.fields[key].fill(value)
    with pytest.raises(ValueError):
        bundle.validate()


@pytest.mark.parametrize("key", ["extension", "bending"])
def test_beam_elastic_matrix_domain(key: str) -> None:
    bundle = regular_solid("beam", precision=PRECISION)
    original = bundle.fields[key].copy()
    bundle.fields[key][0, 0, 1] += 1.0
    with pytest.raises(ValueError, match="symmetric"):
        bundle.validate()
    bundle.fields[key] = original
    bundle.fields[key][0] = -np.eye(3, dtype=PRECISION)
    with pytest.raises(ValueError, match="semidefinite"):
        bundle.validate()
    bundle.fields[key].fill(0.0)
    bundle.validate()  # Empty/zero-stiffness links are legitimate resolved input.


@pytest.mark.parametrize("kind", ["spring", "beam", "fem"])
def test_resolved_schema_handles_inactive_nodes_and_rejects_bad_connectivity(kind: str) -> None:
    bundle = regular_solid(kind, precision=PRECISION, properties={"occupancy": [1.0, 0.0]})
    bundle.validate()
    bundle.fields["entities"][0, 0] = len(bundle.fields["reference"])
    with pytest.raises(ValueError, match="connectivity"):
        bundle.validate()


def test_single_voxel_beam_without_links_is_valid() -> None:
    bundle = regular_solid("beam", shape=(1, 1, 1), precision=PRECISION)
    assert len(bundle.fields["entities"]) == 0
    bundle.validate()


def terrain(dim: int) -> dict[str, np.ndarray]:
    normal = np.zeros(dim, dtype=PRECISION)
    normal[1] = 1.0
    return dict(
        normal=normal,
        offset=np.asarray(0.0, dtype=PRECISION),
        centers=np.zeros((1, dim), dtype=PRECISION),
        half_sizes=np.full((1, dim), 0.01, dtype=PRECISION),
        rounding=np.zeros(1, dtype=PRECISION),
        active=np.ones(1, dtype=np.int32),
    )


@pytest.mark.parametrize("key", ["normal", "offset", "centers", "half_sizes", "rounding", "active"])
def test_terrain_schema_rejected_before_native_upload(key: str) -> None:
    bundle = regular_solid("beam", precision=PRECISION, terrain=terrain(3))
    bundle.validate()
    bundle.terrain[key] = bundle.terrain[key][..., None]
    with pytest.raises(ValueError):
        bundle.validate()


@pytest.mark.parametrize(
    "key,value",
    [("normal", 0.0), ("half_sizes", 0.0), ("rounding", -0.1), ("active", 2), ("offset", np.nan)],
)
def test_invalid_terrain_values_rejected(key: str, value: float | int) -> None:
    bundle = regular_solid("spring", precision=PRECISION, terrain=terrain(2))
    bundle.terrain[key].fill(value)
    with pytest.raises(ValueError):
        bundle.validate()


def initial(bundle: SolidBundle) -> dict[str, np.ndarray]:
    """Create correctly shaped solid state, including orientation and work ledgers."""
    n, d = bundle.fields["reference"].shape
    return dict(
        u=np.zeros((n, d), dtype=PRECISION),
        v=np.zeros((n, d), dtype=PRECISION),
        rotation=np.broadcast_to(np.eye(d, dtype=PRECISION), (n, d, d)).copy(),
        omega=np.zeros((n, d), dtype=PRECISION),
        actuator=np.zeros(bundle.channels, dtype=PRECISION),
        ledger=np.zeros(4, dtype=PRECISION),
        status=np.zeros(1, dtype=np.int32),
    )


@pytest.mark.parametrize("key", ["u", "v", "rotation", "omega", "actuator", "ledger", "status"])
def test_initial_state_validated_by_bundle_before_device_allocation(
    key: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = regular_solid("beam", precision=PRECISION)
    bundle.initial = initial(bundle)
    bundle.validate()
    bundle.initial[key] = bundle.initial[key][..., None]
    attempted_init = []
    monkeypatch.setattr(wp, "init", lambda: attempted_init.append(True))
    with pytest.raises(ValueError, match="initial"):
        SolidDeviceModel(bundle, "cpu")
    assert not attempted_init


class HostArray:
    def __init__(self, value: ArrayLike) -> None:
        self.value = np.array(value, copy=True)

    def numpy(self) -> np.ndarray:
        return self.value.copy()


class FakeOwner:
    """Replace physics assembly while retaining the complete native state contract."""

    def __init__(self, bundle: Bundle | SolidBundle, device: str) -> None:
        self.bundle = bundle
        n, dim = bundle.fields["reference"].shape
        real = wp.float32 if PRECISION == "float32" else wp.float64
        vector = wp.types.vector(length=dim, dtype=real)
        matrix = wp.types.matrix(shape=(dim, dim), dtype=real)
        values = dict(
            v=np.zeros((n, dim), dtype=PRECISION),
            actuator=np.zeros(bundle.channels, dtype=PRECISION),
            ledger=np.zeros(4, dtype=PRECISION),
            status=np.zeros(1, dtype=np.int32),
        )
        identity = np.broadcast_to(np.eye(dim, dtype=PRECISION), (n, dim, dim)).copy()
        if "entities" in bundle.fields:
            values.update(
                u=np.zeros((n, dim), dtype=PRECISION),
                rotation=identity,
                omega=np.zeros((n, dim), dtype=PRECISION),
            )
        else:
            values.update(
                x=bundle.fields["reference"].copy(),
                F=identity,
                C=np.zeros((n, dim, dim), dtype=PRECISION),
            )
        if bundle.initial is not None:
            values.update(bundle.initial)
        self.initial = SimpleNamespace()
        for name, value in values.items():
            dtype = (
                wp.int32
                if name == "status"
                else matrix
                if name in ("F", "C", "rotation")
                else vector
                if name in ("x", "u", "v", "omega")
                else real
            )
            setattr(self.initial, name, wp.array(value, dtype=dtype, device="cpu"))
        self.model = SimpleNamespace(
            **{key: HostArray(value) for key, value in bundle.fields.items()}
        )


class FakeEpisode:
    def __init__(self, owner: FakeOwner) -> None:
        self.final = owner.initial
        self.travel = HostArray(np.zeros_like(owner.bundle.fields["reference"]))
        self.step = SimpleNamespace(diagnostics=lambda: {})
        self.code = 0

    def set_commands(self, commands: np.ndarray) -> None:
        self.commands = commands.copy()

    def run(self) -> dict[str, int | float]:
        return dict(status=self.code, execution_seconds=0.0)


@pytest.fixture
def native_api(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Replace assembly/stepping with fakes, retaining API validation and real snapshots."""
    from envs.soft_walker.engine.warp_native import env as native

    monkeypatch.setattr(native, "SolidDeviceModel", FakeOwner)
    monkeypatch.setattr(native, "DeviceModel", FakeOwner)
    monkeypatch.setattr(
        native.NativeEnv, "episode", lambda self, commands, **kw: FakeEpisode(self.owner)
    )
    monkeypatch.setattr(native.NativeEnv, "_copy_final", lambda self, episode: None)
    monkeypatch.setattr(
        native.NativeEnv,
        "_observation",
        lambda self, state: np.zeros(self.observation_size, self.dtype),
    )
    return native


@pytest.mark.parametrize(
    "env_id",
    ["SoftSpring2D-v0", "SoftBeam3D-v0", "SoftHexFEM3D-v0", "SoftMPM2D-v0", "SoftMPM3D-v0"],
)
def test_scalar_reset_shape_and_persistent_embodiment(native_api: ModuleType, env_id: str) -> None:
    env, base = native_api.make(env_id, dtype=PRECISION)
    modified = deepcopy(base)
    modified["reference"][:, 0] += 0.01
    _, rejected, _ = env.reset(embodiment=modified)
    assert rejected.shape == (1,) and rejected.dtype == np.bool_ and not rejected[0]
    modified["reference"][:, 0] += 1.0  # Accepted design must be copied.
    env.reset()
    np.testing.assert_allclose(
        env.owner.bundle.fields["reference"][:, 0], base["reference"][:, 0] + 0.01
    )
    invalid = deepcopy(base)
    invalid["mass" if "mass" in invalid else "young"].fill(-1.0)
    _, rejected, _ = env.reset(embodiment=invalid)
    assert rejected.shape == (1,) and rejected[0]
    with pytest.raises(native_api.ResetNeeded):
        env.step(np.zeros(env.action_size))
    # Explicit reset may recover the last accepted design; rejected reset itself
    # must leave the environment invalid until that explicit reset occurs.
    env.reset()
    np.testing.assert_allclose(
        env.owner.bundle.fields["reference"][:, 0], base["reference"][:, 0] + 0.01
    )


@pytest.mark.parametrize("failure", [False, True])
def test_scalar_step_types_and_failure_precedence(native_api: ModuleType, failure: bool) -> None:
    env, _ = native_api.make("SoftSpring2D-v0", dtype=PRECISION, horizon=1)
    env.reset()
    env._episode.code = 1 if failure else 0
    _, reward, terminated, truncated, _ = env.step(np.zeros(env.action_size))
    assert type(reward) is float and type(terminated) is bool and type(truncated) is bool
    assert terminated is failure and truncated is (not failure)


def test_vector_reset_shape_and_persistent_embodiment(native_api: ModuleType) -> None:
    env, base = native_vector.make_vec("SoftSpring2D-v0", dtype=PRECISION, num_envs=2, horizon=1)
    design = deepcopy(base)
    design["reference"][0, :, 0] += 0.01
    design["reference"][1, :, 0] += 0.02
    _, rejected, _ = env.reset(embodiment=design)
    assert rejected.shape == (2,) and rejected.dtype == np.bool_
    env.reset()
    for i, lane in enumerate(env.envs):
        np.testing.assert_array_equal(lane.owner.bundle.fields["reference"], design["reference"][i])
    actions = np.zeros(env.action_space.shape)
    assert env.step(actions)[3].all()
    obs, reward, terminated, truncated, info = env.step(actions)
    assert reward.shape == terminated.shape == truncated.shape == (2,)
    assert not terminated.any() and not truncated.any()
    for i, lane in enumerate(env.envs):
        np.testing.assert_array_equal(lane.owner.bundle.fields["reference"], design["reference"][i])


@pytest.mark.parametrize(
    "options", [{"unsupported": True}, {"reset_mask": np.array([True, False])}, []]
)
def test_scalar_unsupported_reset_options_preserve_initialized_state(
    native_api: ModuleType, options: dict[str, object] | list[object]
) -> None:
    env, _ = native_api.make("SoftSpring2D-v0", dtype=PRECISION, horizon=2)
    env.reset()
    env.step(np.zeros(env.action_size))
    owner, episode, steps = env.owner, env._episode, env._steps
    with pytest.raises(ValueError, match="reset options"):
        env.reset(options=options)
    assert env.initialized and env.owner is owner and env._episode is episode
    assert env._steps == steps
    _, rejected, _ = env.reset(options={})
    assert not rejected[0] and env.initialized and env._steps == 0


@pytest.mark.parametrize("options", [{"unsupported": True}, {"reset_mask": np.array([1, 0])}, []])
def test_vector_unsupported_reset_options_preserve_all_lanes(
    native_api: ModuleType, options: dict[str, object] | list[object]
) -> None:
    env, _ = native_vector.make_vec("SoftSpring2D-v0", dtype=PRECISION, num_envs=2, horizon=1)
    env.reset()
    env.step(np.zeros(env.action_space.shape))
    owners = [lane.owner for lane in env.envs]
    episodes = [lane._episode for lane in env.envs]
    pending = env._next_reset.copy()
    with pytest.raises(ValueError, match="reset options|reset_mask"):
        env.reset(options=options)
    np.testing.assert_array_equal(env._next_reset, pending)
    for lane, owner, episode in zip(env.envs, owners, episodes):
        assert lane.initialized and lane.owner is owner and lane._episode is episode
        assert lane._steps == 1
    _, rejected, _ = env.reset(options={})
    assert rejected.shape == (2,) and not rejected.any() and not env._next_reset.any()
    assert all(lane.initialized and lane._steps == 0 for lane in env.envs)


def test_documented_gymnasium_adapter_accepts_native_reset(native_api: ModuleType) -> None:
    pytest.importorskip("gymnasium")
    guide = SOFT_ROOT / "docs" / "COMPONENTS_AND_GYMNASIUM.md"
    blocks = guide.read_text(encoding="utf-8").split("```python\n")[1:]
    snippet = next(
        block.split("```", 1)[0] for block in blocks if "class GymnasiumAdapter(" in block
    )
    namespace = {}
    exec(compile(snippet, str(guide), "exec"), namespace)
    adapter = namespace["GymnasiumAdapter"](backend="warp", dtype=PRECISION)
    try:
        observation, info = adapter.reset(seed=7)
        assert adapter.observation_space.contains(observation)
        result = adapter.step(np.zeros(adapter.action_space.shape, dtype=PRECISION))
        assert len(result) == 5
    finally:
        adapter.close()


# Numerical cache lifetime checks run on the explicitly selected test device.
from envs.soft_walker.engine.warp_native.solid_step import SolidStep

DEVICE = os.environ.get("WARP_TEST_DEVICE", "cpu")


@pytest.mark.parametrize("kind", ["spring", "beam", "fem"])
@pytest.mark.parametrize("invalidate_with_prepare", [False, True])
def test_solid_direct_products_discard_stale_local_blocks(
    kind: str, invalidate_with_prepare: bool
) -> None:
    owner = SolidDeviceModel(regular_solid(kind, precision=PRECISION), DEVICE)
    step = SolidStep(owner)
    step.command.fill_(0.3)
    rng = np.random.default_rng(193)
    direction = wp.array(
        rng.normal(size=(owner.n, owner.k.Q)).astype(PRECISION),
        dtype=owner.k.qvec,
        device=owner.device,
    )
    actual = [wp.zeros_like(direction), wp.zeros_like(direction)]
    expected = [wp.zeros_like(direction), wp.zeros_like(direction)]

    def program() -> None:
        step.forward()
        if invalidate_with_prepare:
            step.refresh_linearization()
            step.prepare()
        # Poisoning makes the stale-cache failure independent of how close the
        # final Newton trial is to its predecessor, including rest-state solves.
        step.rows.grad.fill_(float("nan"))
        step.node_rows.grad.fill_(float("nan"))
        step.matvec(direction, actual[0], actual[0])
        step.transpose_matvec(direction, actual[1], actual[1])
        step.refresh_linearization()
        step.matvec(direction, expected[0], expected[0])
        step.transpose_matvec(direction, expected[1], expected[1])

    if owner.device.is_cuda:
        with wp.ScopedCapture(device=owner.device, force_module_load=True) as capture:
            program()
        wp.capture_launch(capture.graph)
    else:
        program()
    wp.synchronize_device(owner.device)
    assert step.code.numpy()[0] == 0
    for observed, reference in zip(actual, expected):
        assert np.all(np.isfinite(observed.numpy()))
        np.testing.assert_allclose(
            observed.numpy(),
            reference.numpy(),
            rtol=4e-5 if PRECISION == "float32" else 2e-10,
            atol=2e-6 if PRECISION == "float32" else 2e-12,
        )
