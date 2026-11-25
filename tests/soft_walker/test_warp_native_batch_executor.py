"""Graph submission ordering, ownership, cleanup, and scalar numerical parity."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from envs.soft_walker.engine.warp_native.episode import Episode
    from envs.soft_walker.engine.warp_native.solid_episode import SolidEpisode


import os
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

wp = pytest.importorskip("warp")
from envs.soft_walker.engine.warp_native.batching import EpisodeBatchExecutor


class _Readback:
    def __init__(self, events: list[tuple[object, ...]], label: str, value: float | int) -> None:
        self.events, self.label, self.value = events, label, value

    def numpy(self) -> np.ndarray:
        self.events.append(("read", self.label))
        return np.asarray([self.value])


class _Episode:
    def __init__(
        self, device: SimpleNamespace, events: list[tuple[object, ...]], label: str
    ) -> None:
        self.device, self.events, self.label = device, events, label
        self.owner = object()
        self.graphs = {"forward": label, "gradient": label + "_gradient"}
        self.loss = _Readback(events, label, 1.0)
        self.final = SimpleNamespace(status=_Readback(events, label, 0))
        self.adjoint_status = _Readback(events, label, 0)
        self.history_bytes, self.steps, self.root_cache_bytes = 20, 2, 0
        self.reuse_replay_roots = False
        self.step = SimpleNamespace(capacity=4)
        self.compile_seconds = {}
        self.completed = self.last_was_gradient = False

    def compile(self, gradient: bool = False) -> None:
        self.events.append(("compile", self.label, gradient))
        self.compile_seconds["gradient" if gradient else "forward"] = 0.01

    def run(self, gradient: bool = False) -> dict[str, int | float]:
        self.events.append(("scalar", self.label, gradient))
        return dict(loss=1.0, status=0, execution_seconds=0.001)


@pytest.fixture
def fake_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[SimpleNamespace, list[tuple[object, ...]], list[object]]:
    """Record stream submission and readback order without allocating GPU resources."""
    events, streams = [], []
    device = SimpleNamespace(is_cuda=True, alias="cuda:0", stream=object())

    class Stream:
        def __init__(self, _device: SimpleNamespace) -> None:
            assert _device is device
            self.index = len(streams)
            streams.append(self)

        def wait_stream(self, other: object) -> None:
            assert other is device.stream
            events.append(("wait", self.index))

    def launch(graph: str, *, stream: object) -> None:
        events.append(("launch", graph, stream.index))

    monkeypatch.setattr(wp, "Stream", Stream)
    monkeypatch.setattr(wp, "capture_launch", launch)
    monkeypatch.setattr(
        wp, "synchronize_stream", lambda stream: events.append(("join", stream.index))
    )
    monkeypatch.setattr(
        wp,
        "synchronize_device",
        lambda *_: pytest.fail("Batch executor must not synchronize the entire device"),
    )
    return device, events, streams


@pytest.mark.parametrize("gradient", [False, True])
def test_compile_all_submit_all_then_join_and_collect(
    fake_cuda: tuple[SimpleNamespace, list[tuple[object, ...]], list[object]], gradient: bool
) -> None:
    device, events, streams = fake_cuda
    episodes = [_Episode(device, events, str(i)) for i in range(3)]
    executor = EpisodeBatchExecutor(execution="parallel")
    results = executor.run(episodes, gradient=gradient)
    labels = [event[0] for event in events]
    assert max(i for i, label in enumerate(labels) if label == "compile") < labels.index("launch")
    assert max(i for i, label in enumerate(labels) if label == "launch") < labels.index("join")
    assert max(i for i, label in enumerate(labels) if label == "join") < labels.index("read")
    assert all(ep.completed and ep.last_was_gradient is gradient for ep in episodes)
    assert all(result["execution_mode"] == "parallel" for result in results)
    assert len({result["batch_execution_seconds"] for result in results}) == 1
    assert all(
        result["execution_seconds"] == result["batch_execution_seconds"] for result in results
    )
    assert all(result["adjoint_status"] == (0 if gradient else None) for result in results)
    executor.run(episodes, gradient=gradient)
    assert len(streams) == 3
    executor.close()
    executor.close()
    with pytest.raises(RuntimeError, match="closed"):
        executor.run(episodes)


def test_launch_exception_joins_every_possibly_submitted_stream(
    fake_cuda: tuple[SimpleNamespace, list[tuple[object, ...]], list[object]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device, events, _ = fake_cuda
    episodes = [_Episode(device, events, str(i)) for i in range(3)]
    for episode in episodes:
        episode.completed = episode.last_was_gradient = True

    def fail_second(graph: str, *, stream: object) -> None:
        events.append(("launch", graph, stream.index))
        if stream.index == 1:
            raise RuntimeError("injected launch failure")

    monkeypatch.setattr(wp, "capture_launch", fail_second)
    executor = EpisodeBatchExecutor()
    with pytest.raises(RuntimeError, match="injected launch failure"):
        executor.run(episodes)
    assert [event[1] for event in events if event[0] == "join"] == [0, 1]
    assert not any(event[0] == "read" for event in events)
    assert not any(ep.completed for ep in episodes)
    executor.close()


def test_compile_exception_never_submits_a_partial_batch(
    fake_cuda: tuple[SimpleNamespace, list[tuple[object, ...]], list[object]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device, events, _ = fake_cuda
    episodes = [_Episode(device, events, str(i)) for i in range(2)]

    def fail_compile(*_: object) -> None:
        raise RuntimeError("injected compile failure")

    monkeypatch.setattr(episodes[1], "compile", fail_compile)
    with pytest.raises(RuntimeError, match="compile failure"):
        EpisodeBatchExecutor().run(episodes)
    assert not any(event[0] in ("launch", "join", "read") for event in events)


@pytest.mark.parametrize(
    "case", ["duplicate_episode", "duplicate_owner", "mixed_device", "shared_storage"]
)
def test_invalid_batch_rejected_before_compile(
    fake_cuda: tuple[SimpleNamespace, list[tuple[object, ...]], list[object]], case: str
) -> None:
    device, events, _ = fake_cuda
    episodes = [_Episode(device, events, str(i)) for i in range(2)]
    if case == "duplicate_episode":
        episodes[1] = episodes[0]
    elif case == "duplicate_owner":
        episodes[1].owner = episodes[0].owner
    elif case == "mixed_device":
        episodes[1].device = SimpleNamespace(is_cuda=False, alias="cpu")
    else:
        shared = wp.zeros(4, dtype=wp.float64, device="cpu")
        episodes[0].step.shared = episodes[1].step.shared = shared
    with pytest.raises(ValueError):
        EpisodeBatchExecutor().run(episodes)
    assert events == []


def test_rebinding_after_reset_checks_new_owner_before_launch(
    fake_cuda: tuple[SimpleNamespace, list[tuple[object, ...]], list[object]],
) -> None:
    device, events, _ = fake_cuda
    episodes = [_Episode(device, events, str(i)) for i in range(2)]
    executor = EpisodeBatchExecutor()
    executor.run(episodes)
    events.clear()
    episodes[1] = _Episode(device, events, "replacement")
    episodes[1].owner = episodes[0].owner
    with pytest.raises(ValueError, match="independent models"):
        executor.run(episodes)
    assert events == []
    executor.close()


def test_serial_supports_run_only_episode_protocol() -> None:
    calls = []

    class RunOnly:
        def run(self, gradient: bool = False) -> dict[str, int | float]:
            calls.append(gradient)
            return dict(status=0, execution_seconds=0.2)

    results = EpisodeBatchExecutor(execution="serial").run([RunOnly(), RunOnly()], gradient=True)
    assert calls == [True, True]
    assert all(r["execution_mode"] == "serial" and r["execution_seconds"] == 0.2 for r in results)
    assert results[0]["batch_execution_seconds"] == results[1]["batch_execution_seconds"]


def test_modes_and_cpu_parallel_rejection() -> None:
    with pytest.raises(ValueError, match="execution"):
        EpisodeBatchExecutor(execution="threads")
    device, events = SimpleNamespace(is_cuda=False, alias="cpu"), []
    episode = _Episode(device, events, "cpu")
    with pytest.raises(ValueError, match="CUDA"):
        EpisodeBatchExecutor(execution="parallel").run([episode])
    assert events == []
    result = EpisodeBatchExecutor().run([episode])[0]
    assert result["execution_mode"] == "serial"
    with pytest.raises(ValueError, match="boolean"):
        EpisodeBatchExecutor().run([episode], gradient=1)
    assert EpisodeBatchExecutor().run([]) == []


def _native_episode(
    physics: str, command: float, device: str, precision: str
) -> Episode | SolidEpisode:
    """Build independent real episodes with nonzero commands for lane-isolation checks."""
    if physics == "mpm2":
        from envs.soft_walker.engine.warp_native.episode import Episode
        from envs.soft_walker.engine.warp_native.model import DeviceModel, uniform_body

        bundle = uniform_body(2, precision=precision)
        bundle.config = replace(
            bundle.config,
            krylov_method="adaptive_dynamic",
            cache_tangent=True,
            dt=0.001,
            gravity=0,
            contact_enabled=False,
        )
        owner = DeviceModel(bundle, device)
    else:
        from envs.soft_walker.engine.warp_native.solid_builders import regular_solid
        from envs.soft_walker.engine.warp_native.solid_episode import SolidEpisode as Episode
        from envs.soft_walker.engine.warp_native.solid_model import SolidDeviceModel

        bundle = regular_solid("spring", shape=(1, 1), channels=1, precision=precision)
        bundle.config = replace(bundle.config, dt=0.001, gravity=0, contact_enabled=False)
        owner = SolidDeviceModel(bundle, device)
    return Episode(
        owner,
        np.asarray([[command], [-0.5 * command]], dtype=precision),
        checkpoint_stride=1,
        velocity_weight=0.1,
        work_weight=0.05,
    )


@pytest.mark.parametrize("physics", ["spring", "mpm2"])
@pytest.mark.parametrize("gradient", [False, True])
def test_native_batch_matches_independent_scalar_episodes(physics: str, gradient: bool) -> None:
    device = os.environ.get("WARP_TEST_DEVICE", "cpu")
    precision = os.environ.get("WARP_TEST_DTYPE", "float64")
    episodes = [_native_episode(physics, command, device, precision) for command in (0.1, -0.15)]
    reference = [ep.run(gradient) for ep in episodes]
    names = (
        ("u", "v", "rotation", "omega", "actuator", "ledger")
        if physics == "spring"
        else ("x", "v", "F", "C", "actuator", "ledger")
    )
    states = [{name: getattr(ep.final, name).numpy().copy() for name in names} for ep in episodes]
    derivatives = [deepcopy(ep.gradients()) for ep in episodes] if gradient else None
    executor = EpisodeBatchExecutor()
    try:
        results = executor.run(episodes, gradient)
        for index, (ep, actual, expected) in enumerate(zip(episodes, results, reference)):
            assert actual["status"] == expected["status"] == 0
            assert (
                actual["adjoint_status"] == expected["adjoint_status"] == (0 if gradient else None)
            )
            rtol, atol = (8e-4, 3e-6) if precision == "float32" else (3e-8, 2e-10)
            np.testing.assert_allclose(actual["loss"], expected["loss"], rtol=rtol, atol=atol)
            for name in names:
                np.testing.assert_allclose(
                    getattr(ep.final, name).numpy(), states[index][name], rtol=rtol, atol=atol
                )
            if gradient:
                actual_grad = ep.gradients()
                for group in ("model", "initial"):
                    for name, expected_grad in derivatives[index][group].items():
                        np.testing.assert_allclose(
                            actual_grad[group][name], expected_grad, rtol=rtol, atol=atol
                        )
                np.testing.assert_allclose(
                    actual_grad["commands"], derivatives[index]["commands"], rtol=rtol, atol=atol
                )
    finally:
        executor.close()
