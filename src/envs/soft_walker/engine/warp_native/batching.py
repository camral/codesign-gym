"""Independent native episodes submitted concurrently on CUDA streams.

Each lane retains its own implicit solver, stopping criteria, and failure state.
This is graph-level concurrency, not a single concatenated Krylov problem. All
commands must be uploaded before ``run``; results remain a host API boundary.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from .episode import Episode
    from .host_types import EpisodeResult
    from .solid_episode import SolidEpisode

import time
from types import SimpleNamespace

import warp as wp

from ._episode_host import collect_result


def _array_ranges(episode: "Episode | SolidEpisode") -> "list[tuple[int, int]]":
    """Inspect storage without downloading arrays or traversing Warp's runtime."""
    pending, seen, ranges = list(vars(episode).values()), set(), set()
    while pending:
        value = pending.pop()
        if id(value) in seen:
            continue
        seen.add(id(value))
        if isinstance(value, wp.array):
            if value.ptr and value.capacity:
                ranges.add((int(value.ptr), int(value.ptr) + int(value.capacity)))
            if value.grad is not None:
                pending.append(value.grad)
        elif isinstance(value, dict):
            pending.extend(value.values())
        elif isinstance(value, (tuple, list)):
            pending.extend(value)
        elif not isinstance(value, type) and hasattr(value, "_cls"):
            pending.extend(getattr(value, name) for name in value._cls.vars)
        elif isinstance(value, SimpleNamespace) or (
            hasattr(value, "__dict__")
            and (
                ".warp_native." in type(value).__module__
                or type(value).__module__.startswith("warp._src.optim.")
            )
        ):
            pending.extend(vars(value).values())
    return sorted(ranges)


def _overlap(left: "Sequence[tuple[int, int]]", right: "Sequence[tuple[int, int]]") -> "bool":
    """Test sorted half-open device address ranges for any shared storage."""
    i = j = 0
    while i < len(left) and j < len(right):
        a, b = left[i], right[j]
        if a[0] < b[1] and b[0] < a[1]:
            return True
        if a[1] <= b[0]:
            i += 1
        else:
            j += 1
    return False


class EpisodeBatchExecutor:
    """Run separately owned episodes, with a persistent CUDA stream per lane.

    ``auto`` uses parallel CUDA graphs and a serial CPU path. ``parallel``
    requires one CUDA device; ``serial`` is the scalar reference execution.
    Mixed devices, duplicate episodes, and shared model/workspace storage are
    rejected before compilation or execution. Episode buffers must remain
    fixed after construction, as their captured graphs already require.

    For parallel execution, ``execution_seconds`` on every result is the same
    batch latency, from graph submission to completion of all participating
    streams. It is *not* an isolated measurement of that lane. Compilation,
    host input-upload calls, and result downloads are excluded; waiting for
    any still-pending input uploads on the caller's stream is included. The identical
    ``batch_execution_seconds`` field makes that timing scope explicit. In
    serial mode ``execution_seconds`` retains the scalar episode timing.

    This object is synchronous and must not be shared between Python threads.
    It does not commit episode final states into their owners' initial states.
    """

    def __init__(self, execution: "str" = "auto") -> "None":
        """Select execution policy; streams are allocated lazily per CUDA lane."""
        if execution not in ("auto", "serial", "parallel"):
            raise ValueError("execution must be 'auto', 'serial', or 'parallel'")
        self.execution = execution
        self._streams: dict[tuple[str, int], wp.Stream] = {}
        self._validated_binding: tuple[tuple[object, object], ...] | None = None
        self._closed = False

    def _validate(self, episodes: "Sequence[Episode | SolidEpisode]") -> "wp.Device | None":
        """Reject duplicate owners, mixed devices and overlapping device buffers."""
        if len({id(ep) for ep in episodes}) != len(episodes):
            raise ValueError("A batch cannot contain the same episode twice")
        owners = [ep.owner for ep in episodes if hasattr(ep, "owner")]
        if len({id(owner) for owner in owners}) != len(owners):
            raise ValueError("Batch episodes must own independent models")
        if self.execution == "serial":
            # Keep the scalar run-only protocol usable by adapters and CPU
            # lifecycle tests; native objects still receive device validation.
            devices = [ep.device for ep in episodes if hasattr(ep, "device")]
            if devices and any(device != devices[0] for device in devices):
                raise ValueError("All batch episodes must use the same device")
            return devices[0] if devices else None
        device = episodes[0].device
        if any(ep.device != device for ep in episodes):
            raise ValueError("All batch episodes must use the same device")
        if self.execution == "parallel" and not device.is_cuda:
            raise ValueError("Parallel episode execution requires a CUDA device")
        binding = tuple((ep, ep.owner) for ep in episodes)
        if binding != self._validated_binding:
            # Captured native episode buffers are fixed. Inspect each new
            # binding once, without adding an array walk to every policy step.
            ranges = [_array_ranges(ep) for ep in episodes]
            for i, current in enumerate(ranges):
                if any(_overlap(current, previous) for previous in ranges[:i]):
                    raise ValueError("Batch episodes must not share model or workspace storage")
            self._validated_binding = binding
        return device

    @staticmethod
    def _collect(
        episode: "Episode | SolidEpisode", gradient: "bool", seconds: "float"
    ) -> "EpisodeResult":
        """Download scalar loss/status metadata after a lane's graph has completed."""
        return collect_result(episode, gradient, seconds)

    def run(
        self, episodes: "Iterable[Episode | SolidEpisode]", gradient: "bool" = False
    ) -> "list[EpisodeResult]":
        """Finish all submitted graphs before reading any lane's host results."""
        if self._closed:
            raise RuntimeError("Episode batch executor is closed")
        if type(gradient) is not bool:
            raise ValueError("gradient must be a boolean")
        episodes = tuple(episodes)
        if not episodes:
            return []
        device = self._validate(episodes)
        parallel = self.execution != "serial" and cast("wp.Device", device).is_cuda
        if not parallel:
            start = time.perf_counter()
            results = [
                episode.run(gradient=True) if gradient else episode.run() for episode in episodes
            ]
            seconds = time.perf_counter() - start
            for result in results:
                result.update(
                    {
                        "execution_mode": "serial",
                        "batch_execution_seconds": seconds,
                        "batch_size": len(episodes),
                    }
                )
            return results

        # Loading modules and recording conditional graphs are setup work.
        # Never compile a later lane while earlier lanes are already running.
        for episode in episodes:
            episode.compile(gradient)
        for episode in episodes:
            episode.completed = False
            episode.last_was_gradient = False
        device = cast("wp.Device", device)
        input_stream = device.stream
        streams = []
        for index in range(len(episodes)):
            key = (device.alias, index)
            if key not in self._streams:
                self._streams[key] = wp.Stream(device)
            streams.append(self._streams[key])
        submitted = []
        start = time.perf_counter()
        try:
            for episode, stream in zip(episodes, streams):
                # Capture and input uploads happened on the caller's stream.
                # This event dependency does not make lane streams wait on
                # one another and does not block the host.
                stream.wait_stream(input_stream)
                submitted.append(stream)
                wp.capture_launch(
                    episode.graphs["gradient" if gradient else "forward"], stream=stream
                )
            for stream in submitted:
                wp.synchronize_stream(stream)
        except BaseException as exc:
            # A launch can fail after earlier lanes were submitted. Their
            # buffers must not be reset/freed while those graphs still run.
            for stream in submitted:
                try:
                    wp.synchronize_stream(stream)
                except BaseException as cleanup_error:
                    exc.add_note(f"Joining a submitted batch stream also failed: {cleanup_error}")
            raise
        seconds = time.perf_counter() - start
        for episode in episodes:
            episode.completed = True
            episode.last_was_gradient = gradient
        results = [self._collect(episode, gradient, seconds) for episode in episodes]
        for result in results:
            result.update(
                {
                    "execution_mode": "parallel",
                    "batch_execution_seconds": seconds,
                    "batch_size": len(episodes),
                }
            )
        return results

    def close(self) -> "None":
        """Join owned streams before releasing their references; idempotent."""
        if self._closed:
            return
        errors = []
        for stream in self._streams.values():
            try:
                wp.synchronize_stream(stream)
            except BaseException as exc:
                errors.append(exc)
        if errors:
            # Retain buffers and streams so a caller can investigate or retry.
            raise errors[0]
        self._streams.clear()
        self._validated_binding = None
        self._closed = True
