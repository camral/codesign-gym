"""Capture-safe support sorting with scratch isolated from other episodes."""

import warp as wp
from warp.utils import radix_sort_pairs

# Warp >= 1.17 makes radix_sort_pairs capture-safe itself: inside a conditional (while/if) body it takes the
# sort scratch from a per-capture side allocation retained until the graph is destroyed (sort.cu cached_side_alloc).
# A separately recorded sort graph, by contrast, now holds a memory-allocation node, which CUDA rejects as a child
# of a conditional node. So on >= 1.17 the sort is called directly and no private graph is recorded.
_NATIVE_CAPTURE_SAFE_SORT = tuple(int(v) for v in wp.__version__.split(".")[:2]) >= (1, 17)


class CapturedSort:
    """Reserve CUB scratch outside loops and retain it for this sort graph.

    Warp 1.16 caches radix-sort scratch per stream and may replace it when a
    larger sort uses that stream. A private setup stream gives each instance
    its own retained cache. The resulting graph contains only sort/copy work;
    replay inserts it into the caller's stream without switching streams.
    """

    def __init__(self, keys: "wp.array", values: "wp.array", count: "int") -> "None":
        """Bind key/value buffers and reserve version-specific capture-safe scratch."""
        self.keys, self.values, self.count = keys, values, count
        self.device = keys.device
        self.graph = None
        self.stream = None
        self.enabled = None
        if self.device.is_cuda and not _NATIVE_CAPTURE_SAFE_SORT:
            if self.device.is_capturing:
                raise RuntimeError("Construct support sorting before graph capture")
            self.stream = wp.Stream(self.device)
            self.enabled = wp.ones(1, dtype=int, device=self.device)
            with wp.ScopedStream(self.stream, sync_enter=True, sync_exit=True):
                # Establish the fixed-size scratch cache before recording.
                radix_sort_pairs(keys, values, count)
                with wp.ScopedCapture(stream=self.stream) as capture:
                    radix_sort_pairs(keys, values, count)
                # Warp instantiates graph executables lazily on first launch;
                # instantiate here, before nesting this graph in another capture.
                wp.capture_launch(capture.graph, stream=self.stream)
            wp.synchronize_stream(self.stream)
            self.graph = capture.graph

    def __call__(self) -> "None":
        """Sort the bound key/value pairs using this instance's retained workspace."""
        if self.graph is None:
            radix_sort_pairs(self.keys, self.values, self.count)
        elif self.device.is_capturing:
            # capture_launch executes a graph; capture_if embeds an existing
            # graph as a child in Warp 1.16. This condition is permanently true.
            wp.capture_if(self.enabled, self.graph)
        else:
            wp.capture_launch(self.graph)
