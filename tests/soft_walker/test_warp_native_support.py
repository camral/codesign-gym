"""Retained support-sort graphs survive other sort sizes and new inputs."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable


import os

import numpy as np
import pytest

wp = pytest.importorskip("warp")
from envs.soft_walker.engine.warp_native.support import CapturedSort

DEVICE = os.environ.get("WARP_TEST_DEVICE", "cpu")


@wp.kernel(enable_backward=False)
def stop_loop(condition: wp.array(dtype=int)):
    condition[0] = 0


def test_sort_graph_reuse_with_independent_larger_sort() -> None:
    device = wp.get_device(DEVICE)

    def make(
        count: "int",
    ) -> "tuple[wp.array, wp.array, CapturedSort, Callable[[], None], wp.Graph | None]":
        keys = wp.zeros(count * 2, dtype=int, device=device)
        values = wp.zeros(count * 2, dtype=int, device=device)
        sort = CapturedSort(keys, values, count)
        condition = wp.zeros(1, dtype=int, device=device)

        def body() -> "None":
            sort()
            wp.launch(stop_loop, dim=1, inputs=[condition], device=device)

        def program() -> "None":
            condition.fill_(1)
            wp.capture_while(condition, body)

        graph = None
        if device.is_cuda:
            with wp.ScopedCapture(device=device, force_module_load=True) as capture:
                program()
            graph = capture.graph
        return keys, values, sort, program, graph

    small = make(37)
    large = make(50003)
    rng = np.random.default_rng(917)
    for row in (small, large, small, large, small):
        keys, values, sort, program, graph = row
        n = sort.count
        data = rng.integers(0, 153, n, dtype=np.int32)
        data[-1] = np.iinfo(np.int32).max
        identities = np.arange(n, dtype=np.int32)
        keys.assign(np.r_[data, np.zeros(n, np.int32)])
        values.assign(np.r_[identities, np.zeros(n, np.int32)])
        if graph is None:
            program()
        else:
            wp.capture_launch(graph)
        wp.synchronize_device(device)
        indices = np.argsort(data, kind="stable")
        np.testing.assert_array_equal(keys.numpy()[:n], data[indices])
        np.testing.assert_array_equal(values.numpy()[:n], identities[indices])
