"""Integer scan boundaries and reuse inside nested CUDA conditional graphs.

Set WARP_TEST_DEVICE=cuda for the capture/allocation regression. CPU runs
verify the common interface and reference behavior, using Warp's CPU scan.
"""

import os

import numpy as np
import pytest

wp = pytest.importorskip("warp")

from envs.soft_walker.engine.warp_native.scan import InclusiveIntScan

DEVICE = os.environ.get("WARP_TEST_DEVICE", "cpu")


@wp.kernel(enable_backward=False)
def _advance(count: wp.array(dtype=int), condition: wp.array(dtype=int)):
    count[0] += 1
    condition[0] = wp.where(count[0] < 2, 1, 0)


@pytest.mark.parametrize("count", [0, 1, 255, 256, 257, 50003, 65537])
def test_scan_nested_capture_reuses_changed_inputs(count: "int") -> None:
    wp.init()
    device = wp.get_device(DEVICE)
    if device.is_cuda and not wp.is_conditional_graph_supported():
        pytest.fail("This regression requires CUDA conditional graph support")
    values = wp.zeros(count, dtype=int, device=device)
    prefix = wp.zeros_like(values)
    scan = InclusiveIntScan(count, device)
    scratch_ptrs = tuple(array.ptr for array in scan._totals + scan._prefixes)
    outer = wp.zeros(1, dtype=int, device=device)
    inner = wp.zeros_like(outer)
    outer_count = wp.zeros_like(outer)
    inner_count = wp.zeros_like(outer)

    def inner_body() -> "None":
        scan(values, prefix)
        wp.launch(_advance, dim=1, inputs=[inner_count, inner], device=device)

    def outer_body() -> "None":
        inner.fill_(1)
        inner_count.zero_()
        wp.capture_while(inner, inner_body)
        wp.launch(_advance, dim=1, inputs=[outer_count, outer], device=device)

    def program() -> "None":
        outer.fill_(1)
        outer_count.zero_()
        wp.capture_while(outer, outer_body)

    if device.is_cuda:
        with wp.ScopedCapture(device=device, force_module_load=True) as capture:
            program()

    rng = np.random.default_rng(1729 + count)
    patterns = (
        np.ones(count, dtype=np.int32),
        np.zeros(count, dtype=np.int32),
        rng.integers(0, 2, size=count, dtype=np.int32),
    )
    for pattern in patterns:
        values.assign(pattern)
        prefix.fill_(-1)
        if device.is_cuda:
            wp.capture_launch(capture.graph)
        else:
            program()
        wp.synchronize_device(device)
        np.testing.assert_array_equal(prefix.numpy(), np.cumsum(pattern, dtype=np.int32))
        assert int(outer_count.numpy()[0]) == 2
        assert int(inner_count.numpy()[0]) == 2
        assert tuple(array.ptr for array in scan._totals + scan._prefixes) == scratch_ptrs


def test_scan_accepts_exact_in_place_use() -> None:
    wp.init()
    device = wp.get_device(DEVICE)
    source = np.arange(513, dtype=np.int32) % 2
    values = wp.array(source, dtype=int, device=device)
    scan = InclusiveIntScan(len(source), device)
    if device.is_cuda:
        with wp.ScopedCapture(device=device, force_module_load=True) as capture:
            scan(values, values)
        wp.capture_launch(capture.graph)
    else:
        scan(values, values)
    wp.synchronize_device(device)
    np.testing.assert_array_equal(values.numpy(), np.cumsum(source, dtype=np.int32))


def test_scan_rejects_incompatible_arrays() -> None:
    wp.init()
    device = wp.get_device(DEVICE)
    scan = InclusiveIntScan(256, device)
    storage = wp.zeros(257, dtype=int, device=device)
    values = storage[:256]
    output = wp.zeros_like(values)
    with pytest.raises(ValueError, match="partially overlap"):
        scan(values, storage[1:])
    with pytest.raises(ValueError, match="shape"):
        scan(storage, output)
    with pytest.raises(TypeError, match="int32"):
        scan(wp.zeros(256, dtype=wp.float32, device=device), output)
