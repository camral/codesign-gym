"""Reusable integer inclusive scan for CUDA conditional graph bodies.

Warp 1.16's CUDA ``array_scan`` allocates CUB scratch on every invocation.
This implementation instead allocates a hierarchy of block totals before
capture, then uses only kernel launches during each CUDA scan. The CPU path
retains Warp's existing scan implementation.
"""

import warp as wp
from warp.utils import array_scan

_BLOCK_SIZE = 256


@wp.kernel(enable_backward=False)
def _scan_blocks(
    values: wp.array(dtype=int),
    prefix: wp.array(dtype=int),
    totals: wp.array(dtype=int),
):
    block, lane = wp.tid()
    # Bounds-checked tile loads pad the final partial block with zeros;
    # bounds-checked stores write only the actual array entries.
    tile = wp.tile_load(values, shape=_BLOCK_SIZE, offset=block * _BLOCK_SIZE)
    scanned = wp.tile_scan_inclusive(tile)
    wp.tile_store(prefix, scanned, offset=block * _BLOCK_SIZE)
    last = wp.tile_extract(scanned, _BLOCK_SIZE - 1)
    if lane == _BLOCK_SIZE - 1:
        totals[block] = last


@wp.kernel(enable_backward=False)
def _add_block_offsets(prefix: wp.array(dtype=int), totals_prefix: wp.array(dtype=int)):
    i = wp.tid()
    block = i // _BLOCK_SIZE
    if block > 0:
        prefix[i] += totals_prefix[block - 1]


class InclusiveIntScan:
    """Inclusive int32 prefix sum with fixed-size, reusable scratch buffers.

    Construct this object before graph capture, then call it with contiguous
    one-dimensional int32 input/output arrays of the specified length and
    device. Exact in-place use is supported; partially overlapping arrays are
    rejected. Prefix sums must fit in int32, as they do for binary flags with
    ``count < 2**31``. A workspace may be reused by sequential scans, including
    graph replays; concurrent scans require separate workspaces.
    """

    def __init__(self, count: "int", device: "str | wp.Device") -> "None":
        """Preallocate a fixed-length int32 hierarchical scan before graph capture."""
        if type(count) is not int or not 0 <= count < 2**31:
            raise ValueError("Scan count must be an integer in 0..2**31-1")
        self.count = count
        self.device = wp.get_device(device)
        if self.device.is_capturing:
            raise RuntimeError("Construct the scan workspace before graph capture")
        self._totals = []
        self._prefixes = []
        if not self.device.is_cuda:
            return

        length = count
        while length:
            blocks = (length + _BLOCK_SIZE - 1) // _BLOCK_SIZE
            self._totals.append(wp.empty(blocks, dtype=int, device=self.device))
            if blocks == 1:
                break
            self._prefixes.append(wp.empty(blocks, dtype=int, device=self.device))
            length = blocks

    def __call__(self, values: "wp.array", prefix: "wp.array") -> "None":
        """Write an inclusive int32 prefix sum without allocating during capture.

        Inputs and outputs must be contiguous arrays on this scan's device.
        Exact in-place operation is allowed; partial storage overlap is rejected."""
        for label, array in (("input", values), ("output", prefix)):
            if not isinstance(array, wp.array) or array.dtype is not wp.int32:
                raise TypeError(f"Scan {label} must be a Warp int32 array")
            if array.ndim != 1 or array.shape != (self.count,) or not array.is_contiguous:
                raise ValueError(f"Scan {label} must be contiguous with shape ({self.count},)")
            if array.device != self.device:
                raise ValueError(f"Scan {label} must reside on {self.device}")

        if self.count == 0:
            return
        if values.ptr != prefix.ptr:
            size_bytes = self.count * 4
            if max(values.ptr, prefix.ptr) < min(values.ptr + size_bytes, prefix.ptr + size_bytes):
                raise ValueError("Scan arrays must not partially overlap")
        if not self.device.is_cuda:
            array_scan(values, prefix, inclusive=True)
            return

        # Each level scans contiguous blocks and supplies the next level's
        # input. The last level fits in one block and is globally scanned.
        for level, totals in enumerate(self._totals):
            source = values if level == 0 else self._totals[level - 1]
            output = prefix if level == 0 else self._prefixes[level - 1]
            wp.launch(
                _scan_blocks,
                dim=(totals.shape[0], _BLOCK_SIZE),
                block_dim=_BLOCK_SIZE,
                inputs=[source, output, totals],
                device=self.device,
                record_tape=False,
            )

        # Propagate global block offsets back down the hierarchy. Every
        # scratch entry is overwritten on each call, so replay needs no reset.
        for level in range(len(self._totals) - 2, -1, -1):
            output = prefix if level == 0 else self._prefixes[level - 1]
            wp.launch(
                _add_block_offsets,
                dim=output.shape[0],
                inputs=[output, self._prefixes[level]],
                device=self.device,
                record_tape=False,
            )
