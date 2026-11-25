"""Precision-specialized bookkeeping for resident policy and objective callbacks.

These kernels move controller carry and scalar costs; they do not implement
physics or replace the implicit adjoint. An action boundary is measured after
settling in physical steps, with the same action hold as the episode.
"""

from functools import lru_cache
from types import SimpleNamespace

import warp as wp

from ...contracts.status_codes import ROLLOUT_INVALID_INPUT


@lru_cache(maxsize=2)
def callback_kernels(precision: str) -> SimpleNamespace:
    """Build float32/64 scalar buffers and device-side control-boundary operations."""
    if precision not in ("float32", "float64"):
        raise ValueError("Callback precision must be float32 or float64")
    real = wp.float32 if precision == "float32" else wp.float64
    nan = wp.constant(real(float("nan")))
    module = f"soft_walker_native_callbacks_{precision}"

    @wp.kernel(module=module, enable_backward=False)
    def guard_loss(status: wp.array(dtype=int), loss: wp.array(dtype=real)):
        if not wp.isfinite(loss[0]):
            wp.atomic_max(status, 0, ROLLOUT_INVALID_INPUT)
        if status[0] != 0:
            loss[0] = nan

    @wp.kernel(module=module, enable_backward=False)
    def boundary(
        clock: wp.array(dtype=int),
        settling: int,
        hold: int,
        condition: wp.array(dtype=int),
        index: wp.array(dtype=int),
    ):
        step = clock[0] - settling
        condition[0] = int(step >= 0 and step % hold == 0)
        index[0] = wp.max(step // hold, 0)

    @wp.kernel(module=module, enable_backward=False)
    def load_carry(
        history: wp.array2d(dtype=real), index: wp.array(dtype=int), carry: wp.array(dtype=real)
    ):
        i = wp.tid()
        carry[i] = history[index[0], i]

    @wp.kernel(module=module, enable_backward=False)
    def save_policy(
        command: wp.array(dtype=real),
        carry: wp.array(dtype=real),
        index: wp.array(dtype=int),
        table: wp.array2d(dtype=real),
        history: wp.array2d(dtype=real),
        status: wp.array(dtype=int),
    ):
        i = wp.tid()
        if i < command.shape[0]:
            value = command[i]
            table[index[0], i] = value
            if not wp.isfinite(value) or wp.abs(value) > real(1):
                wp.atomic_max(status, 0, ROLLOUT_INVALID_INPUT)
        if i < carry.shape[0]:
            value = carry[i]
            history[index[0] + 1, i] = value
            if not wp.isfinite(value):
                wp.atomic_max(status, 0, ROLLOUT_INVALID_INPUT)

    @wp.kernel(module=module, enable_backward=False)
    def initial_carry(value: wp.array(dtype=real), history: wp.array2d(dtype=real)):
        i = wp.tid()
        history[0, i] = value[i]

    @wp.kernel(module=module, enable_backward=False)
    def load_command_seed(
        table: wp.array2d(dtype=real), index: wp.array(dtype=int), seed: wp.array(dtype=real)
    ):
        i = wp.tid()
        seed[i] = table[index[0], i]

    @wp.kernel(module=module, enable_backward=False)
    def accumulate_cost(
        value: wp.array(dtype=real),
        clock: wp.array(dtype=int),
        settling: int,
        include_settling: int,
        total: wp.array(dtype=real),
    ):
        if include_settling != 0 or clock[0] >= settling:
            total[0] = total[0] + value[0]

    @wp.kernel(module=module, enable_backward=False)
    def cost_seed(
        total_gradient: wp.array(dtype=real),
        clock: wp.array(dtype=int),
        settling: int,
        include_settling: int,
        seed: wp.array(dtype=real),
    ):
        seed[0] = real(0)
        if include_settling != 0 or clock[0] >= settling:
            seed[0] = total_gradient[0]

    return SimpleNamespace(**locals())
