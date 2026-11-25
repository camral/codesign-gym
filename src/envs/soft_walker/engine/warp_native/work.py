"""Positive-work value and the reference derivative at its zero-work kink."""

from functools import lru_cache

import warp as wp


@lru_cache(maxsize=2)
def positive_work_function(precision: "str") -> "wp.Function":
    """Keep max(work, 0) values and match JAX's half derivative at exact zero."""
    if precision not in ("float32", "float64"):
        raise ValueError("Expected float32 or float64")
    real = wp.float32 if precision == "float32" else wp.float64
    module = f"envs.soft_walker.engine.warp_native.work_{precision}_v1"

    @wp.func(module=module)
    def positive_work(value: real):
        return wp.max(value, real(0))

    @wp.func_grad(positive_work)
    def positive_work_grad(value: real, adj_ret: real):
        slope = real(0)
        if value > real(0):
            slope = real(1)
        elif value == real(0):
            slope = real(0.5)
        wp.adjoint[value] += slope * adj_ret

    return positive_work
