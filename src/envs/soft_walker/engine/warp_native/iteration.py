"""Device-side convergence, safeguarded Newton, and diagnostic reductions.

Small scalar kernels are synchronization boundaries for globally reduced
quantities. No result is read by Python inside a captured CUDA simulation.
"""

from functools import lru_cache
from types import SimpleNamespace

import warp as wp

from ...contracts.metrics import (
    COUNT_BACKTRACKS,
    COUNT_LINE_SEARCH_TRIALS,
    COUNT_LINEAR_SOLVES,
    COUNT_NEWTON_ITERATIONS,
    WORK_REPLAY_ROOT_CHECK,
    WORK_REPLAY_ROOT_FALLBACK,
)
from ...contracts.status_codes import (
    ROOT_INVALID_INITIAL,
    ROOT_LINE_SEARCH_FAILED,
    ROOT_LINEAR_FAILED,
    ROOT_MAX_NEWTON,
    ROOT_OK,
)
from .kernels import kernels


@lru_cache(maxsize=12)
def iteration_kernels(
    dim: "int", precision: "str", vector_width: "int | None" = None
) -> "SimpleNamespace":
    """Return the cached Newton, backtracking and linear-residual control namespace.

    Types and kernel identities depend on the requested physics/dimension and
    precision. Warp DSL annotations must be evaluated in this factory's
    scope; postponed annotations and relocation change kernel construction."""
    k = kernels(dim, precision)
    real, vec, Parameters = k.real, k.vec, k.Parameters
    D = dim if vector_width is None else vector_width
    vec = wp.types.vector(length=D, dtype=real)
    module = f"envs.soft_walker.engine.warp_native.iteration_d{dim}_v{D}_{precision}_v2"
    not_a_number = wp.constant(real(float("nan")))

    @wp.kernel(module=module, enable_backward=False)
    def maximum(x: wp.array(dtype=vec), norm: wp.array(dtype=real), bad: wp.array(dtype=int)):
        i = wp.tid()
        value = real(0)
        for axis in range(D):
            value = wp.max(value, wp.abs(x[i][axis]))
            if not wp.isfinite(x[i][axis]):
                wp.atomic_max(bad, 0, 1)
        wp.atomic_max(norm, 0, value)

    @wp.kernel(module=module, enable_backward=False)
    def linear_error(ax: wp.array(dtype=vec), b: wp.array(dtype=vec), sums: wp.array(dtype=real)):
        i = wp.tid()
        difference = ax[i] - b[i]
        wp.atomic_add(sums, 0, wp.dot(difference, difference))
        wp.atomic_add(sums, 1, wp.dot(b[i], b[i]))

    @wp.kernel(module=module, enable_backward=False)
    def negate(x: wp.array(dtype=vec), out: wp.array(dtype=vec)):
        i = wp.tid()
        out[i] = -x[i]

    @wp.kernel(module=module, enable_backward=False)
    def fallback(x: wp.array(dtype=vec), bad: wp.array(dtype=int)):
        i = wp.tid()
        if bad[0] != 0:
            x[i] = vec()

    @wp.kernel(module=module, enable_backward=False)
    def initialize(
        norm: wp.array(dtype=real),
        bad: wp.array(dtype=int),
        previous_status: wp.array(dtype=int),
        cfg: Parameters,
        code: wp.array(dtype=int),
        counts: wp.array(dtype=int),
        condition: wp.array(dtype=int),
        residual: wp.array(dtype=real),
    ):
        code[0] = ROOT_OK
        if bad[0] != 0 or previous_status[0] != 0 or not wp.isfinite(norm[0]):
            code[0] = ROOT_INVALID_INITIAL
        for i in range(counts.shape[0]):
            counts[i] = 0
        residual[0] = norm[0]
        condition[0] = int(code[0] == ROOT_OK and norm[0] > cfg.atol)

    @wp.kernel(module=module, enable_backward=False)
    def check_replayed_root(
        norm: wp.array(dtype=real),
        bad: wp.array(dtype=int),
        previous_status: wp.array(dtype=int),
        cfg: Parameters,
        code: wp.array(dtype=int),
        counts: wp.array(dtype=int),
        condition: wp.array(dtype=int),
        residual: wp.array(dtype=real),
        work_counts: wp.array(dtype=int),
    ):
        work_counts[WORK_REPLAY_ROOT_CHECK] += 1
        for i in range(counts.shape[0]):
            counts[i] = 0
        residual[0] = norm[0]
        if previous_status[0] != 0:
            code[0] = ROOT_INVALID_INITIAL
        retry = code[0] == ROOT_OK and (
            bad[0] != 0 or not wp.isfinite(norm[0]) or norm[0] > cfg.atol
        )
        condition[0] = int(retry)
        if retry:
            work_counts[WORK_REPLAY_ROOT_FALLBACK] += 1

    @wp.kernel(module=module, enable_backward=False)
    def start_linesearch(
        sums: wp.array(dtype=real),
        cfg: Parameters,
        code: wp.array(dtype=int),
        counts: wp.array(dtype=int),
        condition: wp.array(dtype=int),
        line_condition: wp.array(dtype=int),
        accepted: wp.array(dtype=int),
        max_linear: wp.array(dtype=real),
    ):
        denominator = sums[1] if sums[1] > real(0) else real(1)
        error = wp.sqrt(sums[0] / denominator)
        max_linear[0] = wp.max(max_linear[0], error)
        counts[COUNT_LINEAR_SOLVES] = counts[COUNT_LINEAR_SOLVES] + 1
        if not wp.isfinite(error) or error > cfg.linear_limit:
            code[0] = ROOT_LINEAR_FAILED
            condition[0] = 0
        counts[COUNT_LINE_SEARCH_TRIALS] = 0
        accepted[0] = 0
        line_condition[0] = int(code[0] == ROOT_OK)

    @wp.kernel(module=module, enable_backward=False)
    def candidate(
        q: wp.array(dtype=vec),
        delta: wp.array(dtype=vec),
        counts: wp.array(dtype=int),
        out: wp.array(dtype=vec),
    ):
        i = wp.tid()
        scale = wp.pow(real(0.5), real(counts[COUNT_LINE_SEARCH_TRIALS]))
        out[i] = q[i] + scale * delta[i]

    @wp.kernel(module=module, enable_backward=False)
    def assess(
        norm: wp.array(dtype=real),
        bad: wp.array(dtype=int),
        cfg: Parameters,
        residual: wp.array(dtype=real),
        code: wp.array(dtype=int),
        counts: wp.array(dtype=int),
        line_condition: wp.array(dtype=int),
        accepted: wp.array(dtype=int),
    ):
        scale = wp.pow(real(0.5), real(counts[COUNT_LINE_SEARCH_TRIALS]))
        sufficient = norm[0] <= (real(1) - cfg.armijo * scale) * residual[0] or norm[0] <= cfg.atol
        success = bad[0] == 0 and wp.isfinite(norm[0]) and sufficient
        accepted[0] = int(success)
        counts[COUNT_LINE_SEARCH_TRIALS] = counts[COUNT_LINE_SEARCH_TRIALS] + 1
        if success:
            residual[0] = norm[0]
            counts[COUNT_BACKTRACKS] = (
                counts[COUNT_BACKTRACKS] + counts[COUNT_LINE_SEARCH_TRIALS] - 1
            )
            line_condition[0] = 0
        elif counts[COUNT_LINE_SEARCH_TRIALS] >= cfg.max_backtracks:
            counts[COUNT_BACKTRACKS] = (
                counts[COUNT_BACKTRACKS] + counts[COUNT_LINE_SEARCH_TRIALS] - 1
            )
            code[0] = ROOT_LINE_SEARCH_FAILED
            line_condition[0] = 0

    @wp.kernel(module=module, enable_backward=False)
    def accept(
        candidate_q: wp.array(dtype=vec),
        candidate_r: wp.array(dtype=vec),
        accepted: wp.array(dtype=int),
        q: wp.array(dtype=vec),
        r: wp.array(dtype=vec),
    ):
        i = wp.tid()
        if accepted[0] != 0:
            q[i] = candidate_q[i]
            r[i] = candidate_r[i]

    @wp.kernel(module=module, enable_backward=False)
    def finish_iteration(
        residual: wp.array(dtype=real),
        cfg: Parameters,
        code: wp.array(dtype=int),
        counts: wp.array(dtype=int),
        condition: wp.array(dtype=int),
    ):
        counts[COUNT_NEWTON_ITERATIONS] = counts[COUNT_NEWTON_ITERATIONS] + 1
        if (
            counts[COUNT_NEWTON_ITERATIONS] >= cfg.max_newton
            and residual[0] > cfg.atol
            and code[0] == ROOT_OK
        ):
            code[0] = ROOT_MAX_NEWTON
        condition[0] = int(code[0] == ROOT_OK and residual[0] > cfg.atol)

    @wp.kernel(module=module, enable_backward=False)
    def check_adjoint(
        sums: wp.array(dtype=real),
        cfg: Parameters,
        multiplier: wp.array(dtype=vec),
        code: wp.array(dtype=int),
    ):
        i = wp.tid()
        denominator = sums[1] if sums[1] > real(0) else real(1)
        error = wp.sqrt(sums[0] / denominator)
        if not wp.isfinite(error) or error > cfg.linear_limit or code[0] != ROOT_OK:
            multiplier[i] = vec(not_a_number)

    @wp.kernel(module=module, enable_backward=False)
    def correction_rhs(rhs: wp.array(dtype=vec), ax: wp.array(dtype=vec), out: wp.array(dtype=vec)):
        i = wp.tid()
        out[i] = rhs[i] - ax[i]

    @wp.kernel(module=module, enable_backward=False)
    def add_correction(
        x: wp.array(dtype=vec), correction: wp.array(dtype=vec), out: wp.array(dtype=vec)
    ):
        i = wp.tid()
        out[i] = x[i] + correction[i]

    @wp.kernel(module=module, enable_backward=False)
    def refinement_begin(
        sums: wp.array(dtype=real),
        target: real,
        old_error: wp.array(dtype=real),
        count: wp.array(dtype=int),
        condition: wp.array(dtype=int),
    ):
        denominator = sums[1] if sums[1] > real(0) else real(1)
        error = wp.sqrt(sums[0] / denominator)
        old_error[0] = error
        count[0] = 0
        condition[0] = int(wp.isfinite(error) and error > target)

    @wp.kernel(module=module, enable_backward=False)
    def refinement_assess(
        sums: wp.array(dtype=real),
        target: real,
        old_error: wp.array(dtype=real),
        count: wp.array(dtype=int),
        budget: int,
        accepted: wp.array(dtype=int),
        condition: wp.array(dtype=int),
    ):
        denominator = sums[1] if sums[1] > real(0) else real(1)
        error = wp.sqrt(sums[0] / denominator)
        improves = wp.isfinite(error) and error < old_error[0]
        accepted[0] = int(improves)
        if improves:
            old_error[0] = error
        count[0] = count[0] + 1
        condition[0] = int(
            count[0] < budget and wp.isfinite(old_error[0]) and old_error[0] > target
        )

    @wp.kernel(module=module, enable_backward=False)
    def refinement_update(
        candidate: wp.array(dtype=vec), accepted: wp.array(dtype=int), value: wp.array(dtype=vec)
    ):
        i = wp.tid()
        if accepted[0] != 0:
            value[i] = candidate[i]

    return SimpleNamespace(**locals())
