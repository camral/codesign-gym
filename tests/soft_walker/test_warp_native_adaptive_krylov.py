"""Adaptive GMRES arithmetic, operand reuse, and nested CUDA graph regressions.

CPU is the default: four precision/storage cases validate the fallback and the
12 CUDA-specific cases skip. WARP_TEST_DEVICE=cuda enables all capture cases.
WARP_TEST_DTYPE may select float32 or float64; otherwise both run. Device work
counters establish the actual 1-versus-40 Arnoldi reduction on an exact fixture.
All imports come from the installed package; this test does not need runs/.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable


import json
import os
from functools import lru_cache
from types import SimpleNamespace

import numpy as np
import pytest

wp = pytest.importorskip("warp")
from warp.optim.linear import LinearOperator

from envs.soft_walker.engine.warp_native.adaptive_krylov import (
    AdaptiveCapturableGMRES,
    _check_estimate,
)
from envs.soft_walker.engine.warp_native.krylov import CapturableGMRES

PRECISIONS = (
    [os.environ["WARP_TEST_DTYPE"]] if "WARP_TEST_DTYPE" in os.environ else ["float64", "float32"]
)
if any(p not in ("float64", "float32") for p in PRECISIONS):
    raise ValueError("WARP_TEST_DTYPE must be float64 or float32")


@wp.kernel(enable_backward=False)
def advance(count: wp.array(dtype=int), condition: wp.array(dtype=int)):
    count[0] += 1
    condition[0] = wp.where(count[0] < 2, 1, 0)


@lru_cache(maxsize=4)
def kernels(precision: "str", components: "int") -> "SimpleNamespace":
    real = wp.float32 if precision == "float32" else wp.float64
    entry = real if components == 1 else wp.types.vector(length=2, dtype=real)
    module = f"codesign_tests.adaptive_gmres_{precision}_{components}"

    if components == 1:

        @wp.kernel(module=module, enable_backward=False)
        def multiply(
            matrix: wp.array2d(dtype=real),
            transpose: int,
            x: wp.array(dtype=entry),
            y: wp.array(dtype=entry),
            alpha: real,
            beta: real,
            z: wp.array(dtype=entry),
            calls: wp.array(dtype=int),
        ):
            i = wp.tid()
            if i == 0:
                calls[0] += 1
            value = real(0.0)
            for j in range(matrix.shape[1]):
                if transpose:
                    value += matrix[j, i] * x[j]
                else:
                    value += matrix[i, j] * x[j]
            result = alpha * value
            if beta != real(0.0):
                result += beta * y[i]
            z[i] = result

        @wp.kernel(module=module, enable_backward=False)
        def jacobi(
            matrix: wp.array2d(dtype=real),
            x: wp.array(dtype=entry),
            y: wp.array(dtype=entry),
            alpha: real,
            beta: real,
            z: wp.array(dtype=entry),
        ):
            i = wp.tid()
            value = alpha * x[i] / matrix[i, i]
            if beta != real(0.0):
                value += beta * y[i]
            z[i] = value
    else:

        @wp.kernel(module=module, enable_backward=False)
        def multiply(
            matrix: wp.array2d(dtype=real),
            transpose: int,
            x: wp.array(dtype=entry),
            y: wp.array(dtype=entry),
            alpha: real,
            beta: real,
            z: wp.array(dtype=entry),
            calls: wp.array(dtype=int),
        ):
            i = wp.tid()
            if i == 0:
                calls[0] += 1
            result = entry(real(0.0))
            for component in range(2):
                row = 2 * i + component
                value = real(0.0)
                for j in range(matrix.shape[1]):
                    if transpose:
                        value += matrix[j, row] * x[j // 2][j % 2]
                    else:
                        value += matrix[row, j] * x[j // 2][j % 2]
                result[component] = alpha * value
                if beta != real(0.0):
                    result[component] += beta * y[i][component]
            z[i] = result

        @wp.kernel(module=module, enable_backward=False)
        def jacobi(
            matrix: wp.array2d(dtype=real),
            x: wp.array(dtype=entry),
            y: wp.array(dtype=entry),
            alpha: real,
            beta: real,
            z: wp.array(dtype=entry),
        ):
            i = wp.tid()
            result = entry(real(0.0))
            for component in range(2):
                row = 2 * i + component
                result[component] = alpha * x[i][component] / matrix[row, row]
                if beta != real(0.0):
                    result[component] += beta * y[i][component]
            z[i] = result

    @wp.kernel(module=module, enable_backward=False)
    def save_residual(
        residual: wp.array(dtype=real),
        threshold: wp.array(dtype=real),
        slot: int,
        output: wp.array(dtype=real),
    ):
        output[2 * slot] = residual[0]
        output[2 * slot + 1] = threshold[0]

    return SimpleNamespace(
        real=real, entry=entry, multiply=multiply, jacobi=jacobi, save_residual=save_residual
    )


@pytest.fixture(scope="module")
def device() -> "wp.context.Device":
    wp.init()
    selected = wp.get_device(os.environ.get("WARP_TEST_DEVICE", "cpu"))
    if not selected.is_cuda:
        pytest.skip("Adaptive conditional capture requires WARP_TEST_DEVICE=cuda")
    assert wp.is_conditional_graph_supported(), "CUDA conditional graph support is required"
    return selected


@pytest.fixture(
    params=[(p, n) for p in PRECISIONS for n in (1, 2)],
    ids=[f"{p}-{'scalar' if n == 1 else 'vec2'}" for p in PRECISIONS for n in (1, 2)],
)
def case(request: "pytest.FixtureRequest", device: "wp.context.Device") -> "SimpleNamespace":
    """Pair a precision-specific kernel family with its existing residual tolerance."""
    precision, components = request.param
    return SimpleNamespace(
        precision=precision,
        components=components,
        device=device,
        k=kernels(precision, components),
        tol=2.0e-6 if precision == "float32" else 1.0e-12,
    )


class Problem:
    """Dense reference operator with an independent count of device executions."""

    def __init__(self, case: "SimpleNamespace", matrix: "np.ndarray") -> "None":
        self.case = case
        self.nscalar = len(matrix)
        assert self.nscalar % case.components == 0
        self.n = self.nscalar // case.components
        self.matrix = wp.array(
            np.asarray(matrix, dtype=case.precision), dtype=case.k.real, device=case.device
        )
        self.calls = wp.zeros(1, dtype=int, device=case.device)
        self.direct = self.operator(0)
        self.transpose = self.operator(1)
        self.preconditioner = LinearOperator(
            (self.n, self.n), case.k.entry, case.device, self.precondition
        )

    def array(self, values: "ArrayLike | None" = None) -> "wp.array":
        if values is None:
            return wp.zeros(self.n, dtype=self.case.k.entry, device=self.case.device)
        return wp.array(self.shaped(values), dtype=self.case.k.entry, device=self.case.device)

    def shaped(self, values: "ArrayLike") -> "np.ndarray":
        values = np.asarray(values, dtype=self.case.precision)
        return values.reshape(self.n) if self.case.components == 1 else values.reshape(self.n, 2)

    def operator(self, transpose: "bool | int") -> "LinearOperator":
        case = self.case

        def matvec(
            x: "wp.array", y: "wp.array", z: "wp.array", alpha: "float" = 1.0, beta: "float" = 0.0
        ) -> "None":
            wp.launch(
                case.k.multiply,
                dim=self.n,
                device=case.device,
                inputs=[
                    self.matrix,
                    transpose,
                    x,
                    y,
                    case.k.real(alpha),
                    case.k.real(beta),
                    z,
                    self.calls,
                ],
            )

        return LinearOperator((self.n, self.n), case.k.entry, case.device, matvec)

    def precondition(
        self, x: "wp.array", y: "wp.array", z: "wp.array", alpha: "float" = 1.0, beta: "float" = 0.0
    ) -> "None":
        case = self.case
        wp.launch(
            case.k.jacobi,
            dim=self.n,
            device=case.device,
            inputs=[self.matrix, x, y, case.k.real(alpha), case.k.real(beta), z],
        )

    def solver(
        self,
        b: "wp.array",
        x: "wp.array",
        *,
        right: "bool" = False,
        restart: "int" = 8,
        maxiter: "int" = 48,
        baseline: "bool" = False,
    ) -> "CapturableGMRES | AdaptiveCapturableGMRES":
        cls = CapturableGMRES if baseline else AdaptiveCapturableGMRES
        return cls(
            self.direct,
            b,
            x,
            M=self.preconditioner if right else None,
            tol=self.case.tol,
            atol=0.0,
            restart=restart,
            maxiter=maxiter,
            check_every=0,
            use_cuda_graph=True,
            is_left_preconditioner=False,
        )


def assert_solution(
    case: "SimpleNamespace",
    matrix: "np.ndarray",
    rhs: "np.ndarray",
    x: "wp.array",
    residual_sq: "float",
    threshold_sq: "float",
) -> "float":
    """Check the reported residual against an independent dense solve and return its norm."""
    actual = x.numpy().reshape(-1).astype(np.float64)
    matrix = np.asarray(matrix, dtype=case.precision).astype(np.float64)
    rhs = np.asarray(rhs, dtype=case.precision).astype(np.float64)
    assert np.all(np.isfinite(actual))
    assert np.isfinite(residual_sq) and np.isfinite(threshold_sq)
    assert 0.0 <= residual_sq <= threshold_sq
    # Independently evaluate b-A*x; allow only a floating-point matvec error
    # bound in addition to the solver's reported tolerance.
    rounding = (
        8
        * np.finfo(case.precision).eps
        * (np.linalg.norm(matrix, np.inf) * np.linalg.norm(actual) + np.linalg.norm(rhs))
    )
    actual_residual = np.linalg.norm(rhs - matrix @ actual)
    assert actual_residual <= np.sqrt(threshold_sq) + rounding
    np.testing.assert_allclose(
        actual, np.linalg.solve(matrix, rhs), rtol=8 * case.tol, atol=8 * case.tol
    )
    return float(actual_residual)


def workspace_pointers(solver: "AdaptiveCapturableGMRES") -> "tuple[int, ...]":
    """Record allocation identities to detect unexpected workspace replacement."""
    return tuple(
        getattr(solver, name).ptr
        for name in (
            "_H",
            "_V",
            "_loop_control",
            "_inner_condition",
            "_qr_remaining",
            "_force_full",
        )
    )


def test_one_column_replaces_forty_actual_columns(
    case: "SimpleNamespace", record_property: "Callable[[str, object], None]"
) -> None:
    """Binary-exact fixtures prove saved work, not just an unchanged solution."""
    measured = []
    for right in (False, True):
        diagonal = np.tile([1.0, 2.0, 4.0, 8.0], 4) if right else np.full(16, 2.0)
        matrix = np.diag(diagonal).astype(case.precision)
        problem = Problem(case, matrix)
        rhs = np.ones(16, dtype=case.precision)  # norm=4 exactly in both precisions
        b, x = problem.array(rhs), problem.array()
        outputs = []
        for baseline in (True, False):
            solver = problem.solver(b, x, right=right, restart=40, maxiter=80, baseline=baseline)
            with wp.ScopedCapture(device=case.device, force_module_load=True) as capture:
                x.zero_()
                result = solver()
            problem.calls.zero_()
            if not baseline:
                solver.reset_adaptive_counters()
            wp.capture_launch(capture.graph)
            wp.synchronize_device(case.device)
            residual_sq = float(result[1].numpy()[0])
            threshold_sq = float(result[2].numpy()[0])
            residual = assert_solution(case, matrix, rhs, x, residual_sq, threshold_sq)
            calls = int(problem.calls.numpy()[0])
            if baseline:
                # One initial residual + 40 Arnoldi matvecs + one true residual.
                assert calls == 42
            else:
                np.testing.assert_array_equal(solver.adaptive_counters.numpy(), [1, 1, 1, 0])
                assert calls == 3
                np.testing.assert_array_equal(solver._V.numpy()[2:], 0)
            outputs.append(x.numpy().copy())
            measured.append(
                dict(
                    right=right,
                    baseline=baseline,
                    operator_calls=calls,
                    arnoldi_columns=calls - 2,
                    true_residual=residual,
                )
            )
        np.testing.assert_array_equal(outputs[0], outputs[1])
    record_property("measured_work", json.dumps(measured))


def test_nested_capture_transpose_replacement_and_replay(
    case: "SimpleNamespace", record_property: "Callable[[str, object], None]"
) -> None:
    matrix = np.array(
        [
            [4.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 3.0, -1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 2.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 5.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 0.5, 4.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, -1.0, 3.0],
        ],
        dtype=case.precision,
    )
    first = np.array([1.0, 2.0, -1.0, -2.0, 0.5, 3.0], dtype=case.precision)
    second = np.array([3.0, -1.0, 0.25, 0.5, 2.0, -4.0], dtype=case.precision)
    replays = [
        (matrix, first, second),
        (2 * np.eye(6, dtype=case.precision), np.ones(6), -np.ones(6)),
        (matrix, np.zeros(6), np.zeros(6)),
        (matrix + 0.25 * np.eye(6), second, first),
    ]
    measured = []
    for right in (False, True):
        problem = Problem(case, matrix)
        b, bt, x, xt = (problem.array() for _ in range(4))
        solver = problem.solver(b, x, right=right)
        pointers = workspace_pointers(solver)
        outer = wp.zeros(1, dtype=int, device=case.device)
        enabled = wp.ones(1, dtype=int, device=case.device)
        count = wp.zeros(1, dtype=int, device=case.device)
        stats = wp.zeros(4, dtype=case.k.real, device=case.device)

        def save(result: "CapturableGMRES | AdaptiveCapturableGMRES", slot: "int") -> "None":
            wp.launch(
                case.k.save_residual,
                dim=1,
                device=case.device,
                inputs=[result[1], result[2], slot, stats],
            )

        def solve_pair() -> "None":
            x.zero_()
            xt.zero_()
            save(solver(), 0)
            save(solver(A=problem.transpose, b=bt, x=xt), 1)

        def body() -> "None":
            wp.capture_if(enabled, solve_pair)
            wp.launch(advance, dim=1, device=case.device, inputs=[count, outer])

        with wp.ScopedCapture(device=case.device, force_module_load=True) as capture:
            count.zero_()
            outer.fill_(1)
            wp.capture_while(outer, body)

        for replay, (current_matrix, rhs, rhs_t) in enumerate(replays):
            problem.matrix.assign(np.asarray(current_matrix, dtype=case.precision))
            b.assign(problem.shaped(rhs))
            bt.assign(problem.shaped(rhs_t))
            solver.reset_adaptive_counters()
            problem.calls.zero_()
            wp.capture_launch(capture.graph)
            wp.synchronize_device(case.device)
            values = stats.numpy().astype(np.float64)
            assert_solution(case, current_matrix, rhs, x, *values[:2])
            assert_solution(case, current_matrix.T, rhs_t, xt, *values[2:])
            counters = solver.adaptive_counters.numpy()
            assert int(count.numpy()[0]) == 2
            assert workspace_pointers(solver) == pointers
            # Four solves execute: two outer iterations, each forward+transpose.
            assert int(problem.calls.numpy()[0]) == int(counters[0] + counters[1] + 4)
            if not np.any(rhs) and not np.any(rhs_t):
                np.testing.assert_array_equal(counters, 0)
                np.testing.assert_array_equal(x.numpy(), 0)
                np.testing.assert_array_equal(xt.numpy(), 0)
            else:
                assert 0 < counters[0] <= solver._restart * counters[1]
                assert counters[1] >= 4
            measured.append(dict(right=right, replay=replay, counters=counters.tolist()))

        # Exercise the false branch of the enclosing if using the same graph.
        enabled.zero_()
        x.assign(problem.shaped(np.full(6, 7.0)))
        xt.assign(problem.shaped(np.full(6, -3.0)))
        solver.reset_adaptive_counters()
        problem.calls.zero_()
        wp.capture_launch(capture.graph)
        wp.synchronize_device(case.device)
        np.testing.assert_array_equal(x.numpy(), 7.0)
        np.testing.assert_array_equal(xt.numpy(), -3.0)
        np.testing.assert_array_equal(solver.adaptive_counters.numpy(), 0)
        assert int(problem.calls.numpy()[0]) == 0
        assert int(count.numpy()[0]) == 2
    record_property("replay_counters", json.dumps(measured))


def test_zero_singular_and_nonfinite_are_not_false_success(
    case: "SimpleNamespace", record_property: "Callable[[str, object], None]"
) -> None:
    n = 4
    identity = np.eye(n, dtype=case.precision)
    problem = Problem(case, identity)
    b, x = problem.array(), problem.array()
    solver = problem.solver(b, x, restart=8, maxiter=24)
    pointers = workspace_pointers(solver)
    with wp.ScopedCapture(device=case.device, force_module_load=True) as capture:
        x.zero_()
        result = solver()

    nan_matrix, inf_matrix = identity.copy(), identity.copy()
    nan_matrix[0, 0] = np.nan
    inf_matrix[0, 0] = np.inf
    scenarios = [
        ("valid_before", identity, np.ones(n)),
        ("zero_rhs", identity, np.zeros(n)),
        ("singular", np.zeros((n, n)), np.ones(n)),
        ("nan_operator", nan_matrix, np.ones(n)),
        ("inf_operator", inf_matrix, np.ones(n)),
        ("nan_rhs", identity, np.array([np.nan, 1.0, 1.0, 1.0])),
        ("valid_after", identity, np.array([1.0, -1.0, 1.0, -1.0])),
    ]
    measured = []
    for label, matrix, rhs in scenarios:
        problem.matrix.assign(np.asarray(matrix, dtype=case.precision))
        b.assign(problem.shaped(rhs))
        solver.reset_adaptive_counters()
        problem.calls.zero_()
        wp.capture_launch(capture.graph)
        wp.synchronize_device(case.device)
        residual_sq = float(result[1].numpy()[0])
        threshold_sq = float(result[2].numpy()[0])
        counters = solver.adaptive_counters.numpy()
        accepted = (
            np.isfinite(residual_sq)
            and np.isfinite(threshold_sq)
            and residual_sq >= 0
            and residual_sq <= threshold_sq
            and np.all(np.isfinite(x.numpy()))
        )
        if label in ("valid_before", "valid_after", "zero_rhs"):
            assert accepted, label
            assert_solution(case, matrix, rhs, x, residual_sq, threshold_sq)
            if label == "zero_rhs":
                np.testing.assert_array_equal(counters, 0)
                np.testing.assert_array_equal(x.numpy(), 0)
            else:
                np.testing.assert_array_equal(counters, [1, 1, 1, 0])
        else:
            assert not accepted, label
            if label == "singular":
                assert np.isfinite(residual_sq) and residual_sq >= np.linalg.norm(rhs) ** 2 * 0.99
                assert np.linalg.norm(rhs - matrix @ x.numpy().reshape(-1)) > 1.0
                assert counters[0] == counters[1] > 0
                assert counters[2] == 0
            else:
                assert not np.isfinite(residual_sq), label
        assert workspace_pointers(solver) == pointers
        measured.append(dict(case=label, accepted=bool(accepted), counters=counters.tolist()))

    # A well-conditioned solve need not encounter an optimistic QR estimate.
    # Exercise its rejection explicitly, then prove the next captured solve
    # clears that persistent per-solve flag rather than inheriting full restart.
    probe_residual = wp.array([4.0], dtype=case.k.real, device=case.device)
    probe_threshold = wp.ones(1, dtype=case.k.real, device=case.device)
    with wp.ScopedCapture(device=case.device, force_module_load=True) as rejection_capture:
        wp.launch(
            _check_estimate,
            dim=1,
            device=case.device,
            inputs=[
                probe_residual,
                probe_threshold,
                solver._estimated_convergence,
                solver._force_full,
                solver.adaptive_counters,
            ],
        )
    for value in (4.0, np.nan):
        probe_residual.assign(np.asarray([value], dtype=case.precision))
        solver._estimated_convergence.fill_(1)
        solver._force_full.zero_()
        solver.reset_adaptive_counters()
        wp.capture_launch(rejection_capture.graph)
        wp.synchronize_device(case.device)
        assert int(solver._force_full.numpy()[0]) == 1
        np.testing.assert_array_equal(solver.adaptive_counters.numpy(), [0, 0, 0, 1])
        solver.reset_adaptive_counters()
        wp.capture_launch(capture.graph)
        wp.synchronize_device(case.device)
        assert int(solver._force_full.numpy()[0]) == 0
        np.testing.assert_array_equal(solver.adaptive_counters.numpy(), [1, 1, 1, 0])
        assert_solution(
            case,
            identity,
            scenarios[-1][2],
            x,
            float(result[1].numpy()[0]),
            float(result[2].numpy()[0]),
        )
    record_property("failure_replays", json.dumps(measured))


@pytest.mark.parametrize(
    "precision,components",
    [(p, n) for p in PRECISIONS for n in (1, 2)],
    ids=[f"{p}-{'scalar' if n == 1 else 'vec2'}" for p in PRECISIONS for n in (1, 2)],
)
def test_cpu_fallback_true_residual_and_operand_reuse(precision: "str", components: "int") -> None:
    """The candidate's documented CPU fallback retains the original equations."""
    wp.init()
    case = SimpleNamespace(
        precision=precision,
        components=components,
        device=wp.get_device("cpu"),
        k=kernels(precision, components),
        tol=2.0e-6 if precision == "float32" else 1.0e-12,
    )
    matrix = np.array(
        [[4.0, 1.0, 0.0, 0.0], [0.0, 3.0, -1.0, 0.0], [1.0, 0.0, 2.0, 0.0], [0.0, 1.0, 0.0, 8.0]],
        dtype=precision,
    )
    problem = Problem(case, matrix)
    b, bt, x, xt = (problem.array() for _ in range(4))
    solver = AdaptiveCapturableGMRES(
        problem.direct,
        b,
        x,
        M=problem.preconditioner,
        tol=case.tol,
        atol=0.0,
        restart=4,
        maxiter=32,
        check_every=1,
        use_cuda_graph=False,
        is_left_preconditioner=False,
    )
    pointers = (solver._H.ptr, solver._V.ptr)
    for rhs, rhs_t in (
        ([1.0, 2.0, -1.0, 3.0], [-2.0, 0.5, 3.0, 1.0]),
        ([0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]),
        ([3.0, -1.0, 0.25, 2.0], [0.5, 2.0, -4.0, -1.0]),
    ):
        b.assign(problem.shaped(rhs))
        bt.assign(problem.shaped(rhs_t))
        x.zero_()
        xt.zero_()
        _, residual, threshold = solver()
        assert_solution(case, matrix, rhs, x, float(residual) ** 2, float(threshold) ** 2)
        _, residual, threshold = solver(A=problem.transpose, b=bt, x=xt)
        assert_solution(case, matrix.T, rhs_t, xt, float(residual) ** 2, float(threshold) ** 2)
        assert (solver._H.ptr, solver._V.ptr) == pointers
        np.testing.assert_array_equal(solver.adaptive_counters.numpy(), 0)
