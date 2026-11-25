"""GMRES operand reuse, including calls inside a CUDA conditional graph.

Use WARP_TEST_DEVICE=cuda and WARP_TEST_DTYPE=float32/float64 to exercise the
capture regression. The default CPU run exercises the same device-loop
adapter arithmetic, without claiming CUDA graph coverage.
"""

import os

import numpy as np
import pytest

wp = pytest.importorskip("warp")

from warp.optim.linear import LinearOperator

from envs.soft_walker.engine.warp_native.krylov import gmres

DTYPE = os.environ.get("WARP_TEST_DTYPE", "float64")
REAL = wp.float32 if DTYPE == "float32" else wp.float64
DEVICE = os.environ.get("WARP_TEST_DEVICE", "cpu")


@wp.kernel(enable_backward=False)
def multiply(
    matrix: wp.array2d(dtype=REAL),
    transpose: int,
    x: wp.array(dtype=REAL),
    y: wp.array(dtype=REAL),
    alpha: REAL,
    beta: REAL,
    z: wp.array(dtype=REAL),
):
    i = wp.tid()
    value = REAL(0.0)
    for j in range(matrix.shape[1]):
        if transpose:
            value += matrix[j, i] * x[j]
        else:
            value += matrix[i, j] * x[j]
    z[i] = alpha * value + beta * y[i]


@wp.kernel(enable_backward=False)
def precondition(
    matrix: wp.array2d(dtype=REAL),
    x: wp.array(dtype=REAL),
    y: wp.array(dtype=REAL),
    alpha: REAL,
    beta: REAL,
    z: wp.array(dtype=REAL),
):
    i = wp.tid()
    z[i] = alpha * x[i] / matrix[i, i] + beta * y[i]


@wp.kernel(enable_backward=False)
def advance(count: wp.array(dtype=int), condition: wp.array(dtype=int)):
    count[0] += 1
    condition[0] = wp.where(count[0] < 2, 1, 0)


def test_reusable_gmres_forward_transpose_inside_conditional() -> None:
    wp.init()
    device = wp.get_device(DEVICE)
    if device.is_cuda and not wp.is_conditional_graph_supported():
        pytest.fail("This regression requires CUDA conditional graph support")

    matrix_host = np.array([[4.0, 1.0, 0.0], [0.0, 3.0, -1.0], [1.0, 0.0, 2.0]], dtype=DTYPE)
    matrix = wp.array(matrix_host, dtype=REAL, device=device)
    n = len(matrix_host)

    def operator(transpose: "bool | int") -> "LinearOperator":
        def matvec(
            x: "wp.array", y: "wp.array", z: "wp.array", alpha: "float" = 1.0, beta: "float" = 0.0
        ) -> "None":
            wp.launch(
                multiply,
                dim=n,
                inputs=[matrix, transpose, x, y, REAL(alpha), REAL(beta), z],
                device=device,
            )

        return LinearOperator((n, n), REAL, device, matvec)

    def apply_preconditioner(
        x: "wp.array", y: "wp.array", z: "wp.array", alpha: "float" = 1.0, beta: "float" = 0.0
    ) -> "None":
        wp.launch(
            precondition,
            dim=n,
            inputs=[matrix, x, y, REAL(alpha), REAL(beta), z],
            device=device,
        )

    direct = operator(0)
    transpose = operator(1)
    M = LinearOperator((n, n), REAL, device, apply_preconditioner)
    b = wp.zeros(n, dtype=REAL, device=device)
    bt = wp.zeros_like(b)
    x = wp.zeros_like(b)
    xt = wp.zeros_like(b)
    tol = 2e-6 if DTYPE == "float32" else 1e-12
    solver = gmres(
        direct,
        b,
        x,
        M=M,
        tol=tol,
        atol=0.0,
        restart=2,
        maxiter=24,
        check_every=0,
        use_cuda_graph=device.is_cuda,
        is_left_preconditioner=False,
        run=False,
    )
    control_ptr = solver._loop_control.ptr
    outer_condition = wp.zeros(1, dtype=int, device=device)
    count = wp.zeros(1, dtype=int, device=device)

    def body() -> "None":
        x.zero_()
        xt.zero_()
        solver()
        solver(A=transpose, b=bt, x=xt)
        wp.launch(advance, dim=1, inputs=[count, outer_condition], device=device)

    def program() -> "None":
        count.zero_()
        outer_condition.fill_(1)
        wp.capture_while(outer_condition, body)

    if device.is_cuda:
        with wp.ScopedCapture(device=device, force_module_load=True) as capture:
            program()

    rhs_cases = (
        ([1.0, 2.0, -1.0], [-2.0, 0.5, 3.0]),
        ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
        ([3.0, -1.0, 0.25], [0.5, 2.0, -4.0]),
    )
    for rhs, rhs_transpose in rhs_cases:
        rhs = np.asarray(rhs, dtype=DTYPE)
        rhs_transpose = np.asarray(rhs_transpose, dtype=DTYPE)
        b.assign(rhs)
        bt.assign(rhs_transpose)
        if device.is_cuda:
            wp.capture_launch(capture.graph)
        else:
            program()
        wp.synchronize_device(device)
        error = 2e-5 if DTYPE == "float32" else 2e-10
        np.testing.assert_allclose(
            x.numpy(), np.linalg.solve(matrix_host, rhs), rtol=error, atol=error
        )
        np.testing.assert_allclose(
            xt.numpy(), np.linalg.solve(matrix_host.T, rhs_transpose), rtol=error, atol=error
        )
        assert int(count.numpy()[0]) == 2
        assert solver._loop_control.ptr == control_ptr
