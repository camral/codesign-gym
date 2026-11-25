# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Warp 1.16 GMRES with reusable device-side convergence controls.

Adapted from Warp 1.16's ``warp._src.optim.linear.GMRES._run`` and
``_run_capturable_loop``. The upstream ``run=False`` constructor allocates the
Krylov workspace, but its device-side loop still allocates a two-integer array
on every solve. CUDA forbids that allocation inside a conditional graph body.

This adapter allocates those controls with the other solver workspace and
resets them on every call. Arithmetic, stopping rules, operand substitution,
and right/left preconditioning follow the installed Warp 1.16 implementation.
The host-checked path delegates to Warp unchanged. Private Warp kernels are
used here, so a Warp upgrade requires rechecking this adapter against upstream.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from warp.optim.linear import LinearOperator

import warp as wp
from warp._src.optim import linear as _linear


class CapturableGMRES(_linear.GMRES):
    """A reusable GMRES state that can run inside nested conditional graphs."""

    def _allocate(self) -> "None":
        """Reserve Krylov vectors, reductions and device loop controls before capture."""
        if self._device.is_capturing:
            raise RuntimeError("Construct the GMRES workspace before graph capture")
        super()._allocate()
        if self._check_every > 0:
            return

        self._loop_control = wp.empty(2, dtype=int, device=self._device)
        self._loop_iteration = self._loop_control[0:1]
        self._loop_condition = self._loop_control[1:2]
        kernel, tile_size = _linear._create_update_condition_kernel(self._batch_count)
        self._update_loop = wp.launch(
            kernel,
            dim=(1, tile_size),
            block_dim=tile_size,
            device=self._device,
            inputs=[
                self._maxiter,
                self._restart,
                self._loop_iteration,
                self._tiled_dot.col(0),
                self._atol_sq,
                self._loop_condition,
            ],
            record_cmd=True,
        )
        self._loop_callback = None
        if isinstance(self._callback, wp.Kernel):
            self._loop_callback = wp.launch(
                self._callback,
                dim=1,
                device=self._device,
                inputs=[self._loop_iteration, self._tiled_dot.col(0), self._atol_sq],
                record_cmd=True,
            )

    def _run(
        self, A: "LinearOperator", b: "wp.array", x: "wp.array", M: "LinearOperator | None"
    ) -> "tuple[wp.array, wp.array, wp.array]":
        """Solve Ax = b in place using retained Krylov workspace and device stopping rules."""
        if self._check_every > 0:
            return super()._run(A, b, x, M)

        device = self._device
        scalar_dtype = self._scalar_type
        restart = self._restart
        is_left_preconditioner = self._is_left_preconditioner
        batch_offsets = A.batch_offsets
        dofs_per_entry = self._dofs_per_entry

        tiled_dot = self._tiled_dot
        r = self._r
        w = self._w
        H = self._H
        y = self._y
        V = self._V
        beta = self._beta
        atol_sq = self._atol_sq
        w_repeated = self._w_repeated
        r_norm_sq = tiled_dot.col(0)

        least_squares_solve = self._least_squares_solve
        normalize_arnoldi_vec = self._normalize_arnoldi_vec
        arnoldi_axpy = self._arnoldi_axpy
        copy_hessenberg_col = self._copy_hessenberg_col

        _linear._initialize_absolute_tolerance(b, self._tol, self._atol, tiled_dot, atol_sq)
        if getattr(self, "zero_initial_guess", False):
            x.zero_()
            wp.copy(r, b)
        else:
            A.matvec(x, b, r, alpha=-1.0, beta=1.0)
        tiled_dot.compute(r, r, col_offset=0)
        w.zero_()

        def do_arnoldi_iteration(j: "int") -> "None":
            """Orthogonalize one restart column and update the residual estimate."""
            if M is not None:
                tmp = V[j + 1]
                if is_left_preconditioner:
                    A.matvec(V[j], tmp, tmp, alpha=1, beta=0)
                    M.matvec(tmp, w, w, alpha=1, beta=0)
                else:
                    M.matvec(V[j], tmp, tmp, alpha=1, beta=0)
                    A.matvec(tmp, w, w, alpha=1, beta=0)
            else:
                A.matvec(V[j], w, w, alpha=1, beta=0)

            tiled_dot.compute(w_repeated[: j + 1], V[: j + 1])
            copy_hessenberg_col.set_param_at_index(0, j)
            copy_hessenberg_col.launch()
            arnoldi_axpy.set_param_at_index(0, j)
            arnoldi_axpy.launch()
            tiled_dot.compute(w, w)
            normalize_arnoldi_vec.set_params(
                [w, V[j + 1], tiled_dot.col(0), H[:, j + 1, j], batch_offsets, dofs_per_entry]
            )
            normalize_arnoldi_vec.launch()

        def do_restart_cycle() -> "None":
            """Complete one GMRES restart and recompute its true residual."""
            if M is not None and is_left_preconditioner:
                M.matvec(r, w, w, alpha=1, beta=0)
                rh = w
            else:
                rh = r

            tiled_dot.compute(rh, rh)
            normalize_arnoldi_vec.set_params(
                [rh, V[0], tiled_dot.col(0), beta, batch_offsets, dofs_per_entry]
            )
            normalize_arnoldi_vec.launch()
            for j in range(restart):
                do_arnoldi_iteration(j)
            least_squares_solve.launch()

            if M is None or is_left_preconditioner:
                wp.launch(
                    _linear._gmres_update_x_kernel,
                    dim=x.shape,
                    device=device,
                    inputs=[restart, scalar_dtype(1.0), y, V, x, batch_offsets, dofs_per_entry],
                )
            else:
                wp.launch(
                    _linear._gmres_update_x_kernel,
                    dim=x.shape,
                    device=device,
                    inputs=[restart, scalar_dtype(0.0), y, V, w, batch_offsets, dofs_per_entry],
                )
                M.matvec(w, r, r, alpha=1, beta=0)
                wp.launch(
                    _linear._gmres_add_active_entries_kernel,
                    dim=x.shape,
                    device=device,
                    inputs=[r, x, batch_offsets, dofs_per_entry],
                )

            wp.copy(src=b, dest=r)
            A.matvec(x, b, r, alpha=-1.0, beta=1.0)
            tiled_dot.compute(r, r)

        # Match Warp 1.16's initialization and update order exactly. In
        # particular, the initial update adds restart to the initial -1.
        self._loop_control.fill_(-1)
        self._update_loop.launch()
        if self._loop_callback is not None:
            self._loop_callback.launch()

        def do_cycle_with_condition() -> "None":
            """Run a restart and update the resident loop/callback controls."""
            do_restart_cycle()
            self._update_loop.launch()
            if self._loop_callback is not None:
                self._loop_callback.launch()

        if self._use_cuda_graph and device.is_cuda:
            if device.is_capturing:
                wp.capture_while(self._loop_condition, do_cycle_with_condition)
            else:
                with wp.ScopedCapture(device=device) as capture:
                    wp.capture_while(self._loop_condition, do_cycle_with_condition)
                wp.capture_launch(capture.graph)
        else:
            # CPU is the numerical-reference path, but must honor the same
            # device-resident condition. A fixed Python restart-budget loop
            # unnecessarily kept applying A after the solve had converged.
            wp.capture_while(self._loop_condition, do_cycle_with_condition)

        return self._loop_iteration, r_norm_sq, atol_sq


def gmres(
    A: "LinearOperator",
    b: "wp.array",
    x: "wp.array",
    tol: "float | None" = None,
    atol: "float | None" = None,
    restart: "int" = 31,
    maxiter: "int" = 0,
    M: "LinearOperator | None" = None,
    callback: "Callable[..., None] | None" = None,
    check_every: "int" = 31,
    use_cuda_graph: "bool" = True,
    is_left_preconditioner: "bool" = False,
    run: "bool" = True,
) -> "CapturableGMRES | tuple[int, float, float]":
    """Warp-compatible GMRES factory with preallocated capture loop controls."""
    state = CapturableGMRES(
        A,
        b,
        x,
        tol=tol,
        atol=atol,
        restart=restart,
        maxiter=maxiter,
        M=M,
        callback=callback,
        check_every=check_every,
        use_cuda_graph=use_cuda_graph,
        is_left_preconditioner=is_left_preconditioner,
    )
    return state() if run else state
