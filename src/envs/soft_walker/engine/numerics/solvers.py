"""Matrix-free safeguarded Newton, with implicit JVP and transpose solves.

No jacfwd of an N-vector, global dense Jacobian, or dense fallback. Restarted
GMRES stores O(N * restart) Krylov data. A small restart-space least-squares
solve is allowed; it is NOT a global N x N tangent. The independent true linear
residual is checked (the library's info flag alone is insufficient).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import gmres
from numpy.typing import DTypeLike

from ...contracts.status_codes import (
    ROOT_INVALID_INITIAL,
    ROOT_LINE_SEARCH_FAILED,
    ROOT_LINEAR_FAILED,
    ROOT_MAX_NEWTON,
    ROOT_OK,
)
from .krylov import fused_gmres, fused_gmres_with_residual
from .policy import (
    LINEAR_ACCEPTANCE_EPS_FACTOR,
    LINEAR_ACCEPTANCE_RTOL_FACTOR,
    PRECONDITIONER_DIAGONAL_FLOOR,
)


@dataclass(frozen=True)
class NewtonConfig:
    """Immutable nonlinear and matrix-free Krylov acceptance settings."""

    atol64: float = 2e-11
    atol32: float = 3e-06
    max_steps: int = 16
    backtracks: int = 10
    restart: int = 20
    max_restarts: int = 12
    linear_rtol64: float = 2e-10
    linear_rtol32: float = 2e-05
    armijo: float = 0.0001
    debug_linear: bool = False
    # Numerical choice, not an environment-batch size. Keep the previously
    # validated QR/Givens path as the default until device evidence warrants a
    # change. All choices pass the same true-residual acceptance test. "fused"
    # removes nested Arnoldi control flow without hardware-dependent dispatch.
    gmres_method: str = "incremental"
    preconditioner_side: str = "left"
    linear_refinements: int = 0
    linear_work_reuse: bool = False

    def __post_init__(self) -> None:
        """Reject incompatible Newton, line-search, and Krylov settings."""
        if type(self.linear_refinements) is not int or not 0 <= self.linear_refinements <= 3:
            raise ValueError("linear_refinements must be 0..3")
        if self.preconditioner_side not in ("left", "right"):
            raise ValueError("preconditioner_side must be left or right")
        if self.gmres_method not in ("incremental", "batched", "fused"):
            raise ValueError("gmres_method must be incremental, batched, or fused")
        if type(self.linear_work_reuse) is not bool:
            raise ValueError("linear_work_reuse must be boolean")
        if self.linear_work_reuse and self.gmres_method != "fused":
            raise ValueError("linear_work_reuse requires gmres_method='fused'")
        if not all(
            math.isfinite(x)
            for x in (
                self.atol64,
                self.atol32,
                self.linear_rtol64,
                self.linear_rtol32,
                self.armijo,
            )
        ):
            raise ValueError("Solver tolerances and Armijo parameter must be finite")
        if min(self.atol64, self.atol32, self.linear_rtol64, self.linear_rtol32) <= 0:
            raise ValueError("Tolerances must be positive")
        capacities = (self.max_steps, self.backtracks, self.restart, self.max_restarts)
        if any(
            isinstance(value, bool) or not isinstance(value, Integral) or value < 1
            for value in capacities
        ):
            raise ValueError("Iteration capacities must be positive integers")
        if not 0 < self.armijo < 1:
            raise ValueError("armijo must lie in (0,1)")


class RootInfo(NamedTuple):
    """Device scalar root diagnostics; acceptance uses independently checked residuals."""

    residual: jax.Array
    iterations: jax.Array
    accepted: jax.Array
    backtracks: jax.Array
    linear_residual: jax.Array
    linear_solves: jax.Array
    code: jax.Array


# All aux leaves are floating: lax.custom_root uses zero tangents for aux.
CODES = {
    ROOT_OK: "OK",
    ROOT_MAX_NEWTON: "MAX_NEWTON",
    ROOT_LINE_SEARCH_FAILED: "LINE_SEARCH_FAILED",
    ROOT_LINEAR_FAILED: "LINEAR_FAILED",
    ROOT_INVALID_INITIAL: "INVALID_INITIAL",
}


def tolerance(dtype: DTypeLike, cfg: NewtonConfig) -> float:
    """Select the configured nonlinear tolerance for the state precision."""
    return cfg.atol64 if dtype == jnp.float64 else cfg.atol32


def _raw_linear(
    matvec: Callable[[jax.Array], jax.Array],
    b: jax.Array,
    diag: jax.Array,
    cfg: NewtonConfig,
    *,
    inverse_blocks: jax.Array | None = None,
    transpose_blocks: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Run the selected GMRES and independently audit its true residual."""
    rt = cfg.linear_rtol64 if b.dtype == jnp.float64 else cfg.linear_rtol32
    n = b.size
    restart = min(cfg.restart, n)
    diag = jax.lax.stop_gradient(
        jnp.where(jnp.isfinite(diag) & (jnp.abs(diag) > PRECONDITIONER_DIAGONAL_FLOOR), diag, 1.0)
    )
    if inverse_blocks is not None:
        inverse_blocks = jax.lax.stop_gradient(inverse_blocks)
        if transpose_blocks:
            inverse_blocks = jnp.swapaxes(inverse_blocks, -1, -2)

    def precondition(vector: jax.Array) -> jax.Array:
        """Apply the supplied inverse local blocks, or the existing scalar diagonal."""
        if inverse_blocks is None:
            return vector / diag
        shaped = vector.reshape(inverse_blocks.shape[0], inverse_blocks.shape[1])
        return jnp.einsum("nij,nj->ni", inverse_blocks, shaped).reshape(vector.shape)

    # M is a local left preconditioner. Incremental GMRES uses QR/Givens;
    # JAX's batched alternative reduces per-iteration GPU overhead but changes
    # the small restart-space least-squares solve. It is opt-in, not a tolerance
    # relaxation. Restart-space arrays remain bounded independently of N.
    # Right preconditioning preserves the physical residual norm used by the
    # acceptance check. The left-preconditioned iteration can stop on a small
    # scaled residual while tiny-mass MPM rows still fail that check.
    right = cfg.preconditioner_side == "right"
    operator = (lambda y: matvec(precondition(y))) if right else matvec

    def krylov(rhs: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array | None]:
        """Select the configured fused or JAX Krylov implementation for one right-hand side."""
        if cfg.gmres_method == "fused":
            if cfg.linear_work_reuse:
                value, detail, physical_residual = fused_gmres_with_residual(
                    operator,
                    rhs,
                    rtol=rt,
                    restart=restart,
                    max_restarts=cfg.max_restarts,
                    precondition=None if right else precondition,
                )
                return value, jnp.where(detail.finite, 0, -1), physical_residual
            value, detail = fused_gmres(
                operator,
                rhs,
                rtol=rt,
                restart=restart,
                max_restarts=cfg.max_restarts,
                precondition=None if right else precondition,
            )
            return value, jnp.where(detail.finite, 0, -1), None
        value, detail = gmres(
            operator,
            rhs,
            tol=rt,
            atol=0.0,
            restart=restart,
            maxiter=cfg.max_restarts,
            M=None if right else precondition,
            solve_method=cfg.gmres_method,
        )
        return value, detail, None

    x, libinfo, physical_residual = krylov(b)
    x = precondition(x) if right else x
    normb = jnp.linalg.norm(b)
    denominator = jnp.where(normb > 0, normb, 1.0)
    limit = (
        LINEAR_ACCEPTANCE_RTOL_FACTOR * rt + LINEAR_ACCEPTANCE_EPS_FACTOR * jnp.finfo(b.dtype).eps
    )

    def true_relative(value: jax.Array) -> jax.Array:
        """Compute the true operator residual rather than the Krylov estimate."""
        return jnp.linalg.norm(matvec(value) - b) / denominator

    rel = (
        true_relative(x)
        if physical_residual is None
        else jnp.linalg.norm(physical_residual) / denominator
    )
    if cfg.linear_refinements:
        # Restart from the independently recomputed physical residual. This
        # improves an inaccurate Krylov estimate; it does NOT relax acceptance.
        def needs_correction(carry: tuple[jax.Array, ...]) -> jax.Array:
            """Require another refinement only while the true residual exceeds its acceptance
            limit.
            """
            count, _, error = carry
            return (count < cfg.linear_refinements) & jnp.isfinite(error) & (error > limit)

        def correct(carry: tuple[jax.Array, ...]) -> tuple[jax.Array, ...]:
            """Apply a residual correction only if it reduces the independently measured error."""
            count, value, error = carry
            correction, _, _ = krylov(b - matvec(value))
            candidate = value + (precondition(correction) if right else correction)
            candidate_error = true_relative(candidate)
            improves = jnp.isfinite(candidate_error) & (candidate_error < error)
            return (
                count + 1,
                jnp.where(improves, candidate, value),
                jnp.where(improves, candidate_error, error),
            )

        corrections, x, rel = jax.lax.while_loop(
            needs_correction, correct, (jnp.asarray(0), x, rel)
        )
        if cfg.linear_work_reuse:
            rel = jax.lax.cond(corrections > 0, lambda _: true_relative(x), lambda _: rel, None)
    ok = jnp.isfinite(rel) & jnp.all(jnp.isfinite(x)) & (rel <= limit)
    if cfg.debug_linear:

        def report(_: None) -> jax.Array:
            """Emit opt-in linear failure diagnostics through a JAX debug callback."""
            jax.debug.print(
                "LINEAR FAILURE n={n} rhs_norm={b} relative={r} limit={t} libinfo={i}",
                n=n,
                b=normb,
                r=rel,
                t=limit,
                i=libinfo,
            )
            return jnp.asarray(0)

        jax.lax.cond(ok, lambda _: jnp.asarray(0), report, None)

    return (
        jnp.where(ok, x, jnp.full_like(x, jnp.nan)),
        jnp.stack((rel, ok.astype(b.dtype), libinfo.astype(b.dtype))),
    )


def linear_solve(
    matvec: Callable[[jax.Array], jax.Array],
    b: jax.Array,
    diag: jax.Array,
    cfg: NewtonConfig = NewtonConfig(),
    *,
    inverse_blocks: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Native implicit linear primitive with explicit transpose solve.

    Returning NaNs on unconverged linear solves makes invalid adjoints visible.
    Detailed adjoint residuals can also be requested with adjoint_diagnostics.
    The common diagnostic shape is unchanged. No per-Krylov counts are invented;
    direct fused_gmres callers can inspect its separate KrylovInfo counters.

    Optional inverse node blocks have shape (nodes, dim, dim) and act on the
    flattened node-major right-hand side. The adjoint uses their true transpose.
    Blocks are a frozen numerical preconditioner, not part of the physical
    equation; nonfinite blocks fail the unchanged true-residual acceptance gate.
    """
    if inverse_blocks is not None and (
        inverse_blocks.ndim != 3
        or inverse_blocks.shape[1] != inverse_blocks.shape[2]
        or inverse_blocks.shape[0] * inverse_blocks.shape[1] != b.size
    ):
        raise ValueError("Inverse preconditioner blocks must match the flattened unknowns")
    raw = lambda op, rhs: _raw_linear(op, rhs, diag, cfg, inverse_blocks=inverse_blocks)
    transpose = lambda op, rhs: _raw_linear(
        op, rhs, diag, cfg, inverse_blocks=inverse_blocks, transpose_blocks=True
    )
    return jax.lax.custom_linear_solve(
        matvec,
        b,
        solve=raw,
        transpose_solve=transpose,
        symmetric=False,
        has_aux=True,
    )


def solve_root(
    residual: Callable[[jax.Array], jax.Array],
    guess: jax.Array,
    cfg: NewtonConfig = NewtonConfig(),
    *,
    diagonal: jax.Array | None = None,
    admissible: Callable[[jax.Array], jax.Array] | None = None,
    inverse_blocks: jax.Array | None = None,
    inverse_blocks_factory: Callable[[jax.Array], jax.Array] | None = None,
) -> tuple[jax.Array, RootInfo]:
    """Solve a nonlinear residual with backtracking and an implicit adjoint.

    Supply either frozen inverse blocks or a factory rebuilding them at each
    Newton iterate. The factory path also rebuilds at the final root for the
    implicit adjoint; its numerical preconditioner is never differentiated.
    """
    if guess.ndim != 1:
        raise ValueError("Flatten root unknowns explicitly")
    if inverse_blocks is not None and inverse_blocks_factory is not None:
        raise ValueError("Choose frozen inverse blocks or a Newton-update factory")
    diag = jnp.ones_like(guess) if diagonal is None else jnp.broadcast_to(diagonal, guess.shape)
    admissible = (lambda z: jnp.asarray(True)) if admissible is None else admissible
    tol = tolerance(guess.dtype, cfg)

    def solve(f: Callable[[jax.Array], jax.Array], y0: jax.Array) -> tuple[jax.Array, RootInfo]:
        """Run admissible Newton iterations and return status-bearing root diagnostics."""
        valid = admissible(y0) & jnp.all(jnp.isfinite(y0))
        r0 = jnp.max(jnp.abs(f(y0)))
        code = jnp.where(
            valid & jnp.isfinite(r0), float(ROOT_OK), float(ROOT_INVALID_INITIAL)
        ).astype(y0.dtype)
        z = jnp.asarray(0.0, y0.dtype)

        # Newton carry: unknown, infinity-norm residual, iteration count,
        # backtrack count, maximum linear residual, solve count, status code.
        def cond(c: tuple[jax.Array, ...]) -> jax.Array:
            """Continue Newton updates while residual, status, and iteration limits permit."""
            y, r, i, bt, lr, ls, code = c
            return (r > tol) & (i < cfg.max_steps) & (code == ROOT_OK)

        def body(c: tuple[jax.Array, ...]) -> tuple[jax.Array, ...]:
            """Linearize the current residual and run an admissible backtracking line search."""
            y, r, i, bt, lr, ls, code = c
            fy, tangent = jax.linearize(f, y)
            blocks = (
                inverse_blocks
                if inverse_blocks_factory is None
                else inverse_blocks_factory(jax.lax.stop_gradient(y))
            )
            delta, linfo = linear_solve(tangent, -fy, diag, cfg, inverse_blocks=blocks)
            good_linear = linfo[1] > 0

            def trial_cond(c: tuple[jax.Array, ...]) -> jax.Array:
                """Continue admissible backtracking while no acceptable candidate has been found."""
                return (c[0] < cfg.backtracks) & (~c[3]) & good_linear

            def trial(c: tuple[jax.Array, ...]) -> tuple[jax.Array, ...]:
                """Evaluate one Armijo candidate, recording the first accepted root update."""
                k, out, rr, found = c
                scale = jnp.exp2(-k.astype(y.dtype))
                candidate = y + scale * delta
                valid = admissible(candidate) & jnp.all(jnp.isfinite(candidate))
                rc = jax.lax.cond(
                    valid,
                    lambda q: jnp.max(jnp.abs(f(q))),
                    lambda q: jnp.asarray(jnp.inf, y.dtype),
                    candidate,
                )
                accepted = (
                    valid & jnp.isfinite(rc) & ((rc <= (1 - cfg.armijo * scale) * r) | (rc <= tol))
                )
                return (
                    k + 1,
                    jnp.where(accepted, candidate, out),
                    jnp.where(accepted, rc, rr),
                    accepted,
                )

            k, yn, rn, found = jax.lax.while_loop(
                trial_cond,
                trial,
                (jnp.asarray(0, jnp.int32), y, r, jnp.asarray(False)),
            )
            code = jnp.where(
                ~good_linear,
                float(ROOT_LINEAR_FAILED),
                jnp.where(~found, float(ROOT_LINE_SEARCH_FAILED), float(ROOT_OK)),
            ).astype(y.dtype)
            return (
                yn,
                rn,
                i + 1,
                bt + jnp.maximum(k - 1, 0),
                jnp.maximum(lr, linfo[0]),
                ls + 1,
                code,
            )

        y, r, i, bt, lr, ls, code = jax.lax.while_loop(cond, body, (y0, r0, z, z, z, z, code))
        ok = (r <= tol) & (code == ROOT_OK) & admissible(y)
        code = jnp.where((~ok) & (code == ROOT_OK), float(ROOT_MAX_NEWTON), code)
        return y, RootInfo(r, i, ok.astype(y.dtype), bt, lr, ls, code)

    adjoint_blocks = inverse_blocks
    root_solver = solve
    if inverse_blocks_factory is not None:
        # Materialize the primal once so the transpose preconditioner can use
        # its final point. custom_root still defines the physical derivative;
        # neither the Newton search nor preconditioner construction is traced
        # as an optimization trajectory in the reverse pass.
        solved, information = solve(residual, guess)
        solved = jax.lax.stop_gradient(solved)
        information = jax.tree.map(jax.lax.stop_gradient, information)
        adjoint_blocks = jax.lax.stop_gradient(inverse_blocks_factory(solved))
        guess = solved

        def solved_root(
            f: Callable[[jax.Array], jax.Array], y0: jax.Array
        ) -> tuple[jax.Array, RootInfo]:
            """Attach the implicit-root derivative to the already computed primal result."""
            return solved, information

        root_solver = solved_root

    def tangent_solve(op: Callable[[jax.Array], jax.Array], rhs: jax.Array) -> jax.Array:
        """Solve the root tangent equation through the same configured linear solver."""
        return linear_solve(op, rhs, diag, cfg, inverse_blocks=adjoint_blocks)[0]

    return jax.lax.custom_root(residual, guess, root_solver, tangent_solve, has_aux=True)


def adjoint_diagnostics(
    residual: Callable[[jax.Array], jax.Array],
    root: jax.Array,
    cotangent: jax.Array,
    cfg: NewtonConfig = NewtonConfig(),
    diagonal: jax.Array | None = None,
    *,
    inverse_blocks: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Independently reproduce the implicit transpose solve for evidence."""
    _, vjp = jax.vjp(residual, root)
    op = lambda y: vjp(y)[0]
    diag = jnp.ones_like(root) if diagonal is None else diagonal
    blocks = None if inverse_blocks is None else jnp.swapaxes(inverse_blocks, -1, -2)
    return linear_solve(op, cotangent, diag, cfg, inverse_blocks=blocks)
