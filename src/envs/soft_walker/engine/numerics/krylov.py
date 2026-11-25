"""Restarted real GMRES without nested data-dependent Arnoldi loops.

The outer restart and Arnoldi iterations still stop adaptively. Orthogonalization
uses two fixed classical Gram--Schmidt passes, and previous Givens rotations are
applied with a parallel affine prefix scan. Neither operation contains a dynamic
WHILE or a host callback. This targets accelerator launch/synchronization cost;
it does not change the equations or relax the caller's true-residual checks.

Only O(n * restart + restart**2) arrays are constructed. The back substitution
is confined to the restart-sized Hessenberg factor, never a global tangent.
It lowers to ordinary array operations, not the non-capturable CUDA TRSM thunk.
Differentiation belongs to the caller's custom_linear_solve, not these iterates.
"""

from __future__ import annotations

from typing import Callable, NamedTuple, cast

import jax
import jax.numpy as jnp


@jax.named_call
def triangular_backsolve(upper: jax.Array, rhs: jax.Array) -> jax.Array:
    """Solve one small real upper-triangular system with array operations.

    Only the GMRES restart-space system belongs here, never a global tangent.
    The Python loop expands at trace time over the *static* restart capacity;
    there is no runtime Python/host iteration, library TRSM, or custom call.
    Independent solves are batched by ``vmap``. Reverse/JVP transformations
    see ordinary JAX arithmetic, including the dependence on ``upper``.

    Values below the diagonal are not read. Singular diagonals are not clamped:
    NaN/Inf remains visible to the caller's unchanged true-residual check.
    """
    if upper.ndim != 2 or rhs.ndim != 1 or upper.shape != (rhs.size, rhs.size):
        raise ValueError("Expected a square restart-space matrix and matching vector")
    if rhs.size < 1 or rhs.dtype not in (jnp.float32, jnp.float64):
        raise ValueError("triangular_backsolve requires a nonempty real floating vector")
    if upper.dtype != rhs.dtype:
        raise TypeError("Matrix and right-hand side must have the same precision")

    # Column elimination: after column i is eliminated, earlier rows carry
    # b[:i] - R[:i, i:] @ x[i:]. The completed tail holds solved coefficients.
    # Keep each row as a scalar expression during tracing. Indexed array
    # writes can become XLA __dynamic_memcpy operations, splitting a resident
    # CUDA graph even when the Python index is static. This list is trace-time
    # bookkeeping only; the executable contains ordinary scalar arithmetic
    # and a final stack, with the same descending-column elimination order.
    answer = [rhs[i] for i in range(rhs.size)]
    for i in range(rhs.size - 1, -1, -1):
        coefficient = answer[i] / upper[i, i]
        answer[i] = coefficient
        for row in range(i):
            answer[row] = answer[row] + (-upper[row, i] * coefficient)
    return jnp.stack(answer)


class KrylovInfo(NamedTuple):
    """Actual work of this invocation, excluding the caller's residual audit."""

    restarts: jax.Array
    iterations: jax.Array
    residual_norm: jax.Array
    finite: jax.Array


@jax.named_call
def apply_rotations(h: jax.Array, cosine: jax.Array, sine: jax.Array) -> jax.Array:
    """Apply adjacent Givens rotations in order using an affine prefix scan.

    Rotation i maps (t_i, h[i+1]) to
        (c_i*t_i - s_i*h[i+1], s_i*t_i + c_i*h[i+1]).
    Thus t[i+1] = s_i*t_i + c_i*h[i+1]. Composing these affine
    maps is associative. ``lax.associative_scan`` expands a fixed-depth,
    data-parallel expression, rather than a dynamic loop up to iteration k.
    Unused rotations are exactly (cosine=1, sine=0).
    """

    def compose(
        left: tuple[jax.Array, jax.Array], right: tuple[jax.Array, jax.Array]
    ) -> tuple[jax.Array, jax.Array]:
        """Compose adjacent affine Givens maps for the associative rotation scan."""
        a_left, b_left = left
        a_right, b_right = right
        return a_right * a_left, a_right * b_left + b_right

    a, b = jax.lax.associative_scan(compose, (sine, cosine * h[1:]))
    incoming = jnp.concatenate((h[:1], a[:-1] * h[0] + b[:-1]))
    rotated = cosine * incoming - sine * h[1:]
    tail = a[-1:] * h[0] + b[-1:]
    return jnp.concatenate((rotated, tail))


def _normalize(vector: jax.Array, *, threshold: float = 0.0) -> tuple[jax.Array, jax.Array]:
    """Return a unit vector and a magnitude that keeps nonfinite failures visible."""
    norm = jnp.linalg.norm(vector)
    finite = jnp.isfinite(norm)
    usable = finite & (norm > threshold)
    safe_norm = jnp.where(usable, norm, jnp.asarray(1, vector.dtype))
    # Finite entries can still overflow their norm. Keep that failure visible
    # instead of reporting the zero magnitude reserved for finite breakdown.
    magnitude = jnp.where(finite, jnp.where(usable, norm, 0), jnp.inf)
    return jnp.where(usable, vector / safe_norm, 0), magnitude


@jax.named_call
def fused_gmres(
    matvec: Callable[[jax.Array], jax.Array],
    b: jax.Array,
    *,
    rtol: float,
    restart: int,
    max_restarts: int,
    precondition: Callable[[jax.Array], jax.Array] | None = None,
) -> tuple[jax.Array, KrylovInfo]:
    """Zero-start GMRES for a flattened real floating-point right-hand side.

    Matches the existing wrapper's left-preconditioned stopping convention:
    ``rtol*||b||`` at restart boundaries and ``rtol*||M b||`` inside Arnoldi.
    The production right-preconditioned path supplies its transformed operator
    and no left preconditioner. The caller independently checks ||A x-b||.

    Two-pass orthogonalization and a different rotation evaluation order can
    change rounding/iteration counts relative to JAX's incremental algorithm.
    Nonconvergence is not repaired by a dense or legacy-solver fallback.
    """
    return cast(
        tuple[jax.Array, KrylovInfo],
        _fused_gmres(
            matvec,
            b,
            rtol=rtol,
            restart=restart,
            max_restarts=max_restarts,
            precondition=precondition,
            retain_residual=False,
        ),
    )


def fused_gmres_with_residual(
    matvec: Callable[[jax.Array], jax.Array],
    b: jax.Array,
    *,
    rtol: float,
    restart: int,
    max_restarts: int,
    precondition: Callable[[jax.Array], jax.Array] | None = None,
) -> tuple[jax.Array, KrylovInfo, jax.Array]:
    """Return the independently evaluated unpreconditioned residual along with GMRES.

    The extra vector is exactly ``b - matvec(x)`` from the last completed restart,
    or ``b`` for the known zero initial solution. It is not the Arnoldi estimate
    or the preconditioned residual. A caller can reuse it for a physical residual
    audit while the solution remains unchanged; refinements require a fresh audit.
    """
    return cast(
        tuple[jax.Array, KrylovInfo, jax.Array],
        _fused_gmres(
            matvec,
            b,
            rtol=rtol,
            restart=restart,
            max_restarts=max_restarts,
            precondition=precondition,
            retain_residual=True,
        ),
    )


def _fused_gmres(
    matvec: Callable[[jax.Array], jax.Array],
    b: jax.Array,
    *,
    rtol: float,
    restart: int,
    max_restarts: int,
    precondition: Callable[[jax.Array], jax.Array] | None,
    retain_residual: bool,
) -> tuple[jax.Array, KrylovInfo] | tuple[jax.Array, KrylovInfo, jax.Array]:
    """Run the same Krylov algorithm with an optional final physical-residual buffer."""
    if b.ndim != 1 or b.size < 1 or b.dtype not in (jnp.float32, jnp.float64):
        raise ValueError("fused_gmres requires a nonempty float32/float64 vector")
    if not 0 < rtol < 1 or restart < 1 or max_restarts < 1:
        raise ValueError("Invalid GMRES tolerance or iteration capacities")
    restart = min(int(restart), b.size)
    precondition = (lambda x: x) if precondition is None else precondition
    dtype = b.dtype
    eps = jnp.finfo(dtype).eps
    residual = precondition(b)
    unit, residual_norm = _normalize(residual)
    outer_tol = rtol * jnp.linalg.norm(b)
    inner_tol = rtol * jnp.linalg.norm(residual)

    def cycle(x: jax.Array, unit: jax.Array, residual_norm: jax.Array) -> tuple[jax.Array, ...]:
        """Run one restarted Krylov cycle while retaining the current residual and convergence
        flag.
        """
        basis = jnp.zeros((b.size, restart + 1), dtype).at[:, 0].set(unit)
        # Unvisited columns represent an identity block, so early termination
        # leaves a nonsingular triangular tail with a zero right-hand side.
        factor = jnp.eye(restart, restart + 1, dtype=dtype)
        beta = jnp.zeros(restart + 1, dtype).at[0].set(residual_norm)
        cosine = jnp.ones(restart, dtype)
        sine = jnp.zeros(restart, dtype)

        def keep_going(carry: tuple[jax.Array, ...]) -> jax.Array:
            """Continue Arnoldi expansion while the residual and basis capacity permit it."""
            k, error, _, _, _, _, _, breakdown = carry
            return (k < restart) & (error > inner_tol) & jnp.isfinite(error) & ~breakdown

        def arnoldi(carry: tuple[jax.Array, ...]) -> tuple[jax.Array, ...]:
            """Extend the masked Krylov basis and update its small least-squares system on device."""
            k, _, basis, factor, beta, cosine, sine, _ = carry
            candidate = precondition(matvec(basis[:, k]))
            original_norm = jnp.linalg.norm(candidate)
            # Classical Gram--Schmidt twice: fixed work, no conditional norm
            # loop and no iteration-dependent sequence of small projections.
            h = jnp.matmul(basis.T, candidate, precision=jax.lax.Precision.HIGHEST)
            remainder = candidate - jnp.matmul(basis, h, precision=jax.lax.Precision.HIGHEST)
            correction = jnp.matmul(basis.T, remainder, precision=jax.lax.Precision.HIGHEST)
            remainder = remainder - jnp.matmul(
                basis, correction, precision=jax.lax.Precision.HIGHEST
            )
            h = h + correction
            next_vector, next_norm = _normalize(remainder, threshold=eps * original_norm)
            basis = basis.at[:, k + 1].set(next_vector)
            h = h.at[k + 1].set(next_norm)
            rotated = apply_rotations(h, cosine, sine)

            a, tail = rotated[k], rotated[k + 1]
            magnitude = jnp.hypot(a, tail)
            safe_magnitude = jnp.where(magnitude > 0, magnitude, 1)
            c = jnp.where(magnitude > 0, a / safe_magnitude, 1)
            s = jnp.where(magnitude > 0, -tail / safe_magnitude, 0)
            # Keep the small QR system in its original scale; no normal
            # equations and no extra conditioning loss from H.T @ H.
            rotated = rotated.at[k].set(c * a - s * tail)
            rotated = rotated.at[k + 1].set(0)
            factor = factor.at[k, :].set(rotated)
            cosine = cosine.at[k].set(c)
            sine = sine.at[k].set(s)
            beta_k, beta_next = beta[k], beta[k + 1]
            beta = beta.at[k].set(c * beta_k - s * beta_next)
            beta = beta.at[k + 1].set(s * beta_k + c * beta_next)
            return (k + 1, jnp.abs(beta[k + 1]), basis, factor, beta, cosine, sine, next_norm == 0)

        initial = (
            jnp.asarray(0, jnp.int32),
            residual_norm,
            basis,
            factor,
            beta,
            cosine,
            sine,
            jnp.asarray(False),
        )
        count, _, basis, factor, beta, _, _, _ = jax.lax.while_loop(keep_going, arnoldi, initial)
        coefficients = triangular_backsolve(factor[:, :-1].T, beta[:-1])
        candidate = x + jnp.matmul(basis[:, :-1], coefficients, precision=jax.lax.Precision.HIGHEST)
        physical_residual = b - matvec(candidate)
        residual = precondition(physical_residual)
        unit, norm = _normalize(residual)
        # _normalize deliberately makes zero/invalid vectors safe. Do not
        # accidentally classify NaN/Inf residuals as converged zero residuals.
        norm = jnp.where(jnp.all(jnp.isfinite(residual)), norm, jnp.inf)
        result = (candidate, unit, norm, count)
        return result + (physical_residual,) if retain_residual else result

    def keep_restarting(carry: tuple[jax.Array, ...]) -> jax.Array:
        """Continue while the true residual exceeds tolerance and restart capacity remains."""
        _, _, norm, cycles, _ = carry[:5]
        return (cycles < max_restarts) & (norm > outer_tol) & jnp.isfinite(norm)

    def restart_cycle(carry: tuple[jax.Array, ...]) -> tuple[jax.Array, ...]:
        """Run one fixed-capacity cycle and update restart diagnostics."""
        x, unit, norm, cycles, count = carry[:5]
        updated = cycle(x, unit, norm)
        x, unit, norm, used = updated[:4]
        result = (x, unit, norm, cycles + 1, count + used)
        return result + (updated[4],) if retain_residual else result

    residual_norm = jnp.where(jnp.all(jnp.isfinite(residual)), residual_norm, jnp.inf)
    initial: tuple[jax.Array, ...] = (
        jnp.zeros_like(b),
        unit,
        residual_norm,
        jnp.asarray(0, jnp.int32),
        jnp.asarray(0, jnp.int32),
    )
    if retain_residual:
        initial = initial + (b,)
    final = jax.lax.while_loop(keep_restarting, restart_cycle, initial)
    x, _, norm, cycles, count = final[:5]
    finite = jnp.all(jnp.isfinite(x)) & jnp.isfinite(norm)
    result = (x, KrylovInfo(cycles, count, norm, finite))
    return result + (final[5],) if retain_residual else result
