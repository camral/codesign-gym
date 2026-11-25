"""Raw Krylov diagnostics must expose overflowing finite residual norms."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from envs.soft_walker.engine.numerics.krylov import fused_gmres


def test_fused_finite_rhs_norm_overflow_is_not_convergence(dtype: np.dtype[np.floating]) -> None:
    large = 1e30 if dtype == np.dtype("float32") else 1e200
    rhs = jnp.full(2, large, dtype=dtype)
    assert np.isfinite(np.asarray(rhs)).all()
    _, info = jax.jit(
        lambda b: fused_gmres(
            lambda v: v,
            b,
            rtol=1e-5,
            restart=2,
            max_restarts=2,
        )
    )(rhs)
    assert not bool(info.finite)
    assert np.isposinf(np.asarray(info.residual_norm))
    assert int(info.iterations) == 0
    assert int(info.restarts) == 0
