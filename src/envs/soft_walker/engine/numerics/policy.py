"""Fixed solver policies shared by equivalent backend implementations.

These host scalars import no accelerator runtime. Callers retain their original
arithmetic order and precision casts; mathematical series coefficients remain
in their equations instead of being mixed with acceptance policies here.
These are inherited implementation policies, not newly calibrated thresholds.
"""

# Independently recomputed linear residual: factor * configured rtol + factor * eps.
LINEAR_ACCEPTANCE_RTOL_FACTOR = 8
LINEAR_ACCEPTANCE_EPS_FACTOR = 32

# Historical scalar-preconditioner guard: entries with magnitude at or below
# this floor, or nonfinite entries, use the existing unit fallback. This is not
# a precision-relative tolerance.
PRECONDITIONER_DIAGONAL_FLOOR = 1e-20
