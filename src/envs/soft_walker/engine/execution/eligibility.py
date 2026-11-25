"""Fail early on known capture obstacles; this is not a residency certificate.

The project's pinned XLA source has no command-buffer conversion for its
separate TriangularSolveThunk. Merely enabling CUBLAS/CUSTOM_CALL does not
make that thunk convertible. Runtime certification is still mandatory even
when this static screen finds no obstacle.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping

from .audit_types import EligibilityReport


def capture_eligibility(
    primitive_counts: Mapping[str, int], optimized_hlo: str | None = None
) -> EligibilityReport:
    """Inventory primitives/library calls without executing the program.

    Keep the blacklist narrow and evidence-based. An unknown operation is
    reported, not declared supported. The CUPTI boundary audit remains the
    authority for an executable on the actual installed compiler/runtime.
    """
    errors = []
    if primitive_counts.get("triangular_solve", 0):
        errors.append(
            "triangular_solve lowers to a non-convertible CUDA TRSM thunk in "
            "the referenced XLA runtime; select gmres_method='fused' with the "
            "graph-compatible back substitution (no automatic solver switch)"
        )
    targets: Counter[str] = Counter()
    if optimized_hlo is not None:
        targets.update(re.findall(r'custom_call_target="([^"]+)"', optimized_hlo))
        for target in targets:
            if any(tag in target.lower() for tag in ("triangularsolve", "trsm")):
                errors.append("Non-convertible triangular-solve custom call: " + target)
            if "dynamic" in target.lower() and "slice" in target.lower():
                errors.append("Potential host-offset dynamic-slice custom call: " + target)
    return {
        "passed": not errors,
        "errors": errors,
        "custom_call_targets": dict(sorted(targets.items())),
        "primitive_triangular_solve_count": primitive_counts.get("triangular_solve", 0),
        "runtime_residency_proven": False,
        "source_scope": "XLA bb760b047bdbfeff962f0366ad5cc782c98657e0; installed build may differ",
    }
