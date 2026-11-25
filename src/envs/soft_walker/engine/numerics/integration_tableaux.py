"""Shared host-only integration coefficients with no accelerator imports.

Kvaerno3 values match the published tableau as shipped by pinned Diffrax 0.7.2
(Apache-2.0; see NOTICE.md). Preserve these decimal constants and operation order:
increasing their apparent precision would change the established integrator.
"""

from typing import Final

KVAERNO3_GAMMA: Final = 0.4358665215
KVAERNO3_A: Final = (
    (0.0, 0.0, 0.0, 0.0),
    (KVAERNO3_GAMMA, KVAERNO3_GAMMA, 0.0, 0.0),
    (0.4905633884191082, 0.0735700900808919, KVAERNO3_GAMMA, 0.0),
    (0.30880996997303606, 1.490563388254108, -1.2352398797271447, KVAERNO3_GAMMA),
)
KVAERNO3_C: Final = (0.0, 2 * KVAERNO3_GAMMA, 1.0, 1.0)
