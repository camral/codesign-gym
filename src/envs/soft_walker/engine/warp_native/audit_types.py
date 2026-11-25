"""Oracle inputs and native numerical evidence layered on the CUDA API audit."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

from ..execution.audit_types import SubmissionReport

if TYPE_CHECKING:
    from ..types import Array
    from .host_types import EpisodeResult, Report


class ResolvedParameters(TypedDict):
    """Differentiable commands, resolved model fields, and initial-state arrays.

    Integer connectivity and status remain fixed outside the differentiated tree.
    Array placement is selected by the oracle caller before compilation.
    """

    commands: Array
    model: dict[str, Array]
    initial: dict[str, Array]


class NativeSubmissionReport(SubmissionReport, total=False):
    """Submission evidence augmented with independently downloaded numerical checks."""

    numerical_replay_matches: bool
    numerical_match: bool
    output_matches: bool
    numerical_comparisons: dict[str, Report]
    first: EpisodeResult
    second: EpisodeResult
    native_warp_version: str
