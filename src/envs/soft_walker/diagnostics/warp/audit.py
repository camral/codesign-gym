"""Observe warmed native whole-forward and whole-adjoint graph submissions.

Input uploads, graph construction, warmup, and output downloads are outside the
observed boundary. The observer does not change the simulation or inspect a
residual. This module deliberately imports no JAX.
"""

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from envs.soft_walker.engine.warp_native.audit_types import NativeSubmissionReport
    from envs.soft_walker.engine.warp_native.host_types import HostArray, PathLike

import json
from pathlib import Path

import numpy as np
import warp as wp

from envs.soft_walker.engine.warp_native.episode import Episode
from envs.soft_walker.engine.warp_native.model import DeviceModel

from .checks import contact_fixture


def audit_episode(precision: "str", out: "PathLike") -> "dict[str, NativeSubmissionReport]":
    """Audit warmed resident forward/adjoint submissions and save replay evidence.

    CUDA conditional graphs and CUPTI are required. This observer excludes
    uploads, compilation and downloads; failed numerics or residency checks
    raise rather than reporting their timings as successful measurements."""
    from envs.soft_walker.engine.execution.cupti import DriverAudit

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    device = wp.get_device("cuda")
    if not device.is_cuda or not wp.is_conditional_graph_supported():
        raise RuntimeError("A CUDA device with conditional-graph support is required")
    episode = Episode(
        DeviceModel(contact_fixture(2, precision), device),
        np.array([[0.4], [-0.2], [0.15]], precision),
        settling_steps=1,
        checkpoint_stride=2,
        velocity_weight=0.1,
        work_weight=0.05,
    )
    reports = {}
    for gradient in (False, True):
        name = "value_gradient" if gradient else "forward"
        first = episode.run(gradient=gradient)
        second = episode.run(gradient=gradient)
        if any(
            result["status"] != 0
            or not np.isfinite(result["loss"])
            or (gradient and result["adjoint_status"] != 0)
            for result in (first, second)
        ):
            raise RuntimeError(f"{name}: warmup failed numerically")
        reference = episode.loss.numpy().copy()
        reference_gradient = episode.gradients()["commands"].copy() if gradient else None
        stream = wp.get_stream(device)
        wp.synchronize_stream(stream)
        observer = DriverAudit()
        with observer:
            wp.capture_launch(episode.graphs["gradient" if gradient else "forward"], stream=stream)
            wp.synchronize_stream(stream)
        report = cast("NativeSubmissionReport", observer.report())
        value = episode.loss.numpy()
        numerical = np.all(np.isfinite(value)) and np.allclose(
            value, reference, rtol=2e-5, atol=1e-12
        )
        if gradient:
            result_gradient = episode.gradients()["commands"]
            numerical = (
                numerical
                and np.all(np.isfinite(result_gradient))
                and np.allclose(
                    result_gradient, cast("HostArray", reference_gradient), rtol=3e-4, atol=1e-10
                )
            )
        report.update(
            dict(
                numerical_replay_matches=bool(numerical),
                first=first,
                second=second,
                native_warp_version=wp.__version__,
            )
        )
        reports[name] = report
        (out / f"{name}_audit.json").write_text(json.dumps(report, indent=2))
        if not report["passed"] or not numerical:
            raise RuntimeError(
                f"{name}: native whole-execution audit failed; inspect {name}_audit.json"
            )
    return reports
