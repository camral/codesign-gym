"""Classify Warp's metadata query without admitting capture mutations."""

from __future__ import annotations

from collections import Counter

import pytest

from envs.soft_walker.engine.execution.submissions import inspect_submissions


@pytest.mark.parametrize(
    "name,status,passed",
    [
        ("cuStreamIsCapturing", 0, True),
        ("cuStreamIsCapturing_ptsz", 0, True),
        ("cuStreamIsCapturing", 900, False),
        ("cuStreamBeginCapture", 0, False),
        ("cuStreamEndCapture", 0, False),
        ("cuStreamUnknownOperation", 0, False),
    ],
)
def test_warp_capture_query_classification(name: str, status: int, passed: bool) -> None:
    calls = [
        ("cuGraphLaunch", {"stream": 4096, "graph_exec": 8192}, 0),
        (name, {}, status),
        ("cuStreamSynchronize", {}, 0),
    ]
    records = [
        dict(
            sequence=i,
            name=operation,
            parameters=parameters,
            context=16384,
            enter_ns=10 * i,
            exit_ns=10 * i + 2,
            return_status=result,
        )
        for i, (operation, parameters, result) in enumerate(calls)
    ]
    report = inspect_submissions(Counter(row["name"] for row in records), records)
    assert report["passed"] is passed
    assert report["driver_api_counts"][name] == 1
