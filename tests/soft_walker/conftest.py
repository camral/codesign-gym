"""Precision, tolerance, evidence, and compile-memory fixtures for soft-robot tests."""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import TYPE_CHECKING

import jax
import numpy as np
import pytest

from envs.soft_walker.shared import evidence_record as record

if TYPE_CHECKING:
    from envs.soft_walker.engine.physics.spring_config import SpringConfig

# Precision is scoped to each soft_walker test (not set process-wide) so the other JAX envs collected in the
# same root pytest session keep their float32 defaults.
_X64 = os.environ.get("CODESIGN_TEST_DTYPE", "float64") == "float64"


@pytest.fixture(autouse=True)
def _soft_walker_precision() -> Iterator[None]:
    """Scope JAX precision changes so unrelated environment tests keep defaults."""
    with jax.enable_x64(_X64), jax.default_matmul_precision("highest"):
        yield


@pytest.fixture(scope="session")
def dtype() -> np.dtype[np.floating]:
    """Return the precision selected by the isolated validation process."""
    return np.dtype(os.environ.get("CODESIGN_TEST_DTYPE", "float64"))


@pytest.fixture(scope="session")
def tolerance(dtype: np.dtype[np.floating]) -> float:
    """Return the suite's existing precision-dependent comparison tolerance."""
    return 3e-5 if dtype == np.dtype("float32") else 2e-9


@pytest.fixture
def small_config() -> SpringConfig:
    """Build the small spring case shared by API and differentiation checks."""
    from envs.soft_walker.engine.physics.spring_config import SpringConfig

    return SpringConfig(grid_shape=(2, 1), substeps=4, settle_steps=1, horizon=3)


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    record(
        "pytest",
        test=report.nodeid,
        phase=report.when,
        outcome=report.outcome,
        duration_s=report.duration,
        failure=str(report.longrepr) if report.failed else None,
    )


def pytest_sessionfinish(session: pytest.Session, exitstatus: int | pytest.ExitCode) -> None:
    record("pytest_session_finish", exitstatus=int(exitstatus))


@pytest.fixture(autouse=True)
def bounded_m3_test_compile_cache(request: pytest.FixtureRequest) -> Iterator[None]:
    """Release large driver executable families between independent unit cases.

    This does not change numerical inputs, tolerances, or benchmarks. Persistent
    disk caching remains enabled, so later processes can reload compiled code.
    It prevents the full collection of beam/MPM/FEM JAXPRs exhausting small hosts.
    """
    yield
    name = request.node.name
    boundaries = (
        "test_m3_driver_api_metrics_and_pure_rollout",
        "test_m3_vector_scalar_autoreset",
        "test_m3_closed_loop_policy_same_actions_and_gradient",
        "test_m3_failure_transaction_and_invalid_objective",
        "test_m3_beam_energy_objectivity_and_force_torque",
    )
    if name.startswith(boundaries):
        # Small containers need conservative cleanup. Larger machines retain
        # executable families until a measured RSS budget is reached, avoiding
        # unnecessary retracing on the user's workstation.
        from pathlib import Path

        limits = []
        try:
            limits.append(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
        except (OSError, ValueError, AttributeError):
            pass
        for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
            try:
                value = Path(path).read_text().strip()
                if value != "max":
                    limits.append(int(value))
            except (OSError, ValueError):
                pass
        limit = min(limits) if limits else 4 * 1024**3
        budget = min(0.3 * limit, 8 * 1024**3)
        try:
            rss = next(
                int(line.split()[1]) * 1024
                for line in Path("/proc/self/status").read_text().splitlines()
                if line.startswith("VmRSS:")
            )
        except (OSError, StopIteration, ValueError):
            rss = budget
        release = limit <= 6 * 1024**3 or rss >= budget
        record(
            "unit_compile_cache_budget",
            after_test=request.node.nodeid,
            memory_limit_bytes=limit,
            rss_bytes=rss,
            budget_bytes=budget,
            released=release,
        )
        if not release:
            return
        import gc

        from envs.soft_walker.engine.drivers import environment as drivers

        drivers.physics_kernels.cache_clear()
        drivers.kernels.cache_clear()
        drivers.instantiation_kernel.cache_clear()
        jax.clear_caches()
        gc.collect()
        record(
            "unit_compile_cache_release",
            after_test=request.node.nodeid,
            purpose="bound host memory; persistent disk cache retained",
        )
