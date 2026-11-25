"""Canonical solver imports preserve host-only policies, record identity and cached kernels."""

from __future__ import annotations

import importlib
import pickle
import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest

from envs.soft_walker.engine.drivers.environment import physics_kernels
from envs.soft_walker.engine.drivers.task import BodyConfig
from envs.soft_walker.engine.numerics import integration_tableaux, integrators, krylov, solvers
from envs.soft_walker.engine.physics import common


def test_numerics_namespace_coefficients_and_policy_are_host_only() -> None:
    """Choosing numerical coefficients or policy constants does not initialize a backend."""
    script = """
import importlib.util, sys
from envs.soft_walker.engine import numerics
from envs.soft_walker.engine.numerics import integration_tableaux, policy
assert not hasattr(numerics, 'NewtonConfig')
assert not {'jax', 'warp', 'newton'}.intersection(sys.modules)
for name in ('integration_tableaux', 'numerical_policy',
             'physics.integrators', 'physics.solvers', 'physics.krylov'):
    assert importlib.util.find_spec('envs.soft_walker.engine.' + name) is None
"""
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)


def test_numerical_callers_share_canonical_solver_and_coefficient_owners() -> None:
    """Integrator and physics adapters use the same solver records and published tableau."""
    assert integrators.solve_root is solvers.solve_root
    assert integrators.NewtonConfig is common.NewtonConfig is solvers.NewtonConfig
    assert integrators.RootInfo is common.RootInfo is solvers.RootInfo
    assert solvers.fused_gmres is krylov.fused_gmres
    assert integrators.A is integration_tableaux.KVAERNO3_A
    assert integrators.C is integration_tableaux.KVAERNO3_C
    assert integrators.GAMMA == integration_tableaux.KVAERNO3_GAMMA


@pytest.mark.parametrize(
    "module_name, record_name", (("krylov", "KrylovInfo"), ("solvers", "RootInfo"))
)
def test_numerical_record_pickles_use_canonical_modules(module_name: str, record_name: str) -> None:
    """New diagnostic record pickles retain their concrete class and ordered scalar fields."""
    canonical = f"envs.soft_walker.engine.numerics.{module_name}"
    record_type = getattr(importlib.import_module(canonical), record_name)
    assert record_type.__module__ == canonical
    assert pickle.loads(pickle.dumps(record_type)) is record_type
    values = [np.asarray(index, dtype=np.float32) for index in range(len(record_type._fields))]
    record = record_type(*values)
    restored = pickle.loads(pickle.dumps(record))
    assert type(restored) is record_type
    assert restored._fields == record._fields
    for expected, actual in zip(record, restored):
        np.testing.assert_array_equal(actual, expected)


def test_newton_config_pickle_preserves_production_kernel_cache_key() -> None:
    """Equal restored numerical settings reuse the existing cached body-kernel factory."""
    configuration = solvers.NewtonConfig()
    restored = pickle.loads(pickle.dumps(configuration))
    assert type(restored) is solvers.NewtonConfig
    assert type(restored).__module__ == "envs.soft_walker.engine.numerics.solvers"
    assert restored == configuration and hash(restored) == hash(configuration)
    body = BodyConfig(dynamics=common.DynamicsConfig(newton=configuration))
    restored_body = replace(body, dynamics=replace(body.dynamics, newton=restored))
    functions = physics_kernels("fem", body, None)
    hits_before = physics_kernels.cache_info().hits
    assert physics_kernels("fem", restored_body, None) is functions
    assert physics_kernels.cache_info().hits == hits_before + 1
