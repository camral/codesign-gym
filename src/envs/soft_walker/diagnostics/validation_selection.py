"""Explicit, reviewable incremental selection. Full suites remain available.

Each old test listed here exercises a changed path. This is not a probabilistic
selection based on last-run successes. Updating this file is part of a patch.
"""

from envs.soft_walker.shared import (
    ENGINE_ROOT as ENGINE_ROOT,
    REPO_ROOT as REPO_ROOT,
    SCRIPTS_ROOT as SCRIPTS_ROOT,
    SOFT_ROOT as SOFT_ROOT,
    SRC_ROOT as SRC_ROOT,
    TESTS_ROOT as TESTS_ROOT,
)

LEGACY_STAGES = [
    "unit",
    "numerics",
    "benchmark",
    "m2_unit",
    "m2_numerics",
    "m2_mechanics",
    "m2_benchmark",
]
M3_STAGES = ["m3_unit", "m3_numerics", "m3_benchmark"]
CHANGED_STAGES = ["affected"] + M3_STAGES
FULL_STAGES = LEGACY_STAGES + M3_STAGES
AFFECTED_TESTS = [
    "tests/soft_walker/test_contract.py::test_factory_registry",
    "tests/soft_walker/test_contract.py::test_arities_observations_and_metrics",
    "tests/soft_walker/test_contract.py::test_vector_scalar_and_next_step_autoreset",
    "tests/soft_walker/test_contract.py::test_vector_rejection_and_partial_reset",
    "tests/soft_walker/test_contract.py::test_real_gymnasium_types",
    "tests/soft_walker/test_contract.py::test_padding_invariance",
    "tests/soft_walker/test_contract.py::test_unsupported_step_update_warns_and_ignores",
    "tests/soft_walker/test_mechanics.py::test_resolution_mass_and_affine_energy_scaling",
    "tests/soft_walker/test_mechanics.py::test_contact_energy_force_and_friction",
    "tests/soft_walker/test_mechanics.py::test_rollout_matches_api_including_settling",
    "tests/soft_walker/test_mechanics.py::test_invalid_pure_rollout_has_nan_objectives",
    "tests/soft_walker/test_m2_continuum.py::test_m2_mpm_internal_force_variational_and_contact",
    "tests/soft_walker/test_m2_continuum.py::test_m2_mpm_freeflight",
    "tests/soft_walker/test_m2_continuum.py::test_m2_hex_freeflight_and_fixed_nodes",
    "tests/soft_walker/test_m2_continuum.py::test_m2_hex_empty_nodes_nan_safety",
    "tests/soft_walker/test_m2_solvers.py::test_m2_retry_transaction_and_replay",
    "tests/soft_walker/test_decoders.py::test_decoder_fields_and_roundtrip[hard-2-direct]",
    "tests/soft_walker/test_decoders.py::test_decoder_fields_and_roundtrip[relaxed-3-cppn]",
    "tests/soft_walker/test_decoders.py::test_serialization_rejects_descriptor_mismatch",
]
REASONS = {
    "env.py": "driver factory dispatch and shared vector rewards",
    "differentiation.py": "ND validation and optional dynamic terrain",
    "physics/spring.py": "gray-surface refinement correction and optional terrain",
    "physics/contact.py": "normal/tangent projection generalized to analytical terrain",
    "physics/fem.py": "dynamic terrain model payload and contact diagnostics",
    "physics/mpm.py": "dynamic terrain model payload and contact diagnostics",
    "serialization.py": "backward-compatible v1 plus explicit v2 global-Poisson parameter schema",
}
