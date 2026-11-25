# Shared-helper relocation

The requested single temporary owner is `envs.soft_walker.shared`. This moves
implementations, rather than adding forwarding files. Existing repository callers
use the new owner. Signatures, array computations, optimizer/checkpoint schemas
and CLI flags are preserved. JAX/physics imports occur only inside helpers that
need them; importing host utilities does not initialize a backend.

The two JSON policies remain separate: `write_json` maps nonfinite values to
null; `evidence_write_json` keeps explicit nonfinite markers. Diagnostics' former
short names now have an `evidence_` prefix to avoid collisions in the shared file.

This table describes this consolidation, after commit `4079d56`. The separate
[historical breaking API inventory](../../../../BREAKING_API_CHANGES.md) covers
the earlier shim removals and package reorganizations.

Names below omit the common `envs.soft_walker.` prefix.

| Previous object | New owner |
| --- | --- |
| `diagnostics._paths.require_source_checkout` | `shared.require_source_checkout` |
| `diagnostics._paths.diagnostic_sources` | `shared.diagnostic_sources` |
| `diagnostics._paths.source_group_files` | `shared.source_group_files` |
| `experiments.artifacts.json_value` | `shared.json_value` |
| `experiments.artifacts.write_json` | `shared.write_json` |
| `experiments.artifacts.write_npz` | `shared.write_npz` |
| `experiments.artifacts.append_event` | `shared.append_event` |
| `experiments.artifacts.source_hashes` | `shared.source_hashes` |
| `experiments.artifacts.environment_record` | `shared.environment_record` |
| `experiments.artifacts.archive_run` | `shared.archive_run` |
| `diagnostics.evidence.jsonable` | `shared.evidence_jsonable` |
| `diagnostics.evidence.write_json` | `shared.evidence_write_json` |
| `diagnostics.evidence.record` | `shared.evidence_record` |
| `diagnostics.evidence.sha256` | `shared.evidence_sha256` |
| `experiments.controllers.periodic_action` | `shared.periodic_action` |
| `experiments.optimization.bounded.InvalidEvaluation` | `shared.InvalidEvaluation` |
| `experiments.optimization.bounded.OptimizationResult` | `shared.OptimizationResult` |
| `experiments.optimization.bounded.OptimizationCallback` | `shared.OptimizationCallback` |
| `experiments.optimization.bounded.optimize` | `shared.optimize` |
| `experiments.optimization.checkpoint.AdamCheckpoint` | `shared.AdamCheckpoint` |
| `experiments.optimization.checkpoint.AdamCallback` | `shared.AdamCallback` |
| `experiments.optimization.checkpoint.load_checkpoint` | `shared.load_checkpoint` |
| `experiments.optimization.checkpoint.evaluate_objective` | `shared.evaluate_objective` |
| `experiments.optimization.checkpoint.save_checkpoint` | `shared.save_checkpoint` |
| `experiments.optimization.checkpoint.Objective` | `shared.Objective` |
| `experiments.optimization.adam.run_adam` | `shared.run_adam` |
| `experiments.embodiments._output_groups` | `shared._output_groups` |
| `experiments.embodiments._bounded_delta` | `shared._bounded_delta` |
| `experiments.embodiments.CPPNParameterization` | `shared.CPPNParameterization` |
| `experiments.embodiments.make_implicit_design` | `shared.make_implicit_design` |
| `diagnostics.gradient_checks.difference_ladder` | `shared.difference_ladder` |
| `diagnostics.gradient_checks.DifferenceSample` | `shared.DifferenceSample` |
| `diagnostics.gradient_checks.DifferenceReport` | `shared.DifferenceReport` |
| `diagnostics.gradient_checks.finite_difference_ladder` | `shared.finite_difference_ladder` |
| `diagnostics.scenario_preflight.physical_budget` | `shared.physical_budget` |
| `diagnostics.scenario_preflight.preflight` | `shared.preflight` |
| `diagnostics._paths.DIAGNOSTICS_ROOT` | `shared.DIAGNOSTICS_ROOT` |
| `diagnostics._paths.SOFT_ROOT` | `shared.SOFT_ROOT` |
| `diagnostics._paths.SCRIPTS_ROOT` | `shared.SCRIPTS_ROOT` |
| `diagnostics._paths.ENGINE_ROOT` | `shared.ENGINE_ROOT` |
| `diagnostics._paths.SRC_ROOT` | `shared.SRC_ROOT` |
| `diagnostics._paths.REPO_ROOT` | `shared.REPO_ROOT` |
| `diagnostics._paths.TESTS_ROOT` | `shared.TESTS_ROOT` |
| `experiments.locomotion.rollout.region_tags` | `shared.region_tags` |
| `experiments.locomotion.policies.REGION_BOUNDARY_TOLERANCE` | `shared.REGION_BOUNDARY_TOLERANCE` |
| `experiments.locomotion.policies.TORSO_REGION_START` | `shared.TORSO_REGION_START` |
| `experiments.locomotion.policies.FOOT_REGION_END` | `shared.FOOT_REGION_END` |

The old ten implementation modules and empty `experiments/optimization/__init__.py`
are removed; no compatibility shims are added. The `region_tags` implementation
and its three tagging constants also move out of the locomotion workflow.

Study-specific rollout/objective/resident adapters remain under `experiments/`.
They are not generic helper functions. Their public CLI consumers still need a
release decision before that entire directory can be excluded. Diagnostic
programs remain available in the development checkout, but public helper callers
no longer need the four moved diagnostic modules.

## Validation

Validation was performed against the committed implementations at `4079d56`.
The physics engine and registered Gym implementations are unchanged by this move.

| Check | Result |
| --- | --- |
| Before/after optimizer, checkpoint/resume, CPPN fields and analytical gradients | Exact equality for 812 arrays/scalars and 16 normalized JSON cases, across float32/float64. Only wall-clock timestamps and measured durations were excluded. |
| Helper and existing Adam tests | 23 passed in each precision. |
| Diagnostic, experiment, locomotion, scenario and implicit-body caller regressions | All 204 cases passed in float32. Float64 initially passed 203 with one obsolete filename expectation; that assertion was updated and passed in the focused follow-up. |
| Float64 follow-up: import boundaries, renderer geometry/recordings, spatial materials and body generators | 109 passed, including the corrected filename assertion. |
| CLI help | Locomotion, scenarios, random creatures, Warp MPM, Warp solids and source/results packager passed. |
| Fresh host import with accelerator and private workflow namespaces blocked | Passed without importing JAX, JAXLIB, Warp, Newton, experiments or diagnostics. |
| Current guide links and Python examples | 37 passed. |
| Scoped Ruff formatting/lint, mypy and strict Sphinx build | Passed; mypy checked 236 source files. |

The float32 regression run emitted one dtype warning from an existing test that
requests a float64 scenario while x64 is disabled. No numerical policy was changed.
Moved function bodies were also compared structurally, allowing only local import
placement and the documented evidence-name changes. Active source/test imports
resolve; limited forward-study and native-tool source manifests now include
`shared.py` so future results capture the helpers used.

These checks validate relocation, host tooling and CPU computations. They do not
constitute a new CUDA performance measurement or an interactive viewer session.
Historical results, galleries, baselines and handoff inventories were preserved.
Reproduction scripts, comparison payloads and test logs for this change are kept
outside the repository under the workspace's `runs/shared_helpers/` directory.
