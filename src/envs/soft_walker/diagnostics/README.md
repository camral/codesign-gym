# Diagnostics and benchmark programs

This package owns offline validation, numerical audits, profiling, benchmark
workers, evidence comparisons, and experiment summaries. It does not own runtime
validity checks: solver convergence, resident execution certificates, array and
topology validation, contact reporting, and differentiation contracts remain in
the engine and execute during normal simulation.

Directional finite-difference audits live in `../shared.py`. Import
`envs.soft_walker.shared.difference_ladder` for the full
epsilon ladder, observed convergence windows and agreement report. This replaces
`engine.locomotion.numerics`; it is separate from the production time integrators
and linear/nonlinear solvers in `engine.numerics`.

From a source checkout with an editable install, run a diagnostic as a module:

```sh
python -m envs.soft_walker.diagnostics.run_resident_tests --help
python -m envs.soft_walker.diagnostics.benchmark_action_hold --help
python -m envs.soft_walker.diagnostics.compare_performance_audits --help
```

The former `scripts/<name>.py` diagnostic forwarding files have been removed.
Update those commands to `python -m envs.soft_walker.diagnostics.<name>`; arguments,
defaults, report schemas, exit behavior, and subprocess isolation are unchanged.
Canonical diagnostic files also support direct execution from a source checkout,
including from a different working directory:

```sh
python /path/to/codesign-gym/src/envs/soft_walker/diagnostics/run_resident_tests.py --help
```

Python callers should import `envs.soft_walker.diagnostics.<name>` directly.
Replace former helper imports `scripts._layout` and `scripts._host_types` with
`envs.soft_walker.shared` and
`envs.soft_walker.diagnostics.host_types`, respectively. There is no replacement
for the removed `_compat` source-execution helper. A caller that replaces a worker
or writer hook must change that attribute on the canonical diagnostic module;
loading a removed forwarding file with `importlib` no longer works.

The `shared.py` module resolves repository, source, test, and script roots
independently of the caller's working directory without modifying `sys.path`.
Direct-file entry points explicitly select their checkout before package imports.
Evidence inventories include the canonical implementation and its host contracts.

The following runners require the checkout, including its `tests/` tree:
`run_validation`, `run_locomotion_tests`, `run_scenario_tests`,
`run_resident_tests`, `run_krylov_audit`, `run_performance_audit`,
`run_warp_followup`, `run_warp_native_tests`, `run_warp_ports_tests`, and
`run_warp_speed_audit`. Their repository-relative stage commands and source
inventories depend on that layout, even when an individual test stage is skipped.
`resident_worker` also imports repository test fixtures. The `worker` unit-test
stages and `performance_worker --unit` require the same checkout.

These commands reject a wheel-only installation with a checkout-specific error
before creating an evidence directory or launching work. Use an editable source
install (`python -m pip install -e .`) with `tests/soft_walker` present. Imports
and `--help` remain available in a wheel; neither runs or qualifies a simulation.
Saved-evidence comparisons do not require the checkout and accept explicit input
and output paths.

The followup parity comparison requires identical nonempty fixture coverage and
matching numerical selections. Device, output location, timeout, repetition,
optional unit-test collection, and additional performance-audit settings may
differ; they do not select the saved numerical fixtures. Both the followup and
focused-performance comparisons require paired array dtypes to match and select
their existing precision tolerances from the stored arrays, independently of
directory names. Each comparison retains its own norm or elementwise rule.

Shared implementation details remain private to this package:

- `_precision_comparison` implements the identical M2/M3 numerical gate;
  public commands retain their report names and entry points.
- `_focused_tests` executes the locomotion/scenario precision stages and writes
  partial evidence on failure; each command retains its selection and writer hook.
- `_processes` starts logged children and performs kill/termination mechanics.
  Each orchestrator retains its timeout, cancellation, return-code and scheduling
  policy, including whether later device work must stop.
- `_evidence` hashes explicitly selected files and writes checksums/ZIPs.
  Callers retain file selection, ordering, path conventions, schemas and exclusions.

The historical [consolidation validation record](../docs/CONSOLIDATION_VALIDATION.md)
records focused checks and matched warm-performance results from the earlier
extraction, before the forwarding files were removed.

User-facing replay, training, scenario generation, robot/terrain galleries,
rendering, and source packaging remain in `scripts/`. In particular `warp_mpm`
and `warp_solids` provide bundle export/replay and gradient workflows; their audit
workers moved here, while those production commands remain in place. Historical
archives and generated results have not moved or changed.
