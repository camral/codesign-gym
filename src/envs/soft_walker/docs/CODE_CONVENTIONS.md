# Readable soft-robot code

These conventions apply to first-party Python in the engine, wrapper, scripts,
examples, viewer, and associated tests. Third-party code, saved runs, generated
galleries, and historical evidence are excluded from automated rewrites.

## Formatting and names

Use the repository's scoped Ruff configuration and a 100-character line target.
Break complex expressions into named intermediate values when that explains
the computation, keeping numerical operation order intact. Short mathematical
names such as `F`, `C`, and `dt` are appropriate when their meaning and units are
documented. Use descriptive names for orchestration and ownership.

Run from the repository root:

```sh
python -m ruff format --check
python -m ruff check
python -m mypy --config-file mypy-soft.ini
```

The `dev` extra installs the checked formatter/type-checker versions. The scope
is configured explicitly; unrelated environments in codesign-gym are not part
of this cleanup. Import-root bootstrap code in standalone scripts and optional
backend selection in tests intentionally precede some imports.

Mypy checks host-side source with required argument and return annotations. Its
`mypy_path` resolves the canonical `envs.soft_walker` package from `src`; do not
check the same module again under a `src.envs` name. Optional third-party
implementations are not followed. The runtime-generated Warp kernel
modules listed in `mypy-soft.ini` are excluded from mypy body checks; their
ordinary host functions still carry annotations, while Warp checks DSL types.
Associated tests are formatted, annotated, and executed, but are not included
in the host-source mypy target.

## Types and docstrings

Annotate Python function inputs and returns with concrete scalar, array,
configuration, protocol, and report types. Use `TypedDict` for stable dictionary
contracts without changing their runtime representation. Use protocols for
structural backend interfaces; do not introduce inheritance only to satisfy a
type checker. Keep optional accelerator imports behind `TYPE_CHECKING` where
annotations are the only reason for importing them.

Type annotations do not encode all numerical contracts. Document shape axes,
SI units, normalization, precision/device assumptions, whether an array is host
or device data, and whether it is borrowed or copied. Explain rejection/failure
results and observable mutations. Do not add comments or docstrings that merely
repeat a function's name or claim unsupported physical guarantees.

Dynamic compiler/library boundaries may need an explicit narrow typing escape.
Explain it at the boundary rather than propagating untyped dictionaries or
blanket `Any` throughout otherwise structured code.

## Compiled-code boundaries

Warp kernels use evaluated annotations involving runtime-generated scalar,
vector, array, and struct types. Do not mechanically add postponed annotations
to these factories, move generated kernels between modules, or replace their
annotations with ordinary NumPy types. Their compiler is responsible for kernel
type checking; host protocols describe the interface used by orchestration.

Preserve JAX pytree layouts, static arguments, `jit`/`vmap`/scan boundaries,
custom derivatives, and cache keys during readability changes. Do not replace
a numerical `NamedTuple` with an ordinary dataclass or turn a static choice into
an array-valued branch without treating that as a behavioral change.

Host-only extraction is usually the safer first refactor: separate validation,
configuration, observation/reporting, checkpoint I/O, and recording geometry
from simulation transitions. Preserve established imports using explicit
re-exports, and keep public signatures/defaults and artifact schemas stable.

Shared host helpers own mechanics rather than merging backend policies.
For example, `registered/task.py` shares validated task setup while each backend
owns its episode state; `warp_native/_episode_host.py` shares checkpoint
orchestration while concrete classes retain physics steps and buffer allocation.
Diagnostic process helpers leave exception handling, timeout exit codes and stage
scheduling in their command modules. Import canonical diagnostic modules before
replacing worker or writer hooks; the former script forwarding files and legacy
execution namespaces have been removed. See the
[breaking cleanup migration](BREAKING_CLEANUP_MIGRATION.md) for current paths and
the historical [consolidation record](CONSOLIDATION_VALIDATION.md) for the earlier
extraction checks.

## Evidence and support claims

The [capability matrix](CAPABILITIES.md) is the implementation/support reference.
Tests establish only their stated inputs, precision, device, and execution path.
Recorded optimization results do not promise that another design will walk or
that an arbitrary timestep will converge. Keep historical outputs unchanged and
record new verification evidence separately. Formatting, refactoring, and
behavior changes should be independently reviewable.
