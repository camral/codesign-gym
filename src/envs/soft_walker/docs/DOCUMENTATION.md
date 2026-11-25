# Documentation maintenance

The [component guide](COMPONENTS_AND_GYMNASIUM.md) is the authored architecture
reference. The [capability matrix](CAPABILITIES.md) records support and validation
limits, while the [modularity matrix](MODULARITY_MATRIX.md) lists configurable
choices. Update these together when a public contract changes. Keep experimental
protocols and dated measurements in the [historical records](HISTORY.md), with
links from current guides when their evidence remains relevant.

## API reference on Read the Docs

The engine's host-facing API reference is part of the Codesign Gym docs, in the Soft Walker section
(`docs/envs/soft_walker/engine.md`, `engine-api.md` and `guides.md`). It documents the host-facing entry points only;
it does not generate pages for internal kernels. The guides in this folder stay the authored reference; the docs link
to them on GitHub. Build the whole site from the repository root:

```sh
python -m pip install -r docs/requirements.txt
python -m sphinx -b html docs docs/_build/html
```

## Distribution contents

The wheel includes the first-party simulator modules, CLI modules, example JSON
inputs, small numerical reference data, and Markdown guides. It excludes saved
galleries, `runs/`, `handoff/`, local CMD/PowerShell launchers and generated HTML. Run installed tools with their module paths, for example:

```sh
python -m envs.soft_walker.scripts.scenarios --help
python -m envs.soft_walker.viewer.view_newton --run /path/to/your/recording
```

Newton and its graphics dependencies remain separate from the base installation.
The registered `SoftWalker-*` task runs on Warp only. The low-level JAX engine works without Warp. The optional legacy spring Diffrax
method requires `python -m pip install diffrax optimistix`; native integration
methods do not require Diffrax, Equinox, Optimistix, Optax or Lineax.

The source distribution additionally contains `tests/` and their JSON fixtures and
`mypy-soft.ini`, so numerical tests and scoped typing can run from its extracted directory. The full checkout is
still required to follow historical media/evidence links or run the complete
checkout-link validation in `tests/test_soft_walker_documentation.py`. Those
links deliberately retain the original evidence rather than duplicating hundreds
of megabytes into an installation archive. A source distribution is not the
full gallery/results handoff.

## Import and validation boundaries

The Codesign Gym docs build mocks JAX, Warp and MuJoCo (`autodoc_mock_imports` in `docs/conf.py`); the engine API
pages import only the lazy engine factories. Authored tables describe shapes, units and forwarded factory keywords
that a `**kwargs` signature cannot explain. A successful build checks the documentation structure and the selected
imports; it does not check tutorial numerics, optional backend interfaces or remote web links.

Do not add a previously generated HTML tree or duplicate API stubs to source
inventories. Update the current guides and migration map after module moves;
retain historical reports' original paths and append supersession notices only
where a reader could mistake them for current instructions.
