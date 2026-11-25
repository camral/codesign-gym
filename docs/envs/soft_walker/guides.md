# Engine guides and history

The engine's detailed guides are Markdown files in the source tree, maintained with the engine. Their links to source
files, tests and evidence work when you open them on GitHub or in a checkout.

## Current guides

- [Components and Gymnasium integration](../../../src/envs/soft_walker/docs/COMPONENTS_AND_GYMNASIUM.md): contracts, extension examples and the full pipeline.
- [Capability matrix](../../../src/envs/soft_walker/docs/CAPABILITIES.md): what is supported, the limits and the validation evidence.
- [Modularity matrix](../../../src/envs/soft_walker/docs/MODULARITY_MATRIX.md): the choices and which ones work together.
- [Integrator selection](../../../src/envs/soft_walker/docs/INTEGRATOR_SELECTION.md): the selectors and which one takes priority.
- [Action holding](../../../src/envs/soft_walker/docs/ACTION_HOLD.md) and [parallel environments](../../../src/envs/soft_walker/docs/PARALLEL_ENVIRONMENTS.md): command timing, resets and batching.
- [Spatial materials and terrain](../../../src/envs/soft_walker/docs/SPATIAL_MATERIALS_AND_TERRAIN.md).
- [Defaults and policy ownership](../../../src/envs/soft_walker/docs/EMBODIMENT_DEFAULTS.md): units, sources, and the physical, experimental and display policies.
- [Native callback contracts](../../../src/envs/soft_walker/docs/NATIVE_CALLBACKS.md) and [native embodiment gradients](../../../src/envs/soft_walker/docs/EMBODIMENT_NATIVE_GRADIENTS.md).
- [Code conventions](../../../src/envs/soft_walker/docs/CODE_CONVENTIONS.md).

## Migration and evidence

The migration guides describe the code at their own revision. Read the later notes before you use their imports.

- [Release organization](../../../src/envs/soft_walker/docs/RELEASE_ORGANIZATION.md): the current workflow and preset moves.
- [Shared-helper migration](../../../src/envs/soft_walker/docs/SHARED_HELPER_MIGRATION.md): the helpers moved into one module.
- [Release cleanup validation](../../../src/envs/soft_walker/docs/RELEASE_CLEANUP_VALIDATION.md): the checks, the performance scope and the open qualifications.
- [Breaking package cleanup](../../../src/envs/soft_walker/docs/BREAKING_CLEANUP_MIGRATION.md) and [responsibility consolidation](../../../src/envs/soft_walker/docs/RESPONSIBILITY_CLEANUP.md): earlier records.
- [Historical records](../../../src/envs/soft_walker/docs/HISTORY.md): the experiments and validation reports, kept apart from current guidance.

After those records, the registered task moved back to the flat Codesign Gym layout (`config.py`, `embodiment.py`,
`vector_env.py`, `render.py`), and the registered JAX backend was removed. The engine's own JAX backend is unchanged.
