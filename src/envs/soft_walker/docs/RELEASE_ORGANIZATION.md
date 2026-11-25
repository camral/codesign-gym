# Experiment, preset and rendering organization

This release separates reusable simulation mechanisms from experiment workflows,
named robot/terrain presets and presentation. The public engine factories,
registered Gym interfaces, CLI flags, decoder descriptor IDs and array-only
checkpoint schemas remain the same. Imports of the implementation modules below
change; no forwarding shims are retained.

Paths in the table are relative to `envs.soft_walker`. Earlier paths in saved
evidence remain provenance and are not rewritten.

| Previous module or object | Current owner |
| --- | --- |
| `engine.locomotion.robots` | `presets.locomotion` |
| `engine.locomotion.controllers.periodic_action` | `shared.periodic_action` |
| `engine.locomotion.controllers.seed_controller` | `presets.gaits.seed_controller` |
| `engine.locomotion.artifacts` | `shared` |
| `engine.locomotion.optimize` | `shared` |
| `engine.locomotion.adam_experiment` | `shared` |
| `engine.locomotion.adam_state` | `shared` |
| `engine.locomotion.implicit_design` | `shared` |
| `engine.locomotion.rollout`, `engine.locomotion.low_bounce` | `experiments.locomotion.<same name>` |
| `engine.locomotion.rollout.region_tags` | `shared.region_tags` |
| `engine.locomotion.rollout.physical_state` | `engine.drivers.model_access.physical_state` |
| `engine.locomotion.rollout.positions`, `reference_positions` | `engine.drivers.model_access.task_positions`, `task_reference_positions` |
| `engine.locomotion.animation` | `viewer.trajectory_animation` |
| `engine.scenarios.catalog.Scenario` | `presets.scenarios.config.Scenario` |
| `engine.scenarios.catalog.suite` | `presets.scenarios.suites.suite` |
| `engine.scenarios.embodiments` | `presets.embodiments.chambers` |
| `engine.scenarios.chamber_profile` | `presets.embodiments.chamber_profile` |
| `engine.scenarios.build` construction helpers | `presets.scenarios.build` |
| `engine.scenarios.build.preflight`, `physical_budget` | `shared` |
| `engine.scenarios.controllers`, `objectives`, `rollout` | `experiments.scenarios.<same name>` |
| `engine.scenarios.pictures` | `viewer.scenario_pictures` |
| `engine.scenarios.terrain_recipes` schema and validation | `terrain.recipes` |
| `engine.scenarios.terrain_recipes.make_recipe`, `PRESETS`, `PRESET_DEFAULTS` | `presets.terrain.editor_courses` |
| `engine.scenarios.terrains` named courses | `presets.terrain.scenario_courses` |
| `engine.scenarios.terrains.height_at` | `viewer.terrain_geometry.height_at` |
| `engine.execution.bind_demo_episode` / `program.bind_demo_episode` | `experiments.locomotion.resident.bind_demo_episode` |
| `engine.execution.bind_scenario_episode` / `program.bind_scenario_episode` | `experiments.scenarios.resident.bind_scenario_episode` |

General decoders remain in `engine.embodiments`. The chamber decoder and legged
seed builder are presets because their topology, region assignments and
initialization assume a particular torso-and-legs construction. Shared Adam
updates/checkpoints, serialization, periodic control, reference-region
labels, path helpers, preflight and derivative checks now live in `shared.py`.
This is the single temporary helper owner requested for release preparation.
The two optimizer policies and two JSON serialization policies remain distinct.
JAX imports are local to the helpers that need them; importing the module is
host-only. Specialized scenario/locomotion episode adapters remain workflows.
The [shared-helper migration](SHARED_HELPER_MIGRATION.md) lists every relocated
name, including its more recent `experiments` or `diagnostics` owner.

`experiments.locomotion.policies` names the inherited flight/walking assessment
criteria, while `experiments.forward_policy` supplies implicit-objective weights
to both computation and recorded metadata. `presets.embodiments.chamber_profile`
names chamber geometry and CPPN seed choices. See the
[defaults and policy guide](EMBODIMENT_DEFAULTS.md) for units, provenance and
why these choices remain separate from generic material bounds and solver tolerances.

The editor and scenario terrain generators retain their separate random number
generators and dimensional defaults. `terrain.recipes` is the portable, host-only
schema owner; a shared terrain name does not imply byte-identical courses.
Specialized resident binders live with their objectives/controllers so generic
`engine.execution` consumes callbacks without importing applications.

The package now requires Pillow 10.1 or later. Scenario previews already used
`ImageFont.load_default(size=...)`, whose `size` argument was introduced in 10.1;
the previous `pillow>=9` declaration admitted versions that could not render
those previews. This corrects the dependency floor without changing image output
on an already-compatible installation.

`engine.action_timing` remains a shared cadence service for both backends.
Registered horizon rounding stays with the registered task. Rendering prepares
host geometry separately from drawing; cosmetic camera/style policies do not
become physical configuration.

For the full responsibility map, see [components](COMPONENTS_AND_GYMNASIUM.md).
For the optional generated API site and build boundaries, see
[documentation maintenance](DOCUMENTATION.md). Module moves do not constitute a
new physics/performance qualification; retain the validation limits recorded in
the capability matrix and dated evidence.
