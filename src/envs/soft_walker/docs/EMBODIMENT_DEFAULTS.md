# Embodiment contracts and inherited defaults

The decoder and registered wrapper share a material envelope, but their initial
bodies and numerical safeguards have different purposes. The source separates
those responsibilities without changing existing values or random draw order.

| Owner | Contract and units | Provenance and limits |
| --- | --- | --- |
| [`contracts/fields.py`](../contracts/fields.py) | Host NumPy arrays; Young modulus 5–100 kPa, density 800–1250 kg/m³, Poisson ratio 0.25–0.45 with default 0.35 | Existing semantic-field and wrapper envelope. These are search/validation bounds, not universal limits of elasticity. |
| [`engine/embodiments/constants.py`](../engine/embodiments/constants.py) | Dimensionless score limits, normalized-coordinate primitive ranges, material palette in Pa, separate norm and squared-norm safeguards | Inherited decoder priors and numerical guards. No original empirical calibration record is available. Different guards remain separate even when their numeric values match. |
| [`engine/embodiments/priors.py`](../engine/embodiments/priors.py) | Sampling distributions, score weights and noise scales in normalized material coordinates; angular frequencies in radians | Hand-chosen priors from the existing random-creature generator, not fitted morphology distributions. Naming the parameters preserves the fixed-seed sequence and candidate scoring. |
| [`presets/embodiments/chamber_profile.py`](../presets/embodiments/chamber_profile.py) | Normalized torso/leg/chamber geometry, CPPN node roles and seed coefficients; material priors in SI units | Existing two/four-leg construction, not generic decoder limits. Seed calibration was not recorded. Shared material bounds and numerical guards still come from their generic contract owners. |
| [`engine/warp_native/solid_builders.py`](../engine/warp_native/solid_builders.py) | Regular-body dimensions, material defaults and family-specific contact mappings | Preserves native builder defaults and assembly scaling; similarly named spring and continuum contact parameters are not assumed to have the same units. |
| [`config.py`](../config.py) (presets) | Physical dimensions in metres, modulus in Pa, density in kg/m³, work in J, timing in seconds, routing/grid capacities as counts | Registered coarse locomotion preset and original native builder defaults. Comments distinguish standard gravity, task choices, and inherited contact/reward scales whose tuning history is unknown. |
| [`config.py`](../config.py) (terrain) | Stair/platform/gap geometry in metres | Existing demonstration terrain recipe. Overlap and padding are explicit; these presets are examples rather than terrain constraints. |
| [`config.py`](../config.py) (builders) | Assembly mappings, background-grid padding counts and the historical spring contact scale | Existing resolved-model mappings. The eight-cell reference used for forward MPM capacity is intentionally separate from a customizable body grid. |

`FieldSpec` can narrow or otherwise customize supported material settings exactly
as before. Sharing its default envelope with the host wrapper does not remove
those instance-specific settings. The task seed's modulus is also independent of
the decoder seed and palette: sharing a number just because it happens to match
would obscure a distinct configuration choice.

The fixed field offsets are serialization layout, not tuning parameters. Semantic
decoder rows contain four scalar scores, a direction vector, three material
scores, three actuator-kind scores, routing scores and a Poisson score. The Gym
embodiment remains a C-order flat vector with `7 + dimension` values per cell.
The `0.5` occupancy cutoff retains its inclusive comparison. Index offsets,
Boolean masks, polynomial coefficients and the two cells per axis in a topology
neighborhood remain close to the equations they implement.

`engine.embodiments.decoders` is the canonical implementation module. Its classes
now use their defining module names. Legacy decoder import aliases and the
registered wrapper's `config` forwarding facade were removed in the
[breaking cleanup](BREAKING_CLEANUP_MIGRATION.md). Customize the live owner modules
listed above; construction, validation and rendering read the same shared presets.
Live engine settings follow their responsibility: spring configuration is in
`engine.physics.spring_config`, body-task configuration in `engine.drivers.task`,
and native model configurations remain alongside their native implementations.

This audit covers decoder/field interpretation, morphology priors, registered
embodiments and task assembly. Solver tolerances, constitutive regularization,
integration coefficients, and demonstration controller seeds remain owned and
documented by their respective engine, preset or experiment modules; they are not merged
into a single global constants file. The relocation does not retune physics or
claim validation outside the recorded backend/device tests.

## Experiment and display policies

The release follow-up names additional choices at their actual owners:

- `experiments.locomotion.policies` owns flight/walking qualification, geometric
  assessment thresholds and distinct unit-aware guards. These are inherited
  experimental criteria, not universal evidence that a robot walks or flies.
- `experiments.forward_policy` owns implicit-body optimization weights and the
  metadata generated from those same weights. Checkpoint-continuity tolerances
  describe that experiment's acceptance policy, not the physics solver.
- `terrain.recipes` owns typed recipe/record validation and fixed-box layout;
  `presets.terrain.editor_courses` and `scenario_courses` retain their distinct
  seeded generators, dimensions and random draw order.
- Viewer recording, gallery camera and preview-style policies remain local to
  presentation. Display tolerances, colors and cosmetic geometry do not change
  physical collision or material definitions.

The naming records units and purpose. Where the original tuning rationale is
unknown, the source says so rather than attributing the value to a nonexistent
calibration. Arithmetic coefficients and topology entries remain close to their
equations when a new global name would not clarify the contract.

The subsequent [release-readiness review](RELEASE_READINESS_REVIEW.md) inventories
remaining literals across the whole Python implementation. It identifies shared
layout descriptors and validation policies still worth extracting; it does not
claim that this earlier, bounded constants pass removed every unexplained value.
