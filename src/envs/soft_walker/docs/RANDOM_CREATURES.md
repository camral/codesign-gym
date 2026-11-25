# Random creature designs

`scripts/random_creatures.py` demonstrates all four general-purpose design
representations in both 2D and 3D. It saves actual decoder parameters, decoded
fields, seeds, rejection reports and hashes. These are structured random designs,
not optimized walkers. No silhouette is repaired after decoding.

## Generate

Run with the project's JAX environment, from the project root:

```bash
python scripts/random_creatures.py generate --dim 2 --count 6 --seed 7 --out runs/random_creatures_2d
python scripts/random_creatures.py generate --dim 3 --count 3 --seed 7 --out runs/random_creatures_3d
```

Use fresh output paths. Defaults sample primitives, direct grids, B-splines and
CPPNs; `--representations` can restrict the list. `--count` is per representation.
The CLI configures CPU generation and the requested float32/float64 precision.
`--grid 12 10` (2D) or `--grid 6 4 4` (3D) sets the mechanical reference lattice;
`--cell-size .02` sets physical size in metres. Each axis requires at least three
cells. The exported display fields use twice as many samples per axis, matching
the default MPM particle sampling.

| Representation | Random prior used by this demo |
|---|---|
| Primitives | Ellipsoid torso and overlapping random capsule/box appendages |
| Direct grid | Branching walks painted into native grid coefficients |
| B-spline | Smooth overlapping lobes in the native spline coefficient grid |
| CPPN | Random sigmoid gates and lobe connections around a radial body prior |

The priors deliberately differ. They demonstrate reproducible variety, not
uniform sampling or a fair comparison of representation expressiveness. CPPN
examples are broad, rounded bodies; the other priors tend to be smaller or more
branched. All colors/properties are decoded from the saved parameters. This
sampler uses directional actuation, with varied strengths, directions and
channel routing. Other actuator types remain available in the underlying API.

Every accepted sample passes embedding bounds, finite-property checks, nontrivial
occupancy, actuation, face connectivity and local nonmanifold checks at both the
coarse mechanical cells and doubled MPM sampling sites. Bounded rejection reports
record every rejection reason and the accepted child seed. A design may still
fail a longer dynamics run; this is not a locomotion certificate.

Each candidate contains:

- `design.json`: portable decoder descriptor and parameters, loadable with
  `envs.soft_walker.engine.serialization.load_design`.
- `fields.npz`: normalized coordinates, physical envelope, grid shape, raw
  decoder scores and occupancy/material/actuation fields.
- `generation.json`: seed, prior, attempts, validation and exact serialization
  roundtrip result.

`manifest.json` inventories all candidates and failures. Repeated occupancy masks
are explicitly flagged; none occurred in the supplied 36-design example.

## Render with Newton

The renderer uses the installed native Newton Python environment independently
of the WSL/JAX environment. In PowerShell:

```powershell
$env:WARP_CACHE_PATH = Join-Path (Get-Location) 'runs/warp-cache'
& 'C:\Users\Andy\Documents\ResearchCode\newton\.venv\Scripts\python.exe' viewer/render_morphologies.py --input runs/random_creatures_2d
& 'C:\Users\Andy\Documents\ResearchCode\newton\.venv\Scripts\python.exe' viewer/render_morphologies.py --input runs/random_creatures_3d
```

Open `rendered/index.html` within either run. It offers material/actuator toggles,
individual previews, contact sheets and links to the reusable designs. Material
colors show Young's modulus using one shared scale per gallery. Actuator colors
show channel identity; gray means passive. Camera and physical scale are shared.
All surfaces are actual occupied sampling-cell faces. They are **not** solver
elements. The thickness of 2D pieces is cosmetic; generation does not run dynamics.

## Reuse the same design with different mechanics

```bash
python scripts/random_creatures.py replay --design runs/random_creatures_2d/primitives_00_s7/design.json --drivers spring mpm2 --execution legacy --device cpu --steps 3 --out runs/random_creature_swap_2d
python scripts/random_creatures.py replay --design runs/random_creatures_3d/primitives_00_s7/design.json --drivers mpm3 beam fem --execution legacy --device cpu --steps 3 --out runs/random_creature_swap_3d
```

Each driver gets the same stored design, physical envelope, terrain and modest
external action sequence. The continuum/beam backends consume the design's spatial
Poisson field; new random designs vary it alongside the other material fields.
An explicit legacy global override is preserved for body drivers. Spring carries
the field but has no independent continuum Poisson parameter. See
[spatial material programming](SPATIAL_MATERIALS_AND_TERRAIN.md).
Each driver runs in a separate process
with a configurable timeout. Outputs include configuration, reset/step reports,
actual sampled fields, mechanical connectivity, trajectory and final scalar loss.
Material values in the instantiated model are compared against the decoder.

| Driver | Dimension | Instantiation |
|---|---|---|
| `spring` | 2D | Cross-braced spring lattice and nodal masses |
| `mpm2` / `mpm3` | 2D / 3D | Material particles plus computational grid |
| `beam` | 3D | Voxel-centred masses, rotations and beam links |
| `fem` | 3D | Structured Hex8 elements and nodal masses |

The conceptual pipeline is:

```text
design parameters + coordinates
    -> decoder -> occupancy, material and actuator fields
    -> backend sampling/assembly -> mechanical model
    -> dynamics driven by controller actions and terrain
    -> trajectory/metrics -> loss
```

Fields need not be an explicit intermediate grid: CPPNs and primitives can be
queried directly at backend sample coordinates. Material and actuator properties
are part of those fields. Current backends use fixed reference lattices, not an
arbitrary unstructured remesher. Sampling can change thin features and total mass.
Backend contact, constitutive approximations and actuation maps differ; switching
backend preserves the design interface, not identical trajectories or calibrated
physics. Controllers using backend-specific observations need adaptation.
Discrete hard occupancy/channel choices also limit smooth morphology gradients.

## Validation supplied with the demo

The focused generator suite passed 20 tests in float64 and 20 in float32:
determinism, distinct seeds, parameter sensitivity, both dimensions, all four
representations, portable roundtrips and bounded rejection. All 36 generated
designs roundtripped with identical fields. The five CPU legacy integrations
passed three steps each at dt=0.00025 seconds, for 0.00075 seconds of physical time.
Those checks establish wiring and finite output, not useful gait behavior or
physical equivalence. No GPU solver changes are part of this demo.
