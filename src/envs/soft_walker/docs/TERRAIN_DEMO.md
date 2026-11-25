# Create and replay terrain

Open the terrain editor from the project directory in PowerShell:

```powershell
.\viewer\terrain.cmd
```

The editor uses the installed NVIDIA Newton renderer. Choose a preset, change
its metric dimensions, orbit the view, and save the terrain. The preview shows
the entire course and a robot-size reference. It is a geometry editor; that
reference is not a simulated robot trajectory.

The eight starting points are flat, uphill, downhill, cross slope, low blocks,
rough blocks, stairs, and hurdles. Cross slope requires 3D. Controls cover
slope angle, obstacle count, starting position, spacing, width, height, corner
rounding, and the seed for roughness. For stairs, height means the rise per
tread. The editor uses centimetres/millimetres where labeled; saved geometry
uses metres, with x forward and y up.

```powershell
.\viewer\terrain.cmd -Preset stairs -Out runs/terrain_demo/custom
.\viewer\terrain.cmd -Gallery runs/my_terrain_gallery
```

Saves go into fresh numbered directories rather than replacing earlier work.
Each save contains `recipe.json`, the editable and self-contained recipe, and
`terrain.json`, its resolved contact geometry. Gallery export also produces
eight PNG previews, `overview.png`, an HTML index, and rendering provenance.
Open the gallery's `index.html` directly in a browser; no server is needed.

The editor runs in the existing Windows Newton environment. Simulation commands
run in the existing WSL project virtualenv. `-Python` selects another compatible
Newton Python installation without changing the simulation environment.

## Use a saved design in the simulator

From the project directory in WSL, with `.venv` activated:

```bash
python scripts/scenarios.py replay \
  --terrain-file examples/terrain_demo/stairs.recipe.json \
  --driver mpm2 --tier smoke --dtype float64 \
  --device cpu --execution legacy --api-check \
  --out runs/my_stairs_smoke
```

Replace the recipe path with the editor's saved `recipe.json`. The custom recipe
is embedded in the output's `case.json`, and the actual terrain used by the
simulator is saved as `terrain.json`. A later replay can use that saved case
without access to the original recipe file. A 3D recipe requires a 3D driver,
such as `mpm3`; mismatches are rejected. Select either a custom terrain file or
`--terrain PRESET`, rather than both.

The `smoke` tier advances one short interval and checks initialization and API
equivalence. It does not demonstrate walking over obstacles. For an ordinary
longer simulation, explicitly select `--tier coarse --cycles 1` and retain
`--execution legacy`.

The supplied two-case legacy check can be repeated with:

```bash
python scripts/run_scenario_suite.py \
  --plan examples/terrain_demo/smoke_plan.json --mode smoke \
  --device cpu --dtype float64 --execution legacy \
  --out runs/my_terrain_smoke
```

The suite runner now accepts and forwards `--execution`; its default is legacy.
The individual scenario CLI retains its existing default, so the examples
select legacy explicitly.

## Create terrain in Python

```python
import json
from pathlib import Path
from envs.soft_walker.presets.terrain.editor_courses import make_recipe

recipe = make_recipe(
    "rough_blocks", dim=2, seed=23,
    count=6, start=.26, spacing=.10, width=.055, height=.018,
)
Path("my_terrain.recipe.json").write_text(json.dumps(recipe, indent=2))
```

`python scripts/create_terrain_demo.py --out runs/my_terrain_recipes` generates
all eight examples plus preview and smoke plans without running physics.

The contact model supports a plane or sloped plane, and a flat ground plane
with a union of up to 32 rounded, axis-aligned boxes. Recipes freeze box geometry
and slope values, so changing the environment's reset seed cannot change the
saved course. Physics and rendering resolve the same recipe. This editor does
not add pits, arbitrary mesh terrain, or deformable ground.
