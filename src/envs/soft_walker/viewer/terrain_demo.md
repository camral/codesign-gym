# Terrain Studio

Open `viewer\terrain.cmd` from Windows Explorer, or run it from the project directory:

```powershell
viewer\terrain.cmd -Preset stairs -Dim 2
```

This uses the locally installed native NVIDIA Newton viewer. It edits analytical
terrain geometry without running a physics solver. Drag to orbit, scroll to zoom,
and use **Reset view** to fit the entire course. The orange ruler is the robot's
16 cm length reference; it is not a simulated robot. The grid is 5 cm.

Choose flat, uphill, downhill, cross slope, low blocks, rough blocks, stairs, or
hurdles. Controls show degrees and millimeters; saved JSON uses meters. Cross
slope requires 3D. In 2D, displayed obstacle depth is cosmetic; in 3D it follows
the physical boxes. Roughness comes from a reproducible seed, and every recipe
stores the resulting boxes explicitly.

**Save recipe + terrain** writes `recipe.json` and `terrain.json` under
`runs/terrain_demo/custom`. Each subsequent save gets a fresh numbered directory.
The simulator accepts the recipe through `scripts/scenarios.py replay
--terrain-file <recipe.json> --execution legacy`, using a driver with the same
dimension. The editor does not automatically launch simulations.

Render all eight presets as hidden-window PNGs, with downloadable JSON and an
HTML gallery:

```powershell
viewer\terrain.cmd -Gallery runs/terrain_demo/gallery
```

The gallery path must be new. Its `index.html` is a static browser gallery;
`overview.png` shows all eight presets. To exercise the save operation and close
the editor after three frames:

```powershell
viewer\terrain.cmd -Preset stairs -Save -Hidden -Frames 3 -Out runs/terrain_demo/save_smoke
```

`-Python` overrides the existing Newton Python executable. By default it uses
`Documents/ResearchCode/newton/.venv/Scripts/python.exe`. The launcher keeps Warp's
graphics cache within this project and restores the previous cache environment
afterward.
