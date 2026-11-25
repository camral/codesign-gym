# Newton trajectory viewer

To create or edit terrain, open `viewer\terrain.cmd`. The separate
[Terrain Studio](terrain_demo.md) provides eight presets, metric controls,
recipe saving, and a PNG gallery. Saved recipes can be used by the legacy
scenario simulator; see [terrain workflow](../docs/TERRAIN_DEMO.md).

A studio view of saved simulation trajectories using NVIDIA Newton's actual
OpenGL renderer: solid surfaces, lighting, shadows, a metric ground grid, and
interactive playback. The original diagnostic GIF renderer remains available.

From the codesign-gym repository root in PowerShell, enter the soft-walker
package directory first. All `viewer/` and `runs/` paths below are relative to
that directory:

```powershell
cd src/envs/soft_walker
.\viewer\open.cmd
```

This opens the final six-cycle replay after 30 total Adam updates. To select a
different completed run or export a GIF, poster PNG, and provenance JSON:

```powershell
.\viewer\open.cmd -Run runs/mpm2_candidate
.\viewer\open.cmd -Run runs/mpm2_adam_to30_six_cycles -Gif runs/my_newton_export/replay.gif
```

The launcher uses the existing native Windows Newton environment at
`Documents/ResearchCode/newton/.venv/Scripts/python.exe`. Use `-Python` to select
another Newton environment. It places Warp's compilation cache inside this
project and restores the calling shell's cache setting afterward. It does not
change the WSL simulation environment or rerun physics.
The CMD wrapper permits this local PowerShell script in its own child process;
it does not change Windows execution policy settings.

The viewer provides play/pause, frame scrubbing, speed, looping, and a fixed
camera. Disable **Fixed camera** to use Newton's camera controls. **Reset view**
returns to a camera that includes the full recorded trajectory. Colors identify
the torso and individual legs; they do not show actuator values or stress.

**Particles** shows saved active MPM particle centers. **Elements** shows the
saved spring links (including diagonal springs), beam links, or FEM boundary
element edges. FEM is labeled **Surface elements**, since these recordings do
not contain interior element connectivity. Element nodes can also be shown.
The first time you enable a detail view, the solid surface is hidden so that
the detail is visible; you can then toggle the surface independently. Particle
and node marker sizes are cosmetic, not physical particle volumes.

These views also work from the launcher:

```powershell
.\viewer\open.cmd -Particles -NoSurface
.\viewer\open.cmd -Particles -NoSurface -Gif runs/my_particle_export/replay.gif
```

Use `-Elements -NoSurface` with `-Run` pointing to a spring, beam, or FEM recording.

GIF exports use a hidden graphics window. Existing output files are protected
against overwrite. `-Frames 8` selects eight saved samples across the entire
recording for a quick export; omit it for the full movie. Export timing follows
the selected physical sample times, including when samples are skipped.

For direct use in a Newton environment:

```text
python viewer/view_newton.py --run runs/mpm2_adam_to30_six_cycles
python viewer/view_newton.py --run runs/mpm2_adam_to30_six_cycles --gif runs/my_newton_export/replay.gif
```

The standalone viewer needs NumPy, Pillow, Newton, Warp, pyglet, and imgui-bundle.
It was exercised with the locally installed Newton 0.2.0 and its existing
dependencies. This installation uses CUDA/OpenGL interoperability for captures.
It reads saved `trajectory.npz` and optional topology, terrain, protocol/case,
and assessment files; it does not import JAX or `envs.soft_walker.engine`.

## Display geometry and measurement limits

2D simulations appear as thin pieces in a 3D scene. Their thickness is cosmetic.
Spring cells and FEM boundary faces use saved connectivity. MPM surfaces connect
adjacent active reference particles; this is a particle-center display surface,
not the physical material boundary. The saved MPM files do not contain enough
information to reconstruct exact deformed particle boundaries. Beam tubes follow
saved edge endpoints, with a cosmetic radius.

Every displayed center/vertex follows a saved physical sample. There is no time
interpolation or smoothing. Invalid tails are excluded and labeled as failure
previews. The display axis mapping and scale are recorded in each export's
provenance file, together with input hashes, frame indices, physical times,
package versions, and the recorded gait assessment.

Use the numerical contact/clearance measurements to decide whether liftoff
occurred. The cosmetic skin alone cannot establish contact or a successful gait.
