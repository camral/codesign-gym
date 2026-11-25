# Soft Walker

Ids: `SoftWalker-<Model>`, `SoftWalker-<Model>Stairs`, `SoftWalker-<Model>Gaps` (15 ids). Engine: native Warp soft-body simulator.

A soft robot built from a grid of material cells learns to walk forward.

- **Design:** one flat vector giving, for each cell, its occupancy, Young's modulus (5–100 kPa), density (800–1250 kg/m³), actuator strength, fibre direction, actuator kind (none, directional, isotropic), channel and Poisson's ratio (0.25–0.45).
- **Control:** the actuation channels over time.
- **Reward:** forward centre-of-mass travel over body length, minus positive actuator work over a work scale (0.4 J in 2D, 4 J in 3D). The return range is not yet characterised.
- **Design changes in `step`:** not supported; the body is rebuilt at each reset.

## Models

| `<Model>` | Physics | Grid | Channels |
|---|---|---|---|
| `Spring2D` | 2D spring lattice | 8x6 | 4 |
| `MPM2D` | 2D material point method | 8x6 | 4 |
| `MPM3D` | 3D material point method | 8x6x6 | 16 |
| `Beam3D` | 3D Timoshenko beam lattice | 8x6x6 | 16 |
| `HexFEM3D` | 3D hexahedral finite elements (neo-Hookean) | 8x6x6 | 16 |

Cells are 2 cm.

## Terrain

| Suffix | Terrain |
|---|---|
| (none) | Flat ground. |
| `Stairs` | Four 7.6 cm treads, each 6 mm higher than the last, starting about 8 cm ahead of the body. |
| `Gaps` | 5 cm platforms over 5 cm-deep pits, with gaps of 4, 20, 2 and 8 cm. The 20 cm gap is wider than the body. |

## Design rules

- A cell is solid when occupancy ≥ 0.5; occupancy has no effect on mass, stiffness or actuation.
- `kind` and `channel` must be integers, and directional actuators in solid cells need a fibre direction. In an empty cell the model decides: MPM2D and MPM3D accept a directional actuator without a fibre, the other models reject it.
- The body must be non-empty and face-connected (cells joined only at corners or edges don't count).
- `validate_embodiment` builds the body, so each model's own rules apply. A valid design can still fail at reset if it doesn't settle under gravity.
- The base design is a solid block. In the MPM models it can't walk, since every cell is on channel 0 with a vertical fibre.
- The spring lattice carries the Poisson field but has no continuum Poisson control.

## Task

- Control step: 0.01 s (40 physics steps).
- Settling at reset: 0.2 s with zero command.
- Episode: 240 steps (2.4 s).
- Observation: the simulator's full state. Its size is fixed because empty cells keep their massless nodes or particles.

## Device

The env runs in float64 on CUDA when Warp finds a GPU. Options: `dtype`, `device`, `execution` and `physics_steps_per_action`. On a CPU, a reset takes minutes and a step takes seconds.

## Render

`render_mode="human"` opens the live Newton viewer. With `render_mode="rgb_array"`, `render()` returns a 3D view (1280x800) that tracks the body. Without an OpenGL context it warns once and falls back to a 2D side view (720x420); for headless 3D frames, install EGL (see [README.md](README.md)).

## Engine

The engine is Andrew Spielberg's soft-body simulator in [`engine/`](engine/). This env uses only its native Warp backend and never imports JAX. Engine docs are in [README.md](README.md) and [`docs/`](docs/).
