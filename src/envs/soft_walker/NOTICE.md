# Provenance

This source package implements the project's supplied co-design API and M1 plan. Its numerical kernels and evidence scripts are written for this implementation, except for the attributed Warp GMRES adaptation described below. It does not bundle dependency wheels or complete third-party libraries.

The CPPN architecture fixture is based on the documented EvoGym NEAT configuration and NEAT-Python sigmoid convention. Its parameters were generated anew for reproducible testing; it is not a trained robot copied from a paper. Source references are recorded in the fixture and README. Spring-model similarities to EvoGym are architectural precedents, not an assertion that this implements EvoGym's exact mechanical constants, solver, or strain-limiting rules.

Third-party dependencies retain their respective licenses. The project owner should select the repository's release license; no third-party project ownership or endorsement is implied here.

## Warp GMRES capture adapter

`engine/warp_native/krylov.py` adapts the GMRES execution code from
NVIDIA Warp 1.16.0, `warp/_src/optim/linear.py` (Copyright 2024 NVIDIA CORPORATION
& AFFILIATES), under Apache-2.0. The modified adapter allocates loop-control
storage before graph capture so the solver can execute inside conditional CUDA
graphs. It retains upstream numerical operations and delegates the CPU path to
the installed Warp implementation. The copyright and SPDX notice are retained
in the file; the license is included at `third_party/WARP_LICENSE.md`.

## M2 numerical precedents

The native fixed-step Kvaerno3 implementation uses the numerical tableau
published by Anne Kværnø and encoded in Diffrax 0.7.2. The tableau is checked
against that pinned dependency; the stage driver, matrix-free residual solves,
and explicit transpose solves were implemented separately here. No Diffrax
monkey patch, whole solver source, or bundled library is included.

- Anne Kværnø, *Singly Diagonally Implicit Runge–Kutta Methods with an Explicit First Stage*, BIT Numerical Mathematics 44 (2004), 489–502.
- Diffrax coefficient reference: https://github.com/patrick-kidger/diffrax/blob/v0.7.2/diffrax/_solver/kvaerno3.py
- Diffrax license: https://github.com/patrick-kidger/diffrax/blob/v0.7.2/LICENSE (Apache-2.0).
- APIC transfer formulation: Jiang et al., https://arxiv.org/abs/1603.06188 .

The Hex8, material, reference-quadrature and evidence kernels are independently
written implementations of the equations documented in docs/M2_MODEL.md, not
copies of an existing simulator. No continuum-convergence or third-party
endorsement is implied by sharing a constitutive law or quadrature convention.
