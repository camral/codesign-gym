# Environment summary and construction-output audit

Audited 2026-09-28 using fresh processes for every registered preset and both factories. The 13 presets of BallCatcher, SolarCleaner, TruckUnload, RotateInHand and HandPickUp were audited again on 2026-10-27, on upstream mujoco-warp, including HandPickUp-SmallHammer, which was registered after the first audit. All 26 of those cases constructed, and each preset's single and vector summaries matched. HandPickUp-SquashBall was removed on 2026-10-29, when the hand envs moved to the updated hand models, and the four remaining hand presets were audited again: all 8 cases constructed and their summaries matched.

- 86 registered presets; 172 factory cases.
- 169 constructed and printed successfully; all 83 comparable single/vector summary pairs matched.
- Three expected single-env failures: NeroGrasp-All (10 objects), NeroGrasp-Small (5), NeroGrasp-Tiny (4). Their vector cases passed with valid batch sizes.
- All hand presets passed. The first audit used the pinned morphhand backend in an isolated source checkout, leaving the shared installation unchanged. The later ones used upstream mujoco-warp.
- No legacy Gym migration banners or Warp initialization banners remained in the final construction captures.
- SoftWalker summaries are compact and name each per-cell field; hand summaries abbreviate repeated bounds.
- All Pokémon defaults printed TeamDesignSpace(0 Pokemon, 0 indices). A separate configured-EV test verified a non-empty space in both factories.

Warnings retained: MuJoCo contact limitations, geometry minimum-length clamping, unsupported noslip solver, and NeroGrasp VHACD degenerate-part diagnostics. Matplotlib may report one-time font cache creation on a fresh cache.

Validation: 89 output/summary/device checks, 45 Microgrid behavior checks, four hand summary tests, and one configured Pokémon test passed. Logging restoration and warning preservation are covered, including constructor failures.

Run `python tests/audit_environment_output.py --output /tmp/codesign-audit` to regenerate per-case summaries and stdout/stderr captures. Optional backends, local server ports, and OS shared memory are required for complete coverage.

| Preset | make | make_vec | Summary type |
|---|---|---|---|
| Ant-v4 | ok | ok | EmbodimentSpace |
| BallCatcher-Arc | ok | ok | EmbodimentSpace |
| BallCatcher-Fielding | ok | ok | EmbodimentSpace |
| BallCatcher-Pitch | ok | ok | EmbodimentSpace |
| HalfCheetah-v4 | ok | ok | EmbodimentSpace |
| HandPickUp-Cube | ok | ok | EmbodimentSpace |
| HandPickUp-Hammer | ok | ok | EmbodimentSpace |
| HandPickUp-SmallHammer | ok | ok | EmbodimentSpace |
| Hopper-v4 | ok | ok | EmbodimentSpace |
| Humanoid-v4 | ok | ok | EmbodimentSpace |
| HumanoidStandup-v4 | ok | ok | EmbodimentSpace |
| LQStructure-LQG | ok | ok | Dict |
| LQStructure-LQR | ok | ok | Dict |
| LQStructure-QP | ok | ok | Dict |
| Microgrid-Campus | ok | ok | Dict |
| Microgrid-Forecast | ok | ok | Dict |
| Microgrid-OffGridCampus | ok | ok | Dict |
| Microgrid-OffGridSingle | ok | ok | Dict |
| Microgrid-Random | ok | ok | Dict |
| NeroGrasp-All | error | ok | MeshSpace |
| NeroGrasp-LockNut | ok | ok | MeshSpace |
| NeroGrasp-Scissors | ok | ok | MeshSpace |
| NeroGrasp-Screw | ok | ok | MeshSpace |
| NeroGrasp-ScrewLong | ok | ok | MeshSpace |
| NeroGrasp-SetScrew | ok | ok | MeshSpace |
| NeroGrasp-Small | error | ok | MeshSpace |
| NeroGrasp-Spatula | ok | ok | MeshSpace |
| NeroGrasp-Stick | ok | ok | MeshSpace |
| NeroGrasp-TNut | ok | ok | MeshSpace |
| NeroGrasp-Thin | ok | ok | MeshSpace |
| NeroGrasp-Tiny | error | ok | MeshSpace |
| NeroGrasp-Tweezers | ok | ok | MeshSpace |
| Network-Motorway | ok | ok | Dict |
| Network-Suburban | ok | ok | Dict |
| Network-Urban | ok | ok | Dict |
| Pokenv-Heuristic | ok | ok | TeamDesignSpace |
| Pokenv-LeagueG1 | ok | ok | TeamDesignSpace |
| Pokenv-LeagueG1H | ok | ok | TeamDesignSpace |
| Pokenv-LeagueG2 | ok | ok | TeamDesignSpace |
| Pokenv-LeagueG2H | ok | ok | TeamDesignSpace |
| Pokenv-LeagueG3 | ok | ok | TeamDesignSpace |
| Pokenv-LeagueG3H | ok | ok | TeamDesignSpace |
| Pokenv-LeagueG4 | ok | ok | TeamDesignSpace |
| Pokenv-LeagueG4H | ok | ok | TeamDesignSpace |
| Pokenv-LeagueG5 | ok | ok | TeamDesignSpace |
| Pokenv-LeagueG5H | ok | ok | TeamDesignSpace |
| Pokenv-LeagueG6 | ok | ok | TeamDesignSpace |
| Pokenv-LeagueG6H | ok | ok | TeamDesignSpace |
| Pokenv-LeagueG7 | ok | ok | TeamDesignSpace |
| Pokenv-LeagueG7H | ok | ok | TeamDesignSpace |
| Pokenv-LeagueG8 | ok | ok | TeamDesignSpace |
| Pokenv-LeagueG8H | ok | ok | TeamDesignSpace |
| Pokenv-LeagueG9 | ok | ok | TeamDesignSpace |
| Pokenv-LeagueG9H | ok | ok | TeamDesignSpace |
| Pokenv-LeagueRand | ok | ok | TeamDesignSpace |
| Pokenv-LeagueRandH | ok | ok | TeamDesignSpace |
| Pokenv-MaxPower | ok | ok | TeamDesignSpace |
| Pokenv-Random | ok | ok | TeamDesignSpace |
| Racing-Levine | ok | ok | Box |
| Racing-Spielberg | ok | ok | Box |
| RotateInHand-Cube | ok | ok | EmbodimentSpace |
| SoftWalker-Beam3D | ok | ok | CellEmbodimentSpace |
| SoftWalker-Beam3DGaps | ok | ok | CellEmbodimentSpace |
| SoftWalker-Beam3DStairs | ok | ok | CellEmbodimentSpace |
| SoftWalker-HexFEM3D | ok | ok | CellEmbodimentSpace |
| SoftWalker-HexFEM3DGaps | ok | ok | CellEmbodimentSpace |
| SoftWalker-HexFEM3DStairs | ok | ok | CellEmbodimentSpace |
| SoftWalker-MPM2D | ok | ok | CellEmbodimentSpace |
| SoftWalker-MPM2DGaps | ok | ok | CellEmbodimentSpace |
| SoftWalker-MPM2DStairs | ok | ok | CellEmbodimentSpace |
| SoftWalker-MPM3D | ok | ok | CellEmbodimentSpace |
| SoftWalker-MPM3DGaps | ok | ok | CellEmbodimentSpace |
| SoftWalker-MPM3DStairs | ok | ok | CellEmbodimentSpace |
| SoftWalker-Spring2D | ok | ok | CellEmbodimentSpace |
| SoftWalker-Spring2DGaps | ok | ok | CellEmbodimentSpace |
| SoftWalker-Spring2DStairs | ok | ok | CellEmbodimentSpace |
| SolarCleaner-Reach | ok | ok | Box |
| SolarCleaner-Stability | ok | ok | Box |
| SolarCleaner-Traverse | ok | ok | Box |
| Swimmer-v4 | ok | ok | EmbodimentSpace |
| TruckUnload-Multi | ok | ok | Box |
| TruckUnload-Single | ok | ok | Box |
| Walker2d-v4 | ok | ok | EmbodimentSpace |
| Warehouse-Congested | ok | ok | MultiBinary |
| Warehouse-Medium | ok | ok | MultiBinary |
| Warehouse-Small | ok | ok | MultiBinary |
