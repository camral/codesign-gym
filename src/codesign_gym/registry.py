"""
Authors: Aviraj Newatia

Environment registry for codesign_gym.

Each entry maps an ``env_id`` to a lazy ``"module:function"`` import string for that
environment's ``build_vec`` builder, plus default keyword arguments. The string is
imported only when the id is requested, so heavy environments (e.g. MuJoCo-based ones)
do not load their dependencies until used.

Each ``build_vec`` owns its environment's native vectorization and returns an env that
exposes a ``base_embodiment`` attribute. Register a new environment by adding a row here.
"""

from __future__ import annotations

# env_id -> ("module:function", default kwargs)
REGISTRY: dict[str, tuple[str, dict]] = {
    "LQStructure-LQR": ("envs.lq_structure.registration:build_vec", {"setup": "lqr"}),
    "LQStructure-LQG": ("envs.lq_structure.registration:build_vec", {"setup": "lqg"}),
    "LQStructure-QP":  ("envs.lq_structure.registration:build_vec", {"setup": "qp"}),
    "HalfCheetah-v4": ("envs.half_cheetah.registration:build_vec", {}),
    "Swimmer-v4": ("envs.swimmer.registration:build_vec", {}),
    "Hopper-v4": ("envs.hopper.registration:build_vec", {}),
    "Walker2d-v4": ("envs.walker2d.registration:build_vec", {}),
    "Ant-v4": ("envs.ant.registration:build_vec", {}),
    "Humanoid-v4": ("envs.humanoid.registration:build_vec", {}),
    "HumanoidStandup-v4": ("envs.humanoid_standup.registration:build_vec", {}),
    "BallCatcher-Pitch":    ("envs.ball_catcher.registration:build_vec", {"setup": "pitch"}),
    "BallCatcher-Arc":      ("envs.ball_catcher.registration:build_vec", {"setup": "arc"}),
    "BallCatcher-Fielding": ("envs.ball_catcher.registration:build_vec", {"setup": "fielding"}),
    "Microgrid-Random":        ("envs.microgrid.registration:build_vec", {"setup": "random"}),
    "Microgrid-Forecast":      ("envs.microgrid.registration:build_vec", {"setup": "forecast"}),
    "Microgrid-Campus":        ("envs.microgrid.registration:build_vec", {"setup": "campus"}),
    "Microgrid-OffGridSingle":       ("envs.microgrid.registration:build_vec", {"setup": "offgrid"}),
    "Microgrid-OffGridCampus": ("envs.microgrid.registration:build_vec", {"setup": "offgridcampus"}),
    "Pokenv-Random":    ("envs.pokenv.registration:build_vec", {"opponent": "random"}),
    "Pokenv-MaxPower":  ("envs.pokenv.registration:build_vec", {"opponent": "max_base_power"}),
    "Pokenv-Heuristic": ("envs.pokenv.registration:build_vec", {"opponent": "simple_heuristics"}),
    "Network-Urban":   ("envs.mobile_env.registration:build_vec", {"setup": "dense_urban"}),
    "Network-Suburban":     ("envs.mobile_env.registration:build_vec", {"setup": "suburban"}),
    "Network-Motorway": ("envs.mobile_env.registration:build_vec", {"setup": "rural_highway"}),
    "Warehouse-Small":     ("envs.robot_warehouse.registration:build_vec", {"setup": "small"}),
    "Warehouse-Medium":    ("envs.robot_warehouse.registration:build_vec", {"setup": "medium"}),
    "Warehouse-Congested": ("envs.robot_warehouse.registration:build_vec", {"setup": "congested"}),
    "Racing-Spielberg": ("envs.f1tenth.registration:build_vec", {"setup": "spielberg"}),
    "Racing-Levine":    ("envs.f1tenth.registration:build_vec", {"setup": "levine"}),
    "SolarCleaner-Reach":     ("envs.panel_cleaner.registration:build_vec", {"setup": "reach"}),
    "SolarCleaner-Stability": ("envs.panel_cleaner.registration:build_vec", {"setup": "stability"}),
    "SolarCleaner-Traverse":  ("envs.panel_cleaner.registration:build_vec", {"setup": "traverse"}),
    "RotateInHand-Cube": ("envs.rotate_in_hand.registration:build_vec", {"setup": "cube"}),
    "HandPickUp-Cube":       ("envs.hand_pick_up.registration:build_vec", {"setup": "cube"}),
    "HandPickUp-Hammer":     ("envs.hand_pick_up.registration:build_vec", {"setup": "hammer"}),
    "HandPickUp-SmallHammer": ("envs.hand_pick_up.registration:build_vec", {"setup": "small_hammer"}),
    "TruckUnload-Single": ("envs.truck_unload.registration:build_vec", {"setup": "single"}),
    "TruckUnload-Multi":  ("envs.truck_unload.registration:build_vec", {"setup": "multi"}),
    "SoftWalker-Spring2D": ("envs.soft_walker.registration:build_vec", {"driver": "spring"}),
    "SoftWalker-MPM2D": ("envs.soft_walker.registration:build_vec", {"driver": "mpm2"}),
    "SoftWalker-MPM3D": ("envs.soft_walker.registration:build_vec", {"driver": "mpm3"}),
    "SoftWalker-Beam3D": ("envs.soft_walker.registration:build_vec", {"driver": "beam"}),
    "SoftWalker-HexFEM3D": ("envs.soft_walker.registration:build_vec", {"driver": "fem"}),
}

for _name, _driver in (("Spring2D", "spring"), ("MPM2D", "mpm2"), ("MPM3D", "mpm3"), ("Beam3D", "beam"), ("HexFEM3D", "fem")):
    for _terrain in ("stairs", "gaps"):
        REGISTRY[f"SoftWalker-{_name}{_terrain.capitalize()}"] = ("envs.soft_walker.registration:build_vec", {"driver": _driver, "terrain": _terrain})

for _selector in ("all", "small", "tiny", "thin", "set_screw", "t_nut", "lock_nut", "screw", "scissors", "tweezers", "spatula", "stick", "screw_long"):
    _camel = "".join(_part.capitalize() for _part in _selector.split("_"))
    REGISTRY[f"NeroGrasp-{_camel}"] = ("envs.nerograsp.registration:build_vec", {"object_name": _selector})

for _tool in ("carabiner", "wire_cutters"):
    _camel = "".join(_part.capitalize() for _part in _tool.split("_"))
    REGISTRY[f"NeroGrasp-Tools-{_camel}"] = ("envs.nerograsp.registration:build_tools_vec", {"object_name": _tool})

for _generation in range(1, 10):
    REGISTRY[f"Pokenv-LeagueG{_generation}"] = ("envs.pokenv.registration:build_vec", {"opponent": "league", "generation": _generation, "difficulty": 0})
    REGISTRY[f"Pokenv-LeagueG{_generation}H"] = ("envs.pokenv.registration:build_vec", {"opponent": "league", "generation": _generation, "difficulty": 1})

REGISTRY["Pokenv-LeagueRand"] = ("envs.pokenv.registration:build_vec", {"opponent": "league", "random_league": True, "difficulty": 0})
REGISTRY["Pokenv-LeagueRandH"] = ("envs.pokenv.registration:build_vec", {"opponent": "league", "random_league": True, "difficulty": 1})
