"""Presets for the TruckUnload co-design environment.

Each preset is a kwargs dict for :class:`codesign_env.TruckUnloadCoDesignEnv`,
selected by ``setup=`` in ``registration.build_vec``. Both put every robot -- a
SCARA loader and SCARA stackers -- on a mobile x/y/yaw base, and co-design the
same thing: the length of each robot's column and two arm links, every robot on
its own entries.

* ``single`` -- the tested one-stacker line: the loader empties a 2x2x1
                container onto the belt, and one stacker shelves the red boxes.
* ``multi``  -- two stackers with different jobs, so their best arms differ: the
                red stacker shelves its boxes, the blue one sets them down
                anywhere on a flat 3x3 area. A 2x2x2 container, 4 boxes of each
                colour.
"""

from __future__ import annotations

_COMMON = dict(
    loader_xml="scara_loader_arm.xml", loader_base_xml="base_xy_loader.xml",
    stacker_xml="scara_stacker_arm.xml", stacker_base_xml="base_xy_stacker.xml",
    scene_xml="scene_truck_unload.xml", init_key=("init_pos", "init_pos2"),
    # From a stability sweep (random designs x random actions, 0 to 4 m): no
    # world diverges at any value, but past 1.0 m impacts fling the arms faster
    # than their actuators drive them and contacts penetrate 40+ mm. At 1.0 a
    # stacker arm link grows from 0.1 to 1.1 m; stacker_field_reach widens the
    # stackers' fields past the belt so every design still spawns inside its
    # field (worst corner design ~0.27 m inside).
    max_offset=1.0, min_link_length=0.02, min_length_pad=0.02,
    reach_weight=0.25, carry_weight=0.25, ungrasp_weight=0.25,
    ungrasp_threshold=0.1, goal_threshold=0.05, goal_dwell_steps=10,
    loader_goal_return=5.0, stacker_goal_return=5.0,
    time_limit_s=120.0, inactivity_limit_s=15.0, truncate_at_time_limit=False,
    terminate_out_of_field=True, platform_walls=False,
    container_opening="all", baseplate_dims=(2.0, 1.0), ramp_tilt_degs=20.0,
    ramp_lip=0.0, ramp_x_offset=0.8, container_z_offset=0.45, z_playing_field=0.45,
    belt_velocity=0.2, divider_dims=(3.0, 3.0), wall_thickness=0.025,
    box_size=0.1, box_clearance=0.02, box_mass=0.5, field_clearance=0.1,
    stacker_field_reach=1.2,
    max_rel_dist=1.75, vel_scale=10.0, time_in_obs=True, timestep=0.005,
    phys_substeps=8, camera_mode="fixed",
)

SINGLE = dict(
    _COMMON,
    n_stackers=1, stacker_targets=("shelf",), stacker_colors=("red",),
    box_array=(2, 2, 1),
)

# Twice the container and an extra stacker: allow substantially more time for
# the longer loader sequence and downstream deliveries.
MULTI = dict(
    _COMMON,
    n_stackers=2, stacker_targets=("shelf", "floor"), stacker_colors=("red", "blue"),
    floor_area_n=3, box_array=(2, 2, 2), time_limit_s=300.0,
)

PRESETS = {"single": SINGLE, "multi": MULTI}
