"""Presets for the RotateInHand co-design environment.

Each preset is a kwargs dict for :class:`codesign_env.RotateInHandCoDesignEnv`,
selected by ``setup=`` in ``registration.build_vec``, as with the other
multi-preset envs.

In ``cube``, the wrist-only hand (``hand_no_xyz.xml``), whose heightfield
pads (one embodiment entry per MJCF cell, subdivided 2x2 for the physics) and
link-offset slots are the embodiment, turns a box through an endless stream of
palm-frame goals, each ``goal_dist`` from where the last one was met. The shape
is set at reset only, so both deform windows are 0.
"""

from __future__ import annotations

CUBE = dict(
    hand_xml='hand_no_xyz.xml',
    object_xml='test_object_big',
    scene_xml='scene_rotate_in_hand.xml',
    init_key=('init_pos',),
    init_quat=(1.0, 0.0, 0.0, 0.0),
    randomise_init_quat=False,
    deform_hfield_during_rollout_s=0.0,
    deform_links_during_rollout_s=0.0,
    max_link_offset=0.015,
    hfield_offset_fraction=1.0,
    # each pad cell is a 2x2 patch of spline samples; one embodiment entry per cell
    grid_subd=2,
    spline_degree=2,
    # the finger-link pads taper to 0 at the joints over 2 samples
    relief_length=2,
    min_link_length=0.002,
    max_pose_return=3.0,
    max_goal_return=7.0,
    goal_dwell_steps=10,
    goal_rot_tol=0.1,
    # goal_pose=None draws a fresh goal every time one is met
    goal_pose=None,
    goal_frame='palm',
    goal_dist=1.0,
    force_mag=0.0,
    force_every=0,
    force_hold=0,
    force_viz_scale=0.02,
    goal_viz_scale=1.0,
    max_rel_dist=0.1,
    vel_scale=10.0,
    contact_steps=2,
    timestep=0.005,
    phys_substeps=8,
    goal_limit_s=7.5,
    total_limit_s=15.0,
    use_graph=True,
    check_finite=False,
    njmax=256,
    nconmax=64,
)

PRESETS = {
    "cube": CUBE,
}
