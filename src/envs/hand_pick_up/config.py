"""Presets for the HandPickUp co-design environment.

Each preset is a kwargs dict for :class:`codesign_env.HandPickUpCoDesignEnv`,
selected by ``setup=`` in ``registration.build_vec``, as with the other
multi-preset envs.

The hand hangs palm-down over a table. Its heightfield pads (one embodiment
entry per MJCF cell, subdivided 2x2 for the physics) and link-offset slots are
the embodiment. The shape is set at reset only, so both deform windows are 0.

* ``cube``: a box, turned through an endless stream of palm-frame goals 0.5 rad
  apart once lifted, by the wrist-only hand (``hand_no_xyz.xml``). The table
  gap is derived from the box.
* ``hammer``: the hand with a sliding wrist (``hand.xml``) lifts a hammer lying
  on the table and holds it at one world-fixed orientation *and* place --
  ``rel_pos``, given in the palm frame and fixed where it lies at the palm's
  reset pose -- for ``goal_dwell_steps``. Meeting it ends the episode. Once
  lifted off the table, the object is perturbed by random 3D torques while held.
* ``small_hammer``: the same, with a slimmer hammer: a 2-3 cm thick handle and
  a 2.5 cm square head (0.43 kg, against the hammer's 4 cm handle and 6 cm
  head, 0.45 kg).
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
    max_goal_return=6.0,
    max_ftip_return=1.0,
    goal_dwell_steps=20,
    goal_rot_tol=0.05,
    train_rel_pos=False,
    rel_pos=(0.0, 0.0, 0.03),
    goal_pos_tol=0.05,
    # goal_pose=None draws a fresh goal every time one is met
    goal_pose=None,
    goal_frame='palm',
    goal_dist=0.5,
    force_mag=0.0,
    force_every=0,
    force_hold=0,
    force_viz_scale=0.02,
    goal_viz_scale=0.3,
    max_rel_dist=0.1,
    vel_scale=10.0,
    termination_dist=0.4,
    table_gap=0.0,
    contact_steps=2,
    timestep=0.005,
    phys_substeps=8,
    goal_limit_s=6.0,
    total_limit_s=15.0,
    use_graph=True,
    check_finite=False,
    njmax=256,
    nconmax=64,
)

HAMMER = dict(
    hand_xml='hand.xml',
    object_xml='hammer',
    scene_xml='scene_rotate_in_hand.xml',
    init_key=('init_pos',),
    init_quat=(0.68296, -0.68296, 0.18311, -0.18311),
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
    max_goal_return=6.0,
    max_ftip_return=1.0,
    goal_dwell_steps=125,
    goal_rot_tol=0.05,
    train_rel_pos=True,
    rel_pos=(0.0, 0.0, -0.07),
    goal_pos_tol=0.05,
    goal_pose=(0.991, 0.0, 0.13, 0.0),
    goal_frame='world',
    goal_dist=0.5,
    # Random 3D perturbation torque [N*m], gated on liftoff (~3 N at 10 cm CoM lever arm)
    force_mag=0.3,
    force_every=20,
    force_hold=5,
    force_viz_scale=0.02,
    goal_viz_scale=0.3,
    max_rel_dist=0.1,
    vel_scale=10.0,
    termination_dist=0.4,
    table_gap=0.145,
    contact_steps=2,
    timestep=0.005,
    phys_substeps=8,
    goal_limit_s=8.0,
    total_limit_s=9.0,
    use_graph=True,
    check_finite=False,
    njmax=256,
    nconmax=64,
)

SMALL_HAMMER = dict(
    HAMMER,
    object_xml='small_hammer',
)

PRESETS = {
    "cube": CUBE,
    "hammer": HAMMER,
    "small_hammer": SMALL_HAMMER,
}
