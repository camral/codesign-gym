"""Presets for the PanelCleaner co-design environment.

Each preset is a kwargs dict for :class:`codesign_env.PanelCleanerCoDesignEnv`,
selected by ``setup=`` in ``registration.build_vec`` (the shared-builder
discriminator, as with the other multi-id envs). They pose three different
co-design challenges over the same robot -- an ant carrying a roller-tipped arm,
whose every leg and arm link deforms independently:

* ``reach``     -- arm-morphology dominant: tall, far panels, no disturbance.
* ``stability`` -- leg-morphology dominant: uneven terrain and a shove, while the
                   roller has to dwell on a cell.
* ``traverse``  -- careful locomotion: a tightly packed array whose plates are
                   solid obstacles, so a mis-placed body part ends the episode.

``field_pos``/``field_size`` are sized per preset to contain every cell the sweep
visits; a goal outside the field is refused at build time.
"""

from __future__ import annotations

# The episode ends when the base sinks below this fraction of the design's
# standing height, measured above the ground under it. Without it, the policy
# lets the lower legs fold flat and crawls on them: that posture sits the base
# at about 0.6-0.67 of standing height, while a robot on its feet stays above 0.85.
MIN_BASE_HEIGHT_FRAC = 0.75

# Arm-morphology dominant. A single row of tall panels set well away from the
# robot: the only way to reach the high, far cells is to lengthen and re-aim the
# arm links (the 24 arm offsets), so the legs barely matter and control is easy.
# No disturbance, generous budgets, a wider offset range so the arm can grow.
REACH = dict(
    panel_rows=1, panel_cols=3,
    first_panel_pos=[2.4, 0.0, 0.0],
    panel_length=1.3, panel_width=1.9, panel_height=2.0, panel_tilt_degs=20.0,
    panel_gap_x=1.0, panel_gap_y=0.35,
    # Cells a little larger than the roller (0.25 m): 1.3/4 = 0.33, 1.9/6 = 0.32.
    n_sensor_rows=4, n_sensor_cols=6,
    max_offset=0.15,
    midpoints=True,
    min_base_height_frac=MIN_BASE_HEIGHT_FRAC,
    # Spawn well back from the array; the extra approach gets a bigger walk budget.
    spawn_offset=1.5,
    next_cell_step_limit=150, next_panel_step_limit=700,
    collide_roller_only=True,
    field_pos=[2.4, 0.0, 1.2], field_size=[3.0, 3.8, 2.6],
)

# Leg-morphology dominant. Staying upright is the challenge: the ground rolls in
# smooth hills and valleys, a random per-step force jostles the legs to emulate
# rough footing, and a lateral shove hits the base on a schedule -- all while the
# roller has to sit on a cell for the dwell. Independent legs let the optimizer
# lengthen or splay individual legs to find footing and brace against the push.
STABILITY = dict(
    # A 2x2 scatter: panels spread apart in BOTH x and y, so the robot crosses
    # open ground -- smooth hills and valleys -- to get from one to the next.
    panel_rows=2, panel_cols=2,
    first_panel_pos=[1.8, 0.0, 0.0],
    panel_length=1.2, panel_width=1.6, panel_height=1.4, panel_tilt_degs=30.0,
    panel_gap_x=2.2, panel_gap_y=2.2,
    # Cells a little larger than the roller (0.25 m): 1.2/4 = 0.30, 1.6/5 = 0.32.
    n_sensor_rows=4, n_sensor_cols=5,
    panel_dwell_steps=15,
    max_offset=0.1,
    midpoints=True,
    min_base_height_frac=MIN_BASE_HEIGHT_FRAC,
    # Spawn well back from the array; the extra approach gets a bigger walk budget.
    spawn_offset=1.5,
    next_cell_step_limit=200, next_panel_step_limit=700,
    # Panels are solid and must not be struck: crossing the smooth hilly ground between
    # them, a firm body-panel contact ends the episode (no extra penalty, as in
    # Traverse -- the deterrent is forfeiting the rest of the sweep).
    collide_roller_only=False,
    terminate_on_body_contact=True, body_contact_penalty=0.0,
    body_contact_depth=0.005,
    # The shove: a lateral force on the base, pulsed on and held. Tuned for the
    # ~50 kg quadruped so the passive robot is disturbed to ~25-30 deg of tilt but
    # reliably recovers across seeds (past ~360 N it tips past the point of no
    # return on a bad shove); the original ant's 40 N was for a 10x lighter body.
    force_mag=320.0, force_every=80, force_hold=25,
    # The roughness of the ground is emulated with a random force jostling the
    # legs every step, rather than modelled as bumps in the surface -- fine bumps
    # gave the feet messy, catching contacts. So the height field is kept SMOOTH
    # (few, low-frequency octaves) and only carries the big hills and valleys;
    # the leg noise supplies the texture.
    leg_noise_mag=55.0,
    # Leaving the green terrain patch (walking off its edge) ends the episode.
    terminate_off_green=True,
    # Smooth rolling hills over a large expanse, regenerated fresh every episode:
    # a single low-frequency octave (no added detail octaves) with a tall peak and
    # high contrast for pronounced but perfectly smooth hills and valleys. The
    # panels sit on the surface and the base is dropped on at spawn. The base
    # elevation is lifted off zero so even the lowest valley sits above the floor
    # plane -- the robot always stands on the terrain, never on the ground seam.
    terrain=True,
    terrain_region=((-2.0, 8.5), (-5.2, 5.2)),
    terrain_height=(0.3, 1.2),
    # A 16x16 grid resolves the deliberately low-frequency single-octave hills
    # while keeping every large panel/robot geom below MJWarp's 50-triangle
    # heightfield collision limit. At 64x64 the panel footprints can overlap
    # more than 50 terrain triangles, invalidating randomized training worlds.
    terrain_res=(16, 16),
    terrain_octaves=1,
    terrain_base_freq=2.6,
    terrain_lacunarity=2.2,
    terrain_roughness=0.3,
    terrain_contrast=1.9,
    field_pos=[2.6, 0.0, 1.6], field_size=[4.0, 3.4, 3.3],
)

# Careful locomotion. A tightly packed array whose plates are solid obstacles
# (collide_roller_only=False): the robot must place its body precisely among the
# panels, because any part other than the roller touching one ends the episode.
# Spawn jitter and tight budgets. Co-design pressure is to shape the legs to fit
# the gaps.
TRAVERSE = dict(
    # A larger 3x4 array to work across.
    panel_rows=3, panel_cols=4,
    first_panel_pos=[1.8, 0.0, 0.0],
    panel_length=1.1, panel_width=1.3, panel_height=1.4, panel_tilt_degs=20.0,
    # Tightly packed, but a moderately wide gap so the array is navigable
    # without instantly fouling a panel.
    panel_gap_x=0.8, panel_gap_y=0.45,
    # Cells a little larger than the roller (0.25 m): 1.1/4 = 0.28, 1.3/4 = 0.33.
    n_sensor_rows=4, n_sensor_cols=4,
    max_offset=0.1,
    midpoints=True,
    min_base_height_frac=MIN_BASE_HEIGHT_FRAC,
    # Spawn well back from the array; the extra approach gets a bigger walk budget.
    spawn_offset=1.5,
    next_cell_step_limit=120, next_panel_step_limit=600,
    rand_pos_range=0.15, rand_heading_range=10.0,
    collide_roller_only=False,
    # Touching a panel with anything but the roller ends the episode -- a robot
    # cleaning panels must not damage them. There is no extra penalty: the
    # deterrent is forfeiting the rest of the sweep (up to max_return), while the
    # cleaning banked so far is kept. This keeps "attempt and occasionally foul"
    # strictly better than stalling, so the agent has no do-nothing equilibrium
    # to fall into -- carefulness is instrumental to finishing the job.
    terminate_on_body_contact=True, body_contact_penalty=0.0,
    body_contact_depth=0.005,
    field_pos=[2.9, 0.0, 0.95], field_size=[3.7, 3.7, 2.0],
)

PRESETS = {
    "reach": REACH,
    "stability": STABILITY,
    "traverse": TRAVERSE,
}
