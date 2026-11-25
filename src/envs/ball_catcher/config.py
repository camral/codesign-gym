"""Presets for the BallCatcher co-design environment.

Each preset is a kwargs dict for :class:`codesign_env.BallCatcherCoDesignEnv`,
selected by ``setup=`` in ``registration.build_vec``, and overrides the class
defaults. The three share the robot and differ only in the throw (see the
README's "Throw Parameters"):

* ``pitch``    -- arrival mode: the ball crosses a broad, low band at a set time.
* ``arc``      -- speed mode: farther and lofted, with moderate spread.
* ``fielding`` -- speed mode: distant and high, with a wide spread in speed.
"""

from __future__ import annotations

PITCH = dict(
    pitch_to=[0.4648899, 0.2, 1.5],
    pitch_arrival_time=0.4,
    pitch_arrival_spread=0.025,
    pitch_target_half_widths=[0.0, 1.5, 0.15],
)

ARC = dict(
    pitch_to=[0.4648899, 0.2, 5.5],
    pitch_to_radius=0.5,
    pitch_vel=14.0,
    pitch_vel_spread=0.5,
)

FIELDING = dict(
    pitch_to=[0, 0, 4.5],
    pitch_to_radius=1.0,
    pitch_vel=15.0,
    pitch_vel_spread=2.0,
)

PRESETS = {
    "pitch": PITCH,
    "arc": ARC,
    "fielding": FIELDING,
}
