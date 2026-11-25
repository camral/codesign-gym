# Levine map provenance

`levine.png` and `levine.yaml` are copied without modification from
`f1tenth/f1tenth_gym_ros` revision `883394df0964c555ee05bea69c3002daf6f2d405`.
They are distributed under the adjacent MIT `LICENSE`.

The current `f1tenth_gym_jax` track loader additionally requires a closed
`<map>_centerline.csv`, although its former Levine assets did not include one.
`levine_centerline.csv` is a Codesign Gym addition describing the centre of the
main closed corridor in the official occupancy map. Coordinates use the map's
0.05 m resolution and `[-51.224998, -51.224998, 0]` origin. The final two
columns are the approximately 0.7 m left and right corridor half-widths expected
by the upstream CSV schema.
