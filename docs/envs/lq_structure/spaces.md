# Observations, actions and rewards

## Observation

A `Box(4n,)` float64 vector (24 values for `n = 6`):

| Slice | Content |
|---|---|
| `[0:n]` | Position of each mass |
| `[n:2n]` | Velocity of each mass |
| `[2n:3n]` | Actuator mask (1 = actuated) |
| `[3n:4n]` | Sensor mask (1 = sensed) |

In `LQStructure-LQG` the state is measured only at sensed masses, with Gaussian noise (std 0.05); unsensed entries are 0.
In LQR and QP the full state is observed without noise.

## Action

A `Box(n,)` vector of forces in **actuator order**: entry `k` is the force on the `k`-th actuated mass (in increasing mass index). With 3 actuators, only the first 3 entries are used and the rest are ignored.
`LQStructure-QP` clips forces to `[-1, 1]`; the other setups are unbounded.

To apply a per-mass force `f` (length `n`), take the actuated entries:

```python
import numpy as np

def to_action(f, actuator_mask):
    """Per-mass forces -> the compact action this environment expects."""
    u = np.zeros_like(f)
    active = f[actuator_mask.astype(bool)]
    u[: active.size] = active
    return u
```

The actuator mask is in the observation, so a policy can always read the current placement.

## Reward

Each step pays the negative quadratic stage cost times `dt`:

```text
reward = -(xᵀ Q x + uᵀ R u) · dt        Q = I (positions and velocities), R = I
```

The best possible return is 0 (the chain at rest with no force). All setups add process noise: a random force (std 0.05) on each mass, held over the step, and the start state is random (position std 0.5).

## Episode

200 steps of 0.05 s (10 s). Episodes end by truncation only.

## Info

| Key | Content |
|---|---|
| `state` | The true state `[positions, velocities]` |
| `embodiment` | The design in use (a list with one dict per copy from `make_vec`) |
| `plant` | The discrete plant matrices for the current design |
| `stage_cost` | The cost before scaling by `dt` |
| `applied_control` | The force after masking and clipping |
| `n_actuators` | Number of actuators in the design |
| `embodiment_rejected` | Rejection flag |
