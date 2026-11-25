# LQ Structure

Ids: `LQStructure-LQR`, `LQStructure-LQG`, `LQStructure-QP`. Uses: python-control, cvxpy.

A chain of masses, springs and dampers that the policy keeps steady against disturbances.

- **Design:** masses, stiffnesses, damping, and where the actuators and sensors go.
- **Control:** the actuator forces.
- **Reward:** a negative quadratic cost, so the best possible return is 0.
- **Design changes in `step`:** supported.

The optimal controller for any fixed design can be computed exactly, which makes this small, fast env a good sanity check for co-design algorithms.

| Id | Problem | Return [max / min] |
|---|---|---|
| `LQStructure-LQR` | Continuous design with unconstrained LQR control. | 0 / −∞ |
| `LQStructure-LQG` | Actuator and sensor placement, with LQG control under process and measurement noise. | 0 / −∞ |
| `LQStructure-QP` | Actuator saturation under a mass and actuator budget, with control solved as a QP. | 0 / −∞ |
