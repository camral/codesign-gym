"""Integer status values, with distinct names for distinct reporting domains.

Root diagnostics and native solid state preserve the specific root failure.
JAX continuum and native MPM physical steps use a binary success/failure flag;
their value 1 does not mean MAX_NEWTON. Step diagnostics add geometric failure
codes. Adjoint status may carry a primal status or an adjoint-specific failure.
Policy/callback validation can inject ROLLOUT_INVALID_INPUT independently of a
root solve. JAX spring's root/ODE labels and spring's state codes are separate
domains; native spring uses the native solid root-code domain.

Plain Python integers can be cast by JAX or bound as Warp constants without
importing either runtime into this module. Existing serialized values stay fixed.
"""

ROOT_OK = 0
ROOT_MAX_NEWTON = 1
ROOT_LINE_SEARCH_FAILED = 2
ROOT_LINEAR_FAILED = 3
ROOT_INVALID_INITIAL = 4

STEP_INVALID_SUPPORT = 10
STEP_INVALID_GEOMETRY = 11
STEP_REJECTED = 12
STEP_COMPACT_CAPACITY_EXCEEDED = 13

BINARY_STATE_OK = 0
BINARY_STATE_FAILED = 1

ROLLOUT_INVALID_INPUT = 4

ADJOINT_REPLAY_FAILED = 20
ADJOINT_REVERSE_FAILED = 21

SPRING_STATE_OK = 0
SPRING_STATE_SOLVE_FAILED = 1
SPRING_STATE_INVALID_GEOMETRY = 2
SPRING_STATE_NONFINITE = 3

SPRING_SOLVER_OK = 0
SPRING_SOLVER_ROOT_NOT_CONVERGED = 1
SPRING_SOLVER_ODE_MAX_STEPS = 2
SPRING_SOLVER_NONLINEAR_MAX_STEPS = 3
SPRING_SOLVER_NONLINEAR_DIVERGENCE = 4
SPRING_SOLVER_SINGULAR_LINEAR_SYSTEM = 5
SPRING_SOLVER_NONFINITE_SOLVE = 6
SPRING_SOLVER_OTHER_DIFFRAX_ERROR = 7
