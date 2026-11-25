"""Linear-Quadratic Structure Co-Design environment.

A wall-anchored mass-spring-damper chain whose embodiment (masses, stiffness, damping,
actuator/sensor placement) is co-designed with a regulating control policy. Registered
with codesign_gym under LQStructure-LQR / -LQG-v0 / -QP-v0; construct via
``codesign_gym.make`` / ``codesign_gym.make_vec``.
"""

from .config import LQConfig, make_config, ControllerType
from .env import LQStructureEnv
from .vector_env import LQStructureVectorEnv
from .registration import build_vec

__all__ = [
    "LQStructureEnv",
    "LQStructureVectorEnv",
    "LQConfig",
    "make_config",
    "ControllerType",
    "build_vec",
]
