"""Microgrid Sizing and Dispatch Co-Design environment.

An energy-system co-design environment: the embodiment specifies the
installation's sizing (solar capacity, battery size, generator rating) and the
control policy dispatches charge / discharge / import / export decisions against
stochastic demand and irradiance. Wraps python-microgrid (pymgrid) for the
underlying dynamics and its bundled TMY-solar / reference-building load data.

Registered with codesign_gym under ``Microgrid-Random``, ``Microgrid-Forecast``,
``Microgrid-Campus``, ``Microgrid-OffGridSingle``, and
``Microgrid-OffGridCampus``; construct via ``codesign_gym.make`` /
``codesign_gym.make_vec``.
"""

from . import embodiment
from .env import MicrogridCodesignEnv
from .vector_env import MicrogridVectorEnv
from .registration import build_vec, SETUP_CONFIGS

__all__ = [
    "MicrogridCodesignEnv",
    "MicrogridVectorEnv",
    "build_vec",
    "SETUP_CONFIGS",
    "embodiment",
]
