"""Pokemon Showdown team-design co-design environment.

A co-design environment over Pokemon Showdown battles: the embodiment is a team design drawn from
``TeamDesignSpace`` and the control policy plays the battle the team is locked into, against a fixed
opponent task suite. Wraps the ``pokemon_codesign_env`` package (which bundles Pokemon Showdown and
poke-env) for the dynamics, legality, and its own server-owning, process-sharded vectorization.

Registered with codesign_gym under Pokenv-Random, Pokenv-MaxPower, Pokenv-Heuristic, and the
PokenvLeague generation/random variants; construct via ``codesign_gym.make`` /
``codesign_gym.make_vec``.

Requires the optional ``showdown`` dependencies (``pip install codesign-gym[showdown]``) and a built
bundled Pokemon Showdown server; these submodules import ``pokemon_codesign_env`` at import time, so
they load only when the env is actually requested through codesign_gym's lazy registry.
"""

from .env import PokenvCodesignEnv
from .registration import OPPONENTS, build_vec
from .vector_env import PokenvVectorEnv

__all__ = ["PokenvCodesignEnv", "PokenvVectorEnv", "build_vec", "OPPONENTS"]
