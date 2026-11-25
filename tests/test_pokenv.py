"""Tests for the Pokemon Showdown env and its wiring into the central codesign_gym API.

The whole module is skipped where the optional ``showdown`` dependencies are not installed. The
contract tests need only imports; the live tests launch a real (bundled) Showdown server and are
skipped if one cannot be started (e.g. Node too old), rather than failing.
"""

from __future__ import annotations

import numpy as np
import pytest

import codesign_gym as cg

pytest.importorskip("pokemon_codesign_env")

from codesign_gym import SingleEnv
from codesign_gym.registry import REGISTRY
from core.codesign_env import CoDesignEnv
from envs.pokenv.env import PokenvCodesignEnv
from envs.pokenv.registration import OPPONENTS, build_vec

POKEMON_IDS = ["Pokenv-Random", "Pokenv-MaxPower", "Pokenv-Heuristic", "Pokenv-LeagueRand", "Pokenv-LeagueRandH",
    *[f"Pokenv-LeagueG{generation}" for generation in range(1, 10)],
    *[f"Pokenv-LeagueG{generation}H" for generation in range(1, 10)]]


# --- contract / wiring (no Showdown server) ----------------------------------------

def test_registry_lists_all_pokemon_envs():
    assert set(POKEMON_IDS) <= set(cg.list_envs())


def test_all_ids_use_one_builder_module():
    """One fixed builder module per env, mirroring lq_structure and microgrid."""
    paths = {REGISTRY[env_id][0] for env_id in POKEMON_IDS}
    assert paths == {"envs.pokenv.registration:build_vec"}


def test_pokemon_preset_mappings():
    assert REGISTRY["Pokenv-Random"][1] == {"opponent": "random"}
    assert REGISTRY["Pokenv-MaxPower"][1] == {"opponent": "max_base_power"}
    assert REGISTRY["Pokenv-Heuristic"][1] == {"opponent": "simple_heuristics"}
    for generation in range(1, 10):
        assert REGISTRY[f"Pokenv-LeagueG{generation}"][1] == {"opponent": "league", "generation": generation, "difficulty": 0}
        assert REGISTRY[f"Pokenv-LeagueG{generation}H"][1] == {"opponent": "league", "generation": generation, "difficulty": 1}
    assert REGISTRY["Pokenv-LeagueRand"][1] == {"opponent": "league", "random_league": True, "difficulty": 0}
    assert REGISTRY["Pokenv-LeagueRandH"][1] == {"opponent": "league", "random_league": True, "difficulty": 1}


def test_old_pokemon_ids_are_removed():
    assert not {"Pokemon-Random-v0", "Pokemon-MaxBasePower-v0", "Pokemon-SimpleHeuristics-v0", "Pokemon-League-v0"} & set(cg.list_envs())


def test_opponents_match_the_package():
    from pokemon_codesign_env.gym_environment_factory import OPPONENTS as PACKAGE_OPPONENTS

    assert set(OPPONENTS) == set(PACKAGE_OPPONENTS)


def test_build_vec_rejects_unknown_opponent():
    # Raises before constructing anything, so no server is launched.
    with pytest.raises(ValueError, match="unknown opponent"):
        build_vec(1, opponent="not-a-real-opponent")


def test_codesign_env_subclasses_base_and_overrides_hooks():
    assert issubclass(PokenvCodesignEnv, CoDesignEnv)
    for hook in ("_reset", "_step", "_validate_embodiment"):
        assert getattr(PokenvCodesignEnv, hook) is not getattr(CoDesignEnv, hook)


# --- live single env (launches a Showdown server) ----------------------------------

@pytest.fixture(scope="module")
def single_env():
    try:
        env, base = cg.make("Pokenv-Random")
    except Exception as error:  # noqa: BLE001 - any launch failure means "cannot test live here"
        pytest.skip(f"could not launch a Showdown single env: {error}")
    yield env, base
    env.close()


def _legal_action(observation) -> int:
    legal = np.flatnonzero(np.asarray(observation["action_mask"]))
    return int(legal[0]) if legal.size else 0


def test_make_returns_squeezed_single_env(single_env):
    env, base = single_env
    assert isinstance(env, SingleEnv)
    assert isinstance(base, dict)
    obs, rejected, info = env.reset(seed=0)
    # Dict observation, squeezed: each value has no leading batch axis.
    assert isinstance(obs, dict)
    assert np.asarray(obs["action_mask"]).ndim == 1
    assert isinstance(rejected, bool) and not rejected


def test_single_step_takes_a_scalar_discrete_action(single_env):
    env, _ = single_env
    obs, _rejected, _info = env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(_legal_action(obs))
    assert isinstance(reward, float)
    assert isinstance(terminated, bool) and isinstance(truncated, bool)
    assert np.asarray(obs["action_mask"]).ndim == 1


def test_validate_embodiment_returns_bool(single_env):
    env, _ = single_env
    assert isinstance(env.validate_embodiment(env.embodiment_space.sample()), bool)
