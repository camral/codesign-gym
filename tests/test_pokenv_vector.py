"""Tests for the vectorized Pokemon Showdown env and the central codesign_gym API.

Skipped where the optional ``showdown`` dependencies are absent; the live vector env is skipped if a
Showdown cluster cannot be launched. One num_envs=2 env is shared across the tests (module scope) so
the Showdown servers are launched once.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from core.seeding import lane_seeds

import codesign_gym as cg

pytest.importorskip("pokemon_codesign_env")

from envs.pokenv.vector_env import PokenvVectorEnv

NUM_ENVS = 2


@pytest.fixture(scope="module")
def vec():
    try:
        venv, base = cg.make_vec("Pokenv-Random", num_envs=NUM_ENVS, shards=1)
    except Exception as error:  # noqa: BLE001
        pytest.skip(f"could not launch a Showdown cluster: {error}")
    yield venv, base
    venv.close()


def _legal_actions(observation) -> np.ndarray:
    masks = np.asarray(observation["action_mask"])
    return np.array([int(np.flatnonzero(row)[0]) if row.any() else 0 for row in masks], dtype=np.int64)


def test_make_vec_returns_env_and_single_base_embodiment(vec):
    venv, base = vec
    assert isinstance(venv, PokenvVectorEnv)
    # codesign_gym.make_vec returns a single base_embodiment; per-env designs stay in base_embodiments.
    assert isinstance(base, dict)
    assert len(venv.base_embodiments) == NUM_ENVS
    assert venv.embodiment_space is not None
    assert venv.single_observation_space is not None and venv.single_action_space is not None


def test_reset_returns_obs_rejected_info(vec):
    venv, _ = vec
    obs, rejected, infos = venv.reset(seed=0)
    assert isinstance(obs, dict)
    assert rejected.shape == (NUM_ENVS,) and rejected.dtype == bool
    assert not rejected.any()
    assert isinstance(infos, dict)


def test_step_forwards_actions_and_returns_batched_five_tuple(vec):
    venv, _ = vec
    obs, _rejected, _infos = venv.reset(seed=0)
    obs, rewards, terms, truncs, infos = venv.step(_legal_actions(obs))
    assert np.asarray(rewards).shape == (NUM_ENVS,)
    assert np.asarray(terms).shape == (NUM_ENVS,)
    assert np.asarray(truncs).shape == (NUM_ENVS,)


def test_step_embodiment_is_ignored_with_a_warning(vec):
    """The Pokemon battle locks the team, so a design passed to step is ignored with a warning."""
    venv, _ = vec
    obs, _rejected, _infos = venv.reset(seed=0)
    designs = [venv.embodiment_space.sample() for _ in range(NUM_ENVS)]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        venv.step(_legal_actions(obs), embodiment=designs)
    assert any("transition-level" in str(w.message).lower() for w in caught)


def test_validate_embodiment_batch(vec):
    venv, _ = vec
    designs = [venv.embodiment_space.sample() for _ in range(NUM_ENVS)]
    result = venv.validate_embodiment(designs)
    assert result.shape == (NUM_ENVS,) and result.dtype == bool


def test_reset_with_designs_reports_rejection_per_env(vec):
    venv, _ = vec
    designs = [venv.embodiment_space.sample() for _ in range(NUM_ENVS)]
    _obs, rejected, _infos = venv.reset(seed=1, embodiment=designs)
    assert rejected.shape == (NUM_ENVS,) and rejected.dtype == bool


def test_none_means_the_base_team_and_seeds_are_normalised(vec, monkeypatch):
    """None (or a None entry) resets that env with its base team, not whatever was installed last,
    and NumPy / per-env seeds reach the package as per-env Python ints."""
    venv, _ = vec
    calls = []
    real = venv._venv.reset
    monkeypatch.setattr(venv._venv, "reset", lambda **kw: calls.append(kw) or real(**kw))
    design = venv.embodiment_space.sample()
    venv.reset(seed=np.int64(3), embodiment=[None, design])
    teams = [o["embodiment"] for o in calls[-1]["options"]]
    assert teams[0] is venv.base_embodiments[0] and teams[1] is design
    assert calls[-1]["seed"] == lane_seeds(3, 2) and all(type(s) is int for s in calls[-1]["seed"])
    venv.reset(seed=[7, 9])
    assert calls[-1]["seed"] == [7, 9] and [o["embodiment"] for o in calls[-1]["options"]] == venv.base_embodiments


def test_validate_embodiment_accepts_none(vec):
    venv, _ = vec
    assert venv.validate_embodiment(None).tolist() == [True] * NUM_ENVS
    assert venv.validate_embodiment([None, venv.base_embodiments[1]]).tolist() == [True] * NUM_ENVS
    _obs, rejected, _infos = venv.reset(embodiment=[None, None])
    assert not rejected.any()


def test_partial_reset_is_refused_and_a_full_mask_is_a_full_reset(vec):
    venv, _ = vec
    assert venv.metadata["supports_partial_reset"] is False
    with pytest.raises(NotImplementedError, match="every lane together"):
        venv.reset(options={"reset_mask": np.array([True, False])})
    _obs, rejected, _infos = venv.reset(seed=0, options={"reset_mask": np.ones(NUM_ENVS, bool)})
    assert rejected.shape == (NUM_ENVS,) and not rejected.any()
