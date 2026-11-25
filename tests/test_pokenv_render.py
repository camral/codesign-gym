"""Pokémon replays are HTML artifacts from completed battles, not image frames."""

import os
from pathlib import Path

import numpy as np
import pytest

import codesign_gym as cg

pytest.importorskip("pokemon_codesign_env")

from envs.pokenv.vector_env import PokenvVectorEnv


def _showdown_kwargs():
    path = os.environ.get("CODESIGN_POKEMON_SHOWDOWN_PATH")
    return {"showdown_path": Path(path)} if path else {}


def test_unsupported_render_modes_fail_before_server_start():
    with pytest.raises(ValueError, match="render_mode"):
        PokenvVectorEnv(1, render_mode="human")
    with pytest.raises(ValueError, match="render_mode"):
        PokenvVectorEnv(1, render_mode="rgb_array")


@pytest.mark.slow
def test_completed_battle_html_is_available_per_lane():
    try:
        env, _ = cg.make_vec(
            "Pokenv-Random", num_envs=2, shards=1, render_mode="html",
            **_showdown_kwargs(),
        )
    except Exception as error:  # noqa: BLE001
        pytest.skip(f"could not launch a Showdown cluster: {error}")
    try:
        obs, rejected, _ = env.reset(seed=11)
        assert not rejected.any()
        assert env.metadata["render_modes"] == ["html"]
        assert env.render() == (None, None)
        rng = np.random.default_rng(11)
        for _ in range(350):
            actions = np.asarray([
                rng.choice(np.flatnonzero(mask)) for mask in obs["action_mask"]
            ])
            obs, _, _, _, info = env.step(actions)
            html = env.render()
            assert isinstance(html, tuple) and len(html) == 2
            if any(page is not None for page in html):
                assert all(page is None or page.startswith("<!DOCTYPE html>") for page in html)
                assert env.render() == html
                for lane, page in enumerate(html):
                    if page is not None:
                        assert info["_replay_html"][lane]
                        assert info["replay_html"][lane] == page
                break
        else:
            pytest.fail("no battle finished within 350 turns")
        actions = np.asarray([
            rng.choice(np.flatnonzero(mask)) for mask in obs["action_mask"]
        ])
        _, _, _, _, next_info = env.step(actions)
        next_html = env.render()
        for lane, previous in enumerate(html):
            if previous is not None and not next_info.get("_replay_html", np.zeros(2, dtype=bool))[lane]:
                assert next_html[lane] == previous
        env.reset(seed=12)
        assert env.render() == (None, None)
    finally:
        env.close()


@pytest.mark.slow
def test_single_env_returns_one_html_string():
    try:
        env, _ = cg.make("Pokenv-Random", shards=1, render_mode="html", **_showdown_kwargs())
    except Exception as error:  # noqa: BLE001
        pytest.skip(f"could not launch a Showdown cluster: {error}")
    try:
        obs, rejected, _ = env.reset(seed=17)
        assert not rejected
        assert env.render() is None
        rng = np.random.default_rng(17)
        for _ in range(350):
            action = int(rng.choice(np.flatnonzero(obs["action_mask"])))
            obs, _, _, _, _ = env.step(action)
            html = env.render()
            if html is not None:
                assert isinstance(html, str) and html.startswith("<!DOCTYPE html>")
                break
        else:
            pytest.fail("no battle finished within 350 turns")
        action = int(rng.choice(np.flatnonzero(obs["action_mask"])))
        _, _, _, _, info = env.step(action)
        if "replay_html" not in info:
            assert env.render() == html
        env.reset(seed=18)
        assert env.render() is None
    finally:
        env.close()
