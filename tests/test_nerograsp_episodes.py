"""NeroGraspAlt (src/envs/nerograsp) runs every world on one episode clock (Gymnasium DISABLED autoreset):
episodes end together, the caller resets the whole batch, a world that ends early is
flagged once and stays ended, and the vector spaces are batched."""

from __future__ import annotations

import numpy as np
import pytest
import torch

pytest.importorskip("mujoco_warp")

import codesign_gym as cg  # noqa: E402
from gymnasium.vector import AutoresetMode  # noqa: E402


@pytest.fixture(scope="module")
def thin():
    env, _ = cg.make_vec("NeroGrasp-Thin", 2, use_cpu=True)
    yield env
    env.close()


def test_declares_disabled_and_batched_spaces(thin):
    assert thin.metadata["autoreset_mode"] == AutoresetMode.DISABLED
    assert thin.action_space.shape == (2, thin.single_action_space.shape[0])
    assert thin.observation_space["actor"].shape == (2, thin.single_observation_space["actor"].shape[0])


def test_step_before_reset_and_partial_reset_are_refused():
    env, _ = cg.make_vec("NeroGrasp-Thin", 2, use_cpu=True)
    try:
        with pytest.raises(RuntimeError, match="reset"):
            env.step(torch.zeros(2, env.single_action_space.shape[0]))
        env.reset(seed=0)
        with pytest.raises(NotImplementedError):
            env.reset(options={"reset_mask": np.array([True, False])})
    finally:
        env.close()


def test_episode_ends_together_and_needs_a_reset(thin):
    thin.reset(seed=0)
    a = torch.zeros(2, thin.single_action_space.shape[0])
    thin._finished[1] = True                          # as if world 1 ended early
    terminal = 0
    for _ in range(thin.max_episode_steps):
        _, reward, terminated, truncated, info = thin.step(a)
        assert not bool(terminated[1]) and not bool(truncated[1])      # already ended: not flagged again
        terminal += int(bool(truncated[0]) or bool(terminated[0]))
    assert terminal == 1 and bool(info["finished"].all())
    with pytest.raises(RuntimeError, match="reset"):
        thin.step(a)
    thin.reset(seed=0)
    assert not bool(thin._finished.any())
    thin.step(a)


def _task(env):
    inner = env
    while not hasattr(inner, "_assess_embodiment"):
        inner = getattr(inner, "env", None) or getattr(inner, "_env")
    return inner


def test_rejection_messages_name_the_rejected_design_row(thin, monkeypatch):
    from envs.nerograsp import base_env
    task = _task(thin)
    monkeypatch.setattr(base_env, "rejection_reasons", lambda metrics, *a, **k: [["forced"] for _ in metrics])
    reasons = task._assess_embodiment(list(thin.base_embodiment))[4]
    per_design = reasons[::task.num_unique_objects]
    assert all(f"embodiment row {row}," in r and not any(f"embodiment row {o}," in r for o in range(len(per_design)) if o != row)
               for row, r in enumerate(per_design))


def test_mesh_quality_columns_are_thumb_then_index_like_contacts(thin, monkeypatch):
    task = _task(thin)
    records = [{"env_id": 0, "effector": "leap_th_tip", "q": 1.0}, {"env_id": 0, "effector": "leap_if_tip", "q": 2.0}]
    monkeypatch.setattr(task, "last_quality_metrics", records)
    assert task._quality_info()["mesh_quality/q"][0].tolist() == [1.0, 2.0]
