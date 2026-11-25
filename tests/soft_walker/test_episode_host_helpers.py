"""Shared episode callbacks retain the concrete owner's launch arguments."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from envs.soft_walker.engine.warp_native._episode_host import EpisodeHost
from envs.soft_walker.engine.warp_native.episode import Episode
from envs.soft_walker.engine.warp_native.solid_episode import SolidEpisode


@pytest.mark.parametrize("episode_type", (Episode, SolidEpisode))
def test_saved_root_callback_has_one_inherited_implementation(episode_type: type) -> None:
    assert "load_saved_root" not in episode_type.__dict__
    assert episode_type.load_saved_root is EpisodeHost.load_saved_root


@pytest.mark.parametrize("episode_type", (Episode, SolidEpisode))
def test_saved_root_callback_keeps_arrays_and_kernel_with_each_owner(episode_type: type) -> None:
    """Binding the callback never shares mutable episode state or allocates a device."""
    owners = []
    calls = []
    for capacity in (7, 11):
        owner = episode_type.__new__(episode_type)
        owner.e = SimpleNamespace(load_root=object())
        owner.step = SimpleNamespace(capacity=capacity, q=object(), code=object())
        owner.replay_roots = object()
        owner.replay_codes = object()
        owner.clock = object()
        owner.launch = lambda *args: calls.append(args)
        owners.append(owner)

    callbacks = [owner.load_saved_root for owner in owners]
    for owner, callback in zip(owners, callbacks):
        assert callback.__self__ is owner
        assert callback.__func__ is EpisodeHost.load_saved_root
        callback()
    assert len(calls) == 2
    for owner, arguments in zip(owners, calls):
        assert arguments == (
            owner.e.load_root,
            owner.step.capacity,
            owner.replay_roots,
            owner.replay_codes,
            owner.clock,
            owner.step.q,
            owner.step.code,
        )
    assert calls[0][0] is not calls[1][0]
    assert calls[0][2] is not calls[1][2]
