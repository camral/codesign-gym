"""Resolved actuator masks have one host contract across the two backends."""

from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from envs.soft_walker.contracts.fields import HostFields, resolved_action_mask


@pytest.mark.parametrize("kind", ("spring", "beam"))
def test_signed_routes_activate_channels_and_pad(kind: str) -> None:
    fields = {"routing": np.asarray([[0.0, -0.5, 0.0], [0.0, 0.0, 0.25]])}
    before = fields["routing"].copy()
    mask = resolved_action_mask(fields, kind, channels=3, action_size=5)
    np.testing.assert_array_equal(mask, [False, True, True, False, False])
    assert mask.dtype == np.dtype(bool)
    np.testing.assert_array_equal(fields["routing"], before)
    mask[:] = False
    np.testing.assert_array_equal(
        resolved_action_mask(fields, kind, 3, 5), [False, True, True, False, False]
    )


@pytest.mark.parametrize("kind", ("fem", "mpm2", "mpm3"))
@pytest.mark.parametrize("channel_dtype", (np.int32, np.int64))
def test_continuum_mask_requires_all_three_actuator_gates(
    kind: str, channel_dtype: type[np.integer]
) -> None:
    fields: HostFields = {
        "alpha": np.asarray([1.0, 0.0, -0.1, 1.0, 1.0, 1.0, 0.5]),
        "strength": np.asarray([0.1, 0.1, 0.1, 0.0, -0.1, 0.1, 0.2]),
        "kind": np.asarray([1, 1, 1, 1, 1, 0, 2], np.int32),
        "channel": np.asarray([2, 0, 0, 0, 0, 0, 2], channel_dtype),
    }
    before = {name: value.copy() for name, value in fields.items()}
    mask = resolved_action_mask(fields, kind, channels=3, action_size=4)
    np.testing.assert_array_equal(mask, [False, False, True, False])
    assert mask.dtype == np.dtype(bool)
    for name, value in fields.items():
        np.testing.assert_array_equal(value, before[name])


@pytest.mark.parametrize("kind", ("spring", "beam", "fem", "mpm2", "mpm3"))
def test_empty_resolved_fields_leave_all_channels_inactive(kind: str) -> None:
    fields: HostFields = {
        "routing": np.zeros((0, 3)),
        "alpha": np.zeros(0),
        "strength": np.zeros(0),
        "kind": np.zeros(0, np.int32),
        "channel": np.zeros(0, np.int32),
    }
    mask = resolved_action_mask(fields, kind, channels=3, action_size=3)
    assert mask.shape == (3,)
    assert mask.dtype == np.dtype(bool)
    assert not mask.any()


def test_native_mask_delegates_without_allocating_device_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from envs.soft_walker.engine.warp_native import env

    fields: HostFields = {"routing": np.zeros((1, 2))}
    bundle = SimpleNamespace(fields=fields)
    owner = SimpleNamespace(kind="spring", channels=2, action_size=4)
    expected = np.asarray([False, True, False, False])
    calls = []

    def capture(actual_fields: HostFields, kind: str, channels: int, action_size: int):
        calls.append((actual_fields, kind, channels, action_size))
        return expected

    monkeypatch.setattr(env, "resolved_action_mask", capture)
    assert env.NativeEnv._mask(owner, bundle) is expected
    assert len(calls) == 1
    assert calls[0][0] is fields
    assert calls[0][1:] == ("spring", 2, 4)


def test_mask_helper_import_stays_independent_of_backends() -> None:
    script = """
import sys
from envs.soft_walker.contracts.fields import resolved_action_mask
assert callable(resolved_action_mask)
assert not {'jax', 'jaxlib', 'warp', 'newton', 'diffrax'}.intersection(sys.modules)
assert not any(name.startswith('envs.soft_walker.engine') for name in sys.modules)
assert not {'envs.soft_walker.' + m for m in ('config', 'embodiment', 'vector_env', 'registration')}.intersection(sys.modules)
"""
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)
