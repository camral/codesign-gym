"""The registered task's host modules (config, embodiment, render) keep lazy imports and live configuration values."""

from __future__ import annotations

import json
import pickle
import subprocess
import sys

from envs.soft_walker import config, registration


def test_registered_host_modules_do_not_initialize_backends() -> None:
    """Host imports stay lazy; the retired ``registered`` package and the JAX wrapper are absent."""
    script = """
import importlib, importlib.util, json, sys
for name in ('config', 'embodiment', 'render', 'registration'):
    importlib.import_module('envs.soft_walker.' + name)
for name in ('registered', 'registered_task', 'task_presets', 'terrain_presets', 'bundle_builders',
             'jax_vector_env', 'jax_resolved'):
    assert importlib.util.find_spec('envs.soft_walker.' + name) is None, name
print(json.dumps({name: name in sys.modules for name in ('jax', 'warp', 'newton')}))
"""
    result = subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)
    assert json.loads(result.stdout.splitlines()[-1]) == {"jax": False, "warp": False, "newton": False}


def test_timing_metadata_uses_its_natural_module_identity() -> None:
    """Host metadata pickles resolve directly to ``config``."""
    timing = config.env_kwargs("spring")
    metadata = config.timing_metadata(timing, timing, config.CONTROL_DT / config.SUBSTEPS)
    assert type(metadata).__module__ == "envs.soft_walker.config"
    restored = pickle.loads(pickle.dumps(metadata))
    assert type(restored) is config.TimingMetadata
    assert restored == metadata
    assert restored.physics_steps_per_action == config.SUBSTEPS
    assert restored.horizon == config.HORIZON


def test_registration_shares_the_live_config_and_keeps_the_registry_factory() -> None:
    """Mutable preset hooks live in one module, which the registry factory reads."""
    assert registration.C is config
    assert registration.build_vec.__module__ == "envs.soft_walker.registration"
