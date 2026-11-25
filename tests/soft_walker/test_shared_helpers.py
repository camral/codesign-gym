"""Release-safe shared helpers retain serialization and differentiable contracts."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from envs.soft_walker import shared


def test_shared_host_import_and_writers_do_not_load_accelerators(tmp_path: Path) -> None:
    """Installed-style host consumers need neither device runtimes nor internal studies."""
    source = Path(__file__).resolve().parents[2] / "src"
    script = f"""
import json, sys
from pathlib import Path
sys.path.insert(0, {str(source)!r})
before = list(sys.path)
from envs.soft_walker import shared
shared.write_json(Path('artifact.json'), {{'value': float('nan')}})
shared.evidence_write_json(Path('evidence.json'), {{'value': float('nan')}})
assert sys.path == before
blocked = ('jax', 'warp', 'newton', 'envs.soft_walker.experiments',
           'envs.soft_walker.diagnostics')
assert not any(name == prefix or name.startswith(prefix + '.')
               for name in sys.modules for prefix in blocked)
print(json.dumps({{'artifact': json.loads(Path('artifact.json').read_text()),
                  'evidence': json.loads(Path('evidence.json').read_text())}}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert json.loads(completed.stdout) == {
        "artifact": {"value": None},
        "evidence": {"value": {"nonfinite": "nan"}},
    }


def test_shared_paths_retain_checkout_roots() -> None:
    """Moving the helper up one package level must not shift public command roots."""
    repo = Path(__file__).resolve().parents[2]
    package = repo / "src/envs/soft_walker"
    assert shared.REPO_ROOT == repo
    assert shared.SRC_ROOT == repo / "src"
    assert shared.SOFT_ROOT == package
    assert shared.DIAGNOSTICS_ROOT == package / "diagnostics"
    assert shared.SCRIPTS_ROOT == package / "scripts"
    assert shared.ENGINE_ROOT == package / "engine"
    assert shared.TESTS_ROOT == repo / "tests/soft_walker"


@dataclass
class _Artifact:
    """A representative caller-owned artifact payload."""

    samples: np.ndarray
    checkpoint: Path


def test_artifact_and_evidence_json_keep_distinct_nonfinite_policies(tmp_path: Path) -> None:
    """Artifacts use null, while diagnostic evidence must retain signed failure markers."""
    values = np.asarray([1.0, np.nan, np.inf, -np.inf])
    payload = {7: (values, Path("last.npz"))}
    assert shared.json_value(payload) == {"7": [[1.0, None, None, None], "last.npz"]}
    assert shared.evidence_jsonable(payload) == {
        "7": [
            [1.0, {"nonfinite": "nan"}, {"nonfinite": "inf"}, {"nonfinite": "-inf"}],
            "last.npz",
        ]
    }
    artifact = tmp_path / "nested/artifact.json"
    shared.write_json(artifact, _Artifact(values, Path("last.npz")))
    assert json.loads(artifact.read_text()) == {
        "samples": [1.0, None, None, None],
        "checkpoint": "last.npz",
    }
    assert not artifact.with_name(artifact.name + ".tmp").exists()
    before = artifact.read_bytes()
    with pytest.raises(TypeError):
        shared.write_json(artifact, {"unsupported": object()})
    assert artifact.read_bytes() == before, "Failed serialization must preserve the last record"


def test_numeric_npz_and_evidence_hash_preserve_portable_arrays(tmp_path: Path) -> None:
    """Numeric checkpoints are readable without pickle and hashes describe their exact bytes."""
    path = tmp_path / "state.npz"
    original = np.arange(12, dtype=np.float32).reshape(3, 4)
    shared.write_npz(path, positions=original, iteration=np.int64(3))
    with np.load(path, allow_pickle=False) as saved:
        assert set(saved.files) == {"positions", "iteration"}
        assert saved["positions"].dtype == original.dtype
        np.testing.assert_array_equal(saved["positions"], original)
        assert saved["iteration"].shape == ()
        assert int(saved["iteration"]) == 3
    assert shared.evidence_sha256(path) == hashlib.sha256(path.read_bytes()).hexdigest()
    assert not path.with_name(path.name + ".tmp").exists()


def test_periodic_action_values_and_parameter_gradients(dtype: np.dtype) -> None:
    """Moving controller code preserves harmonic ordering, one tanh, and JAX gradients."""
    coefficients = np.asarray([[0.2, 0.4, -0.3, 0.1, 0.6], [-0.1, -0.2, 0.5, 0.7, -0.4]], dtype)
    time, period = 0.17, 0.8
    phase = 2 * np.pi * time / period
    basis = np.asarray([1, np.sin(phase), np.cos(phase), np.sin(2 * phase), np.cos(2 * phase)])
    expected = np.tanh(coefficients @ basis)
    parameters = jnp.asarray(coefficients)
    actual = jax.jit(shared.periodic_action)(parameters, jnp.asarray(time, dtype), period)
    gradient = jax.grad(lambda p: jnp.sum(shared.periodic_action(p, time, period)))(parameters)
    expected_gradient = (1 - expected**2)[:, None] * basis
    tolerance = 5e-6 if dtype == np.dtype("float32") else 2e-12
    np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(gradient, expected_gradient, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("shape", [(3,), (2, 1), (2, 4)])
def test_periodic_action_retains_shape_rejection(shape: tuple[int, ...]) -> None:
    """Malformed controller coefficients remain caller errors rather than broadcasting."""
    with pytest.raises(ValueError, match=r"Expected \(channels, 1 \+ 2\*harmonics\)"):
        shared.periodic_action(jnp.zeros(shape), jnp.asarray(0.0), 0.8)


@pytest.mark.parametrize("kind", ["spring", "fem", "mpm2", "beam"])
def test_region_tags_preserve_driver_reference_offsets(kind: str, dtype: np.dtype) -> None:
    """Particle and beam cell centers retain their half-cell envelope correction."""
    reference = jnp.asarray([[0, 0], [0.8, 0.16], [0, 0.66], [0.8, 1]], dtype)
    config = SimpleNamespace(size=(1, 1), particles_per_axis=2)
    env = SimpleNamespace(config=config)
    model = SimpleNamespace(active=jnp.ones(4, dtype=bool), reference=reference)
    if kind != "spring":
        env.kind = kind
    if kind == "fem":
        model.mesh = SimpleNamespace(reference=reference)
    elif kind == "mpm2":
        model.cloud = SimpleNamespace(reference=reference)
        env._grid = SimpleNamespace(spacing=0.08)
    elif kind == "beam":
        model.topology = SimpleNamespace(reference=reference, h=0.04)
    torso, feet = shared.region_tags(env, model)
    center_offset = kind in ("mpm2", "beam")
    np.testing.assert_array_equal(torso, [False, False, center_offset, True])
    np.testing.assert_array_equal(
        feet, [[True, False, False, False], [False, not center_offset, False, False]]
    )


def test_region_tags_preserve_spatial_leg_order_and_active_mask(dtype: np.dtype) -> None:
    """Quadruped feet retain the 2*x+z ordering, and inactive material has no label."""
    model = SimpleNamespace(
        reference=jnp.asarray(
            [[0, 0, 0], [0.8, 0, 0], [0, 0, 0.8], [0.8, 0, 0.8], [0, 0.9, 0], [0.8, 0.9, 0.8]],
            dtype,
        ),
        active=jnp.asarray([True, True, True, True, True, False]),
    )
    env = SimpleNamespace(config=SimpleNamespace(size=(1, 1, 1)))
    torso, feet = shared.region_tags(env, model)
    np.testing.assert_array_equal(torso, [False, False, False, False, True, False])
    expected_feet = np.zeros((4, 6), dtype=bool)
    expected_feet[np.arange(4), [0, 2, 1, 3]] = True
    np.testing.assert_array_equal(feet, expected_feet)
