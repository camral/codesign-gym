"""Cheap contract checks for evidence selection, saved designs, and cache policy."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from soft_paths import REPO_ROOT, SOFT_ROOT, needs_scripts

from envs.soft_walker.engine import make
from envs.soft_walker.engine.drivers.task import BodyConfig
from envs.soft_walker.engine.embodiments.serialization import load_design, save_design


@pytest.mark.parametrize("decoder", ["direct", "cppn", "bspline", "primitives"])
def test_m3_environment_design_serialization(
    decoder: str, dtype: np.dtype[np.floating], tmp_path: Path
) -> None:
    e, b = make("SoftHexFEM3D-v0", dtype=dtype, decoder=decoder, config=BodyConfig(settle_steps=0))
    b["poisson"] = np.asarray(0.31, dtype)
    path = tmp_path / "body.json"
    save_design(path, b, e.decoder, {"env_id": e.env_id})
    loaded, d, metadata = load_design(path)
    for k in b:
        np.testing.assert_array_equal(b[k], loaded[k])
    assert e.validate_embodiment(loaded) and metadata["env_id"] == e.env_id
    doc = json.loads(path.read_text())
    assert doc["version"] == 3
    assert doc["legacy_global_poisson_override"]
    doc["parameter_schema"] = "unrecognized"
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="schema"):
        load_design(path)
    e.close()


@needs_scripts
def test_m3_selection_retains_full_suite() -> None:
    path = SOFT_ROOT / "diagnostics" / "validation_selection.py"
    spec = importlib.util.spec_from_file_location("selection_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert set(module.CHANGED_STAGES) == {"affected", "m3_unit", "m3_numerics", "m3_benchmark"}
    assert set(module.LEGACY_STAGES) <= set(module.FULL_STAGES)
    assert set(module.M3_STAGES) <= set(module.FULL_STAGES)
    assert len(module.AFFECTED_TESTS) == len(set(module.AFFECTED_TESTS))
    for node in module.AFFECTED_TESTS:
        f, name = node.split("::")
        name = name.split("[")[0]
        assert ("def " + name + "(") in (REPO_ROOT / f).read_text()


def test_m3_no_blocks_roundtrip(dtype: np.dtype[np.floating]) -> None:
    from envs.soft_walker.engine.physics import terrain

    t = terrain.generate(terrain.TerrainConfig(blocks=0), 9, 3, dtype)
    restored = terrain.from_dict(terrain.serialize(t), dtype)
    for a, b in zip(t, restored):
        np.testing.assert_array_equal(a, b)
