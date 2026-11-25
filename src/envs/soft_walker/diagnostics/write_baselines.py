"""Regenerate deterministic semantic fixtures, never optimized scores."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    # Direct-file execution must select this checkout before package imports.
    _source_root = str(Path(__file__).resolve().parents[3])
    if _source_root not in sys.path:
        sys.path.insert(0, _source_root)

from envs.soft_walker.shared import (
    ENGINE_ROOT as ENGINE_ROOT,
    REPO_ROOT as REPO_ROOT,
    SCRIPTS_ROOT as SCRIPTS_ROOT,
    SOFT_ROOT as SOFT_ROOT,
    SRC_ROOT as SRC_ROOT,
    TESTS_ROOT as TESTS_ROOT,
)


def main() -> None:
    """Regenerate the explicit baseline artifact set selected by this maintenance command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=SOFT_ROOT / "baselines")
    args = parser.parse_args()
    from envs.soft_walker.engine.embodiments.decoders import FieldSpec, make_decoder
    from envs.soft_walker.engine.embodiments.precedents import evogym_fixture
    from envs.soft_walker.engine.embodiments.serialization import save_design

    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=True)
    for dtype in ("float64", "float32"):
        for dim in (2, 3):
            for name in ("direct", "cppn", "primitives", "bspline"):
                d = make_decoder(name, spec=FieldSpec(dim=dim), dtype=dtype)
                save_design(
                    root / f"{name}_{dim}d_{dtype}.json",
                    d.base(),
                    d,
                    {"purpose": "unoptimized M1 semantic baseline", "physics_available": dim == 2},
                )
    f = evogym_fixture()
    (root / "evogym_cppn_architecture.json").write_text(
        json.dumps(
            {
                "source": "https://github.com/EvolutionGym/evogym/blob/main/examples/cppn_neat/neat.cfg",
                "activation_source": "https://github.com/CodeReclaimers/neat-python/blob/master/neat/activations.py",
                "layout": "3 inputs (x,y,r), 1 sigmoid hidden, 5 sigmoid category outputs; gain5, clamp60; 12/23 initial edges",
                "category_order": [
                    "empty",
                    "rigid",
                    "soft",
                    "horizontal_actuator",
                    "vertical_actuator",
                ],
                "weights_provenance": "New deterministic seed13 fixture, NOT a published trained robot",
                "embedding": {k: v.tolist() for k, v in f.items()},
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
