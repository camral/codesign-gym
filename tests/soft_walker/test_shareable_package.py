"""Small saved-gallery fixtures exercise packaging without historical runs or physics."""

from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType

import pytest

from envs.soft_walker.scripts import package_shareable


class FrozenDatetime(datetime):
    """Keep the generated archive name and captured timestamps deterministic."""

    @classmethod
    def now(cls, tz: object = None) -> FrozenDatetime:
        return cls(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)


def gallery_checkout(root: Path) -> list[str]:
    """Create the fixed gallery dependencies, plus selections and link-repair canaries."""
    sources = [
        "pyproject.toml",
        "README.md",
        "LICENSE",
        "src/envs/__init__.py",
        "src/envs/soft_walker/docs/COMPONENTS_AND_GYMNASIUM.md",
        "src/envs/soft_walker/docs/PERFORMANCE_HANDOFF.md",
        "src/envs/soft_walker/docs/RANDOM_CREATURES.md",
        "src/envs/soft_walker/docs/TERRAIN_DEMO.md",
        "src/envs/soft_walker/viewer/README.md",
        "src/envs/soft_walker/engine/example.py",
        "tests/soft_walker/test_example.py",
    ]
    runs = [
        "terrain_demo/gallery/index.html",
        "terrain_demo/gallery/overview.png",
        "random_creatures_2d/rendered/index.html",
        "random_creatures_2d/rendered/overview.png",
        "random_creatures_3d/rendered/index.html",
        "random_creatures_3d/rendered/overview.png",
        "newton_renderer_v2_return.html",
        "newton_v2_mpm3_particles/replay.png",
        "forward_adam50/index.html",
        "forward_adam50/biped/best/replay.gif",
        "forward_adam50/quadruped/iterations/007/replay.gif",
        "forward_adam50/implicit2d/best/replay.gif",
        "forward_adam50/implicit2d/hard_final/replay.gif",
        "non_biped_priority/checks.html",
        "non_biped_priority/implicit3d_gate/initial/replay.gif",
        "biped_cadence_study/index.html",
        "biped_cadence_study/cover.png",
        "biped_low_bounce_adam50/iterations/011/replay.gif",
        "pronk_stress_gallery/index.html",
        "pronk_stress_gallery/cover.png",
        "newton_pronk_stress_fast_final/replay.gif",
        "mpm2_adam_to30_cpu_gallery/index.html",
        "mpm2_adam_to30_cpu_gallery/cover.png",
        "newton_adam30_final/replay.gif",
        "terrain_demo/executables/small.json",
        "terrain_demo/executables/compiler.txt",
        ".cache/skipped.gif",
        "gait_reference_20260922/skipped.gif",
        "helper.sh",
    ]
    for relative in sources + ["src/envs/soft_walker/runs/" + name for name in runs]:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"<html></html>" if path.suffix == ".html" else b"fixture\n")
    gallery = root / "src/envs/soft_walker/runs/terrain_demo/gallery/index.html"
    gallery.write_text(
        '<a href="missing.html">old report</a><a href="../../../examples/shared.json">design</a>'
    )
    shared = root / "src/envs/soft_walker/examples/shared.json"
    shared.parent.mkdir(parents=True)
    shared.write_text('{"saved": true}')
    for relative, data in (
        ("forward_adam50/biped/status.json", {"completed": True, "device": "cpu"}),
        ("forward_adam50/biped/optimization/optimization_summary.json", {"accepted_updates": 50}),
        ("non_biped_priority/implicit3d_gate/status.json", {"pilot_passed": True}),
        (
            "biped_low_bounce_adam50/status.json",
            {"paused": True, "accepted_updates": 11, "extra": 9},
        ),
    ):
        path = root / "src/envs/soft_walker/runs" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))
    return sources


def configure_package(
    module: ModuleType, root: Path, sources: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Supply deterministic Git metadata without accessing a real repository."""

    def git(*arguments: str) -> bytes:
        if arguments[0] == "ls-files":
            return "\0".join(sources).encode() + b"\0"
        if arguments[0] == "rev-parse":
            return b"0123456789abcdef\n"
        if arguments[0] == "status":
            return b" M fixture.py\n"
        if arguments[0] == "diff":
            return b"synthetic source patch\n"
        raise AssertionError(arguments)

    monkeypatch.setattr(module, "ROOT", root)
    monkeypatch.setattr(module, "git", git)
    monkeypatch.setattr(module, "datetime", FrozenDatetime)
    monkeypatch.setattr(module, "utc", lambda: "2026-01-02T03:04:05+00:00")


def test_complete_shareable_archive_preserves_selection_and_schemas(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Selection, copied-link repair, source immutability, inventory and ZIP hashes agree."""
    root = tmp_path / "checkout"
    sources = gallery_checkout(root)
    before = {str(path): path.read_bytes() for path in root.rglob("*") if path.is_file()}
    configure_package(package_shareable, root, sources, monkeypatch)
    output = tmp_path / "output"
    monkeypatch.setattr(sys, "argv", ["package_shareable", "--out", str(output)])
    package_shareable.main()
    report = json.loads(capsys.readouterr().out)
    assert set(report) == {
        "archive",
        "extracted_folder",
        "bytes",
        "sha256",
        "files",
        "source_files",
        "media_counts",
        "local_html_links_checked",
        "zip_crc_and_all_hashes_verified",
        "status",
    }
    assert report["zip_crc_and_all_hashes_verified"] is True
    assert report["source_files"] == len(sources)
    assert report["status"]["quadruped"]["latest_recorded_iteration"] == 7
    assert report["status"]["biped"]["accepted_updates"] == 50
    assert report["status"]["implicit3d"]["pilot_passed"]
    assert report["status"]["lower_bounce_biped"] == {"paused": True, "accepted_updates": 11}
    snapshot = Path(report["extracted_folder"])
    manifest = json.loads((snapshot / "src/envs/soft_walker/handoff/MANIFEST.json").read_text())
    assert set(manifest) == {
        "schema",
        "snapshot_started_utc",
        "snapshot_finished_utc",
        "git_commit",
        "contains_uncommitted_changes",
        "source_files",
        "media_counts",
        "included_visual_roots",
        "local_html_links_checked",
        "historical_links_not_available",
        "files",
    }
    assert manifest["schema"] == 1
    assert len(manifest["historical_links_not_available"]) == 1
    files = manifest["files"]
    prefix = "src/envs/soft_walker/"
    assert files[prefix + "examples/shared.json"]["category"] == "gallery_dependency"
    assert files[prefix + "runs/helper.sh"]["category"] == "support"
    assert prefix + "runs/terrain_demo/executables/small.json" in files
    assert prefix + "runs/terrain_demo/executables/compiler.txt" not in files
    assert not any("skipped.gif" in name for name in files)
    assert (
        "(not included)"
        in (snapshot / (prefix + "runs/terrain_demo/gallery/index.html")).read_text()
    )
    archive = Path(report["archive"])
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == report["sha256"]
    with zipfile.ZipFile(archive) as saved:
        assert saved.testzip() is None and len(saved.namelist()) == report["files"]
        for info in saved.infolist():
            relative = info.filename.split("/", 1)[1]
            assert saved.read(info) == (snapshot / relative).read_bytes()
            assert info.create_system == 3
            assert info.external_attr >> 16 == (0o100755 if relative.endswith(".sh") else 0o100644)
    for line in (snapshot / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        assert hashlib.sha256((snapshot / name).read_bytes()).hexdigest() == digest
    assert {str(path): path.read_bytes() for path in root.rglob("*") if path.is_file()} == before


def test_missing_embedded_gallery_asset_still_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only unavailable download anchors are repaired; missing embedded media remains fatal."""
    root = tmp_path / "checkout"
    sources = gallery_checkout(root)
    page = root / "src/envs/soft_walker/runs/terrain_demo/gallery/index.html"
    page.write_text('<img src="missing.png">')
    configure_package(package_shareable, root, sources, monkeypatch)
    monkeypatch.setattr(sys, "argv", ["package_shareable", "--out", str(tmp_path / "output")])
    with pytest.raises(RuntimeError, match="Unresolved HTML dependencies"):
        package_shareable.main()
    assert not list((tmp_path / "output").glob("*.zip"))


def test_changed_source_still_blocks_manifest_and_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stable first read does not excuse source changes later in packaging."""
    source = tmp_path / "source.py"
    source.write_bytes(b"before")
    monkeypatch.setattr(package_shareable, "ROOT", tmp_path)
    package = package_shareable.PackageSnapshot(tmp_path / "copy")
    package.add("source.py", "source")
    source.write_bytes(b"after")
    with pytest.raises(RuntimeError, match="Source changed during packaging"):
        package_shareable.validate_source_hashes(package)
