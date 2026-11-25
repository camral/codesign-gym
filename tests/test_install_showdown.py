"""Unit tests for the Showdown setup helper (``codesign_gym.install_showdown``).

These cover the auto-fetch fallback used when ``pip install '.[showdown]'`` leaves no Showdown
checkout and no submodule to initialize (pip does not recurse git submodules). They stub out git
and the package metadata, so they need neither network, Node, nor pokemon-codesign-env installed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from codesign_gym import install_showdown as inst


# --- _pce_git_pin ---------------------------------------------------------------------------------

def _fake_direct_url(monkeypatch, payload):
    """Make Distribution.from_name('pokemon-codesign-env').read_text('direct_url.json') return payload."""
    class _Dist:
        def read_text(self, name):
            assert name == "direct_url.json"
            return payload

    monkeypatch.setattr(inst, "Distribution", type("D", (), {"from_name": staticmethod(lambda _n: _Dist())}))


def test_pin_none_for_editable_install(monkeypatch):
    _fake_direct_url(monkeypatch, json.dumps({"dir_info": {"editable": True}, "url": "file:///x"}))
    assert inst._pce_git_pin() is None


def test_pin_prefers_commit_id(monkeypatch):
    _fake_direct_url(monkeypatch, json.dumps({
        "url": "https://example.com/pce.git",
        "vcs_info": {"vcs": "git", "requested_revision": "gym-integration", "commit_id": "abc123"},
    }))
    assert inst._pce_git_pin() == ("https://example.com/pce.git", "abc123")


def test_pin_none_when_not_installed(monkeypatch):
    def _raise(_name):
        raise inst.PackageNotFoundError
    monkeypatch.setattr(inst, "Distribution", type("D", (), {"from_name": staticmethod(_raise)}))
    assert inst._pce_git_pin() is None


# --- _fetch_showdown_from_source ------------------------------------------------------------------

def test_fetch_relocates_tree_and_cleans_up(monkeypatch, tmp_path):
    monkeypatch.setattr(inst.shutil, "which", lambda _n: "/usr/bin/git")
    monkeypatch.setattr(inst, "_pce_git_pin", lambda: ("https://example.com/pce.git", "abc123"))

    scratch = tmp_path / "scratch"

    def fake_mkdtemp(prefix=""):
        scratch.mkdir()
        return str(scratch)

    monkeypatch.setattr(inst.tempfile, "mkdtemp", fake_mkdtemp)

    calls = []

    def fake_run(argv, cwd=None, check=False):
        calls.append((argv, cwd))
        if argv[:2] == ["git", "clone"]:  # clone creates the pce dir + Showdown submodule tree
            target = Path(argv[-1])
            sd = target / "pokemon-showdown"
            sd.mkdir(parents=True)
            (sd / "package.json").write_text("{}")
            (sd / ".git").write_text("gitdir: ../.git/modules/pokemon-showdown\n")

    monkeypatch.setattr(inst.subprocess, "run", fake_run)

    dest = tmp_path / "env" / "pokemon-showdown"
    inst._fetch_showdown_from_source(dest)

    assert (dest / "package.json").is_file()          # relocated to where the package looks
    assert not (dest / ".git").exists()               # dangling gitlink dropped
    assert not scratch.exists()                       # scratch cleaned up
    assert any(c[0][:2] == ["git", "submodule"] for c in calls)  # submodules were materialized


def test_fetch_falls_back_to_full_clone(monkeypatch, tmp_path):
    monkeypatch.setattr(inst.shutil, "which", lambda _n: "/usr/bin/git")
    monkeypatch.setattr(inst, "_pce_git_pin", lambda: ("https://example.com/pce.git", "abc123"))
    scratch = tmp_path / "scratch"
    monkeypatch.setattr(inst.tempfile, "mkdtemp", lambda prefix="": (scratch.mkdir() or str(scratch)))

    clones = []

    def fake_run(argv, cwd=None, check=False):
        if argv[:2] == ["git", "clone"]:
            clones.append(argv)
            if "--filter=blob:none" in argv:  # server rejects partial clone
                raise inst.subprocess.CalledProcessError(1, argv)
            target = Path(argv[-1])
            (target / "pokemon-showdown").mkdir(parents=True)
            (target / "pokemon-showdown" / "package.json").write_text("{}")

    monkeypatch.setattr(inst.subprocess, "run", fake_run)

    inst._fetch_showdown_from_source(tmp_path / "env" / "pokemon-showdown")

    assert len(clones) == 2 and "--filter=blob:none" not in clones[1]  # retried without the filter


def test_fetch_exits_when_not_a_git_install(monkeypatch, tmp_path):
    monkeypatch.setattr(inst.shutil, "which", lambda _n: "/usr/bin/git")
    monkeypatch.setattr(inst, "_pce_git_pin", lambda: None)
    with pytest.raises(SystemExit) as exc:
        inst._fetch_showdown_from_source(tmp_path / "pokemon-showdown")
    assert "recurse-submodules" in str(exc.value)


def test_fetch_exits_without_git(monkeypatch, tmp_path):
    monkeypatch.setattr(inst.shutil, "which", lambda _n: None)
    with pytest.raises(SystemExit) as exc:
        inst._fetch_showdown_from_source(tmp_path / "pokemon-showdown")
    assert "git" in str(exc.value)
