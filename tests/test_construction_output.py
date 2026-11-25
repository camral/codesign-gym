"""Startup noise must not hide actionable warnings or alter global logging."""
import os
from pathlib import Path
import subprocess
import sys
import warnings

import pytest

from core.construction import warp_construction


@pytest.mark.parametrize("verbose", [False, True])
@pytest.mark.parametrize("fails", [False, True])
def test_warp_logging_scope_restores_settings(verbose, fails):
    import warp as wp
    attr = "log_level" if hasattr(wp.config, "log_level") else "quiet"
    previous = getattr(wp.config, attr)

    @warp_construction
    def build():
        if attr == "log_level":
            assert wp.config.log_level == (wp.LOG_INFO if verbose else max(previous, wp.LOG_WARNING))
        else:
            assert wp.config.quiet == (not verbose)
        warnings.warn("An actionable warning", UserWarning)
        if fails:
            raise ValueError("construction failed")
        return "env"

    with pytest.warns(UserWarning, match="actionable"):
        if fails:
            with pytest.raises(ValueError, match="construction failed"):
                build(verbose=verbose)
        else:
            assert build(verbose=verbose) == "env"
    assert getattr(wp.config, attr) == previous


def test_warp_legacy_logging(monkeypatch):
    from types import SimpleNamespace
    fake = SimpleNamespace(config=SimpleNamespace(quiet=False))
    monkeypatch.setitem(sys.modules, "warp", fake)

    @warp_construction
    def build():
        return fake.config.quiet

    assert build() is True
    assert build(verbose=True) is False
    assert fake.config.quiet is False


def fresh_process(code, tmp_path):
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
           "MPLCONFIGDIR": str(tmp_path / "mpl"), "XDG_CACHE_HOME": str(tmp_path / "cache")}
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                            cwd=tmp_path, env=env, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


@pytest.mark.parametrize("factory", ["make('Microgrid-Random')", "make_vec('Microgrid-Random', 2)"])
def test_microgrid_has_no_legacy_gym_banner(factory, tmp_path):
    result = fresh_process(
        "from gym_notices.notices import notices; before = dict(notices); "
        f"import codesign_gym as cg; env, base = cg.{factory}; "
        "print(env.embodiment_space); env.close(); assert notices == before; "
        "import warnings; warnings.warn('visible warning after construction')", tmp_path)
    assert "Gym has been unmaintained" not in result.stderr
    assert "visible warning after construction" in result.stderr
    assert "battery_capacity_kwh" in result.stdout


@pytest.mark.parametrize("verbose", [False, True])
def test_real_warp_construction_logging(verbose, tmp_path):
    result = fresh_process(
        "import codesign_gym as cg; "
        f"env, base = cg.make('HalfCheetah-v4', device='cpu', verbose={verbose}); "
        "print(env.embodiment_space); env.close()", tmp_path)
    assert ("Warp " in result.stdout) == verbose
    assert (" load on device " in result.stdout) == verbose
    assert "Box(" in result.stdout


def test_microgrid_notice_registry_restored_on_import_failure(monkeypatch):
    import importlib
    from gym_notices.notices import notices
    from envs.microgrid._compat import import_pymgrid
    before = dict(notices)

    def fail(name):
        assert name == "pymgrid"
        assert not any(text.startswith("Gym has been unmaintained since 2022") for text in notices.values())
        warnings.warn("Dependency failure details", UserWarning)
        raise ImportError("dependency failure")

    monkeypatch.setattr(importlib, "import_module", fail)
    with pytest.warns(UserWarning, match="failure details"), pytest.raises(ImportError, match="dependency failure"):
        import_pymgrid()
    assert notices == before


def test_warp_verbose_forwarded_only_when_requested():
    from types import SimpleNamespace
    import warp as wp

    @warp_construction(forward_verbose=True)
    def build(**kwargs):
        if hasattr(wp.config, "log_level"):
            assert wp.config.log_level == wp.LOG_INFO
        else:
            assert wp.config.quiet is False
        return kwargs

    assert build(verbose=True) == {"verbose": True}
    config = SimpleNamespace(verbose=True)
    assert build(config=config) == {"config": config}
