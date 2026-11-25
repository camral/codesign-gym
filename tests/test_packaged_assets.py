"""Exercise release archives: editable installs hide missing package data."""
from pathlib import Path
import os
import shutil
import subprocess
import sys
import tarfile
import zipfile


ROOT = Path(__file__).resolve().parents[1]


def test_runtime_assets_survive_sdist_and_wheel(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    shutil.copytree(ROOT / "src", source / "src", ignore=shutil.ignore_patterns(
        "__pycache__", "*.egg-info", "*.pyc"))
    for name in ("pyproject.toml", "README.md", "LICENSE"):
        shutil.copy2(ROOT / name, source / name)
    expected = {
        p.relative_to(ROOT / "src").as_posix()
        for p in (ROOT / "src" / "envs").rglob("*")
        if p.is_file() and "assets" in p.parts and "__pycache__" not in p.parts
    }
    assert expected
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}

    def run(code, cwd):
        result = subprocess.run([sys.executable, "-c", code], cwd=cwd, env=env,
                                capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr

    dist = tmp_path / "dist"
    dist.mkdir()
    run(f"from setuptools.build_meta import build_sdist; build_sdist({str(dist)!r})", source)
    with tarfile.open(next(dist.glob("*.tar.gz"))) as archive:
        names = {"/".join(Path(n).parts[2:]) for n in archive.getnames() if "/src/" in n}
        assert not expected - names, sorted(expected - names)
        archive.extractall(tmp_path / "unpacked", filter="data")
    unpacked = next((tmp_path / "unpacked").iterdir())
    run(f"from setuptools.build_meta import build_wheel; build_wheel({str(dist)!r})", unpacked)
    with zipfile.ZipFile(next(dist.glob("*.whl"))) as archive:
        assert not expected - set(archive.namelist()), sorted(expected - set(archive.namelist()))
        installed = tmp_path / "installed"
        archive.extractall(installed)
    # Import the installed package away from the source checkout, then resolve all
    # asset bytes relative to it. No optional simulator imports are needed.
    run(f"import sys; sys.path.insert(0, {str(installed)!r}); "
        "import envs; from pathlib import Path; "
        "root = Path(envs.__file__).parent.parent; "
        f"assert all((root / name).is_file() for name in {sorted(expected)!r})", tmp_path)
