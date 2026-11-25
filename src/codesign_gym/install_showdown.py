"""
Authors: Aviraj Newatia

Console script: build the bundled Pokemon Showdown server for the Pokemon co-design env.

``pip install`` handles the Python packages, but the Pokemon Showdown env also needs its bundled
Node server, which is not a Python artifact and so cannot be shipped in a wheel or built during
pip's (Node-less, network-isolated) build phase:

  1. the Pokemon Showdown source -- a git submodule of ``pokemon-codesign-env``;
  2. its npm dependencies (``npm ci``);
  3. a TypeScript build plus the packaged co-design League formats (``node build``).

This command runs those three steps on demand, after ``pip install``. It needs Node.js >= 22 and
npm on PATH; everything else it locates from the installed ``pokemon-codesign-env`` package.

    codesign-gym-install-showdown

A ``pip install '.[showdown]'`` pulls ``pokemon-codesign-env`` from git, but pip does not recurse
git submodules -- so the bundled Showdown source never reaches site-packages and there is no
``.gitmodules`` there to initialize. In that case this command fetches the source automatically:
it reads the git pin ``pokemon-codesign-env`` was installed from (its PEP 610 ``direct_url.json``),
clones that pin *with* submodules into a scratch dir, and moves the Showdown checkout to where the
installed package looks for it. No URL or commit is hardcoded here.

Re-running is safe: the source, submodule and npm steps are skipped when already present, and the
format install is idempotent.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from importlib.metadata import Distribution, PackageNotFoundError
from pathlib import Path

MIN_NODEJS = 22


def _node_version() -> tuple[int, int, int] | None:
    """Return the installed Node.js version as a tuple, or None if Node is not on PATH."""
    node = shutil.which("node")
    if node is None:
        return None
    result = subprocess.run([node, "--version"], capture_output=True, text=True, check=False)
    match = re.match(r"v?(\d+)\.(\d+)\.(\d+)", result.stdout.strip())
    return (int(match[1]), int(match[2]), int(match[3])) if match else None


def _require_node() -> None:
    version = _node_version()
    if version is None:
        sys.exit(
            "Node.js was not found on PATH. The bundled Pokemon Showdown server needs Node.js "
            f">= {MIN_NODEJS}; install it (https://nodejs.org) and run this again."
        )
    if version[0] < MIN_NODEJS:
        pretty = ".".join(str(part) for part in version)
        sys.exit(
            f"Node.js {pretty} is too old for the bundled Pokemon Showdown, which needs "
            f">= {MIN_NODEJS}. Upgrade Node.js and run this again."
        )


def _resolve_showdown_path(explicit: Path | None) -> Path:
    """Locate the Pokemon Showdown checkout, defaulting to the one bundled with the package."""
    if explicit is not None:
        return explicit.resolve()
    try:
        from pokemon_codesign_env.gym_utils import default_showdown_path
    except ImportError:
        sys.exit(
            "pokemon-codesign-env is not installed. Install the Pokemon extra first, e.g.\n"
            "  pip install -e '.[showdown]'\n"
            "or clone camral/pokemon-codesign-env (with --recurse-submodules) and pip install it."
        )
    return default_showdown_path()


def _pce_git_pin() -> tuple[str, str] | None:
    """The ``(git url, ref)`` pokemon-codesign-env was installed from, or ``None``.

    Read from the package's PEP 610 ``direct_url.json`` so the Showdown source is fetched from the
    exact pin pip resolved -- nothing is duplicated here. Returns ``None`` when the package was not
    installed from git (an editable/local dir or a plain sdist have no ``vcs_info``).
    """
    try:
        raw = Distribution.from_name("pokemon-codesign-env").read_text("direct_url.json")
    except PackageNotFoundError:
        return None
    if not raw:
        return None
    info = json.loads(raw)
    vcs = info.get("vcs_info")
    if not vcs or vcs.get("vcs") != "git":
        return None
    url = info.get("url")
    ref = vcs.get("commit_id") or vcs.get("requested_revision")
    return (url, ref) if url and ref else None


def _git(*args: str, cwd: Path | None = None) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True)


def _fetch_showdown_from_source(showdown_path: Path) -> None:
    """Fetch the Pokemon Showdown source when pip stripped it from pokemon-codesign-env.

    A ``pip install`` of ``pokemon-codesign-env`` copies only the Python package: pip does not
    recurse git submodules, so the bundled Showdown checkout is absent and there is no
    ``.gitmodules`` to initialize. Recover it by cloning ``pokemon-codesign-env`` (at the pin it was
    installed from) *with* its submodules into a scratch dir, then moving the Showdown working tree
    into the location the installed package looks for it (:func:`default_showdown_path`).
    """
    if shutil.which("git") is None:
        sys.exit("git was not found on PATH; it is needed to fetch the Pokemon Showdown source.")
    pin = _pce_git_pin()
    if pin is None:
        sys.exit(
            f"No Pokemon Showdown checkout at {showdown_path}, and pokemon-codesign-env was not "
            "installed from git so its Showdown source cannot be located automatically.\n"
            "Reinstall it from git with submodules, e.g.\n"
            "  git clone --recurse-submodules -b gym-integration "
            "https://github.com/camral/pokemon-codesign-env.git\n"
            "  pip install -e ./pokemon-codesign-env\n"
            "then run this again, or pass --showdown-path to an existing Showdown checkout."
        )
    url, ref = pin
    scratch = Path(tempfile.mkdtemp(prefix="codesign-showdown-"))
    try:
        clone = scratch / "pokemon-codesign-env"
        print(f"Fetching the Pokemon Showdown source via pokemon-codesign-env ({url}@{ref}) ...", flush=True)
        try:
            # Partial clone: only the commits/trees are needed to reach the submodule pin.
            _git("clone", "--filter=blob:none", url, str(clone))
        except subprocess.CalledProcessError:
            shutil.rmtree(clone, ignore_errors=True)  # server without partial-clone support
            _git("clone", url, str(clone))
        try:
            _git("checkout", ref, cwd=clone)
        except subprocess.CalledProcessError:
            print(f"  could not check out {ref}; using the cloned default branch instead.", flush=True)
        _git("submodule", "update", "--init", "--recursive", cwd=clone)

        fetched = clone / "pokemon-showdown"
        if not (fetched / "package.json").is_file():
            sys.exit(f"The pokemon-codesign-env clone did not contain a Showdown checkout at {fetched}.")
        showdown_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(fetched), str(showdown_path))
        # The moved tree keeps a dangling submodule gitlink file; drop it (the build needs only files).
        gitlink = showdown_path / ".git"
        if gitlink.is_file():
            gitlink.unlink()
        print(f"Pokemon Showdown source placed at {showdown_path}", flush=True)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--showdown-path", type=Path, default=None,
        help="Path to the Pokemon Showdown checkout; defaults to the one bundled with pokemon-codesign-env.",
    )
    parser.add_argument("--skip-npm", action="store_true", help="Skip 'npm ci' (npm dependencies already installed).")
    args = parser.parse_args()

    _require_node()
    showdown_path = _resolve_showdown_path(args.showdown_path)
    repo_root = showdown_path.parent

    # 1. Ensure the Pokemon Showdown source is checked out. An editable/recursive clone has a
    #    submodule to initialize; a pip-from-git install has neither the checkout nor .gitmodules,
    #    so the source is fetched from the pin pokemon-codesign-env was installed from.
    if not (showdown_path / "package.json").is_file():
        if (repo_root / ".gitmodules").is_file():
            print(f"Initializing the Pokemon Showdown submodule under {repo_root} ...", flush=True)
            subprocess.run(["git", "submodule", "update", "--init", "--recursive"], cwd=repo_root, check=True)
        else:
            _fetch_showdown_from_source(showdown_path)
    if not (showdown_path / "package.json").is_file():
        sys.exit(f"Still no Pokemon Showdown checkout at {showdown_path}; cannot continue.")

    # 2. npm dependencies (skip if already installed, unless forced off).
    if args.skip_npm:
        print("Skipping npm ci (--skip-npm).", flush=True)
    elif (showdown_path / "node_modules").is_dir():
        print("npm dependencies already present; skipping npm ci (delete node_modules to force).", flush=True)
    else:
        npm = shutil.which("npm")
        if npm is None:
            sys.exit("npm was not found on PATH (it ships with Node.js). Install Node.js and run this again.")
        print("Installing Pokemon Showdown npm dependencies (npm ci) ...", flush=True)
        subprocess.run([npm, "ci"], cwd=showdown_path, check=True)

    # 3. Install the co-design League formats and build the server (node build).
    from pokemon_codesign_env.showdown_extension.install import install_showdown_extension

    print("Installing the co-design League formats and building the server (node build) ...", flush=True)
    install_showdown_extension(showdown_path, build=True)
    print(f"\nPokemon Showdown server is ready at {showdown_path}", flush=True)


if __name__ == "__main__":
    main()
