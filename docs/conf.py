"""Sphinx configuration for the Codesign Gym docs (built on Read the Docs, see ../.readthedocs.yaml).

Environment pages include each environment's README from ``src/envs/<env>/README.md`` so the README stays the single
source. Before the build reads any page, this file copies those READMEs into ``_generated/readmes`` and rewrites their
relative links (other env READMEs -> that env's docs page, anything else -> GitHub). The Pokémon package docs live in
the pokemon-codesign-env repo; they are copied into ``envs/pokenv/package`` from ``POKEMON_DOCS_PATH`` (a local
checkout) or, when that is unset, from a shallow clone of ``POKEMON_DOCS_REF``.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

DOCS = Path(__file__).resolve().parent
ROOT = DOCS.parent
sys.path.insert(0, str(ROOT / "src"))

GITHUB = "https://github.com/camral/codesign-gym/blob/main"
POKEMON_REPO = "https://github.com/camral/pokemon-codesign-env"
POKEMON_DOCS_REF = os.environ.get("POKEMON_DOCS_REF", "gym-integration")

project = "Codesign Gym"
author = "CamRAL"
copyright = "2026, CamRAL"
release = "1.0.0"

extensions = ["myst_parser", "sphinx.ext.autodoc", "sphinx.ext.napoleon", "sphinx.ext.viewcode", "sphinx_copybutton", "sphinx_design"]
myst_enable_extensions = ["colon_fence", "deflist", "html_image"]
myst_heading_anchors = 3
source_suffix = {".md": "markdown"}
exclude_patterns = ["_build", "_generated", "requirements.txt"]

# The docs never import an environment, only codesign_gym and core; anything heavy they might touch is mocked.
autodoc_mock_imports = ["torch", "jax", "jaxlib", "warp", "mujoco", "mujoco_warp"]
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_inherit_docstrings = False  # gymnasium's own docstrings do not parse cleanly; the API page documents return values

html_theme = "furo"
html_title = "Codesign Gym"
html_theme_options = {"source_repository": "https://github.com/camral/codesign-gym", "source_branch": "main", "source_directory": "docs/"}
copybutton_prompt_text = r">>> |\$ "
copybutton_prompt_is_regexp = True

# Env folder -> docs section. Folders not listed (the seven Gym ports) map to the shared gym_mujoco section.
ENV_SECTIONS = {name: name for name in ("ball_catcher", "panel_cleaner", "truck_unload", "nerograsp", "rotate_in_hand", "hand_pick_up", "soft_walker",
                                        "lq_structure", "microgrid", "mobile_env", "f1tenth", "robot_warehouse", "pokenv")}
GYM_PORTS = ("half_cheetah", "swimmer", "hopper", "walker2d", "ant", "humanoid", "humanoid_standup")
LINK = re.compile(r"(!?\[[^\]]*\])\(([^)\s]+)\)")


def _env_page(folder: str) -> str | None:
    if folder in ENV_SECTIONS:
        return f"/envs/{folder}/index.md"
    if folder in GYM_PORTS:
        return f"/envs/gym_mujoco/{folder}.md"
    return None


def _rewrite_env_links(text: str, folder: str) -> str:
    """Point a README's relative links at docs pages (other env READMEs) or GitHub (everything else)."""
    def fix(match: re.Match) -> str:
        label, target = match.groups()
        if re.match(r"[a-z]+:", target) or target.startswith("#"):
            return match.group(0)
        path, _, anchor = target.partition("#")
        resolved = (ROOT / "src" / "envs" / folder / path).resolve()
        rel = resolved.relative_to(ROOT).as_posix()
        # soft_walker's README.md is the engine handoff doc (its env page comes from ENVIRONMENT.md), so it stays a GitHub link.
        page = _env_page(resolved.parent.name) if resolved.name == "README.md" and resolved.parent.parent == ROOT / "src" / "envs" and resolved.parent.name != "soft_walker" else None
        if page is None and resolved.name == "ENVIRONMENT.md":
            page = "/envs/soft_walker/index.md"
        if page:
            return f"{label}({page}{'#' + anchor if anchor else ''})"
        kind = "tree" if resolved.is_dir() else "blob"
        return f"{label}({GITHUB.replace('/blob/', f'/{kind}/')}/{rel}{'#' + anchor if anchor else ''})"
    return LINK.sub(fix, text)


def _copy_env_readmes() -> None:
    out = DOCS / "_generated" / "readmes"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    for folder in (*ENV_SECTIONS, *GYM_PORTS):
        src = ROOT / "src" / "envs" / folder / ("ENVIRONMENT.md" if folder == "soft_walker" else "README.md")
        text = src.read_text(encoding="utf-8").split("\n", 1)[1].lstrip("\n")  # the docs page supplies the title
        (out / f"{folder}.md").write_text(_rewrite_env_links(text, folder), encoding="utf-8")


# Pokémon package docs: upstream path -> page name in envs/pokenv/package.
POKEMON_PAGES = {"README.md": "guide.md", "docs/codesign_gym_api.md": "design-space-api.md", "src/pokemon_codesign_env/envs/README.md": "env-classes.md"}
POKEMON_URLS = {f"{POKEMON_REPO}/blob/main/{path}": f"/envs/pokenv/package/{page}" for path, page in POKEMON_PAGES.items()}


def _pokemon_source() -> Path:
    local = os.environ.get("POKEMON_DOCS_PATH")
    if local:
        return Path(local).expanduser().resolve()
    clone = DOCS / "_generated" / "pokemon-codesign-env"
    if not (clone / "README.md").exists():
        shutil.rmtree(clone, ignore_errors=True)
        subprocess.run(["git", "clone", "--depth", "1", "--branch", POKEMON_DOCS_REF, f"{POKEMON_REPO}.git", str(clone)], check=True)
    return clone


def _copy_pokemon_docs() -> None:
    src = _pokemon_source()
    out = DOCS / "envs" / "pokenv" / "package"
    shutil.rmtree(out, ignore_errors=True)
    (out / "figures").mkdir(parents=True)
    for upstream, page in POKEMON_PAGES.items():
        text = (src / upstream).read_text(encoding="utf-8")
        # <p align="center"><img src=... alt=...></p> -> a Markdown image Sphinx copies into the build.
        text = re.sub(r'<p align="center">\s*<img src="([^"]+)" alt="([^"]*)"[^>]*>\s*</p>', lambda m: f"![{m.group(2)}]({m.group(1)})", text)
        base = Path(upstream).parent

        def fix(match: re.Match) -> str:
            label, target = match.groups()
            if re.match(r"[a-z]+:", target) or target.startswith("#"):
                return match.group(0)
            path, _, anchor = target.partition("#")
            rel = os.path.normpath(base / path).replace(os.sep, "/")
            if rel in POKEMON_PAGES:
                return f"{label}({POKEMON_PAGES[rel]}{'#' + anchor if anchor else ''})"
            if label.startswith("!") and rel.startswith("docs/figures/"):
                shutil.copy2(src / rel, out / "figures" / Path(rel).name)
                return f"{label}(figures/{Path(rel).name})"
            kind = "tree" if (src / rel).is_dir() else "blob"
            return f"{label}({POKEMON_REPO}/{kind}/main/{rel}{'#' + anchor if anchor else ''})"
        (out / page).write_text(LINK.sub(fix, text), encoding="utf-8")


def _point_pokemon_links_in_site(app, docname, source) -> None:
    """Send links to the Pokémon docs on GitHub to the copies hosted in this site."""
    for url, page in POKEMON_URLS.items():
        source[0] = source[0].replace(f"]({url}", f"]({page}")


# DOCS_SKIP_GENERATE=1 reuses the copies from an earlier build (for several builds running at once).
if not os.environ.get("DOCS_SKIP_GENERATE"):
    _copy_env_readmes()
    _copy_pokemon_docs()


def _repo_links_to_github(text: str, base: Path) -> str:
    """Docs pages link to repository files (e.g. ``src/envs/...``) by relative path, so the Markdown also works when the
    repository is browsed directly; the built site sends links that leave ``docs/`` to GitHub instead."""
    def fix(match: re.Match) -> str:
        label, target = match.groups()
        if re.match(r"[a-z]+:", target) or target.startswith(("#", "/")):
            return match.group(0)
        path, _, anchor = target.partition("#")
        resolved = (base / path).resolve()
        if resolved.is_relative_to(DOCS) or not resolved.is_relative_to(ROOT):
            return match.group(0)
        kind = "tree" if resolved.is_dir() else "blob"
        return f"{label}({GITHUB.replace('/blob/', f'/{kind}/')}/{resolved.relative_to(ROOT).as_posix()}{'#' + anchor if anchor else ''})"
    return LINK.sub(fix, text)


def _source_read(app, docname, source) -> None:
    _point_pokemon_links_in_site(app, docname, source)
    source[0] = _repo_links_to_github(source[0], (DOCS / docname).parent)


def setup(app):
    app.connect("source-read", _source_read)
    app.connect("include-read", lambda app, path, docname, source: _point_pokemon_links_in_site(app, docname, source))
