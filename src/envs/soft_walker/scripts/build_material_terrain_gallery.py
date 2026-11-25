#!/usr/bin/env python3
"""Assemble saved material/terrain demos into an offline, shareable gallery.

This command only reads finished demo manifests. It never runs a simulation.
"""

from __future__ import annotations

import argparse

# Resolve integrated imports independently of the caller's working directory.
import sys as _script_sys
from pathlib import Path, Path as _ScriptPath
from typing import TYPE_CHECKING, Any

_source_root = str(_ScriptPath(__file__).resolve().parents[3])
if _source_root not in _script_sys.path:
    _script_sys.path.insert(0, _source_root)
_script_dir = str(_ScriptPath(__file__).resolve().parent)
if _script_dir not in _script_sys.path:
    _script_sys.path.insert(0, _script_dir)
import json
import shutil
import zipfile
from datetime import datetime, timezone
from html import escape
from urllib.parse import quote

from envs.soft_walker.shared import (
    ENGINE_ROOT as ENGINE_ROOT,
    REPO_ROOT as REPO_ROOT,
    SCRIPTS_ROOT as SCRIPTS_ROOT,
    SOFT_ROOT as SOFT_ROOT,
    SRC_ROOT as SRC_ROOT,
    TESTS_ROOT as TESTS_ROOT,
)

if TYPE_CHECKING:
    from pathlib import Path


ROOT = REPO_ROOT
LABELS = {
    "field": "Decoded material field",
    "quasistatic": "Quasi-static FEM solve",
    "physics": "Dynamic spring simulation",
}

# Serialized reports can contain backend-specific auxiliary diagnostic trees.
Report = dict[str, Any]


def collect(root: Path) -> tuple[list[Report], list[Report]]:
    """Collect available result metadata and relative asset links for the gallery."""
    cards, reports = [], []
    for group in ("materials", "mechanics", "terrain"):
        folder = root / group
        manifest = folder / "manifest.json"
        if not manifest.is_file():
            raise FileNotFoundError(f"Demo generation is unfinished: {manifest}")
        data = json.loads(manifest.read_text(encoding="utf-8-sig"))
        if not data.get("items"):
            raise ValueError(f"No demo cards in {manifest}")
        if group == "mechanics" and data.get("all_frames_converged") is not True:
            raise ValueError("The mechanical comparison has not passed equilibrium checks")
        reports.append({"group": group, "manifest": f"{group}/manifest.json"})
        for entry in data["items"]:
            item: Report = dict(entry, group=group)
            item.setdefault(
                "kind",
                {"materials": "field", "mechanics": "quasistatic", "terrain": "physics"}[group],
            )
            if item["kind"] == "physics" and item.get("status") != 0:
                raise ValueError(f"Terrain demo has not completed successfully: {item['id']}")

            def asset(relative: str) -> str:
                target = (folder / relative).resolve()
                if not target.is_relative_to(root.resolve()) or not target.is_file():
                    raise ValueError(f"Missing or unsafe gallery asset: {relative}")
                return target.relative_to(root.resolve()).as_posix()

            item["image"] = asset(item["image"])
            if item.get("animation"):
                item["animation"] = asset(item["animation"])
            item["links"] = [dict(link, href=asset(link["href"])) for link in item.get("links", [])]
            item["links"].append({"label": "Demo record", "href": f"{group}/manifest.json"})
            cards.append(item)
    order = {"mechanics": 0, "terrain": 1, "materials": 2}
    return sorted(cards, key=lambda item: order[item["group"]]), reports


def card(item: Report) -> str:
    """Render an artifact preview card linked to the underlying evidence."""
    kind = item["kind"]
    ident = escape(item["group"] + "-" + str(item["id"]), quote=True)
    image = quote(item["image"], safe="/")
    animation = quote(item.get("animation", ""), safe="/")
    motion = (
        (
            f'<button class="motion" type="button" aria-pressed="false" data-still="{image}" '
            f'data-motion="{animation}">Play animation</button>'
        )
        if animation
        else '<span class="still-label">Still preview</span>'
    )
    links = "".join(
        f'<a href="{quote(link["href"], safe="/")}" download>{escape(link["label"])}</a>'
        for link in item["links"]
    )
    return f'''<article class="card" id="{ident}" data-kind="{escape(kind)}" data-dim="{int(item.get("dim", 2))}">
      <div class="card-media"><button class="zoom" aria-label="Enlarge {escape(item["title"], quote=True)}" type="button">
        <img src="{image}" alt="{escape(item["title"], quote=True)}" loading="lazy" width="1000" height="640">
        <span class="enlarge">Enlarge ↗</span></button></div>
      <div class="card-copy"><div class="eyebrow"><span class="dot {escape(kind)}"></span>{escape(LABELS[kind])}<span class="dim">{int(item.get("dim", 2))}D</span></div>
      <h3>{escape(item["title"])}</h3><p>{escape(item.get("description", ""))}</p>
      <div class="controls">{motion}<a href="{animation or image}" target="_blank" rel="noopener">Open file ↗</a></div>
      <details><summary>Designs &amp; numerical data</summary><div class="data-links">{links}</div></details></div></article>'''


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="dark"><title>Material / Motion — Co-design gallery</title>
<style>
:root{--bg:#10171f;--panel:#18232d;--line:#2b3a45;--text:#e8eef2;--muted:#a8bac6;--cyan:#75d7ce;--coral:#ffa680;--purple:#c4b4f5}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:16px/1.6 system-ui,-apple-system,"Segoe UI",sans-serif}a{color:var(--cyan);text-decoration:none}a:hover{text-decoration:underline}button,select{font:inherit}button{cursor:pointer}button:focus-visible,a:focus-visible,summary:focus-visible,select:focus-visible{outline:3px solid var(--cyan);outline-offset:4px}button:disabled{cursor:default;opacity:.5}
.wrap{max-width:1460px;margin:0 auto;padding:0 42px}header{display:flex;align-items:center;justify-content:space-between;padding:24px 0;border-bottom:1px solid var(--line);gap:16px}.brand{font-size:14px;font-weight:700;letter-spacing:.16em;text-transform:uppercase}.brand b{color:var(--cyan)}header nav{display:flex;gap:24px;font-size:14px}.hero{padding:66px 0 38px;display:grid;grid-template-columns:1fr 1.04fr;gap:52px;align-items:center}.kicker{font-size:12px;text-transform:uppercase;letter-spacing:.2em;color:var(--cyan);font-weight:700}h1{font-size:clamp(38px,4.6vw,68px);line-height:1.06;letter-spacing:-.048em;font-weight:600;margin:18px 0 24px}h1 em{font-style:normal;color:var(--cyan)}.intro{color:var(--muted);max-width:580px;font-size:17px;line-height:1.8}.hero-image{margin:0;border:1px solid var(--line);border-radius:20px;overflow:hidden;background:#15232e;box-shadow:0 25px 80px #0003}.hero-image img{display:block;width:100%;height:auto}.hero-image figcaption{padding:14px 20px;color:var(--muted);font-size:13px}.stats{display:flex;gap:36px;margin-top:32px}.stat strong{font-size:26px;display:block;font-weight:550}.stat span{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.09em}
.notes{display:grid;grid-template-columns:1fr 1fr 1fr;gap:22px;margin:28px 0 44px}.note{border-top:2px solid var(--line);padding-top:18px}.note:nth-child(1){border-color:var(--cyan)}.note:nth-child(2){border-color:var(--purple)}.note:nth-child(3){border-color:var(--coral)}.note h2{font-size:16px;margin:0 0 8px;font-weight:600}.note p{font-size:13px;color:var(--muted);margin:0;line-height:1.7}.legend{display:flex;align-items:center;gap:20px;padding:20px 24px;border:1px solid var(--line);border-radius:12px;flex-wrap:wrap}.legend-title{font-size:14px;font-weight:600}.legend-scale{width:260px}.gradient{height:8px;border-radius:4px;background:linear-gradient(90deg,#440154,#3b528b,#21918c,#5ec962,#fde725)}.ticks{display:flex;justify-content:space-between;color:var(--muted);font-size:11px;margin-top:4px}.legend p{font-size:12px;color:var(--muted);margin:0;flex:1;min-width:220px}
.gallery-heading{display:flex;justify-content:space-between;align-items:end;gap:24px;margin:50px 0 18px}.gallery-heading h2{font-size:28px;margin:0;font-weight:550;letter-spacing:-.03em}.gallery-heading p{font-size:13px;color:var(--muted);margin:4px 0 0}.toolbar{display:flex;align-items:center;justify-content:space-between;gap:18px;margin:0 0 26px;flex-wrap:wrap}.tabs{display:flex;gap:8px;flex-wrap:wrap}.tabs button,.secondary,select{background:transparent;border:1px solid var(--line);color:var(--muted);border-radius:24px;padding:8px 15px;font-size:13px}.tabs button[aria-pressed=true]{background:var(--cyan);color:#102128;border-color:var(--cyan)}.tools{display:flex;gap:10px;align-items:center}select{border-radius:8px;background:var(--bg)}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:26px}.card{border:1px solid var(--line);border-radius:15px;background:var(--panel);overflow:hidden;display:flex;flex-direction:column}.card[hidden]{display:none}.card-media{background:#0c1420;border-bottom:1px solid var(--line)}.zoom{display:block;position:relative;width:100%;padding:0;border:0;background:transparent}.zoom img{display:block;width:100%;height:auto;aspect-ratio:1.62;object-fit:contain}.enlarge{position:absolute;right:14px;bottom:12px;background:#10171fdd;border:1px solid #526778;border-radius:7px;padding:4px 9px;color:#d9e9ee;font-size:11px;opacity:0;transition:opacity .15s}.zoom:hover .enlarge,.zoom:focus-visible .enlarge{opacity:1}.card-copy{padding:24px;display:flex;flex:1;flex-direction:column}.eyebrow{font-size:11px;letter-spacing:.05em;text-transform:uppercase;color:var(--muted);display:flex;align-items:center;gap:7px}.dot{width:6px;height:6px;border-radius:50%;background:var(--cyan)}.dot.quasistatic{background:var(--purple)}.dot.physics{background:var(--coral)}.dim{margin-left:auto;border:1px solid var(--line);padding:1px 7px;border-radius:5px;font-size:10px}.card h3{font-size:22px;line-height:1.3;letter-spacing:-.02em;margin:14px 0 10px;font-weight:600}.card p{color:var(--muted);font-size:14px;line-height:1.7;margin:0 0 22px;flex:1}.controls{display:flex;align-items:center;justify-content:space-between;gap:12px;font-size:12px}.motion{font-size:12px;border:1px solid #547478;color:var(--cyan);background:#122c30;padding:8px 13px;border-radius:7px}.motion[aria-pressed=true]{color:#dcd3ff;background:#2f2746;border-color:#655781}.still-label{font-size:12px;color:#899da9}details{font-size:12px;margin-top:18px;border-top:1px solid var(--line);padding-top:12px}summary{cursor:pointer;color:var(--muted)}.data-links{display:flex;flex-wrap:wrap;gap:10px 18px;padding-top:12px}.footnote{margin:38px 0;color:var(--muted);font-size:13px;border-left:2px solid var(--line);padding-left:20px;max-width:1000px}footer{border-top:1px solid var(--line);margin-top:45px;padding:24px 0 34px;color:var(--muted);font-size:12px;display:flex;justify-content:space-between;gap:15px;flex-wrap:wrap}dialog{background:#111b26;color:var(--text);border:1px solid #53616c;border-radius:14px;padding:18px;width:min(1350px,96vw);max-height:95vh}dialog::backdrop{background:#050a10dc;backdrop-filter:blur(4px)}dialog .modal-head{display:flex;align-items:center;justify-content:space-between;margin-bottom:15px;gap:15px}dialog h2{font-size:18px;margin:0}dialog img{display:block;width:100%;max-height:75vh;object-fit:contain}dialog button{background:var(--panel);color:var(--text);border:1px solid var(--line);padding:6px 13px;border-radius:6px;font-size:13px}.sr-only{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0,0,0,0)}
@media(min-width:1600px){.grid{grid-template-columns:repeat(3,minmax(0,1fr))}}@media(max-width:900px){.hero{grid-template-columns:1fr;gap:28px;padding-top:36px}.hero-image{max-width:700px}.notes{gap:16px}.wrap{padding:0 24px}.card-copy{padding:18px}.stats{gap:26px}}@media(max-width:620px){.grid,.notes{grid-template-columns:1fr}.wrap{padding:0 16px}header nav{gap:14px}header .brand{font-size:11px}.hero{padding-top:30px}h1{font-size:43px}.notes{margin:20px 0 28px}.note{padding-top:12px}.toolbar{align-items:stretch}.tools{width:100%;justify-content:space-between}.legend{padding:17px}.legend-scale{width:100%}.gallery-heading{margin-top:34px}.stats{gap:24px}.stat strong{font-size:23px}.card h3{font-size:21px}}@media(prefers-reduced-motion:reduce){*{scroll-behavior:auto!important;transition:none!important}}
.gradient{background:linear-gradient(90deg,#436bd7,#28b5c4,#f0dc79,#f47852)}
.card[data-kind=field] .zoom img{aspect-ratio:1}
</style></head><body><div class="wrap">
<header><a class="brand" href="#"><b>Co-design</b> / Visual studies</a><nav><a href="#gallery">Gallery</a><a href="README.md">Read me</a><a href="gallery_manifest.json">Data index</a></nav></header>
<main><section class="hero"><div><div class="kicker">Material fields &amp; terrain contact</div><h1>Material /<br><em>Motion.</em></h1><p class="intro">Program material properties across a body. See how those choices change its response, then watch a spring lattice meet uneven ground.</p><div class="stats"><div class="stat"><strong>__COUNT__</strong><span>Visual demos</span></div><div class="stat"><strong>4</strong><span>Representations</span></div><div class="stat"><strong>2D + 3D</strong><span>Embodiments</span></div></div></div><figure class="hero-image"><img src="__HERO__" alt="__HERO_ALT__" width="1100" height="640"><figcaption>__HERO_CAPTION__</figcaption></figure></section>
<section class="notes" aria-label="What the demos show"><div class="note"><h2>01 / Design the field</h2><p>Direct grids, B-splines, CPPNs and primitives produce real spatial Poisson values. These views show sampled material fields, without dynamics.</p></div><div class="note"><h2>02 / Compare a material response</h2><p>The same FEM specimen is stretched with different Poisson fields. Free coordinates relax through the repository’s elastic energy.</p></div><div class="note"><h2>03 / Meet the terrain</h2><p>Small spring bodies land on flat ground, a slope and a rounded step. Animations come from accepted physical simulation steps.</p></div></section>
<div class="legend"><span class="legend-title">Poisson ratio · ν</span><div class="legend-scale"><div class="gradient"></div><div class="ticks"><span>0.25</span><span>0.35</span><span>0.45</span></div></div><p>One fixed color scale across material demos. Terrain clips have their own contact and body legends. Sampling cells in the morphology views are not solver elements.</p></div>
<section id="gallery"><div class="gallery-heading"><div><h2>Explore the studies</h2><p id="count" aria-live="polite">__COUNT__ demos · Select any image to enlarge</p></div></div><div class="toolbar"><div class="tabs" role="group" aria-label="Demo type"><button data-filter="all" aria-pressed="true">All studies</button><button data-filter="field" aria-pressed="false">Material fields</button><button data-filter="quasistatic" aria-pressed="false">FEM response</button><button data-filter="physics" aria-pressed="false">Terrain motion</button></div><div class="tools"><label class="sr-only" for="dimension">Dimension</label><select id="dimension"><option value="all">2D &amp; 3D</option value="2">2D</option><option value="3">3D</option></select><button class="secondary" id="play-all" aria-pressed="false">Play visible animations</button></div></div><div class="grid">__CARDS__</div></section>
<p class="footnote">The spring lattice has no independent continuum Poisson control; its demos here test nonflat contact. FEM clips are quasi-static loading sequences, while terrain clips use time integration. These are small visual demonstrations, not a certification of long-horizon locomotion or a performance comparison. Each card links to its setup and numerical evidence.</p>
<p class="footnote">Development note: an additional separated-support terrain example did not converge and is excluded from these completed demos. Its attempts are retained with the <a href="terrain/README.md">terrain evidence and limitations</a>.</p>
</main><footer><span>Generated __DATE__ · Opens offline · No server or external assets required</span><span><a href="source_scripts/">Generation scripts</a> · <a href="docs/SPATIAL_MATERIALS_AND_TERRAIN.md">Material API notes</a></span></footer></div>
<dialog id="lightbox"><div class="modal-head"><h2 id="modal-title"></h2><button id="modal-close" autofocus>Close ×</button></div><img id="modal-image" alt=""></dialog>
<script>
const cards=[...document.querySelectorAll('.card')];let current='all';
function play(card,on){const b=card.querySelector('.motion');if(!b)return;card.querySelector('.zoom img').src=on?b.dataset.motion:b.dataset.still;b.setAttribute('aria-pressed',String(on));b.textContent=on?'Stop animation':'Play animation';}
function syncPlay(){const active=cards.filter(c=>!c.hidden&&c.querySelector('.motion'));const on=active.length>0&&active.every(c=>c.querySelector('.motion').getAttribute('aria-pressed')==='true');const b=document.querySelector('#play-all');b.disabled=active.length===0;b.setAttribute('aria-pressed',String(on));b.textContent=on?'Stop visible animations':'Play visible animations';}
function filter(){const dim=document.querySelector('#dimension').value;let count=0;cards.forEach(c=>{c.hidden=!((current==='all'||c.dataset.kind===current)&&(dim==='all'||c.dataset.dim===dim));if(c.hidden)play(c,false);else count++;});document.querySelector('#count').textContent=count+' '+(count===1?'demo':'demos')+' · Select any image to enlarge';syncPlay();}
document.querySelectorAll('[data-filter]').forEach(b=>b.addEventListener('click',()=>{current=b.dataset.filter;document.querySelectorAll('[data-filter]').forEach(x=>x.setAttribute('aria-pressed',String(x===b)));filter();}));
document.querySelector('#dimension').addEventListener('change',filter);
cards.forEach(c=>{const b=c.querySelector('.motion');if(b)b.addEventListener('click',()=>{play(c,b.getAttribute('aria-pressed')!=='true');syncPlay();});c.querySelector('.zoom').addEventListener('click',()=>{const image=c.querySelector('img');document.querySelector('#modal-title').textContent=c.querySelector('h3').textContent;const enlarged=document.querySelector('#modal-image');enlarged.src=image.src;enlarged.alt=image.alt;document.querySelector('#lightbox').showModal();});});
document.querySelector('#play-all').addEventListener('click',()=>{const on=document.querySelector('#play-all').getAttribute('aria-pressed')!=='true';cards.filter(c=>!c.hidden).forEach(c=>play(c,on));syncPlay();});
const modal=document.querySelector('#lightbox');document.querySelector('#modal-close').addEventListener('click',()=>modal.close());modal.addEventListener('click',e=>{if(e.target===modal){const r=modal.getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)modal.close();}});modal.addEventListener('close',()=>document.querySelector('#modal-image').removeAttribute('src'));
filter();
</script></body></html>"""


def build(root: Path, destination: Path | None = None) -> None:
    """Write the offline gallery using existing evidence and explicit missing-data notices."""
    cards, reports = collect(root)
    preferred = next((c for c in cards if c["kind"] == "quasistatic"), cards[0])
    page = PAGE.replace("__COUNT__", str(len(cards))).replace(
        "__HERO__", quote(preferred["image"], safe="/")
    )
    page = page.replace("__HERO_ALT__", escape(preferred["title"], quote=True))
    page = page.replace(
        "__HERO_CAPTION__", escape(preferred["title"]) + " · Actual solved geometry"
    )
    page = page.replace("__CARDS__", "\n".join(map(card, cards)))
    page = page.replace("__DATE__", datetime.now(timezone.utc).strftime("%d %B %Y"))
    for filename in ("index.html", "START_HERE.html"):
        (root / filename).write_text(page, encoding="utf-8")
    manifest: Report = {
        "schema": "codesign_material_terrain_gallery_v1",
        "items": cards,
        "source_manifests": reports,
        "image_count": len(cards),
        "animation_count": sum(bool(c.get("animation")) for c in cards),
    }
    (root / "gallery_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    sources = root / "source_scripts"
    sources.mkdir(exist_ok=True)
    for filename in (
        "create_poisson_gallery.py",
        "create_poisson_mechanics_demo.py",
        "create_spring_terrain_gallery.py",
        "build_material_terrain_gallery.py",
    ):
        shutil.copy2(SCRIPTS_ROOT / filename, sources / filename)
    shutil.copy2(SOFT_ROOT / "shared.py", sources / "shared.py")
    docs = root / "docs"
    docs.mkdir(exist_ok=True)
    for filename in ("SPATIAL_MATERIALS_AND_TERRAIN.md", "NONFLAT_TERRAIN.md"):
        shutil.copy2(SOFT_ROOT / "docs" / filename, docs / filename)
    readme = """# Material / Motion gallery

Open START_HERE.html (or index.html) in a browser. Everything works offline.
Use the type and dimension filters, Play animation buttons, and image enlargement.
Each demo provides PNG/GIF previews and links to designs, raw states or metrics.
Keep the whole folder together when sharing it.

The materials folder shows actual decoded Poisson fields. Its field sweep edits
coefficients and is not a dynamics simulation. Mechanics contains quasi-static
FEM elastic-energy minimization. Terrain contains real spring time integration.
Read each manifest for solver settings, convergence, timing and playback scale.
The spring model does not implement independent continuum Poisson control.

Generator scripts and their shared.py helper are included in source_scripts
for reference. Run the originals at the paths below from a codesign-gym checkout
with its Python dependencies installed; the copied scripts expect that package
layout and are not standalone entry points. They import envs.soft_walker.engine.
This gallery is not a full source release. Install codesign-gym with its dev
extra; plotting dependencies are already included. In Linux/WSL, activate the
virtualenv, change to the codesign-gym repository root, and use a fresh output folder:

    export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
    export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
    python src/envs/soft_walker/scripts/create_poisson_gallery.py --out runs/my_visual_gallery/materials
    python src/envs/soft_walker/scripts/create_poisson_mechanics_demo.py --out runs/my_visual_gallery/mechanics
    python src/envs/soft_walker/scripts/create_spring_terrain_gallery.py --out runs/my_visual_gallery/terrain
    python src/envs/soft_walker/scripts/build_material_terrain_gallery.py --root runs/my_visual_gallery

Poisson bounds are 0.25 to 0.45. Colors use one shared blue/cyan/yellow/coral scale. The FEM and
terrain meshes have their own explicitly labelled physical interpretations.
"""
    (root / "README.md").write_text(readme, encoding="utf-8")
    if destination:
        destination = destination.resolve()
        if destination.is_relative_to(root.resolve()):
            raise ValueError("Share folder must be outside the generated gallery")
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite share folder: {destination}")
        archive = destination.with_suffix(".zip")
        if archive.exists():
            raise FileExistsError(archive)
        shutil.copytree(
            root, destination, ignore=shutil.ignore_patterns("__pycache__", "*.log", ".cache")
        )
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for path in sorted(destination.rglob("*")):
                if path.is_file():
                    z.write(path, path.relative_to(destination.parent))
        with zipfile.ZipFile(archive) as z:
            if z.testzip() is not None:
                raise ValueError("ZIP CRC check failed")
        manifest["share_folder"] = str(destination)
        manifest["archive"] = str(archive)
    print(
        json.dumps(
            {k: v for k, v in manifest.items() if k not in ("items", "source_manifests")}, indent=2
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "runs/spatial_material_terrain_gallery")
    parser.add_argument("--share-folder", type=Path)
    arguments = parser.parse_args()
    build(arguments.root.resolve(), arguments.share_folder)
