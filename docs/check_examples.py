"""Run the Python examples in the docs.

Each page's ```python blocks run in order, in one fresh subprocess per page (so later blocks can use earlier names).
A block containing ``# notest`` is skipped. Pass pages to check; with none, every page under docs/ is checked.

    python docs/check_examples.py docs/getting-started.md docs/envs/lq_structure/*.md
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

DOCS = Path(__file__).resolve().parent
BLOCK = re.compile(r"^```python\n(.*?)^```", re.S | re.M)


def check(page: Path) -> bool:
    blocks = [b for b in BLOCK.findall(page.read_text(encoding="utf-8")) if "# notest" not in b]
    if not blocks:
        return True
    code = "\n\n".join(blocks)
    if "__main__" not in code:
        code = "if __name__ == '__main__':\n" + "\n".join("    " + line for line in code.splitlines()) + "\n    pass\n"
    result = subprocess.run([sys.executable, "-c", code], cwd=DOCS.parent, capture_output=True, text=True)
    status = "ok" if result.returncode == 0 else "FAIL"
    print(f"{status:4s} {page.relative_to(DOCS)} ({len(blocks)} blocks)")
    if result.returncode:
        print(result.stderr[-3000:])
    return result.returncode == 0


if __name__ == "__main__":
    pages = [Path(p).resolve() for p in sys.argv[1:]] or sorted(p for p in DOCS.rglob("*.md") if "_build" not in p.parts and "_generated" not in p.parts and "package" not in p.parts)
    sys.exit(0 if all([check(p) for p in pages]) else 1)
