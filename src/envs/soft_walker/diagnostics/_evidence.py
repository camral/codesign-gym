"""Host-only evidence mechanics; callers own inventories, schemas, and exclusions."""

from __future__ import annotations

import hashlib
import zipfile
from collections.abc import Iterable, Mapping
from pathlib import Path


def hash_files(root: Path, paths: Iterable[Path], *, posix_names: bool = False) -> dict[str, str]:
    """Hash exactly the selected files in their supplied order and path convention."""
    return {
        (
            path.relative_to(root).as_posix() if posix_names else str(path.relative_to(root))
        ): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }


def write_checksums(path: Path, hashes: Mapping[str, str]) -> None:
    """Write the existing two-space-delimited checksum format in inventory order."""
    path.write_text("".join(f"{digest}  {name}\n" for name, digest in hashes.items()))


def write_archive(folder: Path, archive: Path, *, posix_names: bool = False) -> None:
    """Archive all files under the run directory using its name as the ZIP root."""
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zipped:
        for path in sorted(folder.rglob("*")):
            if path.is_file():
                relative = path.relative_to(folder)
                name = relative.as_posix() if posix_names else str(relative)
                zipped.write(path, folder.name + "/" + name)
