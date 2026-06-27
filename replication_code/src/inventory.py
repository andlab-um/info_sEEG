"""Repository inventory helpers for the demo package."""

from __future__ import annotations

from pathlib import Path

from .io_utils import write_csv


def build_inventory(repo_root: Path, configured_paths: dict[str, str], output_dir: Path) -> Path:
    rows = []
    for key, rel_path in configured_paths.items():
        if key == "outputs":
            continue
        path = repo_root / rel_path
        if path.is_dir():
            files = sorted(p for p in path.rglob("*") if p.is_file())
            rows.append(
                {
                    "config_key": key,
                    "relative_path": rel_path,
                    "exists": str(path.exists()),
                    "kind": "directory",
                    "file_count": str(len(files)),
                    "bytes": str(sum(p.stat().st_size for p in files)),
                }
            )
        else:
            rows.append(
                {
                    "config_key": key,
                    "relative_path": rel_path,
                    "exists": str(path.exists()),
                    "kind": "file",
                    "file_count": "1" if path.exists() else "0",
                    "bytes": str(path.stat().st_size if path.exists() else 0),
                }
            )

    out = output_dir / "input_inventory.csv"
    write_csv(out, rows, ["config_key", "relative_path", "exists", "kind", "file_count", "bytes"])
    return out

