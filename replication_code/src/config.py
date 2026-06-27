"""Minimal YAML-like config reader for the demo configuration file."""

from __future__ import annotations

from pathlib import Path


def load_demo_config(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    config: dict[str, dict[str, str]] = {}
    section: str | None = None
    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if not raw_line.startswith(" ") and stripped.endswith(":"):
            section = stripped[:-1]
            config[section] = {}
            continue
        if section is None or ":" not in stripped:
            raise ValueError(f"Unsupported config syntax at line {line_no}: {raw_line}")
        key, value = stripped.split(":", 1)
        value = value.strip()
        if value.startswith('"') and value.endswith('"'):
            value = value[1:-1]
        config[section][key.strip()] = value
    return config


def require_section(config: dict[str, dict[str, str]], section: str) -> dict[str, str]:
    if section not in config:
        raise KeyError(f"Missing config section: {section}")
    return config[section]

