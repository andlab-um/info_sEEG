#!/usr/bin/env python3
"""Run the one-participant replication-code demo from the repository root."""

from __future__ import annotations

import argparse
import datetime as dt
import platform
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from src.behavior_demo import run_behavior_demo
from src.config import load_demo_config, require_section
from src.inventory import build_inventory
from src.io_utils import resolve_repo_path, repo_root_from_config, write_json
from src.neural_demo import run_neural_demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the information sEEG one-participant demo.")
    parser.add_argument(
        "--config",
        default="replication_code/config/demo_config.yaml",
        help="Relative path to the demo config file.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path
    config = load_demo_config(config_path)
    repo_root = repo_root_from_config(config_path)
    paths = require_section(config, "paths")
    output_dir = resolve_repo_path(repo_root, paths["outputs"])
    output_dir.mkdir(parents=True, exist_ok=True)

    generated: list[Path] = []
    generated.append(build_inventory(repo_root, paths, output_dir))
    generated.extend(
        run_behavior_demo(
            resolve_repo_path(repo_root, paths["behavioral_metrics"]),
            resolve_repo_path(repo_root, paths["subject_parameters"]),
            output_dir,
        )
    )
    generated.extend(
        run_neural_demo(
            resolve_repo_path(repo_root, paths["reveal_features"]),
            resolve_repo_path(repo_root, paths["normative_state_metrics"]),
            output_dir,
        )
    )

    manifest = {
        "demo_name": config.get("project", {}).get("name", "info_seeg_replication_demo"),
        "timestamp_local": dt.datetime.now().isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "output_dir": str(output_dir.relative_to(repo_root)),
        "analysis_parameters": config.get("analysis_parameters", {}),
        "generated_files": [str(path.relative_to(repo_root)) for path in generated],
        "scope_note": (
            "This one-participant demo validates inputs and emits expected output structures. "
            "Full group-level inference requires the complete private cohort dataset."
        ),
    }
    manifest_path = output_dir / "manifest.json"
    write_json(manifest_path, manifest)
    generated.append(manifest_path)

    print("Demo completed successfully.")
    print(f"Outputs written under: {output_dir.relative_to(repo_root)}")
    for path in generated:
        print(f"- {path.relative_to(repo_root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
