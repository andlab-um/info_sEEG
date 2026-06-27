#!/usr/bin/env python3
"""Run the full PCA-family reviewer scripts in sequence."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full PCA/state-space reviewer workflow.")
    parser.add_argument(
        "--config",
        default="replication_code/config/full_analysis_config.yaml",
        help="Relative path to the full-analysis config file.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scripts = [
        "run_pca_state_space.py",
        "run_bestdeck_reconfiguration.py",
        "run_reveal3_centroid_loso.py",
        "run_acc_rsa.py",
    ]
    script_dir = Path(__file__).resolve().parent
    for script_name in scripts:
        script_path = script_dir / script_name
        print(f"\n=== Running {script_name} ===")
        result = subprocess.run([sys.executable, str(script_path), "--config", args.config], check=False)
        if result.returncode != 0:
            print(f"FAILED: {script_name}", file=sys.stderr)
            return result.returncode
    print("\nFull PCA-family workflow complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
