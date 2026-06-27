#!/usr/bin/env python3
"""Run all reviewer-facing command-line analysis scripts in sequence."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full reviewer-facing manuscript pipeline.")
    parser.add_argument(
        "--config",
        default="replication_code/config/full_analysis_config.yaml",
        help="Relative path to the full-analysis config file.",
    )
    parser.add_argument(
        "--skip-heavy-model",
        action="store_true",
        help="Skip PyMC behavioral model refitting and use existing model-derived tables for figures.",
    )
    parser.add_argument(
        "--skip-seeg",
        action="store_true",
        help="Skip TFR, PLV, and PSI analyses that require MNE/EEGLAB epoch dependencies.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    scripts = []
    if not args.skip_heavy_model:
        scripts.append("run_behavior_model.py")
    scripts.append("run_behavior_figures.py")
    if not args.skip_seeg:
        scripts.extend(["run_tfr_glm.py", "run_plv.py", "run_psi.py"])
    scripts.extend([
        "run_pca_state_space.py",
        "run_bestdeck_reconfiguration.py",
        "run_reveal3_centroid_loso.py",
        "run_acc_rsa.py",
    ])

    for script_name in scripts:
        script_path = script_dir / script_name
        print(f"\n=== Running {script_name} ===")
        command = [sys.executable, str(script_path), "--config", args.config]
        result = subprocess.run(command, check=False)
        if result.returncode != 0:
            print(f"FAILED: {script_name}", file=sys.stderr)
            return result.returncode
    print("\nFull reviewer-facing manuscript pipeline complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
