#!/usr/bin/env python3
"""Replicate reveal-wise decoding of best-deck reconfiguration from ACC PCs."""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from src.config import load_demo_config, require_section
from src.io_utils import resolve_repo_path, repo_root_from_config


def _parse_int(value: object, default: int) -> int:
    try:
        return int(str(value).strip())
    except Exception:
        return default


def _parse_subjects(value: str) -> list[str] | None:
    text = str(value).strip()
    if not text or text.lower() in {"all", "*"}:
        return None
    return [part.strip() for part in text.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run reveal-wise ACC decoding of best-deck reconfiguration."
    )
    parser.add_argument(
        "--config",
        default="replication_code/config/full_analysis_config.yaml",
        help="Relative path to the full-analysis config file.",
    )
    return parser.parse_args()


def main() -> int:
    warnings.filterwarnings("ignore")
    try:
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, cross_val_score
    except ImportError as exc:
        raise SystemExit(
            "Missing full-analysis dependency. Install the full environment in INSTALL.md. "
            f"Original import error: {exc}"
        ) from exc

    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path

    config = load_demo_config(config_path)
    repo_root = repo_root_from_config(config_path)
    paths = require_section(config, "paths")
    params = require_section(config, "analysis_parameters")

    input_csv = resolve_repo_path(repo_root, paths["normative_state_metrics"])
    output_dir = resolve_repo_path(repo_root, paths["outputs"]) / "bestdeck_reconfiguration"
    output_dir.mkdir(parents=True, exist_ok=True)

    random_state = _parse_int(params.get("bestdeck_random_state", 42), 42)
    by_reveal_permutations = _parse_int(params.get("bestdeck_by_reveal_permutations", 1000), 1000)
    subject_filter = _parse_subjects(params.get("subjects", "all"))

    print(f"Input table: {input_csv.relative_to(repo_root)}")
    print(f"Output directory: {output_dir.relative_to(repo_root)}")

    df = pd.read_csv(input_csv)
    print("Loaded:", df.shape)

    required_cols = [
        "Subject",
        "Game",
        "reveal_idx",
        "condition_label",
        "acc_PC1",
        "acc_PC2",
        "acc_PC3",
        "bestdeck_changed_after_reveal",
    ]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if subject_filter is not None:
        df = df[df["Subject"].astype(str).isin(subject_filter)].copy()
        if len(df) == 0:
            raise ValueError(f"No rows remain after subject filter: {', '.join(subject_filter)}")
        print(f"After subject filter ({', '.join(subject_filter)}):", df.shape)

    df = df.sort_values(["Subject", "Game", "reveal_idx"]).reset_index(drop=True)
    df["bestdeck_changed_after_reveal"] = pd.to_numeric(
        df["bestdeck_changed_after_reveal"], errors="coerce"
    )
    df = df.dropna(
        subset=[
            "Subject",
            "Game",
            "reveal_idx",
            "acc_PC1",
            "acc_PC2",
            "acc_PC3",
            "bestdeck_changed_after_reveal",
        ]
    ).copy()

    if "game_complete_6" in df.columns:
        df = df[df["game_complete_6"] == True].copy()

    def cv_auc_score(x, y, n_splits=5):
        y = np.asarray(y).astype(int)
        if len(np.unique(y)) < 2:
            return np.nan
        binc = np.bincount(y)
        if len(binc) < 2:
            return np.nan
        min_class = np.min(binc)
        ncv = min(n_splits, min_class)
        if ncv < 2:
            return np.nan
        clf = LogisticRegression(max_iter=2000)
        cv = StratifiedKFold(n_splits=ncv, shuffle=True, random_state=random_state)
        scores = cross_val_score(clf, x, y, cv=cv, scoring="roc_auc")
        return np.mean(scores)

    def permutation_auc(x, y, n_perm=1000, seed=42):
        obs = cv_auc_score(x, y)
        if np.isnan(obs):
            return np.nan, np.nan
        rng = np.random.default_rng(seed)
        perm_scores = []
        for _ in range(n_perm):
            yp = rng.permutation(y)
            score = cv_auc_score(x, yp)
            if not np.isnan(score):
                perm_scores.append(score)
        perm_scores = np.asarray(perm_scores, dtype=float)
        if len(perm_scores) == 0:
            return obs, np.nan
        p_value = (np.sum(perm_scores >= obs) + 1) / (len(perm_scores) + 1)
        return obs, p_value

    decode_rows = []
    for reveal in sorted(df["reveal_idx"].unique()):
        sub = df[df["reveal_idx"] == reveal].copy()
        sub = sub.dropna(
            subset=["acc_PC1", "acc_PC2", "acc_PC3", "bestdeck_changed_after_reveal"]
        )
        y = sub["bestdeck_changed_after_reveal"].astype(int).to_numpy()
        x = sub[["acc_PC1", "acc_PC2", "acc_PC3"]].to_numpy(dtype=float)
        auc, p_value = permutation_auc(
            x,
            y,
            n_perm=by_reveal_permutations,
            seed=random_state,
        )
        decode_rows.append(
            {
                "scope": "by_reveal",
                "condition_label": "all",
                "reveal_idx": int(reveal),
                "auc": auc,
                "perm_p": p_value,
                "n_trials": len(sub),
                "n_switch0": int(np.sum(y == 0)),
                "n_switch1": int(np.sum(y == 1)),
            }
        )

    decode_df = pd.DataFrame(decode_rows)
    decode_df.to_csv(output_dir / "decode_auc_by_reveal.csv", index=False)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "font.size": 7,
            "axes.labelsize": 7.5,
            "axes.titlesize": 8.8,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.8,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.6,
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    fig_w = 183 / 25.4
    if len(decode_df) > 0:
        fig, ax = plt.subplots(figsize=(fig_w * 0.65, fig_w * 0.48))
        ax.plot(
            decode_df["reveal_idx"],
            decode_df["auc"],
            marker="o",
            color="#2980B9",
            linewidth=2.2,
            markersize=5.5,
        )
        ax.axhline(0.5, linestyle="--", linewidth=1.2, color="gray", alpha=0.7)
        for _, row in decode_df.iterrows():
            if not np.isnan(row["perm_p"]):
                ax.text(
                    row["reveal_idx"] + 0.08,
                    row["auc"] + 0.012,
                    f"p={row['perm_p']:.3f}",
                    fontsize=6.5,
                    color="#2C3E50",
                )
        ax.set_xlabel("Reveal index")
        ax.set_ylabel("Decode AUC (best-deck change)")
        ax.set_title(
            "ACC decodes belief reconfiguration\nafter reveal",
            pad=12,
            fontweight="bold",
        )
        ax.set_xticks(range(1, 7))
        ax.set_ylim(0.4, 1.02)
        ax.grid(True, alpha=0.25, linestyle="--")
        ax.legend(
            ["Observed AUC", "Chance level"],
            frameon=False,
            fontsize=6.8,
            loc="lower right",
        )
        fig.tight_layout()
        for ext in ["pdf", "svg", "png"]:
            fig.savefig(output_dir / f"fig1_decode_auc_by_reveal.{ext}", transparent=True)
        plt.close(fig)

    print("Reveal-wise best-deck reconfiguration decoding complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
