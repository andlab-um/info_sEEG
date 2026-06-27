#!/usr/bin/env python3
"""Run reveal-3 ACC leave-one-subject-out centroid decoding analyses."""

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


def _parse_int(value: str, default: int) -> int:
    try:
        return int(str(value).strip())
    except Exception:
        return default


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run reveal-3 ACC LOSO centroid decoding analyses.")
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
        from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
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
    output_dir = resolve_repo_path(repo_root, paths["outputs"]) / "reveal3_centroid_loso"
    output_dir.mkdir(parents=True, exist_ok=True)

    n_perm = _parse_int(params.get("reveal3_permutations", "5000"), 5000)
    random_state = _parse_int(params.get("random_state", "42"), 42)
    state_cols = ["acc_PC1", "acc_PC2", "acc_PC3"]
    target_col = "bestdeck_changed_after_reveal"

    df = pd.read_csv(input_csv)
    required_cols = ["Subject", "Game", "reveal_idx", target_col] + state_cols
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    for col in ["reveal_idx", target_col] + state_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=required_cols).copy()
    if "game_complete_6" in df.columns:
        df = df[df["game_complete_6"] == True].copy()
    df = df[df[target_col].isin([0, 1])].copy()
    df[target_col] = df[target_col].astype(int)
    df = df[df["reveal_idx"] == 3].copy()
    if "has_acc_data" in df.columns:
        df = df[df["has_acc_data"] == 1].copy()
    df = df.sort_values(["Subject", "Game"]).reset_index(drop=True)

    print(f"Input table: {input_csv.relative_to(repo_root)}")
    print(f"Output directory: {output_dir.relative_to(repo_root)}")
    print(f"Reveal-3 rows: {len(df)}, subjects: {df['Subject'].nunique()}")

    def euclid(a, b):
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        return float(np.sqrt(np.sum((a - b) ** 2)))

    def safe_auc(y_true, score):
        y_true = np.asarray(y_true).astype(int)
        score = np.asarray(score, dtype=float)
        if len(np.unique(y_true)) < 2:
            return np.nan
        try:
            return float(roc_auc_score(y_true, score))
        except Exception:
            return np.nan

    def loso_distance_to_centroid(dataframe):
        fold_rows = []
        trial_rows = []
        pooled_scores = []
        pooled_y = []
        pooled_pred = []
        for held_out in sorted(dataframe["Subject"].unique()):
            train = dataframe[dataframe["Subject"] != held_out].copy()
            test = dataframe[dataframe["Subject"] == held_out].copy()
            y_train = train[target_col].to_numpy(dtype=int)
            y_test = test[target_col].to_numpy(dtype=int)
            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                fold_rows.append({
                    "test_subject": held_out,
                    "auc": np.nan,
                    "accuracy": np.nan,
                    "balanced_accuracy": np.nan,
                    "n_test": len(test),
                    "n_test_switch0": int(np.sum(y_test == 0)),
                    "n_test_switch1": int(np.sum(y_test == 1)),
                    "train_centroid0_PC1": np.nan,
                    "train_centroid0_PC2": np.nan,
                    "train_centroid0_PC3": np.nan,
                    "train_centroid1_PC1": np.nan,
                    "train_centroid1_PC2": np.nan,
                    "train_centroid1_PC3": np.nan,
                    "train_centroid_distance": np.nan,
                })
                continue
            c0 = train[train[target_col] == 0][state_cols].to_numpy(dtype=float).mean(axis=0)
            c1 = train[train[target_col] == 1][state_cols].to_numpy(dtype=float).mean(axis=0)
            x_test = test[state_cols].to_numpy(dtype=float)
            d0 = np.sqrt(np.sum((x_test - c0) ** 2, axis=1))
            d1 = np.sqrt(np.sum((x_test - c1) ** 2, axis=1))
            score = d0 - d1
            pred = (score > 0).astype(int)
            auc = safe_auc(y_test, score)
            acc = accuracy_score(y_test, pred)
            bacc = balanced_accuracy_score(y_test, pred)
            fold_rows.append({
                "test_subject": held_out,
                "auc": auc,
                "accuracy": acc,
                "balanced_accuracy": bacc,
                "n_test": len(test),
                "n_test_switch0": int(np.sum(y_test == 0)),
                "n_test_switch1": int(np.sum(y_test == 1)),
                "train_centroid0_PC1": c0[0],
                "train_centroid0_PC2": c0[1],
                "train_centroid0_PC3": c0[2],
                "train_centroid1_PC1": c1[0],
                "train_centroid1_PC2": c1[1],
                "train_centroid1_PC3": c1[2],
                "train_centroid_distance": euclid(c0, c1),
            })
            out = test[["Subject", "Game", target_col]].copy()
            out["decision_score"] = score
            out["d_no_reconfiguration"] = d0
            out["d_reconfiguration"] = d1
            trial_rows.append(out)
            pooled_scores.extend(score.tolist())
            pooled_y.extend(y_test.tolist())
            pooled_pred.extend(pred.tolist())
        fold_df = pd.DataFrame(fold_rows)
        score_df = pd.concat(trial_rows, ignore_index=True) if trial_rows else pd.DataFrame()
        overall_auc = safe_auc(pooled_y, pooled_scores) if pooled_y else np.nan
        overall_acc = accuracy_score(pooled_y, pooled_pred) if pooled_y else np.nan
        overall_bacc = balanced_accuracy_score(pooled_y, pooled_pred) if pooled_y else np.nan
        mean_auc = fold_df["auc"].mean() if len(fold_df) else np.nan
        mean_bacc = fold_df["balanced_accuracy"].mean() if len(fold_df) else np.nan
        return {
            "obs_auc": overall_auc,
            "obs_acc": overall_acc,
            "obs_bacc": overall_bacc,
            "obs_mean_subject_auc": mean_auc,
            "obs_mean_subject_bacc": mean_bacc,
            "fold_df": fold_df,
            "score_df": score_df,
        }

    def loso_distance_to_centroid_permutation(dataframe, seed):
        rng_local = np.random.default_rng(seed)
        observed = loso_distance_to_centroid(dataframe)
        perm_auc = []
        perm_bacc = []
        perm_mean_auc = []
        perm_mean_bacc = []
        for _ in range(n_perm):
            perm_df = dataframe.copy()
            for _, idx in perm_df.groupby("Subject").groups.items():
                idx = list(idx)
                perm_df.loc[idx, target_col] = rng_local.permutation(perm_df.loc[idx, target_col].to_numpy())
            perm = loso_distance_to_centroid(perm_df)
            if np.isfinite(perm["obs_auc"]):
                perm_auc.append(perm["obs_auc"])
            if np.isfinite(perm["obs_bacc"]):
                perm_bacc.append(perm["obs_bacc"])
            if np.isfinite(perm["obs_mean_subject_auc"]):
                perm_mean_auc.append(perm["obs_mean_subject_auc"])
            if np.isfinite(perm["obs_mean_subject_bacc"]):
                perm_mean_bacc.append(perm["obs_mean_subject_bacc"])

        def p_value(obs, perm_values):
            perm_values = np.asarray(perm_values, dtype=float)
            if len(perm_values) == 0 or not np.isfinite(obs):
                return np.nan
            return float((np.sum(perm_values >= obs) + 1) / (len(perm_values) + 1))

        observed["perm_p_auc"] = p_value(observed["obs_auc"], perm_auc)
        observed["perm_p_bacc"] = p_value(observed["obs_bacc"], perm_bacc)
        observed["perm_p_mean_subject_auc"] = p_value(observed["obs_mean_subject_auc"], perm_mean_auc)
        observed["perm_p_mean_subject_bacc"] = p_value(observed["obs_mean_subject_bacc"], perm_mean_bacc)
        return observed

    loso_res = loso_distance_to_centroid_permutation(df, random_state)
    fold_df = loso_res["fold_df"].copy()
    fold_df["reveal_idx"] = 3
    fold_df.to_csv(output_dir / "reveal3_loso_distance_to_centroid_fold_details.csv", index=False)
    score_df = loso_res["score_df"].copy()
    score_df.to_csv(output_dir / "reveal3_loso_trialwise_decision_scores.csv", index=False)

    loso_summary = pd.DataFrame([{
        "reveal_idx": 3,
        "loso_pooled_auc": loso_res["obs_auc"],
        "loso_pooled_accuracy": loso_res["obs_acc"],
        "loso_pooled_balanced_accuracy": loso_res["obs_bacc"],
        "loso_mean_subject_auc": loso_res["obs_mean_subject_auc"],
        "loso_mean_subject_balanced_accuracy": loso_res["obs_mean_subject_bacc"],
        "perm_p_auc": loso_res["perm_p_auc"],
        "perm_p_bacc": loso_res["perm_p_bacc"],
        "perm_p_mean_subject_auc": loso_res["perm_p_mean_subject_auc"],
        "perm_p_mean_subject_bacc": loso_res["perm_p_mean_subject_bacc"],
        "n_subjects": df["Subject"].nunique(),
        "n_trials": len(df),
        "n_switch0": int(np.sum(df[target_col] == 0)),
        "n_switch1": int(np.sum(df[target_col] == 1)),
        "n_perm": n_perm,
    }])
    loso_summary.to_csv(output_dir / "reveal3_loso_distance_to_centroid_summary.csv", index=False)

    if len(fold_df) > 0:
        fig, ax = plt.subplots(figsize=(4.8, 3.6))
        ax.hist(fold_df["auc"].dropna(), bins=12, color="#2980B9", alpha=0.75)
        ax.axvline(loso_res["obs_mean_subject_auc"], color="black", linestyle="--", label=f"Mean AUC={loso_res['obs_mean_subject_auc']:.3f}")
        ax.set_xlabel("Subject-level LOSO AUC")
        ax.set_ylabel("Count")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(output_dir / "fig4_reveal3_loso_distance_to_centroid_auc.png", dpi=250)
        plt.close(fig)

    if len(score_df) > 0:
        fig, ax = plt.subplots(figsize=(4.8, 3.6))
        score0 = score_df[score_df[target_col] == 0]["decision_score"].dropna().to_numpy()
        score1 = score_df[score_df[target_col] == 1]["decision_score"].dropna().to_numpy()
        ax.hist(score0, bins=30, alpha=0.5, density=True, label="no-switch")
        ax.hist(score1, bins=30, alpha=0.5, density=True, label="switch")
        ax.axvline(0, linestyle="--", linewidth=1)
        ax.set_xlabel("Decision score")
        ax.set_ylabel("Density")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(output_dir / "fig5_reveal3_loso_decision_score_hist.png", dpi=250)
        plt.close(fig)

    summary_lines = [
        "=== Reveal 3 ACC LOSO distance-to-centroid decoding ===",
        f"N trials: {len(df)}",
        f"N subjects: {df['Subject'].nunique()}",
        f"N no-switch: {int(np.sum(df[target_col] == 0))}",
        f"N switch: {int(np.sum(df[target_col] == 1))}",
        "",
        f"Pooled AUC: {loso_res['obs_auc']:.6f}",
        f"Pooled accuracy: {loso_res['obs_acc']:.6f}",
        f"Pooled balanced accuracy: {loso_res['obs_bacc']:.6f}",
        f"Mean subject AUC: {loso_res['obs_mean_subject_auc']:.6f}",
        f"Mean subject balanced accuracy: {loso_res['obs_mean_subject_bacc']:.6f}",
        f"Permutation p (AUC): {loso_res['perm_p_auc']:.6f}",
        f"Permutation p (balanced accuracy): {loso_res['perm_p_bacc']:.6f}",
        f"Permutation p (mean subject AUC): {loso_res['perm_p_mean_subject_auc']:.6f}",
        f"Permutation p (mean subject balanced accuracy): {loso_res['perm_p_mean_subject_bacc']:.6f}",
        "",
    ]
    (output_dir / "reveal3_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")

    print("Reveal-3 LOSO centroid decoding analysis complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
