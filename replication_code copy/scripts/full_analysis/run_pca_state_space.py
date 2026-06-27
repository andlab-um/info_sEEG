#!/usr/bin/env python3
"""Run the full PCA/state-space workflow from the original PCA notebook.

This script ports the reviewer-relevant PCA logic from
`code/scipts/6_pca_decoding.ipynb` into an English-only command-line script
with relative, config-controlled paths.
"""

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


def _parse_int(config_value: str, default: int) -> int:
    try:
        return int(str(config_value).strip())
    except Exception:
        return default


def _parse_subjects(value: str) -> list[str] | None:
    text = str(value).strip()
    if not text or text.lower() in {"all", "*"}:
        return None
    return [part.strip() for part in text.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ROI PCA and state-space analyses.")
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
        from sklearn.decomposition import PCA
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LinearRegression, LogisticRegression
        from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
        from sklearn.preprocessing import StandardScaler
        import statsmodels.formula.api as smf
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

    input_csv = resolve_repo_path(repo_root, paths["reveal_features"])
    output_dir = resolve_repo_path(repo_root, paths["outputs"]) / "pca_state_space"
    output_dir.mkdir(parents=True, exist_ok=True)

    n_components = _parse_int(params.get("n_pca_components", "3"), 3)
    random_state = _parse_int(params.get("random_state", "42"), 42)
    center_mode = params.get("center_mode", "subject_centered")
    subject_filter = _parse_subjects(params.get("subjects", "all"))

    print(f"Input table: {input_csv.relative_to(repo_root)}")
    print(f"Output directory: {output_dir.relative_to(repo_root)}")

    df = pd.read_csv(input_csv)
    print("Loaded:", df.shape)

    if subject_filter is not None:
        if "Subject" not in df.columns:
            raise ValueError("Missing required column for subject filtering: Subject")
        df = df[df["Subject"].astype(str).isin(subject_filter)].copy()
        if len(df) == 0:
            raise ValueError(f"No rows remain after subject filter: {', '.join(subject_filter)}")
        print(f"After subject filter ({', '.join(subject_filter)}):", df.shape)

    if "game_complete_6" in df.columns:
        df = df[df["game_complete_6"] == True].copy()
    df = df.sort_values(["Subject", "Game", "reveal_idx"]).reset_index(drop=True)
    print("After complete-game filter:", df.shape)

    def pick_roi_features(columns, roi_name):
        feats = []
        for col in columns:
            if not col.startswith(f"{roi_name}_"):
                continue
            if (
                "_bin" in col
                or col.endswith("_full_mean")
                or col.endswith("_full_auc")
                or col.endswith("_channel_sd_full")
            ):
                feats.append(col)
        return feats

    acc_cols = [col for col in pick_roi_features(df.columns, "acc") if df[col].notna().sum() > 0]
    vmpfc_cols = [col for col in pick_roi_features(df.columns, "vmpfc") if df[col].notna().sum() > 0]
    print("ACC feature count:", len(acc_cols))
    print("vmPFC feature count:", len(vmpfc_cols))

    required_cols = ["Subject", "Game", "condition_label", "reveal_idx"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    candidate_targets = [
        "current_reward",
        "best_deck_snapshot",
        "value_gap_snapshot",
        "sampling_imbalance_after",
        "sampling_imbalance_before",
        "imbalance_change",
        "current_deck_was_least_sampled_before",
        "current_deck_is_best_snapshot",
    ]
    available_targets = [col for col in candidate_targets if col in df.columns]
    print("Available representation targets:", available_targets)

    def get_valid_subjects_for_roi(df_in, feature_cols):
        if len(feature_cols) == 0:
            return []
        subj_valid = df_in.groupby("Subject")[feature_cols].apply(lambda x: x.notna().any().any())
        return subj_valid[subj_valid].index.tolist()

    def get_valid_row_mask(df_in, feature_cols):
        if len(feature_cols) == 0:
            return pd.Series(False, index=df_in.index)
        return df_in[feature_cols].notna().any(axis=1)

    acc_valid_subjects = get_valid_subjects_for_roi(df, acc_cols)
    vmpfc_valid_subjects = get_valid_subjects_for_roi(df, vmpfc_cols)
    print("ACC valid subjects:", len(acc_valid_subjects))
    print("vmPFC valid subjects:", len(vmpfc_valid_subjects))

    df["has_acc_data"] = df["Subject"].isin(acc_valid_subjects).astype(int)
    df["has_vmpfc_data"] = df["Subject"].isin(vmpfc_valid_subjects).astype(int)

    def preprocess_matrix(x):
        imputer = SimpleImputer(strategy="median")
        x_imp = imputer.fit_transform(x)
        scaler = StandardScaler()
        x_std = scaler.fit_transform(x_imp)
        return x_std, imputer, scaler

    def center_features(df_in, feature_cols, mode="subject_centered"):
        x = df_in[feature_cols].copy()
        if mode == "raw":
            return x
        if mode == "subject_centered":
            return x - df_in.groupby("Subject")[feature_cols].transform("mean")
        if mode == "subject_condition_centered":
            return x - df_in.groupby(["Subject", "condition_label"])[feature_cols].transform("mean")
        raise ValueError(f"Unknown center mode: {mode}")

    def run_roi_pca(df_in, roi_name, feature_cols):
        if len(feature_cols) == 0:
            raise ValueError(f"No feature columns found for ROI: {roi_name}")
        valid_row_mask = get_valid_row_mask(df_in, feature_cols)
        df_roi = df_in.loc[valid_row_mask].copy()
        if len(df_roi) == 0:
            raise ValueError(f"No valid rows for ROI: {roi_name}")

        x_centered = center_features(df_roi, feature_cols, mode=center_mode)
        x_std, imputer, scaler = preprocess_matrix(x_centered.to_numpy(dtype=float))

        pca = PCA(n_components=n_components, random_state=random_state)
        x_pca = pca.fit_transform(x_std)

        key_cols = ["Subject", "Game", "condition_label", "reveal_idx"]
        if "Condition" in df_roi.columns:
            key_cols.append("Condition")

        score_df = df_roi[key_cols].copy().reset_index(drop=True)
        for idx in range(n_components):
            score_df[f"{roi_name}_PC{idx + 1}"] = x_pca[:, idx]

        loadings = pd.DataFrame(
            pca.components_.T,
            index=feature_cols,
            columns=[f"{roi_name}_PC{idx + 1}" for idx in range(n_components)],
        )
        explained = pd.Series(
            pca.explained_variance_ratio_,
            index=[f"{roi_name}_PC{idx + 1}" for idx in range(n_components)],
            name="explained_variance_ratio",
        )
        return score_df, loadings, explained, df_roi

    acc_scores, acc_loadings, acc_explained, _ = run_roi_pca(
        df[df["has_acc_data"] == 1].copy(), "acc", acc_cols
    )
    vmpfc_scores, vmpfc_loadings, vmpfc_explained, _ = run_roi_pca(
        df[df["has_vmpfc_data"] == 1].copy(), "vmpfc", vmpfc_cols
    )

    acc_loadings.to_csv(output_dir / "acc_pca_loadings.csv")
    vmpfc_loadings.to_csv(output_dir / "vmpfc_pca_loadings.csv")
    acc_explained.to_csv(output_dir / "acc_explained_variance.csv", header=True)
    vmpfc_explained.to_csv(output_dir / "vmpfc_explained_variance.csv", header=True)

    df_state = df.copy()
    merge_keys = ["Subject", "Game", "condition_label", "reveal_idx"]
    if "Condition" in df_state.columns and "Condition" in acc_scores.columns and "Condition" in vmpfc_scores.columns:
        merge_keys = ["Subject", "Game", "Condition", "condition_label", "reveal_idx"]

    df_state = df_state.merge(
        acc_scores[merge_keys + ["acc_PC1", "acc_PC2", "acc_PC3"]],
        on=merge_keys,
        how="left",
    )
    df_state = df_state.merge(
        vmpfc_scores[merge_keys + ["vmpfc_PC1", "vmpfc_PC2", "vmpfc_PC3"]],
        on=merge_keys,
        how="left",
    )

    for idx in [1, 2, 3]:
        df_state[f"roi_pc{idx}_diff"] = df_state[f"acc_PC{idx}"] - df_state[f"vmpfc_PC{idx}"]

    def plot_roi_trajectory(df_in, roi_prefix, explained_series, outpath, valid_flag_col):
        sub0 = df_in[df_in[valid_flag_col] == 1].copy()
        sub0 = sub0.dropna(subset=[f"{roi_prefix}_PC1", f"{roi_prefix}_PC2"])
        if len(sub0) == 0:
            return
        subj_traj = (
            sub0.groupby(["Subject", "condition_label", "reveal_idx"])[
                [f"{roi_prefix}_PC1", f"{roi_prefix}_PC2"]
            ]
            .mean()
            .reset_index()
        )
        grand = (
            subj_traj.groupby(["condition_label", "reveal_idx"])[
                [f"{roi_prefix}_PC1", f"{roi_prefix}_PC2"]
            ]
            .mean()
            .reset_index()
        )
        plt.figure(figsize=(8, 6))
        for cond in ["equal", "unequal"]:
            sub = grand[grand["condition_label"] == cond].sort_values("reveal_idx")
            if len(sub) == 0:
                continue
            plt.plot(sub[f"{roi_prefix}_PC1"], sub[f"{roi_prefix}_PC2"], marker="o", label=cond)
            for _, row in sub.iterrows():
                plt.text(row[f"{roi_prefix}_PC1"], row[f"{roi_prefix}_PC2"], str(int(row["reveal_idx"])), fontsize=9)
        plt.axhline(0, linewidth=0.8)
        plt.axvline(0, linewidth=0.8)
        plt.xlabel(f"{roi_prefix.upper()} PC1 ({explained_series.iloc[0] * 100:.1f}%)")
        plt.ylabel(f"{roi_prefix.upper()} PC2 ({explained_series.iloc[1] * 100:.1f}%)")
        plt.title(f"{roi_prefix.upper()} subject-centered trajectory")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outpath, dpi=200)
        plt.close()

    plot_roi_trajectory(df_state, "acc", acc_explained, output_dir / "acc_subject_centered_trajectory.png", "has_acc_data")
    plot_roi_trajectory(df_state, "vmpfc", vmpfc_explained, output_dir / "vmpfc_subject_centered_trajectory.png", "has_vmpfc_data")

    def compute_geometry_metrics_for_group(group, pc_cols, prefix):
        group = group.sort_values("reveal_idx").copy().dropna(subset=pc_cols)
        x = group[pc_cols].to_numpy(dtype=float)
        if x.shape[0] < 2:
            return None
        step_vecs = np.diff(x, axis=0)
        step_sizes = np.sqrt(np.sum(step_vecs ** 2, axis=1))
        traj_length = np.sum(step_sizes)
        displacement = np.sqrt(np.sum((x[-1] - x[0]) ** 2))
        turning_angles = []
        for idx in range(len(step_vecs) - 1):
            a = step_vecs[idx]
            b = step_vecs[idx + 1]
            na = np.linalg.norm(a)
            nb = np.linalg.norm(b)
            if na < 1e-8 or nb < 1e-8:
                turning_angles.append(np.nan)
                continue
            cosang = np.dot(a, b) / (na * nb)
            turning_angles.append(np.arccos(np.clip(cosang, -1, 1)))
        terminal = x[-1]
        dist_to_terminal = np.sqrt(np.sum((x - terminal) ** 2, axis=1))
        return {
            f"{prefix}_traj_length": traj_length,
            f"{prefix}_mean_step": np.mean(step_sizes),
            f"{prefix}_max_step": np.max(step_sizes),
            f"{prefix}_displacement": displacement,
            f"{prefix}_efficiency": displacement / traj_length if traj_length > 1e-8 else np.nan,
            f"{prefix}_mean_turn_angle": np.nanmean(turning_angles) if len(turning_angles) > 0 else np.nan,
            f"{prefix}_mean_dist_to_terminal": np.mean(dist_to_terminal[:-1]) if len(dist_to_terminal) > 1 else np.nan,
        }

    geometry_rows = []
    for (subj, game, cond), group in df_state.groupby(["Subject", "Game", "condition_label"]):
        row = {
            "Subject": subj,
            "Game": game,
            "condition_label": cond,
            "has_acc_data": int(group["has_acc_data"].iloc[0]),
            "has_vmpfc_data": int(group["has_vmpfc_data"].iloc[0]),
        }
        if row["has_acc_data"] == 1:
            metrics = compute_geometry_metrics_for_group(group, ["acc_PC1", "acc_PC2", "acc_PC3"], "acc")
            if metrics is not None:
                row.update(metrics)
        if row["has_vmpfc_data"] == 1:
            metrics = compute_geometry_metrics_for_group(group, ["vmpfc_PC1", "vmpfc_PC2", "vmpfc_PC3"], "vmpfc")
            if metrics is not None:
                row.update(metrics)
        if row["has_acc_data"] == 1 and row["has_vmpfc_data"] == 1:
            metrics = compute_geometry_metrics_for_group(
                group,
                ["acc_PC1", "acc_PC2", "acc_PC3", "vmpfc_PC1", "vmpfc_PC2", "vmpfc_PC3"],
                "joint",
            )
            if metrics is not None:
                row.update(metrics)
        geometry_rows.append(row)

    geometry_df = pd.DataFrame(geometry_rows)
    geometry_df.to_csv(output_dir / "trajectory_geometry_metrics.csv", index=False)
    geom_summary = geometry_df.groupby("condition_label").mean(numeric_only=True)
    geom_summary.to_csv(output_dir / "trajectory_geometry_condition_means.csv")

    geom_stats_txt = []
    geom_metric_cols = [
        col for col in geometry_df.columns
        if col not in ["Subject", "Game", "condition_label", "has_acc_data", "has_vmpfc_data"]
    ]
    for metric in geom_metric_cols:
        subdf = geometry_df[["Subject", "condition_label", metric]].dropna().copy()
        if len(subdf) < 10 or subdf[metric].nunique() < 3:
            continue
        try:
            model = smf.ols(f"{metric} ~ C(condition_label)", data=subdf).fit(
                cov_type="cluster",
                cov_kwds={"groups": subdf["Subject"]},
            )
            geom_stats_txt.append(f"\n=== {metric} ===\n")
            geom_stats_txt.append(str(model.summary()))
        except Exception as exc:
            geom_stats_txt.append(f"\n=== {metric} ===\nFAILED: {exc}\n")
    (output_dir / "trajectory_geometry_stats.txt").write_text("\n".join(geom_stats_txt), encoding="utf-8")

    def is_binary_series(series):
        vals = pd.Series(series).dropna().unique()
        if len(vals) <= 2:
            return True
        return set(vals).issubset({0, 1, False, True})

    def cross_validated_decode(x, y, binary=True, n_splits=5):
        if len(y) < 20:
            return np.nan
        if binary:
            y2 = pd.Series(y).astype(int).to_numpy()
            if len(np.unique(y2)) < 2:
                return np.nan
            class_counts = np.bincount(y2)
            if len(class_counts) < 2 or np.min(class_counts) < 2:
                return np.nan
            nfold = min(n_splits, int(np.min(class_counts)))
            if nfold < 2:
                return np.nan
            clf = LogisticRegression(max_iter=2000)
            cv = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=random_state)
            return np.mean(cross_val_score(clf, x, y2, cv=cv, scoring="roc_auc"))
        y2 = pd.Series(y).astype(float).to_numpy()
        if np.nanstd(y2) < 1e-8:
            return np.nan
        nfold = min(n_splits, len(y2))
        if nfold < 2:
            return np.nan
        reg = LinearRegression()
        cv = KFold(n_splits=nfold, shuffle=True, random_state=random_state)
        return np.mean(cross_val_score(reg, x, y2, cv=cv, scoring="r2"))

    repr_rows = []
    for reveal in sorted(df_state["reveal_idx"].unique()):
        sub_r = df_state[df_state["reveal_idx"] == reveal].copy()
        for target in available_targets:
            yt = sub_r[target]
            binary = is_binary_series(yt)
            roi_specs = [
                ("acc", ["acc_PC1", "acc_PC2", "acc_PC3"], sub_r["has_acc_data"].eq(1)),
                ("vmpfc", ["vmpfc_PC1", "vmpfc_PC2", "vmpfc_PC3"], sub_r["has_vmpfc_data"].eq(1)),
                (
                    "joint",
                    ["acc_PC1", "acc_PC2", "acc_PC3", "vmpfc_PC1", "vmpfc_PC2", "vmpfc_PC3"],
                    sub_r["has_acc_data"].eq(1) & sub_r["has_vmpfc_data"].eq(1),
                ),
            ]
            for roi_name, roi_cols, roi_mask in roi_specs:
                valid = yt.notna() & roi_mask & sub_r[roi_cols].notna().all(axis=1)
                x = sub_r.loc[valid, roi_cols].to_numpy(dtype=float)
                y = yt.loc[valid].to_numpy()
                score = cross_validated_decode(x, y, binary=binary, n_splits=5) if len(y) >= 20 else np.nan
                repr_rows.append({
                    "reveal_idx": reveal,
                    "target": target,
                    "roi": roi_name,
                    "target_type": "binary" if binary else "continuous",
                    "cv_score": score,
                    "n_rows": len(y),
                })

    repr_df = pd.DataFrame(repr_rows)
    repr_df.to_csv(output_dir / "representation_emergence_scores.csv", index=False)

    def plot_repr_target(repr_df_in, target_name, outpath):
        sub = repr_df_in[repr_df_in["target"] == target_name].copy()
        if len(sub) == 0:
            return
        plt.figure(figsize=(8, 5))
        for roi_name in ["acc", "vmpfc", "joint"]:
            ss = sub[sub["roi"] == roi_name].sort_values("reveal_idx")
            if len(ss) == 0:
                continue
            plt.plot(ss["reveal_idx"], ss["cv_score"], marker="o", label=roi_name)
        plt.xlabel("Reveal index")
        plt.ylabel("CV ROC-AUC / CV R2")
        plt.title(f"Representation emergence: {target_name}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outpath, dpi=200)
        plt.close()

    for target in [
        "best_deck_snapshot",
        "value_gap_snapshot",
        "sampling_imbalance_after",
        "current_deck_was_least_sampled_before",
        "current_reward",
    ]:
        if target in available_targets:
            plot_repr_target(repr_df, target, output_dir / f"repr_emergence_{target}.png")

    emergence_stats = []
    for (target, roi), group in repr_df.groupby(["target", "roi"]):
        group = group.dropna(subset=["cv_score"]).sort_values("reveal_idx")
        if len(group) < 3:
            continue
        x = group["reveal_idx"].to_numpy(dtype=float)
        y = group["cv_score"].to_numpy(dtype=float)
        emergence_stats.append({
            "target": target,
            "roi": roi,
            "emergence_slope": np.polyfit(x, y, deg=1)[0],
            "mean_score": np.mean(y),
            "max_score": np.max(y),
        })
    emergence_stats_df = pd.DataFrame(emergence_stats)
    emergence_stats_df.to_csv(output_dir / "representation_emergence_slopes.csv", index=False)

    terminal = df_state[df_state["reveal_idx"] == 6].copy()
    terminal_results = []

    def run_terminal_decode(df_term, roi_name, roi_cols, y, y_name, binary=True):
        valid = pd.Series(y).notna().values
        x = df_term.loc[valid, roi_cols].to_numpy(dtype=float)
        yy = pd.Series(y).loc[valid].to_numpy()
        score = cross_validated_decode(x, yy, binary=binary, n_splits=5)
        return {
            "analysis": y_name,
            "roi": roi_name,
            "score": score,
            "score_type": "ROC-AUC" if binary else "CV_R2",
            "n_rows": len(yy),
        }

    term_acc = terminal[(terminal["has_acc_data"] == 1) & terminal[["acc_PC1", "acc_PC2", "acc_PC3"]].notna().all(axis=1)].copy()
    term_vmpfc = terminal[(terminal["has_vmpfc_data"] == 1) & terminal[["vmpfc_PC1", "vmpfc_PC2", "vmpfc_PC3"]].notna().all(axis=1)].copy()
    term_joint = terminal[
        (terminal["has_acc_data"] == 1)
        & (terminal["has_vmpfc_data"] == 1)
        & terminal[["acc_PC1", "acc_PC2", "acc_PC3", "vmpfc_PC1", "vmpfc_PC2", "vmpfc_PC3"]].notna().all(axis=1)
    ].copy()

    terminal_results.append(run_terminal_decode(term_acc, "acc", ["acc_PC1", "acc_PC2", "acc_PC3"], (term_acc["condition_label"] == "unequal").astype(int), "decode_condition", binary=True))
    terminal_results.append(run_terminal_decode(term_vmpfc, "vmpfc", ["vmpfc_PC1", "vmpfc_PC2", "vmpfc_PC3"], (term_vmpfc["condition_label"] == "unequal").astype(int), "decode_condition", binary=True))
    terminal_results.append(run_terminal_decode(term_joint, "joint", ["acc_PC1", "acc_PC2", "acc_PC3", "vmpfc_PC1", "vmpfc_PC2", "vmpfc_PC3"], (term_joint["condition_label"] == "unequal").astype(int), "decode_condition", binary=True))

    if "best_deck_snapshot" in terminal.columns and is_binary_series(terminal["best_deck_snapshot"]):
        terminal_results.append(run_terminal_decode(term_acc, "acc", ["acc_PC1", "acc_PC2", "acc_PC3"], term_acc["best_deck_snapshot"], "decode_best_deck_snapshot", binary=True))
        terminal_results.append(run_terminal_decode(term_vmpfc, "vmpfc", ["vmpfc_PC1", "vmpfc_PC2", "vmpfc_PC3"], term_vmpfc["best_deck_snapshot"], "decode_best_deck_snapshot", binary=True))
        terminal_results.append(run_terminal_decode(term_joint, "joint", ["acc_PC1", "acc_PC2", "acc_PC3", "vmpfc_PC1", "vmpfc_PC2", "vmpfc_PC3"], term_joint["best_deck_snapshot"], "decode_best_deck_snapshot", binary=True))

    if "value_gap_snapshot" in terminal.columns:
        terminal_results.append(run_terminal_decode(term_acc, "acc", ["acc_PC1", "acc_PC2", "acc_PC3"], term_acc["value_gap_snapshot"], "predict_value_gap_snapshot", binary=False))
        terminal_results.append(run_terminal_decode(term_vmpfc, "vmpfc", ["vmpfc_PC1", "vmpfc_PC2", "vmpfc_PC3"], term_vmpfc["value_gap_snapshot"], "predict_value_gap_snapshot", binary=False))
        terminal_results.append(run_terminal_decode(term_joint, "joint", ["acc_PC1", "acc_PC2", "acc_PC3", "vmpfc_PC1", "vmpfc_PC2", "vmpfc_PC3"], term_joint["value_gap_snapshot"], "predict_value_gap_snapshot", binary=False))

    terminal_results_df = pd.DataFrame(terminal_results)
    terminal_results_df.to_csv(output_dir / "terminal_state_decode_results.csv", index=False)

    dispersion_rows = []
    roi_term_map = {
        "acc": (term_acc, ["acc_PC1", "acc_PC2", "acc_PC3"]),
        "vmpfc": (term_vmpfc, ["vmpfc_PC1", "vmpfc_PC2", "vmpfc_PC3"]),
        "joint": (term_joint, ["acc_PC1", "acc_PC2", "acc_PC3", "vmpfc_PC1", "vmpfc_PC2", "vmpfc_PC3"]),
    }
    for roi_name, (df_term_roi, roi_cols) in roi_term_map.items():
        for cond, group in df_term_roi.groupby("condition_label"):
            group = group.dropna(subset=roi_cols)
            if len(group) == 0:
                continue
            x = group[roi_cols].to_numpy(dtype=float)
            centroid = x.mean(axis=0, keepdims=True)
            distances = np.sqrt(np.sum((x - centroid) ** 2, axis=1))
            for idx, value in enumerate(distances):
                dispersion_rows.append({
                    "roi": roi_name,
                    "condition_label": cond,
                    "dispersion": value,
                    "Subject": group.iloc[idx]["Subject"],
                    "Game": group.iloc[idx]["Game"],
                })
    dispersion_df = pd.DataFrame(dispersion_rows)
    dispersion_df.to_csv(output_dir / "terminal_state_dispersion.csv", index=False)

    dispersion_stats_txt = []
    for roi_name in dispersion_df["roi"].unique():
        sub = dispersion_df[dispersion_df["roi"] == roi_name].copy()
        try:
            model = smf.ols("dispersion ~ C(condition_label)", data=sub).fit(
                cov_type="cluster",
                cov_kwds={"groups": sub["Subject"]},
            )
            dispersion_stats_txt.append(f"\n=== terminal dispersion: {roi_name} ===\n")
            dispersion_stats_txt.append(str(model.summary()))
        except Exception as exc:
            dispersion_stats_txt.append(f"\n=== terminal dispersion: {roi_name} ===\nFAILED: {exc}\n")
    (output_dir / "terminal_dispersion_stats.txt").write_text("\n".join(dispersion_stats_txt), encoding="utf-8")

    dist_rows = []
    for (subj, game, cond), group in df_state.groupby(["Subject", "Game", "condition_label"]):
        group = group.sort_values("reveal_idx").copy()
        distance_specs = [
            ("acc", ["acc_PC1", "acc_PC2", "acc_PC3"], group["has_acc_data"].iloc[0] == 1),
            ("vmpfc", ["vmpfc_PC1", "vmpfc_PC2", "vmpfc_PC3"], group["has_vmpfc_data"].iloc[0] == 1),
            (
                "joint",
                ["acc_PC1", "acc_PC2", "acc_PC3", "vmpfc_PC1", "vmpfc_PC2", "vmpfc_PC3"],
                group["has_acc_data"].iloc[0] == 1 and group["has_vmpfc_data"].iloc[0] == 1,
            ),
        ]
        for roi_name, roi_cols, enabled in distance_specs:
            if not enabled:
                continue
            gg = group.dropna(subset=roi_cols)
            if len(gg) < 1:
                continue
            x = gg[roi_cols].to_numpy(dtype=float)
            terminal_state = x[-1]
            distances = np.sqrt(np.sum((x - terminal_state) ** 2, axis=1))
            for reveal, distance in zip(gg["reveal_idx"].values, distances):
                dist_rows.append({
                    "Subject": subj,
                    "Game": game,
                    "condition_label": cond,
                    "reveal_idx": reveal,
                    "roi": roi_name,
                    "distance_to_terminal": distance,
                })
    dist_to_terminal_df = pd.DataFrame(dist_rows)
    dist_to_terminal_df.to_csv(output_dir / "distance_to_terminal_by_reveal.csv", index=False)

    for roi_name in ["acc", "vmpfc", "joint"]:
        sub = dist_to_terminal_df[dist_to_terminal_df["roi"] == roi_name].copy()
        if len(sub) == 0:
            continue
        mean_df = sub.groupby(["condition_label", "reveal_idx"])["distance_to_terminal"].mean().reset_index()
        plt.figure(figsize=(7, 5))
        for cond in ["equal", "unequal"]:
            ss = mean_df[mean_df["condition_label"] == cond].sort_values("reveal_idx")
            if len(ss) == 0:
                continue
            plt.plot(ss["reveal_idx"], ss["distance_to_terminal"], marker="o", label=cond)
        plt.xlabel("Reveal index")
        plt.ylabel("Distance to terminal state")
        plt.title(f"{roi_name.upper()} convergence to terminal state")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"{roi_name}_distance_to_terminal.png", dpi=200)
        plt.close()

    roi_div_df = df_state[
        (df_state["has_acc_data"] == 1)
        & (df_state["has_vmpfc_data"] == 1)
        & df_state[["roi_pc1_diff", "roi_pc2_diff", "roi_pc3_diff"]].notna().all(axis=1)
    ].copy()
    roi_div_summary = (
        roi_div_df.groupby(["condition_label", "reveal_idx"])[
            ["roi_pc1_diff", "roi_pc2_diff", "roi_pc3_diff"]
        ]
        .mean()
        .reset_index()
    )
    roi_div_summary.to_csv(output_dir / "roi_divergence_summary.csv", index=False)

    for diff_col in ["roi_pc1_diff", "roi_pc2_diff", "roi_pc3_diff"]:
        plt.figure(figsize=(7, 5))
        for cond in ["equal", "unequal"]:
            sub = roi_div_summary[roi_div_summary["condition_label"] == cond].sort_values("reveal_idx")
            if len(sub) == 0:
                continue
            plt.plot(sub["reveal_idx"], sub[diff_col], marker="o", label=cond)
        plt.xlabel("Reveal index")
        plt.ylabel(diff_col)
        plt.title(f"ACC-vmPFC divergence: {diff_col}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"{diff_col}_trajectory.png", dpi=200)
        plt.close()

    df_state.to_csv(output_dir / "reveal_level_with_roi_state_scores.csv", index=False)

    subject_inclusion = pd.DataFrame({"Subject": sorted(df["Subject"].unique())})
    subject_inclusion["has_acc_data"] = subject_inclusion["Subject"].isin(acc_valid_subjects).astype(int)
    subject_inclusion["has_vmpfc_data"] = subject_inclusion["Subject"].isin(vmpfc_valid_subjects).astype(int)
    subject_inclusion.to_csv(output_dir / "roi_subject_inclusion.csv", index=False)

    summary_txt = [
        f"Subject filter: {', '.join(subject_filter) if subject_filter is not None else 'all'}",
        "",
        "ROI-valid subject counts:",
        f"ACC valid subjects: {len(acc_valid_subjects)}",
        f"vmPFC valid subjects: {len(vmpfc_valid_subjects)}",
        "",
        "ACC explained variance:",
        str(acc_explained),
        "",
        "vmPFC explained variance:",
        str(vmpfc_explained),
        "",
        "Geometry condition means:",
        str(geom_summary),
        "",
        "Terminal decode results:",
        str(terminal_results_df),
        "",
        "Representation emergence slopes:",
        str(emergence_stats_df.sort_values(["target", "roi"]).head(50)) if len(emergence_stats_df) > 0 else "No emergence stats available.",
    ]
    (output_dir / "quick_summary.txt").write_text("\n".join(summary_txt), encoding="utf-8")

    print("PCA/state-space analysis complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
