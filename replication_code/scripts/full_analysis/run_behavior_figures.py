#!/usr/bin/env python3
"""Generate behavioral summary statistics and panels from gkRL outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from src.config import load_demo_config, require_section
from src.io_utils import resolve_repo_path, repo_root_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate behavioral figures from model-derived outputs.")
    parser.add_argument(
        "--config",
        default="replication_code/config/full_analysis_config.yaml",
        help="Relative path to the full-analysis config file.",
    )
    return parser.parse_args()


def main() -> int:
    try:
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from scipy.optimize import curve_fit
        from scipy.stats import wilcoxon
    except ImportError as exc:
        raise SystemExit(
            "Missing behavioral-figure dependency. Install the full environment in INSTALL.md. "
            f"Original import error: {exc}"
        ) from exc

    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path
    config = load_demo_config(config_path)
    repo_root = repo_root_from_config(config_path)
    paths = require_section(config, "paths")

    model_output_dir = resolve_repo_path(repo_root, paths["behavioral_model_outputs"]) / "weighted"
    configured_result_dir = resolve_repo_path(repo_root, paths["behavioral_figure_inputs"])
    result_dir = model_output_dir if list(model_output_dir.glob("*_game_metrics.csv")) else configured_result_dir
    param_dir = resolve_repo_path(repo_root, paths["behavioral_param_inputs"])
    output_dir = resolve_repo_path(repo_root, paths["outputs"]) / "behavior_figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model metrics directory: {result_dir.relative_to(repo_root)}")
    print(f"Output directory: {output_dir.relative_to(repo_root)}")

    def sigmoid_2p(x, slope, midpoint):
        return 1.0 / (1.0 + np.exp(-slope * (x - midpoint)))

    def cohen_d_paired(a, b):
        diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
        if len(diff) < 2 or np.nanstd(diff, ddof=1) == 0:
            return np.nan
        return np.nanmean(diff) / np.nanstd(diff, ddof=1)

    def argmax_deck(row, prefix):
        values = [row[f"{prefix}_Deck_{idx}"] for idx in [1, 2, 3]]
        return int(np.nanargmax(values) + 1)

    metric_files = sorted(result_dir.glob("*_game_metrics.csv"))
    if not metric_files:
        raise FileNotFoundError(f"No *_game_metrics.csv files found in {result_dir}")

    trial_rows = []
    for path in metric_files:
        subject = path.name.replace("_game_metrics.csv", "")
        df = pd.read_csv(path)
        df["Subject"] = subject
        for _, row in df.iterrows():
            chosen = int(row["Chosen_Deck"])
            q_chosen = row[f"Q_Deck_{chosen}"]
            v_chosen = row[f"V_Deck_{chosen}"]
            q_unchosen = np.mean([row[f"Q_Deck_{idx}"] for idx in [1, 2, 3] if idx != chosen])
            v_unchosen = np.mean([row[f"V_Deck_{idx}"] for idx in [1, 2, 3] if idx != chosen])
            best_q = argmax_deck(row, "Q")
            best_v = argmax_deck(row, "V")
            trial_rows.append({
                "Subject": subject,
                "Game": row["Game"],
                "Chosen_Deck": chosen,
                "Chosen_Prob": row["Chosen_Prob"],
                "omega": row["omega"],
                "beta": row["beta"],
                "relative_Q": q_chosen - q_unchosen,
                "relative_V": v_chosen - v_unchosen,
                "best_Q_deck": best_q,
                "best_V_deck": best_v,
                "conflict": int(best_q != best_v),
                "choose_best_Q": int(chosen == best_q),
                "choose_best_V": int(chosen == best_v),
            })
    trial_df = pd.DataFrame(trial_rows)
    trial_df.to_csv(output_dir / "behavior_trial_level_summary.csv", index=False)

    subj_df = (
        trial_df.groupby("Subject")
        .agg(
            mean_chosen_prob=("Chosen_Prob", "mean"),
            mean_omega=("omega", "mean"),
            mean_beta=("beta", "mean"),
            n_games=("Game", "count"),
            n_conflict=("conflict", "sum"),
            conflict_choose_best_Q=("choose_best_Q", lambda x: np.nan),
            conflict_choose_best_V=("choose_best_V", lambda x: np.nan),
        )
        .reset_index()
    )
    conflict_subject_rows = []
    for subject, group in trial_df.groupby("Subject"):
        conflict = group[group["conflict"] == 1]
        conflict_subject_rows.append({
            "Subject": subject,
            "n_conflict": len(conflict),
            "prop_choose_best_Q": conflict["choose_best_Q"].mean() if len(conflict) else np.nan,
            "prop_choose_best_V": conflict["choose_best_V"].mean() if len(conflict) else np.nan,
        })
    conflict_subj_df = pd.DataFrame(conflict_subject_rows)
    subj_df = subj_df.drop(columns=["conflict_choose_best_Q", "conflict_choose_best_V"]).merge(conflict_subj_df, on="Subject", how="left")
    subj_df.to_csv(output_dir / "behavior_subject_level_summary.csv", index=False)

    param_rows = []
    for path in sorted(param_dir.glob("*_subject_params.csv")):
        subject = path.name.replace("_subject_params.csv", "")
        tmp = pd.read_csv(path)
        name_col = tmp.columns[0]
        for _, row in tmp.iterrows():
            param_rows.append({
                "Subject": subject,
                "parameter": row[name_col],
                "mean": row.get("mean", np.nan),
                "sd": row.get("sd", np.nan),
                "hdi_3_percent": row.get("hdi_3%", np.nan),
                "hdi_97_percent": row.get("hdi_97%", np.nan),
                "r_hat": row.get("r_hat", np.nan),
            })
    if param_rows:
        pd.DataFrame(param_rows).to_csv(output_dir / "behavior_parameter_summary.csv", index=False)

    stats_rows = []
    chance = 1.0 / 3.0
    if subj_df["mean_chosen_prob"].notna().sum() > 0:
        vals = subj_df["mean_chosen_prob"].dropna()
        if len(vals) > 1:
            stat = wilcoxon(vals - chance)
            stats_rows.append({"test": "chosen_probability_vs_chance", "statistic": stat.statistic, "p_value": stat.pvalue, "n": len(vals)})
        else:
            stats_rows.append({"test": "chosen_probability_vs_chance", "statistic": np.nan, "p_value": np.nan, "n": len(vals)})
    if subj_df["mean_omega"].notna().sum() > 0:
        vals = subj_df["mean_omega"].dropna()
        if len(vals) > 1:
            stat = wilcoxon(vals)
            stats_rows.append({"test": "omega_vs_zero", "statistic": stat.statistic, "p_value": stat.pvalue, "n": len(vals)})
        else:
            stats_rows.append({"test": "omega_vs_zero", "statistic": np.nan, "p_value": np.nan, "n": len(vals)})
    conflict = conflict_subj_df.dropna(subset=["prop_choose_best_V", "prop_choose_best_Q"])
    if len(conflict) > 1:
        stat = wilcoxon(conflict["prop_choose_best_V"], conflict["prop_choose_best_Q"])
        stats_rows.append({
            "test": "conflict_choose_best_V_vs_best_Q",
            "statistic": stat.statistic,
            "p_value": stat.pvalue,
            "n": len(conflict),
            "cohen_d_paired": cohen_d_paired(conflict["prop_choose_best_V"], conflict["prop_choose_best_Q"]),
        })
    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(output_dir / "behavior_stats.csv", index=False)
    (output_dir / "behavior_stats.txt").write_text(stats_df.to_string(index=False), encoding="utf-8")

    plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42, "ps.fonttype": 42})

    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.hist(trial_df["Chosen_Prob"].dropna(), bins=20, color="#3C5488", alpha=0.8)
    ax.axvline(chance, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Predicted probability assigned to chosen option")
    ax.set_ylabel("Games")
    fig.tight_layout()
    fig.savefig(output_dir / "panel_A_chosen_probability.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    sub = trial_df.dropna(subset=["relative_V", "choose_best_V"]).copy()
    if len(sub) > 5:
        bins = np.quantile(sub["relative_V"], np.linspace(0, 1, 8))
        bins = np.unique(bins)
        if len(bins) > 2:
            sub["bin"] = pd.cut(sub["relative_V"], bins=bins, include_lowest=True)
            grouped = sub.groupby("bin", observed=True).agg(x=("relative_V", "mean"), y=("choose_best_V", "mean")).dropna()
            ax.scatter(grouped["x"], grouped["y"], color="#D36179", s=20)
            try:
                popt, _ = curve_fit(sigmoid_2p, grouped["x"], grouped["y"], p0=[1.0, 0.0], maxfev=10000)
                x_fit = np.linspace(grouped["x"].min(), grouped["x"].max(), 200)
                ax.plot(x_fit, sigmoid_2p(x_fit, *popt), color="black", linewidth=1.5)
            except Exception:
                pass
    ax.set_xlabel("Relative composite value")
    ax.set_ylabel("Choice probability")
    fig.tight_layout()
    fig.savefig(output_dir / "panel_B_value_psychometric.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(2.4, 2.6))
    ax.scatter(np.ones(len(subj_df)), subj_df["mean_omega"], color="#185FA5", alpha=0.7)
    ax.boxplot(subj_df["mean_omega"].dropna(), positions=[1], widths=0.35)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xticks([1])
    ax.set_xticklabels(["omega"])
    ax.set_ylabel("Subject mean omega")
    fig.tight_layout()
    fig.savefig(output_dir / "panel_C_omega_distribution.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.0, 2.8))
    if len(conflict_subj_df) > 0:
        xs = [0, 1]
        for _, row in conflict_subj_df.iterrows():
            ax.plot(xs, [row["prop_choose_best_Q"], row["prop_choose_best_V"]], color="0.65", linewidth=0.8)
            ax.scatter(xs, [row["prop_choose_best_Q"], row["prop_choose_best_V"]], color=["#7F8C8E", "#E74C3C"], s=18)
        ax.set_xticks(xs)
        ax.set_xticklabels(["Highest Q", "Highest V"])
        ax.set_ylabel("Conflict-trial choice proportion")
    fig.tight_layout()
    fig.savefig(output_dir / "panel_D_conflict_choices.png", dpi=300)
    plt.close(fig)

    print("Behavioral figure/statistics generation complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
