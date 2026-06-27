#!/usr/bin/env python3
"""Run theta-band ACC-vmPFC PLV analysis from the original PLV notebook."""

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


def _parse_float(value: str, default: float) -> float:
    try:
        return float(str(value).strip())
    except Exception:
        return default


def _parse_subjects(value: str) -> list[str] | None:
    text = str(value).strip()
    if not text or text.lower() in {"all", "*"}:
        return None
    return [part.strip() for part in text.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run theta PLV analysis.")
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
        import mne
        import statsmodels.api as sm
        from scipy.signal import hilbert
        from scipy.stats import ttest_1samp
    except ImportError as exc:
        raise SystemExit(
            "Missing PLV dependency. Install the full environment in INSTALL.md. "
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

    acc_dir = resolve_repo_path(repo_root, paths["seeg_roi_main_acc"])
    vmpfc_dir = resolve_repo_path(repo_root, paths["seeg_roi_main_vmpfc"])
    model_dir = resolve_repo_path(repo_root, paths["weighted_model_metrics"])
    output_dir = resolve_repo_path(repo_root, paths["outputs"]) / "plv"
    output_dir.mkdir(parents=True, exist_ok=True)

    fmin = _parse_float(params.get("theta_plv_fmin_hz", "4"), 4.0)
    fmax = _parse_float(params.get("theta_plv_fmax_hz", "8"), 8.0)
    tmin = _parse_float(params.get("plv_decision_tmin_s", "-1.1"), -1.1)
    tmax = _parse_float(params.get("plv_decision_tmax_s", "0.15"), 0.15)
    n_perm = _parse_int(params.get("plv_permutations", "5000"), 5000)
    random_state = _parse_int(params.get("random_state", "42"), 42)
    subject_filter = _parse_subjects(params.get("subjects", "all"))

    print(f"PLV band: {fmin:g}-{fmax:g} Hz")
    print(f"Window: {tmin} to {tmax} s")

    def zscore(values):
        values = np.asarray(values, dtype=float)
        sd = np.nanstd(values)
        if sd < 1e-12:
            return np.full_like(values, np.nan)
        return (values - np.nanmean(values)) / sd

    def build_behavior_df(subject_id):
        path = model_dir / f"{subject_id}_game_metrics.csv"
        if not path.exists():
            return None
        gd = pd.read_csv(path)
        rows = []
        for trial_index in range(len(gd)):
            game = trial_index + 1
            game_data = gd[gd["Game"] == game]
            if game_data.empty:
                continue
            row = game_data.iloc[0]
            chosen = int(row["Chosen_Deck"])
            q_cols = [f"Q_Deck_{idx}" for idx in [1, 2, 3]]
            info_col = f"Weighted_I_transformed_Deck_{chosen}"
            reward = row[f"Q_Deck_{chosen}"] - row[q_cols].mean()
            info = row[info_col]
            rows.append({"trial_index": trial_index, "relative_reward": reward, "information_gain": info})
        return pd.DataFrame(rows) if rows else None

    def drop_rejected_epochs(epochs):
        tags = [tag for tag in epochs.event_id if "99" not in tag]
        return epochs[tags] if tags else epochs[:0]

    def align_epochs(ep1, ep2):
        ep1 = drop_rejected_epochs(ep1)
        ep2 = drop_rejected_epochs(ep2)
        if len(ep1) != len(ep2):
            raise ValueError(f"Epoch count mismatch after removing tag 99: {len(ep1)} vs {len(ep2)}")
        return ep1, ep2

    trial_rows = []
    subjects = sorted(path.stem for path in acc_dir.glob("*.set") if (vmpfc_dir / path.name).exists())
    if subject_filter is not None:
        subjects = [subject for subject in subjects if subject in subject_filter]
    if not subjects:
        raise FileNotFoundError("No subjects with both ACC and vmPFC .set files were found.")

    for subject in subjects:
        print(f"Subject {subject}")
        try:
            ep_roi1 = mne.read_epochs_eeglab(str(vmpfc_dir / f"{subject}.set"), verbose=False)
            ep_roi2 = mne.read_epochs_eeglab(str(acc_dir / f"{subject}.set"), verbose=False)
            ep_roi1, ep_roi2 = align_epochs(ep_roi1, ep_roi2)
            ep_roi1 = ep_roi1.copy().filter(fmin, fmax, fir_design="firwin", verbose=False).crop(tmin=tmin, tmax=tmax)
            ep_roi2 = ep_roi2.copy().filter(fmin, fmax, fir_design="firwin", verbose=False).crop(tmin=tmin, tmax=tmax)
            data1 = ep_roi1.get_data(copy=False)
            data2 = ep_roi2.get_data(copy=False)
            phase1 = np.angle(hilbert(data1, axis=-1))
            phase2 = np.angle(hilbert(data2, axis=-1))
            for roi1_idx, roi1_name in enumerate(ep_roi1.ch_names):
                for roi2_idx, roi2_name in enumerate(ep_roi2.ch_names):
                    phase_diff = phase1[:, roi1_idx, :] - phase2[:, roi2_idx, :]
                    plv = np.abs(np.mean(np.exp(1j * phase_diff), axis=1))
                    for trial_index, value in enumerate(plv):
                        trial_rows.append({
                            "subject": subject,
                            "trial_index": trial_index,
                            "roi1_channel": roi1_name,
                            "roi2_channel": roi2_name,
                            "plv": float(value),
                        })
        except Exception as exc:
            print(f"  Failed: {exc}")

    plv_df = pd.DataFrame(trial_rows)
    plv_path = output_dir / f"trial_level_plv_{int(fmin)}-{int(fmax)}Hz.csv"
    plv_df.to_csv(plv_path, index=False)

    behavior_frames = []
    for subject in sorted(plv_df["subject"].dropna().astype(str).unique()):
        behav = build_behavior_df(subject)
        if behav is None:
            continue
        behav["subject"] = str(subject)
        behavior_frames.append(behav)
    if not behavior_frames:
        raise RuntimeError("No behavior metrics were available for PLV regression.")
    behavior_df = pd.concat(behavior_frames, ignore_index=True)
    plv_df["subject"] = plv_df["subject"].astype(str)
    merged = plv_df.merge(behavior_df, on=["subject", "trial_index"], how="inner")

    regression_rows = []
    x_list = []
    y_list = []
    channel_pairs = merged[["roi1_channel", "roi2_channel"]].drop_duplicates()
    for _, row in channel_pairs.iterrows():
        ch1 = row["roi1_channel"]
        ch2 = row["roi2_channel"]
        group = merged[(merged["roi1_channel"] == ch1) & (merged["roi2_channel"] == ch2)].copy()
        if len(group) < 20:
            continue
        group["reward_z"] = zscore(group["relative_reward"])
        group["info_z"] = zscore(group["information_gain"])
        group["plv_z"] = zscore(group["plv"])
        if group[["reward_z", "info_z", "plv_z"]].isna().any().any():
            continue
        slope, intercept = np.polyfit(group["reward_z"], group["info_z"], 1)
        group["info_orth"] = group["info_z"] - (slope * group["reward_z"] + intercept)
        x = sm.add_constant(group[["reward_z", "info_orth"]])
        y = group["plv_z"]
        try:
            model = sm.OLS(y, x, missing="drop").fit()
            regression_rows.append({
                "roi1_channel": ch1,
                "roi2_channel": ch2,
                "n_trials": len(group),
                "beta_reward": model.params["reward_z"],
                "p_reward": model.pvalues["reward_z"],
                "beta_info": model.params["info_orth"],
                "p_info": model.pvalues["info_orth"],
                "r_squared": model.rsquared,
            })
            x_list.append(x.to_numpy(dtype=float))
            y_list.append(y.to_numpy(dtype=float))
        except Exception as exc:
            warnings.warn(f"PLV regression failed for {ch1}-{ch2}: {exc}")

    results_df = pd.DataFrame(regression_rows)
    results_df.to_csv(output_dir / "observed_predict_plv_results.csv", index=False)
    results_df.to_csv(output_dir / "behavior_predict_plv_results.csv", index=False)

    summary_rows = []
    for var in ["reward", "info"]:
        betas = results_df[f"beta_{var}"].dropna().to_numpy(float)
        if len(betas) == 0:
            continue
        t_stat, p_val = ttest_1samp(betas, 0)
        summary_rows.append({
            "regressor": var,
            "mean_beta": np.mean(betas),
            "t_stat": t_stat,
            "p_parametric": p_val,
            "n_pairs": len(betas),
        })

    rng = np.random.default_rng(random_state)
    null_reward = np.zeros(n_perm)
    null_info = np.zeros(n_perm)
    obs_reward = results_df["beta_reward"].mean() if len(results_df) else np.nan
    obs_info = results_df["beta_info"].mean() if len(results_df) else np.nan
    x_pinv_list = [np.linalg.pinv(x) for x in x_list]
    for idx in range(n_perm):
        temp_reward = []
        temp_info = []
        for y_true, x_pinv in zip(y_list, x_pinv_list):
            betas = x_pinv @ rng.permutation(y_true)
            temp_reward.append(betas[1])
            temp_info.append(betas[2])
        null_reward[idx] = np.mean(temp_reward) if temp_reward else np.nan
        null_info[idx] = np.mean(temp_info) if temp_info else np.nan
    p_reward = (np.sum(np.abs(null_reward) >= abs(obs_reward)) + 1) / (n_perm + 1) if np.isfinite(obs_reward) else np.nan
    p_info = (np.sum(np.abs(null_info) >= abs(obs_info)) + 1) / (n_perm + 1) if np.isfinite(obs_info) else np.nan
    perm_summary = pd.DataFrame([
        {"regressor": "reward", "observed_mean_beta": obs_reward, "permutation_p": p_reward, "n_perm": n_perm},
        {"regressor": "info", "observed_mean_beta": obs_info, "permutation_p": p_info, "n_perm": n_perm},
    ])
    perm_summary.to_csv(output_dir / "permutation_summary.csv", index=False)
    pd.DataFrame({"reward_null": null_reward, "info_null": null_info}).to_csv(output_dir / "null_distributions.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    for ax, var, null, obs, p_value, color in [
        (axes[0], "reward", null_reward, obs_reward, p_reward, "#D36179"),
        (axes[1], "information", null_info, obs_info, p_info, "#7BB5B7"),
    ]:
        ax.hist(null[np.isfinite(null)], bins=40, color="0.75", edgecolor="white")
        ax.axvline(obs, color=color, linewidth=2, label=f"Observed, p={p_value:.4f}")
        ax.set_title(f"{var.capitalize()} PLV permutation")
        ax.set_xlabel("Mean beta under shuffled PLV")
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "permutation_null_distribution.png", dpi=300)
    plt.close(fig)

    pd.DataFrame(summary_rows).to_csv(output_dir / "pair_level_parametric_summary.csv", index=False)
    print("PLV analysis complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
