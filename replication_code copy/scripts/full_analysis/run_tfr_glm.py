#!/usr/bin/env python3
"""Run time-frequency neural encoding GLM from the original TFR notebook."""

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
    parser = argparse.ArgumentParser(description="Run ROI TFR GLM analysis.")
    parser.add_argument(
        "--config",
        default="replication_code/config/full_analysis_config.yaml",
        help="Relative path to the full-analysis config file.",
    )
    parser.add_argument("--roi", choices=["acc", "vmpfc", "both"], default="both")
    return parser.parse_args()


def main() -> int:
    warnings.filterwarnings("ignore")
    try:
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import mne
        from scipy.ndimage import gaussian_filter1d
        from sklearn.linear_model import LinearRegression
    except ImportError as exc:
        raise SystemExit(
            "Missing TFR dependency. Install the full environment in INSTALL.md. "
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

    model_dir = resolve_repo_path(repo_root, paths["weighted_model_metrics"])
    output_root = resolve_repo_path(repo_root, paths["outputs"]) / "tfr_glm"
    output_root.mkdir(parents=True, exist_ok=True)

    subject_filter = _parse_subjects(params.get("subjects", "all"))
    n_permutations = _parse_int(params.get("tfr_permutations", "5000"), 5000)
    random_state = _parse_int(params.get("random_state", "42"), 42)
    fmin = _parse_int(params.get("bga_fmin_hz", "40"), 40)
    fmax = _parse_int(params.get("bga_fmax_hz", "150"), 150)
    fstep = _parse_int(params.get("tfr_frequency_step_hz", "5"), 5)
    decision_start = _parse_float(params.get("decision_window_start_s", "-0.6"), -0.6)
    decision_end = _parse_float(params.get("decision_window_end_s", "0.1"), 0.1)
    baseline = (
        _parse_float(params.get("tfr_baseline_start_s", "-3.498"), -3.498),
        _parse_float(params.get("tfr_baseline_end_s", "-3.002"), -3.002),
    )

    roi_specs = []
    if args.roi in {"acc", "both"}:
        roi_specs.append((
            "acc",
            resolve_repo_path(repo_root, paths["seeg_roi_main_acc"]),
            resolve_repo_path(repo_root, paths["seeg_roi_baseline_acc"]),
        ))
    if args.roi in {"vmpfc", "both"}:
        roi_specs.append((
            "vmpfc",
            resolve_repo_path(repo_root, paths["seeg_roi_main_vmpfc"]),
            resolve_repo_path(repo_root, paths["seeg_roi_baseline_vmpfc"]),
        ))

    def zscore(values):
        values = np.asarray(values, dtype=float)
        sd = np.nanstd(values)
        if sd < 1e-12:
            return np.full_like(values, np.nan)
        return (values - np.nanmean(values)) / sd

    def orthogonalize(info_z, reward_z):
        valid = np.isfinite(info_z) & np.isfinite(reward_z)
        out = np.full_like(info_z, np.nan, dtype=float)
        if valid.sum() < 3:
            return out
        x = reward_z[valid].reshape(-1, 1)
        model = LinearRegression().fit(x, info_z[valid])
        out[valid] = info_z[valid] - model.predict(x)
        return out

    def build_behavior_df(subject_id):
        path = model_dir / f"{subject_id}_game_metrics.csv"
        if not path.exists():
            return None
        gd = pd.read_csv(path)
        rows = []
        for trial_index, game in enumerate(gd["Game"].values):
            game_data = gd[gd["Game"] == game]
            if game_data.empty:
                continue
            row = game_data.iloc[0]
            chosen = int(row["Chosen_Deck"])
            q_cols = [f"Q_Deck_{idx}" for idx in [1, 2, 3]]
            q_chosen = row[f"Q_Deck_{chosen}"]
            q_unchosen = [row[col] for col in q_cols if col != f"Q_Deck_{chosen}"]
            info_col = (
                f"Weighted_I_transformed_Deck_{chosen}"
                if f"Weighted_I_transformed_Deck_{chosen}" in gd.columns
                else f"I_transformed_Deck_{chosen}"
            )
            info_chosen = row[info_col]
            rows.append({
                "trial_index": trial_index,
                "reward": q_chosen - np.mean(q_unchosen),
                "info": -info_chosen,
            })
        out = pd.DataFrame(rows)
        if out.empty:
            return None
        out["reward_z"] = zscore(out["reward"])
        out["info_z"] = zscore(out["info"])
        out["info_orth"] = orthogonalize(out["info_z"].to_numpy(float), out["reward_z"].to_numpy(float))
        if out[["reward_z", "info_orth"]].isna().any().any():
            return None
        return out

    def apply_common_baseline_zlogratio(tfr_data, times):
        baseline_mask = (times >= baseline[0]) & (times <= baseline[1])
        if not baseline_mask.any():
            raise ValueError(f"Baseline window {baseline} has no matching samples.")
        baseline_data_all_trials = tfr_data[:, :, :, baseline_mask]
        baseline_data_mean_across_trials = np.mean(baseline_data_all_trials, axis=0)
        baseline_mean = np.mean(baseline_data_mean_across_trials, axis=-1, keepdims=True)
        baseline_mean = np.maximum(baseline_mean, 1e-12)
        baseline_logratio = np.log10(np.maximum(baseline_data_mean_across_trials, 1e-12) / baseline_mean)
        baseline_sd = np.std(baseline_logratio, axis=-1, keepdims=True)
        baseline_sd = np.maximum(baseline_sd, 1e-12)
        return np.log10(np.maximum(tfr_data, 1e-12) / baseline_mean[None, :, :, :]) / baseline_sd[None, :, :, :]

    def drop_rejected_epochs(epochs):
        tags = [tag for tag in epochs.event_id if "99" not in tag]
        return epochs[tags] if tags else epochs[:0]

    def make_concatenated_times(n_times, sfreq):
        return baseline[0] + np.arange(n_times, dtype=float) / sfreq

    def run_single_roi(roi_name, roi_dir, baseline_dir):
        output_dir = output_root / roi_name
        output_dir.mkdir(parents=True, exist_ok=True)
        set_files = sorted(roi_dir.glob("*.set"))
        if subject_filter is not None:
            set_files = [path for path in set_files if path.stem in subject_filter]
        if not set_files:
            raise FileNotFoundError(f"No .set files found for ROI {roi_name} in {roi_dir}")

        channel_reward_betas = []
        channel_info_betas = []
        channel_rows = []
        decision_times = None
        freqs = np.arange(fmin, fmax + 1, fstep)

        for set_path in set_files:
            subject_id = set_path.stem
            print(f"\nROI={roi_name}, subject={subject_id}")
            behav = build_behavior_df(subject_id)
            if behav is None:
                print("  Missing model-derived metrics, skip.")
                continue
            baseline_path = baseline_dir / set_path.name
            if not baseline_path.exists():
                print("  Missing baseline epochs, skip.")
                continue
            epochs = drop_rejected_epochs(mne.read_epochs_eeglab(str(set_path), verbose=False))
            baseline_epochs = drop_rejected_epochs(mne.read_epochs_eeglab(str(baseline_path), verbose=False))
            if len(epochs) == 0:
                print("  No valid epochs after removing tag 99, skip.")
                continue
            if len(baseline_epochs) != len(epochs):
                print(f"  Baseline/main epoch count mismatch ({len(baseline_epochs)} vs {len(epochs)}), skip.")
                continue
            data_n = min(len(epochs), len(behav))
            if data_n < 10:
                print(f"  Too few aligned trials ({data_n}), skip.")
                continue
            epochs = epochs[:data_n]
            baseline_epochs = baseline_epochs[:data_n]
            behav = behav.iloc[:data_n].reset_index(drop=True)
            power = mne.time_frequency.tfr_morlet(
                epochs,
                freqs=freqs,
                n_cycles=freqs / 4.0,
                use_fft=True,
                return_itc=False,
                average=False,
                decim=1,
                n_jobs=1,
                verbose=False,
            )
            baseline_power = mne.time_frequency.tfr_morlet(
                baseline_epochs,
                freqs=freqs,
                n_cycles=freqs / 4.0,
                use_fft=True,
                return_itc=False,
                average=False,
                decim=1,
                n_jobs=1,
                verbose=False,
            )
            baseline_crop = baseline_power.copy().crop(tmin=0.002, tmax=0.498)
            tfr_data = np.concatenate((baseline_crop.data, power.data), axis=3)
            times = make_concatenated_times(tfr_data.shape[-1], float(epochs.info["sfreq"]))
            zlog = apply_common_baseline_zlogratio(tfr_data, times)
            high_gamma = np.nanmean(zlog, axis=2)
            high_gamma = gaussian_filter1d(high_gamma, sigma=50, axis=-1)
            decision_mask = (times >= decision_start) & (times <= decision_end)
            if not decision_mask.any():
                raise ValueError(f"Decision window {decision_start} to {decision_end} has no samples.")
            current_decision_times = times[decision_mask]
            if decision_times is None:
                decision_times = current_decision_times
            elif len(decision_times) != len(current_decision_times) or not np.allclose(decision_times, current_decision_times):
                print("  Inconsistent time axis, skip.")
                continue

            y = high_gamma[:, :, decision_mask]
            reward_z = behav["reward_z"].to_numpy(float)
            info_orth = behav["info_orth"].to_numpy(float)
            x = np.column_stack([reward_z, info_orth, np.ones_like(reward_z)])
            xtx_inv = np.linalg.pinv(x.T @ x) @ x.T
            betas = np.einsum("pn,nct->pct", xtx_inv, y)
            reward_betas = betas[0]
            info_betas = betas[1]
            channel_reward_betas.append(reward_betas)
            channel_info_betas.append(info_betas)
            for channel_index, channel_name in enumerate(epochs.ch_names):
                channel_rows.append({
                    "subject": subject_id,
                    "roi": roi_name,
                    "channel": channel_name,
                    "channel_index": channel_index,
                })

        if not channel_reward_betas:
            raise RuntimeError(f"No valid subjects for ROI {roi_name}.")

        channel_reward = np.concatenate(channel_reward_betas, axis=0)
        channel_info = np.concatenate(channel_info_betas, axis=0)
        channel_df = pd.DataFrame(channel_rows)
        channel_df.to_csv(output_dir / "channel_info.csv", index=False)

        def permutation_cluster_1samp_compatible(data, seed):
            kwargs = {
                "out_type": "mask",
                "n_permutations": n_permutations,
                "n_jobs": 1,
                "verbose": False,
            }
            try:
                return mne.stats.permutation_cluster_1samp_test(data, seed=seed, **kwargs)
            except TypeError as exc:
                if "seed" not in str(exc):
                    raise
                return mne.stats.permutation_cluster_1samp_test(data, **kwargs)

        print("Running channel-level cluster permutations.")
        t_obs_reward, clusters_reward, p_reward, h0_reward = permutation_cluster_1samp_compatible(
            channel_reward,
            random_state,
        )
        t_obs_info, clusters_info, p_info, h0_info = permutation_cluster_1samp_compatible(
            channel_info,
            random_state + 1,
        )
        diff_data = channel_reward - channel_info
        t_obs_diff, clusters_diff, p_diff, h0_diff = permutation_cluster_1samp_compatible(
            diff_data,
            random_state + 2,
        )

        def cluster_rows(clusters, p_values, beta_values, signal_name):
            rows = []
            mean_beta = np.nanmean(beta_values, axis=0)
            for idx, cluster in enumerate(clusters):
                if not np.any(cluster):
                    continue
                cluster_indices = np.where(cluster)[0]
                rows.append({
                    "signal": signal_name,
                    "cluster_index": idx,
                    "p_value": p_values[idx],
                    "start_time": decision_times[cluster_indices[0]],
                    "end_time": decision_times[cluster_indices[-1]],
                    "duration": decision_times[cluster_indices[-1]] - decision_times[cluster_indices[0]],
                    "peak_time": decision_times[cluster_indices[np.argmax(np.abs(mean_beta[cluster_indices]))]],
                    "peak_beta": mean_beta[cluster_indices[np.argmax(np.abs(mean_beta[cluster_indices]))]],
                    "significant": bool(p_values[idx] < 0.05),
                })
            return rows

        clusters = []
        clusters.extend(cluster_rows(clusters_reward, p_reward, channel_reward, "reward"))
        clusters.extend(cluster_rows(clusters_info, p_info, channel_info, "information"))
        clusters.extend(cluster_rows(clusters_diff, p_diff, diff_data, "reward_minus_information"))
        pd.DataFrame(clusters).to_csv(output_dir / "significant_clusters.csv", index=False)

        np.save(
            output_dir / "channel_level_permutation_results.npy",
            {
                "decision_times": decision_times,
                "channel_reward": channel_reward,
                "channel_info": channel_info,
                "t_obs_reward": t_obs_reward,
                "t_obs_info": t_obs_info,
                "t_obs_diff": t_obs_diff,
                "p_reward": p_reward,
                "p_info": p_info,
                "p_diff": p_diff,
            },
            allow_pickle=True,
        )

        mean_reward = np.nanmean(channel_reward, axis=0)
        sem_reward = np.nanstd(channel_reward, axis=0, ddof=1) / np.sqrt(channel_reward.shape[0])
        mean_info = np.nanmean(channel_info, axis=0)
        sem_info = np.nanstd(channel_info, axis=0, ddof=1) / np.sqrt(channel_info.shape[0])
        fig, ax = plt.subplots(figsize=(6.0, 3.4))
        ax.plot(decision_times, mean_reward, color="#D36179", label="Relative reward")
        ax.fill_between(decision_times, mean_reward - sem_reward, mean_reward + sem_reward, color="#D36179", alpha=0.18)
        ax.plot(decision_times, mean_info, color="#7BB5B7", label="Information status")
        ax.fill_between(decision_times, mean_info - sem_info, mean_info + sem_info, color="#7BB5B7", alpha=0.18)
        ax.axhline(0, color="0.5", linestyle="--", linewidth=0.8)
        ax.axvline(0, color="0.5", linestyle=":", linewidth=0.8)
        ax.set_xlabel("Time relative to choice (s)")
        ax.set_ylabel("GLM beta")
        ax.set_title(f"{roi_name.upper()} BGA encoding GLM ({fmin}-{fmax} Hz)")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(output_dir / "channel_level_reward_info_comparison_highgamma.png", dpi=300)
        plt.close(fig)

        report = [
            f"ROI: {roi_name}",
            f"Frequency range: {fmin}-{fmax} Hz",
            f"Decision window: {decision_start} to {decision_end} s",
            f"Number of channels: {channel_reward.shape[0]}",
            f"Number of permutations: {n_permutations}",
            "",
            "Significant clusters are saved in significant_clusters.csv.",
        ]
        (output_dir / "statistical_report.txt").write_text("\n".join(report), encoding="utf-8")

    for roi_name, roi_dir, baseline_dir in roi_specs:
        run_single_roi(roi_name, roi_dir, baseline_dir)

    print("TFR GLM analysis complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
