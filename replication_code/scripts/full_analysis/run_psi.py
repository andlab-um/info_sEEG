#!/usr/bin/env python3
"""Run sliding-window beta PSI analysis from the original PSI notebook."""

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
    parser = argparse.ArgumentParser(description="Run beta PSI analysis.")
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
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        import mne
        from numpy.fft import rfft, rfftfreq
        from scipy.signal import detrend
        from scipy.signal.windows import dpss as dpss_windows
    except ImportError as exc:
        raise SystemExit(
            "Missing PSI dependency. Install the full environment in INSTALL.md. "
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
    output_dir = resolve_repo_path(repo_root, paths["outputs"]) / "psi"
    output_dir.mkdir(parents=True, exist_ok=True)

    subject_filter = _parse_subjects(params.get("subjects", "all"))
    n_perm = _parse_int(params.get("psi_permutations", "5000"), 5000)
    random_state = _parse_int(params.get("random_state", "42"), 42)
    analysis_tmin = _parse_float(params.get("psi_analysis_tmin_s", "-1.4"), -1.4)
    analysis_tmax = _parse_float(params.get("psi_analysis_tmax_s", "0.2"), 0.2)
    decision_tmin = _parse_float(params.get("psi_decision_tmin_s", "-1.0"), -1.0)
    decision_tmax = _parse_float(params.get("psi_decision_tmax_s", "0.1"), 0.1)
    win_len_sec = _parse_float(params.get("psi_window_seconds", "0.30"), 0.30)
    step_sec = _parse_float(params.get("psi_step_seconds", "0.02"), 0.02)
    fmin = _parse_float(params.get("psi_beta_fmin_hz", "13"), 13.0)
    fmax = _parse_float(params.get("psi_beta_fmax_hz", "30"), 30.0)
    bandwidth = _parse_float(params.get("psi_multitaper_bandwidth_hz", "4"), 4.0)
    alpha = 0.05

    def zscore_safe(values):
        values = np.asarray(values, dtype=float)
        sd = np.nanstd(values)
        if sd < 1e-12:
            return np.full_like(values, np.nan)
        return (values - np.nanmean(values)) / sd

    def drop_rejected_epochs(epochs):
        tags = [tag for tag in epochs.event_id if "99" not in tag]
        return epochs[tags] if tags else epochs[:0]

    def align_epochs(ep1, ep2):
        ep1 = drop_rejected_epochs(ep1)
        ep2 = drop_rejected_epochs(ep2)
        n = min(len(ep1), len(ep2))
        return ep1[:n], ep2[:n]

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
            info_col = (
                f"I_transformed_Deck_{chosen}"
                if f"I_transformed_Deck_{chosen}" in gd.columns
                else f"Weighted_I_transformed_Deck_{chosen}"
            )
            reward = row[f"Q_Deck_{chosen}"] - row[q_cols].mean()
            info = row[info_col]
            rows.append({"trial_index": trial_index, "reward": reward, "info": info})
        out = pd.DataFrame(rows)
        out["reward_z"] = zscore_safe(out["reward"])
        out["info_z"] = zscore_safe(out["info"])
        return out

    def build_windows(times):
        dt = times[1] - times[0]
        sfreq = 1.0 / dt
        win_samp = int(round(win_len_sec * sfreq))
        step_samp = max(1, int(round(step_sec * sfreq)))
        half = win_samp // 2
        centres, starts, ends = [], [], []
        for centre_idx in range(half, len(times) - half, step_samp):
            centre_time = times[centre_idx]
            if centre_time < decision_tmin or centre_time > decision_tmax:
                continue
            start_idx = centre_idx - half
            end_idx = centre_idx - half + win_samp
            if end_idx > len(times):
                break
            centres.append(centre_time)
            starts.append(start_idx)
            ends.append(end_idx)
        return np.asarray(centres), np.asarray(starts, dtype=int), np.asarray(ends, dtype=int)

    def compute_psi_one_pair_one_window(x_seg, y_seg, sfreq):
        n = x_seg.shape[-1]
        if n < 4 or np.all(x_seg == 0) or np.all(y_seg == 0):
            return np.nan
        x_seg = detrend(x_seg, type="linear")
        y_seg = detrend(y_seg, type="linear")
        nw = max(1.5, bandwidth * n / (2.0 * sfreq))
        kmax = max(1, int(2 * nw) - 1)
        try:
            tapers = dpss_windows(n, NW=nw, Kmax=kmax, sym=False)
            if tapers.ndim == 1:
                tapers = tapers[np.newaxis, :]
        except Exception:
            tapers = np.array([np.ones(n) / np.sqrt(n)])
        freqs = rfftfreq(n, 1.0 / sfreq)
        fmask = (freqs >= fmin) & (freqs <= fmax)
        freq_indices = np.where(fmask)[0]
        if len(freq_indices) < 2:
            return np.nan
        psi_acc = 0.0
        for taper in tapers:
            xf = rfft(x_seg * taper)
            yf = rfft(y_seg * taper)
            cs = np.conj(xf) * yf
            psi_acc += np.imag(np.sum(cs[freq_indices[:-1]] * np.conj(cs[freq_indices[1:]])))
        return float(psi_acc / len(tapers))

    def compute_sliding_window_psi(data_seed, data_target, seed_idx, target_idx, sfreq, starts, ends):
        n_trials = data_seed.shape[0]
        n_pairs = len(seed_idx)
        n_windows = len(starts)
        out = np.full((n_trials, n_pairs, n_windows), np.nan, dtype=float)
        for trial in range(n_trials):
            for pair in range(n_pairs):
                sx = data_seed[trial, seed_idx[pair]]
                ty = data_target[trial, target_idx[pair]]
                for window_idx, (start, end) in enumerate(zip(starts, ends)):
                    out[trial, pair, window_idx] = compute_psi_one_pair_one_window(sx[start:end], ty[start:end], sfreq)
        return out

    def timepoint_regression_single(y_mat, x_z):
        n_windows = y_mat.shape[1]
        beta = np.full(n_windows, np.nan)
        for window_idx in range(n_windows):
            y = y_mat[:, window_idx]
            valid = np.isfinite(y) & np.isfinite(x_z)
            if valid.sum() < 5 or np.nanstd(x_z[valid]) < 1e-12:
                continue
            x = x_z[valid]
            yy = y[valid]
            beta[window_idx] = np.dot(x, yy) / (np.dot(x, x) + 1e-12)
        return beta

    def signflip_stat(subject_groups):
        per_subject = [np.nanmean(values) for values in subject_groups.values() if len(values) > 0]
        return float(np.nanmean(per_subject)) if per_subject else np.nan

    def signflip_test_one_time(df_t, subject_col, value_col, seed):
        rng = np.random.default_rng(seed)
        groups = {
            subject: group[value_col].to_numpy(float)
            for subject, group in df_t.groupby(subject_col)
            if len(group) > 0
        }
        obs = signflip_stat(groups)
        null = np.empty(n_perm)
        subjects = list(groups.keys())
        for idx in range(n_perm):
            perm = {subject: groups[subject] * rng.choice([-1, 1]) for subject in subjects}
            null[idx] = signflip_stat(perm)
        p_value = (np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1)
        return obs, p_value, len(groups)

    def run_time_resolved_signflip(df_pairs, value_col, seed):
        rows = []
        for idx, time_value in enumerate(np.sort(df_pairs["time"].dropna().unique())):
            obs, p_value, n_subjects = signflip_test_one_time(
                df_pairs[df_pairs["time"] == time_value],
                "subject",
                value_col,
                seed + idx,
            )
            rows.append({"time": float(time_value), "observed_stat": obs, "p_value": p_value, "n_subjects": n_subjects})
        return pd.DataFrame(rows)

    def add_fdr_bh(df, p_col="p_value", out_col="p_fdr_bh"):
        out = df.copy()
        pvals = out[p_col].to_numpy(float)
        valid = np.isfinite(pvals)
        qvals = np.full_like(pvals, np.nan)
        if valid.sum() > 0:
            p = pvals[valid]
            order = np.argsort(p)
            q = p[order] * len(p) / np.arange(1, len(p) + 1)
            q = np.minimum.accumulate(q[::-1])[::-1]
            q = np.clip(q, 0, 1)
            q_out = np.empty_like(q)
            q_out[order] = q
            qvals[valid] = q_out
        out[out_col] = qvals
        return out

    subjects = sorted(path.stem for path in acc_dir.glob("*.set") if (vmpfc_dir / path.name).exists())
    if subject_filter is not None:
        subjects = [subject for subject in subjects if subject in subject_filter]
    if not subjects:
        raise FileNotFoundError("No subjects with both ACC and vmPFC .set files were found.")

    pairwise_rows = []
    subject_betas = {"reward": [], "info": []}
    kept_subjects = []
    times_final = None

    for subject in subjects:
        print(f"Subject {subject}")
        try:
            ep_acc = mne.read_epochs_eeglab(str(acc_dir / f"{subject}.set"), verbose=False)
            ep_vmpfc = mne.read_epochs_eeglab(str(vmpfc_dir / f"{subject}.set"), verbose=False)
            ep_acc, ep_vmpfc = align_epochs(ep_acc, ep_vmpfc)
            ep_acc = ep_acc.crop(analysis_tmin, analysis_tmax)
            ep_vmpfc = ep_vmpfc.crop(analysis_tmin, analysis_tmax)
            sfreq = float(ep_acc.info["sfreq"])
            times = ep_acc.times
            data_acc = ep_acc.get_data(copy=False)
            data_vmpfc = ep_vmpfc.get_data(copy=False)
            n_trials = min(data_acc.shape[0], data_vmpfc.shape[0])
            data_acc = data_acc[:n_trials]
            data_vmpfc = data_vmpfc[:n_trials]
            n_acc = data_acc.shape[1]
            n_vmpfc = data_vmpfc.shape[1]
            seed_idx = np.repeat(np.arange(n_acc), n_vmpfc)
            target_idx = np.tile(np.arange(n_vmpfc), n_acc)
            centres, starts, ends = build_windows(times)
            if len(centres) == 0:
                print("  No valid windows, skip.")
                continue
            psi_arr = compute_sliding_window_psi(data_acc, data_vmpfc, seed_idx, target_idx, sfreq, starts, ends)
            behav = build_behavior_df(subject)
            if behav is None:
                print("  Missing behavior metrics, skip.")
                continue
            behav = behav[behav["trial_index"] < n_trials].reset_index(drop=True)
            n_trials = min(len(behav), psi_arr.shape[0])
            if n_trials < 10:
                print("  Too few trials, skip.")
                continue
            behav = behav.iloc[:n_trials]
            psi_arr = psi_arr[:n_trials]
            reward_z = behav["reward_z"].to_numpy(float)
            info_z = behav["info_z"].to_numpy(float)
            pair_beta_reward = []
            pair_beta_info = []
            for pair_idx in range(len(seed_idx)):
                y_mat = psi_arr[:, pair_idx, :]
                beta_reward = timepoint_regression_single(y_mat, reward_z)
                beta_info = timepoint_regression_single(y_mat, info_z)
                if np.all(np.isnan(beta_reward)) and np.all(np.isnan(beta_info)):
                    continue
                pair_beta_reward.append(beta_reward)
                pair_beta_info.append(beta_info)
                ch_acc = ep_acc.ch_names[seed_idx[pair_idx]]
                ch_vmpfc = ep_vmpfc.ch_names[target_idx[pair_idx]]
                for window_idx, time_value in enumerate(centres):
                    pairwise_rows.append({
                        "subject": subject,
                        "seed_ch": ch_acc,
                        "target_ch": ch_vmpfc,
                        "time": float(time_value),
                        "beta_reward": float(beta_reward[window_idx]),
                        "beta_info": float(beta_info[window_idx]),
                    })
            if not pair_beta_reward:
                continue
            pair_beta_reward = np.asarray(pair_beta_reward, dtype=float)
            pair_beta_info = np.asarray(pair_beta_info, dtype=float)
            if times_final is None:
                times_final = centres.copy()
            elif len(times_final) != len(centres) or not np.allclose(times_final, centres):
                pairwise_rows = [row for row in pairwise_rows if row["subject"] != subject]
                continue
            subject_betas["reward"].append(np.nanmean(pair_beta_reward, axis=0))
            subject_betas["info"].append(np.nanmean(pair_beta_info, axis=0))
            kept_subjects.append(subject)
            print(f"  Kept pairs: {len(pair_beta_reward)}")
        except Exception as exc:
            print(f"  Failed: {exc}")

    if not subject_betas["reward"]:
        raise RuntimeError("No subjects completed successfully.")

    subj_beta_reward = np.asarray(subject_betas["reward"], dtype=float)
    subj_beta_info = np.asarray(subject_betas["info"], dtype=float)
    np.save(output_dir / "times_centres.npy", times_final)
    np.save(output_dir / "subj_beta_reward.npy", subj_beta_reward)
    np.save(output_dir / "subj_beta_info.npy", subj_beta_info)
    pd.DataFrame({"subject": kept_subjects}).to_csv(output_dir / "kept_subjects.csv", index=False)
    pairwise_df = pd.DataFrame(pairwise_rows)
    pairwise_df.to_csv(output_dir / "pairwise_beta_long.csv", index=False)

    reward_pairs = pairwise_df[["subject", "time", "beta_reward"]].rename(columns={"beta_reward": "value"}).dropna()
    info_pairs = pairwise_df[["subject", "time", "beta_info"]].rename(columns={"beta_info": "value"}).dropna()
    reward_perm = add_fdr_bh(run_time_resolved_signflip(reward_pairs, "value", random_state))
    info_perm = add_fdr_bh(run_time_resolved_signflip(info_pairs, "value", random_state + 1000))
    reward_perm["sig_raw"] = reward_perm["p_value"] < alpha
    reward_perm["sig_fdr"] = reward_perm["p_fdr_bh"] < alpha
    info_perm["sig_raw"] = info_perm["p_value"] < alpha
    info_perm["sig_fdr"] = info_perm["p_fdr_bh"] < alpha
    reward_perm.to_csv(output_dir / "reward_perm_results.csv", index=False)
    info_perm.to_csv(output_dir / "info_perm_results.csv", index=False)

    mpl.rcParams.update({"font.size": 8, "pdf.fonttype": 42, "ps.fonttype": 42})

    def find_sig_segments(x, mask):
        x = np.asarray(x, dtype=float)
        mask = np.asarray(mask, dtype=bool)
        if not mask.any():
            return []
        idx = np.where(mask)[0]
        segments = []
        start = previous = idx[0]
        for value in idx[1:]:
            if value == previous + 1:
                previous = value
            else:
                segments.append((x[start], x[previous]))
                start = previous = value
        segments.append((x[start], x[previous]))
        return segments

    def plot_psi_timecourse(subj_beta, perm_df, effect_name, line_color):
        win = (times_final >= -0.6) & (times_final <= 0.1)
        times = times_final[win]
        mean = np.nanmean(subj_beta[:, win], axis=0)
        sem = np.nanstd(subj_beta[:, win], axis=0, ddof=1) / np.sqrt(subj_beta.shape[0])
        perm = perm_df.sort_values("time").reset_index(drop=True)
        mask = (perm["time"].to_numpy(float) >= -0.6) & (perm["time"].to_numpy(float) <= 0.1)
        pt = perm.loc[mask, "time"].to_numpy(float)
        stat = perm.loc[mask, "observed_stat"].to_numpy(float)
        raw_sig = perm.loc[mask, "sig_raw"].to_numpy(bool)
        fdr_sig = perm.loc[mask, "sig_fdr"].to_numpy(bool)
        fig, ax = plt.subplots(figsize=(3.5, 2.6))
        ax.fill_between(times, mean - sem, mean + sem, color=line_color, alpha=0.20, linewidth=0)
        ax.plot(pt, stat, color=line_color, linewidth=1.8, label=f"{effect_name} beta")
        ax.axhline(0, color="0.55", linewidth=0.8, linestyle="--")
        ax.axvline(0, color="0.55", linewidth=0.8, linestyle=":")
        y0 = np.nanmin([np.nanmin(mean - sem), np.nanmin(stat)])
        y1 = np.nanmax([np.nanmax(mean + sem), np.nanmax(stat)])
        yr = y1 - y0 if y1 > y0 else 1.0
        for x0, x1 in find_sig_segments(pt, raw_sig):
            ax.plot([x0, x1], [y0 - 0.10 * yr, y0 - 0.10 * yr], color="#BF3363", linewidth=2.0, clip_on=False)
        for x0, x1 in find_sig_segments(pt, fdr_sig):
            ax.plot([x0, x1], [y0 - 0.18 * yr, y0 - 0.18 * yr], color="#BF3363", linewidth=3.5, clip_on=False)
        ax.set_xlabel("Time relative to choice onset (s)")
        ax.set_ylabel("PSI beta (ACC to vmPFC, 13-30 Hz)")
        ax.legend(frameon=False)
        fig.tight_layout()
        for ext in ["pdf", "svg", "png"]:
            fig.savefig(output_dir / f"psi_{effect_name.lower()}_beta_timecourse.{ext}", dpi=600 if ext == "png" else None, transparent=True)
        plt.close(fig)

    plot_psi_timecourse(subj_beta_reward, reward_perm, "Reward", "#D36179")
    plot_psi_timecourse(subj_beta_info, info_perm, "Info", "#7BB5B7")

    print("PSI analysis complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
