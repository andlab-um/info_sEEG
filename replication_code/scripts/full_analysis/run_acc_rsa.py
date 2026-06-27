#!/usr/bin/env python3
"""Run the final ACC RSA analysis from the PCA/RSA notebook."""

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


def _parse_int_list(value: str, default: list[int]) -> list[int]:
    try:
        return [int(part.strip()) for part in str(value).split(",") if part.strip()]
    except Exception:
        return default


def _parse_subjects(value: str) -> list[str] | None:
    text = str(value).strip()
    if not text or text.lower() in {"all", "*"}:
        return None
    return [part.strip() for part in text.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run final ACC RSA early/late analysis.")
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
        from scipy.spatial.distance import pdist
        from scipy.stats import spearmanr
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
    output_dir = resolve_repo_path(repo_root, paths["outputs"]) / "acc_rsa"
    output_dir.mkdir(parents=True, exist_ok=True)

    state_cols = ["acc_PC1", "acc_PC2", "acc_PC3"]
    focus_var_candidates = [
        ("bestdeck_entropy_before", ["bestdeck_entropy_before", "uncertainty_entropy_before"]),
        (
            "uncertainty_imbalance_before",
            ["uncertainty_imbalance_before", "uncertainty_range_before", "uncertainty_gini_before"],
        ),
    ]
    n_perm = _parse_int(params.get("rsa_permutations", "5000"), 5000)
    n_boot = _parse_int(params.get("rsa_bootstrap", "2000"), 2000)
    random_state_base = _parse_int(params.get("random_state", "42"), 42)
    min_trials_per_subject = _parse_int(params.get("min_trials_per_subject", "6"), 6)
    min_subjects_per_block = _parse_int(params.get("min_subjects_per_rsa_block", "5"), 5)
    early_reveals = _parse_int_list(params.get("early_reveals", "1, 2, 3"), [1, 2, 3])
    late_reveals = _parse_int_list(params.get("late_reveals", "4, 5, 6"), [4, 5, 6])
    subject_filter = _parse_subjects(params.get("subjects", "all"))
    if subject_filter is not None:
        min_subjects_per_block = min(min_subjects_per_block, len(subject_filter))
    block_seeds = {"early13": random_state_base + 101, "late46": random_state_base + 202}

    print(f"Input table: {input_csv.relative_to(repo_root)}")
    print(f"Output directory: {output_dir.relative_to(repo_root)}")
    print(f"N_PERM={n_perm}, N_BOOT={n_boot}")

    def is_binary_series(series):
        series = series.dropna()
        vals = pd.unique(series)
        if len(vals) <= 2:
            return True
        try:
            vals_num = pd.to_numeric(vals)
            return set(vals_num).issubset({0, 1})
        except Exception:
            return False

    def is_categorical_series(series):
        series = pd.Series(series).dropna()
        if len(series) == 0:
            return True
        if series.dtype == object or str(series.dtype).startswith("category"):
            return True
        return is_binary_series(series)

    def zscore_safe(values):
        values = np.asarray(values, dtype=float)
        sd = np.nanstd(values)
        if sd < 1e-12:
            return np.zeros_like(values, dtype=float)
        return (values - np.nanmean(values)) / sd

    def neural_rdm_vector(x):
        return pdist(np.asarray(x, dtype=float), metric="euclidean")

    def model_rdm_vector(values, categorical=False, normalize_continuous=True):
        series = pd.Series(values)
        if categorical:
            x = series.astype(str).to_numpy()
            return np.array([float(x[i] != x[j]) for i in range(len(x) - 1) for j in range(i + 1, len(x))], dtype=float)
        x = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
        if normalize_continuous:
            x = zscore_safe(x)
        return np.array([abs(x[i] - x[j]) for i in range(len(x) - 1) for j in range(i + 1, len(x))], dtype=float)

    def compute_rsa_single_subject(df_sub, var, categorical):
        df_sub = df_sub.dropna(subset=state_cols + [var]).copy().reset_index(drop=True)
        if len(df_sub) < min_trials_per_subject:
            return np.nan, 0
        neural_vec = neural_rdm_vector(df_sub[state_cols].to_numpy(dtype=float))
        model_vec = model_rdm_vector(df_sub[var], categorical=categorical)
        valid = np.isfinite(neural_vec) & np.isfinite(model_vec)
        neural_vec = neural_vec[valid]
        model_vec = model_vec[valid]
        if len(neural_vec) < 5 or np.nanstd(neural_vec) < 1e-12 or np.nanstd(model_vec) < 1e-12:
            return np.nan, len(neural_vec)
        rho, _ = spearmanr(neural_vec, model_vec)
        return float(rho), int(len(neural_vec))

    def subjectwise_observed_rhos(df_block, var, categorical):
        rows = []
        for subj, group in df_block.groupby("Subject"):
            rho, n_pairs = compute_rsa_single_subject(group.reset_index(drop=True), var, categorical)
            rows.append({"Subject": subj, "rho": rho, "n_pairs": n_pairs, "n_trials": len(group)})
        out = pd.DataFrame(rows)
        return out[np.isfinite(out["rho"])].copy()

    def permute_var_within_subject(df_in, var, rng):
        out = df_in.copy().reset_index(drop=True)
        vals = out[var].to_numpy(copy=True)
        for _, idx in out.groupby("Subject").groups.items():
            idx = np.array(list(idx), dtype=int)
            vals[idx] = rng.permutation(vals[idx])
        out[var] = vals
        return out

    def permutation_test_subjectwise_mean(df_block, var, categorical, seed):
        rng = np.random.default_rng(seed)
        obs_df = subjectwise_observed_rhos(df_block, var, categorical)
        if len(obs_df) < min_subjects_per_block:
            return np.nan, np.nan, 0, 0, obs_df
        obs_mean = obs_df["rho"].mean()
        perm_means = []
        for _ in range(n_perm):
            perm_df = permute_var_within_subject(df_block, var, rng)
            tmp = subjectwise_observed_rhos(perm_df, var, categorical)
            if len(tmp) >= min_subjects_per_block:
                perm_means.append(tmp["rho"].mean())
        perm_means = np.asarray(perm_means, dtype=float)
        perm_means = perm_means[np.isfinite(perm_means)]
        if len(perm_means) == 0:
            return float(obs_mean), np.nan, int(obs_df["Subject"].nunique()), int(obs_df["n_pairs"].sum()), obs_df
        p_one_sided_positive = (np.sum(perm_means >= obs_mean) + 1) / (len(perm_means) + 1)
        return float(obs_mean), float(p_one_sided_positive), int(obs_df["Subject"].nunique()), int(obs_df["n_pairs"].sum()), obs_df

    def bootstrap_ci_subjectwise_mean(df_block, var, categorical, seed):
        rng = np.random.default_rng(seed)
        obs_df = subjectwise_observed_rhos(df_block, var, categorical)
        if len(obs_df) < min_subjects_per_block:
            return np.nan, np.nan
        subjects = obs_df["Subject"].unique()
        subj_to_rho = dict(zip(obs_df["Subject"], obs_df["rho"]))
        boots = []
        for _ in range(n_boot):
            samp_subs = rng.choice(subjects, size=len(subjects), replace=True)
            samp_rhos = [subj_to_rho[subj] for subj in samp_subs if np.isfinite(subj_to_rho.get(subj, np.nan))]
            if len(samp_rhos) >= min_subjects_per_block:
                boots.append(np.mean(samp_rhos))
        boots = np.asarray(boots, dtype=float)
        boots = boots[np.isfinite(boots)]
        if len(boots) == 0:
            return np.nan, np.nan
        return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))

    focus_vars: list[str] = []
    focus_var_source: dict[str, str] = {}

    def resolve_focus_vars(columns):
        resolved = []
        source = {}
        for source_var, candidates in focus_var_candidates:
            match = next((candidate for candidate in candidates if candidate in columns), None)
            if match is None:
                print(f"Skip {source_var}: none of the expected columns were found.")
                continue
            resolved.append(match)
            source[match] = source_var
            if match != source_var:
                print(f"Using {match} as available-table fallback for {source_var}.")
        return resolved, source

    def run_rsa_block(df_block, block_name):
        rows = []
        subj_detail_rows = []
        df_block = df_block.dropna(subset=state_cols + ["Subject", "Game", "reveal_idx"]).copy().reset_index(drop=True)
        block_seed = block_seeds.get(block_name, random_state_base + 999)
        print(f"Running RSA block {block_name}: rows={len(df_block)}, subjects={df_block['Subject'].nunique()}")
        for var_index, var in enumerate(focus_vars):
            if var not in df_block.columns:
                print(f"Skip {var}: column not found.")
                continue
            sub = df_block.dropna(subset=[var] + state_cols).copy().reset_index(drop=True)
            if len(sub) < 10:
                print(f"Skip {var}: too few valid rows ({len(sub)}).")
                continue
            categorical = is_categorical_series(sub[var])
            seed = block_seed + var_index * 1000
            rho, p_one_pos, n_subjects, n_pairs, obs_df = permutation_test_subjectwise_mean(sub, var, categorical, seed)
            ci_lo, ci_hi = bootstrap_ci_subjectwise_mean(sub, var, categorical, seed + 123)
            rows.append({
                "block": block_name,
                "model_var": var,
                "source_model_var": focus_var_source.get(var, var),
                "rsa_rho": rho,
                "perm_p_one_sided_positive": p_one_pos,
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
                "n_trials": len(sub),
                "n_subjects": n_subjects,
                "n_pairs_used": n_pairs,
                "var_type": "categorical" if categorical else "continuous",
                "n_perm": n_perm,
                "n_boot": n_boot,
            })
            if len(obs_df) > 0:
                tmp = obs_df.copy()
                tmp["block"] = block_name
                tmp["model_var"] = var
                tmp["source_model_var"] = focus_var_source.get(var, var)
                subj_detail_rows.append(tmp)
        out_df = pd.DataFrame(rows)
        out_df.to_csv(output_dir / f"rsa_{block_name}.csv", index=False)
        if subj_detail_rows:
            pd.concat(subj_detail_rows, ignore_index=True).to_csv(
                output_dir / f"rsa_{block_name}_subject_details.csv",
                index=False,
            )
        return out_df

    df = pd.read_csv(input_csv)
    if subject_filter is not None:
        if "Subject" not in df.columns:
            raise ValueError("Missing required column for subject filtering: Subject")
        df = df[df["Subject"].astype(str).isin(subject_filter)].copy()
        if len(df) == 0:
            raise ValueError(f"No rows remain after subject filter: {', '.join(subject_filter)}")
        print(f"After subject filter ({', '.join(subject_filter)}):", df.shape)
    if "has_acc_data" in df.columns:
        df = df[df["has_acc_data"] == 1].copy()
    df = df.dropna(subset=state_cols + ["Subject", "Game", "reveal_idx"]).copy().reset_index(drop=True)
    print("Filtered data shape:", df.shape)
    focus_vars, focus_var_source = resolve_focus_vars(df.columns)
    if not focus_vars:
        raise ValueError("No RSA model variables are available in the input table.")

    rsa_early = run_rsa_block(df[df["reveal_idx"].isin(early_reveals)].copy(), "early13")
    rsa_late = run_rsa_block(df[df["reveal_idx"].isin(late_reveals)].copy(), "late46")
    if len(rsa_early) > 0:
        rsa_early = rsa_early.copy()
        rsa_early["stage"] = "Early (rev 1-3)"
    if len(rsa_late) > 0:
        rsa_late = rsa_late.copy()
        rsa_late["stage"] = "Late (rev 4-6)"
    compare_df = pd.concat([rsa_early, rsa_late], ignore_index=True)
    compare_df.to_csv(output_dir / "rsa_early_vs_late_comparison.csv", index=False)
    compare_df.round(6).to_csv(output_dir / "rsa_early_vs_late_comparison_rounded.csv", index=False)

    print("ACC RSA analysis complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
