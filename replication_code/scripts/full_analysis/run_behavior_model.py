#!/usr/bin/env python3
"""Run the single-subject hierarchical gkRL model from the original HBM notebook."""

from __future__ import annotations

import argparse
import gc
import logging
import os
import sys
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


def _parse_subjects(value: str) -> list[str] | None:
    text = str(value).strip()
    if not text or text.lower() in {"all", "*"}:
        return None
    return [part.strip() for part in text.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit the behavioral hierarchical gkRL model.")
    parser.add_argument(
        "--config",
        default="replication_code/config/full_analysis_config.yaml",
        help="Relative path to the full-analysis config file.",
    )
    return parser.parse_args()


def main() -> int:
    os.environ.setdefault("PYTENSOR_FLAGS", "floatX=float32,optimizer=fast_compile")
    os.environ.setdefault("MKL_NUM_THREADS", "4")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")
    os.environ.setdefault("OMP_NUM_THREADS", "4")

    try:
        import numpy as np
        import pandas as pd
        import pymc as pm
        import pytensor.tensor as pt
        import arviz as az
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "Missing behavioral-model dependency. Install the full environment in INSTALL.md. "
            f"Original import error: {exc}"
        ) from exc

    try:
        import jax
        jax.config.update("jax_enable_x64", False)
        jax.config.update("jax_platform_name", "cpu")
        import numpyro  # noqa: F401
        jax_available = True
    except ImportError:
        jax_available = False

    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path
    config = load_demo_config(config_path)
    repo_root = repo_root_from_config(config_path)
    paths = require_section(config, "paths")
    params = require_section(config, "analysis_parameters")

    data_dir = resolve_repo_path(repo_root, paths["behavioral_logs"])
    output_dir = resolve_repo_path(repo_root, paths["behavioral_model_outputs"])
    weighted_dir = output_dir / "weighted"
    output_dir.mkdir(parents=True, exist_ok=True)
    weighted_dir.mkdir(parents=True, exist_ok=True)

    draws = _parse_int(params.get("behavioral_draws", "3000"), 3000)
    tune = _parse_int(params.get("behavioral_tune", "1500"), 1500)
    chains = _parse_int(params.get("behavioral_chains", "4"), 4)
    cores = _parse_int(params.get("behavioral_cores", "4"), 4)
    random_state = _parse_int(params.get("random_state", "42"), 42)
    requested_subjects = _parse_subjects(params.get("subjects", "all"))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(output_dir / "behavior_model.log"),
        ],
    )
    logger = logging.getLogger("behavior_model")
    logger.info("Input directory: %s", data_dir.relative_to(repo_root))
    logger.info("Output directory: %s", output_dir.relative_to(repo_root))
    logger.info("PyMC version: %s", pm.__version__)
    logger.info("JAX/NumPyro available: %s", jax_available)

    def preprocess_data(df_raw):
        df_filtered = df_raw[(df_raw["total_reward"] >= 0) | (df_raw["trial_general"] > 0)].copy()
        df_filtered.reset_index(drop=True, inplace=True)
        num_trials_per_game = 7
        num_games_estimated = len(df_filtered) // num_trials_per_game
        if len(df_filtered) == 0:
            return None
        rows = []
        for game_num_est in range(num_games_estimated):
            cumulative_counts = [0, 0, 0]
            start_idx = game_num_est * num_trials_per_game
            end_idx = start_idx + num_trials_per_game
            if end_idx > len(df_filtered):
                break
            for trial_in_game in range(num_trials_per_game):
                index = start_idx + trial_in_game
                chosen_deck_val = np.nan
                rewards = [np.nan, np.nan, np.nan]
                if trial_in_game < 6:
                    if "key_resp.keys" in df_filtered.columns:
                        chosen_deck_val = df_filtered.loc[index, "key_resp.keys"]
                    for deck in range(3):
                        col = f"generated_number_{deck + 1}"
                        if col in df_filtered.columns:
                            rewards[deck] = df_filtered.loc[index, col]
                else:
                    if "key_resp_2.keys" in df_filtered.columns:
                        chosen_deck_val = df_filtered.loc[index, "key_resp_2.keys"]
                    for deck in range(3):
                        col = f"true_reward_{deck + 1}"
                        if col in df_filtered.columns:
                            rewards[deck] = df_filtered.loc[index, col]
                try:
                    if pd.notna(chosen_deck_val):
                        choice_int = int(float(chosen_deck_val))
                        if 1 <= choice_int <= 3:
                            cumulative_counts[choice_int - 1] += 1
                except (ValueError, TypeError):
                    pass
                condition_val = np.nan
                if "trial_general" in df_filtered.columns and pd.notna(df_filtered.loc[index, "trial_general"]):
                    condition_val = 1 if df_filtered.loc[index, "trial_general"] < 7 else 2
                rows.append({
                    "Game": game_num_est + 1,
                    "Trial": trial_in_game + 1,
                    "Condition": condition_val,
                    "Chosen Deck": chosen_deck_val,
                    "Reward Card 1": rewards[0],
                    "Reward Card 2": rewards[1],
                    "Reward Card 3": rewards[2],
                    "Revealed Count Card 1": cumulative_counts[0],
                    "Revealed Count Card 2": cumulative_counts[1],
                    "Revealed Count Card 3": cumulative_counts[2],
                })
        return pd.DataFrame(rows) if rows else None

    def extract_trial_data_for_bhm(df_processed, num_decks=3, num_force_trials=6, alpha_init=0.1):
        trial_data_list = []
        for game_idx in range(df_processed["Game"].nunique()):
            game_data = df_processed[df_processed["Game"] == game_idx + 1].sort_values("Trial").reset_index(drop=True)
            if len(game_data) < num_force_trials + 1:
                continue
            q_values = np.zeros(num_decks)
            forced_choices = []
            forced_rewards = []
            for trial in range(num_force_trials):
                row = game_data.iloc[trial]
                try:
                    deck_val = int(row["Chosen Deck"])
                    if not (1 <= deck_val <= num_decks):
                        raise ValueError
                    chosen_deck = deck_val - 1
                except (ValueError, TypeError):
                    chosen_deck = int(np.argmax(q_values))
                forced_choices.append(chosen_deck)
                rewards = [row[f"Reward Card {idx + 1}"] for idx in range(num_decks)]
                forced_rewards.append(rewards)
                reward = rewards[chosen_deck]
                if pd.notna(reward):
                    delta = reward - q_values[chosen_deck]
                    q_values[chosen_deck] += alpha_init * delta
            free_trial_row = game_data.iloc[num_force_trials]
            try:
                free_choice = int(free_trial_row["Chosen Deck"]) - 1
                if not (0 <= free_choice < num_decks):
                    raise ValueError
            except (ValueError, TypeError):
                continue
            trial_data_list.append({
                "game_idx": game_idx,
                "forced_choices": np.array(forced_choices, dtype=np.int32),
                "forced_rewards": np.array(forced_rewards, dtype=np.float32),
                "free_choice": free_choice,
            })
        return trial_data_list

    def build_single_subject_model(trial_data_list, num_decks=3, num_force_trials=6):
        n_games = len(trial_data_list)
        if n_games == 0:
            return None
        all_forced_choices = pt.as_tensor_variable(
            np.array([item["forced_choices"] for item in trial_data_list], dtype=np.int32)
        )
        all_forced_rewards = pt.as_tensor_variable(
            np.array([item["forced_rewards"] for item in trial_data_list], dtype=np.float32)
        )
        all_free_choices = np.array([item["free_choice"] for item in trial_data_list], dtype=np.int32)

        with pm.Model() as model:
            alpha_subj = pm.Beta("alpha_subj", alpha=2, beta=2)
            gamma_subj = pm.Gamma("gamma_subj", alpha=2, beta=2)
            omega_subj = pm.Gamma("omega_subj", alpha=2, beta=2)
            beta_subj = pm.Gamma("beta_subj", alpha=1.5, beta=2)

            alpha_game_raw = pm.Normal("alpha_game_raw", mu=0, sigma=1, shape=n_games)
            alpha_game_sigma = pm.HalfNormal("alpha_game_sigma", sigma=0.15)
            alpha_logit_games = pm.math.logit(alpha_subj) + alpha_game_sigma * alpha_game_raw
            alpha_games = pm.Deterministic("alpha_games", pm.math.invlogit(alpha_logit_games))

            gamma_game_raw = pm.Normal("gamma_game_raw", mu=0, sigma=1, shape=n_games)
            gamma_game_sigma = pm.HalfNormal("gamma_game_sigma", sigma=0.3)
            gamma_games = pm.Deterministic("gamma_games", pm.math.exp(pm.math.log(gamma_subj) + gamma_game_sigma * gamma_game_raw))

            omega_game_raw = pm.Normal("omega_game_raw", mu=0, sigma=1, shape=n_games)
            omega_game_sigma = pm.HalfNormal("omega_game_sigma", sigma=0.3)
            omega_games = pm.Deterministic("omega_games", pm.math.exp(pm.math.log(omega_subj) + omega_game_sigma * omega_game_raw))
            beta_games = pt.ones(n_games) * beta_subj

            def compute_q_i_single_game(game_idx):
                forced_choices_g = all_forced_choices[game_idx]
                forced_rewards_g = all_forced_rewards[game_idx]
                alpha_g = alpha_games[game_idx]
                q = pt.zeros(num_decks)
                info = pt.zeros(num_decks)
                for step in range(num_force_trials):
                    choice_idx = forced_choices_g[step]
                    reward = forced_rewards_g[step, choice_idx]
                    is_valid = ~pt.isnan(reward)
                    delta = reward - q[choice_idx]
                    q = pt.set_subtensor(q[choice_idx], q[choice_idx] + pt.switch(is_valid, alpha_g * delta, 0.0))
                    info = pt.set_subtensor(info[choice_idx], info[choice_idx] + 1)
                return q, info

            q_all, info_all = [], []
            for idx in range(n_games):
                q, info = compute_q_i_single_game(idx)
                q_all.append(q)
                info_all.append(info)
            q_all = pt.stack(q_all)
            info_all = pt.stack(info_all)

            gamma_safe = pt.clip(gamma_games[:, None], 0.05, 2.0)
            info_transformed = pt.pow(info_all, gamma_safe)
            omega_safe = pt.clip(omega_games[:, None], 0.01, 5.0)
            value_all = q_all - omega_safe * info_transformed
            value_safe = pt.clip(value_all, -8, 8)
            beta_safe = pt.clip(beta_games[:, None], 0.1, 5)
            value_shifted = value_safe - pt.max(value_safe, axis=1, keepdims=True)
            scaled_value = pt.clip(beta_safe * value_shifted, -15, 15)
            exp_value = pt.exp(scaled_value)
            probs = exp_value / (pt.sum(exp_value, axis=1, keepdims=True) + 1e-8)
            probs = pt.clip(probs, 1e-8, 1 - 1e-8)
            pm.Categorical("choices", p=probs, observed=all_free_choices)
        return model

    def sample_model(model):
        with model:
            if jax_available:
                try:
                    from pymc.sampling.jax import sample_numpyro_nuts
                    return sample_numpyro_nuts(
                        draws=draws,
                        tune=tune,
                        chains=chains,
                        target_accept=0.98,
                        random_seed=random_state,
                        chain_method="vectorized",
                        idata_kwargs={"log_likelihood": True},
                    )
                except Exception as exc:
                    logger.warning("JAX sampler failed, falling back to PyMC NUTS: %s", exc)
            return pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                cores=cores,
                return_inferencedata=True,
                target_accept=0.98,
                max_treedepth=12,
                random_seed=random_state,
                idata_kwargs={"log_likelihood": True},
            )

    def extract_game_metrics(trace, trial_data_list, df_processed, num_decks=3, num_force_trials=6):
        posterior_mean = trace.posterior.mean(dim=["chain", "draw"])
        beta_subj = float(posterior_mean["beta_subj"].values)
        alpha_games = np.atleast_1d(posterior_mean["alpha_games"].values)
        gamma_games = np.atleast_1d(posterior_mean["gamma_games"].values)
        omega_games = np.atleast_1d(posterior_mean["omega_games"].values)
        expected = len(trial_data_list)
        if len(alpha_games) != expected:
            alpha_games = np.resize(alpha_games, expected)
            gamma_games = np.resize(gamma_games, expected)
            omega_games = np.resize(omega_games, expected)
        trial_idx_map = {item["game_idx"]: idx for idx, item in enumerate(trial_data_list)}
        rows = []
        for game_idx in range(df_processed["Game"].nunique()):
            game_data = df_processed[df_processed["Game"] == game_idx + 1].sort_values("Trial").reset_index(drop=True)
            if len(game_data) < num_force_trials + 1:
                continue
            trial_idx = trial_idx_map.get(game_idx)
            if trial_idx is None:
                continue
            alpha_g = float(alpha_games[trial_idx])
            gamma_g = float(gamma_games[trial_idx])
            omega_g = float(omega_games[trial_idx])
            beta_g = beta_subj
            q = np.zeros(num_decks)
            info = np.zeros(num_decks)
            for trial in range(num_force_trials):
                row = game_data.iloc[trial]
                try:
                    chosen_deck = int(row["Chosen Deck"]) - 1
                    if not (0 <= chosen_deck < num_decks):
                        raise ValueError
                except (ValueError, TypeError):
                    chosen_deck = int(np.argmax(q))
                reward = row[f"Reward Card {chosen_deck + 1}"]
                if pd.notna(reward):
                    q[chosen_deck] += alpha_g * (reward - q[chosen_deck])
                info[chosen_deck] += 1
            info_transformed = np.power(info, gamma_g)
            weighted_info = omega_g * info_transformed
            value = q - weighted_info
            value_shifted = value - np.max(value)
            exp_value = np.exp(np.clip(beta_g * value_shifted, -15, 15))
            probs = exp_value / (np.sum(exp_value) + 1e-8)
            free_choice = int(game_data.iloc[num_force_trials]["Chosen Deck"])
            row = {
                "Game": game_idx + 1,
                "alpha": alpha_g,
                "gamma": gamma_g,
                "omega": omega_g,
                "beta": beta_g,
                "Chosen_Deck": free_choice,
                "Chosen_Prob": probs[free_choice - 1] if 1 <= free_choice <= num_decks else np.nan,
            }
            for deck in range(num_decks):
                row[f"Q_Deck_{deck + 1}"] = q[deck]
                row[f"I_Deck_{deck + 1}"] = info[deck]
                row[f"I_transformed_Deck_{deck + 1}"] = info_transformed[deck]
                row[f"V_Deck_{deck + 1}"] = value[deck]
                row[f"Softmax_Prob_Deck_{deck + 1}"] = probs[deck]
                row[f"Weighted_I_transformed_Deck_{deck + 1}"] = weighted_info[deck]
            rows.append(row)
        ordered = ["Game", "alpha", "gamma", "omega", "beta"]
        ordered += [f"Q_Deck_{idx}" for idx in [1, 2, 3]]
        ordered += [f"I_Deck_{idx}" for idx in [1, 2, 3]]
        ordered += [f"I_transformed_Deck_{idx}" for idx in [1, 2, 3]]
        ordered += [f"V_Deck_{idx}" for idx in [1, 2, 3]]
        ordered += [f"Softmax_Prob_Deck_{idx}" for idx in [1, 2, 3]]
        ordered += ["Chosen_Deck", "Chosen_Prob"]
        ordered += [f"Weighted_I_transformed_Deck_{idx}" for idx in [1, 2, 3]]
        return pd.DataFrame(rows)[ordered]

    csv_files = sorted(data_dir.glob("*.csv"))
    if requested_subjects is not None:
        csv_files = [path for path in csv_files if path.stem in requested_subjects]
    if not csv_files:
        raise FileNotFoundError(f"No behavioral CSV files found in {data_dir}")

    for csv_file in csv_files:
        base_name = csv_file.stem
        logger.info("Processing subject file: %s", csv_file.relative_to(repo_root))
        df_raw = pd.read_csv(csv_file, encoding="utf-8-sig")
        df_processed = preprocess_data(df_raw)
        if df_processed is None:
            logger.warning("Skipping %s because preprocessing returned no rows.", base_name)
            continue
        trial_data_list = extract_trial_data_for_bhm(df_processed)
        if not trial_data_list:
            logger.warning("Skipping %s because no valid games were extracted.", base_name)
            continue
        model = build_single_subject_model(trial_data_list)
        if model is None:
            continue
        trace = sample_model(model)
        game_metrics = extract_game_metrics(trace, trial_data_list, df_processed)
        game_metrics.to_csv(output_dir / f"{base_name}_game_metrics.csv", index=False)
        game_metrics.to_csv(weighted_dir / f"{base_name}_game_metrics.csv", index=False)
        summary = az.summary(trace, var_names=["alpha_subj", "gamma_subj", "omega_subj", "beta_subj"])
        summary.to_csv(output_dir / f"{base_name}_subject_params.csv")
        trace.to_netcdf(output_dir / f"{base_name}_trace.nc")
        try:
            az.plot_trace(trace, var_names=["alpha_subj", "gamma_subj", "omega_subj", "beta_subj"])
            plt.savefig(output_dir / f"{base_name}_trace.png", dpi=100)
            plt.close("all")
            az.plot_posterior(trace, var_names=["alpha_subj", "gamma_subj", "omega_subj", "beta_subj"])
            plt.savefig(output_dir / f"{base_name}_posterior.png", dpi=100)
            plt.close("all")
        except Exception as exc:
            logger.warning("Posterior plotting failed for %s: %s", base_name, exc)
        del model, trace
        gc.collect()

    logger.info("Behavioral model fitting complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
