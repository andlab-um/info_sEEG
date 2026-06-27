"""Behavioral one-participant demo summaries."""

from __future__ import annotations

from pathlib import Path

from .io_utils import fmt, mean, read_csv_rows, to_float, write_csv


REQUIRED_GAME_COLUMNS = [
    "Game",
    "omega",
    "beta",
    "Q_Deck_1",
    "Q_Deck_2",
    "Q_Deck_3",
    "V_Deck_1",
    "V_Deck_2",
    "V_Deck_3",
    "Chosen_Deck",
    "Chosen_Prob",
]


def _argmax_deck(row: dict[str, str], prefix: str) -> int | None:
    values: list[tuple[int, float]] = []
    for deck in (1, 2, 3):
        value = to_float(row.get(f"{prefix}_Deck_{deck}"))
        if value is not None:
            values.append((deck, value))
    if len(values) != 3:
        return None
    return max(values, key=lambda item: item[1])[0]


def run_behavior_demo(metrics_path: Path, subject_params_path: Path, output_dir: Path) -> list[Path]:
    header, rows = read_csv_rows(metrics_path)
    missing = [col for col in REQUIRED_GAME_COLUMNS if col not in header]
    if missing:
        raise ValueError(f"Behavioral metrics file is missing columns: {', '.join(missing)}")

    conflict_rows = []
    choice_matches_value = 0
    choice_matches_reward = 0
    for row in rows:
        best_v = _argmax_deck(row, "V")
        best_q = _argmax_deck(row, "Q")
        chosen = to_float(row.get("Chosen_Deck"))
        if best_v is None or best_q is None or chosen is None or best_v == best_q:
            continue
        conflict_rows.append(row)
        if int(chosen) == best_v:
            choice_matches_value += 1
        if int(chosen) == best_q:
            choice_matches_reward += 1

    summary_rows = [
        {
            "metric": "games",
            "value": str(len(rows)),
            "note": "Number of free-choice games in the sample model-derived table.",
        },
        {
            "metric": "mean_chosen_probability",
            "value": fmt(mean(to_float(row.get("Chosen_Prob")) for row in rows)),
            "note": "Mean gkRL probability assigned to the observed choice.",
        },
        {
            "metric": "mean_omega",
            "value": fmt(mean(to_float(row.get("omega")) for row in rows)),
            "note": "Mean information-weight parameter across sample games.",
        },
        {
            "metric": "mean_beta",
            "value": fmt(mean(to_float(row.get("beta")) for row in rows)),
            "note": "Mean inverse-temperature value in the sample game table.",
        },
        {
            "metric": "conflict_games",
            "value": str(len(conflict_rows)),
            "note": "Games where argmax composite value differs from argmax reward.",
        },
        {
            "metric": "conflict_choice_matches_highest_value",
            "value": str(choice_matches_value),
            "note": "Count of conflict games where the chosen deck has the highest composite value.",
        },
        {
            "metric": "conflict_choice_matches_highest_reward",
            "value": str(choice_matches_reward),
            "note": "Count of conflict games where the chosen deck has the highest reward estimate.",
        },
    ]

    params_header, params_rows = read_csv_rows(subject_params_path)
    param_name_col = params_header[0]
    param_summary = []
    for row in params_rows:
        param_summary.append(
            {
                "parameter": row.get(param_name_col, ""),
                "mean": row.get("mean", ""),
                "sd": row.get("sd", ""),
                "hdi_3_percent": row.get("hdi_3%", ""),
                "hdi_97_percent": row.get("hdi_97%", ""),
                "r_hat": row.get("r_hat", ""),
            }
        )

    behavior_summary = output_dir / "behavior_summary.csv"
    parameter_summary = output_dir / "subject_parameter_summary.csv"
    write_csv(behavior_summary, summary_rows, ["metric", "value", "note"])
    write_csv(
        parameter_summary,
        param_summary,
        ["parameter", "mean", "sd", "hdi_3_percent", "hdi_97_percent", "r_hat"],
    )
    return [behavior_summary, parameter_summary]

