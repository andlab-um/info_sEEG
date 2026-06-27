"""Sample-data checks and expected-output summaries for neural analyses."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from .io_utils import fmt, mean, read_csv_rows, to_float, write_csv


def _feature_groups(header: list[str]) -> Counter[str]:
    groups: Counter[str] = Counter()
    for col in header:
        lower = col.lower()
        if lower.startswith("acc_"):
            groups["acc_features"] += 1
        elif lower.startswith("vmpfc_"):
            groups["vmpfc_features"] += 1
        elif lower.startswith("bestdeck") or "entropy" in lower or "uncertainty" in lower:
            groups["belief_metrics"] += 1
    return groups


def _count_unique(rows: list[dict[str, str]], column: str) -> int:
    return len({row.get(column, "") for row in rows if row.get(column, "") != ""})


def _condition_counts(rows: list[dict[str, str]]) -> str:
    counts = Counter(row.get("condition_label", "missing") or "missing" for row in rows)
    return ";".join(f"{key}={counts[key]}" for key in sorted(counts))


def run_neural_demo(reveal_path: Path, normative_path: Path, output_dir: Path) -> list[Path]:
    reveal_header, reveal_rows = read_csv_rows(reveal_path)
    normative_header, normative_rows = read_csv_rows(normative_path)

    reveal_groups = _feature_groups(reveal_header)
    normative_groups = _feature_groups(normative_header)

    summary_rows = [
        {
            "table": "reveal_features",
            "rows": str(len(reveal_rows)),
            "subjects": str(_count_unique(reveal_rows, "Subject")),
            "games": str(_count_unique(reveal_rows, "Game")),
            "conditions": _condition_counts(reveal_rows),
            "acc_features": str(reveal_groups["acc_features"]),
            "vmpfc_features": str(reveal_groups["vmpfc_features"]),
            "belief_metrics": str(reveal_groups["belief_metrics"]),
            "note": "One-participant reveal-level neural feature table.",
        },
        {
            "table": "normative_state_metrics",
            "rows": str(len(normative_rows)),
            "subjects": str(_count_unique(normative_rows, "Subject")),
            "games": str(_count_unique(normative_rows, "Game")),
            "conditions": _condition_counts(normative_rows),
            "acc_features": str(normative_groups["acc_features"]),
            "vmpfc_features": str(normative_groups["vmpfc_features"]),
            "belief_metrics": str(normative_groups["belief_metrics"]),
            "note": "Full-analysis-style state metrics table included with the sample repository.",
        },
    ]

    switch_by_reveal = []
    if "bestdeck_changed_after_reveal" in normative_header:
        reveal_values = sorted({row.get("reveal_idx", "") for row in normative_rows if row.get("reveal_idx", "")})
        for reveal_idx in reveal_values:
            sub = [row for row in normative_rows if row.get("reveal_idx") == reveal_idx]
            switched = sum(1 for row in sub if str(row.get("bestdeck_changed_after_reveal", "")).strip() in {"1", "1.0", "True", "true"})
            switch_by_reveal.append(
                {
                    "reveal_idx": reveal_idx,
                    "trials": str(len(sub)),
                    "bestdeck_changed_count": str(switched),
                    "bestdeck_changed_rate": fmt(switched / len(sub) if sub else None),
                }
            )

    entropy_rows = []
    if "uncertainty_entropy_after" in normative_header:
        for label, reveal_set in [("early_1_to_3", {"1", "2", "3"}), ("late_4_to_6", {"4", "5", "6"})]:
            sub = [row for row in normative_rows if row.get("reveal_idx") in reveal_set]
            entropy_rows.append(
                {
                    "phase": label,
                    "rows": str(len(sub)),
                    "mean_uncertainty_entropy_after": fmt(mean(to_float(row.get("uncertainty_entropy_after")) for row in sub)),
                }
            )

    neural_summary = output_dir / "neural_feature_summary.csv"
    switch_summary = output_dir / "belief_reconfiguration_by_reveal.csv"
    entropy_summary = output_dir / "rsa_input_phase_summary.csv"
    write_csv(
        neural_summary,
        summary_rows,
        ["table", "rows", "subjects", "games", "conditions", "acc_features", "vmpfc_features", "belief_metrics", "note"],
    )
    write_csv(switch_summary, switch_by_reveal, ["reveal_idx", "trials", "bestdeck_changed_count", "bestdeck_changed_rate"])
    write_csv(entropy_summary, entropy_rows, ["phase", "rows", "mean_uncertainty_entropy_after"])
    return [neural_summary, switch_summary, entropy_summary]

