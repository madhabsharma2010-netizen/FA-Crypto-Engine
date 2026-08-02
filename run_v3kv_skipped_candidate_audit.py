from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


WINDOWS = [
    ("2022", "2022", 93),
    ("2023", "2023", 223),
    ("2024", "2024", 103),
    ("2025H1", "2025h1", 83),
]

EXPECTED_DISPOSITIONS = {
    "EXECUTED": 139,
    "MAX_POSITIONS": 226,
    "MIN_NOTIONAL": 78,
    "HIGH_BETA": 34,
    "ONE_ENTRY_PER_HOUR": 25,
}

REQUIRED_COLUMNS = {
    "window",
    "disposition",
    "symbol",
    "candidate_id",
    "candidate_rank",
    "due_candidate_count",
    "breakout_strength",
    "position_count",
    "position_symbols",
    "entry_time",
    "exit_time",
    "exit_reason",
    "net_r",
    "holding_hours",
}

REPORT_DIR = Path("reports")
EPSILON = 1e-12


def metric_row(
    frame: pd.DataFrame,
) -> dict[str, object]:

    r_values = pd.to_numeric(
        frame["net_r"],
        errors="raise",
    )

    holding = pd.to_numeric(
        frame["holding_hours"],
        errors="coerce",
    )

    ranks = pd.to_numeric(
        frame["candidate_rank"],
        errors="coerce",
    )

    due_counts = pd.to_numeric(
        frame["due_candidate_count"],
        errors="coerce",
    )

    position_counts = pd.to_numeric(
        frame["position_count"],
        errors="coerce",
    )

    positive = r_values[
        r_values > EPSILON
    ].sort_values(
        ascending=False
    )

    negative = r_values[
        r_values < -EPSILON
    ]

    gross_positive = float(
        positive.sum()
    )

    gross_negative = float(
        negative.sum()
    )

    if gross_negative < 0:
        diagnostic_pf = (
            gross_positive
            / abs(gross_negative)
        )
    else:
        diagnostic_pf = np.inf

    if gross_positive > 0:
        top_five_share = (
            float(
                positive.head(5).sum()
            )
            / gross_positive
            * 100.0
        )
    else:
        top_five_share = 0.0

    entry_hours = (
        pd.to_datetime(
            frame["entry_time"],
            errors="raise",
        )
        .nunique()
    )

    return {
        "candidates":
            int(len(frame)),

        "unique_entry_hours":
            int(entry_hours),

        "same_hour_extra_candidates":
            int(
                len(frame)
                - entry_hours
            ),

        "wins":
            int(
                (
                    r_values > EPSILON
                ).sum()
            ),

        "losses":
            int(
                (
                    r_values < -EPSILON
                ).sum()
            ),

        "flat":
            int(
                (
                    r_values.abs()
                    <= EPSILON
                ).sum()
            ),

        "win_rate_pct":
            float(
                (
                    r_values > EPSILON
                ).mean()
                * 100.0
            ),

        "mean_r":
            float(
                r_values.mean()
            ),

        "median_r":
            float(
                r_values.median()
            ),

        "mean_r_capped_3r":
            float(
                r_values
                .clip(
                    lower=-3.0,
                    upper=3.0,
                )
                .mean()
            ),

        "sum_r_diagnostic":
            float(
                r_values.sum()
            ),

        "gross_positive_r":
            gross_positive,

        "gross_negative_r":
            gross_negative,

        "diagnostic_r_profit_factor":
            float(
                diagnostic_pf
            ),

        "minimum_r":
            float(
                r_values.min()
            ),

        "maximum_r":
            float(
                r_values.max()
            ),

        "p25_r":
            float(
                r_values.quantile(0.25)
            ),

        "p75_r":
            float(
                r_values.quantile(0.75)
            ),

        "ge_0_5r":
            int(
                (
                    r_values >= 0.5
                ).sum()
            ),

        "ge_1r":
            int(
                (
                    r_values >= 1.0
                ).sum()
            ),

        "ge_2r":
            int(
                (
                    r_values >= 2.0
                ).sum()
            ),

        "ge_3r":
            int(
                (
                    r_values >= 3.0
                ).sum()
            ),

        "le_minus_0_5r":
            int(
                (
                    r_values <= -0.5
                ).sum()
            ),

        "le_minus_1r":
            int(
                (
                    r_values <= -1.0
                ).sum()
            ),

        "top5_positive_r_share_pct":
            float(
                top_five_share
            ),

        "average_holding_hours":
            float(
                holding.mean()
            ),

        "median_holding_hours":
            float(
                holding.median()
            ),

        "average_candidate_rank":
            float(
                ranks.mean()
            ),

        "rank_1_pct":
            float(
                (
                    ranks == 1
                ).mean()
                * 100.0
            ),

        "average_due_candidates":
            float(
                due_counts.mean()
            ),

        "average_position_count":
            float(
                position_counts.mean()
            ),
    }


def grouped_metrics(
    frame: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:

    output_rows: list[
        dict[str, object]
    ] = []

    grouped = frame.groupby(
        group_columns,
        dropna=False,
        sort=True,
    )

    for keys, group in grouped:

        if not isinstance(
            keys,
            tuple,
        ):
            keys = (
                keys,
            )

        output = {
            column: value
            for column, value in zip(
                group_columns,
                keys,
            )
        }

        output.update(
            metric_row(group)
        )

        output_rows.append(
            output
        )

    return pd.DataFrame(
        output_rows
    )


frames: list[pd.DataFrame] = []

for audit_window, tag, expected_rows in WINDOWS:

    path = (
        REPORT_DIR
        / f"v3kv_shadow_outcomes_{tag}.csv"
    )

    if not path.exists():
        raise RuntimeError(
            f"MISSING_SHADOW_REPORT: {path}"
        )

    frame = pd.read_csv(path)

    if len(frame) != expected_rows:
        raise RuntimeError(
            "WINDOW_COUNT_MISMATCH: "
            f"window={audit_window}, "
            f"expected={expected_rows}, "
            f"actual={len(frame)}"
        )

    missing = sorted(
        REQUIRED_COLUMNS
        - set(frame.columns)
    )

    if missing:
        raise RuntimeError(
            "MISSING_REQUIRED_COLUMNS: "
            f"window={audit_window}, "
            f"columns={missing}"
        )

    frame.insert(
        0,
        "audit_window",
        audit_window,
    )

    frames.append(frame)


combined = pd.concat(
    frames,
    ignore_index=True,
    sort=False,
)

if len(combined) != 502:
    raise RuntimeError(
        "COMBINED_COUNT_MISMATCH: "
        f"expected=502, "
        f"actual={len(combined)}"
    )

duplicate_count = int(
    combined.duplicated(
        subset=[
            "audit_window",
            "candidate_id",
        ]
    ).sum()
)

if duplicate_count:
    raise RuntimeError(
        "DUPLICATE_CANDIDATES: "
        f"{duplicate_count}"
    )

actual_dispositions = (
    combined[
        "disposition"
    ]
    .value_counts()
    .to_dict()
)

if actual_dispositions != EXPECTED_DISPOSITIONS:
    raise RuntimeError(
        "DISPOSITION_COUNT_MISMATCH: "
        f"expected={EXPECTED_DISPOSITIONS}, "
        f"actual={actual_dispositions}"
    )

combined["net_r"] = pd.to_numeric(
    combined["net_r"],
    errors="raise",
)

combined["entry_time"] = pd.to_datetime(
    combined["entry_time"],
    errors="raise",
)

combined["exit_time"] = pd.to_datetime(
    combined["exit_time"],
    errors="raise",
)

executed = combined[
    combined["disposition"]
    == "EXECUTED"
].copy()

skipped = combined[
    combined["disposition"]
    != "EXECUTED"
].copy()

if len(executed) != 139:
    raise RuntimeError(
        "EXECUTED_COUNT_MISMATCH: "
        f"{len(executed)}"
    )

if len(skipped) != 363:
    raise RuntimeError(
        "SKIPPED_COUNT_MISMATCH: "
        f"{len(skipped)}"
    )


skipped["is_positive_r"] = (
    skipped["net_r"] > EPSILON
)

skipped["is_ge_1r"] = (
    skipped["net_r"] >= 1.0
)

skipped["is_ge_2r"] = (
    skipped["net_r"] >= 2.0
)

skipped["is_le_minus_1r"] = (
    skipped["net_r"] <= -1.0
)


overall_skipped = pd.DataFrame(
    [
        {
            "scope": "ALL_SKIPPED",
            **metric_row(skipped),
        }
    ]
)

reason_summary = grouped_metrics(
    skipped,
    [
        "disposition",
    ],
)

window_summary = grouped_metrics(
    skipped,
    [
        "audit_window",
    ],
)

window_reason_summary = grouped_metrics(
    skipped,
    [
        "audit_window",
        "disposition",
    ],
)

reason_symbol_summary = grouped_metrics(
    skipped,
    [
        "disposition",
        "symbol",
    ],
)

window_symbol_summary = grouped_metrics(
    skipped,
    [
        "audit_window",
        "symbol",
    ],
)


stability_rows: list[
    dict[str, object]
] = []

for disposition, group in (
    window_reason_summary
    .groupby(
        "disposition",
        sort=True,
    )
):

    positive_mean_windows = int(
        (
            group["mean_r"]
            > EPSILON
        ).sum()
    )

    positive_median_windows = int(
        (
            group["median_r"]
            > EPSILON
        ).sum()
    )

    windows_present = int(
        len(group)
    )

    overall_row = (
        reason_summary[
            reason_summary[
                "disposition"
            ] == disposition
        ]
        .iloc[0]
    )

    stability_rows.append(
        {
            "disposition":
                disposition,

            "windows_present":
                windows_present,

            "positive_mean_windows":
                positive_mean_windows,

            "negative_or_flat_mean_windows":
                (
                    windows_present
                    - positive_mean_windows
                ),

            "positive_median_windows":
                positive_median_windows,

            "minimum_window_mean_r":
                float(
                    group[
                        "mean_r"
                    ].min()
                ),

            "maximum_window_mean_r":
                float(
                    group[
                        "mean_r"
                    ].max()
                ),

            "minimum_window_win_rate_pct":
                float(
                    group[
                        "win_rate_pct"
                    ].min()
                ),

            "maximum_window_win_rate_pct":
                float(
                    group[
                        "win_rate_pct"
                    ].max()
                ),

            "combined_candidates":
                int(
                    overall_row[
                        "candidates"
                    ]
                ),

            "combined_mean_r":
                float(
                    overall_row[
                        "mean_r"
                    ]
                ),

            "combined_median_r":
                float(
                    overall_row[
                        "median_r"
                    ]
                ),

            "combined_mean_r_capped_3r":
                float(
                    overall_row[
                        "mean_r_capped_3r"
                    ]
                ),

            "combined_ge_1r":
                int(
                    overall_row[
                        "ge_1r"
                    ]
                ),

            "combined_ge_2r":
                int(
                    overall_row[
                        "ge_2r"
                    ]
                ),

            "combined_le_minus_1r":
                int(
                    overall_row[
                        "le_minus_1r"
                    ]
                ),

            "all_window_means_positive":
                bool(
                    positive_mean_windows
                    == windows_present
                ),
        }
    )

stability_summary = pd.DataFrame(
    stability_rows
)


entry_hour_counts = (
    skipped
    .groupby(
        [
            "audit_window",
            "entry_time",
        ]
    )
    .size()
    .reset_index(
        name="skipped_candidates"
    )
)

max_same_hour_candidates = int(
    entry_hour_counts[
        "skipped_candidates"
    ].max()
)

multi_candidate_hours = int(
    (
        entry_hour_counts[
            "skipped_candidates"
        ] > 1
    ).sum()
)


detail_columns = [
    "audit_window",
    "disposition",
    "symbol",
    "candidate_id",
    "candidate_rank",
    "due_candidate_count",
    "breakout_strength",
    "position_count",
    "position_symbols",
    "entry_time",
    "exit_time",
    "exit_reason",
    "net_r",
    "holding_hours",
    "is_positive_r",
    "is_ge_1r",
    "is_ge_2r",
    "is_le_minus_1r",
]

skipped_detail = (
    skipped[
        detail_columns
    ]
    .sort_values(
        [
            "net_r",
            "audit_window",
            "entry_time",
        ],
        ascending=[
            False,
            True,
            True,
        ],
    )
)

top_opportunities = (
    skipped_detail[
        skipped_detail[
            "net_r"
        ] >= 1.0
    ]
    .copy()
)

large_avoided_losses = (
    skipped_detail[
        skipped_detail[
            "net_r"
        ] <= -1.0
    ]
    .sort_values(
        "net_r",
        ascending=True,
    )
    .copy()
)

exit_reason_cross_tab = (
    pd.crosstab(
        skipped["disposition"],
        skipped["exit_reason"],
        margins=True,
    )
    .reset_index()
)


outputs = {
    "v3kv_skipped_candidate_audit_overall.csv":
        overall_skipped,

    "v3kv_skipped_candidate_audit_by_reason.csv":
        reason_summary,

    "v3kv_skipped_candidate_audit_by_window.csv":
        window_summary,

    "v3kv_skipped_candidate_audit_by_window_reason.csv":
        window_reason_summary,

    "v3kv_skipped_candidate_audit_by_reason_symbol.csv":
        reason_symbol_summary,

    "v3kv_skipped_candidate_audit_by_window_symbol.csv":
        window_symbol_summary,

    "v3kv_skipped_candidate_audit_stability.csv":
        stability_summary,

    "v3kv_skipped_candidate_audit_detail.csv":
        skipped_detail,

    "v3kv_skipped_candidate_audit_ge_1r.csv":
        top_opportunities,

    "v3kv_skipped_candidate_audit_le_minus_1r.csv":
        large_avoided_losses,

    "v3kv_skipped_candidate_audit_exit_reasons.csv":
        exit_reason_cross_tab,

    "v3kv_skipped_candidate_audit_entry_hour_density.csv":
        entry_hour_counts,
}

for filename, frame in outputs.items():

    frame.to_csv(
        REPORT_DIR / filename,
        index=False,
    )


lines: list[str] = []


def emit(
    text: str = "",
) -> None:

    print(text)
    lines.append(text)


emit("")
emit("=" * 118)
emit("V3KV SKIPPED-CANDIDATE AUDIT")
emit("=" * 118)
emit("Historical inspected windows only — not untouched OOS.")
emit("Primary metric: net R. Unit-quantity EUR P&L is not used for cross-asset conclusions.")
emit("Independent shadow outcomes are diagnostic and are not additive feasible portfolio returns.")
emit("")

emit(
    "Candidates: "
    f"total={len(combined)}, "
    f"executed={len(executed)}, "
    f"skipped={len(skipped)}"
)

emit(
    "Entry-hour overlap: "
    f"unique skipped entry hours="
    f"{skipped['entry_time'].nunique()}, "
    f"multi-candidate hours="
    f"{multi_candidate_hours}, "
    f"maximum candidates in one hour="
    f"{max_same_hour_candidates}"
)

emit("")
emit("-" * 118)
emit("SKIPPED RESULTS BY REASON")
emit("-" * 118)

reason_print_columns = [
    "disposition",
    "candidates",
    "win_rate_pct",
    "mean_r",
    "median_r",
    "mean_r_capped_3r",
    "ge_1r",
    "ge_2r",
    "le_minus_1r",
    "top5_positive_r_share_pct",
]

emit(
    reason_summary[
        reason_print_columns
    ]
    .sort_values(
        "candidates",
        ascending=False,
    )
    .to_string(
        index=False,
        float_format=lambda value: (
            f"{value:.4f}"
        ),
    )
)

emit("")
emit("-" * 118)
emit("SKIPPED RESULTS BY WINDOW")
emit("-" * 118)

window_print_columns = [
    "audit_window",
    "candidates",
    "win_rate_pct",
    "mean_r",
    "median_r",
    "mean_r_capped_3r",
    "ge_1r",
    "ge_2r",
    "le_minus_1r",
]

emit(
    window_summary[
        window_print_columns
    ]
    .to_string(
        index=False,
        float_format=lambda value: (
            f"{value:.4f}"
        ),
    )
)

emit("")
emit("-" * 118)
emit("CROSS-WINDOW STABILITY BY SKIP REASON")
emit("-" * 118)

stability_print_columns = [
    "disposition",
    "combined_candidates",
    "windows_present",
    "positive_mean_windows",
    "negative_or_flat_mean_windows",
    "minimum_window_mean_r",
    "maximum_window_mean_r",
    "combined_mean_r",
    "combined_median_r",
    "combined_mean_r_capped_3r",
    "combined_ge_1r",
    "combined_ge_2r",
    "combined_le_minus_1r",
]

emit(
    stability_summary[
        stability_print_columns
    ]
    .sort_values(
        "combined_candidates",
        ascending=False,
    )
    .to_string(
        index=False,
        float_format=lambda value: (
            f"{value:.4f}"
        ),
    )
)

emit("")
emit("-" * 118)
emit("MASTER OBJECTIVE CHECK — DIAGNOSTIC STAGE")
emit("-" * 118)
emit("1. Loss/drawdown improvement: NOT YET MEASURED. No portfolio rule was changed or replayed.")
emit("2. Upside preservation: Measured only as counts/distribution of skipped positive-R and 1R/2R shadows.")
emit("3. Stability: Window/reason and symbol breakdowns were produced; sparse groups require caution.")
emit("4. EUR 200 weekly-average target: NOT YET MEASURABLE from overlapping unit-R shadows.")
emit("Next gate: select one narrow candidate policy, then run causal shared-portfolio replay with weekly EUR audit.")
emit("")
emit("V3KV skipped-candidate audit: PASS")

summary_path = (
    REPORT_DIR
    / "v3kv_skipped_candidate_audit_summary.txt"
)

summary_path.write_text(
    "\n".join(lines) + "\n",
    encoding="utf-8",
)
