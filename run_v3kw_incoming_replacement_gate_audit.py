from __future__ import annotations

from pathlib import Path

import pandas as pd


REPORT_DIR = Path("reports")

INPUT_PATH = (
    REPORT_DIR
    / "v3kw_weakest_selector_event_detail.csv"
)

EXPECTED_EVENTS_PER_SELECTOR = 226

SELECTORS = [
    "OLDEST_HOLDING",
    "WEAKEST_BREAKOUT_STRENGTH",
]

GATES = [
    "ALL_EVENTS",
    "RANK1",
    "RANK1_STRENGTH_ADVANTAGE",
    "RANK1_STRENGTH_ADVANTAGE_SELECTED_NONPOSITIVE",
    "RANK1_STRENGTH_ADVANTAGE_SELECTED_NONPOSITIVE_NO_STOP_PROGRESS",
]


if not INPUT_PATH.exists():
    raise RuntimeError(
        f"SELECTOR_DETAIL_MISSING: {INPUT_PATH}"
    )


detail = pd.read_csv(
    INPUT_PATH
)


required_columns = {
    "selector",
    "audit_window",
    "incoming_candidate_id",
    "incoming_candidate_rank",
    "incoming_breakout_strength",
    "incoming_shadow_net_r",
    "selected_symbol",
    "selected_exit_now_net_r",
    "selected_stop_progress_r",
    "selected_breakout_strength",
    "selected_holding_hours",
    "selected_future_incremental_r",
    "oracle_hit",
    "oracle_regret_r",
    "incoming_minus_selected_future_r",
}

missing_columns = sorted(
    required_columns
    - set(detail.columns)
)

if missing_columns:
    raise RuntimeError(
        "GATE_INPUT_COLUMNS_MISSING: "
        f"{missing_columns}"
    )


detail = detail[
    detail["selector"].isin(
        SELECTORS
    )
].copy()


for selector in SELECTORS:

    selector_count = int(
        (
            detail["selector"]
            == selector
        ).sum()
    )

    if selector_count != EXPECTED_EVENTS_PER_SELECTOR:
        raise RuntimeError(
            "SELECTOR_EVENT_COUNT_MISMATCH: "
            f"selector={selector}, "
            f"expected={EXPECTED_EVENTS_PER_SELECTOR}, "
            f"actual={selector_count}"
        )


numeric_columns = [
    "incoming_candidate_rank",
    "incoming_breakout_strength",
    "incoming_shadow_net_r",
    "selected_exit_now_net_r",
    "selected_stop_progress_r",
    "selected_breakout_strength",
    "selected_holding_hours",
    "selected_future_incremental_r",
    "oracle_regret_r",
    "incoming_minus_selected_future_r",
]

for column in numeric_columns:

    detail[column] = pd.to_numeric(
        detail[column],
        errors="raise",
    )


detail["incoming_strength_gap"] = (
    detail["incoming_breakout_strength"]
    - detail["selected_breakout_strength"]
)


def gate_mask(
    frame: pd.DataFrame,
    gate: str,
) -> pd.Series:

    rank1 = (
        frame["incoming_candidate_rank"]
        == 1
    )

    strength_advantage = (
        frame["incoming_strength_gap"]
        > 0
    )

    selected_nonpositive = (
        frame["selected_exit_now_net_r"]
        <= 0
    )

    no_stop_progress = (
        frame["selected_stop_progress_r"]
        <= 0
    )

    if gate == "ALL_EVENTS":
        return pd.Series(
            True,
            index=frame.index,
        )

    if gate == "RANK1":
        return rank1

    if gate == "RANK1_STRENGTH_ADVANTAGE":
        return (
            rank1
            & strength_advantage
        )

    if (
        gate
        == "RANK1_STRENGTH_ADVANTAGE_SELECTED_NONPOSITIVE"
    ):
        return (
            rank1
            & strength_advantage
            & selected_nonpositive
        )

    if (
        gate
        == "RANK1_STRENGTH_ADVANTAGE_SELECTED_NONPOSITIVE_NO_STOP_PROGRESS"
    ):
        return (
            rank1
            & strength_advantage
            & selected_nonpositive
            & no_stop_progress
        )

    raise RuntimeError(
        f"UNKNOWN_GATE: {gate}"
    )


def summarize(
    frame: pd.DataFrame,
) -> dict[str, object]:

    if frame.empty:

        return {
            "events": 0,
            "incoming_positive": 0,
            "incoming_positive_pct": 0.0,
            "incoming_beats_selected": 0,
            "incoming_beats_selected_pct": 0.0,
            "positive_incoming_selected_negative": 0,
            "oracle_hits": 0,
            "oracle_hit_rate_pct": 0.0,
            "mean_oracle_regret_r": 0.0,
            "mean_replacement_edge_r": 0.0,
            "median_replacement_edge_r": 0.0,
            "capped_3r_mean_replacement_edge": 0.0,
            "negative_edge_events": 0,
            "negative_edge_pct": 0.0,
        }

    edge = pd.to_numeric(
        frame[
            "incoming_minus_selected_future_r"
        ],
        errors="raise",
    )

    incoming_r = pd.to_numeric(
        frame[
            "incoming_shadow_net_r"
        ],
        errors="raise",
    )

    selected_future_r = pd.to_numeric(
        frame[
            "selected_future_incremental_r"
        ],
        errors="raise",
    )

    oracle_regret = pd.to_numeric(
        frame[
            "oracle_regret_r"
        ],
        errors="raise",
    )

    incoming_positive = (
        incoming_r > 0
    )

    incoming_beats_selected = (
        edge > 0
    )

    positive_incoming_selected_negative = (
        incoming_positive
        & (
            selected_future_r < 0
        )
    )

    oracle_hit = (
        oracle_regret <= 1e-12
    )

    negative_edge = (
        edge < 0
    )

    return {
        "events":
            int(
                len(frame)
            ),

        "incoming_positive":
            int(
                incoming_positive.sum()
            ),

        "incoming_positive_pct":
            float(
                incoming_positive.mean()
                * 100.0
            ),

        "incoming_beats_selected":
            int(
                incoming_beats_selected.sum()
            ),

        "incoming_beats_selected_pct":
            float(
                incoming_beats_selected.mean()
                * 100.0
            ),

        "positive_incoming_selected_negative":
            int(
                positive_incoming_selected_negative.sum()
            ),

        "oracle_hits":
            int(
                oracle_hit.sum()
            ),

        "oracle_hit_rate_pct":
            float(
                oracle_hit.mean()
                * 100.0
            ),

        "mean_oracle_regret_r":
            float(
                oracle_regret.mean()
            ),

        "mean_replacement_edge_r":
            float(
                edge.mean()
            ),

        "median_replacement_edge_r":
            float(
                edge.median()
            ),

        "capped_3r_mean_replacement_edge":
            float(
                edge.clip(
                    lower=-3.0,
                    upper=3.0,
                ).mean()
            ),

        "negative_edge_events":
            int(
                negative_edge.sum()
            ),

        "negative_edge_pct":
            float(
                negative_edge.mean()
                * 100.0
            ),
    }


gate_detail_frames: list[
    pd.DataFrame
] = []

combined_rows: list[
    dict[str, object]
] = []

window_rows: list[
    dict[str, object]
] = []


for selector in SELECTORS:

    selector_frame = detail[
        detail["selector"]
        == selector
    ].copy()

    for gate in GATES:

        mask = gate_mask(
            selector_frame,
            gate,
        )

        gated = selector_frame[
            mask
        ].copy()

        gated.insert(
            1,
            "replacement_gate",
            gate,
        )

        gate_detail_frames.append(
            gated
        )

        combined_rows.append(
            {
                "selector":
                    selector,

                "replacement_gate":
                    gate,

                **summarize(
                    gated
                ),
            }
        )

        for audit_window in [
            "2022",
            "2023",
            "2024",
            "2025H1",
        ]:

            window_frame = gated[
                gated["audit_window"]
                == audit_window
            ]

            window_rows.append(
                {
                    "selector":
                        selector,

                    "replacement_gate":
                        gate,

                    "audit_window":
                        audit_window,

                    **summarize(
                        window_frame
                    ),
                }
            )


gate_detail = pd.concat(
    gate_detail_frames,
    ignore_index=True,
    sort=False,
)

combined_summary = pd.DataFrame(
    combined_rows
)

window_summary = pd.DataFrame(
    window_rows
)


stability_rows: list[
    dict[str, object]
] = []


for (
    selector,
    replacement_gate,
), group in window_summary.groupby(
    [
        "selector",
        "replacement_gate",
    ],
    sort=True,
):

    combined_row = combined_summary[
        (
            combined_summary["selector"]
            == selector
        )
        & (
            combined_summary["replacement_gate"]
            == replacement_gate
        )
    ].iloc[0]

    windows_with_events = int(
        (
            group["events"]
            > 0
        ).sum()
    )

    nonempty = group[
        group["events"]
        > 0
    ]

    if nonempty.empty:

        minimum_window_events = 0
        positive_median_edge_windows = 0
        positive_capped_edge_windows = 0
        worst_window_median_edge_r = 0.0
        worst_window_capped_edge_r = 0.0
        minimum_window_beats_selected_pct = 0.0

    else:

        minimum_window_events = int(
            nonempty["events"].min()
        )

        positive_median_edge_windows = int(
            (
                nonempty[
                    "median_replacement_edge_r"
                ]
                > 0
            ).sum()
        )

        positive_capped_edge_windows = int(
            (
                nonempty[
                    "capped_3r_mean_replacement_edge"
                ]
                > 0
            ).sum()
        )

        worst_window_median_edge_r = float(
            nonempty[
                "median_replacement_edge_r"
            ].min()
        )

        worst_window_capped_edge_r = float(
            nonempty[
                "capped_3r_mean_replacement_edge"
            ].min()
        )

        minimum_window_beats_selected_pct = float(
            nonempty[
                "incoming_beats_selected_pct"
            ].min()
        )

    stability_rows.append(
        {
            "selector":
                selector,

            "replacement_gate":
                replacement_gate,

            "combined_events":
                int(
                    combined_row["events"]
                ),

            "windows_with_events":
                windows_with_events,

            "minimum_window_events":
                minimum_window_events,

            "positive_median_edge_windows":
                positive_median_edge_windows,

            "positive_capped_edge_windows":
                positive_capped_edge_windows,

            "minimum_window_beats_selected_pct":
                minimum_window_beats_selected_pct,

            "worst_window_median_edge_r":
                worst_window_median_edge_r,

            "worst_window_capped_edge_r":
                worst_window_capped_edge_r,

            "combined_incoming_positive_pct":
                float(
                    combined_row[
                        "incoming_positive_pct"
                    ]
                ),

            "combined_incoming_beats_selected_pct":
                float(
                    combined_row[
                        "incoming_beats_selected_pct"
                    ]
                ),

            "combined_median_replacement_edge_r":
                float(
                    combined_row[
                        "median_replacement_edge_r"
                    ]
                ),

            "combined_capped_3r_mean_edge":
                float(
                    combined_row[
                        "capped_3r_mean_replacement_edge"
                    ]
                ),

            "combined_negative_edge_pct":
                float(
                    combined_row[
                        "negative_edge_pct"
                    ]
                ),
        }
    )


stability_summary = pd.DataFrame(
    stability_rows
)


detail_output = (
    REPORT_DIR
    / "v3kw_incoming_gate_event_detail.csv"
)

combined_output = (
    REPORT_DIR
    / "v3kw_incoming_gate_combined.csv"
)

window_output = (
    REPORT_DIR
    / "v3kw_incoming_gate_by_window.csv"
)

stability_output = (
    REPORT_DIR
    / "v3kw_incoming_gate_stability.csv"
)

summary_output = (
    REPORT_DIR
    / "v3kw_incoming_gate_audit_summary.txt"
)


gate_detail.to_csv(
    detail_output,
    index=False,
)

combined_summary.to_csv(
    combined_output,
    index=False,
)

window_summary.to_csv(
    window_output,
    index=False,
)

stability_summary.to_csv(
    stability_output,
    index=False,
)


combined_display_columns = [
    "selector",
    "replacement_gate",
    "events",
    "incoming_positive_pct",
    "incoming_beats_selected_pct",
    "oracle_hit_rate_pct",
    "mean_oracle_regret_r",
    "median_replacement_edge_r",
    "capped_3r_mean_replacement_edge",
    "negative_edge_pct",
]

stability_display_columns = [
    "selector",
    "replacement_gate",
    "combined_events",
    "windows_with_events",
    "minimum_window_events",
    "positive_median_edge_windows",
    "positive_capped_edge_windows",
    "minimum_window_beats_selected_pct",
    "worst_window_median_edge_r",
    "worst_window_capped_edge_r",
    "combined_incoming_beats_selected_pct",
    "combined_median_replacement_edge_r",
    "combined_capped_3r_mean_edge",
]


lines: list[str] = []


def emit(
    text: str = "",
) -> None:

    print(text)
    lines.append(text)


emit("")
emit("=" * 150)
emit("V3KW CAUSAL INCOMING-REPLACEMENT GATE AUDIT")
emit("=" * 150)
emit(
    "Historical inspected windows only; "
    "not untouched out-of-sample."
)
emit(
    "Only decision-time rank, relative breakout strength, "
    "current net R and stop progress are used."
)
emit(
    "Zero and relative comparisons were prespecified; "
    "no profit-derived threshold search occurred."
)
emit(
    "No replacement action, portfolio mutation "
    "or risk-rule change occurred."
)
emit("")

emit(
    f"Selectors retained:          "
    f"{len(SELECTORS)}"
)

emit(
    f"Nested causal gates:         "
    f"{len(GATES)}"
)

emit("")

emit("-" * 150)
emit("COMBINED CAUSAL-GATE DIAGNOSTICS")
emit("-" * 150)

emit(
    combined_summary[
        combined_display_columns
    ].to_string(
        index=False,
        float_format=lambda value: (
            f"{value:.4f}"
        ),
    )
)

emit("")

emit("-" * 150)
emit("CROSS-WINDOW CAUSAL-GATE STABILITY")
emit("-" * 150)

emit(
    stability_summary[
        stability_display_columns
    ].to_string(
        index=False,
        float_format=lambda value: (
            f"{value:.4f}"
        ),
    )
)

emit("")

emit("-" * 150)
emit("MASTER OBJECTIVE CHECK")
emit("-" * 150)

emit(
    "1. Loss/drawdown: NOT YET MEASURED; "
    "no shared-portfolio replacement replay occurred."
)

emit(
    "2. Upside preservation: incoming candidates were "
    "filtered causally before comparing future outcomes."
)

emit(
    "3. Stability: every retained selector/gate pair "
    "was separated across all four inspected windows."
)

emit(
    "4. EUR 200 weekly-average target: NOT measurable "
    "from overlapping replacement diagnostics."
)

emit(
    "Next decision: reject V3KW or nominate at most one "
    "prespecified selector/gate pair for isolated portfolio replay."
)

emit("")
emit(
    "V3KW incoming replacement gate audit: PASS"
)


summary_output.write_text(
    "\n".join(lines) + "\n",
    encoding="utf-8",
)
