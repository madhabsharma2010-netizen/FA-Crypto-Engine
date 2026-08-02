from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


REPORT_DIR = Path("reports")

INPUT_PATH = (
    REPORT_DIR
    / "v3kw_replacement_outcome_join_detail.csv"
)

EXPECTED_EVENTS = 226
EXPECTED_ROWS = 452

TOLERANCE = 1e-12

SYMBOL_ORDER = {
    "BTCUSDT": 0,
    "ETHUSDT": 1,
    "SOLUSDT": 2,
    "XRPUSDT": 3,
    "LINKUSDT": 4,
    "DOGEUSDT": 5,
}


if not INPUT_PATH.exists():
    raise RuntimeError(
        f"JOIN_DETAIL_MISSING: {INPUT_PATH}"
    )


detail = pd.read_csv(
    INPUT_PATH
)


required_columns = {
    "audit_window",
    "incoming_candidate_id",
    "event_time",
    "incoming_symbol",
    "incoming_candidate_rank",
    "incoming_breakout_strength",
    "incoming_shadow_net_r",
    "existing_symbol",
    "existing_holding_hours",
    "existing_exit_now_net_r",
    "existing_gross_unrealized_r",
    "existing_stop_progress_r",
    "existing_stop_level_r",
    "existing_distance_to_stop_r",
    "existing_breakout_strength",
    "existing_pending_exit",
    "incumbent_future_incremental_r",
}

missing_columns = sorted(
    required_columns
    - set(detail.columns)
)

if missing_columns:
    raise RuntimeError(
        "SELECTOR_INPUT_COLUMNS_MISSING: "
        f"{missing_columns}"
    )


if len(detail) != EXPECTED_ROWS:
    raise RuntimeError(
        "SELECTOR_INPUT_ROW_MISMATCH: "
        f"expected={EXPECTED_ROWS}, "
        f"actual={len(detail)}"
    )


event_keys = [
    "audit_window",
    "incoming_candidate_id",
]

event_counts = (
    detail
    .groupby(
        event_keys,
        dropna=False,
    )
    .size()
)

if len(event_counts) != EXPECTED_EVENTS:
    raise RuntimeError(
        "SELECTOR_EVENT_COUNT_MISMATCH: "
        f"expected={EXPECTED_EVENTS}, "
        f"actual={len(event_counts)}"
    )

if not (
    event_counts == 2
).all():
    raise RuntimeError(
        "SELECTOR_ROWS_PER_EVENT_FAILED"
    )


numeric_columns = [
    "incoming_candidate_rank",
    "incoming_breakout_strength",
    "incoming_shadow_net_r",
    "existing_holding_hours",
    "existing_exit_now_net_r",
    "existing_gross_unrealized_r",
    "existing_stop_progress_r",
    "existing_stop_level_r",
    "existing_distance_to_stop_r",
    "existing_breakout_strength",
    "incumbent_future_incremental_r",
]

for column in numeric_columns:

    detail[column] = pd.to_numeric(
        detail[column],
        errors="raise",
    )


def parse_bool(
    value: object,
) -> bool:

    if isinstance(
        value,
        bool,
    ):
        return value

    normalized = str(
        value
    ).strip().lower()

    if normalized in {
        "true",
        "1",
        "yes",
    }:
        return True

    if normalized in {
        "false",
        "0",
        "no",
    }:
        return False

    raise RuntimeError(
        "INVALID_BOOLEAN_VALUE: "
        f"{value}"
    )


detail["existing_pending_exit"] = (
    detail[
        "existing_pending_exit"
    ]
    .map(
        parse_bool
    )
)

detail["_symbol_order"] = (
    detail[
        "existing_symbol"
    ]
    .map(
        SYMBOL_ORDER
    )
)

if detail[
    "_symbol_order"
].isna().any():

    unknown_symbols = sorted(
        detail.loc[
            detail[
                "_symbol_order"
            ].isna(),
            "existing_symbol",
        ]
        .astype(str)
        .unique()
        .tolist()
    )

    raise RuntimeError(
        "UNKNOWN_EXISTING_SYMBOLS: "
        f"{unknown_symbols}"
    )


SELECTORS = [
    "LOWEST_EXIT_NOW_NET_R",
    "LOWEST_GROSS_UNREALIZED_R",
    "LOWEST_STOP_PROGRESS_R",
    "LOWEST_STOP_LEVEL_R",
    "SMALLEST_DISTANCE_TO_STOP_R",
    "WEAKEST_BREAKOUT_STRENGTH",
    "OLDEST_HOLDING",
    "PENDING_EXIT_THEN_LOWEST_EXIT_R",
    "MAJORITY_WEAKNESS_VOTE",
    "SYMBOL_ORDER_CONTROL",
    "STRONGEST_EXIT_NOW_CONTROL",
]


def choose_sorted(
    group: pd.DataFrame,
    columns: list[str],
    ascending: list[bool],
) -> pd.Series:

    ordered = group.sort_values(
        columns,
        ascending=ascending,
        kind="mergesort",
    )

    return ordered.iloc[0]


def choose_majority_vote(
    group: pd.DataFrame,
) -> pd.Series:

    scored = group.copy()

    scored["_weakness_votes"] = 0

    minimum_fields = [
        "existing_exit_now_net_r",
        "existing_gross_unrealized_r",
        "existing_stop_progress_r",
        "existing_stop_level_r",
        "existing_distance_to_stop_r",
        "existing_breakout_strength",
    ]

    for column in minimum_fields:

        minimum_value = float(
            scored[column].min()
        )

        scored["_weakness_votes"] += (
            scored[column]
            <= minimum_value
            + TOLERANCE
        ).astype(int)

    maximum_holding = float(
        scored[
            "existing_holding_hours"
        ].max()
    )

    scored["_weakness_votes"] += (
        scored[
            "existing_holding_hours"
        ]
        >= maximum_holding
        - TOLERANCE
    ).astype(int)

    scored["_weakness_votes"] += (
        scored[
            "existing_pending_exit"
        ].astype(int)
    )

    scored = scored.sort_values(
        [
            "_weakness_votes",
            "existing_exit_now_net_r",
            "_symbol_order",
        ],
        ascending=[
            False,
            True,
            True,
        ],
        kind="mergesort",
    )

    return scored.iloc[0]


def select_position(
    selector: str,
    group: pd.DataFrame,
) -> pd.Series:

    if selector == "LOWEST_EXIT_NOW_NET_R":

        return choose_sorted(
            group,
            [
                "existing_exit_now_net_r",
                "_symbol_order",
            ],
            [
                True,
                True,
            ],
        )

    if selector == "LOWEST_GROSS_UNREALIZED_R":

        return choose_sorted(
            group,
            [
                "existing_gross_unrealized_r",
                "_symbol_order",
            ],
            [
                True,
                True,
            ],
        )

    if selector == "LOWEST_STOP_PROGRESS_R":

        return choose_sorted(
            group,
            [
                "existing_stop_progress_r",
                "_symbol_order",
            ],
            [
                True,
                True,
            ],
        )

    if selector == "LOWEST_STOP_LEVEL_R":

        return choose_sorted(
            group,
            [
                "existing_stop_level_r",
                "_symbol_order",
            ],
            [
                True,
                True,
            ],
        )

    if selector == "SMALLEST_DISTANCE_TO_STOP_R":

        return choose_sorted(
            group,
            [
                "existing_distance_to_stop_r",
                "_symbol_order",
            ],
            [
                True,
                True,
            ],
        )

    if selector == "WEAKEST_BREAKOUT_STRENGTH":

        return choose_sorted(
            group,
            [
                "existing_breakout_strength",
                "_symbol_order",
            ],
            [
                True,
                True,
            ],
        )

    if selector == "OLDEST_HOLDING":

        return choose_sorted(
            group,
            [
                "existing_holding_hours",
                "_symbol_order",
            ],
            [
                False,
                True,
            ],
        )

    if selector == "PENDING_EXIT_THEN_LOWEST_EXIT_R":

        return choose_sorted(
            group,
            [
                "existing_pending_exit",
                "existing_exit_now_net_r",
                "_symbol_order",
            ],
            [
                False,
                True,
                True,
            ],
        )

    if selector == "MAJORITY_WEAKNESS_VOTE":

        return choose_majority_vote(
            group
        )

    if selector == "SYMBOL_ORDER_CONTROL":

        return choose_sorted(
            group,
            [
                "_symbol_order",
            ],
            [
                True,
            ],
        )

    if selector == "STRONGEST_EXIT_NOW_CONTROL":

        return choose_sorted(
            group,
            [
                "existing_exit_now_net_r",
                "_symbol_order",
            ],
            [
                False,
                True,
            ],
        )

    raise RuntimeError(
        f"UNKNOWN_SELECTOR: {selector}"
    )


selector_rows: list[
    dict[str, object]
] = []


for (
    audit_window,
    incoming_candidate_id,
), group in detail.groupby(
    event_keys,
    sort=True,
    dropna=False,
):

    if len(group) != 2:
        raise RuntimeError(
            "SELECTOR_GROUP_SIZE_FAILED: "
            f"window={audit_window}, "
            f"candidate={incoming_candidate_id}"
        )

    constant_columns = [
        "event_time",
        "incoming_symbol",
        "incoming_candidate_rank",
        "incoming_breakout_strength",
        "incoming_shadow_net_r",
    ]

    for column in constant_columns:

        if group[column].nunique(
            dropna=False
        ) != 1:

            raise RuntimeError(
                "SELECTOR_EVENT_CONSTANT_MISMATCH: "
                f"window={audit_window}, "
                f"candidate={incoming_candidate_id}, "
                f"column={column}"
            )

    oracle_worst_future_r = float(
        group[
            "incumbent_future_incremental_r"
        ].min()
    )

    oracle_best_future_r = float(
        group[
            "incumbent_future_incremental_r"
        ].max()
    )

    incoming_shadow_r = float(
        group[
            "incoming_shadow_net_r"
        ].iloc[0]
    )

    for selector in SELECTORS:

        selected = select_position(
            selector,
            group,
        )

        selected_future_r = float(
            selected[
                "incumbent_future_incremental_r"
            ]
        )

        oracle_regret_r = (
            selected_future_r
            - oracle_worst_future_r
        )

        if oracle_regret_r < -TOLERANCE:
            raise RuntimeError(
                "NEGATIVE_ORACLE_REGRET: "
                f"selector={selector}, "
                f"window={audit_window}, "
                f"candidate={incoming_candidate_id}, "
                f"regret={oracle_regret_r}"
            )

        replacement_edge_r = (
            incoming_shadow_r
            - selected_future_r
        )

        selector_rows.append(
            {
                "selector":
                    selector,

                "audit_window":
                    audit_window,

                "incoming_candidate_id":
                    incoming_candidate_id,

                "event_time":
                    group[
                        "event_time"
                    ].iloc[0],

                "incoming_symbol":
                    group[
                        "incoming_symbol"
                    ].iloc[0],

                "incoming_candidate_rank":
                    int(
                        group[
                            "incoming_candidate_rank"
                        ].iloc[0]
                    ),

                "incoming_breakout_strength":
                    float(
                        group[
                            "incoming_breakout_strength"
                        ].iloc[0]
                    ),

                "incoming_shadow_net_r":
                    incoming_shadow_r,

                "selected_symbol":
                    selected[
                        "existing_symbol"
                    ],

                "selected_exit_now_net_r":
                    float(
                        selected[
                            "existing_exit_now_net_r"
                        ]
                    ),

                "selected_gross_unrealized_r":
                    float(
                        selected[
                            "existing_gross_unrealized_r"
                        ]
                    ),

                "selected_stop_progress_r":
                    float(
                        selected[
                            "existing_stop_progress_r"
                        ]
                    ),

                "selected_stop_level_r":
                    float(
                        selected[
                            "existing_stop_level_r"
                        ]
                    ),

                "selected_distance_to_stop_r":
                    float(
                        selected[
                            "existing_distance_to_stop_r"
                        ]
                    ),

                "selected_breakout_strength":
                    float(
                        selected[
                            "existing_breakout_strength"
                        ]
                    ),

                "selected_holding_hours":
                    float(
                        selected[
                            "existing_holding_hours"
                        ]
                    ),

                "selected_pending_exit":
                    bool(
                        selected[
                            "existing_pending_exit"
                        ]
                    ),

                "selected_future_incremental_r":
                    selected_future_r,

                "oracle_worst_future_r":
                    oracle_worst_future_r,

                "oracle_best_future_r":
                    oracle_best_future_r,

                "oracle_hit":
                    bool(
                        selected_future_r
                        <= oracle_worst_future_r
                        + TOLERANCE
                    ),

                "oracle_regret_r":
                    max(
                        0.0,
                        oracle_regret_r,
                    ),

                "incoming_minus_selected_future_r":
                    replacement_edge_r,

                "incoming_beats_selected":
                    bool(
                        replacement_edge_r
                        > TOLERANCE
                    ),

                "selected_future_negative":
                    bool(
                        selected_future_r
                        < -TOLERANCE
                    ),

                "incoming_positive_and_selected_negative":
                    bool(
                        incoming_shadow_r
                        > TOLERANCE
                        and selected_future_r
                        < -TOLERANCE
                    ),
            }
        )


selector_detail = pd.DataFrame(
    selector_rows
)


expected_selector_rows = (
    EXPECTED_EVENTS
    * len(
        SELECTORS
    )
)

if len(
    selector_detail
) != expected_selector_rows:

    raise RuntimeError(
        "SELECTOR_OUTPUT_ROW_MISMATCH: "
        f"expected={expected_selector_rows}, "
        f"actual={len(selector_detail)}"
    )


def summarize(
    frame: pd.DataFrame,
) -> dict[str, object]:

    edge = pd.to_numeric(
        frame[
            "incoming_minus_selected_future_r"
        ],
        errors="raise",
    )

    regret = pd.to_numeric(
        frame[
            "oracle_regret_r"
        ],
        errors="raise",
    )

    selected_future = pd.to_numeric(
        frame[
            "selected_future_incremental_r"
        ],
        errors="raise",
    )

    return {
        "events":
            int(
                len(frame)
            ),

        "oracle_hits":
            int(
                frame[
                    "oracle_hit"
                ].sum()
            ),

        "oracle_hit_rate_pct":
            float(
                frame[
                    "oracle_hit"
                ].mean()
                * 100.0
            ),

        "mean_oracle_regret_r":
            float(
                regret.mean()
            ),

        "median_oracle_regret_r":
            float(
                regret.median()
            ),

        "p75_oracle_regret_r":
            float(
                regret.quantile(
                    0.75
                )
            ),

        "mean_selected_future_r":
            float(
                selected_future.mean()
            ),

        "median_selected_future_r":
            float(
                selected_future.median()
            ),

        "selected_future_negative":
            int(
                frame[
                    "selected_future_negative"
                ].sum()
            ),

        "selected_future_negative_pct":
            float(
                frame[
                    "selected_future_negative"
                ].mean()
                * 100.0
            ),

        "incoming_beats_selected":
            int(
                frame[
                    "incoming_beats_selected"
                ].sum()
            ),

        "incoming_beats_selected_pct":
            float(
                frame[
                    "incoming_beats_selected"
                ].mean()
                * 100.0
            ),

        "positive_incoming_selected_negative":
            int(
                frame[
                    "incoming_positive_and_selected_negative"
                ].sum()
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
    }


combined_rows = []

for selector, group in selector_detail.groupby(
    "selector",
    sort=True,
):

    combined_rows.append(
        {
            "selector":
                selector,

            **summarize(
                group
            ),
        }
    )


combined_summary = pd.DataFrame(
    combined_rows
)


window_rows = []

for (
    selector,
    audit_window,
), group in selector_detail.groupby(
    [
        "selector",
        "audit_window",
    ],
    sort=True,
):

    window_rows.append(
        {
            "selector":
                selector,

            "audit_window":
                audit_window,

            **summarize(
                group
            ),
        }
    )


window_summary = pd.DataFrame(
    window_rows
)


stability_rows = []

for selector, group in window_summary.groupby(
    "selector",
    sort=True,
):

    combined_row = (
        combined_summary[
            combined_summary[
                "selector"
            ] == selector
        ]
        .iloc[0]
    )

    stability_rows.append(
        {
            "selector":
                selector,

            "windows_present":
                int(
                    len(group)
                ),

            "minimum_window_oracle_hit_rate_pct":
                float(
                    group[
                        "oracle_hit_rate_pct"
                    ].min()
                ),

            "maximum_window_oracle_hit_rate_pct":
                float(
                    group[
                        "oracle_hit_rate_pct"
                    ].max()
                ),

            "maximum_window_mean_regret_r":
                float(
                    group[
                        "mean_oracle_regret_r"
                    ].max()
                ),

            "positive_median_edge_windows":
                int(
                    (
                        group[
                            "median_replacement_edge_r"
                        ]
                        > TOLERANCE
                    ).sum()
                ),

            "positive_capped_edge_windows":
                int(
                    (
                        group[
                            "capped_3r_mean_replacement_edge"
                        ]
                        > TOLERANCE
                    ).sum()
                ),

            "combined_oracle_hit_rate_pct":
                float(
                    combined_row[
                        "oracle_hit_rate_pct"
                    ]
                ),

            "combined_mean_regret_r":
                float(
                    combined_row[
                        "mean_oracle_regret_r"
                    ]
                ),

            "combined_median_regret_r":
                float(
                    combined_row[
                        "median_oracle_regret_r"
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
        }
    )


stability_summary = pd.DataFrame(
    stability_rows
)


detail_output = (
    REPORT_DIR
    / "v3kw_weakest_selector_event_detail.csv"
)

combined_output = (
    REPORT_DIR
    / "v3kw_weakest_selector_combined.csv"
)

window_output = (
    REPORT_DIR
    / "v3kw_weakest_selector_by_window.csv"
)

stability_output = (
    REPORT_DIR
    / "v3kw_weakest_selector_stability.csv"
)

summary_output = (
    REPORT_DIR
    / "v3kw_weakest_selector_audit_summary.txt"
)


selector_detail.to_csv(
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


display_combined = [
    "selector",
    "events",
    "oracle_hit_rate_pct",
    "mean_oracle_regret_r",
    "median_oracle_regret_r",
    "incoming_beats_selected_pct",
    "median_replacement_edge_r",
    "capped_3r_mean_replacement_edge",
    "positive_incoming_selected_negative",
]

display_stability = [
    "selector",
    "windows_present",
    "minimum_window_oracle_hit_rate_pct",
    "maximum_window_mean_regret_r",
    "positive_median_edge_windows",
    "positive_capped_edge_windows",
    "combined_oracle_hit_rate_pct",
    "combined_mean_regret_r",
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
emit("=" * 130)
emit("V3KW CAUSAL WEAKEST-POSITION SELECTOR AUDIT")
emit("=" * 130)
emit(
    "Historical inspected windows only; "
    "not untouched out-of-sample."
)
emit(
    "Selectors use decision-time position fields only."
)
emit(
    "Future incumbent outcome is used only as "
    "the evaluation oracle."
)
emit(
    "No replacement action, portfolio mutation "
    "or risk-rule change occurred."
)
emit("")

emit(
    f"MAX_POSITIONS events:       "
    f"{EXPECTED_EVENTS}"
)

emit(
    f"Selectors tested:           "
    f"{len(SELECTORS)}"
)

emit(
    f"Selector-event rows:        "
    f"{len(selector_detail)}"
)

emit("")

emit("-" * 130)
emit("COMBINED SELECTOR PERFORMANCE")
emit("-" * 130)

emit(
    combined_summary
    .sort_values(
        [
            "oracle_hit_rate_pct",
            "mean_oracle_regret_r",
        ],
        ascending=[
            False,
            True,
        ],
    )[
        display_combined
    ]
    .to_string(
        index=False,
        float_format=lambda value: (
            f"{value:.4f}"
        ),
    )
)

emit("")

emit("-" * 130)
emit("CROSS-WINDOW SELECTOR STABILITY")
emit("-" * 130)

emit(
    stability_summary
    .sort_values(
        [
            "minimum_window_oracle_hit_rate_pct",
            "combined_oracle_hit_rate_pct",
            "combined_mean_regret_r",
        ],
        ascending=[
            False,
            False,
            True,
        ],
    )[
        display_stability
    ]
    .to_string(
        index=False,
        float_format=lambda value: (
            f"{value:.4f}"
        ),
    )
)

emit("")

emit("-" * 130)
emit("MASTER OBJECTIVE CHECK")
emit("-" * 130)

emit(
    "1. Loss/drawdown: NOT YET MEASURED; "
    "no shared-portfolio replacement replay occurred."
)

emit(
    "2. Upside preservation: selector ability to identify "
    "the weaker future incumbent was measured."
)

emit(
    "3. Stability: each selector was evaluated separately "
    "in all four inspected windows."
)

emit(
    "4. EUR 200 weekly-average target: NOT measurable "
    "from selector diagnostics."
)

emit(
    "Next gate: only a selector with stable oracle accuracy "
    "and low regret may be combined with a causal incoming-entry gate."
)

emit("")
emit(
    "V3KW weakest-position selector audit: PASS"
)


summary_output.write_text(
    "\n".join(lines) + "\n",
    encoding="utf-8",
)
