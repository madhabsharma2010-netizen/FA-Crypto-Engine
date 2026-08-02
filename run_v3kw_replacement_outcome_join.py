from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


REPORT_DIR = Path("reports")

WINDOWS = [
    ("2022", "2022"),
    ("2023", "2023"),
    ("2024", "2024"),
    ("2025H1", "2025h1"),
]

EXPECTED_EVENTS = 226
EXPECTED_SNAPSHOT_ROWS = 452
EXPECTED_EXECUTED_TRADES = 139


snapshot_path = (
    REPORT_DIR
    / "v3kw_position_state_snapshots_combined.csv"
)

if not snapshot_path.exists():
    raise RuntimeError(
        f"SNAPSHOT_REPORT_MISSING: {snapshot_path}"
    )


snapshots = pd.read_csv(
    snapshot_path
)

required_snapshot_columns = {
    "audit_window",
    "window",
    "event_time",
    "incoming_candidate_id",
    "incoming_symbol",
    "incoming_signal_time",
    "incoming_candidate_rank",
    "incoming_due_candidate_count",
    "incoming_breakout_strength",
    "incoming_entry_price",
    "existing_symbol",
    "existing_entry_time",
    "existing_holding_hours",
    "existing_exit_now_net_r",
    "existing_gross_unrealized_r",
    "existing_stop_progress_r",
    "existing_stop_level_r",
    "existing_distance_to_stop_r",
    "existing_breakout_strength",
    "incoming_strength_gap",
    "existing_pending_exit",
    "snapshot_key",
}

missing_snapshot_columns = sorted(
    required_snapshot_columns
    - set(snapshots.columns)
)

if missing_snapshot_columns:
    raise RuntimeError(
        "SNAPSHOT_COLUMNS_MISSING: "
        f"{missing_snapshot_columns}"
    )


if len(snapshots) != EXPECTED_SNAPSHOT_ROWS:
    raise RuntimeError(
        "SNAPSHOT_ROW_COUNT_MISMATCH: "
        f"expected={EXPECTED_SNAPSHOT_ROWS}, "
        f"actual={len(snapshots)}"
    )


snapshot_events = (
    snapshots[
        [
            "audit_window",
            "incoming_candidate_id",
        ]
    ]
    .drop_duplicates()
)

if len(snapshot_events) != EXPECTED_EVENTS:
    raise RuntimeError(
        "SNAPSHOT_EVENT_COUNT_MISMATCH: "
        f"expected={EXPECTED_EVENTS}, "
        f"actual={len(snapshot_events)}"
    )


rows_per_event = (
    snapshots
    .groupby(
        [
            "audit_window",
            "incoming_candidate_id",
        ],
        dropna=False,
    )
    .size()
)

if not (
    rows_per_event == 2
).all():
    raise RuntimeError(
        "SNAPSHOT_ROWS_PER_EVENT_FAILED"
    )


snapshots["event_time"] = pd.to_datetime(
    snapshots["event_time"],
    utc=True,
    errors="raise",
)

snapshots["existing_entry_time"] = pd.to_datetime(
    snapshots["existing_entry_time"],
    utc=True,
    errors="raise",
)

snapshots["incoming_signal_time"] = pd.to_datetime(
    snapshots["incoming_signal_time"],
    utc=True,
    errors="raise",
)


shadow_frames: list[pd.DataFrame] = []
trade_frames: list[pd.DataFrame] = []


for audit_window, tag in WINDOWS:

    shadow_path = (
        REPORT_DIR
        / f"v3kv_shadow_outcomes_{tag}.csv"
    )

    trade_path = (
        REPORT_DIR
        / f"v3kv_observer_{tag}_trades.csv"
    )

    if not shadow_path.exists():
        raise RuntimeError(
            f"SHADOW_REPORT_MISSING: {shadow_path}"
        )

    if not trade_path.exists():
        raise RuntimeError(
            f"TRADE_REPORT_MISSING: {trade_path}"
        )


    shadow = pd.read_csv(
        shadow_path
    )

    required_shadow_columns = {
        "candidate_id",
        "disposition",
        "symbol",
        "signal_time",
        "entry_time",
        "exit_time",
        "exit_reason",
        "net_r",
        "holding_hours",
    }

    missing_shadow_columns = sorted(
        required_shadow_columns
        - set(shadow.columns)
    )

    if missing_shadow_columns:
        raise RuntimeError(
            "SHADOW_COLUMNS_MISSING: "
            f"window={audit_window}, "
            f"columns={missing_shadow_columns}"
        )

    shadow = shadow[
        shadow["disposition"]
        == "MAX_POSITIONS"
    ].copy()

    shadow.insert(
        0,
        "audit_window",
        audit_window,
    )

    shadow_frames.append(
        shadow
    )


    trades = pd.read_csv(
        trade_path
    )

    required_trade_columns = {
        "symbol",
        "entry_time",
        "exit_time",
        "exit_reason",
        "net_pnl_eur",
        "net_r",
        "holding_hours",
    }

    missing_trade_columns = sorted(
        required_trade_columns
        - set(trades.columns)
    )

    if missing_trade_columns:
        raise RuntimeError(
            "TRADE_COLUMNS_MISSING: "
            f"window={audit_window}, "
            f"columns={missing_trade_columns}"
        )

    trades.insert(
        0,
        "audit_window",
        audit_window,
    )

    trade_frames.append(
        trades
    )


shadow_candidates = pd.concat(
    shadow_frames,
    ignore_index=True,
    sort=False,
)

canonical_trades = pd.concat(
    trade_frames,
    ignore_index=True,
    sort=False,
)


if len(shadow_candidates) != EXPECTED_EVENTS:
    raise RuntimeError(
        "MAX_POSITIONS_SHADOW_COUNT_MISMATCH: "
        f"expected={EXPECTED_EVENTS}, "
        f"actual={len(shadow_candidates)}"
    )


if len(canonical_trades) != EXPECTED_EXECUTED_TRADES:
    raise RuntimeError(
        "CANONICAL_TRADE_COUNT_MISMATCH: "
        f"expected={EXPECTED_EXECUTED_TRADES}, "
        f"actual={len(canonical_trades)}"
    )


shadow_duplicate_count = int(
    shadow_candidates.duplicated(
        subset=[
            "audit_window",
            "candidate_id",
        ]
    ).sum()
)

if shadow_duplicate_count != 0:
    raise RuntimeError(
        "DUPLICATE_SHADOW_CANDIDATES: "
        f"{shadow_duplicate_count}"
    )


canonical_trades["entry_time"] = pd.to_datetime(
    canonical_trades["entry_time"],
    utc=True,
    errors="raise",
)

canonical_trades["exit_time"] = pd.to_datetime(
    canonical_trades["exit_time"],
    utc=True,
    errors="raise",
)


trade_duplicate_count = int(
    canonical_trades.duplicated(
        subset=[
            "audit_window",
            "symbol",
            "entry_time",
        ]
    ).sum()
)

if trade_duplicate_count != 0:
    raise RuntimeError(
        "DUPLICATE_CANONICAL_TRADE_KEYS: "
        f"{trade_duplicate_count}"
    )


shadow_join = shadow_candidates[
    [
        "audit_window",
        "candidate_id",
        "symbol",
        "entry_time",
        "exit_time",
        "exit_reason",
        "net_r",
        "holding_hours",
    ]
].rename(
    columns={
        "candidate_id":
            "incoming_candidate_id",

        "symbol":
            "incoming_shadow_symbol",

        "entry_time":
            "incoming_shadow_entry_time",

        "exit_time":
            "incoming_shadow_exit_time",

        "exit_reason":
            "incoming_shadow_exit_reason",

        "net_r":
            "incoming_shadow_net_r",

        "holding_hours":
            "incoming_shadow_holding_hours",
    }
)


shadow_join[
    "incoming_shadow_entry_time"
] = pd.to_datetime(
    shadow_join[
        "incoming_shadow_entry_time"
    ],
    utc=True,
    errors="raise",
)

shadow_join[
    "incoming_shadow_exit_time"
] = pd.to_datetime(
    shadow_join[
        "incoming_shadow_exit_time"
    ],
    utc=True,
    errors="raise",
)


joined = snapshots.merge(
    shadow_join,
    on=[
        "audit_window",
        "incoming_candidate_id",
    ],
    how="left",
    validate="many_to_one",
    indicator="_incoming_join",
)


incoming_join_failures = int(
    (
        joined["_incoming_join"]
        != "both"
    ).sum()
)

if incoming_join_failures != 0:
    raise RuntimeError(
        "INCOMING_SHADOW_JOIN_FAILURES: "
        f"{incoming_join_failures}"
    )


symbol_mismatches = int(
    (
        joined["incoming_symbol"]
        != joined["incoming_shadow_symbol"]
    ).sum()
)

if symbol_mismatches != 0:
    raise RuntimeError(
        "INCOMING_SYMBOL_MISMATCHES: "
        f"{symbol_mismatches}"
    )


trade_join = canonical_trades[
    [
        "audit_window",
        "symbol",
        "entry_time",
        "exit_time",
        "exit_reason",
        "net_pnl_eur",
        "net_r",
        "holding_hours",
    ]
].rename(
    columns={
        "symbol":
            "incumbent_symbol",

        "entry_time":
            "incumbent_entry_time",

        "exit_time":
            "incumbent_exit_time",

        "exit_reason":
            "incumbent_exit_reason",

        "net_pnl_eur":
            "incumbent_total_net_pnl_eur",

        "net_r":
            "incumbent_total_net_r",

        "holding_hours":
            "incumbent_total_holding_hours",
    }
)


joined = joined.merge(
    trade_join,
    left_on=[
        "audit_window",
        "existing_symbol",
        "existing_entry_time",
    ],
    right_on=[
        "audit_window",
        "incumbent_symbol",
        "incumbent_entry_time",
    ],
    how="left",
    validate="many_to_one",
    indicator="_incumbent_join",
)


incumbent_join_failures = int(
    (
        joined["_incumbent_join"]
        != "both"
    ).sum()
)

if incumbent_join_failures != 0:
    failed = joined[
        joined["_incumbent_join"]
        != "both"
    ][
        [
            "audit_window",
            "existing_symbol",
            "existing_entry_time",
            "incoming_candidate_id",
        ]
    ]

    raise RuntimeError(
        "INCUMBENT_TRADE_JOIN_FAILURES: "
        f"count={incumbent_join_failures}, "
        f"examples="
        f"{failed.head(10).to_dict('records')}"
    )


if len(joined) != EXPECTED_SNAPSHOT_ROWS:
    raise RuntimeError(
        "JOINED_ROW_COUNT_MISMATCH: "
        f"expected={EXPECTED_SNAPSHOT_ROWS}, "
        f"actual={len(joined)}"
    )


invalid_event_before_entry = int(
    (
        joined["event_time"]
        < joined["incumbent_entry_time"]
    ).sum()
)

if invalid_event_before_entry != 0:
    raise RuntimeError(
        "EVENT_BEFORE_INCUMBENT_ENTRY: "
        f"{invalid_event_before_entry}"
    )


invalid_event_after_exit = int(
    (
        joined["event_time"]
        > joined["incumbent_exit_time"]
    ).sum()
)

if invalid_event_after_exit != 0:
    raise RuntimeError(
        "EVENT_AFTER_INCUMBENT_EXIT: "
        f"{invalid_event_after_exit}"
    )


joined["incumbent_future_incremental_r"] = (
    pd.to_numeric(
        joined["incumbent_total_net_r"],
        errors="raise",
    )
    - pd.to_numeric(
        joined["existing_exit_now_net_r"],
        errors="raise",
    )
)


joined[
    "incoming_minus_incumbent_future_r_diagnostic"
] = (
    pd.to_numeric(
        joined["incoming_shadow_net_r"],
        errors="raise",
    )
    - joined[
        "incumbent_future_incremental_r"
    ]
)


joined["incoming_positive_r"] = (
    joined["incoming_shadow_net_r"]
    > 0
)

joined["incoming_ge_1r"] = (
    joined["incoming_shadow_net_r"]
    >= 1.0
)

joined["incoming_ge_2r"] = (
    joined["incoming_shadow_net_r"]
    >= 2.0
)

joined["incumbent_future_negative_r"] = (
    joined[
        "incumbent_future_incremental_r"
    ]
    < 0
)


event_rows: list[
    dict[str, object]
] = []


event_group_columns = [
    "audit_window",
    "incoming_candidate_id",
]


for (
    audit_window,
    incoming_candidate_id,
), group in joined.groupby(
    event_group_columns,
    sort=True,
    dropna=False,
):

    if len(group) != 2:
        raise RuntimeError(
            "EVENT_INCUMBENT_COUNT_MISMATCH: "
            f"window={audit_window}, "
            f"candidate={incoming_candidate_id}, "
            f"rows={len(group)}"
        )


    constant_fields = [
        "event_time",
        "incoming_symbol",
        "incoming_candidate_rank",
        "incoming_due_candidate_count",
        "incoming_breakout_strength",
        "incoming_shadow_net_r",
        "incoming_shadow_exit_time",
        "incoming_shadow_exit_reason",
    ]

    for column in constant_fields:

        if group[column].nunique(
            dropna=False
        ) != 1:

            raise RuntimeError(
                "EVENT_CONSTANT_FIELD_MISMATCH: "
                f"window={audit_window}, "
                f"candidate={incoming_candidate_id}, "
                f"column={column}"
            )


    ordered = group.sort_values(
        [
            "incumbent_future_incremental_r",
            "existing_symbol",
        ],
        ascending=[
            True,
            True,
        ],
    )


    oracle_worst = ordered.iloc[0]
    oracle_best = ordered.iloc[-1]

    incoming_r = float(
        group[
            "incoming_shadow_net_r"
        ].iloc[0]
    )

    worst_future_r = float(
        oracle_worst[
            "incumbent_future_incremental_r"
        ]
    )

    best_future_r = float(
        oracle_best[
            "incumbent_future_incremental_r"
        ]
    )


    event_rows.append(
        {
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

            "incoming_due_candidate_count":
                int(
                    group[
                        "incoming_due_candidate_count"
                    ].iloc[0]
                ),

            "incoming_breakout_strength":
                float(
                    group[
                        "incoming_breakout_strength"
                    ].iloc[0]
                ),

            "incoming_shadow_net_r":
                incoming_r,

            "incoming_shadow_exit_time":
                group[
                    "incoming_shadow_exit_time"
                ].iloc[0],

            "incoming_shadow_exit_reason":
                group[
                    "incoming_shadow_exit_reason"
                ].iloc[0],

            "oracle_worst_incumbent_symbol":
                oracle_worst[
                    "existing_symbol"
                ],

            "oracle_worst_incumbent_future_r":
                worst_future_r,

            "oracle_best_incumbent_symbol":
                oracle_best[
                    "existing_symbol"
                ],

            "oracle_best_incumbent_future_r":
                best_future_r,

            "oracle_replacement_edge_r_diagnostic":
                (
                    incoming_r
                    - worst_future_r
                ),

            "incoming_beats_at_least_one_incumbent":
                bool(
                    incoming_r
                    > worst_future_r
                ),

            "incoming_beats_both_incumbents":
                bool(
                    incoming_r
                    > best_future_r
                ),

            "incoming_positive_and_any_incumbent_future_negative":
                bool(
                    incoming_r > 0
                    and (
                        group[
                            "incumbent_future_incremental_r"
                        ]
                        < 0
                    ).any()
                ),

            "both_incumbents_future_negative":
                bool(
                    (
                        group[
                            "incumbent_future_incremental_r"
                        ]
                        < 0
                    ).all()
                ),

            "incoming_positive_r":
                bool(
                    incoming_r > 0
                ),

            "incoming_ge_1r":
                bool(
                    incoming_r >= 1.0
                ),

            "incoming_ge_2r":
                bool(
                    incoming_r >= 2.0
                ),
        }
    )


events = pd.DataFrame(
    event_rows
)


if len(events) != EXPECTED_EVENTS:
    raise RuntimeError(
        "EVENT_OUTPUT_COUNT_MISMATCH: "
        f"expected={EXPECTED_EVENTS}, "
        f"actual={len(events)}"
    )


window_summary = (
    events
    .groupby(
        "audit_window",
        sort=True,
    )
    .agg(
        events=(
            "incoming_candidate_id",
            "size",
        ),

        incoming_positive=(
            "incoming_positive_r",
            "sum",
        ),

        incoming_ge_1r=(
            "incoming_ge_1r",
            "sum",
        ),

        incoming_ge_2r=(
            "incoming_ge_2r",
            "sum",
        ),

        beats_at_least_one=(
            "incoming_beats_at_least_one_incumbent",
            "sum",
        ),

        beats_both=(
            "incoming_beats_both_incumbents",
            "sum",
        ),

        positive_with_negative_incumbent=(
            "incoming_positive_and_any_incumbent_future_negative",
            "sum",
        ),

        both_incumbents_future_negative=(
            "both_incumbents_future_negative",
            "sum",
        ),

        mean_incoming_shadow_r=(
            "incoming_shadow_net_r",
            "mean",
        ),

        median_incoming_shadow_r=(
            "incoming_shadow_net_r",
            "median",
        ),

        mean_oracle_edge_r_diagnostic=(
            "oracle_replacement_edge_r_diagnostic",
            "mean",
        ),

        median_oracle_edge_r_diagnostic=(
            "oracle_replacement_edge_r_diagnostic",
            "median",
        ),
    )
    .reset_index()
)


overall_summary = pd.DataFrame(
    [
        {
            "events":
                len(events),

            "incoming_positive":
                int(
                    events[
                        "incoming_positive_r"
                    ].sum()
                ),

            "incoming_ge_1r":
                int(
                    events[
                        "incoming_ge_1r"
                    ].sum()
                ),

            "incoming_ge_2r":
                int(
                    events[
                        "incoming_ge_2r"
                    ].sum()
                ),

            "beats_at_least_one":
                int(
                    events[
                        "incoming_beats_at_least_one_incumbent"
                    ].sum()
                ),

            "beats_both":
                int(
                    events[
                        "incoming_beats_both_incumbents"
                    ].sum()
                ),

            "positive_with_negative_incumbent":
                int(
                    events[
                        "incoming_positive_and_any_incumbent_future_negative"
                    ].sum()
                ),

            "both_incumbents_future_negative":
                int(
                    events[
                        "both_incumbents_future_negative"
                    ].sum()
                ),

            "mean_incoming_shadow_r":
                float(
                    events[
                        "incoming_shadow_net_r"
                    ].mean()
                ),

            "median_incoming_shadow_r":
                float(
                    events[
                        "incoming_shadow_net_r"
                    ].median()
                ),

            "mean_oracle_edge_r_diagnostic":
                float(
                    events[
                        "oracle_replacement_edge_r_diagnostic"
                    ].mean()
                ),

            "median_oracle_edge_r_diagnostic":
                float(
                    events[
                        "oracle_replacement_edge_r_diagnostic"
                    ].median()
                ),
        }
    ]
)


joined_output = (
    REPORT_DIR
    / "v3kw_replacement_outcome_join_detail.csv"
)

event_output = (
    REPORT_DIR
    / "v3kw_replacement_outcome_join_events.csv"
)

window_output = (
    REPORT_DIR
    / "v3kw_replacement_outcome_join_by_window.csv"
)

overall_output = (
    REPORT_DIR
    / "v3kw_replacement_outcome_join_overall.csv"
)

summary_output = (
    REPORT_DIR
    / "v3kw_replacement_outcome_join_summary.txt"
)


joined.drop(
    columns=[
        "_incoming_join",
        "_incumbent_join",
    ]
).to_csv(
    joined_output,
    index=False,
)

events.to_csv(
    event_output,
    index=False,
)

window_summary.to_csv(
    window_output,
    index=False,
)

overall_summary.to_csv(
    overall_output,
    index=False,
)


lines: list[str] = []


def emit(
    text: str = "",
) -> None:

    print(text)
    lines.append(text)


emit("")
emit("=" * 116)
emit("V3KW REPLACEMENT OUTCOME JOIN")
emit("=" * 116)
emit(
    "Historical inspected windows only; "
    "not untouched out-of-sample."
)
emit(
    "No replacement action, portfolio mutation "
    "or risk-rule change occurred."
)
emit(
    "Incoming-versus-incumbent R comparisons "
    "are diagnostics, not feasible portfolio P&L."
)
emit("")

emit(
    f"Incoming MAX_POSITIONS events: "
    f"{len(events)}"
)

emit(
    f"Incumbent outcome rows:        "
    f"{len(joined)}"
)

emit(
    f"Incoming join failures:        "
    f"{incoming_join_failures}"
)

emit(
    f"Incumbent join failures:       "
    f"{incumbent_join_failures}"
)

emit("")

emit("-" * 116)
emit("OUTCOME JOIN BY WINDOW")
emit("-" * 116)

emit(
    window_summary.to_string(
        index=False,
        float_format=lambda value: (
            f"{value:.4f}"
        ),
    )
)

emit("")

emit("-" * 116)
emit("OVERALL DIAGNOSTIC")
emit("-" * 116)

emit(
    overall_summary.to_string(
        index=False,
        float_format=lambda value: (
            f"{value:.4f}"
        ),
    )
)

emit("")

emit("-" * 116)
emit("MASTER OBJECTIVE CHECK")
emit("-" * 116)

emit(
    "1. Loss/drawdown improvement: "
    "NOT YET MEASURED; no replacement replay occurred."
)

emit(
    "2. Upside preservation: incoming candidate outcomes "
    "were joined to incumbent future outcomes."
)

emit(
    "3. Stability: joined results are separated "
    "across all four inspected windows."
)

emit(
    "4. EUR 200 weekly-average target: "
    "NOT measurable from overlapping R diagnostics."
)

emit(
    "Next gate: test causal weakest-position selectors "
    "against the hindsight oracle before any portfolio replay."
)

emit("")
emit(
    "V3KW replacement outcome join: PASS"
)


summary_output.write_text(
    "\n".join(lines) + "\n",
    encoding="utf-8",
)
