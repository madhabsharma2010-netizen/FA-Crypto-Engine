from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


REPORT_DIR = Path("reports")

INPUT_PATH = (
    REPORT_DIR
    / "v3kx_trade_quality_combined_source.csv"
)

EXPECTED_TRADES = 139
EXPECTED_NET_R = 24.453676
EXPECTED_NET_PNL = 959.347931

WINDOW_ORDER = [
    "2022",
    "2023",
    "2024",
    "2025H1",
]


if not INPUT_PATH.exists():
    raise RuntimeError(
        f"V3KX_SOURCE_MISSING: {INPUT_PATH}"
    )


trades = pd.read_csv(
    INPUT_PATH
)


required_columns = {
    "audit_window",
    "symbol",
    "signal_time",
    "entry_time",
    "exit_time",
    "entry_price",
    "initial_stop",
    "final_stop",
    "quantity",
    "entry_notional",
    "buy_fee",
    "sell_fee",
    "initial_risk_eur",
    "account_risk_percent",
    "breakout_strength",
    "exit_reason",
    "net_pnl_eur",
    "net_r",
    "holding_hours",
}

missing_columns = sorted(
    required_columns
    - set(trades.columns)
)

if missing_columns:
    raise RuntimeError(
        "V3KX_REQUIRED_COLUMNS_MISSING: "
        f"{missing_columns}"
    )


if len(trades) != EXPECTED_TRADES:
    raise RuntimeError(
        "V3KX_TRADE_COUNT_MISMATCH: "
        f"expected={EXPECTED_TRADES}, "
        f"actual={len(trades)}"
    )


for column in [
    "signal_time",
    "entry_time",
    "exit_time",
]:
    trades[column] = pd.to_datetime(
        trades[column],
        utc=True,
        errors="raise",
    )


numeric_columns = [
    "entry_price",
    "initial_stop",
    "final_stop",
    "quantity",
    "entry_notional",
    "buy_fee",
    "sell_fee",
    "initial_risk_eur",
    "account_risk_percent",
    "breakout_strength",
    "net_pnl_eur",
    "net_r",
    "holding_hours",
]

for column in numeric_columns:
    trades[column] = pd.to_numeric(
        trades[column],
        errors="raise",
    )


duplicate_keys = int(
    trades.duplicated(
        subset=[
            "audit_window",
            "symbol",
            "entry_time",
        ]
    ).sum()
)

if duplicate_keys != 0:
    raise RuntimeError(
        "V3KX_DUPLICATE_TRADE_KEYS: "
        f"{duplicate_keys}"
    )


negative_holding_rows = int(
    (
        trades["holding_hours"] < 0
    ).sum()
)

if negative_holding_rows != 0:
    raise RuntimeError(
        "V3KX_NEGATIVE_HOLDING_ROWS: "
        f"{negative_holding_rows}"
    )


total_net_r = float(
    trades["net_r"].sum()
)

total_net_pnl = float(
    trades["net_pnl_eur"].sum()
)

if not np.isclose(
    total_net_r,
    EXPECTED_NET_R,
    atol=1e-6,
    rtol=0.0,
):
    raise RuntimeError(
        "V3KX_NET_R_PARITY_FAILED: "
        f"expected={EXPECTED_NET_R}, "
        f"actual={total_net_r}"
    )

if not np.isclose(
    total_net_pnl,
    EXPECTED_NET_PNL,
    atol=1e-6,
    rtol=0.0,
):
    raise RuntimeError(
        "V3KX_NET_PNL_PARITY_FAILED: "
        f"expected={EXPECTED_NET_PNL}, "
        f"actual={total_net_pnl}"
    )


trades["holding_bucket"] = pd.cut(
    trades["holding_hours"],
    bins=[
        -0.000001,
        24.0,
        48.0,
        96.0,
        168.0,
        float("inf"),
    ],
    labels=[
        "00-24H",
        "24-48H",
        "48-96H",
        "96-168H",
        "168H+",
    ],
    right=False,
)


trades["strength_percentile_in_window"] = (
    trades
    .groupby(
        "audit_window"
    )[
        "breakout_strength"
    ]
    .rank(
        method="average",
        pct=True,
    )
)


trades["strength_quartile_descriptive"] = pd.cut(
    trades[
        "strength_percentile_in_window"
    ],
    bins=[
        0.0,
        0.25,
        0.50,
        0.75,
        1.000001,
    ],
    labels=[
        "Q1_LOW",
        "Q2",
        "Q3",
        "Q4_HIGH",
    ],
    include_lowest=True,
)


def profit_factor(
    frame: pd.DataFrame,
) -> float:

    gross_profit = float(
        frame.loc[
            frame["net_r"] > 0,
            "net_r",
        ].sum()
    )

    gross_loss = float(
        -frame.loc[
            frame["net_r"] < 0,
            "net_r",
        ].sum()
    )

    if gross_loss == 0:
        return (
            float("inf")
            if gross_profit > 0
            else 0.0
        )

    return (
        gross_profit
        / gross_loss
    )


def summarize_frame(
    frame: pd.DataFrame,
) -> dict[str, object]:

    winners = frame[
        frame["net_r"] > 0
    ]

    losers = frame[
        frame["net_r"] < 0
    ]

    return {
        "trades":
            int(
                len(frame)
            ),

        "wins":
            int(
                len(winners)
            ),

        "losses":
            int(
                len(losers)
            ),

        "win_rate_pct":
            float(
                len(winners)
                / len(frame)
                * 100.0
                if len(frame)
                else 0.0
            ),

        "total_net_pnl_eur":
            float(
                frame[
                    "net_pnl_eur"
                ].sum()
            ),

        "total_net_r":
            float(
                frame[
                    "net_r"
                ].sum()
            ),

        "mean_net_r":
            float(
                frame[
                    "net_r"
                ].mean()
            ),

        "median_net_r":
            float(
                frame[
                    "net_r"
                ].median()
            ),

        "profit_factor":
            profit_factor(
                frame
            ),

        "average_winner_r":
            float(
                winners[
                    "net_r"
                ].mean()
                if not winners.empty
                else 0.0
            ),

        "average_loser_r":
            float(
                losers[
                    "net_r"
                ].mean()
                if not losers.empty
                else 0.0
            ),

        "best_trade_r":
            float(
                frame[
                    "net_r"
                ].max()
            ),

        "worst_trade_r":
            float(
                frame[
                    "net_r"
                ].min()
            ),

        "median_holding_hours":
            float(
                frame[
                    "holding_hours"
                ].median()
            ),

        "capped_3r_mean":
            float(
                frame[
                    "net_r"
                ]
                .clip(
                    lower=-3.0,
                    upper=3.0,
                )
                .mean()
            ),
    }


def grouped_summary(
    group_columns: list[str],
) -> pd.DataFrame:

    rows: list[
        dict[str, object]
    ] = []

    for keys, frame in trades.groupby(
        group_columns,
        sort=True,
        dropna=False,
        observed=False,
    ):

        if not isinstance(
            keys,
            tuple,
        ):
            keys = (
                keys,
            )

        row = {
            column: value
            for column, value
            in zip(
                group_columns,
                keys,
            )
        }

        row.update(
            summarize_frame(
                frame
            )
        )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


overall = pd.DataFrame(
    [
        summarize_frame(
            trades
        )
    ]
)

by_window = grouped_summary(
    [
        "audit_window",
    ]
)

by_window[
    "audit_window"
] = pd.Categorical(
    by_window[
        "audit_window"
    ],
    categories=WINDOW_ORDER,
    ordered=True,
)

by_window = by_window.sort_values(
    "audit_window"
)


by_symbol = grouped_summary(
    [
        "symbol",
    ]
).sort_values(
    "total_net_r",
    ascending=False,
)


by_symbol_window = grouped_summary(
    [
        "symbol",
        "audit_window",
    ]
)


by_exit_reason = grouped_summary(
    [
        "exit_reason",
    ]
).sort_values(
    "total_net_r",
    ascending=False,
)


by_holding = grouped_summary(
    [
        "holding_bucket",
    ]
)


by_strength = grouped_summary(
    [
        "strength_quartile_descriptive",
    ]
)


symbol_stability_rows: list[
    dict[str, object]
] = []

for symbol, frame in by_symbol_window.groupby(
    "symbol",
    sort=True,
):

    symbol_stability_rows.append(
        {
            "symbol":
                symbol,

            "windows_present":
                int(
                    len(frame)
                ),

            "positive_windows":
                int(
                    (
                        frame[
                            "total_net_r"
                        ] > 0
                    ).sum()
                ),

            "negative_windows":
                int(
                    (
                        frame[
                            "total_net_r"
                        ] < 0
                    ).sum()
                ),

            "combined_trades":
                int(
                    frame[
                        "trades"
                    ].sum()
                ),

            "combined_total_net_r":
                float(
                    frame[
                        "total_net_r"
                    ].sum()
                ),

            "worst_window_net_r":
                float(
                    frame[
                        "total_net_r"
                    ].min()
                ),

            "best_window_net_r":
                float(
                    frame[
                        "total_net_r"
                    ].max()
                ),

            "minimum_window_win_rate_pct":
                float(
                    frame[
                        "win_rate_pct"
                    ].min()
                ),

            "maximum_window_win_rate_pct":
                float(
                    frame[
                        "win_rate_pct"
                    ].max()
                ),
        }
    )


symbol_stability = pd.DataFrame(
    symbol_stability_rows
).sort_values(
    [
        "positive_windows",
        "combined_total_net_r",
    ],
    ascending=[
        False,
        False,
    ],
)


positive_trades = (
    trades[
        trades["net_r"] > 0
    ]
    .sort_values(
        "net_r",
        ascending=False,
    )
)

negative_trades = (
    trades[
        trades["net_r"] < 0
    ]
    .assign(
        absolute_loss_r=lambda frame: (
            -frame["net_r"]
        )
    )
    .sort_values(
        "absolute_loss_r",
        ascending=False,
    )
)


total_positive_r = float(
    positive_trades[
        "net_r"
    ].sum()
)

total_negative_r_abs = float(
    negative_trades[
        "absolute_loss_r"
    ].sum()
)


concentration_rows: list[
    dict[str, object]
] = []

for count in [
    1,
    3,
    5,
    10,
]:

    winner_count = min(
        count,
        len(
            positive_trades
        ),
    )

    loser_count = min(
        count,
        len(
            negative_trades
        ),
    )

    top_winner_r = float(
        positive_trades
        .head(
            winner_count
        )[
            "net_r"
        ]
        .sum()
    )

    worst_loser_r_abs = float(
        negative_trades
        .head(
            loser_count
        )[
            "absolute_loss_r"
        ]
        .sum()
    )

    concentration_rows.append(
        {
            "trade_count":
                count,

            "top_winner_r":
                top_winner_r,

            "top_winner_share_pct":
                float(
                    top_winner_r
                    / total_positive_r
                    * 100.0
                    if total_positive_r
                    else 0.0
                ),

            "baseline_net_r_without_top_winners":
                float(
                    total_net_r
                    - top_winner_r
                ),

            "worst_loser_absolute_r":
                worst_loser_r_abs,

            "worst_loser_share_pct":
                float(
                    worst_loser_r_abs
                    / total_negative_r_abs
                    * 100.0
                    if total_negative_r_abs
                    else 0.0
                ),
        }
    )


concentration = pd.DataFrame(
    concentration_rows
)


output_map = {
    "v3kx_trade_quality_overall.csv":
        overall,

    "v3kx_trade_quality_by_window.csv":
        by_window,

    "v3kx_trade_quality_by_symbol.csv":
        by_symbol,

    "v3kx_trade_quality_by_symbol_window.csv":
        by_symbol_window,

    "v3kx_trade_quality_by_exit_reason.csv":
        by_exit_reason,

    "v3kx_trade_quality_by_holding.csv":
        by_holding,

    "v3kx_trade_quality_by_strength_quartile.csv":
        by_strength,

    "v3kx_trade_quality_symbol_stability.csv":
        symbol_stability,

    "v3kx_trade_quality_concentration.csv":
        concentration,
}


for filename, frame in output_map.items():

    frame.to_csv(
        REPORT_DIR
        / filename,
        index=False,
    )


lines: list[str] = []


def emit(
    text: str = "",
) -> None:

    print(text)
    lines.append(text)


emit("")
emit("=" * 130)
emit("V3KX BASELINE TRADE-QUALITY AND LOSS-CLUSTER AUDIT")
emit("=" * 130)
emit(
    "Historical inspected windows only; "
    "not untouched out-of-sample."
)
emit(
    "Descriptive audit only. No entry filter, exit change, "
    "portfolio mutation or risk change occurred."
)
emit(
    "Breakout-strength quartiles are descriptive and "
    "must not be treated as production thresholds."
)
emit("")

emit("-" * 130)
emit("OVERALL PAYOFF STRUCTURE")
emit("-" * 130)

emit(
    overall.to_string(
        index=False,
        float_format=lambda value: (
            f"{value:.4f}"
        ),
    )
)

emit("")

emit("-" * 130)
emit("BY WINDOW")
emit("-" * 130)

emit(
    by_window.to_string(
        index=False,
        float_format=lambda value: (
            f"{value:.4f}"
        ),
    )
)

emit("")

emit("-" * 130)
emit("BY SYMBOL")
emit("-" * 130)

emit(
    by_symbol.to_string(
        index=False,
        float_format=lambda value: (
            f"{value:.4f}"
        ),
    )
)

emit("")

emit("-" * 130)
emit("SYMBOL CROSS-WINDOW STABILITY")
emit("-" * 130)

emit(
    symbol_stability.to_string(
        index=False,
        float_format=lambda value: (
            f"{value:.4f}"
        ),
    )
)

emit("")

emit("-" * 130)
emit("BY EXIT REASON")
emit("-" * 130)

emit(
    by_exit_reason.to_string(
        index=False,
        float_format=lambda value: (
            f"{value:.4f}"
        ),
    )
)

emit("")

emit("-" * 130)
emit("BY HOLDING-TIME BUCKET")
emit("-" * 130)

emit(
    by_holding.to_string(
        index=False,
        float_format=lambda value: (
            f"{value:.4f}"
        ),
    )
)

emit("")

emit("-" * 130)
emit("DESCRIPTIVE BREAKOUT-STRENGTH QUARTILES")
emit("-" * 130)

emit(
    by_strength.to_string(
        index=False,
        float_format=lambda value: (
            f"{value:.4f}"
        ),
    )
)

emit("")

emit("-" * 130)
emit("WINNER AND LOSER CONCENTRATION")
emit("-" * 130)

emit(
    concentration.to_string(
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
    "1. Loss/drawdown improvement: NOT YET MEASURED; "
    "no portfolio replay occurred."
)

emit(
    "2. Upside preservation: winner concentration and "
    "baseline dependence on the largest winners were measured."
)

emit(
    "3. Stability: results were separated by window, "
    "symbol, exit reason and holding duration."
)

emit(
    "4. EUR 200 weekly-average target: NOT YET MEASURED; "
    "trade clusters are not a substitute for weekly equity replay."
)

emit(
    "Next gate: nominate at most one structural loss cluster "
    "only when it is negative across multiple windows and does "
    "not contain a material share of the largest winners."
)

emit("")
emit(
    "V3KX baseline trade-quality audit: PASS"
)


summary_path = (
    REPORT_DIR
    / "v3kx_baseline_trade_quality_audit_summary.txt"
)

summary_path.write_text(
    "\n".join(lines) + "\n",
    encoding="utf-8",
)
