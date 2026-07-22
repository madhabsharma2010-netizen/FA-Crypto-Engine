from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config.risk_settings import RISK_SETTINGS
import run_v3d_diagnostics as diagnostics


PERIODS = {
    "DEVELOPMENT": {
        "start": pd.Timestamp("2025-08-01 00:00:00"),
        "end": pd.Timestamp("2026-04-30 00:00:00"),
        "trade_file": Path(
            "logs/backtests/v3e6_development_trade_history.csv"
        ),
    },
    "VALIDATION": {
        "start": pd.Timestamp("2026-05-01 00:00:00"),
        "end": pd.Timestamp("2026-07-20 00:00:00"),
        "trade_file": Path(
            "logs/backtests/v3e6_validation_trade_history.csv"
        ),
    },
}

FORWARD_HOURS = 48

DETAIL_FILE = Path(
    "reports/v3e8_trade_path_detail.csv"
)

SUMMARY_FILE = Path(
    "reports/v3e8_trade_path_summary.csv"
)

OUTCOME_FILE = Path(
    "reports/v3e8_trade_path_outcomes.csv"
)

EXIT_FILE = Path(
    "reports/v3e8_post_exit_continuation.csv"
)

FEE_BUFFER_PERCENT = float(
    RISK_SETTINGS.trading_fee_percent
    + RISK_SETTINGS.estimated_slippage_percent
)


def first_hit_time(
    frame: pd.DataFrame,
    target_price: float,
    stop_price: float,
) -> tuple[
    pd.Timestamp | None,
    pd.Timestamp | None,
    str,
]:
    target_hits = frame[
        frame["high"] >= target_price
    ]

    stop_hits = frame[
        frame["low"] <= stop_price
    ]

    target_time = (
        pd.Timestamp(target_hits.index[0])
        if not target_hits.empty
        else None
    )

    stop_time = (
        pd.Timestamp(stop_hits.index[0])
        if not stop_hits.empty
        else None
    )

    if (
        target_time is not None
        and stop_time is not None
        and target_time == stop_time
    ):
        outcome = "AMBIGUOUS"

    elif (
        target_time is not None
        and (
            stop_time is None
            or target_time < stop_time
        )
    ):
        outcome = "TARGET_FIRST"

    elif stop_time is not None:
        outcome = "STOP_FIRST"

    else:
        outcome = "NEITHER"

    return (
        target_time,
        stop_time,
        outcome,
    )


def true_percent(
    series: pd.Series,
) -> float:
    if series.empty:
        return 0.0

    return float(
        series.fillna(False).mean()
        * 100.0
    )


def profit_factor(
    profits: pd.Series,
) -> float:
    gross_profit = float(
        profits[
            profits > 0
        ].sum()
    )

    gross_loss = abs(
        float(
            profits[
                profits < 0
            ].sum()
        )
    )

    if gross_loss > 0:
        return gross_profit / gross_loss

    if gross_profit > 0:
        return float("inf")

    return 0.0


def load_15m(
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    diagnostics.BACKTEST_START = start
    diagnostics.BACKTEST_END = end

    frame = diagnostics.prepare_15m(
        symbol
    ).copy()

    return frame.sort_index()


def classify_trade(
    row: dict[str, Any],
) -> str:
    if row["hit_2r_before_stop_48h"]:
        return "REACHED_2R"

    if row["hit_1r_then_revisit_entry"]:
        return "ONE_R_BREAKEVEN_TRAP"

    if row["stop_before_0_5r_48h"]:
        return "FAILED_BEFORE_0_5R"

    if row["hit_1r_before_stop_48h"]:
        return "REACHED_1R_NOT_2R"

    if row["hit_0_5r_before_stop_48h"]:
        return "REACHED_0_5R_NOT_1R"

    if row["post_exit_additional_mfe_r"] >= 1.0:
        return "EXITED_BEFORE_LATER_MOVE"

    return "NO_EDGE_VISIBLE"


def analyze_period(
    period: str,
    config: dict[str, Any],
) -> pd.DataFrame:
    trade_file: Path = config[
        "trade_file"
    ]

    if not trade_file.exists():
        raise FileNotFoundError(
            f"Trade history not found: {trade_file}"
        )

    trades = pd.read_csv(
        trade_file
    )

    if trades.empty:
        raise ValueError(
            f"{period} trade history is empty."
        )

    trades["entry_time"] = pd.to_datetime(
        trades["entry_time"],
        errors="coerce",
    )

    trades["exit_time"] = pd.to_datetime(
        trades["exit_time"],
        errors="coerce",
    )

    numeric_columns = [
        "entry_price",
        "exit_price",
        "initial_stop",
        "initial_quantity",
        "profit",
        "buy_fee",
        "sell_fees",
    ]

    for column in numeric_columns:
        trades[column] = pd.to_numeric(
            trades.get(column, 0.0),
            errors="coerce",
        ).fillna(0.0)

    frames = {}

    for symbol in sorted(
        trades["symbol"].unique()
    ):
        print(
            f"Loading {period} 15M path data: {symbol}"
        )

        frames[symbol] = load_15m(
            symbol=symbol,
            start=config["start"],
            end=config["end"],
        )

    rows = []

    for trade in trades.to_dict(
        orient="records"
    ):
        symbol = str(
            trade["symbol"]
        )

        entry_time = pd.Timestamp(
            trade["entry_time"]
        )

        exit_time = pd.Timestamp(
            trade["exit_time"]
        )

        entry_price = float(
            trade["entry_price"]
        )

        exit_price = float(
            trade["exit_price"]
        )

        initial_stop = float(
            trade["initial_stop"]
        )

        initial_quantity = float(
            trade["initial_quantity"]
        )

        risk_per_unit = (
            entry_price
            - initial_stop
        )

        if risk_per_unit <= 0:
            continue

        initial_risk_eur = (
            risk_per_unit
            * initial_quantity
        )

        frame = frames[symbol]

        current_path = frame[
            (
                frame.index
                > entry_time
            )
            & (
                frame.index
                <= exit_time
            )
        ]

        horizon_end = min(
            entry_time
            + pd.Timedelta(
                hours=FORWARD_HOURS
            ),
            config["end"],
        )

        forward_path = frame[
            (
                frame.index
                > entry_time
            )
            & (
                frame.index
                <= horizon_end
            )
        ]

        post_exit_path = frame[
            (
                frame.index
                > exit_time
            )
            & (
                frame.index
                <= min(
                    exit_time
                    + pd.Timedelta(
                        hours=24
                    ),
                    config["end"],
                )
            )
        ]

        if forward_path.empty:
            continue

        current_mfe_r = (
            (
                float(
                    current_path[
                        "high"
                    ].max()
                )
                - entry_price
            )
            / risk_per_unit
            if not current_path.empty
            else 0.0
        )

        current_mae_r = (
            (
                entry_price
                - float(
                    current_path[
                        "low"
                    ].min()
                )
            )
            / risk_per_unit
            if not current_path.empty
            else 0.0
        )

        forward_mfe_r = (
            (
                float(
                    forward_path[
                        "high"
                    ].max()
                )
                - entry_price
            )
            / risk_per_unit
        )

        forward_mae_r = (
            (
                entry_price
                - float(
                    forward_path[
                        "low"
                    ].min()
                )
            )
            / risk_per_unit
        )

        post_exit_additional_mfe_r = (
            (
                float(
                    post_exit_path[
                        "high"
                    ].max()
                )
                - exit_price
            )
            / risk_per_unit
            if not post_exit_path.empty
            else 0.0
        )

        row = {
            **trade,
            "period": period,
            "risk_per_unit": (
                risk_per_unit
            ),
            "initial_risk_eur": (
                initial_risk_eur
            ),
            "realized_r": (
                float(trade["profit"])
                / initial_risk_eur
                if initial_risk_eur > 0
                else 0.0
            ),
            "hold_hours": (
                (
                    exit_time
                    - entry_time
                ).total_seconds()
                / 3600.0
            ),
            "current_mfe_r": (
                current_mfe_r
            ),
            "current_mae_r": (
                current_mae_r
            ),
            "forward_48h_mfe_r": (
                forward_mfe_r
            ),
            "forward_48h_mae_r": (
                forward_mae_r
            ),
            "post_exit_additional_mfe_r": (
                post_exit_additional_mfe_r
            ),
        }

        level_results = {}

        for label, multiple in [
            ("0_5r", 0.5),
            ("1r", 1.0),
            ("1_5r", 1.5),
            ("2r", 2.0),
            ("3r", 3.0),
        ]:
            target_price = (
                entry_price
                + risk_per_unit
                * multiple
            )

            (
                target_time,
                stop_time,
                outcome,
            ) = first_hit_time(
                forward_path,
                target_price,
                initial_stop,
            )

            level_results[
                f"{label}_target_time"
            ] = target_time

            level_results[
                f"{label}_stop_time"
            ] = stop_time

            level_results[
                f"{label}_first_outcome"
            ] = outcome

            level_results[
                f"hit_{label}_before_stop_48h"
            ] = (
                outcome
                == "TARGET_FIRST"
            )

        row.update(level_results)

        row["stop_before_0_5r_48h"] = (
            row[
                "0_5r_first_outcome"
            ]
            == "STOP_FIRST"
        )

        one_r_time = row[
            "1r_target_time"
        ]

        two_r_time = row[
            "2r_target_time"
        ]

        fee_buffer_price = (
            entry_price
            * (
                1.0
                + FEE_BUFFER_PERCENT
                / 100.0
            )
        )

        row[
            "hit_1r_then_revisit_entry"
        ] = False

        if (
            one_r_time is not None
            and (
                two_r_time is None
                or one_r_time
                < two_r_time
            )
        ):
            revisit_end = (
                two_r_time
                if two_r_time
                is not None
                else horizon_end
            )

            revisit_path = frame[
                (
                    frame.index
                    > one_r_time
                )
                & (
                    frame.index
                    <= revisit_end
                )
            ]

            if (
                not revisit_path.empty
                and float(
                    revisit_path[
                        "low"
                    ].min()
                )
                <= fee_buffer_price
            ):
                row[
                    "hit_1r_then_revisit_entry"
                ] = True

        row["current_exit_before_2r"] = (
            current_mfe_r < 2.0
        )

        row["future_reached_2r_after_exit"] = (
            current_mfe_r < 2.0
            and forward_mfe_r >= 2.0
        )

        row["path_classification"] = (
            classify_trade(row)
        )

        rows.append(row)

    return pd.DataFrame(rows)


def summarize_group(
    group: pd.DataFrame,
) -> dict[str, Any]:
    profits = pd.to_numeric(
        group["profit"],
        errors="coerce",
    ).fillna(0.0)

    return {
        "trades": len(group),
        "win_rate": true_percent(
            profits > 0
        ),
        "net_profit": float(
            profits.sum()
        ),
        "profit_factor": (
            profit_factor(
                profits
            )
        ),
        "hit_0_5r_before_stop_percent": (
            true_percent(
                group[
                    "hit_0_5r_before_stop_48h"
                ]
            )
        ),
        "hit_1r_before_stop_percent": (
            true_percent(
                group[
                    "hit_1r_before_stop_48h"
                ]
            )
        ),
        "hit_1_5r_before_stop_percent": (
            true_percent(
                group[
                    "hit_1_5r_before_stop_48h"
                ]
            )
        ),
        "hit_2r_before_stop_percent": (
            true_percent(
                group[
                    "hit_2r_before_stop_48h"
                ]
            )
        ),
        "hit_3r_before_stop_percent": (
            true_percent(
                group[
                    "hit_3r_before_stop_48h"
                ]
            )
        ),
        "stop_before_0_5r_percent": (
            true_percent(
                group[
                    "stop_before_0_5r_48h"
                ]
            )
        ),
        "one_r_breakeven_trap_percent": (
            true_percent(
                group[
                    "hit_1r_then_revisit_entry"
                ]
            )
        ),
        "future_2r_after_exit_percent": (
            true_percent(
                group[
                    "future_reached_2r_after_exit"
                ]
            )
        ),
        "median_current_mfe_r": float(
            group[
                "current_mfe_r"
            ].median()
        ),
        "median_forward_48h_mfe_r": float(
            group[
                "forward_48h_mfe_r"
            ].median()
        ),
        "median_forward_48h_mae_r": float(
            group[
                "forward_48h_mae_r"
            ].median()
        ),
        "median_realized_r": float(
            group[
                "realized_r"
            ].median()
        ),
        "median_hold_hours": float(
            group[
                "hold_hours"
            ].median()
        ),
    }


def grouped_summary(
    detail: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    group_columns = [
        "period",
        "route",
        "symbol",
    ]

    for key, group in detail.groupby(
        group_columns,
        dropna=False,
        sort=False,
    ):
        row = {
            column: value
            for column, value
            in zip(
                group_columns,
                key,
            )
        }

        row.update(
            summarize_group(group)
        )

        rows.append(row)

    combined_rows = []

    for key, group in detail.groupby(
        [
            "route",
            "symbol",
        ],
        dropna=False,
        sort=False,
    ):
        route, symbol = key

        row = {
            "period": "COMBINED",
            "route": route,
            "symbol": symbol,
        }

        row.update(
            summarize_group(group)
        )

        combined_rows.append(row)

    return pd.DataFrame(
        rows
        + combined_rows
    )


def print_summary(
    summary: pd.DataFrame,
) -> None:
    print("-" * 185)
    print(
        "TRADE PATH SUMMARY — INITIAL RISK UNCHANGED"
    )
    print("-" * 185)

    print(
        f"{'PERIOD':<12} "
        f"{'ROUTE':<20} "
        f"{'SYMBOL':<10} "
        f"{'N':>4} "
        f"{'P&L':>10} "
        f"{'PF':>6} "
        f"{'0.5R':>7} "
        f"{'1R':>7} "
        f"{'1.5R':>7} "
        f"{'2R':>7} "
        f"{'STOP<.5R':>10} "
        f"{'1R TRAP':>9} "
        f"{'2R AFTER EXIT':>14} "
        f"{'MED MFE':>9} "
        f"{'MED 48H MFE':>12} "
        f"{'MED R':>8}"
    )

    print("-" * 185)

    for row in summary.to_dict(
        orient="records"
    ):
        factor = float(
            row["profit_factor"]
        )

        factor_text = (
            "INF"
            if np.isinf(factor)
            else f"{factor:.2f}"
        )

        print(
            f"{str(row['period']):<12} "
            f"{str(row['route']):<20} "
            f"{str(row['symbol']):<10} "
            f"{int(row['trades']):>4} "
            f"€{float(row['net_profit']):>8.2f} "
            f"{factor_text:>6} "
            f"{float(row['hit_0_5r_before_stop_percent']):>6.1f}% "
            f"{float(row['hit_1r_before_stop_percent']):>6.1f}% "
            f"{float(row['hit_1_5r_before_stop_percent']):>6.1f}% "
            f"{float(row['hit_2r_before_stop_percent']):>6.1f}% "
            f"{float(row['stop_before_0_5r_percent']):>9.1f}% "
            f"{float(row['one_r_breakeven_trap_percent']):>8.1f}% "
            f"{float(row['future_2r_after_exit_percent']):>13.1f}% "
            f"{float(row['median_current_mfe_r']):>8.2f} "
            f"{float(row['median_forward_48h_mfe_r']):>11.2f} "
            f"{float(row['median_realized_r']):>7.2f}"
        )


def main() -> None:
    print("=" * 185)
    print(
        "FA CRYPTO ENGINE — V3E8 ACTUAL TRADE PATH DIAGNOSTICS"
    )
    print("=" * 185)
    print(
        "Read-only study. V3D3 surveillance, position risk, loss limits and stops are not modified."
    )

    frames = []

    for period, config in PERIODS.items():
        frames.append(
            analyze_period(
                period,
                config,
            )
        )

    detail = pd.concat(
        frames,
        ignore_index=True,
    )

    DETAIL_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    detail.to_csv(
        DETAIL_FILE,
        index=False,
    )

    summary = grouped_summary(
        detail
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    outcomes = (
        detail.groupby(
            [
                "period",
                "route",
                "symbol",
                "path_classification",
            ],
            dropna=False,
        )
        .size()
        .reset_index(
            name="trades"
        )
    )

    outcomes.to_csv(
        OUTCOME_FILE,
        index=False,
    )

    exit_analysis = detail[
        [
            "period",
            "route",
            "symbol",
            "entry_time",
            "exit_time",
            "exit_reason",
            "profit",
            "current_mfe_r",
            "forward_48h_mfe_r",
            "post_exit_additional_mfe_r",
            "future_reached_2r_after_exit",
        ]
    ].copy()

    exit_analysis.to_csv(
        EXIT_FILE,
        index=False,
    )

    print_summary(
        summary
    )

    print("-" * 185)
    print(
        "PATH CLASSIFICATION COUNTS"
    )
    print("-" * 185)

    for row in outcomes.to_dict(
        orient="records"
    ):
        print(
            f"{row['period']:<12} | "
            f"{row['route']:<20} | "
            f"{row['symbol']:<10} | "
            f"{row['path_classification']:<30} | "
            f"{int(row['trades']):>3}"
        )

    print("=" * 185)
    print(
        f"Trade path detail      : {DETAIL_FILE}"
    )
    print(
        f"Trade path summary     : {SUMMARY_FILE}"
    )
    print(
        f"Path classifications   : {OUTCOME_FILE}"
    )
    print(
        f"Post-exit continuation : {EXIT_FILE}"
    )
    print("=" * 185)


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        print()
        print(
            "V3E8 diagnostics stopped manually."
        )

    except Exception as error:
        print()
        print("=" * 185)
        print(
            "V3E8 DIAGNOSTIC ERROR"
        )
        print("=" * 185)
        print(
            f"{type(error).__name__}: {error}"
        )
        print("=" * 185)
        raise
