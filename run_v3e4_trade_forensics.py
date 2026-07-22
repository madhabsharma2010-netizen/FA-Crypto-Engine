from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


TRADE_FILE = Path(
    "logs/backtests/v3e3_portfolio_trade_history.csv"
)

DETAIL_FILE = Path(
    "reports/v3e4_route_asset_forensics.csv"
)

ROUTE_FILE = Path(
    "reports/v3e4_route_summary.csv"
)

SYMBOL_FILE = Path(
    "reports/v3e4_symbol_summary.csv"
)

EXIT_FILE = Path(
    "reports/v3e4_exit_reason_summary.csv"
)

SCORE_FILE = Path(
    "reports/v3e4_score_bucket_summary.csv"
)

RR_FILE = Path(
    "reports/v3e4_rr_bucket_summary.csv"
)


def numeric(
    frame: pd.DataFrame,
    column: str,
    default: float = 0.0,
) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(
            default,
            index=frame.index,
            dtype=float,
        )

    return pd.to_numeric(
        frame[column],
        errors="coerce",
    ).fillna(default)


def text(
    frame: pd.DataFrame,
    column: str,
    default: str,
) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(
            default,
            index=frame.index,
            dtype=str,
        )

    return (
        frame[column]
        .fillna(default)
        .astype(str)
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
        return (
            gross_profit
            / gross_loss
        )

    if gross_profit > 0:
        return float("inf")

    return 0.0


def longest_losing_streak(
    profits: pd.Series,
) -> int:
    longest = 0
    current = 0

    for value in profits:
        if value < 0:
            current += 1
            longest = max(
                longest,
                current,
            )
        else:
            current = 0

    return longest


def edge_label(
    trades: int,
    net_profit: float,
    factor: float,
    expectancy: float,
) -> str:
    if trades < 3:
        return "INSUFFICIENT_SAMPLE"

    if (
        net_profit > 0
        and factor >= 1.20
        and expectancy > 0
    ):
        return "KEEP_CANDIDATE"

    if (
        net_profit > 0
        and factor >= 1.00
    ):
        return "WATCH_POSITIVE"

    if (
        factor < 0.70
        or expectancy < -10.0
    ):
        return "REJECT"

    return "WATCH"


def summarize_group(
    group: pd.DataFrame,
) -> dict[str, float | int | str]:
    profits = group["profit"]
    fees = group["total_trade_fees"]

    trades = len(group)
    winners = int(
        (profits > 0).sum()
    )
    losers = int(
        (profits < 0).sum()
    )

    net_profit = float(
        profits.sum()
    )

    total_fees = float(
        fees.sum()
    )

    pre_fee_profit = (
        net_profit
        + total_fees
    )

    factor = profit_factor(
        profits
    )

    expectancy = (
        net_profit
        / trades
        if trades
        else 0.0
    )

    median_hold = float(
        group[
            "hold_hours"
        ].median()
    )

    average_hold = float(
        group[
            "hold_hours"
        ].mean()
    )

    return {
        "trades": trades,
        "winners": winners,
        "losers": losers,
        "win_rate": (
            winners
            / trades
            * 100.0
            if trades
            else 0.0
        ),
        "net_profit": net_profit,
        "gross_profit": float(
            profits[
                profits > 0
            ].sum()
        ),
        "gross_loss": abs(
            float(
                profits[
                    profits < 0
                ].sum()
            )
        ),
        "profit_factor": factor,
        "expectancy": expectancy,
        "total_fees": total_fees,
        "pre_fee_profit": pre_fee_profit,
        "fees_as_percent_of_absolute_net": (
            total_fees
            / abs(net_profit)
            * 100.0
            if net_profit != 0
            else 0.0
        ),
        "average_hold_hours": average_hold,
        "median_hold_hours": median_hold,
        "partial_profit_trades": int(
            group[
                "partial_profit_taken"
            ].sum()
        ),
        "stop_loss_exits": int(
            group[
                "is_stop_exit"
            ].sum()
        ),
        "longest_losing_streak": (
            longest_losing_streak(
                profits
            )
        ),
        "edge_label": edge_label(
            trades,
            net_profit,
            factor,
            expectancy,
        ),
    }


def grouped_summary(
    frame: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    rows = []

    grouper = (
        columns[0]
        if len(columns) == 1
        else columns
    )

    for key, group in frame.groupby(
        grouper,
        dropna=False,
        sort=False,
    ):
        if not isinstance(
            key,
            tuple,
        ):
            key = (key,)

        row = {
            column: value
            for column, value
            in zip(
                columns,
                key,
            )
        }

        row.update(
            summarize_group(group)
        )

        rows.append(row)

    return pd.DataFrame(rows)


def print_table(
    title: str,
    frame: pd.DataFrame,
    key_columns: list[str],
) -> None:
    print("-" * 150)
    print(title)
    print("-" * 150)

    if frame.empty:
        print("No data.")
        return

    header = " | ".join(
        [
            *[
                f"{column:<20}"
                for column
                in key_columns
            ],
            f"{'TRADES':>6}",
            f"{'WIN%':>7}",
            f"{'PF':>7}",
            f"{'NET P&L':>11}",
            f"{'FEES':>9}",
            f"{'PRE-FEE':>11}",
            f"{'AVG HOLD':>9}",
            f"{'STOP%':>7}",
            f"{'EDGE':<20}",
        ]
    )

    print(header)
    print("-" * 150)

    for row in frame.to_dict(
        orient="records"
    ):
        key_text = " | ".join(
            f"{str(row[column]):<20}"
            for column
            in key_columns
        )

        stop_percent = (
            float(
                row[
                    "stop_loss_exits"
                ]
            )
            / int(row["trades"])
            * 100.0
            if int(row["trades"])
            else 0.0
        )

        pf_value = float(
            row["profit_factor"]
        )

        pf_text = (
            "INF"
            if np.isinf(pf_value)
            else f"{pf_value:.2f}"
        )

        print(
            f"{key_text} | "
            f"{int(row['trades']):>6} | "
            f"{float(row['win_rate']):>6.2f}% | "
            f"{pf_text:>7} | "
            f"€{float(row['net_profit']):>9.2f} | "
            f"€{float(row['total_fees']):>7.2f} | "
            f"€{float(row['pre_fee_profit']):>9.2f} | "
            f"{float(row['average_hold_hours']):>8.2f}h | "
            f"{stop_percent:>6.2f}% | "
            f"{str(row['edge_label']):<20}"
        )


def main() -> None:
    if not TRADE_FILE.exists():
        raise FileNotFoundError(
            f"Trade file not found: {TRADE_FILE}"
        )

    trades = pd.read_csv(
        TRADE_FILE
    )

    if trades.empty:
        raise ValueError(
            "V3E3 trade history is empty."
        )

    trades["symbol"] = text(
        trades,
        "symbol",
        "UNKNOWN",
    )

    trades["route"] = text(
        trades,
        "route",
        "UNKNOWN",
    )

    trades["exit_reason"] = text(
        trades,
        "exit_reason",
        "UNKNOWN",
    )

    trades["profit"] = numeric(
        trades,
        "profit",
    )

    trades["buy_fee"] = numeric(
        trades,
        "buy_fee",
    )

    sell_fee_column = (
        "sell_fees"
        if "sell_fees"
        in trades.columns
        else "sell_fee"
    )

    trades["sell_fees_clean"] = (
        numeric(
            trades,
            sell_fee_column,
        )
    )

    trades["total_trade_fees"] = (
        trades["buy_fee"]
        + trades[
            "sell_fees_clean"
        ]
    )

    trades["setup_score"] = numeric(
        trades,
        "setup_score",
    )

    trades["entry_reward_risk"] = (
        numeric(
            trades,
            "entry_reward_risk",
        )
    )

    partial_column = (
        "partial_profit_taken"
        if "partial_profit_taken"
        in trades.columns
        else "partial_done"
    )

    trades[
        "partial_profit_taken"
    ] = (
        text(
            trades,
            partial_column,
            "False",
        )
        .str.lower()
        .isin(
            {
                "true",
                "1",
                "yes",
            }
        )
    )

    trades["is_stop_exit"] = (
        trades["exit_reason"]
        .str.upper()
        .str.contains(
            "STOP",
            regex=False,
        )
    )

    trades["entry_time_clean"] = (
        pd.to_datetime(
            trades.get(
                "entry_time",
                pd.NaT,
            ),
            errors="coerce",
        )
    )

    trades["exit_time_clean"] = (
        pd.to_datetime(
            trades.get(
                "exit_time",
                pd.NaT,
            ),
            errors="coerce",
        )
    )

    trades["hold_hours"] = (
        (
            trades[
                "exit_time_clean"
            ]
            - trades[
                "entry_time_clean"
            ]
        )
        .dt.total_seconds()
        .div(3600.0)
        .fillna(0.0)
        .clip(lower=0.0)
    )

    trades["score_bucket"] = pd.cut(
        trades["setup_score"],
        bins=[
            -np.inf,
            89.999,
            94.999,
            98.999,
            np.inf,
        ],
        labels=[
            "<90",
            "90-94",
            "95-98",
            "99-100",
        ],
    ).astype(str)

    trades["rr_bucket"] = pd.cut(
        trades[
            "entry_reward_risk"
        ],
        bins=[
            -np.inf,
            1.799,
            1.999,
            2.199,
            np.inf,
        ],
        labels=[
            "<1.8",
            "1.8-1.99",
            "2.0-2.19",
            "2.2+",
        ],
    ).astype(str)

    route_asset = grouped_summary(
        trades,
        [
            "route",
            "symbol",
        ],
    ).sort_values(
        [
            "edge_label",
            "net_profit",
        ],
        ascending=[
            True,
            False,
        ],
    )

    route_summary = grouped_summary(
        trades,
        ["route"],
    ).sort_values(
        "net_profit",
        ascending=False,
    )

    symbol_summary = grouped_summary(
        trades,
        ["symbol"],
    ).sort_values(
        "net_profit",
        ascending=False,
    )

    exit_summary = grouped_summary(
        trades,
        ["exit_reason"],
    ).sort_values(
        "net_profit",
        ascending=False,
    )

    score_summary = grouped_summary(
        trades,
        [
            "route",
            "score_bucket",
        ],
    ).sort_values(
        [
            "route",
            "score_bucket",
        ]
    )

    rr_summary = grouped_summary(
        trades,
        [
            "route",
            "rr_bucket",
        ],
    ).sort_values(
        [
            "route",
            "rr_bucket",
        ]
    )

    DETAIL_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    route_asset.to_csv(
        DETAIL_FILE,
        index=False,
    )

    route_summary.to_csv(
        ROUTE_FILE,
        index=False,
    )

    symbol_summary.to_csv(
        SYMBOL_FILE,
        index=False,
    )

    exit_summary.to_csv(
        EXIT_FILE,
        index=False,
    )

    score_summary.to_csv(
        SCORE_FILE,
        index=False,
    )

    rr_summary.to_csv(
        RR_FILE,
        index=False,
    )

    print("=" * 150)
    print(
        "FA CRYPTO ENGINE — V3E4 ROUTE/ASSET TRADE FORENSICS"
    )
    print("=" * 150)

    print_table(
        "ROUTE × ASSET EDGE",
        route_asset,
        [
            "route",
            "symbol",
        ],
    )

    print_table(
        "ROUTE TOTALS",
        route_summary,
        ["route"],
    )

    print_table(
        "SYMBOL TOTALS",
        symbol_summary,
        ["symbol"],
    )

    print_table(
        "EXIT-REASON OUTCOMES",
        exit_summary,
        ["exit_reason"],
    )

    print_table(
        "SETUP-SCORE BUCKETS",
        score_summary,
        [
            "route",
            "score_bucket",
        ],
    )

    print_table(
        "ENTRY R:R BUCKETS",
        rr_summary,
        [
            "route",
            "rr_bucket",
        ],
    )

    total = summarize_group(
        trades
    )

    print("-" * 150)
    print("OVERALL COST DIAGNOSIS")
    print("-" * 150)
    print(
        f"Net P&L                         : "
        f"€{float(total['net_profit']):.2f}"
    )
    print(
        f"Total fees                      : "
        f"€{float(total['total_fees']):.2f}"
    )
    print(
        f"Pre-fee signal P&L              : "
        f"€{float(total['pre_fee_profit']):.2f}"
    )
    print(
        f"Fees / absolute net loss        : "
        f"{float(total['fees_as_percent_of_absolute_net']):.2f}%"
    )
    print(
        f"Average holding time            : "
        f"{float(total['average_hold_hours']):.2f}h"
    )
    print(
        f"Stop-loss exit rate             : "
        f"{float(total['stop_loss_exits']) / int(total['trades']) * 100.0:.2f}%"
    )
    print(
        f"Longest losing streak           : "
        f"{int(total['longest_losing_streak'])}"
    )

    print("=" * 150)
    print(
        f"Route × asset report : {DETAIL_FILE}"
    )
    print(
        f"Route summary        : {ROUTE_FILE}"
    )
    print(
        f"Symbol summary       : {SYMBOL_FILE}"
    )
    print(
        f"Exit summary         : {EXIT_FILE}"
    )
    print(
        f"Score buckets        : {SCORE_FILE}"
    )
    print(
        f"R:R buckets          : {RR_FILE}"
    )
    print("=" * 150)


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        print()
        print(
            "V3E4 trade forensics stopped manually."
        )

    except Exception as error:
        print()
        print("=" * 150)
        print(
            "V3E4 FORENSIC ERROR"
        )
        print("=" * 150)
        print(
            f"{type(error).__name__}: "
            f"{error}"
        )
        print("=" * 150)
        raise
