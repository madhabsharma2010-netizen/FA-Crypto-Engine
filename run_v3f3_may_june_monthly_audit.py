from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pandas as pd

import run_v3d_diagnostics as diagnostics
import run_v3e6_selective_validation as engine


MONTHS: list[tuple[str, pd.Timestamp, pd.Timestamp]] = [
    (
        "MAY_2026",
        pd.Timestamp("2026-05-01 00:00:00"),
        pd.Timestamp("2026-05-31 23:59:59"),
    ),
    (
        "JUNE_2026",
        pd.Timestamp("2026-06-01 00:00:00"),
        pd.Timestamp("2026-06-30 23:59:59"),
    ),
]

COMPARISON_FILE = Path("reports/v3f3_may_june_monthly_comparison.csv")


def classify_result(summary: dict[str, Any]) -> str:
    trades = int(summary.get("completed_trades", 0))
    net_pnl = float(summary.get("total_profit", 0.0))
    profit_factor = float(summary.get("profit_factor", 0.0))

    if trades < 10:
        return "INSUFFICIENT_SAMPLE"
    if net_pnl > 0 and profit_factor >= 1.20:
        return "MONTH_PASS"
    if net_pnl > 0 and profit_factor >= 1.00:
        return "WEAK_MONTH_PASS"
    return "MONTH_FAIL"


def configure_month(label: str, start: pd.Timestamp, end: pd.Timestamp) -> None:
    diagnostics.BACKTEST_START = start
    diagnostics.BACKTEST_END = end
    engine.BACKTEST_START = start
    engine.BACKTEST_END = end

    slug = label.lower()
    engine.TRADE_HISTORY_FILE = Path(
        f"logs/backtests/v3f3_{slug}_trade_history.csv"
    )
    engine.EVENT_HISTORY_FILE = Path(
        f"logs/backtests/v3f3_{slug}_event_history.csv"
    )
    engine.SUMMARY_FILE = Path(
        f"reports/v3f3_{slug}_summary.csv"
    )
    engine.SYMBOL_SUMMARY_FILE = Path(
        f"reports/v3f3_{slug}_symbol_summary.csv"
    )


def comparison_row(
    label: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "month": label,
        "start": start,
        "end": end,
        "completed_trades": int(summary.get("completed_trades", 0)),
        "winning_trades": int(summary.get("winning_trades", 0)),
        "losing_trades": int(summary.get("losing_trades", 0)),
        "win_rate": round(float(summary.get("win_rate", 0.0)), 4),
        "total_profit": round(float(summary.get("total_profit", 0.0)), 8),
        "roi_percent": round(float(summary.get("roi_percent", 0.0)), 4),
        "profit_factor": round(float(summary.get("profit_factor", 0.0)), 4),
        "expectancy_per_trade": round(
            float(summary.get("expectancy_per_trade", 0.0)), 8
        ),
        "maximum_drawdown": round(
            float(summary.get("maximum_drawdown", 0.0)), 4
        ),
        "total_fees": round(float(summary.get("total_fees", 0.0)), 8),
        "verdict": classify_result(summary),
    }


def export_comparison(rows: list[dict[str, Any]]) -> None:
    COMPARISON_FILE.parent.mkdir(parents=True, exist_ok=True)
    with COMPARISON_FILE.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_month_result(row: dict[str, Any]) -> None:
    print("-" * 142)
    print(f"Month                     : {row['month']}")
    print(
        f"Trades / Win-Loss         : {row['completed_trades']} / "
        f"{row['winning_trades']}-{row['losing_trades']}"
    )
    print(f"Win rate                  : {float(row['win_rate']):.2f}%")
    print(f"Net P&L                   : €{float(row['total_profit']):.2f}")
    print(f"ROI                       : {float(row['roi_percent']):.2f}%")
    print(f"Profit factor             : {float(row['profit_factor']):.2f}")
    print(f"Expectancy/trade          : €{float(row['expectancy_per_trade']):.2f}")
    print(f"Maximum drawdown          : {float(row['maximum_drawdown']):.2f}%")
    print(f"Explicit fees             : €{float(row['total_fees']):.2f}")
    print(f"Verdict                   : {row['verdict']}")


def print_comparison(rows: list[dict[str, Any]]) -> None:
    print()
    print("=" * 142)
    print("V3F3 MAY vs JUNE 2026 MONTHLY COMPARISON")
    print("=" * 142)
    print(
        f"{'MONTH':<14} {'TRADES':>8} {'W-L':>9} {'WIN%':>9} "
        f"{'PF':>8} {'NET P&L':>13} {'ROI':>9} {'FEES':>11} {'VERDICT':>22}"
    )
    print("-" * 142)

    for row in rows:
        win_loss = f"{row['winning_trades']}-{row['losing_trades']}"
        print(
            f"{row['month']:<14} "
            f"{int(row['completed_trades']):>8} "
            f"{win_loss:>9} "
            f"{float(row['win_rate']):>8.2f}% "
            f"{float(row['profit_factor']):>8.2f} "
            f"€{float(row['total_profit']):>11.2f} "
            f"{float(row['roi_percent']):>8.2f}% "
            f"€{float(row['total_fees']):>9.2f} "
            f"{str(row['verdict']):>22}"
        )

    total_trades = sum(int(row["completed_trades"]) for row in rows)
    total_profit = sum(float(row["total_profit"]) for row in rows)
    total_fees = sum(float(row["total_fees"]) for row in rows)
    positive_months = sum(float(row["total_profit"]) > 0 for row in rows)

    print("-" * 142)
    print(f"Combined trades           : {total_trades}")
    print(f"Combined net P&L          : €{total_profit:.2f}")
    print(f"Combined explicit fees    : €{total_fees:.2f}")
    print(f"Positive months           : {positive_months}/{len(rows)}")
    print()
    print("Interpretation:")
    print("- One positive month is not profitability proof.")
    print("- Consistency needs positive after-cost results across multiple untouched months.")
    print("- A month with fewer than 10 trades remains diagnostic only.")
    print("=" * 142)
    print(f"Comparison report         : {COMPARISON_FILE}")
    print("=" * 142)


def main() -> None:
    print("=" * 142)
    print("FA CRYPTO ENGINE — V3F3 MAY + JUNE 2026 MONTHLY CAPTURE AUDIT")
    print("=" * 142)
    print("Engine                    : V3E6 selective validation")
    print("Risk/exit architecture    : V3D3 frozen")
    print("Purpose                   : Month-by-month diagnostic")
    print("Normal validation reports : NOT overwritten")
    print("Strategy thresholds       : NOT changed")
    print("=" * 142)

    rows: list[dict[str, Any]] = []

    for label, start, end in MONTHS:
        print()
        print("#" * 142)
        print(f"RUNNING {label}: {start} to {end}")
        print("#" * 142)

        configure_month(label, start, end)
        summary = engine.run_backtest()
        row = comparison_row(label, start, end, summary)
        rows.append(row)
        print_month_result(row)

    export_comparison(rows)
    print_comparison(rows)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nMay-June monthly audit stopped manually.")
    except Exception as error:
        print()
        print("=" * 142)
        print("V3F3 MAY-JUNE MONTHLY AUDIT ERROR")
        print("=" * 142)
        print(f"{type(error).__name__}: {error}")
        print("=" * 142)
        raise
