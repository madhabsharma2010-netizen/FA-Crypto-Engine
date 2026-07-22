from __future__ import annotations

from pathlib import Path

import pandas as pd

import run_v3d_diagnostics as diagnostics
import run_v3e6_selective_validation as engine


MAY_START = pd.Timestamp("2026-05-01 00:00:00")
MAY_END = pd.Timestamp("2026-05-31 23:59:59")


def classify_result(summary: dict[str, object]) -> str:
    trades = int(summary.get("completed_trades", 0))
    net_pnl = float(summary.get("total_profit", 0.0))
    profit_factor = float(summary.get("profit_factor", 0.0))

    if trades < 10:
        return "INSUFFICIENT SAMPLE — May result is diagnostic only."
    if net_pnl > 0 and profit_factor >= 1.20:
        return "MAY PASS — positive after costs, but still not final proof."
    if net_pnl > 0 and profit_factor >= 1.00:
        return "WEAK MAY PASS — positive, but edge is too thin."
    return "MAY FAIL — not profitable after costs."


def main() -> None:
    # Patch both modules because load_frame() uses run_v3d_diagnostics globals,
    # while the V3E6 engine uses its own imported period globals.
    diagnostics.BACKTEST_START = MAY_START
    diagnostics.BACKTEST_END = MAY_END
    engine.BACKTEST_START = MAY_START
    engine.BACKTEST_END = MAY_END

    # Keep the normal validation reports untouched.
    engine.TRADE_HISTORY_FILE = Path(
        "logs/backtests/v3f3_may_2026_trade_history.csv"
    )
    engine.EVENT_HISTORY_FILE = Path(
        "logs/backtests/v3f3_may_2026_event_history.csv"
    )
    engine.SUMMARY_FILE = Path(
        "reports/v3f3_may_2026_summary.csv"
    )
    engine.SYMBOL_SUMMARY_FILE = Path(
        "reports/v3f3_may_2026_symbol_summary.csv"
    )

    print("=" * 126)
    print("FA CRYPTO ENGINE — V3F3 MAY 2026 ISOLATED SAMPLE")
    print("=" * 126)
    print(f"Window                    : {MAY_START} to {MAY_END}")
    print("Engine                    : V3E6 selective validation")
    print("Risk/exit architecture    : V3D3 frozen")
    print("Purpose                   : One-month diagnostic only")
    print("Normal validation reports : NOT overwritten")
    print("=" * 126)

    summary = engine.run_backtest()

    print()
    print("=" * 126)
    print("V3F3 MAY-ONLY VERDICT")
    print("=" * 126)
    print(f"Completed trades          : {int(summary.get('completed_trades', 0))}")
    print(f"Winning / losing          : {int(summary.get('winning_trades', 0))} / {int(summary.get('losing_trades', 0))}")
    print(f"Win rate                  : {float(summary.get('win_rate', 0.0)):.2f}%")
    print(f"Net P&L                   : €{float(summary.get('total_profit', 0.0)):.2f}")
    print(f"ROI                       : {float(summary.get('roi_percent', 0.0)):.2f}%")
    print(f"Profit factor             : {float(summary.get('profit_factor', 0.0)):.2f}")
    print(f"Expectancy/trade          : €{float(summary.get('expectancy_per_trade', 0.0)):.2f}")
    print(f"Maximum drawdown          : {float(summary.get('maximum_drawdown', 0.0)):.2f}%")
    print(f"Explicit fees             : €{float(summary.get('total_fees', 0.0)):.2f}")
    print(f"Verdict                   : {classify_result(summary)}")
    print("=" * 126)
    print(f"Trade log                 : {engine.TRADE_HISTORY_FILE}")
    print(f"Summary                   : {engine.SUMMARY_FILE}")
    print(f"Symbol summary            : {engine.SYMBOL_SUMMARY_FILE}")
    print("=" * 126)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nMay-only sample stopped manually.")
    except Exception as error:
        print()
        print("=" * 126)
        print("V3F3 MAY-ONLY SAMPLE ERROR")
        print("=" * 126)
        print(f"{type(error).__name__}: {error}")
        print("=" * 126)
        raise
