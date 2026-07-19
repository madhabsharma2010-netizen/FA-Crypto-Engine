import csv
from pathlib import Path
from typing import Any

from binance.client import Client

from backtesting.backtest import run_backtest


SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "LINKUSDT",
    "DOGEUSDT",
]

INTERVAL = Client.KLINE_INTERVAL_1HOUR
CANDLE_LIMIT = 1000

SUMMARY_CSV = Path("reports/multi_asset_backtest_summary.csv")


def export_summary(
    results: list[dict[str, Any]],
) -> None:
    """
    Export successful multi-asset backtest results.
    """

    if not results:
        print("No successful results available for export.")
        return

    SUMMARY_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "symbol",
        "interval",
        "candle_limit",
        "starting_capital",
        "final_balance",
        "total_profit",
        "roi_percent",
        "completed_trades",
        "winning_trades",
        "losing_trades",
        "win_rate",
        "maximum_drawdown",
        "final_mode",
        "final_reason",
        "csv_file",
    ]

    with SUMMARY_CSV.open(
        mode="w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(results)

    print(f"Summary saved to: {SUMMARY_CSV}")


def print_summary(
    results: list[dict[str, Any]],
) -> None:
    """
    Print comparison of all successful backtests.
    """

    print()
    print("=" * 95)
    print("FA CRYPTO ENGINE — MULTI-ASSET BACKTEST SUMMARY")
    print("=" * 95)

    if not results:
        print("No backtests completed successfully.")
        print("=" * 95)
        return

    sorted_results = sorted(
        results,
        key=lambda item: float(item["roi_percent"]),
        reverse=True,
    )

    print(
        f"{'SYMBOL':<12}"
        f"{'ROI':>12}"
        f"{'WIN RATE':>14}"
        f"{'DRAWDOWN':>14}"
        f"{'TRADES':>12}"
        f"{'P&L':>14}"
    )

    print("-" * 95)

    for result in sorted_results:
        print(
            f"{result['symbol']:<12}"
            f"{float(result['roi_percent']):>11.2f}%"
            f"{float(result['win_rate']):>13.2f}%"
            f"{float(result['maximum_drawdown']):>13.2f}%"
            f"{int(result['completed_trades']):>12}"
            f"€{float(result['total_profit']):>12.2f}"
        )

    average_roi = sum(
        float(result["roi_percent"])
        for result in results
    ) / len(results)

    profitable_assets = sum(
        1
        for result in results
        if float(result["total_profit"]) > 0
    )

    best_result = sorted_results[0]
    worst_result = sorted_results[-1]

    print("-" * 95)
    print(f"Assets Tested             : {len(results)}")
    print(f"Profitable Assets         : {profitable_assets}")
    print(f"Average ROI               : {average_roi:.2f}%")
    print(
        f"Best Asset                : "
        f"{best_result['symbol']} "
        f"({float(best_result['roi_percent']):.2f}%)"
    )
    print(
        f"Worst Asset               : "
        f"{worst_result['symbol']} "
        f"({float(worst_result['roi_percent']):.2f}%)"
    )
    print("=" * 95)


def main() -> None:
    results: list[dict[str, Any]] = []
    failed_symbols: list[str] = []

    print("=" * 95)
    print("FA CRYPTO ENGINE — MULTI-ASSET BACKTEST RUNNER")
    print("=" * 95)

    for symbol in SYMBOLS:
        print()
        print("#" * 95)
        print(f"RUNNING BACKTEST: {symbol}")
        print("#" * 95)

        trade_csv = (
            f"logs/backtests/"
            f"{symbol.lower()}_trade_history.csv"
        )

        try:
            result = run_backtest(
                symbol=symbol,
                interval=INTERVAL,
                candle_limit=CANDLE_LIMIT,
                csv_file=trade_csv,
            )

            results.append(result)

        except KeyboardInterrupt:
            print()
            print("Multi-asset backtest stopped manually.")
            break

        except Exception as error:
            failed_symbols.append(symbol)

            print()
            print(f"{symbol} failed:")
            print(
                f"{type(error).__name__}: "
                f"{error}"
            )

    print_summary(results)
    export_summary(results)

    if failed_symbols:
        print()
        print(
            "Failed symbols: "
            + ", ".join(failed_symbols)
        )


if __name__ == "__main__":
    main()