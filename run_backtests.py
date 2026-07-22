import csv
import math
from collections import Counter
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

BACKTEST_START = "2026-05-01 00:00:00"
BACKTEST_END = "2026-07-20 00:00:00"

SUMMARY_CSV = Path(
    "reports/multi_asset_backtest_summary.csv"
)


def safe_float(
    value: Any,
    default: float = 0.0,
) -> float:
    """
    Convert a value to float safely.
    """

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def analyze_trade_history(
    csv_file: str,
) -> dict[str, Any]:
    """
    Calculate advanced statistics from one
    asset's exported trade-history CSV.
    """

    file_path = Path(csv_file)

    default_metrics: dict[str, Any] = {
        "gross_profit": 0.0,
        "gross_loss": 0.0,
        "profit_factor": 0.0,
        "average_win": 0.0,
        "average_loss": 0.0,
        "payoff_ratio": 0.0,
        "expectancy": 0.0,
        "largest_win": 0.0,
        "largest_loss": 0.0,
        "total_fees": 0.0,
        "break_even_trades": 0,
        "stop_loss_exits": 0,
        "take_profit_exits": 0,
        "atr_trailing_exits": 0,
        "strategy_sell_exits": 0,
        "profit_extension_exits": 0,
        "end_of_backtest_exits": 0,
        "other_exits": 0,
    }

    if not file_path.exists():
        return default_metrics

    with file_path.open(
        mode="r",
        newline="",
        encoding="utf-8",
    ) as handle:
        rows = list(
            csv.DictReader(handle)
        )

    if not rows:
        return default_metrics

    profits = [
        safe_float(row.get("profit"))
        for row in rows
    ]

    winning_profits = [
        profit
        for profit in profits
        if profit > 0
    ]

    losing_profits = [
        profit
        for profit in profits
        if profit < 0
    ]

    break_even_trades = sum(
        1
        for profit in profits
        if profit == 0
    )

    gross_profit = sum(
        winning_profits
    )

    gross_loss = abs(
        sum(losing_profits)
    )

    if gross_loss > 0:
        profit_factor = (
            gross_profit
            / gross_loss
        )
    elif gross_profit > 0:
        profit_factor = math.inf
    else:
        profit_factor = 0.0

    average_win = (
        gross_profit
        / len(winning_profits)
        if winning_profits
        else 0.0
    )

    average_loss = (
        sum(losing_profits)
        / len(losing_profits)
        if losing_profits
        else 0.0
    )

    if average_loss < 0:
        payoff_ratio = (
            average_win
            / abs(average_loss)
        )
    elif average_win > 0:
        payoff_ratio = math.inf
    else:
        payoff_ratio = 0.0

    expectancy = (
        sum(profits)
        / len(profits)
    )

    largest_win = max(
        winning_profits,
        default=0.0,
    )

    largest_loss = min(
        losing_profits,
        default=0.0,
    )

    total_fees = sum(
        safe_float(row.get("buy_fee"))
        + safe_float(row.get("sell_fee"))
        for row in rows
    )

    exit_counts = Counter(
        str(
            row.get("exit_reason", "")
        ).strip().upper()
        for row in rows
    )

    recognised_exit_total = sum(
        exit_counts.get(reason, 0)
        for reason in [
            "STOP LOSS",
            "TAKE PROFIT",
            "ATR TRAILING",
            "STRATEGY SELL",
            "PROFIT EXTENSION TRAILING STOP",
            "END OF BACKTEST",
        ]
    )

    return {
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "profit_factor": profit_factor,
        "average_win": average_win,
        "average_loss": average_loss,
        "payoff_ratio": payoff_ratio,
        "expectancy": expectancy,
        "largest_win": largest_win,
        "largest_loss": largest_loss,
        "total_fees": total_fees,
        "break_even_trades": break_even_trades,
        "stop_loss_exits": exit_counts.get(
            "STOP LOSS",
            0,
        ),
        "take_profit_exits": exit_counts.get(
            "TAKE PROFIT",
            0,
        ),
        "atr_trailing_exits": exit_counts.get(
            "ATR TRAILING",
            0,
        ),
        "strategy_sell_exits": exit_counts.get(
            "STRATEGY SELL",
            0,
        ),
        "profit_extension_exits": exit_counts.get(
            "PROFIT EXTENSION TRAILING STOP",
            0,
        ),
        "end_of_backtest_exits": exit_counts.get(
            "END OF BACKTEST",
            0,
        ),
        "other_exits": max(
            len(rows) - recognised_exit_total,
            0,
        ),
    }


def export_summary(
    results: list[dict[str, Any]],
) -> None:
    """
    Export successful multi-asset backtest results
    with advanced performance metrics.
    """

    if not results:
        print(
            "No successful results available for export."
        )
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
        "break_even_trades",
        "win_rate",
        "maximum_drawdown",
        "gross_profit",
        "gross_loss",
        "profit_factor",
        "average_win",
        "average_loss",
        "payoff_ratio",
        "expectancy",
        "largest_win",
        "largest_loss",
        "total_fees",
        "stop_loss_exits",
        "take_profit_exits",
        "atr_trailing_exits",
        "strategy_sell_exits",
        "profit_extension_exits",
        "end_of_backtest_exits",
        "other_exits",
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
            extrasaction="ignore",
        )

        writer.writeheader()
        writer.writerows(results)

    print(
        f"Summary saved to: {SUMMARY_CSV}"
    )


def format_ratio(
    value: Any,
) -> str:
    """
    Format finite or infinite ratios.
    """

    numeric_value = safe_float(value)

    if math.isinf(numeric_value):
        return "INF"

    return f"{numeric_value:.2f}"


def print_summary(
    results: list[dict[str, Any]],
) -> None:
    """
    Print portfolio comparison and advanced
    trade-quality statistics.
    """

    print()
    print("=" * 105)
    print(
        "FA CRYPTO ENGINE — "
        "MULTI-ASSET BACKTEST SUMMARY"
    )
    print("=" * 105)

    if not results:
        print(
            "No backtests completed successfully."
        )
        print("=" * 105)
        return

    sorted_results = sorted(
        results,
        key=lambda item: safe_float(
            item.get("roi_percent")
        ),
        reverse=True,
    )

    print(
        f"{'SYMBOL':<12}"
        f"{'ROI':>11}"
        f"{'WIN RATE':>13}"
        f"{'DRAWDOWN':>13}"
        f"{'TRADES':>10}"
        f"{'PROFIT FACTOR':>16}"
        f"{'EXPECTANCY':>14}"
        f"{'P&L':>16}"
    )

    print("-" * 105)

    for result in sorted_results:
        pnl_text = (
            f"€{safe_float(result.get('total_profit')):.2f}"
        )

        expectancy_text = (
            f"€{safe_float(result.get('expectancy')):.2f}"
        )

        print(
            f"{str(result.get('symbol', '')):<12}"
            f"{safe_float(result.get('roi_percent')):>10.2f}%"
            f"{safe_float(result.get('win_rate')):>12.2f}%"
            f"{safe_float(result.get('maximum_drawdown')):>12.2f}%"
            f"{int(result.get('completed_trades', 0)):>10}"
            f"{format_ratio(result.get('profit_factor')):>16}"
            f"{expectancy_text:>14}"
            f"{pnl_text:>16}"
        )

    total_starting_capital = sum(
        safe_float(
            result.get("starting_capital")
        )
        for result in results
    )

    combined_final_balance = sum(
        safe_float(
            result.get("final_balance")
        )
        for result in results
    )

    combined_profit = sum(
        safe_float(
            result.get("total_profit")
        )
        for result in results
    )

    total_trades = sum(
        int(
            result.get(
                "completed_trades",
                0,
            )
        )
        for result in results
    )

    total_winning_trades = sum(
        int(
            result.get(
                "winning_trades",
                0,
            )
        )
        for result in results
    )

    total_losing_trades = sum(
        int(
            result.get(
                "losing_trades",
                0,
            )
        )
        for result in results
    )

    total_break_even_trades = sum(
        int(
            result.get(
                "break_even_trades",
                0,
            )
        )
        for result in results
    )

    combined_gross_profit = sum(
        safe_float(
            result.get("gross_profit")
        )
        for result in results
    )

    combined_gross_loss = sum(
        safe_float(
            result.get("gross_loss")
        )
        for result in results
    )

    if combined_gross_loss > 0:
        combined_profit_factor = (
            combined_gross_profit
            / combined_gross_loss
        )
    elif combined_gross_profit > 0:
        combined_profit_factor = math.inf
    else:
        combined_profit_factor = 0.0

    combined_expectancy = (
        combined_profit
        / total_trades
        if total_trades > 0
        else 0.0
    )

    total_fees = sum(
        safe_float(
            result.get("total_fees")
        )
        for result in results
    )

    total_stop_loss_exits = sum(
        int(
            result.get(
                "stop_loss_exits",
                0,
            )
        )
        for result in results
    )

    total_take_profit_exits = sum(
        int(
            result.get(
                "take_profit_exits",
                0,
            )
        )
        for result in results
    )

    total_atr_exits = sum(
        int(
            result.get(
                "atr_trailing_exits",
                0,
            )
        )
        for result in results
    )

    total_strategy_exits = sum(
        int(
            result.get(
                "strategy_sell_exits",
                0,
            )
        )
        for result in results
    )

    total_profit_extension_exits = sum(
        int(
            result.get(
                "profit_extension_exits",
                0,
            )
        )
        for result in results
    )

    total_end_exits = sum(
        int(
            result.get(
                "end_of_backtest_exits",
                0,
            )
        )
        for result in results
    )

    if total_starting_capital > 0:
        combined_roi = (
            combined_profit
            / total_starting_capital
        ) * 100
    else:
        combined_roi = 0.0

    if total_trades > 0:
        overall_win_rate = (
            total_winning_trades
            / total_trades
        ) * 100
    else:
        overall_win_rate = 0.0

    profitable_assets = sum(
        1
        for result in results
        if safe_float(
            result.get("total_profit")
        ) > 0
    )

    average_roi = sum(
        safe_float(
            result.get("roi_percent")
        )
        for result in results
    ) / len(results)

    highest_asset_drawdown = max(
        safe_float(
            result.get("maximum_drawdown")
        )
        for result in results
    )

    best_result = sorted_results[0]
    worst_result = sorted_results[-1]

    print("-" * 105)
    print(
        f"Assets Tested             : "
        f"{len(results)}"
    )
    print(
        f"Profitable Assets         : "
        f"{profitable_assets}"
    )
    print(
        f"Losing Assets             : "
        f"{len(results) - profitable_assets}"
    )
    print(
        f"Total Starting Capital    : "
        f"€{total_starting_capital:.2f}"
    )
    print(
        f"Combined Final Balance    : "
        f"€{combined_final_balance:.2f}"
    )
    print(
        f"Combined Profit/Loss      : "
        f"€{combined_profit:.2f}"
    )
    print(
        f"Combined Portfolio ROI    : "
        f"{combined_roi:.2f}%"
    )
    print(
        f"Average Asset ROI         : "
        f"{average_roi:.2f}%"
    )
    print(
        f"Total Completed Trades    : "
        f"{total_trades}"
    )
    print(
        f"Total Winning Trades      : "
        f"{total_winning_trades}"
    )
    print(
        f"Total Losing Trades       : "
        f"{total_losing_trades}"
    )
    print(
        f"Break-even Trades         : "
        f"{total_break_even_trades}"
    )
    print(
        f"Overall Win Rate          : "
        f"{overall_win_rate:.2f}%"
    )
    print(
        f"Combined Gross Profit     : "
        f"€{combined_gross_profit:.2f}"
    )
    print(
        f"Combined Gross Loss       : "
        f"€{combined_gross_loss:.2f}"
    )
    print(
        f"Combined Profit Factor    : "
        f"{format_ratio(combined_profit_factor)}"
    )
    print(
        f"Expectancy Per Trade      : "
        f"€{combined_expectancy:.2f}"
    )
    print(
        f"Total Trading Fees        : "
        f"€{total_fees:.2f}"
    )
    print(
        f"Highest Asset Drawdown    : "
        f"{highest_asset_drawdown:.2f}%"
    )
    print(
        f"Stop-loss Exits           : "
        f"{total_stop_loss_exits}"
    )
    print(
        f"Take-profit Exits         : "
        f"{total_take_profit_exits}"
    )
    print(
        f"ATR Trailing Exits        : "
        f"{total_atr_exits}"
    )
    print(
        f"Strategy Sell Exits       : "
        f"{total_strategy_exits}"
    )
    print(
        f"Profit Extension Exits    : "
        f"{total_profit_extension_exits}"
    )
    print(
        f"End-of-Backtest Exits     : "
        f"{total_end_exits}"
    )
    print(
        f"Best Asset                : "
        f"{best_result['symbol']} "
        f"({safe_float(best_result.get('roi_percent')):.2f}%)"
    )
    print(
        f"Worst Asset               : "
        f"{worst_result['symbol']} "
        f"({safe_float(worst_result.get('roi_percent')):.2f}%)"
    )
    print("=" * 105)

    print()
    print("=" * 105)
    print(
        "FA CRYPTO ENGINE — "
        "ADVANCED ASSET DIAGNOSTICS"
    )
    print("=" * 105)

    print(
        f"{'SYMBOL':<12}"
        f"{'AVG WIN':>13}"
        f"{'AVG LOSS':>13}"
        f"{'PAYOFF':>11}"
        f"{'LARGEST WIN':>15}"
        f"{'LARGEST LOSS':>15}"
        f"{'FEES':>13}"
    )

    print("-" * 105)

    for result in sorted_results:
        print(
            f"{str(result.get('symbol', '')):<12}"
            f"{'€' + format(safe_float(result.get('average_win')), '.2f'):>13}"
            f"{'€' + format(safe_float(result.get('average_loss')), '.2f'):>13}"
            f"{format_ratio(result.get('payoff_ratio')):>11}"
            f"{'€' + format(safe_float(result.get('largest_win')), '.2f'):>15}"
            f"{'€' + format(safe_float(result.get('largest_loss')), '.2f'):>15}"
            f"{'€' + format(safe_float(result.get('total_fees')), '.2f'):>13}"
        )

    print("=" * 105)


def main() -> None:
    """
    Run fixed-date backtests for all configured assets.
    """

    results: list[dict[str, Any]] = []
    failed_symbols: list[str] = []

    print("=" * 105)
    print(
        "FA CRYPTO ENGINE — "
        "MULTI-ASSET BACKTEST RUNNER"
    )
    print("=" * 105)

    for symbol in SYMBOLS:
        print()
        print("#" * 105)
        print(
            f"RUNNING BACKTEST: {symbol}"
        )
        print("#" * 105)

        trade_csv = (
            "logs/backtests/"
            f"{symbol.lower()}_trade_history.csv"
        )

        try:
            result = run_backtest(
                symbol=symbol,
                interval=INTERVAL,
                candle_limit=CANDLE_LIMIT,
                csv_file=trade_csv,
                start_time=BACKTEST_START,
                end_time=BACKTEST_END,
            )

            if not isinstance(
                result,
                dict,
            ):
                raise TypeError(
                    "run_backtest() did not return "
                    "a result dictionary."
                )

            advanced_metrics = (
                analyze_trade_history(
                    trade_csv
                )
            )

            result.update(
                advanced_metrics
            )

            results.append(
                result
            )

        except KeyboardInterrupt:
            print()
            print(
                "Multi-asset backtest "
                "stopped manually."
            )
            break

        except Exception as error:
            failed_symbols.append(
                symbol
            )

            print()
            print(
                f"{symbol} failed:"
            )
            print(
                f"{type(error).__name__}: "
                f"{error}"
            )

    print_summary(
        results
    )

    export_summary(
        results
    )

    if failed_symbols:
        print()
        print(
            "Failed symbols: "
            + ", ".join(
                failed_symbols
            )
        )


if __name__ == "__main__":
    main()
