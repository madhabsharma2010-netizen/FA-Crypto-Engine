from pathlib import Path

import pandas as pd
from binance.client import Client

from core.indicators import (
    calculate_adx,
    calculate_atr,
    calculate_ema,
    calculate_rsi,
)
from core.regime_router import (
    MarketRegime,
    RegimeDecision,
    evaluate_regime,
)
from market.historical_data import get_historical_candles


SYMBOLS = (
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "LINKUSDT",
    "DOGEUSDT",
)

INTERVAL = Client.KLINE_INTERVAL_1HOUR
BACKTEST_START = "2026-05-01 00:00:00"
BACKTEST_END = "2026-07-20 00:00:00"

DETAIL_FILE = Path(
    "reports/regime_ranking_detail_v3.csv"
)

SUMMARY_FILE = Path(
    "reports/regime_ranking_summary_v3.csv"
)


def prepare_data(
    symbol: str,
) -> pd.DataFrame:
    data = get_historical_candles(
        symbol=symbol,
        interval=INTERVAL,
        limit=1000,
        start_time=BACKTEST_START,
        end_time=BACKTEST_END,
        drop_incomplete=True,
    )

    if data.empty:
        raise ValueError(
            f"No historical data returned for {symbol}."
        )

    for column in (
        "high",
        "low",
        "close",
        "volume",
    ):
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    data["EMA20"] = calculate_ema(
        data,
        20,
    )
    data["EMA50"] = calculate_ema(
        data,
        50,
    )
    data["EMA200"] = calculate_ema(
        data,
        200,
    )
    data["RSI14"] = calculate_rsi(
        data,
        14,
    )
    data["ADX14"] = calculate_adx(
        data,
        14,
    )
    data["ATR14"] = calculate_atr(
        data,
        14,
    )

    data["VolumeSMA20"] = (
        data["volume"]
        .rolling(20)
        .mean()
    )

    data["MOMENTUM_24H"] = (
        data["close"]
        .pct_change(24)
        * 100
    )

    data["EMA50_SLOPE_12H"] = (
        data["EMA50"]
        .pct_change(12)
        * 100
    )

    return (
        data
        .dropna()
        .set_index("open_time")
        .sort_index()
    )


def allocate_ranked_assets(
    ranked: list[RegimeDecision],
) -> dict[str, float]:
    """
    Portfolio capital allocation.

    Rules:
        - No eligible asset: 100% cash.
        - One exceptional leader: up to 50%.
        - Two closely ranked strong assets: 30% + 20%.
        - Otherwise only the leader receives 30%.
        - Maximum deployed capital: 50%.
    """

    eligible = [
        decision
        for decision in ranked
        if decision.regime
        in {
            MarketRegime.ACTIVE,
            MarketRegime.STRONG,
        }
        and decision.score >= 65.0
    ]

    if not eligible:
        return {}

    leader = eligible[0]

    if len(eligible) == 1:
        allocation = (
            50.0
            if leader.score >= 85.0
            else 30.0
        )

        return {
            leader.symbol: allocation,
        }

    second = eligible[1]
    score_gap = (
        leader.score
        - second.score
    )

    if (
        leader.score >= 85.0
        and score_gap >= 12.0
    ):
        return {
            leader.symbol: 50.0,
        }

    if (
        second.score >= 70.0
        and score_gap <= 12.0
    ):
        return {
            leader.symbol: 30.0,
            second.symbol: 20.0,
        }

    return {
        leader.symbol: 30.0,
    }


def main() -> None:
    data_by_symbol = {
        symbol: prepare_data(symbol)
        for symbol in SYMBOLS
    }

    common_times = None

    for data in data_by_symbol.values():
        if common_times is None:
            common_times = data.index
        else:
            common_times = (
                common_times
                .intersection(data.index)
            )

    if common_times is None or len(common_times) == 0:
        raise ValueError(
            "No common candle timestamps found."
        )

    common_times = common_times.sort_values()

    detail_rows: list[dict[str, object]] = []

    selection_counts = {
        symbol: 0
        for symbol in SYMBOLS
    }

    capital_sum = {
        symbol: 0.0
        for symbol in SYMBOLS
    }

    cash_only_candles = 0
    total_deployed_percent = 0.0
    latest_ranked: list[RegimeDecision] = []
    latest_allocations: dict[str, float] = {}

    for candle_time in common_times:
        decisions = [
            evaluate_regime(
                symbol,
                data_by_symbol[symbol].loc[
                    candle_time
                ],
            )
            for symbol in SYMBOLS
        ]

        ranked = sorted(
            decisions,
            key=lambda item: (
                item.score,
                item.symbol,
            ),
            reverse=True,
        )

        allocations = allocate_ranked_assets(
            ranked
        )

        deployed = sum(
            allocations.values()
        )

        total_deployed_percent += deployed

        if not allocations:
            cash_only_candles += 1

        rank_by_symbol = {
            decision.symbol: rank
            for rank, decision in enumerate(
                ranked,
                start=1,
            )
        }

        for decision in decisions:
            selected = (
                decision.symbol
                in allocations
            )

            capital_percent = (
                allocations.get(
                    decision.symbol,
                    0.0,
                )
            )

            if selected:
                selection_counts[
                    decision.symbol
                ] += 1

                capital_sum[
                    decision.symbol
                ] += capital_percent

            detail_rows.append(
                {
                    "open_time": candle_time,
                    "symbol": decision.symbol,
                    "rank": rank_by_symbol[
                        decision.symbol
                    ],
                    "regime": (
                        decision.regime.value
                    ),
                    "score": decision.score,
                    "selected": selected,
                    "capital_percent": (
                        capital_percent
                    ),
                    "portfolio_deployed_percent": (
                        deployed
                    ),
                }
            )

        latest_ranked = ranked
        latest_allocations = allocations

    detail = pd.DataFrame(
        detail_rows
    )

    total_candles = len(
        common_times
    )

    summary_rows = []

    for symbol in SYMBOLS:
        selected_candles = (
            selection_counts[symbol]
        )

        summary_rows.append(
            {
                "symbol": symbol,
                "selected_candles": (
                    selected_candles
                ),
                "selected_percent": (
                    selected_candles
                    / total_candles
                    * 100
                ),
                "average_capital_when_selected": (
                    capital_sum[symbol]
                    / selected_candles
                    if selected_candles
                    else 0.0
                ),
            }
        )

    summary = pd.DataFrame(
        summary_rows
    ).sort_values(
        "selected_percent",
        ascending=False,
    )

    DETAIL_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    detail.to_csv(
        DETAIL_FILE,
        index=False,
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    print("=" * 112)
    print(
        "FA CRYPTO ENGINE — V3 CROSS-ASSET RANKING DIAGNOSTICS"
    )
    print("=" * 112)

    print(
        f"Common candles analysed   : {total_candles}"
    )
    print(
        f"Cash-only candles         : {cash_only_candles} "
        f"({cash_only_candles / total_candles * 100:.2f}%)"
    )
    print(
        f"Average capital deployed  : "
        f"{total_deployed_percent / total_candles:.2f}%"
    )

    print("-" * 112)
    print(
        f"{'SYMBOL':<10} "
        f"{'SELECTED':>12} "
        f"{'SELECTED %':>14} "
        f"{'AVG CAPITAL':>16}"
    )
    print("-" * 112)

    for row in summary.to_dict(
        orient="records"
    ):
        print(
            f"{row['symbol']:<10} "
            f"{int(row['selected_candles']):>12} "
            f"{row['selected_percent']:>13.2f}% "
            f"{row['average_capital_when_selected']:>15.2f}%"
        )

    print("-" * 112)
    print(
        "LATEST CROSS-ASSET RANKING"
    )
    print("-" * 112)

    for rank, decision in enumerate(
        latest_ranked,
        start=1,
    ):
        allocation = latest_allocations.get(
            decision.symbol,
            0.0,
        )

        status = (
            "SELECTED"
            if allocation > 0
            else "SLEEP"
        )

        print(
            f"{rank}. {decision.symbol:<10} | "
            f"Score {decision.score:>6.2f} | "
            f"Regime {decision.regime.value:<6} | "
            f"{status:<8} | "
            f"Capital {allocation:>5.1f}%"
        )

    print("=" * 112)
    print(
        f"Detail saved to : {DETAIL_FILE}"
    )
    print(
        f"Summary saved to: {SUMMARY_FILE}"
    )
    print("=" * 112)


if __name__ == "__main__":
    main()
