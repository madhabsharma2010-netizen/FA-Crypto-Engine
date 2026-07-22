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
    "reports/regime_ranking_detail_v3b.csv"
)

SUMMARY_FILE = Path(
    "reports/regime_ranking_summary_v3b.csv"
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

    data["MOMENTUM_72H"] = (
        data["close"]
        .pct_change(72)
        * 100
    )

    data["EMA50_SLOPE_12H"] = (
        data["EMA50"]
        .pct_change(12)
        * 100
    )

    data["VOLUME_RATIO"] = (
        data["volume"]
        / data["VolumeSMA20"]
    )

    data["ATR_PERCENT"] = (
        data["ATR14"]
        / data["close"]
        * 100
    )

    return (
        data
        .dropna()
        .set_index("open_time")
        .sort_index()
    )


def percentile_score(
    series: pd.Series,
    higher_is_better: bool = True,
) -> pd.Series:
    """
    Cross-sectional percentile score from 0 to 100.
    """

    result = series.rank(
        method="average",
        pct=True,
        ascending=higher_is_better,
    ) * 100

    if higher_is_better:
        return result

    return 100 - result + (
        100 / max(len(series), 1)
    )


def calculate_relative_score(
    snapshot: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add cross-asset ranking components.

    Final score uses:
        45% absolute regime quality
        20% 24-hour momentum rank
        15% 72-hour momentum rank
        10% EMA50 slope rank
         5% ADX rank
         5% volume-ratio rank

    Excess volatility receives a penalty.
    """

    ranked = snapshot.copy()

    ranked["MOM24_RANK"] = percentile_score(
        ranked["MOMENTUM_24H"]
    )

    ranked["MOM72_RANK"] = percentile_score(
        ranked["MOMENTUM_72H"]
    )

    ranked["SLOPE_RANK"] = percentile_score(
        ranked["EMA50_SLOPE_12H"]
    )

    ranked["ADX_RANK"] = percentile_score(
        ranked["ADX14"]
    )

    ranked["VOLUME_RANK"] = percentile_score(
        ranked["VOLUME_RATIO"]
    )

    ranked["VOLATILITY_PENALTY"] = 0.0

    ranked.loc[
        ranked["ATR_PERCENT"] > 4.5,
        "VOLATILITY_PENALTY",
    ] += 7.5

    ranked.loc[
        ranked["ATR_PERCENT"] > 6.0,
        "VOLATILITY_PENALTY",
    ] += 7.5

    ranked.loc[
        ranked["MOMENTUM_24H"] < 0,
        "VOLATILITY_PENALTY",
    ] += 5.0

    ranked["RELATIVE_SCORE"] = (
        ranked["REGIME_SCORE"] * 0.45
        + ranked["MOM24_RANK"] * 0.20
        + ranked["MOM72_RANK"] * 0.15
        + ranked["SLOPE_RANK"] * 0.10
        + ranked["ADX_RANK"] * 0.05
        + ranked["VOLUME_RANK"] * 0.05
        - ranked["VOLATILITY_PENALTY"]
    )

    ranked["RELATIVE_SCORE"] = (
        ranked["RELATIVE_SCORE"]
        .clip(lower=0.0, upper=100.0)
    )

    return ranked


def allocate_ranked_assets(
    ranked: pd.DataFrame,
) -> dict[str, float]:
    """
    Select at most two assets.

    Eligibility:
        - Regime ACTIVE or STRONG
        - Absolute regime score >= 65
        - Relative score >= 60

    Allocation:
        - Exceptional isolated leader: 50%
        - Two qualified leaders: 30% + 20%
        - Normal single leader: 30%
    """

    eligible = ranked[
        ranked["REGIME"].isin(
            (
                MarketRegime.ACTIVE.value,
                MarketRegime.STRONG.value,
            )
        )
        & (ranked["REGIME_SCORE"] >= 65.0)
        & (ranked["RELATIVE_SCORE"] >= 60.0)
    ].copy()

    if eligible.empty:
        return {}

    eligible = eligible.sort_values(
        [
            "RELATIVE_SCORE",
            "REGIME_SCORE",
            "MOMENTUM_24H",
            "MOMENTUM_72H",
        ],
        ascending=False,
    )

    leader = eligible.iloc[0]

    if len(eligible) == 1:
        allocation = (
            50.0
            if (
                leader["RELATIVE_SCORE"] >= 82.0
                and leader["REGIME_SCORE"] >= 80.0
            )
            else 30.0
        )

        return {
            str(leader["symbol"]): allocation,
        }

    second = eligible.iloc[1]

    relative_gap = (
        float(leader["RELATIVE_SCORE"])
        - float(second["RELATIVE_SCORE"])
    )

    if (
        leader["RELATIVE_SCORE"] >= 85.0
        and relative_gap >= 15.0
    ):
        return {
            str(leader["symbol"]): 50.0,
        }

    if (
        second["RELATIVE_SCORE"] >= 65.0
        and relative_gap <= 15.0
    ):
        return {
            str(leader["symbol"]): 30.0,
            str(second["symbol"]): 20.0,
        }

    return {
        str(leader["symbol"]): 30.0,
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

    latest_ranked = pd.DataFrame()
    latest_allocations: dict[str, float] = {}

    for candle_time in common_times:
        snapshot_rows: list[dict[str, object]] = []

        for symbol in SYMBOLS:
            candle = data_by_symbol[
                symbol
            ].loc[candle_time]

            decision = evaluate_regime(
                symbol,
                candle,
            )

            snapshot_rows.append(
                {
                    "symbol": symbol,
                    "REGIME": decision.regime.value,
                    "REGIME_SCORE": decision.score,
                    "MOMENTUM_24H": float(
                        candle["MOMENTUM_24H"]
                    ),
                    "MOMENTUM_72H": float(
                        candle["MOMENTUM_72H"]
                    ),
                    "EMA50_SLOPE_12H": float(
                        candle["EMA50_SLOPE_12H"]
                    ),
                    "ADX14": float(
                        candle["ADX14"]
                    ),
                    "VOLUME_RATIO": float(
                        candle["VOLUME_RATIO"]
                    ),
                    "ATR_PERCENT": float(
                        candle["ATR_PERCENT"]
                    ),
                }
            )

        snapshot = pd.DataFrame(
            snapshot_rows
        )

        ranked = calculate_relative_score(
            snapshot
        ).sort_values(
            [
                "RELATIVE_SCORE",
                "REGIME_SCORE",
                "MOMENTUM_24H",
                "MOMENTUM_72H",
            ],
            ascending=False,
        ).reset_index(drop=True)

        ranked["RANK"] = (
            ranked.index + 1
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

        for row in ranked.to_dict(
            orient="records"
        ):
            symbol = str(
                row["symbol"]
            )

            selected = symbol in allocations
            capital_percent = allocations.get(
                symbol,
                0.0,
            )

            if selected:
                selection_counts[
                    symbol
                ] += 1

                capital_sum[
                    symbol
                ] += capital_percent

            detail_rows.append(
                {
                    "open_time": candle_time,
                    "symbol": symbol,
                    "rank": int(row["RANK"]),
                    "regime": row["REGIME"],
                    "regime_score": round(
                        float(row["REGIME_SCORE"]),
                        2,
                    ),
                    "relative_score": round(
                        float(row["RELATIVE_SCORE"]),
                        2,
                    ),
                    "momentum_24h": round(
                        float(row["MOMENTUM_24H"]),
                        4,
                    ),
                    "momentum_72h": round(
                        float(row["MOMENTUM_72H"]),
                        4,
                    ),
                    "selected": selected,
                    "capital_percent": (
                        capital_percent
                    ),
                    "portfolio_deployed_percent": (
                        deployed
                    ),
                }
            )

        latest_ranked = ranked.copy()
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

    print("=" * 118)
    print(
        "FA CRYPTO ENGINE — V3B RELATIVE-STRENGTH RANKING DIAGNOSTICS"
    )
    print("=" * 118)

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

    print("-" * 118)
    print(
        f"{'SYMBOL':<10} "
        f"{'SELECTED':>12} "
        f"{'SELECTED %':>14} "
        f"{'AVG CAPITAL':>16}"
    )
    print("-" * 118)

    for row in summary.to_dict(
        orient="records"
    ):
        print(
            f"{row['symbol']:<10} "
            f"{int(row['selected_candles']):>12} "
            f"{row['selected_percent']:>13.2f}% "
            f"{row['average_capital_when_selected']:>15.2f}%"
        )

    print("-" * 118)
    print(
        "LATEST RELATIVE-STRENGTH RANKING"
    )
    print("-" * 118)

    for row in latest_ranked.to_dict(
        orient="records"
    ):
        symbol = str(
            row["symbol"]
        )

        allocation = latest_allocations.get(
            symbol,
            0.0,
        )

        status = (
            "SELECTED"
            if allocation > 0
            else "SLEEP"
        )

        print(
            f"{int(row['RANK'])}. "
            f"{symbol:<10} | "
            f"Relative {float(row['RELATIVE_SCORE']):>6.2f} | "
            f"Regime {float(row['REGIME_SCORE']):>6.2f} | "
            f"24h {float(row['MOMENTUM_24H']):>7.2f}% | "
            f"{status:<8} | "
            f"Capital {allocation:>5.1f}%"
        )

    print("=" * 118)
    print(
        f"Detail saved to : {DETAIL_FILE}"
    )
    print(
        f"Summary saved to: {SUMMARY_FILE}"
    )
    print("=" * 118)


if __name__ == "__main__":
    main()
