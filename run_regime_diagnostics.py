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

OUTPUT_FILE = Path(
    "reports/regime_diagnostics_v3.csv"
)


def prepare_data(symbol: str) -> pd.DataFrame:
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

    data["EMA20"] = calculate_ema(data, 20)
    data["EMA50"] = calculate_ema(data, 50)
    data["EMA200"] = calculate_ema(data, 200)
    data["RSI14"] = calculate_rsi(data, 14)
    data["ADX14"] = calculate_adx(data, 14)
    data["ATR14"] = calculate_atr(data, 14)

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
        .reset_index(drop=True)
    )


def analyse_symbol(
    symbol: str,
) -> dict[str, object]:
    data = prepare_data(symbol)

    decisions = [
        evaluate_regime(
            symbol,
            data.iloc[index],
        )
        for index in range(len(data))
    ]

    counts = {
        regime.value: 0
        for regime in MarketRegime
    }

    for decision in decisions:
        counts[decision.regime.value] += 1

    total = len(decisions)
    latest = decisions[-1]

    active_candles = (
        counts["ACTIVE"]
        + counts["STRONG"]
    )

    return {
        "symbol": symbol,
        "candles_scored": total,
        "sleep_candles": counts["SLEEP"],
        "watch_candles": counts["WATCH"],
        "active_candles": counts["ACTIVE"],
        "strong_candles": counts["STRONG"],
        "tradable_percent": (
            active_candles / total * 100
            if total
            else 0.0
        ),
        "latest_time": data.iloc[-1]["open_time"],
        "latest_regime": latest.regime.value,
        "latest_score": latest.score,
        "recommended_capital_percent": (
            latest.capital_usage_percent
        ),
        "recommended_risk_percent": (
            latest.risk_percent
        ),
        "latest_reasons": " | ".join(
            latest.reasons
        ),
    }


def main() -> None:
    print("=" * 112)
    print(
        "FA CRYPTO ENGINE — V3 REGIME ROUTER DIAGNOSTICS"
    )
    print("=" * 112)

    rows: list[dict[str, object]] = []

    for symbol in SYMBOLS:
        row = analyse_symbol(symbol)
        rows.append(row)

        print(
            f"{symbol:<10} | "
            f"Tradable {row['tradable_percent']:>6.2f}% | "
            f"Latest {row['latest_regime']:<6} | "
            f"Score {row['latest_score']:>6.2f} | "
            f"Capital {row['recommended_capital_percent']:>5.1f}% | "
            f"Risk {row['recommended_risk_percent']:>4.2f}%"
        )

    report = pd.DataFrame(rows)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("=" * 112)
    print(
        f"Diagnostics saved to: {OUTPUT_FILE}"
    )
    print("=" * 112)


if __name__ == "__main__":
    main()
