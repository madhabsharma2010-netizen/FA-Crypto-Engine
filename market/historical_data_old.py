from datetime import datetime, timezone

import pandas as pd
from binance.client import Client

from core.indicators import (
    calculate_adx,
    calculate_atr,
    calculate_ema,
    calculate_rsi,
    calculate_volume_sma,
)
from strategies.ema_rsi_strategy_v2 import EmaRsiStrategyV2


client = Client()


def get_historical_candles(
    symbol: str = "BTCUSDT",
    interval: str = Client.KLINE_INTERVAL_1HOUR,
    limit: int = 1000,
    start_time: str | None = None,
    end_time: str | None = None,
    drop_incomplete: bool = True,
) -> pd.DataFrame:
    """
    Fetch Binance candle data.

    When start_time is supplied, a fixed historical
    date range is used. Otherwise, the latest candles
    are downloaded.

    The latest incomplete candle is removed by default.
    """

    if not symbol:
        raise ValueError(
            "Symbol cannot be empty."
        )

    if limit <= 0:
        raise ValueError(
            "Limit must be greater than zero."
        )

    normalized_symbol = symbol.upper()

    if start_time is not None:
        candles = client.get_historical_klines(
            symbol=normalized_symbol,
            interval=interval,
            start_str=start_time,
            end_str=end_time,
            limit=1000,
        )
    else:
        candles = client.get_klines(
            symbol=normalized_symbol,
            interval=interval,
            limit=limit,
        )

    if not candles:
        return pd.DataFrame(
            columns=[
                "open_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]
        )

    dataframe = pd.DataFrame(
        candles,
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base_volume",
            "taker_buy_quote_volume",
            "ignore",
        ],
    )

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
    ]

    for column in numeric_columns:
        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

    if drop_incomplete:
        current_time_ms = int(
            datetime.now(
                timezone.utc
            ).timestamp() * 1000
        )

        dataframe = dataframe[
            dataframe["close_time"] <= current_time_ms
        ]

    dataframe["open_time"] = pd.to_datetime(
        dataframe["open_time"],
        unit="ms",
        utc=True,
    ).dt.tz_convert(None)

    dataframe = dataframe[
        [
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
    ]

    dataframe = (
        dataframe
        .dropna()
        .drop_duplicates(
            subset=["open_time"],
            keep="last",
        )
        .sort_values("open_time")
        .reset_index(drop=True)
    )

    if len(dataframe) > limit:
        dataframe = (
            dataframe
            .tail(limit)
            .reset_index(drop=True)
        )

    return dataframe


def add_indicators(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add indicators required by Strategy V2A.
    """

    data = dataframe.copy()

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

    data["VolumeSMA20"] = calculate_volume_sma(
        data,
        20,
    )

    return data.dropna().reset_index(
        drop=True
    )


def main() -> None:
    symbol = "BTCUSDT"

    data = get_historical_candles(
        symbol=symbol,
        interval=Client.KLINE_INTERVAL_1HOUR,
        limit=1000,
        start_time="2026-05-01 00:00:00",
        end_time="2026-07-20 00:00:00",
    )

    if data.empty:
        raise ValueError(
            f"No historical data returned for {symbol}."
        )

    data = add_indicators(
        data
    )

    if data.empty:
        raise ValueError(
            "No rows remain after indicator calculation."
        )

    latest_candle = data.iloc[-1]

    signal = (
        EmaRsiStrategyV2.generate_signal(
            latest_candle
        )
    )

    print("=" * 80)
    print(
        f"FA CRYPTO ENGINE — {symbol} INDICATORS"
    )
    print("=" * 80)

    print(
        data[
            [
                "open_time",
                "close",
                "EMA20",
                "EMA50",
                "RSI14",
                "ADX14",
                "ATR14",
                "volume",
                "VolumeSMA20",
            ]
        ].tail(20).to_string(
            index=False
        )
    )

    print("=" * 80)
    print(
        "CURRENT SIGNAL:",
        signal,
    )
    print("=" * 80)


if __name__ == "__main__":
    main()