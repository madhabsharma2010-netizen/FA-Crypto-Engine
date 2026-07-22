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

BINANCE_MAX_BATCH = 1000


def _to_milliseconds(
    value: str,
) -> int:
    """
    Convert a date/time string to UTC milliseconds.
    """

    timestamp = pd.Timestamp(value)

    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")

    return int(
        timestamp.timestamp() * 1000
    )


def _empty_candle_frame() -> pd.DataFrame:
    """
    Return an empty candle dataframe with
    the expected columns.
    """

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


def _fetch_fixed_range(
    symbol: str,
    interval: str,
    start_time: str,
    end_time: str | None,
) -> list[list]:
    """
    Fetch an entire fixed Binance date range
    in batches of up to 1,000 candles.
    """

    start_ms = _to_milliseconds(
        start_time
    )

    if end_time is not None:
        end_ms = _to_milliseconds(
            end_time
        )
    else:
        end_ms = int(
            datetime.now(
                timezone.utc
            ).timestamp() * 1000
        )

    if end_ms <= start_ms:
        raise ValueError(
            "end_time must be later than start_time."
        )

    all_candles: list[list] = []
    next_start_ms = start_ms

    while next_start_ms < end_ms:
        batch = client.get_klines(
            symbol=symbol,
            interval=interval,
            startTime=next_start_ms,
            endTime=end_ms,
            limit=BINANCE_MAX_BATCH,
        )

        if not batch:
            break

        all_candles.extend(
            batch
        )

        last_open_time_ms = int(
            batch[-1][0]
        )

        next_start_ms = (
            last_open_time_ms + 1
        )

        if len(batch) < BINANCE_MAX_BATCH:
            break

    return all_candles


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

    Fixed-date mode:
        start_time is supplied, so the complete date
        range is downloaded using pagination.

    Latest-candle mode:
        start_time is omitted, so only the latest
        `limit` candles are downloaded.
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
        candles = _fetch_fixed_range(
            symbol=normalized_symbol,
            interval=interval,
            start_time=start_time,
            end_time=end_time,
        )
    else:
        candles = client.get_klines(
            symbol=normalized_symbol,
            interval=interval,
            limit=min(
                limit,
                BINANCE_MAX_BATCH,
            ),
        )

    if not candles:
        return _empty_candle_frame()

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
            dataframe["close_time"]
            <= current_time_ms
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
        .sort_values(
            "open_time"
        )
        .reset_index(
            drop=True
        )
    )

    # Only latest-candle mode is limited.
    # Fixed-date mode must preserve the whole range.
    if (
        start_time is None
        and len(dataframe) > limit
    ):
        dataframe = (
            dataframe
            .tail(limit)
            .reset_index(
                drop=True
            )
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

    data["VolumeSMA20"] = (
        calculate_volume_sma(
            data,
            20,
        )
    )

    return (
        data
        .dropna()
        .reset_index(
            drop=True
        )
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

    print("=" * 80)
    print(
        f"FA CRYPTO ENGINE — {symbol} DATA RANGE"
    )
    print("=" * 80)
    print(
        f"Candles : {len(data)}"
    )
    print(
        f"First   : {data['open_time'].min()}"
    )
    print(
        f"Last    : {data['open_time'].max()}"
    )

    indicator_data = add_indicators(
        data
    )

    if indicator_data.empty:
        raise ValueError(
            "No rows remain after indicator calculation."
        )

    latest_candle = (
        indicator_data.iloc[-1]
    )

    signal = (
        EmaRsiStrategyV2.generate_signal(
            latest_candle
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
