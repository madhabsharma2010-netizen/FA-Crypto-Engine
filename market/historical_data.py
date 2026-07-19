from binance.client import Client
import pandas as pd

from core.indicators import (
    calculate_ema,
    calculate_rsi,
    calculate_adx,
    calculate_atr,
    calculate_volume_sma,
)
from strategies.ema_rsi_strategy_v2 import EmaRsiStrategyV2


client = Client()


def get_historical_candles(
    symbol: str = "BTCUSDT",
    interval: str = Client.KLINE_INTERVAL_1HOUR,
    limit: int = 1000,
) -> pd.DataFrame:
    """
    Fetch historical candle data from Binance.
    """

    if not symbol:
        raise ValueError(
            "Symbol cannot be empty."
        )

    if limit <= 0:
        raise ValueError(
            "Limit must be greater than zero."
        )

    candles = client.get_klines(
        symbol=symbol.upper(),
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
    ]

    for column in numeric_columns:
        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

    dataframe["open_time"] = pd.to_datetime(
        dataframe["open_time"],
        unit="ms",
    )

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

    dataframe = dataframe.dropna().reset_index(
        drop=True
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