from binance.client import Client
import pandas as pd

from core.indicators import calculate_ema, calculate_rsi


client = Client()


def get_historical_candles(
    symbol: str = "BTCUSDT",
    interval: str = Client.KLINE_INTERVAL_1HOUR,
    limit: int = 100,
) -> pd.DataFrame:
    """
    Binance se historical candle data fetch karta hai.
    """

    candles = client.get_klines(
        symbol=symbol,
        interval=interval,
        limit=limit,
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

    return dataframe[
        [
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
    ]


def main() -> None:
    symbol = "BTCUSDT"

    data = get_historical_candles(
        symbol=symbol,
        interval=Client.KLINE_INTERVAL_1HOUR,
        limit=100,
    )

    data["EMA20"] = calculate_ema(data, 20)
    data["EMA50"] = calculate_ema(data, 50)
    data["RSI14"] = calculate_rsi(data, 14)

    signal = generate_signal(data)

    print("=" * 80)
    print(f"FA CRYPTO ENGINE — {symbol} INDICATORS")
    print("=" * 80)

    print(
        data[
            [
                "open_time",
                "close",
                "EMA20",
                "EMA50",
                "RSI14",
            ]
        ].tail(20).to_string(index=False)
    )

    print("=" * 80)
    print("CURRENT SIGNAL:", signal)
    print("=" * 80)


if __name__ == "__main__":
    main()
