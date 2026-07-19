import numpy as np
import pandas as pd
def calculate_ema(
    dataframe: pd.DataFrame,
    period: int,
    column: str = "close",
) -> pd.Series:
    """
    Exponential Moving Average calculate karta hai.
    """
    return dataframe[column].ewm(
        span=period,
        adjust=False,
    ).mean()


def calculate_rsi(
    dataframe: pd.DataFrame,
    period: int = 14,
) -> pd.Series:
    """
    Close prices ke basis par RSI calculate karta hai.
    """

    price_change = dataframe["close"].diff()

    gains = price_change.clip(lower=0)
    losses = -price_change.clip(upper=0)

    average_gain = gains.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    average_loss = losses.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    relative_strength = average_gain / average_loss.replace(0, np.nan)

    rsi = 100 - (
        100 / (1 + relative_strength)
    )

    return rsi


def calculate_volume_sma(
    dataframe: pd.DataFrame,
    period: int = 20,
) -> pd.Series:
    """
    Trading volume ka simple moving average calculate karta hai.
    """
    return dataframe["volume"].rolling(
        window=period,
        min_periods=period,
    ).mean()


def calculate_adx(
    dataframe: pd.DataFrame,
    period: int = 14,
) -> pd.Series:
    """
    Average Directional Index calculate karta hai.
    """

    high = dataframe["high"]
    low = dataframe["low"]
    close = dataframe["close"]

    high_change = high.diff()
    low_change = -low.diff()

    plus_dm = pd.Series(
        np.where(
            (high_change > low_change)
            & (high_change > 0),
            high_change,
            0.0,
        ),
        index=dataframe.index,
    )

    minus_dm = pd.Series(
        np.where(
            (low_change > high_change)
            & (low_change > 0),
            low_change,
            0.0,
        ),
        index=dataframe.index,
    )

    true_range = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = true_range.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    smoothed_plus_dm = plus_dm.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    smoothed_minus_dm = minus_dm.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    plus_di = 100 * (
        smoothed_plus_dm / atr.replace(0, np.nan)
    )

    minus_di = 100 * (
        smoothed_minus_dm / atr.replace(0, np.nan)
    )

    directional_sum = (
        plus_di + minus_di
    ).replace(0, np.nan)

    dx = (
        100
        * (plus_di - minus_di).abs()
        / directional_sum
    )

    adx = dx.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    return adx
