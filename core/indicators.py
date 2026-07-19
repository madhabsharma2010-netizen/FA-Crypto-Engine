import pandas as pd
def calculate_ema(dataframe: pd.DataFrame,period: int, column: str = "close") -> pd.Series:
    """
    Exponential Moving Average (EMA) calculate karta hai.
    """
    return dataframe[column].ewm(span=period, adjust=False).mean()


def generate_signal(dataframe):
    """
    EMA20 aur EMA50 ke basis par signal generate karta hai.
    """

    latest = dataframe.iloc[-1]

    ema20 = latest["EMA20"]
    ema50 = latest["EMA50"]
    rsi = latest["RSI14"]


    if ema20 > ema50 and rsi < 70:
        return "BUY"

    elif ema20 < ema50 and rsi > 30:
        return "SELL"

    return "HOLD"

def calculate_rsi(dataframe, period=14):
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

    relative_strength = average_gain / average_loss

    rsi = 100 - (100 / (1 + relative_strength))

    return rsi
