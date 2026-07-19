import pandas as pd


def calculate_ema(
    dataframe: pd.DataFrame,
    period: int,
) -> pd.Series:
    """
    Calculate Exponential Moving Average.
    """

    if period <= 0:
        raise ValueError(
            "EMA period must be greater than zero."
        )

    if "close" not in dataframe.columns:
        raise KeyError(
            "Dataframe must contain a 'close' column."
        )

    close = pd.to_numeric(
        dataframe["close"],
        errors="coerce",
    )

    return close.ewm(
        span=period,
        adjust=False,
    ).mean()


def calculate_rsi(
    dataframe: pd.DataFrame,
    period: int = 14,
) -> pd.Series:
    """
    Calculate Relative Strength Index using
    Wilder-style exponential smoothing.
    """

    if period <= 0:
        raise ValueError(
            "RSI period must be greater than zero."
        )

    if "close" not in dataframe.columns:
        raise KeyError(
            "Dataframe must contain a 'close' column."
        )

    close = pd.to_numeric(
        dataframe["close"],
        errors="coerce",
    )

    price_change = close.diff()

    gains = price_change.clip(
        lower=0,
    )

    losses = (
        -price_change.clip(
            upper=0,
        )
    )

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

    relative_strength = (
        average_gain
        / average_loss.replace(0, float("nan"))
    )

    rsi = (
        100
        - (
            100
            / (
                1
                + relative_strength
            )
        )
    )

    no_losses = (
        average_loss == 0
    )

    no_gains = (
        average_gain == 0
    )

    rsi = rsi.mask(
        no_losses & ~no_gains,
        100.0,
    )

    rsi = rsi.mask(
        no_gains & ~no_losses,
        0.0,
    )

    rsi = rsi.mask(
        no_gains & no_losses,
        50.0,
    )

    return rsi


def calculate_volume_sma(
    dataframe: pd.DataFrame,
    period: int = 20,
) -> pd.Series:
    """
    Calculate Simple Moving Average of volume.
    """

    if period <= 0:
        raise ValueError(
            "Volume SMA period must be greater than zero."
        )

    if "volume" not in dataframe.columns:
        raise KeyError(
            "Dataframe must contain a 'volume' column."
        )

    volume = pd.to_numeric(
        dataframe["volume"],
        errors="coerce",
    )

    return volume.rolling(
        window=period,
        min_periods=period,
    ).mean()


def calculate_atr(
    dataframe: pd.DataFrame,
    period: int = 14,
) -> pd.Series:
    """
    Calculate Average True Range using
    Wilder-style smoothing.
    """

    if period <= 0:
        raise ValueError(
            "ATR period must be greater than zero."
        )

    required_columns = {
        "high",
        "low",
        "close",
    }

    missing_columns = (
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:
        raise KeyError(
            "Dataframe is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    high = pd.to_numeric(
        dataframe["high"],
        errors="coerce",
    )

    low = pd.to_numeric(
        dataframe["low"],
        errors="coerce",
    )

    close = pd.to_numeric(
        dataframe["close"],
        errors="coerce",
    )

    previous_close = close.shift(1)

    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(
        axis=1,
    )

    atr = true_range.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    return atr


def calculate_adx(
    dataframe: pd.DataFrame,
    period: int = 14,
) -> pd.Series:
    """
    Calculate Average Directional Index.

    ADX measures trend strength.
    It does not indicate trend direction.
    """

    if period <= 0:
        raise ValueError(
            "ADX period must be greater than zero."
        )

    required_columns = {
        "high",
        "low",
        "close",
    }

    missing_columns = (
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:
        raise KeyError(
            "Dataframe is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    high = pd.to_numeric(
        dataframe["high"],
        errors="coerce",
    )

    low = pd.to_numeric(
        dataframe["low"],
        errors="coerce",
    )

    close = pd.to_numeric(
        dataframe["close"],
        errors="coerce",
    )

    upward_move = high.diff()

    downward_move = (
        -low.diff()
    )

    positive_dm = upward_move.where(
        (
            upward_move > downward_move
        )
        & (
            upward_move > 0
        ),
        0.0,
    )

    negative_dm = downward_move.where(
        (
            downward_move > upward_move
        )
        & (
            downward_move > 0
        ),
        0.0,
    )

    previous_close = close.shift(1)

    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(
        axis=1,
    )

    atr = true_range.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    smoothed_positive_dm = (
        positive_dm.ewm(
            alpha=1 / period,
            adjust=False,
            min_periods=period,
        ).mean()
    )

    smoothed_negative_dm = (
        negative_dm.ewm(
            alpha=1 / period,
            adjust=False,
            min_periods=period,
        ).mean()
    )

    positive_di = (
        100
        * smoothed_positive_dm
        / atr.replace(
            0,
            float("nan"),
        )
    )

    negative_di = (
        100
        * smoothed_negative_dm
        / atr.replace(
            0,
            float("nan"),
        )
    )

    directional_sum = (
        positive_di
        + negative_di
    )

    directional_difference = (
        positive_di
        - negative_di
    ).abs()

    dx = (
        100
        * directional_difference
        / directional_sum.replace(
            0,
            float("nan"),
        )
    )

    adx = dx.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    return adx