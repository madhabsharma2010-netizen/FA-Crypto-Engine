from __future__ import annotations

from pathlib import Path

import pandas as pd
from binance.client import Client

from market.historical_data import get_historical_candles


SYMBOLS = (
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "LINKUSDT",
    "DOGEUSDT",
)

# Development only.
# May/June 2026 remain untouched validation data.
DEV_START = pd.Timestamp("2025-08-01 00:00:00")
DEV_END = pd.Timestamp("2026-05-01 00:00:00")

# Enough history for a complete trailing 365-day high.
WARMUP_START = DEV_START - pd.Timedelta(days=366)

# Used only to establish the genuine historical ATH before
# the hourly warmup window.
TRUE_HISTORY_START = pd.Timestamp("2017-01-01 00:00:00")

DETAIL_FILE = Path(
    "reports/v3g1_ath_signal_detail.csv"
)

SUMMARY_FILE = Path(
    "reports/v3g1_ath_signal_summary.csv"
)


def load_hourly(symbol: str) -> pd.DataFrame:
    data = get_historical_candles(
        symbol=symbol,
        interval=Client.KLINE_INTERVAL_1HOUR,
        limit=1000,
        start_time=WARMUP_START.strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        end_time=DEV_END.strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        drop_incomplete=True,
    )

    if data.empty:
        raise ValueError(
            f"No hourly data for {symbol}."
        )

    for column in (
        "open",
        "high",
        "low",
        "close",
        "volume",
    ):
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    data["open_time"] = pd.to_datetime(
        data["open_time"]
    )

    data["completion_time"] = (
        data["open_time"]
        + pd.Timedelta(hours=1)
    )

    return (
        data
        .dropna()
        .sort_values("open_time")
        .reset_index(drop=True)
    )


def historical_ath_seed(
    symbol: str,
) -> float:
    data = get_historical_candles(
        symbol=symbol,
        interval=Client.KLINE_INTERVAL_1DAY,
        limit=1000,
        start_time=TRUE_HISTORY_START.strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        end_time=WARMUP_START.strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        drop_incomplete=True,
    )

    if data.empty:
        raise ValueError(
            f"No historical ATH seed data for {symbol}."
        )

    data["high"] = pd.to_numeric(
        data["high"],
        errors="coerce",
    )

    data["open_time"] = pd.to_datetime(
        data["open_time"]
    )

    data = data[
        data["open_time"] < WARMUP_START
    ]

    if data.empty:
        raise ValueError(
            f"No pre-warmup history for {symbol}."
        )

    return float(
        data["high"].max()
    )


def add_ath_signals(
    data: pd.DataFrame,
    true_ath_seed: float,
) -> pd.DataFrame:
    result = data.copy()

    # ---------------------------------------------------------
    # TRUE ATH
    #
    # Current candle high is NEVER included in the breakout
    # level. Only information available before this candle.
    # ---------------------------------------------------------

    prior_running_high = (
        result["high"]
        .shift(1)
        .cummax()
    )

    result["PRIOR_TRUE_ATH"] = (
        prior_running_high
        .clip(lower=true_ath_seed)
        .fillna(true_ath_seed)
    )

    previous_close = (
        result["close"].shift(1)
    )

    previous_true_ath = (
        result["PRIOR_TRUE_ATH"].shift(1)
    )

    result["TRUE_ATH_SIGNAL"] = (
        (
            result["close"]
            > result["PRIOR_TRUE_ATH"]
        )
        & (
            previous_close
            <= previous_true_ath
        )
    )

    # ---------------------------------------------------------
    # 365-DAY HIGH
    #
    # Previous 8,760 completed hourly candles only.
    # Current candle is excluded with shift(1).
    # ---------------------------------------------------------

    hours_365 = 24 * 365

    result["PRIOR_365D_HIGH"] = (
        result["high"]
        .shift(1)
        .rolling(
            hours_365,
            min_periods=hours_365,
        )
        .max()
    )

    previous_365_high = (
        result["PRIOR_365D_HIGH"]
        .shift(1)
    )

    result["HIGH_365D_SIGNAL"] = (
        (
            result["close"]
            > result["PRIOR_365D_HIGH"]
        )
        & (
            previous_close
            <= previous_365_high
        )
    )

    # Entry is the NEXT candle open.
    result["NEXT_OPEN"] = (
        result["open"].shift(-1)
    )

    # Pure forward-edge diagnostics.
    # These are outcomes, never entry filters.
    for hours in (6, 24, 72):
        future_close = (
            result["close"].shift(-hours)
        )

        result[
            f"FORWARD_{hours}H_RETURN_PCT"
        ] = (
            (
                future_close
                / result["NEXT_OPEN"]
                - 1.0
            )
            * 100.0
        )

    return result


def signal_rows(
    symbol: str,
    data: pd.DataFrame,
    variant: str,
    signal_column: str,
    level_column: str,
) -> list[dict]:
    development = data[
        (data["completion_time"] >= DEV_START)
        & (data["completion_time"] < DEV_END)
        & data[signal_column]
    ].copy()

    rows: list[dict] = []

    for _, row in development.iterrows():
        level = float(
            row[level_column]
        )

        signal_close = float(
            row["close"]
        )

        rows.append(
            {
                "symbol": symbol,
                "variant": variant,
                "signal_time": (
                    row["completion_time"]
                ),
                "signal_close": signal_close,
                "breakout_level": level,
                "breakout_percent": (
                    (
                        signal_close
                        / level
                        - 1.0
                    )
                    * 100.0
                ),
                "next_open": row["NEXT_OPEN"],
                "forward_6h_return_pct": (
                    row[
                        "FORWARD_6H_RETURN_PCT"
                    ]
                ),
                "forward_24h_return_pct": (
                    row[
                        "FORWARD_24H_RETURN_PCT"
                    ]
                ),
                "forward_72h_return_pct": (
                    row[
                        "FORWARD_72H_RETURN_PCT"
                    ]
                ),
            }
        )

    return rows


def summarize(
    detail: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for symbol in SYMBOLS:
        for variant in (
            "TRUE_ATH",
            "365D_HIGH",
        ):
            group = detail[
                (detail["symbol"] == symbol)
                & (
                    detail["variant"]
                    == variant
                )
            ]

            def average(column: str) -> float:
                if group.empty:
                    return 0.0

                return float(
                    group[column]
                    .dropna()
                    .mean()
                )

            def positive_rate(
                column: str,
            ) -> float:
                values = (
                    group[column]
                    .dropna()
                )

                if values.empty:
                    return 0.0

                return float(
                    (values > 0).mean()
                    * 100.0
                )

            rows.append(
                {
                    "symbol": symbol,
                    "variant": variant,
                    "signals": len(group),
                    "avg_breakout_percent": (
                        average(
                            "breakout_percent"
                        )
                    ),
                    "avg_forward_6h_pct": (
                        average(
                            "forward_6h_return_pct"
                        )
                    ),
                    "positive_6h_percent": (
                        positive_rate(
                            "forward_6h_return_pct"
                        )
                    ),
                    "avg_forward_24h_pct": (
                        average(
                            "forward_24h_return_pct"
                        )
                    ),
                    "positive_24h_percent": (
                        positive_rate(
                            "forward_24h_return_pct"
                        )
                    ),
                    "avg_forward_72h_pct": (
                        average(
                            "forward_72h_return_pct"
                        )
                    ),
                    "positive_72h_percent": (
                        positive_rate(
                            "forward_72h_return_pct"
                        )
                    ),
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    print("=" * 110)
    print(
        "FA CRYPTO ENGINE — V3G1 ATH "
        "ENTRY-EDGE DIAGNOSTIC"
    )
    print("=" * 110)

    print(
        f"Development window : "
        f"{DEV_START} -> {DEV_END}"
    )

    print(
        "Risk engine        : UNCHANGED / NOT USED"
    )

    print(
        "Validation data    : MAY/JUNE 2026 UNTOUCHED"
    )

    print("=" * 110)

    all_rows: list[dict] = []

    for symbol in SYMBOLS:
        print()
        print(
            f"Loading {symbol} history..."
        )

        seed = historical_ath_seed(
            symbol
        )

        hourly = load_hourly(
            symbol
        )

        diagnostic = add_ath_signals(
            hourly,
            seed,
        )

        true_rows = signal_rows(
            symbol=symbol,
            data=diagnostic,
            variant="TRUE_ATH",
            signal_column="TRUE_ATH_SIGNAL",
            level_column="PRIOR_TRUE_ATH",
        )

        yearly_rows = signal_rows(
            symbol=symbol,
            data=diagnostic,
            variant="365D_HIGH",
            signal_column="HIGH_365D_SIGNAL",
            level_column="PRIOR_365D_HIGH",
        )

        all_rows.extend(true_rows)
        all_rows.extend(yearly_rows)

        print(
            f"{symbol:<10} | "
            f"Historical ATH seed "
            f"{seed:.8f} | "
            f"TRUE ATH signals "
            f"{len(true_rows):>3} | "
            f"365D signals "
            f"{len(yearly_rows):>3}"
        )

    detail = pd.DataFrame(
        all_rows
    )

    DETAIL_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    detail.to_csv(
        DETAIL_FILE,
        index=False,
    )

    summary = summarize(
        detail
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    print()
    print("=" * 110)
    print("V3G1 SUMMARY")
    print("=" * 110)

    if summary.empty:
        print(
            "No ATH breakout signals found."
        )
    else:
        print(
            summary.to_string(
                index=False
            )
        )

    print("=" * 110)
    print(
        f"Detail  : {DETAIL_FILE}"
    )
    print(
        f"Summary : {SUMMARY_FILE}"
    )
    print("=" * 110)


if __name__ == "__main__":
    main()
