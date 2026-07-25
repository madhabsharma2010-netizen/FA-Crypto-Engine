from pathlib import Path

import numpy as np
import pandas as pd
from binance.client import Client

from market.historical_data import get_historical_candles


WINDOWS = {
    "2021": ("2021-01-01", "2022-01-01"),
    "2022": ("2022-01-01", "2023-01-01"),
    "2023": ("2023-01-01", "2024-01-01"),
    "2024": ("2024-01-01", "2025-01-01"),
    "2025h1": ("2025-01-01", "2025-08-01"),
}

TARGET_SYMBOLS = {
    "LINKUSDT",
    "SOLUSDT",
}


def load_trades() -> pd.DataFrame:
    frames = []

    for path in sorted(
        Path("logs/backtests").glob(
            "v3g4_eth365_robustness_*_baseline_trades.csv"
        )
    ):
        window = (
            path.stem
            .replace("v3g4_eth365_robustness_", "")
            .replace("_baseline_trades", "")
        )

        if window not in WINDOWS:
            continue

        df = pd.read_csv(path)

        df = df[
            df["symbol"].isin(TARGET_SYMBOLS)
        ].copy()

        if df.empty:
            continue

        df["window"] = window
        df["entry_time"] = pd.to_datetime(
            df["entry_time"]
        )
        df["exit_time"] = pd.to_datetime(
            df["exit_time"]
        )

        frames.append(df)

    if not frames:
        raise RuntimeError(
            "No SOL/LINK baseline trades found."
        )

    return pd.concat(
        frames,
        ignore_index=True,
    )


trades = load_trades()

market_data = {}

print()
print("=" * 100)
print("V3H - LOADING 15M MARKET DATA")
print("=" * 100)

for window, (start_time, end_time) in WINDOWS.items():

    window_trades = trades[
        trades["window"] == window
    ]

    for symbol in sorted(TARGET_SYMBOLS):

        if not (
            window_trades["symbol"] == symbol
        ).any():
            continue

        print(
            f"Loading {window} {symbol}..."
        )

        candles = get_historical_candles(
            symbol=symbol,
            interval=Client.KLINE_INTERVAL_15MINUTE,
            start_time=start_time,
            end_time=end_time,
        )

        if candles.empty:
            raise RuntimeError(
                f"No candles: {window} {symbol}"
            )

        market_data[
            (window, symbol)
        ] = candles


rows = []

for _, trade in trades.iterrows():

    window = trade["window"]
    symbol = trade["symbol"]

    candles = market_data[
        (window, symbol)
    ]

    entry_time = trade["entry_time"]
    exit_time = trade["exit_time"]

    entry_price = float(
        trade["entry_price"]
    )
    initial_stop = float(
        trade["initial_stop"]
    )

    risk_per_unit = (
        entry_price - initial_stop
    )

    if risk_per_unit <= 0:
        continue

    # IMPORTANT:
    # Exit candle itself is deliberately excluded.
    #
    # A stop/shock may occur intrabar. Using the full
    # exit candle's later high/low could introduce
    # intrabar lookahead.
    #
    # Therefore excursion is measured only using
    # complete 15m candles strictly BEFORE exit_time.
    path = candles[
        (candles["open_time"] >= entry_time)
        & (candles["open_time"] < exit_time)
    ].copy()

    if path.empty:
        max_high = entry_price
        min_low = entry_price
        bars = 0
    else:
        max_high = float(
            path["high"].max()
        )
        min_low = float(
            path["low"].min()
        )
        bars = len(path)

    mfe_price = max(
        0.0,
        max_high - entry_price,
    )

    mae_price = max(
        0.0,
        entry_price - min_low,
    )

    mfe_r = (
        mfe_price / risk_per_unit
    )

    mae_r = (
        mae_price / risk_per_unit
    )

    mfe_percent = (
        mfe_price
        / entry_price
        * 100.0
    )

    mae_percent = (
        mae_price
        / entry_price
        * 100.0
    )

    profit = float(
        trade["profit"]
    )

    quantity = float(
        trade["initial_quantity"]
    )

    initial_risk_eur = (
        risk_per_unit * quantity
    )

    realized_r = (
        profit / initial_risk_eur
        if initial_risk_eur > 0
        else np.nan
    )

    reached_05r = (
        mfe_r >= 0.5
    )

    reached_1r = (
        mfe_r >= 1.0
    )

    reached_2r = (
        mfe_r >= 2.0
    )

    rows.append(
        {
            "window": window,
            "symbol": symbol,
            "route": trade["route"],
            "entry_time": entry_time,
            "exit_time": exit_time,
            "exit_reason": trade[
                "exit_reason"
            ],
            "profit": profit,
            "realized_r": realized_r,
            "strict_mfe_r": mfe_r,
            "strict_mae_r": mae_r,
            "strict_mfe_percent": (
                mfe_percent
            ),
            "strict_mae_percent": (
                mae_percent
            ),
            "reached_0_5r": reached_05r,
            "reached_1r": reached_1r,
            "reached_2r": reached_2r,
            "partial_profit_taken": trade[
                "partial_profit_taken"
            ],
            "bars_before_exit": bars,
        }
    )


detail = pd.DataFrame(rows)

if detail.empty:
    raise RuntimeError(
        "No excursion results generated."
    )


def percentage(series: pd.Series) -> float:
    if len(series) == 0:
        return 0.0

    return float(
        series.astype(bool).mean()
        * 100.0
    )


summary_rows = []

for (
    window,
    symbol,
), group in detail.groupby(
    ["window", "symbol"],
    sort=False,
):

    losers = group[
        group["profit"] <= 0
    ]

    winners = group[
        group["profit"] > 0
    ]

    summary_rows.append(
        {
            "window": window,
            "symbol": symbol,
            "trades": len(group),
            "losers": len(losers),
            "median_mfe_r": (
                group["strict_mfe_r"].median()
            ),
            "median_mae_r": (
                group["strict_mae_r"].median()
            ),
            "pct_reached_0_5r": percentage(
                group["reached_0_5r"]
            ),
            "pct_reached_1r": percentage(
                group["reached_1r"]
            ),
            "pct_reached_2r": percentage(
                group["reached_2r"]
            ),
            "loser_median_mfe_r": (
                losers[
                    "strict_mfe_r"
                ].median()
                if len(losers)
                else np.nan
            ),
            "losers_reached_0_5r_pct": (
                percentage(
                    losers[
                        "reached_0_5r"
                    ]
                )
            ),
            "losers_reached_1r_pct": (
                percentage(
                    losers[
                        "reached_1r"
                    ]
                )
            ),
            "losers_reached_2r_pct": (
                percentage(
                    losers[
                        "reached_2r"
                    ]
                )
            ),
            "winner_median_mfe_r": (
                winners[
                    "strict_mfe_r"
                ].median()
                if len(winners)
                else np.nan
            ),
        }
    )


summary = pd.DataFrame(
    summary_rows
)


print()
print("=" * 120)
print("V3H STRICT PRE-EXIT MFE / MAE")
print("=" * 120)

print(
    summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}",
    )
)


print()
print("=" * 120)
print("TOTAL BY ROUTE")
print("=" * 120)

total_rows = []

for symbol, group in detail.groupby(
    "symbol"
):

    losers = group[
        group["profit"] <= 0
    ]

    winners = group[
        group["profit"] > 0
    ]

    total_rows.append(
        {
            "symbol": symbol,
            "trades": len(group),
            "losers": len(losers),
            "median_mfe_r": (
                group[
                    "strict_mfe_r"
                ].median()
            ),
            "median_mae_r": (
                group[
                    "strict_mae_r"
                ].median()
            ),
            "reached_0_5r_pct": percentage(
                group["reached_0_5r"]
            ),
            "reached_1r_pct": percentage(
                group["reached_1r"]
            ),
            "reached_2r_pct": percentage(
                group["reached_2r"]
            ),
            "loser_median_mfe_r": (
                losers[
                    "strict_mfe_r"
                ].median()
            ),
            "losers_reached_0_5r_pct": (
                percentage(
                    losers[
                        "reached_0_5r"
                    ]
                )
            ),
            "losers_reached_1r_pct": (
                percentage(
                    losers[
                        "reached_1r"
                    ]
                )
            ),
            "losers_reached_2r_pct": (
                percentage(
                    losers[
                        "reached_2r"
                    ]
                )
            ),
            "winner_median_mfe_r": (
                winners[
                    "strict_mfe_r"
                ].median()
            ),
        }
    )

total = pd.DataFrame(
    total_rows
)

print(
    total.to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}",
    )
)


print()
print("=" * 120)
print("LOSING TRADES THAT FIRST HAD >= 1R FAVORABLE EXCURSION")
print("=" * 120)

gave_back = detail[
    (detail["profit"] <= 0)
    & (detail["reached_1r"])
].sort_values(
    "strict_mfe_r",
    ascending=False,
)

if gave_back.empty:
    print(
        "None."
    )
else:
    print(
        gave_back[
            [
                "window",
                "symbol",
                "entry_time",
                "exit_time",
                "profit",
                "realized_r",
                "strict_mfe_r",
                "strict_mae_r",
                "partial_profit_taken",
                "exit_reason",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )


Path("reports").mkdir(
    parents=True,
    exist_ok=True,
)

detail.to_csv(
    "reports/v3h_loss_excursion_detail.csv",
    index=False,
)

summary.to_csv(
    "reports/v3h_loss_excursion_summary.csv",
    index=False,
)

print()
print(
    "Saved: reports/v3h_loss_excursion_detail.csv"
)
print(
    "Saved: reports/v3h_loss_excursion_summary.csv"
)
