from pathlib import Path
import os

import numpy as np
import pandas as pd

import run_v3ka_sol_ema_reload_engine as eng


WINDOW = os.environ.get(
    "V3G4_WINDOW",
    "2024",
).upper()

TAG = WINDOW.lower()

SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "LINKUSDT",
    "DOGEUSDT",
]

OUTPUT = Path(
    f"reports/v3ki_4h_donchian_{TAG}.csv"
)

BREAKOUT_BARS = 20
EXIT_BARS = 10
INITIAL_STOP_ATR = 2.0
EMA_SLOPE_BARS = 3

FEE_RATE = (
    float(eng.TRADING_FEE_PERCENT)
    / 100.0
)


def prepare_4h(frame_1h):
    bars = (
        frame_1h[
            [
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]
        ]
        .resample(
            "4h",
            label="right",
            closed="right",
        )
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        .dropna()
    )

    bars["EMA50"] = (
        bars["close"]
        .ewm(
            span=50,
            adjust=False,
            min_periods=50,
        )
        .mean()
    )

    bars["EMA200"] = (
        bars["close"]
        .ewm(
            span=200,
            adjust=False,
            min_periods=200,
        )
        .mean()
    )

    previous_close = (
        bars["close"].shift(1)
    )

    true_range = pd.concat(
        [
            bars["high"] - bars["low"],
            (
                bars["high"]
                - previous_close
            ).abs(),
            (
                bars["low"]
                - previous_close
            ).abs(),
        ],
        axis=1,
    ).max(axis=1)

    bars["ATR14"] = (
        true_range
        .ewm(
            alpha=1.0 / 14.0,
            adjust=False,
            min_periods=14,
        )
        .mean()
    )

    bars["PRIOR_HIGH_20"] = (
        bars["high"]
        .shift(1)
        .rolling(
            BREAKOUT_BARS,
            min_periods=BREAKOUT_BARS,
        )
        .max()
    )

    bars["PRIOR_LOW_10"] = (
        bars["low"]
        .shift(1)
        .rolling(
            EXIT_BARS,
            min_periods=EXIT_BARS,
        )
        .min()
    )

    bars["EMA50_RISING"] = (
        bars["EMA50"]
        > bars["EMA50"].shift(
            EMA_SLOPE_BARS
        )
    )

    bars["ENTRY_SIGNAL"] = (
        (
            bars["close"]
            > bars["PRIOR_HIGH_20"]
        )
        & (
            bars["close"]
            > bars["EMA50"]
        )
        & (
            bars["EMA50"]
            > bars["EMA200"]
        )
        & bars["EMA50_RISING"]
        & bars["ATR14"].notna()
        & bars["PRIOR_LOW_10"].notna()
    )

    return bars


def sell_price(raw_price):
    return raw_price * (
        1.0
        - float(
            eng.SLIPPAGE_PERCENT
        )
        / 100.0
    )


def close_trade(
    trades,
    position,
    raw_exit,
    exit_time,
    exit_reason,
):
    execution_exit = sell_price(
        float(raw_exit)
    )

    buy_fee = (
        position["entry_price"]
        * FEE_RATE
    )

    sell_fee = (
        execution_exit
        * FEE_RATE
    )

    net_per_unit = (
        execution_exit
        - position["entry_price"]
        - buy_fee
        - sell_fee
    )

    net_r = (
        net_per_unit
        / position["initial_risk"]
    )

    trades.append(
        {
            "window": WINDOW,
            "symbol": position["symbol"],
            "signal_time": (
                position["signal_time"]
            ),
            "entry_time": (
                position["entry_time"]
            ),
            "exit_time": exit_time,
            "entry_price": (
                position["entry_price"]
            ),
            "initial_stop": (
                position["initial_stop"]
            ),
            "final_stop": (
                position["stop_price"]
            ),
            "exit_price": execution_exit,
            "initial_risk": (
                position["initial_risk"]
            ),
            "exit_reason": exit_reason,
            "net_r": net_r,
            "holding_hours": (
                pd.Timestamp(exit_time)
                - pd.Timestamp(
                    position["entry_time"]
                )
            ).total_seconds()
            / 3600.0,
        }
    )


all_trades = []


for symbol in SYMBOLS:
    frame_1h = (
        eng.prepare_1h(symbol)
        .copy()
        .sort_index()
    )

    frame_1h.index = pd.to_datetime(
        frame_1h.index
    )

    bars_4h = prepare_4h(
        frame_1h
    )

    completed_4h = {
        timestamp: row
        for timestamp, row
        in bars_4h.iterrows()
    }

    position = None
    pending_entry = None
    pending_exit = None

    symbol_trades = []

    last_time = None
    last_close = None


    for event_time, candle in (
        frame_1h.iterrows()
    ):
        last_time = event_time
        last_close = float(
            candle["close"]
        )

        raw_open = float(
            candle["open"]
        )

        high_price = float(
            candle["high"]
        )

        low_price = float(
            candle["low"]
        )


        # ----------------------------------------------------
        # Market exit ordered by completed 4h information.
        # Executes at the next 1h open.
        # ----------------------------------------------------

        if (
            position is not None
            and pending_exit is not None
        ):
            close_trade(
                symbol_trades,
                position,
                raw_open,
                event_time,
                pending_exit,
            )

            position = None
            pending_exit = None


        # ----------------------------------------------------
        # Entry ordered by completed 4h information.
        # Executes at the next 1h open.
        # ----------------------------------------------------

        if (
            position is None
            and pending_entry is not None
        ):
            entry_price = (
                eng.apply_buy_slippage(
                    raw_open
                )
            )

            atr_4h = float(
                pending_entry["ATR14"]
            )

            initial_stop = (
                entry_price
                - INITIAL_STOP_ATR
                * atr_4h
            )

            initial_risk = (
                entry_price
                - initial_stop
            )

            if initial_risk > 0:
                position = {
                    "symbol": symbol,
                    "signal_time": (
                        pending_entry[
                            "signal_time"
                        ]
                    ),
                    "entry_time": event_time,
                    "entry_price": (
                        entry_price
                    ),
                    "initial_stop": (
                        initial_stop
                    ),
                    "stop_price": (
                        initial_stop
                    ),
                    "initial_risk": (
                        initial_risk
                    ),
                }

            pending_entry = None


        # ----------------------------------------------------
        # Hard-stop check.
        # Open-gap fill is conservatively taken at the worse
        # of stop or current open.
        # ----------------------------------------------------

        if position is not None:
            stop_price = float(
                position["stop_price"]
            )

            if low_price <= stop_price:
                raw_stop_fill = min(
                    stop_price,
                    raw_open,
                )

                close_trade(
                    symbol_trades,
                    position,
                    raw_stop_fill,
                    event_time,
                    "HARD_OR_TRAILING_STOP",
                )

                position = None
                pending_exit = None


        # ----------------------------------------------------
        # Completed 4h decision processing.
        # New information applies only to future 1h bars.
        # ----------------------------------------------------

        if event_time in completed_4h:
            bar_4h = completed_4h[
                event_time
            ]

            if position is not None:
                prior_low = bar_4h[
                    "PRIOR_LOW_10"
                ]

                if pd.notna(prior_low):
                    candidate_stop = float(
                        prior_low
                    )

                    position[
                        "stop_price"
                    ] = max(
                        float(
                            position[
                                "stop_price"
                            ]
                        ),
                        candidate_stop,
                    )

                    if (
                        float(
                            bar_4h["close"]
                        )
                        <= candidate_stop
                    ):
                        pending_exit = (
                            "4H_DONCHIAN_EXIT"
                        )

            if (
                position is None
                and pending_entry is None
                and bool(
                    bar_4h[
                        "ENTRY_SIGNAL"
                    ]
                )
            ):
                pending_entry = {
                    "signal_time": (
                        event_time
                    ),
                    "ATR14": float(
                        bar_4h["ATR14"]
                    ),
                }


    if (
        position is not None
        and last_time is not None
        and last_close is not None
    ):
        close_trade(
            symbol_trades,
            position,
            last_close,
            last_time,
            "END_MARK",
        )

    all_trades.extend(
        symbol_trades
    )


detail = pd.DataFrame(
    all_trades
)

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

detail.to_csv(
    OUTPUT,
    index=False,
)


print()
print("=" * 108)
print(
    f"V3KI 4H DONCHIAN TREND | "
    f"{WINDOW}"
)
print("=" * 108)

if detail.empty:
    print("No completed trades.")
else:
    winners = detail[
        detail["net_r"] > 0
    ]

    losers = detail[
        detail["net_r"] < 0
    ]

    positive_r = float(
        winners["net_r"].sum()
    )

    negative_r = abs(
        float(
            losers["net_r"].sum()
        )
    )

    profit_factor = (
        positive_r / negative_r
        if negative_r > 0
        else float("inf")
        if positive_r > 0
        else 0.0
    )

    print(
        f"Completed trades             : "
        f"{len(detail)}"
    )

    print(
        f"Winners / Losers            : "
        f"{len(winners)} / "
        f"{len(losers)}"
    )

    print(
        f"Total net R                 : "
        f"{detail['net_r'].sum():+.2f}R"
    )

    print(
        f"Average net R               : "
        f"{detail['net_r'].mean():+.3f}R"
    )

    print(
        f"Profit factor               : "
        f"{profit_factor:.2f}"
    )

    print("-" * 108)
    print("BY SYMBOL")
    print("-" * 108)

    for symbol in SYMBOLS:
        group = detail[
            detail["symbol"] == symbol
        ]

        if group.empty:
            print(
                f"{symbol:<10}"
                f"Trades   0 | "
                f"Net R   +0.00"
            )

            continue

        print(
            f"{symbol:<10}"
            f"Trades {len(group):>3} | "
            f"Wins "
            f"{int((group['net_r'] > 0).sum()):>2} | "
            f"Net R "
            f"{group['net_r'].sum():>+8.2f}"
        )

print("-" * 108)
print(
    f"Detail report               : "
    f"{OUTPUT}"
)
print("=" * 108)
