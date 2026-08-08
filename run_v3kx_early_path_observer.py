from __future__ import annotations

from collections import Counter
from pathlib import Path
import os

import pandas as pd

import run_v3ka_sol_ema_reload_engine as eng

from v3kj_persistent_risk_guard import (
    assert_trading_allowed,
    load_or_create_state,
    update_equity_and_lock,
)


# ============================================================================
# V3KJ SHARED-PORTFOLIO BACKTEST
#
# Entry and normal-exit rules are copied from V3KI without threshold tuning.
# Portfolio controls are applied across all six assets on one hourly timeline.
# ============================================================================


WINDOW = os.environ.get(
    "V3G4_WINDOW",
    "2024",
).upper()

TAG = WINDOW.lower()

# ============================================================================
# V3KV OPPORTUNITY OBSERVER
# ============================================================================

V3KV_EVENT_OUTPUT = Path(
    "reports/"
    f"v3kx_early_path_observer_{TAG}_events.csv"
)

V3KV_DISPOSITION_OUTPUT = Path(
    "reports/"
    f"v3kx_early_path_observer_{TAG}_dispositions.csv"
)

v3kx_early_path_rows: list[
    dict[str, object]
] = []


def v3kv_record_due(
    context: dict[str, object],
    disposition: str,
    **extra: object,
) -> None:
    row = dict(
        context
    )

    row["disposition"] = (
        disposition
    )

    row.update(
        extra
    )

    v3kx_early_path_rows.append(
        row
    )




WINDOW_BOUNDS = {
    "2022": (
        pd.Timestamp("2022-01-01 00:00:00"),
        pd.Timestamp("2023-01-01 00:00:00"),
    ),
    "2023": (
        pd.Timestamp("2023-01-01 00:00:00"),
        pd.Timestamp("2024-01-01 00:00:00"),
    ),
    "2024": (
        pd.Timestamp("2024-01-01 00:00:00"),
        pd.Timestamp("2025-01-01 00:00:00"),
    ),
    "2025H1": (
        pd.Timestamp("2025-01-01 00:00:00"),
        pd.Timestamp("2025-08-01 00:00:00"),
    ),
}


if WINDOW not in WINDOW_BOUNDS:
    raise ValueError(
        f"Unsupported V3KJ window: {WINDOW}"
    )


TEST_START, TEST_END = WINDOW_BOUNDS[
    WINDOW
]


SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "LINKUSDT",
    "DOGEUSDT",
]


SYMBOL_ORDER = {
    symbol: index
    for index, symbol in enumerate(
        SYMBOLS
    )
}


HIGH_BETA_ASSETS = {
    "SOLUSDT",
    "XRPUSDT",
    "LINKUSDT",
    "DOGEUSDT",
}


# Exact V3KI signal/exit constants.
BREAKOUT_BARS = 20
EXIT_BARS = 10
INITIAL_STOP_ATR = 2.0
EMA_SLOPE_BARS = 3


# Frozen portfolio controls.
MAX_OPEN_POSITIONS = 2
MAX_TOTAL_DEPLOYMENT_PERCENT = 50.0
MAX_TOTAL_OPEN_RISK_PERCENT = 0.75
MIN_NOTIONAL_EUR = 500.0
MAX_ONE_NEW_ENTRY_PER_HOUR = True

DAILY_LOSS_LIMIT_PERCENT = 1.0
WEEKLY_LOSS_LIMIT_PERCENT = 2.5
HARD_DRAWDOWN_LIMIT_PERCENT = 5.0
DAILY_PROFIT_ENTRY_LOCK_PERCENT = 2.0


BASE_RISK_PERCENT = {
    "BTCUSDT": 0.45,
    "ETHUSDT": 0.40,
    "SOLUSDT": 0.30,
    "XRPUSDT": 0.30,
    "LINKUSDT": 0.30,
    "DOGEUSDT": 0.15,
}


ASSET_NOTIONAL_CAP_PERCENT = {
    "BTCUSDT": 40.0,
    "ETHUSDT": 35.0,
    "SOLUSDT": 25.0,
    "XRPUSDT": 25.0,
    "LINKUSDT": 25.0,
    "DOGEUSDT": 15.0,
}


STARTING_CAPITAL = float(
    eng.STARTING_CAPITAL
)

TRADING_FEE_PERCENT = float(
    eng.TRADING_FEE_PERCENT
)

SLIPPAGE_PERCENT = float(
    eng.SLIPPAGE_PERCENT
)

FEE_RATE = (
    TRADING_FEE_PERCENT
    / 100.0
)


TRADE_OUTPUT = Path(
    f"reports/"
    f"v3kx_early_path_{TAG}_trades.csv"
)

EQUITY_OUTPUT = Path(
    f"reports/"
    f"v3kx_early_path_{TAG}_equity.csv"
)

SUMMARY_OUTPUT = Path(
    f"reports/"
    f"v3kx_early_path_{TAG}_summary.csv"
)

STATE_FILE = Path(
    f"state/backtests/"
    f"v3kx_early_path_{TAG}_risk_guard.json"
)


RESET_BACKTEST_STATE = (
    os.environ.get(
        "V3KJ_RESET_BACKTEST_STATE",
        "1",
    )
    == "1"
)


# ============================================================================
# V3KI INDICATORS â€” UNCHANGED
# ============================================================================


def prepare_4h(
    frame_1h: pd.DataFrame,
) -> pd.DataFrame:

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


# ============================================================================
# EXECUTION AND PORTFOLIO HELPERS
# ============================================================================


def sell_price(
    raw_price: float,
) -> float:

    return float(raw_price) * (
        1.0
        - SLIPPAGE_PERCENT
        / 100.0
    )


def calculate_fee(
    notional: float,
) -> float:

    return float(notional) * FEE_RATE


def portfolio_market_value(
    positions: dict,
    prices: dict[str, float],
) -> float:

    return sum(
        float(position["quantity"])
        * float(prices[symbol])
        for symbol, position
        in positions.items()
    )


def portfolio_equity(
    cash_balance: float,
    positions: dict,
    prices: dict[str, float],
) -> float:

    return (
        float(cash_balance)
        + portfolio_market_value(
            positions,
            prices,
        )
    )


def total_open_risk_eur(
    positions: dict,
) -> float:

    return sum(
        float(position["quantity"])
        * max(
            float(position["entry_price"])
            - float(position["stop_price"]),
            0.0,
        )
        for position
        in positions.values()
    )


def high_beta_allows(
    symbol: str,
    positions: dict,
) -> bool:

    if symbol not in HIGH_BETA_ASSETS:
        return True

    return not any(
        existing_symbol
        in HIGH_BETA_ASSETS
        for existing_symbol
        in positions
    )


def close_position(
    trades: list[dict],
    positions: dict,
    cash_balance: float,
    symbol: str,
    raw_exit: float,
    exit_time: pd.Timestamp,
    exit_reason: str,
) -> float:

    position = positions.pop(symbol)

    execution_exit = sell_price(
        float(raw_exit)
    )

    sell_notional = (
        float(position["quantity"])
        * execution_exit
    )

    sell_fee = calculate_fee(
        sell_notional
    )

    cash_received = (
        sell_notional
        - sell_fee
    )

    cash_balance += cash_received

    net_pnl_eur = (
        cash_received
        - float(
            position["entry_total_cost"]
        )
    )

    initial_risk_eur = float(
        position["initial_risk_eur"]
    )

    net_r = (
        net_pnl_eur
        / initial_risk_eur
        if initial_risk_eur > 0
        else 0.0
    )

    holding_hours = (
        pd.Timestamp(exit_time)
        - pd.Timestamp(
            position["entry_time"]
        )
    ).total_seconds() / 3600.0

    trades.append(
        {
            "window": WINDOW,
            "symbol": symbol,
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
            "quantity": (
                position["quantity"]
            ),
            "entry_notional": (
                position["entry_notional"]
            ),
            "buy_fee": (
                position["buy_fee"]
            ),
            "sell_fee": sell_fee,
            "initial_risk_eur": (
                initial_risk_eur
            ),
            "account_risk_percent": (
                position[
                    "account_risk_percent"
                ]
            ),
            "breakout_strength": (
                position[
                    "breakout_strength"
                ]
            ),
            "exit_reason": exit_reason,
            "net_pnl_eur": net_pnl_eur,
            "net_r": net_r,
            "holding_hours": holding_hours,
        }
    )

    return cash_balance


def close_all(
    trades: list[dict],
    positions: dict,
    cash_balance: float,
    prices: dict[str, float],
    exit_time: pd.Timestamp,
    exit_reason: str,
) -> float:

    for symbol in list(
        positions.keys()
    ):
        cash_balance = close_position(
            trades=trades,
            positions=positions,
            cash_balance=cash_balance,
            symbol=symbol,
            raw_exit=prices[symbol],
            exit_time=exit_time,
            exit_reason=exit_reason,
        )

    return cash_balance


# ============================================================================
# LOAD DATA
# ============================================================================


frames_1h: dict[
    str,
    pd.DataFrame,
] = {}

bars_4h: dict[
    str,
    pd.DataFrame,
] = {}


for symbol in SYMBOLS:
    frame = (
        eng.prepare_1h(symbol)
        .copy()
        .sort_index()
    )

    frame.index = pd.to_datetime(
        frame.index
    )

    frame = frame[
        ~frame.index.duplicated(
            keep="last"
        )
    ]

    frames_1h[symbol] = frame

    bars_4h[symbol] = prepare_4h(
        frame
    )


timeline_values = set()


for symbol in SYMBOLS:
    frame = frames_1h[symbol]

    selected_index = frame.index[
        (frame.index >= TEST_START)
        & (frame.index < TEST_END)
    ]

    timeline_values.update(
        selected_index.tolist()
    )


timeline = sorted(
    timeline_values
)


if not timeline:
    raise RuntimeError(
        f"No V3KJ hourly data for {WINDOW}."
    )


# ============================================================================
# INITIAL STATE
# ============================================================================


if (
    RESET_BACKTEST_STATE
    and STATE_FILE.exists()
):
    STATE_FILE.unlink()


risk_state = load_or_create_state(
    STATE_FILE,
    STARTING_CAPITAL,
)

assert_trading_allowed(
    STATE_FILE,
    STARTING_CAPITAL,
)


cash_balance = STARTING_CAPITAL

positions: dict[
    str,
    dict,
] = {}

pending_entries: dict[
    str,
    dict,
] = {}

pending_exits: dict[
    str,
    dict,
] = {}


trades: list[dict] = []

V3KX_CHECKPOINT_HOURS = (
    6,
    12,
    24,
    36,
)

v3kx_checkpoint_rows: list[
    dict[str, object]
] = []
equity_rows: list[dict] = []

counters = Counter()


last_prices: dict[
    str,
    float,
] = {}


current_day = None
current_week = None

daily_start_equity = (
    STARTING_CAPITAL
)

weekly_start_equity = (
    STARTING_CAPITAL
)

daily_entry_block = False
weekly_entry_block = False
daily_profit_entry_block = False
hard_lock = False

peak_equity = max(
    STARTING_CAPITAL,
    float(
        risk_state["peak_equity"]
    ),
)

maximum_positions_seen = 0


# ============================================================================
# SHARED HOURLY SIMULATION
# ============================================================================


for event_time in timeline:

    current_candles = {
        symbol: (
            frames_1h[symbol]
            .loc[event_time]
        )
        for symbol in SYMBOLS
        if event_time
        in frames_1h[symbol].index
    }

    if not current_candles:
        continue


    # ------------------------------------------------------------------------
    # Opening marks and calendar resets.
    # ------------------------------------------------------------------------

    opening_prices = dict(
        last_prices
    )

    for symbol, candle in (
        current_candles.items()
    ):
        opening_prices[symbol] = float(
            candle["open"]
        )


    opening_equity = portfolio_equity(
        cash_balance,
        positions,
        opening_prices,
    )

    timestamp = pd.Timestamp(
        event_time
    )

    day_key = timestamp.date()

    iso = timestamp.isocalendar()

    week_key = (
        int(iso.year),
        int(iso.week),
    )


    if current_day != day_key:
        current_day = day_key
        daily_start_equity = (
            opening_equity
        )
        daily_entry_block = False
        daily_profit_entry_block = False


    if current_week != week_key:
        current_week = week_key
        weekly_start_equity = (
            opening_equity
        )
        weekly_entry_block = False


    # ------------------------------------------------------------------------
    # Pending Donchian exits execute first at the next available 1h open.
    # ------------------------------------------------------------------------

    for symbol in list(
        pending_exits.keys()
    ):
        pending = pending_exits[
            symbol
        ]

        if (
            event_time
            < pending["due_time"]
            or symbol
            not in current_candles
        ):
            continue

        if symbol in positions:
            cash_balance = (
                close_position(
                    trades=trades,
                    positions=positions,
                    cash_balance=cash_balance,
                    symbol=symbol,
                    raw_exit=float(
                        current_candles[
                            symbol
                        ]["open"]
                    ),
                    exit_time=event_time,
                    exit_reason=(
                        pending["reason"]
                    ),
                )
            )

        pending_exits.pop(
            symbol,
            None,
        )


    # ------------------------------------------------------------------------
    # Pending entries: same-hour candidates ranked by causal breakout strength.
    # An unfilled candidate is not delayed beyond its next-hour execution.
    # ------------------------------------------------------------------------

    due_candidates = []

    for symbol, candidate in list(
        pending_entries.items()
    ):
        if (
            event_time
            >= candidate["due_time"]
            and symbol
            in current_candles
        ):
            due_candidates.append(
                candidate
            )

            pending_entries.pop(
                symbol,
                None,
            )


    due_candidates.sort(
        key=lambda candidate: (
            -float(
                candidate[
                    "breakout_strength"
                ]
            ),
            SYMBOL_ORDER[
                candidate["symbol"]
            ],
        )
    )


    entries_this_hour = 0

    block_new_entries = any(
        (
            hard_lock,
            daily_entry_block,
            weekly_entry_block,
            daily_profit_entry_block,
        )
    )


    for candidate_rank, candidate in enumerate(
        due_candidates,
        start=1,
    ):

        symbol = candidate["symbol"]

        observer_raw_open = float(
            current_candles[
                symbol
            ]["open"]
        )

        observer_entry_price = float(
            eng.apply_buy_slippage(
                observer_raw_open
            )
        )

        observer_equity = portfolio_equity(
            cash_balance,
            positions,
            opening_prices,
        )

        observer_open_risk = (
            total_open_risk_eur(
                positions
            )
        )

        observer_market_value = (
            portfolio_market_value(
                positions,
                opening_prices,
            )
        )

        observer_context = {
            "window": WINDOW,
            "observer_stage": "DUE",
            "disposition": None,
            "symbol": symbol,
            "signal_time": (
                candidate[
                    "signal_time"
                ]
            ),
            "due_time": (
                candidate[
                    "due_time"
                ]
            ),
            "event_time": event_time,
            "candidate_rank":
                candidate_rank,
            "due_candidate_count":
                len(due_candidates),
            "breakout_strength":
                float(
                    candidate[
                        "breakout_strength"
                    ]
                ),
            "atr_4h":
                float(
                    candidate[
                        "ATR14"
                    ]
                ),
            "raw_open":
                observer_raw_open,
            "entry_price":
                observer_entry_price,
            "cash_balance":
                float(
                    cash_balance
                ),
            "current_equity":
                float(
                    observer_equity
                ),
            "position_count":
                len(positions),
            "position_symbols":
                "|".join(
                    sorted(
                        positions.keys()
                    )
                ),
            "current_open_risk_eur":
                float(
                    observer_open_risk
                ),
            "current_market_value_eur":
                float(
                    observer_market_value
                ),
            "entries_this_hour":
                entries_this_hour,
            "hard_lock":
                bool(
                    hard_lock
                ),
            "daily_entry_block":
                bool(
                    daily_entry_block
                ),
            "weekly_entry_block":
                bool(
                    weekly_entry_block
                ),
            "daily_profit_entry_block":
                bool(
                    daily_profit_entry_block
                ),
            "high_beta_allowed":
                bool(
                    high_beta_allows(
                        symbol,
                        positions,
                    )
                ),
        }


        if block_new_entries:
            v3kv_record_due(
                observer_context,
                "ENTRY_BLOCK",
            )

            counters[
                "skipped_entry_block"
            ] += 1
            continue


        if symbol in positions:
            v3kv_record_due(
                observer_context,
                "EXISTING_POSITION",
            )

            counters[
                "skipped_existing_position"
            ] += 1
            continue


        if (
            len(positions)
            >= MAX_OPEN_POSITIONS
        ):
            v3kv_record_due(
                observer_context,
                "MAX_POSITIONS",
            )

            counters[
                "skipped_max_positions"
            ] += 1
            continue


        if (
            MAX_ONE_NEW_ENTRY_PER_HOUR
            and entries_this_hour >= 1
        ):
            v3kv_record_due(
                observer_context,
                "ONE_ENTRY_PER_HOUR",
            )

            counters[
                "skipped_one_entry_hour"
            ] += 1
            continue


        if not high_beta_allows(
            symbol,
            positions,
        ):
            v3kv_record_due(
                observer_context,
                "HIGH_BETA",
            )

            counters[
                "skipped_high_beta"
            ] += 1
            continue


        raw_open = float(
            current_candles[
                symbol
            ]["open"]
        )

        entry_price = float(
            eng.apply_buy_slippage(
                raw_open
            )
        )

        atr_4h = float(
            candidate["ATR14"]
        )

        initial_stop = (
            entry_price
            - INITIAL_STOP_ATR
            * atr_4h
        )

        initial_risk_per_unit = (
            entry_price
            - initial_stop
        )


        if initial_risk_per_unit <= 0:
            v3kv_record_due(
                observer_context,
                "INVALID_RISK",
                initial_stop=(
                    initial_stop
                ),
                initial_risk_per_unit=(
                    initial_risk_per_unit
                ),
            )

            counters[
                "skipped_invalid_risk"
            ] += 1
            continue


        current_equity = (
            portfolio_equity(
                cash_balance,
                positions,
                opening_prices,
            )
        )

        allowed_risk_eur = (
            current_equity
            * BASE_RISK_PERCENT[symbol]
            / 100.0
        )

        current_open_risk = (
            total_open_risk_eur(
                positions
            )
        )

        portfolio_risk_cap = (
            current_equity
            * MAX_TOTAL_OPEN_RISK_PERCENT
            / 100.0
        )

        remaining_risk_room = max(
            0.0,
            portfolio_risk_cap
            - current_open_risk,
        )

        allowed_risk_eur = min(
            allowed_risk_eur,
            remaining_risk_room,
        )


        if allowed_risk_eur <= 0:
            v3kv_record_due(
                observer_context,
                "RISK_CAP",
                initial_stop=(
                    initial_stop
                ),
                initial_risk_per_unit=(
                    initial_risk_per_unit
                ),
                allowed_risk_eur=(
                    allowed_risk_eur
                ),
                portfolio_risk_cap_eur=(
                    portfolio_risk_cap
                ),
                remaining_risk_room_eur=(
                    remaining_risk_room
                ),
            )

            counters[
                "skipped_risk_cap"
            ] += 1
            continue


        risk_sized_quantity = (
            allowed_risk_eur
            / initial_risk_per_unit
        )

        risk_sized_notional = (
            risk_sized_quantity
            * entry_price
        )

        asset_notional_cap = (
            current_equity
            * ASSET_NOTIONAL_CAP_PERCENT[
                symbol
            ]
            / 100.0
        )

        current_market_value = (
            portfolio_market_value(
                positions,
                opening_prices,
            )
        )

        deployment_cap = (
            current_equity
            * MAX_TOTAL_DEPLOYMENT_PERCENT
            / 100.0
        )

        deployment_room = max(
            0.0,
            deployment_cap
            - current_market_value,
        )

        affordable_notional = (
            cash_balance
            / (
                1.0
                + FEE_RATE
            )
        )

        notional = min(
            risk_sized_notional,
            asset_notional_cap,
            deployment_room,
            affordable_notional,
        )


        if notional < MIN_NOTIONAL_EUR:
            v3kv_record_due(
                observer_context,
                "MIN_NOTIONAL",
                initial_stop=(
                    initial_stop
                ),
                initial_risk_per_unit=(
                    initial_risk_per_unit
                ),
                allowed_risk_eur=(
                    allowed_risk_eur
                ),
                risk_sized_notional=(
                    risk_sized_notional
                ),
                asset_notional_cap=(
                    asset_notional_cap
                ),
                deployment_cap=(
                    deployment_cap
                ),
                deployment_room=(
                    deployment_room
                ),
                affordable_notional=(
                    affordable_notional
                ),
                selected_notional=(
                    notional
                ),
            )

            counters[
                "skipped_min_notional"
            ] += 1
            continue


        quantity = (
            notional
            / entry_price
        )

        buy_fee = calculate_fee(
            notional
        )

        entry_total_cost = (
            notional
            + buy_fee
        )


        if (
            quantity <= 0
            or entry_total_cost
            > cash_balance
        ):
            v3kv_record_due(
                observer_context,
                "AFFORDABILITY",
                initial_stop=(
                    initial_stop
                ),
                initial_risk_per_unit=(
                    initial_risk_per_unit
                ),
                selected_notional=(
                    notional
                ),
                quantity=quantity,
                buy_fee=buy_fee,
                entry_total_cost=(
                    entry_total_cost
                ),
            )

            counters[
                "skipped_affordability"
            ] += 1
            continue


        initial_risk_eur = (
            quantity
            * initial_risk_per_unit
        )

        account_risk_percent = (
            initial_risk_eur
            / current_equity
            * 100.0
            if current_equity > 0
            else 0.0
        )


        cash_balance -= (
            entry_total_cost
        )

        positions[symbol] = {
            "symbol": symbol,
            "signal_time": (
                candidate["signal_time"]
            ),
            "entry_time": event_time,
            "entry_price": entry_price,
            "initial_stop": initial_stop,
            "stop_price": initial_stop,
            "initial_risk_per_unit": (
                initial_risk_per_unit
            ),
            "initial_risk_eur": (
                initial_risk_eur
            ),
            "quantity": quantity,
            "entry_notional": notional,
            "buy_fee": buy_fee,
            "entry_total_cost": (
                entry_total_cost
            ),
            "account_risk_percent": (
                account_risk_percent
            ),
            "breakout_strength": (
                candidate[
                    "breakout_strength"
                ]
            ),

            "v3kx_path_high_price":
                float(
                    current_candles[
                        symbol
                    ]["high"]
                ),

            "v3kx_path_low_price":
                float(
                    current_candles[
                        symbol
                    ]["low"]
                ),

            "v3kx_recorded_checkpoints":
                set(),
        }

        entries_this_hour += 1
        counters["entries"] += 1

        v3kv_record_due(
            observer_context,
            "EXECUTED",
            initial_stop=(
                initial_stop
            ),
            initial_risk_per_unit=(
                initial_risk_per_unit
            ),
            allowed_risk_eur=(
                allowed_risk_eur
            ),
            portfolio_risk_cap_eur=(
                portfolio_risk_cap
            ),
            remaining_risk_room_eur=(
                remaining_risk_room
            ),
            risk_sized_notional=(
                risk_sized_notional
            ),
            asset_notional_cap=(
                asset_notional_cap
            ),
            deployment_cap=(
                deployment_cap
            ),
            deployment_room=(
                deployment_room
            ),
            affordable_notional=(
                affordable_notional
            ),
            selected_notional=(
                notional
            ),
            quantity=quantity,
            buy_fee=buy_fee,
            entry_total_cost=(
                entry_total_cost
            ),
            initial_risk_eur=(
                initial_risk_eur
            ),
            account_risk_percent=(
                account_risk_percent
            ),
        )

        maximum_positions_seen = max(
            maximum_positions_seen,
            len(positions),
        )


    # ------------------------------------------------------------------------
    # Intrabar stop check. This happens after entry and before new 4h updates.
    # ------------------------------------------------------------------------

    for symbol in list(
        positions.keys()
    ):

        if symbol not in current_candles:
            continue

        candle = current_candles[
            symbol
        ]

        raw_open = float(
            candle["open"]
        )

        low_price = float(
            candle["low"]
        )

        stop_price = float(
            positions[
                symbol
            ]["stop_price"]
        )


        if low_price <= stop_price:

            raw_stop_fill = min(
                stop_price,
                raw_open,
            )

            cash_balance = (
                close_position(
                    trades=trades,
                    positions=positions,
                    cash_balance=cash_balance,
                    symbol=symbol,
                    raw_exit=raw_stop_fill,
                    exit_time=event_time,
                    exit_reason=(
                        "HARD_OR_TRAILING_STOP"
                    ),
                )
            )

            pending_exits.pop(
                symbol,
                None,
            )

            counters[
                "stop_exits"
            ] += 1


    # ------------------------------------------------------------------------
    # Hourly closing marks.
    # ------------------------------------------------------------------------

    for symbol, candle in (
        current_candles.items()
    ):
        last_prices[symbol] = float(
            candle["close"]
        )


    close_equity = portfolio_equity(
        cash_balance,
        positions,
        last_prices,
    )

    peak_equity = max(
        peak_equity,
        close_equity,
    )

    drawdown_percent = (
        (
            peak_equity
            - close_equity
        )
        / peak_equity
        * 100.0
        if peak_equity > 0
        else 0.0
    )

    daily_return = (
        (
            close_equity
            - daily_start_equity
        )
        / daily_start_equity
        * 100.0
        if daily_start_equity > 0
        else 0.0
    )

    weekly_return = (
        (
            close_equity
            - weekly_start_equity
        )
        / weekly_start_equity
        * 100.0
        if weekly_start_equity > 0
        else 0.0
    )


    # ------------------------------------------------------------------------
    # Persistent hard drawdown guard.
    # ------------------------------------------------------------------------

    risk_state = (
        update_equity_and_lock(
            state_file=STATE_FILE,
            starting_capital=(
                STARTING_CAPITAL
            ),
            current_equity=(
                close_equity
            ),
            drawdown_limit_percent=(
                HARD_DRAWDOWN_LIMIT_PERCENT
            ),
        )
    )


    if (
        bool(risk_state["hard_lock"])
        and not hard_lock
    ):
        hard_lock = True

        counters[
            "hard_lock_hits"
        ] += 1

        cash_balance = close_all(
            trades=trades,
            positions=positions,
            cash_balance=cash_balance,
            prices=last_prices,
            exit_time=event_time,
            exit_reason=(
                "HARD DRAWDOWN LOCK"
            ),
        )

        close_equity = (
            portfolio_equity(
                cash_balance,
                positions,
                last_prices,
            )
        )

        update_equity_and_lock(
            state_file=STATE_FILE,
            starting_capital=(
                STARTING_CAPITAL
            ),
            current_equity=(
                close_equity
            ),
            drawdown_limit_percent=(
                HARD_DRAWDOWN_LIMIT_PERCENT
            ),
        )


    # ------------------------------------------------------------------------
    # Temporary weekly and daily circuit breakers.
    # ------------------------------------------------------------------------

    if (
        not hard_lock
        and not weekly_entry_block
        and weekly_return
        <= -WEEKLY_LOSS_LIMIT_PERCENT
    ):
        weekly_entry_block = True

        counters[
            "weekly_loss_stops"
        ] += 1

        cash_balance = close_all(
            trades=trades,
            positions=positions,
            cash_balance=cash_balance,
            prices=last_prices,
            exit_time=event_time,
            exit_reason=(
                "WEEKLY LOSS LIMIT"
            ),
        )


    if (
        not hard_lock
        and not daily_entry_block
        and daily_return
        <= -DAILY_LOSS_LIMIT_PERCENT
    ):
        daily_entry_block = True

        counters[
            "daily_loss_stops"
        ] += 1

        cash_balance = close_all(
            trades=trades,
            positions=positions,
            cash_balance=cash_balance,
            prices=last_prices,
            exit_time=event_time,
            exit_reason=(
                "DAILY LOSS LIMIT"
            ),
        )


    if (
        not hard_lock
        and not daily_profit_entry_block
        and daily_return
        >= DAILY_PROFIT_ENTRY_LOCK_PERCENT
    ):
        daily_profit_entry_block = True

        counters[
            "daily_profit_locks"
        ] += 1


    # ------------------------------------------------------------------------
    # Completed 4h decisions. New information applies only to future 1h bars.
    # ------------------------------------------------------------------------

    block_new_signals = any(
        (
            hard_lock,
            daily_entry_block,
            weekly_entry_block,
            daily_profit_entry_block,
        )
    )


    if not hard_lock:

        for symbol in SYMBOLS:

            if (
                event_time
                not in bars_4h[
                    symbol
                ].index
            ):
                continue

            bar_4h = (
                bars_4h[symbol]
                .loc[event_time]
            )


            if bool(
                bar_4h[
                    "ENTRY_SIGNAL"
                ]
            ):
                observer_signal_atr = float(
                    bar_4h[
                        "ATR14"
                    ]
                )

                observer_prior_high = float(
                    bar_4h[
                        "PRIOR_HIGH_20"
                    ]
                )

                observer_signal_strength = (
                    (
                        float(
                            bar_4h[
                                "close"
                            ]
                        )
                        - observer_prior_high
                    )
                    / observer_signal_atr
                    if observer_signal_atr > 0
                    else 0.0
                )

                if symbol in positions:
                    signal_disposition = (
                        "SIGNAL_EXISTING_POSITION"
                    )

                elif symbol in pending_entries:
                    signal_disposition = (
                        "SIGNAL_ALREADY_PENDING"
                    )

                elif block_new_signals:
                    signal_disposition = (
                        "SIGNAL_BLOCK"
                    )

                else:
                    signal_disposition = (
                        "SIGNAL_CREATED"
                    )

                v3kx_early_path_rows.append(
                    {
                        "window": WINDOW,
                        "observer_stage":
                            "SIGNAL",
                        "disposition":
                            signal_disposition,
                        "symbol": symbol,
                        "signal_time":
                            event_time,
                        "due_time": (
                            event_time
                            + pd.Timedelta(
                                hours=1
                            )
                        ),
                        "event_time":
                            event_time,
                        "candidate_rank":
                            None,
                        "due_candidate_count":
                            None,
                        "breakout_strength":
                            observer_signal_strength,
                        "atr_4h":
                            observer_signal_atr,
                        "raw_open":
                            None,
                        "entry_price":
                            None,
                        "cash_balance":
                            float(
                                cash_balance
                            ),
                        "current_equity":
                            None,
                        "position_count":
                            len(positions),
                        "position_symbols":
                            "|".join(
                                sorted(
                                    positions.keys()
                                )
                            ),
                        "current_open_risk_eur":
                            float(
                                total_open_risk_eur(
                                    positions
                                )
                            ),
                        "current_market_value_eur":
                            None,
                        "entries_this_hour":
                            entries_this_hour,
                        "hard_lock":
                            bool(
                                hard_lock
                            ),
                        "daily_entry_block":
                            bool(
                                daily_entry_block
                            ),
                        "weekly_entry_block":
                            bool(
                                weekly_entry_block
                            ),
                        "daily_profit_entry_block":
                            bool(
                                daily_profit_entry_block
                            ),
                        "high_beta_allowed":
                            bool(
                                high_beta_allows(
                                    symbol,
                                    positions,
                                )
                            ),
                    }
                )


            if symbol in positions:

                prior_low = bar_4h[
                    "PRIOR_LOW_10"
                ]

                if pd.notna(
                    prior_low
                ):
                    candidate_stop = float(
                        prior_low
                    )

                    positions[
                        symbol
                    ]["stop_price"] = max(
                        float(
                            positions[
                                symbol
                            ]["stop_price"]
                        ),
                        candidate_stop,
                    )

                    if (
                        float(
                            bar_4h["close"]
                        )
                        <= candidate_stop
                    ):
                        pending_exits[
                            symbol
                        ] = {
                            "due_time": (
                                event_time
                                + pd.Timedelta(
                                    hours=1
                                )
                            ),
                            "reason": (
                                "4H_DONCHIAN_EXIT"
                            ),
                        }


            if (
                symbol not in positions
                and symbol
                not in pending_entries
                and not block_new_signals
                and bool(
                    bar_4h[
                        "ENTRY_SIGNAL"
                    ]
                )
            ):
                atr_4h = float(
                    bar_4h["ATR14"]
                )

                prior_high = float(
                    bar_4h[
                        "PRIOR_HIGH_20"
                    ]
                )

                breakout_strength = (
                    (
                        float(
                            bar_4h["close"]
                        )
                        - prior_high
                    )
                    / atr_4h
                    if atr_4h > 0
                    else 0.0
                )

                pending_entries[
                    symbol
                ] = {
                    "symbol": symbol,
                    "signal_time": (
                        event_time
                    ),
                    "due_time": (
                        event_time
                        + pd.Timedelta(
                            hours=1
                        )
                    ),
                    "ATR14": atr_4h,
                    "breakout_strength": (
                        breakout_strength
                    ),
                }

                counters[
                    "signals"
                ] += 1


    # ------------------------------------------------------------------------
    # V3KX causal early-path snapshots.
    #
    # This executes after:
    #   1. pending exits,
    #   2. entries,
    #   3. intrabar stops,
    #   4. hourly risk transitions,
    #   5. completed 4h stop updates.
    #
    # Only positions still open at the completed hourly close are recorded.
    # ------------------------------------------------------------------------

    for symbol in sorted(
        positions.keys(),
        key=lambda value: (
            SYMBOL_ORDER[value]
        ),
    ):

        if symbol not in current_candles:
            continue

        position = positions[
            symbol
        ]

        candle = current_candles[
            symbol
        ]

        candle_high = float(
            candle["high"]
        )

        candle_low = float(
            candle["low"]
        )

        candle_close = float(
            candle["close"]
        )

        position[
            "v3kx_path_high_price"
        ] = max(
            float(
                position[
                    "v3kx_path_high_price"
                ]
            ),
            candle_high,
        )

        position[
            "v3kx_path_low_price"
        ] = min(
            float(
                position[
                    "v3kx_path_low_price"
                ]
            ),
            candle_low,
        )

        snapshot_time = (
            pd.Timestamp(
                event_time
            )
            + pd.Timedelta(
                hours=1
            )
        )

        completed_hours_float = float(
            (
                snapshot_time
                - pd.Timestamp(
                    position[
                        "entry_time"
                    ]
                )
            )
            / pd.Timedelta(
                hours=1
            )
        )

        completed_hours = int(
            round(
                completed_hours_float
            )
        )

        if abs(
            completed_hours_float
            - completed_hours
        ) > 1e-9:
            continue

        if (
            completed_hours
            not in V3KX_CHECKPOINT_HOURS
        ):
            continue

        recorded_checkpoints = position[
            "v3kx_recorded_checkpoints"
        ]

        if (
            completed_hours
            in recorded_checkpoints
        ):
            continue

        entry_price = float(
            position[
                "entry_price"
            ]
        )

        initial_stop = float(
            position[
                "initial_stop"
            ]
        )

        active_stop = float(
            position[
                "stop_price"
            ]
        )

        initial_risk_per_unit = float(
            position[
                "initial_risk_per_unit"
            ]
        )

        initial_risk_eur = float(
            position[
                "initial_risk_eur"
            ]
        )

        quantity = float(
            position[
                "quantity"
            ]
        )

        execution_exit_price = float(
            sell_price(
                candle_close
            )
        )

        sell_notional = (
            quantity
            * execution_exit_price
        )

        sell_fee = float(
            calculate_fee(
                sell_notional
            )
        )

        exit_now_net_pnl_eur = (
            sell_notional
            - sell_fee
            - float(
                position[
                    "entry_total_cost"
                ]
            )
        )

        exit_now_net_r = (
            exit_now_net_pnl_eur
            / initial_risk_eur
            if initial_risk_eur > 0
            else 0.0
        )

        path_high_price = float(
            position[
                "v3kx_path_high_price"
            ]
        )

        path_low_price = float(
            position[
                "v3kx_path_low_price"
            ]
        )

        mfe_gross_r = (
            (
                path_high_price
                - entry_price
            )
            / initial_risk_per_unit
            if initial_risk_per_unit > 0
            else 0.0
        )

        mae_gross_r = (
            (
                path_low_price
                - entry_price
            )
            / initial_risk_per_unit
            if initial_risk_per_unit > 0
            else 0.0
        )

        close_gross_r = (
            (
                candle_close
                - entry_price
            )
            / initial_risk_per_unit
            if initial_risk_per_unit > 0
            else 0.0
        )

        stop_progress_r = (
            (
                active_stop
                - initial_stop
            )
            / initial_risk_per_unit
            if initial_risk_per_unit > 0
            else 0.0
        )

        stop_level_r = (
            (
                active_stop
                - entry_price
            )
            / initial_risk_per_unit
            if initial_risk_per_unit > 0
            else 0.0
        )

        distance_to_stop_r = (
            (
                candle_close
                - active_stop
            )
            / initial_risk_per_unit
            if initial_risk_per_unit > 0
            else 0.0
        )

        v3kx_checkpoint_rows.append(
            {
                "window":
                    WINDOW,

                "symbol":
                    symbol,

                "signal_time":
                    position[
                        "signal_time"
                    ],

                "entry_time":
                    position[
                        "entry_time"
                    ],

                "snapshot_event_time":
                    event_time,

                "snapshot_time":
                    snapshot_time,

                "checkpoint_hours":
                    completed_hours,

                "entry_price":
                    entry_price,

                "snapshot_close_price":
                    candle_close,

                "snapshot_execution_exit_price":
                    execution_exit_price,

                "initial_stop":
                    initial_stop,

                "active_stop":
                    active_stop,

                "initial_risk_per_unit":
                    initial_risk_per_unit,

                "initial_risk_eur":
                    initial_risk_eur,

                "quantity":
                    quantity,

                "breakout_strength":
                    float(
                        position[
                            "breakout_strength"
                        ]
                    ),

                "path_high_price":
                    path_high_price,

                "path_low_price":
                    path_low_price,

                "mfe_gross_r":
                    mfe_gross_r,

                "mae_gross_r":
                    mae_gross_r,

                "close_gross_r":
                    close_gross_r,

                "exit_now_net_pnl_eur":
                    exit_now_net_pnl_eur,

                "exit_now_net_r":
                    exit_now_net_r,

                "stop_progress_r":
                    stop_progress_r,

                "stop_level_r":
                    stop_level_r,

                "distance_to_stop_r":
                    distance_to_stop_r,

                "pending_exit":
                    bool(
                        symbol
                        in pending_exits
                    ),

                "hard_lock":
                    bool(
                        hard_lock
                    ),

                "daily_entry_block":
                    bool(
                        daily_entry_block
                    ),

                "weekly_entry_block":
                    bool(
                        weekly_entry_block
                    ),

                "daily_profit_entry_block":
                    bool(
                        daily_profit_entry_block
                    ),

                "snapshot_key":
                    (
                        f"{WINDOW}|"
                        f"{symbol}|"
                        f"{pd.Timestamp(
                            position['entry_time']
                        ).isoformat()}|"
                        f"{completed_hours}"
                    ),
            }
        )

        recorded_checkpoints.add(
            completed_hours
        )


    ending_equity = portfolio_equity(
        cash_balance,
        positions,
        last_prices,
    )

    equity_rows.append(
        {
            "window": WINDOW,
            "time": event_time,
            "equity": ending_equity,
            "cash": cash_balance,
            "open_positions": (
                len(positions)
            ),
            "open_risk_eur": (
                total_open_risk_eur(
                    positions
                )
            ),
            "daily_return_percent": (
                daily_return
            ),
            "weekly_return_percent": (
                weekly_return
            ),
            "drawdown_percent": (
                drawdown_percent
            ),
            "daily_entry_block": (
                daily_entry_block
            ),
            "weekly_entry_block": (
                weekly_entry_block
            ),
            "daily_profit_entry_block": (
                daily_profit_entry_block
            ),
            "hard_lock": hard_lock,
        }
    )


    if hard_lock:
        break


# ============================================================================
# END-OF-WINDOW MARK
# ============================================================================


if positions and not hard_lock:

    final_time = timeline[-1]

    cash_balance = close_all(
        trades=trades,
        positions=positions,
        cash_balance=cash_balance,
        prices=last_prices,
        exit_time=final_time,
        exit_reason="END_MARK",
    )

    equity_rows.append(
        {
            "window": WINDOW,
            "time": final_time,
            "equity": cash_balance,
            "cash": cash_balance,
            "open_positions": 0,
            "open_risk_eur": 0.0,
            "daily_return_percent": 0.0,
            "weekly_return_percent": 0.0,
            "drawdown_percent": 0.0,
            "daily_entry_block": (
                daily_entry_block
            ),
            "weekly_entry_block": (
                weekly_entry_block
            ),
            "daily_profit_entry_block": (
                daily_profit_entry_block
            ),
            "hard_lock": hard_lock,
        }
    )


# ============================================================================
# REPORTING
# ============================================================================


detail = pd.DataFrame(
    trades
)

equity_detail = pd.DataFrame(
    equity_rows
)


for output_path in [
    TRADE_OUTPUT,
    EQUITY_OUTPUT,
    SUMMARY_OUTPUT,
]:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


detail.to_csv(
    TRADE_OUTPUT,
    index=False,
)

equity_detail.to_csv(
    EQUITY_OUTPUT,
    index=False,
)


ending_capital = float(
    cash_balance
)

return_percent = (
    (
        ending_capital
        - STARTING_CAPITAL
    )
    / STARTING_CAPITAL
    * 100.0
)


if equity_detail.empty:
    maximum_drawdown = 0.0
else:
    equity_series = (
        equity_detail["equity"]
        .astype(float)
    )

    running_peak = (
        equity_series.cummax()
    )

    drawdowns = (
        (
            running_peak
            - equity_series
        )
        / running_peak
        * 100.0
    )

    maximum_drawdown = float(
        drawdowns.max()
    )


if detail.empty:
    winners = detail
    losers = detail
    gross_profit = 0.0
    gross_loss = 0.0
    profit_factor = 0.0
    total_net_r = 0.0
else:
    winners = detail[
        detail["net_pnl_eur"] > 0
    ]

    losers = detail[
        detail["net_pnl_eur"] < 0
    ]

    gross_profit = float(
        winners[
            "net_pnl_eur"
        ].sum()
    )

    gross_loss = abs(
        float(
            losers[
                "net_pnl_eur"
            ].sum()
        )
    )

    profit_factor = (
        gross_profit
        / gross_loss
        if gross_loss > 0
        else float("inf")
        if gross_profit > 0
        else 0.0
    )

    total_net_r = float(
        detail["net_r"].sum()
    )


summary = pd.DataFrame(
    [
        {
            "window": WINDOW,
            "starting_capital": (
                STARTING_CAPITAL
            ),
            "ending_capital": (
                ending_capital
            ),
            "return_percent": (
                return_percent
            ),
            "maximum_drawdown_percent": (
                maximum_drawdown
            ),
            "completed_trades": (
                len(detail)
            ),
            "winners": (
                len(winners)
            ),
            "losers": (
                len(losers)
            ),
            "profit_factor": (
                profit_factor
            ),
            "total_net_r": (
                total_net_r
            ),
            "maximum_positions_seen": (
                maximum_positions_seen
            ),
            "signals": (
                counters["signals"]
            ),
            "entries": (
                counters["entries"]
            ),
            "hard_lock_hits": (
                counters[
                    "hard_lock_hits"
                ]
            ),
            "daily_loss_stops": (
                counters[
                    "daily_loss_stops"
                ]
            ),
            "weekly_loss_stops": (
                counters[
                    "weekly_loss_stops"
                ]
            ),
            "daily_profit_locks": (
                counters[
                    "daily_profit_locks"
                ]
            ),
            "skipped_max_positions": (
                counters[
                    "skipped_max_positions"
                ]
            ),
            "skipped_one_entry_hour": (
                counters[
                    "skipped_one_entry_hour"
                ]
            ),
            "skipped_high_beta": (
                counters[
                    "skipped_high_beta"
                ]
            ),
            "skipped_min_notional": (
                counters[
                    "skipped_min_notional"
                ]
            ),
            "persistent_hard_lock": (
                bool(
                    risk_state[
                        "hard_lock"
                    ]
                )
            ),
        }
    ]
)


summary.to_csv(
    SUMMARY_OUTPUT,
    index=False,
)


print()
print("=" * 112)
print(
    f"V3KJ SHARED PORTFOLIO | "
    f"{WINDOW}"
)
print("=" * 112)

print(
    f"Starting capital             : "
    f"EUR {STARTING_CAPITAL:,.2f}"
)

print(
    f"Ending capital               : "
    f"EUR {ending_capital:,.2f}"
)

print(
    f"Portfolio return             : "
    f"{return_percent:+.2f}%"
)

print(
    f"Maximum drawdown             : "
    f"{maximum_drawdown:.2f}%"
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
    f"Total net R                  : "
    f"{total_net_r:+.2f}R"
)

print(
    f"Profit factor                : "
    f"{profit_factor:.2f}"
)

print(
    f"Maximum simultaneous trades  : "
    f"{maximum_positions_seen}"
)

print(
    f"Hard-lock hits               : "
    f"{counters['hard_lock_hits']}"
)

print(
    f"Daily / Weekly stops         : "
    f"{counters['daily_loss_stops']} / "
    f"{counters['weekly_loss_stops']}"
)

print(
    f"Daily profit-entry locks     : "
    f"{counters['daily_profit_locks']}"
)

print("-" * 112)
print("SKIPPED ENTRIES")
print("-" * 112)

print(
    f"Maximum positions            : "
    f"{counters['skipped_max_positions']}"
)

print(
    f"One-entry-per-hour           : "
    f"{counters['skipped_one_entry_hour']}"
)

print(
    f"High-beta correlation        : "
    f"{counters['skipped_high_beta']}"
)

print(
    f"Minimum notional             : "
    f"{counters['skipped_min_notional']}"
)


if not detail.empty:

    print("-" * 112)
    print("BY SYMBOL")
    print("-" * 112)

    symbol_summary = (
        detail.groupby("symbol")
        .agg(
            trades=(
                "net_pnl_eur",
                "size",
            ),
            wins=(
                "net_pnl_eur",
                lambda values: int(
                    (values > 0).sum()
                ),
            ),
            net_pnl_eur=(
                "net_pnl_eur",
                "sum",
            ),
            net_r=(
                "net_r",
                "sum",
            ),
        )
        .reset_index()
        .sort_values(
            "net_pnl_eur",
            ascending=False,
        )
    )

    print(
        symbol_summary.to_string(
            index=False,
            formatters={
                "net_pnl_eur": (
                    lambda value:
                    f"{value:+.2f}"
                ),
                "net_r": (
                    lambda value:
                    f"{value:+.2f}"
                ),
            },
        )
    )


print("-" * 112)

print(
    f"Trades report                : "
    f"{TRADE_OUTPUT}"
)

print(
    f"Equity report                : "
    f"{EQUITY_OUTPUT}"
)

print(
    f"Summary report               : "
    f"{SUMMARY_OUTPUT}"
)

print(
    f"Persistent state             : "
    f"{STATE_FILE}"
)

print(
    "Safety evaluation resolution: "
    "1h close (backtest baseline)"
)

print("=" * 112)
# ============================================================================
# V3KV OBSERVER REPORTING
# ============================================================================

V3KV_EVENT_OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

v3kx_early_path_detail = pd.DataFrame(
    v3kx_early_path_rows
)

v3kx_early_path_detail.to_csv(
    V3KV_EVENT_OUTPUT,
    index=False,
)

if v3kx_early_path_detail.empty:
    v3kv_disposition_summary = pd.DataFrame(
        columns=[
            "window",
            "observer_stage",
            "disposition",
            "observations",
        ]
    )
else:
    v3kv_disposition_summary = (
        v3kx_early_path_detail
        .groupby(
            [
                "window",
                "observer_stage",
                "disposition",
            ],
            dropna=False,
        )
        .size()
        .rename(
            "observations"
        )
        .reset_index()
        .sort_values(
            [
                "observer_stage",
                "observations",
                "disposition",
            ],
            ascending=[
                True,
                False,
                True,
            ],
        )
    )

v3kv_disposition_summary.to_csv(
    V3KV_DISPOSITION_OUTPUT,
    index=False,
)

print("")
print(
    "V3KV opportunity observer"
)

print(
    "Observer rows                 : "
    f"{len(v3kx_early_path_detail)}"
)

print(
    "Observer events report        : "
    f"{V3KV_EVENT_OUTPUT}"
)

print(
    "Observer disposition report   : "
    f"{V3KV_DISPOSITION_OUTPUT}"
)

print(
    "Canonical V3KJ source was not modified."
)

# ============================================================================
# V3KX EARLY-PATH SNAPSHOT OUTPUT
# ============================================================================

v3kx_early_path_detail = pd.DataFrame(
    v3kx_checkpoint_rows
)

required_snapshot_columns = [
    "window",
    "symbol",
    "entry_time",
    "snapshot_time",
    "checkpoint_hours",
    "breakout_strength",
    "mfe_gross_r",
    "mae_gross_r",
    "close_gross_r",
    "exit_now_net_r",
    "stop_progress_r",
    "stop_level_r",
    "distance_to_stop_r",
    "snapshot_key",
]

if not v3kx_early_path_detail.empty:

    missing_columns = sorted(
        set(
            required_snapshot_columns
        )
        - set(
            v3kx_early_path_detail.columns
        )
    )

    if missing_columns:
        raise RuntimeError(
            "V3KX_SNAPSHOT_COLUMNS_MISSING: "
            f"{missing_columns}"
        )

    invalid_checkpoints = sorted(
        set(
            v3kx_early_path_detail[
                "checkpoint_hours"
            ]
            .astype(int)
            .unique()
            .tolist()
        )
        - set(
            V3KX_CHECKPOINT_HOURS
        )
    )

    if invalid_checkpoints:
        raise RuntimeError(
            "V3KX_INVALID_CHECKPOINTS: "
            f"{invalid_checkpoints}"
        )

    duplicate_keys = int(
        v3kx_early_path_detail.duplicated(
            subset=[
                "snapshot_key",
            ]
        ).sum()
    )

    if duplicate_keys != 0:
        raise RuntimeError(
            "V3KX_DUPLICATE_SNAPSHOT_KEYS: "
            f"{duplicate_keys}"
        )

    null_counts = (
        v3kx_early_path_detail[
            required_snapshot_columns
        ]
        .isna()
        .sum()
    )

    null_counts = null_counts[
        null_counts > 0
    ]

    if not null_counts.empty:
        raise RuntimeError(
            "V3KX_REQUIRED_SNAPSHOT_NULLS: "
            f"{null_counts.to_dict()}"
        )


v3kx_snapshot_output = (
    Path("reports")
    / (
        "v3kx_early_path_snapshots_"
        f"{TAG}.csv"
    )
)

v3kx_early_path_detail.to_csv(
    v3kx_snapshot_output,
    index=False,
)

checkpoint_counts = (
    v3kx_early_path_detail[
        "checkpoint_hours"
    ]
    .value_counts()
    .sort_index()
    .to_dict()
    if not v3kx_early_path_detail.empty
    else {}
)

print()
print("=" * 112)
print("V3KX CAUSAL EARLY-PATH OBSERVER")
print("=" * 112)
print(
    f"Snapshot rows               : "
    f"{len(v3kx_early_path_detail)}"
)
print(
    f"Checkpoint counts           : "
    f"{checkpoint_counts}"
)
print(
    f"Snapshot report             : "
    f"{v3kx_snapshot_output}"
)
print(
    "Early exits executed        : 0"
)
print(
    "Portfolio/risk changes      : 0"
)
print("=" * 112)

