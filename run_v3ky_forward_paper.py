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


WINDOW = "V3KY001"
TAG = "v3ky001"

TEST_START = pd.Timestamp(
    "2026-08-08T23:00:00Z"
)

TEST_END = pd.Timestamp(
    "2100-01-01T00:00:00Z"
)


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
    f"v3kj_shared_portfolio_{TAG}_trades.csv"
)

EQUITY_OUTPUT = Path(
    f"reports/"
    f"v3kj_shared_portfolio_{TAG}_equity.csv"
)

SUMMARY_OUTPUT = Path(
    f"reports/"
    f"v3kj_shared_portfolio_{TAG}_summary.csv"
)

STATE_FILE = Path(
    "state/v3ky/v3ky_001_runtime_risk_guard.json"
)

# Deterministic replay of accumulated forward bars.
RESET_BACKTEST_STATE = True


# ============================================================================
# V3KI INDICATORS — UNCHANGED
# ============================================================================


from core.v3ky_forward_ledger import append_record, load_config

V3KY_FORWARD_START = pd.Timestamp(
    load_config()["forward_start_utc"]
)


def v3ky_timestamp(value):
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def v3ky_iso(value):
    return (
        v3ky_timestamp(value)
        .isoformat()
        .replace("+00:00", "Z")
    )


def v3ky_record(record_type, event_time, **payload):
    import json

    ts = v3ky_timestamp(event_time)

    if ts <= V3KY_FORWARD_START:
        return None

    event_iso = v3ky_iso(ts)

    candidate = {
        "record_type": record_type,
        "event_timestamp": event_iso,
        **payload,
    }

    identity_fields = (
        "record_type",
        "event_timestamp",
        "symbol",
        "signal_timestamp",
        "entry_decision",
        "entry_execution_timestamp",
        "exit_execution_timestamp",
        "exit_reason",
        "notes",
    )

    def identity(record):
        return tuple(
            record.get(field)
            for field in identity_fields
        )

    if not hasattr(
        v3ky_record,
        "_replay_index",
    ):
        index = {}
        sealed_horizon = None

        ledger_path = Path(
            "state/v3ky/"
            "v3ky_001_forward_ledger.jsonl"
        )

        if ledger_path.exists():
            for raw in ledger_path.read_text(
                encoding="utf-8"
            ).splitlines():

                if not raw.strip():
                    continue

                existing_record = json.loads(raw)

                key = identity(
                    existing_record
                )

                if key in index:
                    raise RuntimeError(
                        "DUPLICATE_EXISTING_"
                        f"LEDGER_IDENTITY:{key}"
                    )

                index[key] = (
                    existing_record
                )

                if (
                    existing_record.get(
                        "record_type"
                    )
                    == "EQUITY"
                ):
                    existing_time = (
                        v3ky_timestamp(
                            existing_record[
                                "event_timestamp"
                            ]
                        )
                    )

                    if (
                        sealed_horizon is None
                        or existing_time
                        > sealed_horizon
                    ):
                        sealed_horizon = (
                            existing_time
                        )

        v3ky_record._replay_index = (
            index
        )

        v3ky_record._sealed_horizon = (
            sealed_horizon
        )

    index = v3ky_record._replay_index
    key = identity(candidate)

    existing = index.get(key)

    if existing is not None:

        for field, value in candidate.items():
            if existing.get(field) != value:
                raise RuntimeError(
                    "V3KY_REPLAY_CONFLICT:"
                    f"{key}:{field}"
                )

        return existing

    sealed_horizon = (
        v3ky_record._sealed_horizon
    )

    if (
        sealed_horizon is not None
        and ts <= sealed_horizon
    ):
        raise RuntimeError(
            "V3KY_RETROACTIVE_RECORD_ATTEMPT:"
            f"{event_iso}:"
            f"{record_type}"
        )

    record_payload = dict(payload)

    record_payload[
        "record_type"
    ] = record_type

    record_payload[
        "event_timestamp"
    ] = event_iso

    record = append_record(
        record_payload
    )

    if not isinstance(record, dict):
        ledger_path = Path(
            "state/v3ky/"
            "v3ky_001_forward_ledger.jsonl"
        )

        latest = [
            line
            for line
            in ledger_path.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ][-1]

        record = json.loads(latest)

    index[key] = record

    return record



def v3ky_decision(event_time, symbol, candidate, decision):
    signal_time = candidate.get("signal_time")

    return v3ky_record(
        "DECISION",
        event_time,
        symbol=symbol,
        signal_state="ENTRY_SIGNAL",
        signal_timestamp=(
            v3ky_iso(signal_time)
            if signal_time is not None
            else None
        ),
        pending_entry_state=False,
        entry_decision=decision,
        data_health_state="OK",
        notes="Canonical V3KJ candidate disposition.",
    )


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

    v3ky_record(
        "EXIT",
        exit_time,
        symbol=symbol,
        signal_timestamp=v3ky_iso(position["signal_time"]),
        exit_decision="EXECUTED",
        exit_reason=exit_reason,
        exit_execution_timestamp=v3ky_iso(exit_time),
        exit_execution_price=float(execution_exit),
        quantity=float(position["quantity"]),
        initial_stop=float(position["initial_stop"]),
        active_stop=float(position["stop_price"]),
        initial_risk_eur=float(initial_risk_eur),
        fees_eur=float(sell_fee),
        realized_pnl_eur=float(net_pnl_eur),
        realized_net_r=float(net_r),
        cash_eur=float(cash_balance),
        open_positions=len(positions),
        open_risk_eur=float(total_open_risk_eur(positions)),
        data_health_state="OK",
        notes="Canonical V3KJ paper exit.",
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


V3KY_MARKET_DIR = Path(
    "state/v3ky/market"
)


def v3ky_load_1h(symbol):
    path = (
        V3KY_MARKET_DIR
        / f"{symbol}_1h.csv"
    )

    if not path.exists():
        raise RuntimeError(
            f"LIVE_DATA_MISSING:{symbol}"
        )

    frame = pd.read_csv(path)

    required = {
        "completion_time_utc",
        "open",
        "high",
        "low",
        "close",
        "volume",
    }

    missing = (
        required
        - set(frame.columns)
    )

    if missing:
        raise RuntimeError(
            "LIVE_DATA_COLUMNS_MISSING:"
            f"{symbol}:"
            f"{sorted(missing)}"
        )

    frame[
        "completion_time_utc"
    ] = pd.to_datetime(
        frame[
            "completion_time_utc"
        ],
        utc=True,
    )

    for column in (
        "open",
        "high",
        "low",
        "close",
        "volume",
    ):
        frame[column] = (
            pd.to_numeric(
                frame[column],
                errors="raise",
            )
        )

    frame = (
        frame[
            [
                "completion_time_utc",
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]
        ]
        .drop_duplicates(
            subset=(
                "completion_time_utc"
            ),
            keep="last",
        )
        .set_index(
            "completion_time_utc"
        )
        .sort_index()
    )

    if len(frame) < 1000:
        raise RuntimeError(
            "INSUFFICIENT_LIVE_WARMUP:"
            f"{symbol}:{len(frame)}"
        )

    if not (
        frame.index
        .is_monotonic_increasing
    ):
        raise RuntimeError(
            "LIVE_INDEX_NOT_MONOTONIC:"
            f"{symbol}"
        )

    return frame


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
        v3ky_load_1h(symbol)
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
    print(
        "V3KY FORWARD RUNNER: "
        "WAITING FOR FIRST "
        "ELIGIBLE CLOSED BAR"
    )
    raise SystemExit(0)


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

v3ky_last_lock_state = (
    False,
    False,
    False,
    False,
)

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


    for candidate in due_candidates:

        symbol = candidate["symbol"]


        if block_new_entries:
            counters[
                "skipped_entry_block"
            ] += 1
            v3ky_decision(
                event_time,
                symbol,
                candidate,
                "SKIPPED_ENTRY_BLOCK",
            )
            continue


        if symbol in positions:
            counters[
                "skipped_existing_position"
            ] += 1
            v3ky_decision(
                event_time,
                symbol,
                candidate,
                "SKIPPED_EXISTING_POSITION",
            )
            continue


        if (
            len(positions)
            >= MAX_OPEN_POSITIONS
        ):
            counters[
                "skipped_max_positions"
            ] += 1
            v3ky_decision(
                event_time,
                symbol,
                candidate,
                "SKIPPED_MAX_POSITIONS",
            )
            continue


        if (
            MAX_ONE_NEW_ENTRY_PER_HOUR
            and entries_this_hour >= 1
        ):
            counters[
                "skipped_one_entry_hour"
            ] += 1
            v3ky_decision(
                event_time,
                symbol,
                candidate,
                "SKIPPED_ONE_ENTRY_HOUR",
            )
            continue


        if not high_beta_allows(
            symbol,
            positions,
        ):
            counters[
                "skipped_high_beta"
            ] += 1
            v3ky_decision(
                event_time,
                symbol,
                candidate,
                "SKIPPED_HIGH_BETA",
            )
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
            counters[
                "skipped_invalid_risk"
            ] += 1
            v3ky_decision(
                event_time,
                symbol,
                candidate,
                "SKIPPED_INVALID_RISK",
            )
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
            counters[
                "skipped_risk_cap"
            ] += 1
            v3ky_decision(
                event_time,
                symbol,
                candidate,
                "SKIPPED_RISK_CAP",
            )
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
            counters[
                "skipped_min_notional"
            ] += 1
            v3ky_decision(
                event_time,
                symbol,
                candidate,
                "SKIPPED_MIN_NOTIONAL",
            )
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
            counters[
                "skipped_affordability"
            ] += 1
            v3ky_decision(
                event_time,
                symbol,
                candidate,
                "SKIPPED_AFFORDABILITY",
            )
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
        }

        v3ky_decision(
            event_time,
            symbol,
            candidate,
            "EXECUTED",
        )

        v3ky_record(
            "ENTRY",
            event_time,
            symbol=symbol,
            signal_state="ENTRY_SIGNAL",
            signal_timestamp=v3ky_iso(candidate["signal_time"]),
            pending_entry_state=False,
            entry_decision="EXECUTED",
            entry_execution_timestamp=v3ky_iso(event_time),
            entry_execution_price=float(entry_price),
            quantity=float(quantity),
            initial_stop=float(initial_stop),
            active_stop=float(initial_stop),
            initial_risk_eur=float(initial_risk_eur),
            fees_eur=float(buy_fee),
            cash_eur=float(cash_balance),
            open_positions=len(positions),
            open_risk_eur=float(total_open_risk_eur(positions)),
            data_health_state="OK",
            notes="Canonical V3KJ paper entry.",
        )

        entries_this_hour += 1
        counters["entries"] += 1

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

                v3ky_record(
                    "DECISION",
                    event_time,
                    symbol=symbol,
                    signal_state="ENTRY_SIGNAL",
                    signal_timestamp=v3ky_iso(event_time),
                    pending_entry_state=True,
                    entry_decision="QUEUED_NEXT_HOUR",
                    data_health_state="OK",
                    notes="Canonical V3KJ signal queued.",
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


    v3ky_record(
        "EQUITY",
        event_time,
        cash_eur=float(cash_balance),
        equity_eur=float(ending_equity),
        open_positions=len(positions),
        open_risk_eur=float(
            total_open_risk_eur(positions)
        ),
        daily_lock_state=bool(
            daily_entry_block
        ),
        weekly_lock_state=bool(
            weekly_entry_block
        ),
        hard_drawdown_lock_state=bool(
            hard_lock
        ),
        data_health_state="OK",
        notes="Canonical hourly portfolio mark.",
    )

    v3ky_lock_state = (
        bool(hard_lock),
        bool(daily_entry_block),
        bool(weekly_entry_block),
        bool(daily_profit_entry_block),
    )

    if v3ky_lock_state != v3ky_last_lock_state:

        v3ky_record(
            "RISK_LOCK",
            event_time,
            cash_eur=float(cash_balance),
            equity_eur=float(ending_equity),
            open_positions=len(positions),
            open_risk_eur=float(
                total_open_risk_eur(positions)
            ),
            daily_lock_state=bool(
                daily_entry_block
            ),
            weekly_lock_state=bool(
                weekly_entry_block
            ),
            hard_drawdown_lock_state=bool(
                hard_lock
            ),
            data_health_state="OK",
            notes=(
                "Risk-state transition: "
                f"hard={hard_lock}; "
                f"daily={daily_entry_block}; "
                f"weekly={weekly_entry_block}; "
                f"profit={daily_profit_entry_block}"
            ),
        )

        v3ky_last_lock_state = (
            v3ky_lock_state
        )

    if hard_lock:
        break


# ============================================================================
# END-OF-WINDOW MARK
# ============================================================================


if False and positions and not hard_lock:

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
