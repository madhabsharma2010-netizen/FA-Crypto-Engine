from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from config.risk_settings import RISK_SETTINGS
from core.surveillance_v3d import (
    AssetShockDecision,
    MarketShockDecision,
    MarketState,
    MarketStateDecision,
    ShockLevel,
    evaluate_asset_shock,
    evaluate_market_shock,
    evaluate_market_state,
)
from run_v3d_diagnostics import (
    BACKTEST_END,
    BACKTEST_START,
    SYMBOLS,
    align_frame,
    breadth_score_at,
    common_index,
    load_frame,
    prepare_15m,
    prepare_1h,
)
from strategies.pullback_candidate_v3d2 import (
    PullbackCandidate,
    evaluate_pullback_candidate,
)


# ============================================================================
# V3D3 PORTFOLIO SETTINGS
# ============================================================================

STARTING_CAPITAL = float(RISK_SETTINGS.starting_capital)
TRADING_FEE_PERCENT = float(RISK_SETTINGS.trading_fee_percent)
SLIPPAGE_PERCENT = float(RISK_SETTINGS.estimated_slippage_percent)

MAX_OPEN_POSITIONS = 2
MAX_TOTAL_DEPLOYMENT_PERCENT = 50.0
MAX_TOTAL_OPEN_RISK_PERCENT = 0.75
MIN_NOTIONAL_EUR = 500.0
MAX_ONE_NEW_ENTRY_PER_HOUR = True

DAILY_LOSS_LIMIT_PERCENT = 1.00
WEEKLY_LOSS_LIMIT_PERCENT = 2.50
HARD_DRAWDOWN_LIMIT_PERCENT = 5.00
DAILY_PROFIT_ENTRY_LOCK_PERCENT = 2.00

PARTIAL_PROFIT_R = 2.00
PARTIAL_SELL_PERCENT = 25.0
BREAKEVEN_TRIGGER_R = 1.00
NO_PROGRESS_HOURS = 36
NO_PROGRESS_MIN_R = 0.50

NORMAL_EXIT_COOLDOWN_HOURS = 6
DEFENSIVE_EXIT_COOLDOWN_HOURS = 12

TRADE_HISTORY_FILE = Path(
    "logs/backtests/v3d3_portfolio_trade_history.csv"
)
EVENT_HISTORY_FILE = Path(
    "logs/backtests/v3d3_portfolio_event_history.csv"
)
SUMMARY_FILE = Path(
    "reports/v3d3_portfolio_backtest_summary.csv"
)
SYMBOL_SUMMARY_FILE = Path(
    "reports/v3d3_portfolio_symbol_summary.csv"
)


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

HIGH_BETA_ASSETS = {
    "SOLUSDT",
    "XRPUSDT",
    "LINKUSDT",
    "DOGEUSDT",
}

DEFENSIVE_EXIT_REASONS = {
    "STOP LOSS",
    "ASSET SEVERE SHOCK",
    "MARKET SEVERE SHOCK",
    "DAILY LOSS LIMIT",
    "WEEKLY LOSS LIMIT",
    "HARD DRAWDOWN LOCK",
}


@dataclass
class Position:
    symbol: str
    entry_time: pd.Timestamp
    entry_price: float
    initial_stop: float
    current_stop: float
    initial_risk_per_unit: float
    initial_quantity: float
    remaining_quantity: float
    entry_notional: float
    buy_fee: float
    entry_total_cost: float
    account_risk_percent: float
    setup_score: float
    entry_reward_risk: float
    highest_price: float
    partial_done: bool = False
    partial_quantity: float = 0.0
    realized_net_proceeds: float = 0.0
    total_sell_fees: float = 0.0
    asset_shock_reduced: bool = False
    market_shock_reduced: bool = False


# ============================================================================
# BASIC EXECUTION HELPERS
# ============================================================================


def calculate_fee(amount: float) -> float:
    return amount * TRADING_FEE_PERCENT / 100.0


def apply_buy_slippage(market_price: float) -> float:
    return market_price * (1.0 + SLIPPAGE_PERCENT / 100.0)


def apply_sell_slippage(market_price: float) -> float:
    return market_price * (1.0 - SLIPPAGE_PERCENT / 100.0)


def export_csv(rows: list[dict[str, Any]], file_path: Path) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        file_path.write_text("", encoding="utf-8")
        return

    with file_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def portfolio_equity(
    cash_balance: float,
    positions: dict[str, Position],
    price_by_symbol: dict[str, float],
) -> float:
    market_value = sum(
        position.remaining_quantity * price_by_symbol[position.symbol]
        for position in positions.values()
    )
    return cash_balance + market_value


def portfolio_market_value(
    positions: dict[str, Position],
    price_by_symbol: dict[str, float],
) -> float:
    return sum(
        position.remaining_quantity * price_by_symbol[position.symbol]
        for position in positions.values()
    )


def maximum_drawdown(equity_history: list[float]) -> float:
    if not equity_history:
        return 0.0

    peak = equity_history[0]
    result = 0.0

    for equity in equity_history:
        peak = max(peak, equity)
        if peak <= 0:
            continue
        result = max(result, (peak - equity) / peak * 100.0)

    return result


def percentile_rank(values: pd.Series) -> pd.Series:
    if len(values) <= 1:
        return pd.Series(100.0, index=values.index)
    return values.rank(method="average", pct=True) * 100.0


# ============================================================================
# DATA PREPARATION AND SURVEILLANCE TABLES
# ============================================================================


def prepare_frames() -> tuple[
    dict[str, pd.DataFrame],
    dict[str, pd.DataFrame],
    pd.DatetimeIndex,
    pd.DatetimeIndex,
]:
    frames_15m = {
        symbol: prepare_15m(symbol)
        for symbol in SYMBOLS
    }

    frames_1h = {
        symbol: prepare_1h(symbol)
        for symbol in SYMBOLS
    }

    for frame in frames_1h.values():
        trend_condition = (
            (frame["close"] > frame["EMA200"])
            & (frame["EMA20"] > frame["EMA50"])
            & (frame["EMA50"] > frame["EMA200"])
        )
        frame["TREND_STABLE_4_OF_6"] = (
            trend_condition.rolling(6).sum().ge(4)
        )

    common_15m = common_index(
        frames_15m,
        BACKTEST_START,
        BACKTEST_END,
    )
    common_1h = common_index(
        frames_1h,
        BACKTEST_START,
        BACKTEST_END,
    )

    market_return_60m = pd.DataFrame(
        {
            symbol: frames_15m[symbol]["RETURN_60M"]
            for symbol in SYMBOLS
        }
    ).median(axis=1)

    for symbol in SYMBOLS:
        frames_15m[symbol]["MARKET_RELATIVE_60M"] = (
            frames_15m[symbol]["RETURN_60M"]
            - market_return_60m
        )

    return frames_15m, frames_1h, common_15m, common_1h


def build_shock_tables(
    frames_15m: dict[str, pd.DataFrame],
    common_15m: pd.DatetimeIndex,
) -> tuple[
    dict[str, dict[pd.Timestamp, AssetShockDecision]],
    dict[pd.Timestamp, MarketShockDecision],
]:
    asset_decisions: dict[
        str,
        dict[pd.Timestamp, AssetShockDecision],
    ] = {symbol: {} for symbol in SYMBOLS}

    market_decisions: dict[
        pd.Timestamp,
        MarketShockDecision,
    ] = {}

    for decision_time in common_15m:
        current_asset_decisions = []
        returns_15m = []
        returns_60m = []

        for symbol in SYMBOLS:
            candle = frames_15m[symbol].loc[decision_time]
            decision = evaluate_asset_shock(symbol, candle)
            asset_decisions[symbol][decision_time] = decision
            current_asset_decisions.append(decision)
            returns_15m.append(float(candle["RETURN_15M"]))
            returns_60m.append(float(candle["RETURN_60M"]))

        market_decisions[decision_time] = evaluate_market_shock(
            current_asset_decisions,
            median_return_15m=float(pd.Series(returns_15m).median()),
            median_return_60m=float(pd.Series(returns_60m).median()),
        )

    return asset_decisions, market_decisions


def build_market_states(
    frames_1h: dict[str, pd.DataFrame],
    common_1h: pd.DatetimeIndex,
    market_shocks: dict[pd.Timestamp, MarketShockDecision],
) -> dict[pd.Timestamp, MarketStateDecision]:
    btc_2h = load_frame("BTCUSDT", "2h")
    btc_4h = load_frame("BTCUSDT", "4h")
    btc_1d = load_frame("BTCUSDT", "1d")
    eth_2h = load_frame("ETHUSDT", "2h")
    eth_4h = load_frame("ETHUSDT", "4h")

    aligned_btc_2h = align_frame(btc_2h, common_1h)
    aligned_btc_4h = align_frame(btc_4h, common_1h)
    aligned_btc_1d = align_frame(btc_1d, common_1h)
    aligned_eth_2h = align_frame(eth_2h, common_1h)
    aligned_eth_4h = align_frame(eth_4h, common_1h)

    market_shock_series = pd.Series(
        {
            timestamp: decision.level.value
            for timestamp, decision in market_shocks.items()
        }
    ).sort_index()

    aligned_market_shocks = market_shock_series.reindex(
        common_1h,
        method="ffill",
    )

    states: dict[pd.Timestamp, MarketStateDecision] = {}

    for decision_time in common_1h:
        breadth = breadth_score_at(frames_1h, decision_time)
        shock_level = ShockLevel(
            str(aligned_market_shocks.loc[decision_time])
        )

        states[decision_time] = evaluate_market_state(
            btc_1h=frames_1h["BTCUSDT"].loc[decision_time],
            btc_2h=aligned_btc_2h.loc[decision_time],
            btc_4h=aligned_btc_4h.loc[decision_time],
            btc_1d=aligned_btc_1d.loc[decision_time],
            eth_1h=frames_1h["ETHUSDT"].loc[decision_time],
            eth_2h=aligned_eth_2h.loc[decision_time],
            eth_4h=aligned_eth_4h.loc[decision_time],
            breadth_score=breadth,
            market_shock=shock_level,
        )

    return states


def build_candidates(
    frames_1h: dict[str, pd.DataFrame],
    common_1h: pd.DatetimeIndex,
    market_states: dict[pd.Timestamp, MarketStateDecision],
    asset_shocks: dict[str, dict[pd.Timestamp, AssetShockDecision]],
) -> dict[pd.Timestamp, list[PullbackCandidate]]:
    shock_frames = {}

    for symbol in SYMBOLS:
        shock_frames[symbol] = pd.Series(
            {
                timestamp: decision
                for timestamp, decision in asset_shocks[symbol].items()
            }
        ).sort_index().reindex(common_1h, method="ffill")

    result: dict[pd.Timestamp, list[PullbackCandidate]] = {}

    for index in range(1, len(common_1h)):
        decision_time = common_1h[index]
        previous_time = common_1h[index - 1]
        candidates = []

        for symbol in SYMBOLS:
            candidate = evaluate_pullback_candidate(
                symbol=symbol,
                candle=frames_1h[symbol].loc[decision_time],
                previous=frames_1h[symbol].loc[previous_time],
                market_state=market_states[decision_time],
                asset_shock=shock_frames[symbol].loc[decision_time],
            )
            candidates.append(candidate)

        result[decision_time] = candidates

    return result


# ============================================================================
# CANDIDATE RANKING AND RISK-BASED SIZING
# ============================================================================


def rank_valid_candidates(
    candidates: list[PullbackCandidate],
    frames_1h: dict[str, pd.DataFrame],
    decision_time: pd.Timestamp,
) -> list[tuple[PullbackCandidate, float]]:
    valid = [candidate for candidate in candidates if candidate.valid]

    if not valid:
        return []

    table = pd.DataFrame(
        {
            "symbol": [candidate.symbol for candidate in valid],
            "setup_score": [candidate.score for candidate in valid],
            "reward_risk": [candidate.reward_risk for candidate in valid],
            "risk_percent": [candidate.risk_percent for candidate in valid],
            "momentum_24h": [
                float(frames_1h[candidate.symbol].loc[decision_time, "MOMENTUM_24H"])
                for candidate in valid
            ],
            "slope": [
                float(frames_1h[candidate.symbol].loc[decision_time, "EMA50_SLOPE_12H"])
                for candidate in valid
            ],
            "adx": [
                float(frames_1h[candidate.symbol].loc[decision_time, "ADX14"])
                for candidate in valid
            ],
        }
    ).set_index("symbol")

    table["setup_rank"] = percentile_rank(table["setup_score"])
    table["rr_rank"] = percentile_rank(table["reward_risk"])
    table["momentum_rank"] = percentile_rank(table["momentum_24h"])
    table["slope_rank"] = percentile_rank(table["slope"])
    table["adx_rank"] = percentile_rank(table["adx"])

    table["selection_score"] = (
        table["setup_rank"] * 0.55
        + table["rr_rank"] * 0.20
        + table["momentum_rank"] * 0.10
        + table["slope_rank"] * 0.10
        + table["adx_rank"] * 0.05
    )

    by_symbol = {candidate.symbol: candidate for candidate in valid}

    return [
        (by_symbol[symbol], float(row["selection_score"]))
        for symbol, row in table.sort_values(
            ["selection_score", "reward_risk", "setup_score"],
            ascending=False,
        ).iterrows()
    ]


def total_open_risk_eur(positions: dict[str, Position]) -> float:
    return sum(
        position.remaining_quantity
        * max(position.entry_price - position.current_stop, 0.0)
        for position in positions.values()
    )


def correlation_allows(
    symbol: str,
    positions: dict[str, Position],
) -> bool:
    if symbol not in HIGH_BETA_ASSETS:
        return True

    return not any(
        existing_symbol in HIGH_BETA_ASSETS
        for existing_symbol in positions
    )


# ============================================================================
# BACKTEST ENGINE
# ============================================================================


def run_backtest() -> dict[str, Any]:
    print("=" * 126)
    print("FA CRYPTO ENGINE — V3D3 SURVEILLANCE + PULLBACK PORTFOLIO BACKTEST")
    print("=" * 126)

    frames_15m, frames_1h, common_15m, common_1h = prepare_frames()

    print("Building 15m asset and market surveillance...")
    asset_shocks, market_shocks = build_shock_tables(
        frames_15m,
        common_15m,
    )

    print("Building multi-timeframe market states...")
    market_states = build_market_states(
        frames_1h,
        common_1h,
        market_shocks,
    )

    print("Building valid 1h pullback candidates...")
    candidates_by_time = build_candidates(
        frames_1h,
        common_1h,
        market_states,
        asset_shocks,
    )

    # Align completed 1h data and market state to each completed 15m candle.
    aligned_1h_to_15m = {
        symbol: frames_1h[symbol].reindex(
            common_15m,
            method="ffill",
        )
        for symbol in SYMBOLS
    }

    state_series = pd.Series(market_states).sort_index()
    aligned_states_15m = state_series.reindex(
        common_15m,
        method="ffill",
    )

    cash_balance = STARTING_CAPITAL
    positions: dict[str, Position] = {}
    cooldown_until: dict[str, pd.Timestamp] = {
        symbol: pd.Timestamp.min
        for symbol in SYMBOLS
    }

    trade_history: list[dict[str, Any]] = []
    event_history: list[dict[str, Any]] = []
    equity_history: list[float] = []

    total_fees = 0.0
    partial_profit_events = 0
    asset_shock_reductions = 0
    market_shock_reductions = 0
    warning_stop_tightens = 0
    skipped_gap_entries = 0
    skipped_risk_entries = 0
    skipped_correlation_entries = 0
    skipped_small_entries = 0
    daily_loss_stops = 0
    weekly_loss_stops = 0
    hard_lock_hits = 0
    daily_profit_locks = 0

    peak_equity = STARTING_CAPITAL
    current_day = None
    current_week = None
    daily_start_equity = STARTING_CAPITAL
    weekly_start_equity = STARTING_CAPITAL
    daily_entry_block = False
    weekly_entry_block = False
    daily_profit_entry_block = False
    hard_lock = False

    previous_market_shock_level = ShockLevel.NORMAL
    previous_asset_shock_level = {
        symbol: ShockLevel.NORMAL
        for symbol in SYMBOLS
    }

    selected_counts = Counter()
    valid_candidate_counts = Counter()
    state_counts = Counter()
    shock_counts = Counter()

    last_price = {
        symbol: float(frames_15m[symbol].loc[common_15m[0], "close"])
        for symbol in SYMBOLS
    }

    def record_event(
        event_time: pd.Timestamp,
        symbol: str,
        event_type: str,
        details: str,
    ) -> None:
        event_history.append(
            {
                "event_time": event_time,
                "symbol": symbol,
                "event_type": event_type,
                "details": details,
            }
        )

    def reduce_position(
        symbol: str,
        percentage: float,
        market_price: float,
        event_time: pd.Timestamp,
        reason: str,
    ) -> None:
        nonlocal cash_balance, total_fees

        position = positions.get(symbol)
        if position is None or percentage <= 0:
            return

        quantity = position.remaining_quantity * percentage / 100.0
        quantity = min(quantity, position.remaining_quantity)

        if quantity <= 0:
            return

        sell_price = apply_sell_slippage(market_price)
        gross_value = quantity * sell_price
        fee = calculate_fee(gross_value)
        net_value = gross_value - fee

        cash_balance += net_value
        total_fees += fee
        position.remaining_quantity -= quantity
        position.realized_net_proceeds += net_value
        position.total_sell_fees += fee

        record_event(
            event_time,
            symbol,
            reason,
            (
                f"Reduced {percentage:.0f}% at {sell_price:.8f}; "
                f"remaining quantity {position.remaining_quantity:.8f}"
            ),
        )

        print(
            f"{event_time} | {symbol} REDUCE {percentage:.0f}% @ "
            f"{sell_price:.6f} | {reason}"
        )

    def close_position(
        symbol: str,
        market_price: float,
        event_time: pd.Timestamp,
        reason: str,
    ) -> None:
        nonlocal cash_balance, total_fees

        position = positions.get(symbol)
        if position is None:
            return

        sell_price = apply_sell_slippage(market_price)
        gross_value = position.remaining_quantity * sell_price
        sell_fee = calculate_fee(gross_value)
        net_value = gross_value - sell_fee

        cash_balance += net_value
        total_fees += sell_fee

        total_net_proceeds = position.realized_net_proceeds + net_value
        total_sell_fees = position.total_sell_fees + sell_fee
        profit = total_net_proceeds - position.entry_total_cost
        profit_percent = (
            profit / position.entry_total_cost * 100.0
            if position.entry_total_cost > 0
            else 0.0
        )

        trade_history.append(
            {
                "trade_number": len(trade_history) + 1,
                "symbol": symbol,
                "entry_time": position.entry_time,
                "exit_time": event_time,
                "entry_price": round(position.entry_price, 8),
                "exit_price": round(sell_price, 8),
                "initial_stop": round(position.initial_stop, 8),
                "final_stop": round(position.current_stop, 8),
                "initial_quantity": round(position.initial_quantity, 8),
                "partial_quantity": round(position.partial_quantity, 8),
                "entry_notional": round(position.entry_notional, 8),
                "account_risk_percent": round(position.account_risk_percent, 4),
                "setup_score": round(position.setup_score, 4),
                "entry_reward_risk": round(position.entry_reward_risk, 4),
                "buy_fee": round(position.buy_fee, 8),
                "sell_fees": round(total_sell_fees, 8),
                "partial_profit_taken": position.partial_done,
                "profit": round(profit, 8),
                "profit_percent": round(profit_percent, 4),
                "exit_reason": reason,
                "cash_after_trade": round(cash_balance, 8),
            }
        )

        cooldown_hours = (
            DEFENSIVE_EXIT_COOLDOWN_HOURS
            if reason in DEFENSIVE_EXIT_REASONS
            else NORMAL_EXIT_COOLDOWN_HOURS
        )
        cooldown_until[symbol] = event_time + pd.Timedelta(hours=cooldown_hours)

        print(
            f"{event_time} | {symbol} SELL @ {sell_price:.6f} | "
            f"{reason} | P&L €{profit:.2f}"
        )

        del positions[symbol]

    def close_all(
        event_time: pd.Timestamp,
        prices: dict[str, float],
        reason: str,
    ) -> None:
        for symbol in list(positions.keys()):
            close_position(
                symbol,
                prices[symbol],
                event_time,
                reason,
            )

    def manage_15m_candle(event_time: pd.Timestamp) -> None:
        nonlocal cash_balance
        nonlocal total_fees
        nonlocal partial_profit_events
        nonlocal asset_shock_reductions
        nonlocal market_shock_reductions
        nonlocal warning_stop_tightens
        nonlocal daily_loss_stops
        nonlocal weekly_loss_stops
        nonlocal hard_lock_hits
        nonlocal daily_profit_locks
        nonlocal peak_equity
        nonlocal current_day
        nonlocal current_week
        nonlocal daily_start_equity
        nonlocal weekly_start_equity
        nonlocal daily_entry_block
        nonlocal weekly_entry_block
        nonlocal daily_profit_entry_block
        nonlocal hard_lock
        nonlocal previous_market_shock_level

        price_by_symbol = {}

        for symbol in SYMBOLS:
            candle = frames_15m[symbol].loc[event_time]
            last_price[symbol] = float(candle["close"])
            price_by_symbol[symbol] = last_price[symbol]

        timestamp = pd.Timestamp(event_time)
        day_key = timestamp.date()
        iso = timestamp.isocalendar()
        week_key = (int(iso.year), int(iso.week))

        current_equity = portfolio_equity(
            cash_balance,
            positions,
            price_by_symbol,
        )

        if current_day != day_key:
            current_day = day_key
            daily_start_equity = current_equity
            daily_entry_block = False
            daily_profit_entry_block = False

        if current_week != week_key:
            current_week = week_key
            weekly_start_equity = current_equity
            weekly_entry_block = False

        peak_equity = max(peak_equity, current_equity)

        drawdown = (
            (peak_equity - current_equity) / peak_equity * 100.0
            if peak_equity > 0
            else 0.0
        )
        daily_return = (
            (current_equity - daily_start_equity)
            / daily_start_equity
            * 100.0
            if daily_start_equity > 0
            else 0.0
        )
        weekly_return = (
            (current_equity - weekly_start_equity)
            / weekly_start_equity
            * 100.0
            if weekly_start_equity > 0
            else 0.0
        )

        if not hard_lock and drawdown >= HARD_DRAWDOWN_LIMIT_PERCENT:
            hard_lock = True
            hard_lock_hits += 1
            close_all(event_time, price_by_symbol, "HARD DRAWDOWN LOCK")
            record_event(
                event_time,
                "PORTFOLIO",
                "HARD LOCK",
                f"Drawdown {drawdown:.2f}%",
            )
            equity_history.append(
                portfolio_equity(
                    cash_balance,
                    positions,
                    price_by_symbol,
                )
            )
            return

        if (
            not hard_lock
            and not weekly_entry_block
            and weekly_return <= -WEEKLY_LOSS_LIMIT_PERCENT
        ):
            weekly_entry_block = True
            weekly_loss_stops += 1
            close_all(event_time, price_by_symbol, "WEEKLY LOSS LIMIT")
            record_event(
                event_time,
                "PORTFOLIO",
                "WEEKLY STOP",
                f"Weekly return {weekly_return:.2f}%",
            )

        if (
            not hard_lock
            and not daily_entry_block
            and daily_return <= -DAILY_LOSS_LIMIT_PERCENT
        ):
            daily_entry_block = True
            daily_loss_stops += 1
            close_all(event_time, price_by_symbol, "DAILY LOSS LIMIT")
            record_event(
                event_time,
                "PORTFOLIO",
                "DAILY STOP",
                f"Daily return {daily_return:.2f}%",
            )

        if (
            not daily_profit_entry_block
            and daily_return >= DAILY_PROFIT_ENTRY_LOCK_PERCENT
        ):
            daily_profit_entry_block = True
            daily_profit_locks += 1
            record_event(
                event_time,
                "PORTFOLIO",
                "DAILY PROFIT LOCK",
                f"Daily return {daily_return:.2f}%; winners continue",
            )

        market_shock = market_shocks[event_time]
        shock_counts[market_shock.level.value] += 1

        market_entered_shock = (
            market_shock.level in {ShockLevel.SHOCK, ShockLevel.SEVERE}
            and previous_market_shock_level
            not in {ShockLevel.SHOCK, ShockLevel.SEVERE}
        )

        if market_shock.level == ShockLevel.SEVERE:
            close_all(event_time, price_by_symbol, "MARKET SEVERE SHOCK")
            previous_market_shock_level = market_shock.level
            for tracked_symbol in SYMBOLS:
                previous_asset_shock_level[tracked_symbol] = (
                    asset_shocks[tracked_symbol][event_time].level
                )
            equity_history.append(
                portfolio_equity(
                    cash_balance,
                    positions,
                    price_by_symbol,
                )
            )
            return

        if market_shock.level == ShockLevel.SHOCK and market_entered_shock:
            for symbol in list(positions.keys()):
                position = positions.get(symbol)
                if position is None or position.market_shock_reduced:
                    continue
                reduce_position(
                    symbol,
                    50.0,
                    price_by_symbol[symbol],
                    event_time,
                    "MARKET SHOCK REDUCTION",
                )
                position.market_shock_reduced = True
                market_shock_reductions += 1

        if market_shock.level == ShockLevel.NORMAL:
            for position in positions.values():
                position.market_shock_reduced = False

        previous_market_shock_level = market_shock.level

        for symbol in list(positions.keys()):
            position = positions.get(symbol)
            if position is None:
                continue

            candle = frames_15m[symbol].loc[event_time]
            close_price = float(candle["close"])
            high_price = float(candle["high"])
            low_price = float(candle["low"])
            atr_15m = float(candle["ATR14"])
            one_hour = aligned_1h_to_15m[symbol].loc[event_time]
            atr_1h = float(one_hour["ATR14"])
            state = aligned_states_15m.loc[event_time]
            asset_shock = asset_shocks[symbol][event_time]

            # Conservative intrabar ordering: an already-active stop is
            # checked before a new high can raise the trailing stop.
            if low_price <= position.current_stop:
                close_position(
                    symbol,
                    position.current_stop,
                    event_time,
                    "STOP LOSS",
                )
                previous_asset_shock_level[symbol] = asset_shock.level
                continue

            position.highest_price = max(
                position.highest_price,
                high_price,
            )

            highest_r = (
                (position.highest_price - position.entry_price)
                / position.initial_risk_per_unit
                if position.initial_risk_per_unit > 0
                else 0.0
            )

            if highest_r >= BREAKEVEN_TRIGGER_R:
                fee_buffer_stop = position.entry_price * (
                    1.0 + (TRADING_FEE_PERCENT + SLIPPAGE_PERCENT) / 100.0
                )
                position.current_stop = max(
                    position.current_stop,
                    fee_buffer_stop,
                )

            if (
                highest_r >= PARTIAL_PROFIT_R
                and not position.partial_done
            ):
                target_price = (
                    position.entry_price
                    + position.initial_risk_per_unit * PARTIAL_PROFIT_R
                )
                quantity_before = position.remaining_quantity
                reduce_position(
                    symbol,
                    PARTIAL_SELL_PERCENT,
                    target_price,
                    event_time,
                    "2R PARTIAL PROFIT",
                )
                position = positions.get(symbol)
                if position is None:
                    continue
                position.partial_done = True
                position.partial_quantity += (
                    quantity_before - position.remaining_quantity
                )
                partial_profit_events += 1

            trailing_multiple = {
                MarketState.STRONG_BULL: 3.20,
                MarketState.BULL: 2.70,
                MarketState.NEUTRAL: 2.20,
                MarketState.BEARISH_RECOVERY: 1.80,
                MarketState.BEAR: 1.50,
                MarketState.SHOCK: 1.00,
            }[state.state]

            if highest_r >= BREAKEVEN_TRIGGER_R:
                normal_trailing_stop = (
                    position.highest_price
                    - atr_1h * trailing_multiple
                )
                position.current_stop = max(
                    position.current_stop,
                    normal_trailing_stop,
                )

            previous_level = previous_asset_shock_level[symbol]
            entered_asset_shock = (
                asset_shock.level in {ShockLevel.SHOCK, ShockLevel.SEVERE}
                and previous_level
                not in {ShockLevel.SHOCK, ShockLevel.SEVERE}
            )

            if asset_shock.level == ShockLevel.SEVERE:
                close_position(
                    symbol,
                    close_price,
                    event_time,
                    "ASSET SEVERE SHOCK",
                )
                previous_asset_shock_level[symbol] = asset_shock.level
                continue

            if (
                asset_shock.level == ShockLevel.SHOCK
                and entered_asset_shock
                and not position.asset_shock_reduced
                and market_shock.level != ShockLevel.SHOCK
            ):
                reduce_position(
                    symbol,
                    35.0,
                    close_price,
                    event_time,
                    "ASSET SHOCK REDUCTION",
                )
                position = positions.get(symbol)
                if position is None:
                    previous_asset_shock_level[symbol] = asset_shock.level
                    continue
                position.asset_shock_reduced = True
                asset_shock_reductions += 1

            if asset_shock.level == ShockLevel.NORMAL:
                position.asset_shock_reduced = False

            if asset_shock.emergency_atr_multiple is not None:
                emergency_stop = (
                    close_price
                    - atr_15m * asset_shock.emergency_atr_multiple
                )
                old_stop = position.current_stop
                position.current_stop = max(
                    position.current_stop,
                    emergency_stop,
                )
                if position.current_stop > old_stop:
                    warning_stop_tightens += 1

            # A portfolio warning also tightens, but never lowers, stops.
            if market_shock.level == ShockLevel.WARNING:
                market_warning_stop = close_price - atr_15m * 1.80
                position.current_stop = max(
                    position.current_stop,
                    market_warning_stop,
                )

            if close_price <= position.current_stop:
                close_position(
                    symbol,
                    close_price,
                    event_time,
                    "EMERGENCY STOP EXIT",
                )
                previous_asset_shock_level[symbol] = asset_shock.level
                continue

            age_hours = (
                event_time - position.entry_time
            ).total_seconds() / 3600.0
            current_r = (
                (close_price - position.entry_price)
                / position.initial_risk_per_unit
                if position.initial_risk_per_unit > 0
                else 0.0
            )

            if (
                age_hours >= NO_PROGRESS_HOURS
                and current_r < NO_PROGRESS_MIN_R
                and state.state
                not in {MarketState.STRONG_BULL}
            ):
                close_position(
                    symbol,
                    close_price,
                    event_time,
                    "TIME STOP",
                )

            previous_asset_shock_level[symbol] = asset_shock.level

        # Keep shock episode state current even when an asset has no open
        # position, so future reductions trigger only on genuinely new shocks.
        for tracked_symbol in SYMBOLS:
            previous_asset_shock_level[tracked_symbol] = (
                asset_shocks[tracked_symbol][event_time].level
            )

        current_equity = portfolio_equity(
            cash_balance,
            positions,
            price_by_symbol,
        )
        equity_history.append(current_equity)

    # Process 15m data in chronological order. At completed 1h timestamps,
    # valid candidates are evaluated after that 15m close and executed at the
    # new 1h candle open with slippage.
    hour_set = set(common_1h)
    next_hour_open = {}

    for symbol in SYMBOLS:
        frame = frames_1h[symbol]
        for index in range(len(common_1h) - 1):
            decision_time = common_1h[index]
            next_time = common_1h[index + 1]
            next_hour_open[(symbol, decision_time)] = float(
                frame.loc[next_time, "open"]
            )

    for event_time in common_15m:
        manage_15m_candle(event_time)

        if hard_lock:
            break

        if event_time not in hour_set:
            continue

        market_state = market_states.get(event_time)
        if market_state is None:
            continue

        state_counts[market_state.state.value] += 1

        candidates = candidates_by_time.get(event_time, [])
        for candidate in candidates:
            if candidate.valid:
                valid_candidate_counts[candidate.symbol] += 1

        latest_market_shock = market_shocks[event_time]

        block_new_entries = any(
            (
                hard_lock,
                daily_entry_block,
                weekly_entry_block,
                daily_profit_entry_block,
                latest_market_shock.freeze_all_entries,
                not market_state.new_entries_allowed,
            )
        )

        if block_new_entries:
            continue

        ranked = rank_valid_candidates(
            candidates,
            frames_1h,
            event_time,
        )

        entries_this_hour = 0

        for candidate, selection_score in ranked:
            symbol = candidate.symbol

            if symbol in positions:
                continue

            if event_time < cooldown_until[symbol]:
                continue

            if len(positions) >= MAX_OPEN_POSITIONS:
                break

            if (
                MAX_ONE_NEW_ENTRY_PER_HOUR
                and entries_this_hour >= 1
            ):
                break

            if not correlation_allows(symbol, positions):
                skipped_correlation_entries += 1
                continue

            if (symbol, event_time) not in next_hour_open:
                continue

            raw_entry_price = next_hour_open[(symbol, event_time)]
            entry_price = apply_buy_slippage(raw_entry_price)
            signal_atr = float(
                frames_1h[symbol].loc[event_time, "ATR14"]
            )

            # Do not chase an opening gap after a valid close signal.
            maximum_entry = candidate.entry_price + signal_atr * 0.35
            if entry_price > maximum_entry:
                skipped_gap_entries += 1
                continue

            stop_price = candidate.stop_price
            risk_per_unit = entry_price - stop_price

            if risk_per_unit <= 0:
                skipped_risk_entries += 1
                continue

            projected_reward = candidate.projected_target - entry_price
            actual_reward_risk = projected_reward / risk_per_unit

            if actual_reward_risk < 1.40:
                skipped_gap_entries += 1
                continue

            current_prices = {
                item: last_price[item]
                for item in SYMBOLS
            }
            current_equity = portfolio_equity(
                cash_balance,
                positions,
                current_prices,
            )

            base_risk = BASE_RISK_PERCENT[symbol]
            risk_percent = base_risk * market_state.risk_multiplier

            # DOGE requires exceptional conditions even when its candidate is valid.
            if symbol == "DOGEUSDT":
                if not (
                    market_state.state == MarketState.STRONG_BULL
                    and candidate.score >= 85.0
                ):
                    continue

            allowed_risk_eur = current_equity * risk_percent / 100.0
            current_open_risk = total_open_risk_eur(positions)
            portfolio_risk_cap = (
                current_equity * MAX_TOTAL_OPEN_RISK_PERCENT / 100.0
            )
            remaining_risk_room = max(
                0.0,
                portfolio_risk_cap - current_open_risk,
            )
            allowed_risk_eur = min(
                allowed_risk_eur,
                remaining_risk_room,
            )

            if allowed_risk_eur <= 0:
                skipped_risk_entries += 1
                continue

            risk_sized_quantity = allowed_risk_eur / risk_per_unit
            risk_sized_notional = risk_sized_quantity * entry_price

            asset_notional_cap = (
                current_equity
                * ASSET_NOTIONAL_CAP_PERCENT[symbol]
                / 100.0
            )
            current_market_value = portfolio_market_value(
                positions,
                current_prices,
            )
            deployment_cap = (
                current_equity
                * MAX_TOTAL_DEPLOYMENT_PERCENT
                / 100.0
            )
            deployment_room = max(
                0.0,
                deployment_cap - current_market_value,
            )
            affordable_notional = cash_balance / (
                1.0 + TRADING_FEE_PERCENT / 100.0
            )

            notional = min(
                risk_sized_notional,
                asset_notional_cap,
                deployment_room,
                affordable_notional,
            )

            if notional < MIN_NOTIONAL_EUR:
                skipped_small_entries += 1
                continue

            quantity = notional / entry_price
            buy_fee = calculate_fee(notional)
            total_required = notional + buy_fee

            if quantity <= 0 or total_required > cash_balance:
                skipped_small_entries += 1
                continue

            actual_risk_eur = quantity * risk_per_unit
            actual_account_risk_percent = (
                actual_risk_eur / current_equity * 100.0
                if current_equity > 0
                else 0.0
            )

            cash_balance -= total_required
            total_fees += buy_fee

            positions[symbol] = Position(
                symbol=symbol,
                entry_time=event_time,
                entry_price=entry_price,
                initial_stop=stop_price,
                current_stop=stop_price,
                initial_risk_per_unit=risk_per_unit,
                initial_quantity=quantity,
                remaining_quantity=quantity,
                entry_notional=notional,
                buy_fee=buy_fee,
                entry_total_cost=total_required,
                account_risk_percent=actual_account_risk_percent,
                setup_score=candidate.score,
                entry_reward_risk=actual_reward_risk,
                highest_price=entry_price,
            )

            selected_counts[symbol] += 1
            entries_this_hour += 1

            print(
                f"{event_time} | {symbol} BUY @ {entry_price:.6f} | "
                f"Setup {candidate.score:.1f} | Selection {selection_score:.1f} | "
                f"R:R {actual_reward_risk:.2f} | Risk {actual_account_risk_percent:.2f}% | "
                f"Notional €{notional:.2f}"
            )

    final_time = common_15m[-1]
    final_prices = {
        symbol: float(frames_15m[symbol].loc[final_time, "close"])
        for symbol in SYMBOLS
    }

    if positions:
        close_all(
            final_time,
            final_prices,
            "END OF BACKTEST",
        )

    final_balance = cash_balance
    total_profit = final_balance - STARTING_CAPITAL
    roi_percent = total_profit / STARTING_CAPITAL * 100.0

    winners = [trade for trade in trade_history if float(trade["profit"]) > 0]
    losers = [trade for trade in trade_history if float(trade["profit"]) < 0]
    gross_profit = sum(float(trade["profit"]) for trade in winners)
    gross_loss = abs(sum(float(trade["profit"]) for trade in losers))
    profit_factor = (
        gross_profit / gross_loss
        if gross_loss > 0
        else float("inf") if gross_profit > 0 else 0.0
    )
    win_rate = (
        len(winners) / len(trade_history) * 100.0
        if trade_history
        else 0.0
    )
    expectancy = (
        total_profit / len(trade_history)
        if trade_history
        else 0.0
    )
    max_drawdown = maximum_drawdown(equity_history)

    # Simple benchmark from the same testing window.
    first_hour = common_1h[0]
    last_hour = common_1h[-1]
    btc_first = float(frames_1h["BTCUSDT"].loc[first_hour, "open"])
    btc_last = float(frames_1h["BTCUSDT"].loc[last_hour, "close"])
    btc_hold_return = (btc_last / btc_first - 1.0) * 100.0

    equal_weight_returns = []
    for symbol in SYMBOLS:
        first_price = float(frames_1h[symbol].loc[first_hour, "open"])
        last_close = float(frames_1h[symbol].loc[last_hour, "close"])
        equal_weight_returns.append(last_close / first_price - 1.0)
    equal_weight_return = (
        sum(equal_weight_returns) / len(equal_weight_returns) * 100.0
    )

    symbol_rows = []

    for symbol in SYMBOLS:
        symbol_trades = [
            trade
            for trade in trade_history
            if trade["symbol"] == symbol
        ]
        symbol_profit = sum(
            float(trade["profit"])
            for trade in symbol_trades
        )
        symbol_winners = sum(
            float(trade["profit"]) > 0
            for trade in symbol_trades
        )

        symbol_rows.append(
            {
                "symbol": symbol,
                "valid_candidates": int(valid_candidate_counts[symbol]),
                "selected_entries": int(selected_counts[symbol]),
                "completed_trades": len(symbol_trades),
                "winning_trades": int(symbol_winners),
                "win_rate": round(
                    symbol_winners / len(symbol_trades) * 100.0
                    if symbol_trades
                    else 0.0,
                    4,
                ),
                "profit": round(symbol_profit, 8),
            }
        )

    summary = {
        "starting_capital": STARTING_CAPITAL,
        "final_balance": round(final_balance, 8),
        "total_profit": round(total_profit, 8),
        "roi_percent": round(roi_percent, 4),
        "completed_trades": len(trade_history),
        "winning_trades": len(winners),
        "losing_trades": len(losers),
        "win_rate": round(win_rate, 4),
        "gross_profit": round(gross_profit, 8),
        "gross_loss": round(gross_loss, 8),
        "profit_factor": round(profit_factor, 4),
        "expectancy_per_trade": round(expectancy, 8),
        "maximum_drawdown": round(max_drawdown, 4),
        "total_fees": round(total_fees, 8),
        "btc_buy_hold_return_percent": round(btc_hold_return, 4),
        "equal_weight_return_percent": round(equal_weight_return, 4),
        "partial_profit_events": partial_profit_events,
        "asset_shock_reductions": asset_shock_reductions,
        "market_shock_reductions": market_shock_reductions,
        "warning_stop_tightens": warning_stop_tightens,
        "daily_profit_locks": daily_profit_locks,
        "daily_loss_stops": daily_loss_stops,
        "weekly_loss_stops": weekly_loss_stops,
        "hard_lock_hits": hard_lock_hits,
        "skipped_gap_entries": skipped_gap_entries,
        "skipped_risk_entries": skipped_risk_entries,
        "skipped_correlation_entries": skipped_correlation_entries,
        "skipped_small_entries": skipped_small_entries,
    }

    export_csv(trade_history, TRADE_HISTORY_FILE)
    export_csv(event_history, EVENT_HISTORY_FILE)
    export_csv([summary], SUMMARY_FILE)
    export_csv(symbol_rows, SYMBOL_SUMMARY_FILE)

    print()
    print("=" * 126)
    print("FA CRYPTO ENGINE — V3D3 PORTFOLIO RESULT")
    print("=" * 126)
    print(f"Starting Capital             : €{STARTING_CAPITAL:.2f}")
    print(f"Final Balance                : €{final_balance:.2f}")
    print(f"Total Profit/Loss            : €{total_profit:.2f}")
    print(f"Portfolio ROI                : {roi_percent:.2f}%")
    print(f"Completed Trades             : {len(trade_history)}")
    print(f"Winning / Losing             : {len(winners)} / {len(losers)}")
    print(f"Win Rate                     : {win_rate:.2f}%")
    print(f"Profit Factor                : {profit_factor:.2f}")
    print(f"Expectancy Per Trade         : €{expectancy:.2f}")
    print(f"Maximum Drawdown             : {max_drawdown:.2f}%")
    print(f"Total Trading Fees           : €{total_fees:.2f}")
    print(f"BTC Buy-and-Hold             : {btc_hold_return:.2f}%")
    print(f"Equal-Weight 6 Assets        : {equal_weight_return:.2f}%")
    print(f"Partial Profit Events        : {partial_profit_events}")
    print(f"Asset Shock Reductions       : {asset_shock_reductions}")
    print(f"Market Shock Reductions      : {market_shock_reductions}")
    print(f"Emergency Stop Tightens      : {warning_stop_tightens}")
    print(f"Daily / Weekly Stops         : {daily_loss_stops} / {weekly_loss_stops}")
    print(f"Hard Locks                   : {hard_lock_hits}")
    print("-" * 126)
    print(
        f"{'SYMBOL':<10} "
        f"{'VALID':>10} "
        f"{'ENTRIES':>10} "
        f"{'TRADES':>10} "
        f"{'WIN RATE':>12} "
        f"{'P&L':>14}"
    )
    print("-" * 126)

    for row in sorted(
        symbol_rows,
        key=lambda item: float(item["profit"]),
        reverse=True,
    ):
        print(
            f"{row['symbol']:<10} "
            f"{int(row['valid_candidates']):>10} "
            f"{int(row['selected_entries']):>10} "
            f"{int(row['completed_trades']):>10} "
            f"{float(row['win_rate']):>11.2f}% "
            f"€{float(row['profit']):>12.2f}"
        )

    print("-" * 126)
    print("MARKET STATE COUNTS")
    print("-" * 126)
    for state in MarketState:
        print(f"{state.value:<20}: {state_counts[state.value]:>6}")

    print("-" * 126)
    print("15M MARKET SHOCK COUNTS")
    print("-" * 126)
    for level in ShockLevel:
        print(f"{level.value:<10}: {shock_counts[level.value]:>6}")

    print("=" * 126)
    print(f"Trade history : {TRADE_HISTORY_FILE}")
    print(f"Event history : {EVENT_HISTORY_FILE}")
    print(f"Summary       : {SUMMARY_FILE}")
    print(f"Symbol report : {SYMBOL_SUMMARY_FILE}")
    print("=" * 126)

    return summary


if __name__ == "__main__":
    try:
        run_backtest()

    except KeyboardInterrupt:
        print()
        print("V3D3 portfolio backtest stopped manually.")

    except Exception as error:
        print()
        print("=" * 126)
        print("V3D3 PORTFOLIO BACKTEST ERROR")
        print("=" * 126)
        print(f"{type(error).__name__}: {error}")
        print("=" * 126)
        raise
