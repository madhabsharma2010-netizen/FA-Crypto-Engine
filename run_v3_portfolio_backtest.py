import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from binance.client import Client

from config.risk_settings import RISK_SETTINGS
from core.indicators import (
    calculate_adx,
    calculate_atr,
    calculate_ema,
    calculate_rsi,
)
from core.regime_router import (
    MarketRegime,
    evaluate_regime,
)
from market.historical_data import get_historical_candles
from strategies.ema_rsi_strategy_v2 import EmaRsiStrategyV2


# ============================================================
# V3 SHARED-PORTFOLIO SETTINGS
# ============================================================

SYMBOLS = (
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "LINKUSDT",
    "DOGEUSDT",
)

INTERVAL = Client.KLINE_INTERVAL_1HOUR
BACKTEST_START = "2026-05-01 00:00:00"
BACKTEST_END = "2026-07-20 00:00:00"

STARTING_CAPITAL = float(
    RISK_SETTINGS.starting_capital
)

MAX_PORTFOLIO_DEPLOYMENT_PERCENT = 50.0
MAX_OPEN_POSITIONS = 2

STOP_LOSS_PERCENT = 2.0
PARTIAL_TARGET_PERCENT = 5.0
PARTIAL_SELL_PERCENT = 25.0
BREAK_EVEN_TRIGGER_PERCENT = 2.0

DAILY_PROFIT_LOCK_PERCENT = 3.0
DAILY_LOSS_LIMIT_PERCENT = 2.0
WEEKLY_LOSS_LIMIT_PERCENT = 5.0
HARD_DRAWDOWN_LIMIT_PERCENT = 10.0

TRADING_FEE_PERCENT = float(
    RISK_SETTINGS.trading_fee_percent
)

SLIPPAGE_PERCENT = float(
    RISK_SETTINGS.estimated_slippage_percent
)

TRADE_HISTORY_FILE = Path(
    "logs/backtests/v3_portfolio_trade_history.csv"
)

SUMMARY_FILE = Path(
    "reports/v3_portfolio_backtest_summary.csv"
)

SYMBOL_SUMMARY_FILE = Path(
    "reports/v3_portfolio_symbol_summary.csv"
)


@dataclass
class Position:
    symbol: str
    entry_time: Any
    entry_price: float
    initial_quantity: float
    remaining_quantity: float
    entry_notional: float
    buy_fee: float
    entry_total_cost: float
    allocation_percent: float
    account_risk_percent: float
    highest_price: float
    partial_done: bool = False
    partial_time: Any = None
    partial_quantity: float = 0.0
    realized_net_proceeds: float = 0.0
    total_sell_fees: float = 0.0


def calculate_fee(
    amount: float,
) -> float:
    return (
        amount
        * TRADING_FEE_PERCENT
        / 100
    )


def apply_buy_slippage(
    market_price: float,
) -> float:
    return market_price * (
        1
        + SLIPPAGE_PERCENT
        / 100
    )


def apply_sell_slippage(
    market_price: float,
) -> float:
    return market_price * (
        1
        - SLIPPAGE_PERCENT
        / 100
    )


def prepare_data(
    symbol: str,
) -> pd.DataFrame:
    data = get_historical_candles(
        symbol=symbol,
        interval=INTERVAL,
        limit=1000,
        start_time=BACKTEST_START,
        end_time=BACKTEST_END,
        drop_incomplete=True,
    )

    if data.empty:
        raise ValueError(
            f"No historical data returned for {symbol}."
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

    data["EMA20"] = calculate_ema(
        data,
        20,
    )
    data["EMA50"] = calculate_ema(
        data,
        50,
    )
    data["EMA200"] = calculate_ema(
        data,
        200,
    )
    data["RSI14"] = calculate_rsi(
        data,
        14,
    )
    data["ADX14"] = calculate_adx(
        data,
        14,
    )
    data["ATR14"] = calculate_atr(
        data,
        14,
    )

    data["VolumeSMA20"] = (
        data["volume"]
        .rolling(20)
        .mean()
    )

    data["MOMENTUM_24H"] = (
        data["close"]
        .pct_change(24)
        * 100
    )

    data["MOMENTUM_72H"] = (
        data["close"]
        .pct_change(72)
        * 100
    )

    data["EMA50_SLOPE_12H"] = (
        data["EMA50"]
        .pct_change(12)
        * 100
    )

    data["VOLUME_RATIO"] = (
        data["volume"]
        / data["VolumeSMA20"]
    )

    data["ATR_PERCENT"] = (
        data["ATR14"]
        / data["close"]
        * 100
    )

    return (
        data
        .dropna()
        .set_index("open_time")
        .sort_index()
    )


def percentile_score(
    series: pd.Series,
) -> pd.Series:
    """
    Highest raw value receives the highest percentile.
    """

    return (
        series
        .rank(
            method="average",
            pct=True,
            ascending=True,
        )
        * 100
    )


def calculate_relative_scores(
    snapshot: pd.DataFrame,
) -> pd.DataFrame:
    ranked = snapshot.copy()

    ranked["MOM24_RANK"] = percentile_score(
        ranked["MOMENTUM_24H"]
    )

    ranked["MOM72_RANK"] = percentile_score(
        ranked["MOMENTUM_72H"]
    )

    ranked["SLOPE_RANK"] = percentile_score(
        ranked["EMA50_SLOPE_12H"]
    )

    ranked["ADX_RANK"] = percentile_score(
        ranked["ADX14"]
    )

    ranked["VOLUME_RANK"] = percentile_score(
        ranked["VOLUME_RATIO"]
    )

    ranked["VOLATILITY_PENALTY"] = 0.0

    ranked.loc[
        ranked["ATR_PERCENT"] > 4.5,
        "VOLATILITY_PENALTY",
    ] += 7.5

    ranked.loc[
        ranked["ATR_PERCENT"] > 6.0,
        "VOLATILITY_PENALTY",
    ] += 7.5

    ranked.loc[
        ranked["MOMENTUM_24H"] < 0,
        "VOLATILITY_PENALTY",
    ] += 5.0

    ranked["RELATIVE_SCORE"] = (
        ranked["REGIME_SCORE"] * 0.45
        + ranked["MOM24_RANK"] * 0.20
        + ranked["MOM72_RANK"] * 0.15
        + ranked["SLOPE_RANK"] * 0.10
        + ranked["ADX_RANK"] * 0.05
        + ranked["VOLUME_RANK"] * 0.05
        - ranked["VOLATILITY_PENALTY"]
    ).clip(
        lower=0.0,
        upper=100.0,
    )

    return (
        ranked
        .sort_values(
            [
                "RELATIVE_SCORE",
                "REGIME_SCORE",
                "MOMENTUM_24H",
                "MOMENTUM_72H",
            ],
            ascending=False,
        )
        .reset_index(drop=True)
    )


def allocate_ranked_assets(
    ranked: pd.DataFrame,
) -> dict[str, float]:
    """
    Allocate a maximum of 50% portfolio capital.

    Possible allocations:
        50% to one exceptional leader,
        30% to one normal leader,
        30% + 20% to two qualified leaders,
        0% when no asset qualifies.
    """

    eligible = ranked[
        ranked["REGIME"].isin(
            (
                MarketRegime.ACTIVE.value,
                MarketRegime.STRONG.value,
            )
        )
        & (
            ranked["REGIME_SCORE"]
            >= 65.0
        )
        & (
            ranked["RELATIVE_SCORE"]
            >= 60.0
        )
    ].copy()

    if eligible.empty:
        return {}

    leader = eligible.iloc[0]

    if len(eligible) == 1:
        allocation = (
            50.0
            if (
                float(
                    leader["RELATIVE_SCORE"]
                )
                >= 82.0
                and float(
                    leader["REGIME_SCORE"]
                )
                >= 80.0
            )
            else 30.0
        )

        return {
            str(leader["symbol"]): allocation,
        }

    second = eligible.iloc[1]

    score_gap = (
        float(
            leader["RELATIVE_SCORE"]
        )
        - float(
            second["RELATIVE_SCORE"]
        )
    )

    if (
        float(
            leader["RELATIVE_SCORE"]
        )
        >= 85.0
        and score_gap >= 15.0
    ):
        return {
            str(leader["symbol"]): 50.0,
        }

    if (
        float(
            second["RELATIVE_SCORE"]
        )
        >= 65.0
        and score_gap <= 15.0
    ):
        return {
            str(leader["symbol"]): 30.0,
            str(second["symbol"]): 20.0,
        }

    return {
        str(leader["symbol"]): 30.0,
    }


def portfolio_equity(
    cash_balance: float,
    positions: dict[str, Position],
    price_by_symbol: dict[str, float],
) -> float:
    position_value = sum(
        position.remaining_quantity
        * price_by_symbol[symbol]
        for symbol, position
        in positions.items()
    )

    return (
        cash_balance
        + position_value
    )


def portfolio_position_value(
    positions: dict[str, Position],
    price_by_symbol: dict[str, float],
) -> float:
    return sum(
        position.remaining_quantity
        * price_by_symbol[symbol]
        for symbol, position
        in positions.items()
    )


def maximum_drawdown(
    equity_history: list[float],
) -> float:
    if not equity_history:
        return 0.0

    peak = equity_history[0]
    result = 0.0

    for equity in equity_history:
        peak = max(
            peak,
            equity,
        )

        if peak <= 0:
            continue

        drawdown = (
            (peak - equity)
            / peak
            * 100
        )

        result = max(
            result,
            drawdown,
        )

    return result


def trailing_atr_multiple(
    regime: str,
    selected: bool,
    profit_lock_active: bool,
) -> float:
    """
    Strong selected winners receive more room.
    Weak/non-selected positions receive tighter protection.
    """

    if regime == MarketRegime.STRONG.value:
        multiple = (
            3.0
            if selected
            else 2.50
        )
    elif regime == MarketRegime.ACTIVE.value:
        multiple = (
            2.50
            if selected
            else 2.00
        )
    elif regime == MarketRegime.WATCH.value:
        multiple = 1.75
    else:
        multiple = 1.25

    if profit_lock_active:
        multiple = max(
            1.50,
            multiple - 0.50,
        )

    return multiple


def export_csv(
    rows: list[dict[str, Any]],
    file_path: Path,
) -> None:
    file_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        file_path.write_text(
            "",
            encoding="utf-8",
        )
        return

    with file_path.open(
        mode="w",
        newline="",
        encoding="utf-8",
    ) as output:
        writer = csv.DictWriter(
            output,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(rows)


def run_backtest() -> dict[str, Any]:
    print("=" * 118)
    print(
        "FA CRYPTO ENGINE — V3 SHARED-CAPITAL PORTFOLIO BACKTEST"
    )
    print("=" * 118)

    data_by_symbol = {
        symbol: prepare_data(symbol)
        for symbol in SYMBOLS
    }

    common_times = None

    for data in data_by_symbol.values():
        if common_times is None:
            common_times = data.index
        else:
            common_times = (
                common_times
                .intersection(data.index)
            )

    if (
        common_times is None
        or len(common_times) == 0
    ):
        raise ValueError(
            "No common candle timestamps found."
        )

    common_times = (
        common_times
        .sort_values()
    )

    cash_balance = STARTING_CAPITAL
    positions: dict[str, Position] = {}
    trade_history: list[dict[str, Any]] = []
    equity_history: list[float] = []

    total_fees = 0.0
    partial_profit_events = 0
    stop_loss_exits = 0
    trailing_exits = 0
    sleep_exits = 0
    strategy_exits = 0
    end_exits = 0
    daily_loss_stops = 0
    weekly_loss_stops = 0
    hard_lock_hits = 0
    daily_profit_locks = 0

    selected_counts = {
        symbol: 0
        for symbol in SYMBOLS
    }

    cash_only_candles = 0
    deployed_percent_sum = 0.0
    max_open_positions = 0

    peak_equity = STARTING_CAPITAL
    current_day = None
    current_week = None
    daily_start_equity = STARTING_CAPITAL
    weekly_start_equity = STARTING_CAPITAL
    daily_entry_block = False
    weekly_entry_block = False
    hard_lock = False
    profit_lock_active = False

    latest_ranked = pd.DataFrame()
    latest_allocations: dict[str, float] = {}

    def close_full_position(
        symbol: str,
        market_exit_price: float,
        close_time: Any,
        exit_reason: str,
    ) -> None:
        nonlocal cash_balance
        nonlocal total_fees
        nonlocal stop_loss_exits
        nonlocal trailing_exits
        nonlocal sleep_exits
        nonlocal strategy_exits
        nonlocal end_exits

        position = positions.get(
            symbol
        )

        if position is None:
            return

        sell_price = apply_sell_slippage(
            market_exit_price
        )

        gross_sell_value = (
            position.remaining_quantity
            * sell_price
        )

        sell_fee = calculate_fee(
            gross_sell_value
        )

        net_sell_value = (
            gross_sell_value
            - sell_fee
        )

        cash_balance += net_sell_value
        total_fees += sell_fee

        total_net_proceeds = (
            position.realized_net_proceeds
            + net_sell_value
        )

        total_sell_fees = (
            position.total_sell_fees
            + sell_fee
        )

        profit = (
            total_net_proceeds
            - position.entry_total_cost
        )

        profit_percent = (
            profit
            / position.entry_total_cost
            * 100
            if position.entry_total_cost > 0
            else 0.0
        )

        if exit_reason == "STOP LOSS":
            stop_loss_exits += 1
        elif "TRAILING" in exit_reason:
            trailing_exits += 1
        elif exit_reason == "REGIME SLEEP":
            sleep_exits += 1
        elif exit_reason == "STRATEGY SELL":
            strategy_exits += 1
        elif exit_reason == "END OF BACKTEST":
            end_exits += 1

        trade_history.append(
            {
                "trade_number": (
                    len(trade_history) + 1
                ),
                "symbol": symbol,
                "entry_time": (
                    position.entry_time
                ),
                "exit_time": close_time,
                "entry_price": round(
                    position.entry_price,
                    8,
                ),
                "final_exit_price": round(
                    sell_price,
                    8,
                ),
                "initial_quantity": round(
                    position.initial_quantity,
                    8,
                ),
                "partial_quantity": round(
                    position.partial_quantity,
                    8,
                ),
                "entry_notional": round(
                    position.entry_notional,
                    8,
                ),
                "allocation_percent": round(
                    position.allocation_percent,
                    4,
                ),
                "account_risk_percent": round(
                    position.account_risk_percent,
                    4,
                ),
                "buy_fee": round(
                    position.buy_fee,
                    8,
                ),
                "sell_fees": round(
                    total_sell_fees,
                    8,
                ),
                "partial_profit_taken": (
                    position.partial_done
                ),
                "profit": round(
                    profit,
                    8,
                ),
                "profit_percent": round(
                    profit_percent,
                    4,
                ),
                "exit_reason": exit_reason,
                "cash_after_trade": round(
                    cash_balance,
                    8,
                ),
            }
        )

        print(
            f"{close_time} | "
            f"{symbol} SELL @ "
            f"{sell_price:.6f} | "
            f"{exit_reason} | "
            f"P&L €{profit:.2f}"
        )

        del positions[symbol]

    def take_partial_profit(
        symbol: str,
        market_exit_price: float,
        close_time: Any,
    ) -> None:
        nonlocal cash_balance
        nonlocal total_fees
        nonlocal partial_profit_events

        position = positions.get(
            symbol
        )

        if (
            position is None
            or position.partial_done
        ):
            return

        partial_quantity = (
            position.initial_quantity
            * PARTIAL_SELL_PERCENT
            / 100
        )

        partial_quantity = min(
            partial_quantity,
            position.remaining_quantity,
        )

        if partial_quantity <= 0:
            return

        sell_price = apply_sell_slippage(
            market_exit_price
        )

        gross_sell_value = (
            partial_quantity
            * sell_price
        )

        sell_fee = calculate_fee(
            gross_sell_value
        )

        net_sell_value = (
            gross_sell_value
            - sell_fee
        )

        cash_balance += net_sell_value
        total_fees += sell_fee

        position.remaining_quantity -= (
            partial_quantity
        )

        position.partial_done = True
        position.partial_time = close_time
        position.partial_quantity = (
            partial_quantity
        )

        position.realized_net_proceeds += (
            net_sell_value
        )

        position.total_sell_fees += (
            sell_fee
        )

        partial_profit_events += 1

        print(
            f"{close_time} | "
            f"{symbol} PARTIAL SELL "
            f"{PARTIAL_SELL_PERCENT:.0f}% @ "
            f"{sell_price:.6f} | "
            f"Trend runner remains open"
        )

    def close_all_positions(
        candle_time: Any,
        price_by_symbol: dict[str, float],
        reason: str,
    ) -> None:
        for symbol in list(
            positions.keys()
        ):
            close_full_position(
                symbol=symbol,
                market_exit_price=(
                    price_by_symbol[symbol]
                ),
                close_time=candle_time,
                exit_reason=reason,
            )

    for candle_time in common_times:
        timestamp = pd.Timestamp(
            candle_time
        )

        day_key = timestamp.date()
        iso_calendar = (
            timestamp.isocalendar()
        )
        week_key = (
            int(iso_calendar.year),
            int(iso_calendar.week),
        )

        price_by_symbol = {
            symbol: float(
                data_by_symbol[
                    symbol
                ].loc[
                    candle_time,
                    "close",
                ]
            )
            for symbol in SYMBOLS
        }

        if current_day != day_key:
            current_day = day_key

            current_equity = portfolio_equity(
                cash_balance,
                positions,
                price_by_symbol,
            )

            daily_start_equity = (
                current_equity
            )

            daily_entry_block = False
            profit_lock_active = False

        if current_week != week_key:
            current_week = week_key

            current_equity = portfolio_equity(
                cash_balance,
                positions,
                price_by_symbol,
            )

            weekly_start_equity = (
                current_equity
            )

            weekly_entry_block = False

        snapshot_rows: list[
            dict[str, object]
        ] = []

        decision_by_symbol = {}

        for symbol in SYMBOLS:
            candle = data_by_symbol[
                symbol
            ].loc[candle_time]

            decision = evaluate_regime(
                symbol,
                candle,
            )

            decision_by_symbol[
                symbol
            ] = decision

            snapshot_rows.append(
                {
                    "symbol": symbol,
                    "REGIME": (
                        decision.regime.value
                    ),
                    "REGIME_SCORE": (
                        decision.score
                    ),
                    "MOMENTUM_24H": float(
                        candle[
                            "MOMENTUM_24H"
                        ]
                    ),
                    "MOMENTUM_72H": float(
                        candle[
                            "MOMENTUM_72H"
                        ]
                    ),
                    "EMA50_SLOPE_12H": float(
                        candle[
                            "EMA50_SLOPE_12H"
                        ]
                    ),
                    "ADX14": float(
                        candle["ADX14"]
                    ),
                    "VOLUME_RATIO": float(
                        candle[
                            "VOLUME_RATIO"
                        ]
                    ),
                    "ATR_PERCENT": float(
                        candle[
                            "ATR_PERCENT"
                        ]
                    ),
                }
            )

        ranked = calculate_relative_scores(
            pd.DataFrame(
                snapshot_rows
            )
        )

        allocations = allocate_ranked_assets(
            ranked
        )

        selected_symbols = set(
            allocations
        )

        for symbol in selected_symbols:
            selected_counts[
                symbol
            ] += 1

        latest_ranked = ranked.copy()
        latest_allocations = (
            allocations.copy()
        )

        current_equity = portfolio_equity(
            cash_balance,
            positions,
            price_by_symbol,
        )

        peak_equity = max(
            peak_equity,
            current_equity,
        )

        current_drawdown = (
            (peak_equity - current_equity)
            / peak_equity
            * 100
            if peak_equity > 0
            else 0.0
        )

        daily_return = (
            (current_equity - daily_start_equity)
            / daily_start_equity
            * 100
            if daily_start_equity > 0
            else 0.0
        )

        weekly_return = (
            (current_equity - weekly_start_equity)
            / weekly_start_equity
            * 100
            if weekly_start_equity > 0
            else 0.0
        )

        if (
            not hard_lock
            and current_drawdown
            <= -HARD_DRAWDOWN_LIMIT_PERCENT
        ):
            # This branch is retained for clarity;
            # current_drawdown is non-negative.
            pass

        if (
            not hard_lock
            and current_drawdown
            >= HARD_DRAWDOWN_LIMIT_PERCENT
        ):
            hard_lock = True
            hard_lock_hits += 1

            print(
                f"{candle_time} | "
                f"HARD LOCK | "
                f"Drawdown {current_drawdown:.2f}%"
            )

            close_all_positions(
                candle_time,
                price_by_symbol,
                "HARD DRAWDOWN LOCK",
            )

        if (
            not hard_lock
            and not weekly_entry_block
            and weekly_return
            <= -WEEKLY_LOSS_LIMIT_PERCENT
        ):
            weekly_entry_block = True
            weekly_loss_stops += 1

            print(
                f"{candle_time} | "
                f"WEEKLY STOP | "
                f"Return {weekly_return:.2f}%"
            )

            close_all_positions(
                candle_time,
                price_by_symbol,
                "WEEKLY LOSS LIMIT",
            )

        if (
            not hard_lock
            and not daily_entry_block
            and daily_return
            <= -DAILY_LOSS_LIMIT_PERCENT
        ):
            daily_entry_block = True
            daily_loss_stops += 1

            print(
                f"{candle_time} | "
                f"DAILY STOP | "
                f"Return {daily_return:.2f}%"
            )

            close_all_positions(
                candle_time,
                price_by_symbol,
                "DAILY LOSS LIMIT",
            )

        if (
            not profit_lock_active
            and daily_return
            >= DAILY_PROFIT_LOCK_PERCENT
        ):
            profit_lock_active = True
            daily_profit_locks += 1

            print(
                f"{candle_time} | "
                f"DAILY PROFIT LOCK | "
                f"Return {daily_return:.2f}% | "
                f"Existing winners continue"
            )

        # ----------------------------------------------------
        # MANAGE OPEN POSITIONS
        # ----------------------------------------------------

        for symbol in list(
            positions.keys()
        ):
            position = positions[
                symbol
            ]

            candle = data_by_symbol[
                symbol
            ].loc[candle_time]

            close_price = float(
                candle["close"]
            )
            candle_high = float(
                candle["high"]
            )
            candle_low = float(
                candle["low"]
            )
            atr = float(
                candle["ATR14"]
            )

            position.highest_price = max(
                position.highest_price,
                candle_high,
            )

            hard_stop = (
                position.entry_price
                * (
                    1
                    - STOP_LOSS_PERCENT
                    / 100
                )
            )

            if candle_low <= hard_stop:
                close_full_position(
                    symbol,
                    hard_stop,
                    candle_time,
                    "STOP LOSS",
                )
                continue

            partial_target = (
                position.entry_price
                * (
                    1
                    + PARTIAL_TARGET_PERCENT
                    / 100
                )
            )

            partial_completed_now = False

            if (
                not position.partial_done
                and candle_high
                >= partial_target
            ):
                take_partial_profit(
                    symbol,
                    partial_target,
                    candle_time,
                )

                partial_completed_now = True

            decision = decision_by_symbol[
                symbol
            ]

            current_regime = (
                decision.regime.value
            )

            selected = (
                symbol
                in selected_symbols
            )

            strategy_signal = (
                EmaRsiStrategyV2
                .generate_signal(candle)
            )

            if (
                current_regime
                == MarketRegime.SLEEP.value
            ):
                close_full_position(
                    symbol,
                    close_price,
                    candle_time,
                    "REGIME SLEEP",
                )
                continue

            if strategy_signal == "SELL":
                close_full_position(
                    symbol,
                    close_price,
                    candle_time,
                    "STRATEGY SELL",
                )
                continue

            break_even_active = (
                position.partial_done
                or close_price
                >= (
                    position.entry_price
                    * (
                        1
                        + BREAK_EVEN_TRIGGER_PERCENT
                        / 100
                    )
                )
            )

            if (
                break_even_active
                and not partial_completed_now
            ):
                atr_multiple = (
                    trailing_atr_multiple(
                        current_regime,
                        selected,
                        profit_lock_active,
                    )
                )

                trailing_stop = (
                    position.highest_price
                    - atr
                    * atr_multiple
                )

                break_even_floor = (
                    position.entry_price
                    * 1.001
                )

                trailing_stop = max(
                    trailing_stop,
                    break_even_floor,
                )

                if candle_low <= trailing_stop:
                    close_full_position(
                        symbol,
                        trailing_stop,
                        candle_time,
                        (
                            "DYNAMIC ATR TRAILING"
                        ),
                    )
                    continue

        # ----------------------------------------------------
        # OPEN NEW POSITIONS
        # ----------------------------------------------------

        block_new_entries = (
            hard_lock
            or daily_entry_block
            or weekly_entry_block
            or profit_lock_active
        )

        if not block_new_entries:
            for symbol, allocation_percent in (
                allocations.items()
            ):
                if (
                    symbol in positions
                    or len(positions)
                    >= MAX_OPEN_POSITIONS
                ):
                    continue

                candle = data_by_symbol[
                    symbol
                ].loc[candle_time]

                signal = (
                    EmaRsiStrategyV2
                    .generate_signal(candle)
                )

                if signal != "BUY":
                    continue

                close_price = float(
                    candle["close"]
                )

                current_equity = (
                    portfolio_equity(
                        cash_balance,
                        positions,
                        price_by_symbol,
                    )
                )

                current_position_value = (
                    portfolio_position_value(
                        positions,
                        price_by_symbol,
                    )
                )

                max_deployed_value = (
                    current_equity
                    * MAX_PORTFOLIO_DEPLOYMENT_PERCENT
                    / 100
                )

                deployment_room = max(
                    0.0,
                    max_deployed_value
                    - current_position_value,
                )

                target_notional = (
                    current_equity
                    * allocation_percent
                    / 100
                )

                affordable_notional = (
                    cash_balance
                    / (
                        1
                        + TRADING_FEE_PERCENT
                        / 100
                    )
                )

                notional = min(
                    target_notional,
                    deployment_room,
                    affordable_notional,
                )

                if notional <= 0:
                    continue

                buy_price = apply_buy_slippage(
                    close_price
                )

                quantity = (
                    notional
                    / buy_price
                )

                buy_fee = calculate_fee(
                    notional
                )

                total_required = (
                    notional
                    + buy_fee
                )

                if (
                    quantity <= 0
                    or total_required
                    > cash_balance
                ):
                    continue

                cash_balance -= (
                    total_required
                )

                total_fees += buy_fee

                account_risk_percent = (
                    notional
                    / current_equity
                    * STOP_LOSS_PERCENT
                )

                positions[symbol] = Position(
                    symbol=symbol,
                    entry_time=candle_time,
                    entry_price=buy_price,
                    initial_quantity=quantity,
                    remaining_quantity=quantity,
                    entry_notional=notional,
                    buy_fee=buy_fee,
                    entry_total_cost=(
                        total_required
                    ),
                    allocation_percent=(
                        allocation_percent
                    ),
                    account_risk_percent=(
                        account_risk_percent
                    ),
                    highest_price=buy_price,
                )

                print(
                    f"{candle_time} | "
                    f"{symbol} BUY @ "
                    f"{buy_price:.6f} | "
                    f"Allocation "
                    f"{allocation_percent:.0f}% | "
                    f"Notional €{notional:.2f} | "
                    f"Account Risk "
                    f"{account_risk_percent:.2f}%"
                )

        end_equity = portfolio_equity(
            cash_balance,
            positions,
            price_by_symbol,
        )

        equity_history.append(
            end_equity
        )

        end_position_value = (
            portfolio_position_value(
                positions,
                price_by_symbol,
            )
        )

        deployed_percent = (
            end_position_value
            / end_equity
            * 100
            if end_equity > 0
            else 0.0
        )

        deployed_percent_sum += (
            deployed_percent
        )

        if not positions:
            cash_only_candles += 1

        max_open_positions = max(
            max_open_positions,
            len(positions),
        )

        if hard_lock:
            break

    final_time = common_times[-1]

    final_prices = {
        symbol: float(
            data_by_symbol[
                symbol
            ].loc[
                final_time,
                "close",
            ]
        )
        for symbol in SYMBOLS
    }

    for symbol in list(
        positions.keys()
    ):
        close_full_position(
            symbol,
            final_prices[symbol],
            final_time,
            "END OF BACKTEST",
        )

    final_balance = cash_balance
    total_profit = (
        final_balance
        - STARTING_CAPITAL
    )

    roi_percent = (
        total_profit
        / STARTING_CAPITAL
        * 100
    )

    completed_trades = len(
        trade_history
    )

    winners = [
        trade
        for trade in trade_history
        if float(trade["profit"]) > 0
    ]

    losers = [
        trade
        for trade in trade_history
        if float(trade["profit"]) < 0
    ]

    gross_profit = sum(
        float(trade["profit"])
        for trade in winners
    )

    gross_loss = abs(
        sum(
            float(trade["profit"])
            for trade in losers
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

    win_rate = (
        len(winners)
        / completed_trades
        * 100
        if completed_trades
        else 0.0
    )

    expectancy = (
        total_profit
        / completed_trades
        if completed_trades
        else 0.0
    )

    max_drawdown = maximum_drawdown(
        equity_history
    )

    average_deployed_percent = (
        deployed_percent_sum
        / len(equity_history)
        if equity_history
        else 0.0
    )

    symbol_rows: list[
        dict[str, Any]
    ] = []

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
            1
            for trade in symbol_trades
            if float(trade["profit"]) > 0
        )

        symbol_rows.append(
            {
                "symbol": symbol,
                "selected_candles": (
                    selected_counts[symbol]
                ),
                "completed_trades": len(
                    symbol_trades
                ),
                "winning_trades": (
                    symbol_winners
                ),
                "win_rate": round(
                    (
                        symbol_winners
                        / len(symbol_trades)
                        * 100
                    )
                    if symbol_trades
                    else 0.0,
                    4,
                ),
                "profit": round(
                    symbol_profit,
                    8,
                ),
            }
        )

    summary_row = {
        "starting_capital": (
            STARTING_CAPITAL
        ),
        "final_balance": round(
            final_balance,
            8,
        ),
        "total_profit": round(
            total_profit,
            8,
        ),
        "roi_percent": round(
            roi_percent,
            4,
        ),
        "completed_trades": (
            completed_trades
        ),
        "winning_trades": len(
            winners
        ),
        "losing_trades": len(
            losers
        ),
        "win_rate": round(
            win_rate,
            4,
        ),
        "gross_profit": round(
            gross_profit,
            8,
        ),
        "gross_loss": round(
            gross_loss,
            8,
        ),
        "profit_factor": round(
            profit_factor,
            4,
        ),
        "expectancy_per_trade": round(
            expectancy,
            8,
        ),
        "maximum_drawdown": round(
            max_drawdown,
            4,
        ),
        "total_fees": round(
            total_fees,
            8,
        ),
        "average_capital_deployed_percent": round(
            average_deployed_percent,
            4,
        ),
        "cash_only_candles": (
            cash_only_candles
        ),
        "cash_only_percent": round(
            (
                cash_only_candles
                / len(equity_history)
                * 100
            )
            if equity_history
            else 0.0,
            4,
        ),
        "maximum_open_positions": (
            max_open_positions
        ),
        "partial_profit_events": (
            partial_profit_events
        ),
        "stop_loss_exits": (
            stop_loss_exits
        ),
        "dynamic_trailing_exits": (
            trailing_exits
        ),
        "regime_sleep_exits": (
            sleep_exits
        ),
        "strategy_sell_exits": (
            strategy_exits
        ),
        "end_of_backtest_exits": (
            end_exits
        ),
        "daily_profit_locks": (
            daily_profit_locks
        ),
        "daily_loss_stops": (
            daily_loss_stops
        ),
        "weekly_loss_stops": (
            weekly_loss_stops
        ),
        "hard_lock_hits": (
            hard_lock_hits
        ),
    }

    export_csv(
        trade_history,
        TRADE_HISTORY_FILE,
    )

    export_csv(
        [summary_row],
        SUMMARY_FILE,
    )

    export_csv(
        symbol_rows,
        SYMBOL_SUMMARY_FILE,
    )

    print()
    print("=" * 118)
    print(
        "FA CRYPTO ENGINE — V3 PORTFOLIO RESULT"
    )
    print("=" * 118)

    print(
        f"Starting Capital             : "
        f"€{STARTING_CAPITAL:.2f}"
    )
    print(
        f"Final Balance                : "
        f"€{final_balance:.2f}"
    )
    print(
        f"Total Profit/Loss            : "
        f"€{total_profit:.2f}"
    )
    print(
        f"Portfolio ROI                : "
        f"{roi_percent:.2f}%"
    )
    print(
        f"Completed Trades             : "
        f"{completed_trades}"
    )
    print(
        f"Winning / Losing             : "
        f"{len(winners)} / {len(losers)}"
    )
    print(
        f"Win Rate                     : "
        f"{win_rate:.2f}%"
    )
    print(
        f"Profit Factor                : "
        f"{profit_factor:.2f}"
    )
    print(
        f"Expectancy Per Trade         : "
        f"€{expectancy:.2f}"
    )
    print(
        f"Maximum Drawdown             : "
        f"{max_drawdown:.2f}%"
    )
    print(
        f"Total Trading Fees           : "
        f"€{total_fees:.2f}"
    )
    print(
        f"Average Capital Deployed     : "
        f"{average_deployed_percent:.2f}%"
    )
    print(
        f"Cash/Sleep Time              : "
        f"{summary_row['cash_only_percent']:.2f}%"
    )
    print(
        f"Partial Profit Events        : "
        f"{partial_profit_events}"
    )
    print(
        f"Daily Profit Locks           : "
        f"{daily_profit_locks}"
    )
    print(
        f"Daily / Weekly Loss Stops    : "
        f"{daily_loss_stops} / "
        f"{weekly_loss_stops}"
    )
    print(
        f"Hard Locks                   : "
        f"{hard_lock_hits}"
    )

    print("-" * 118)
    print(
        f"{'SYMBOL':<10} "
        f"{'SELECTED':>12} "
        f"{'TRADES':>10} "
        f"{'WIN RATE':>12} "
        f"{'P&L':>14}"
    )
    print("-" * 118)

    for row in sorted(
        symbol_rows,
        key=lambda item: float(
            item["profit"]
        ),
        reverse=True,
    ):
        print(
            f"{row['symbol']:<10} "
            f"{int(row['selected_candles']):>12} "
            f"{int(row['completed_trades']):>10} "
            f"{float(row['win_rate']):>11.2f}% "
            f"€{float(row['profit']):>12.2f}"
        )

    print("-" * 118)
    print(
        "LATEST RANKING"
    )
    print("-" * 118)

    for rank, row in enumerate(
        latest_ranked.to_dict(
            orient="records"
        ),
        start=1,
    ):
        symbol = str(
            row["symbol"]
        )

        allocation = (
            latest_allocations.get(
                symbol,
                0.0,
            )
        )

        print(
            f"{rank}. {symbol:<10} | "
            f"Relative "
            f"{float(row['RELATIVE_SCORE']):>6.2f} | "
            f"Regime "
            f"{float(row['REGIME_SCORE']):>6.2f} | "
            f"Allocation "
            f"{allocation:>5.1f}%"
        )

    print("=" * 118)
    print(
        f"Trade history : {TRADE_HISTORY_FILE}"
    )
    print(
        f"Summary       : {SUMMARY_FILE}"
    )
    print(
        f"Symbol report : {SYMBOL_SUMMARY_FILE}"
    )
    print("=" * 118)

    return summary_row


if __name__ == "__main__":
    try:
        run_backtest()

    except KeyboardInterrupt:
        print()
        print(
            "V3 portfolio backtest stopped manually."
        )

    except Exception as error:
        print()
        print("=" * 118)
        print(
            "V3 PORTFOLIO BACKTEST ERROR"
        )
        print("=" * 118)
        print(
            f"{type(error).__name__}: "
            f"{error}"
        )
        print("=" * 118)
        raise
