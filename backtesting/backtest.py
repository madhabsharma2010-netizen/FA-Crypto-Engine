import csv
from pathlib import Path
from typing import Optional

from binance.client import Client

from config.risk_settings import RISK_SETTINGS
from core.indicators import (
    calculate_ema,
    calculate_rsi,
    calculate_adx,
    calculate_atr,
)
from core.risk_manager import RiskManager, TradingMode
from market.historical_data import get_historical_candles
from strategies.ema_rsi_strategy_v2 import EmaRsiStrategyV2 



# ============================================================
# BACKTEST SETTINGS
# ============================================================

SYMBOL = "BTCUSDT"

INTERVAL = Client.KLINE_INTERVAL_1HOUR

CANDLE_LIMIT = 1000

CSV_FILE = "logs/backtest_trade_history.csv"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def calculate_fee(
    amount: float,
) -> float:

    return (
        amount
        * RISK_SETTINGS.trading_fee_percent
        / 100
    )


def apply_buy_slippage(
    market_price: float,
) -> float:

    return (
        market_price
        * (
            1
            + RISK_SETTINGS.estimated_slippage_percent
            / 100
        )
    )


def apply_sell_slippage(
    market_price: float,
) -> float:

    return (
        market_price
        * (
            1
            - RISK_SETTINGS.estimated_slippage_percent
            / 100
        )
    )


def calculate_equity(
    cash_balance: float,
    position_open: bool,
    quantity: float,
    market_price: float,
) -> float:

    if not position_open:
        return cash_balance

    return (
        cash_balance
        + quantity * market_price
    )


def calculate_max_drawdown(
    equity_history: list[float],
) -> float:

    if not equity_history:
        return 0.0

    peak_equity = equity_history[0]

    maximum_drawdown = 0.0

    for equity in equity_history:

        if equity > peak_equity:
            peak_equity = equity

        if peak_equity <= 0:
            continue

        drawdown = (
            (peak_equity - equity)
            / peak_equity
        ) * 100

        maximum_drawdown = max(
            maximum_drawdown,
            drawdown,
        )

    return maximum_drawdown


def export_trade_history(
    trade_history: list[dict],
    csv_file: str,
) -> None:

    if not trade_history:
        print("No completed trades available.")
        return

    file_path = Path(csv_file)


    file_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "trade_number",
        "trade_type",
        "buy_time",
        "sell_time",
        "buy_price",
        "sell_price",
        "quantity",
        "notional",
        "risk_amount",
        "risk_percent",
        "buy_fee",
        "sell_fee",
        "profit",
        "profit_percent",
        "exit_reason",
        "balance_after_trade",
    ]

    with file_path.open(
        mode="w",
        newline="",
        encoding="utf-8",
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            trade_history
        )

    print(
        f"Trade history saved to: {file_path}"
    )


# ============================================================
# BACKTEST ENGINE
# ============================================================

def run_backtest(
    symbol: str = SYMBOL,
    interval: str = INTERVAL,
    candle_limit: int = CANDLE_LIMIT,
    csv_file: str = CSV_FILE,
) -> None:

    settings = RISK_SETTINGS

    settings.validate()

    print("=" * 95)
    print("FA CRYPTO ENGINE — CAPITAL PRESERVATION BACKTEST")
    print("=" * 95)

    print(
        f"Master Control             : "
        f"{settings.engine_control}"
    )

    if settings.engine_control == "STOP":
        print("Engine manually stopped.")
        return

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

# --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    data = get_historical_candles(
        symbol=symbol,
        interval=interval,
        limit=candle_limit,
    )

    if data.empty:
        raise ValueError("Historical data is empty.")

    required_columns = {
        "open_time",
        "high",
        "low",
        "close",
        "volume",
    }

    missing_columns = required_columns - set(data.columns)

    if missing_columns:
        raise ValueError(
            f"Historical data missing columns: "
            f"{sorted(missing_columns)}"
        )

    numeric_columns = [
        "high",
        "low",
        "close",
        "volume",
    ]

    for column in numeric_columns:
        data[column] = data[column].astype(float)

    # --------------------------------------------------------
    # INDICATORS
    # --------------------------------------------------------

    data["EMA20"] = calculate_ema(data, 20)
    data["EMA50"] = calculate_ema(data, 50)
    data["EMA200"] = calculate_ema(data, 200)
    data["RSI14"] = calculate_rsi(data, 14)
    data["ADX14"] = calculate_adx(data, 14)
    data["ATR14"] = calculate_atr(data, 14)

    data["VOLUME_AVG"] = (
        data["volume"]
        .rolling(settings.volume_average_period)
        .mean()
    )

    data["VolumeSMA20"] = data["VOLUME_AVG"]

    data = data.dropna().reset_index(drop=True)

    if len(data) < 2:
        raise ValueError("Not enough candles after indicator calculation.")

    # --------------------------------------------------------
    # PORTFOLIO STATE
    # --------------------------------------------------------

    cash_balance = float(settings.starting_capital)
    position_open = False
    quantity = 0.0
    entry_price = 0.0
    entry_time = None
    entry_notional = 0.0
    entry_total_cost = 0.0
    entry_buy_fee = 0.0
    entry_risk_amount = 0.0
    entry_risk_percent = 0.0
    highest_price = 0.0
    break_even_active = False
    recovery_trade_open = False

    # --------------------------------------------------------
    # PERFORMANCE STATE
    # --------------------------------------------------------

    trade_history: list[dict] = []
    equity_history: list[float] = []
    winning_trades = 0
    losing_trades = 0
    daily_profit_lock_hits = 0
    daily_loss_hits = 0
    weekly_stop_hits = 0
    hard_lock_hits = 0
    recovery_setup_detections = 0
    recovery_trades = 0

    risk_manager = RiskManager(
        settings=settings,
        starting_equity=settings.starting_capital,
    )

    # --------------------------------------------------------
    # POSITION CLOSING FUNCTION
    # --------------------------------------------------------

    def close_position(
        market_exit_price: float,
        close_time,
        exit_reason: str,
    ) -> None:
        nonlocal cash_balance
        nonlocal position_open
        nonlocal quantity
        nonlocal entry_price
        nonlocal entry_time
        nonlocal entry_notional
        nonlocal entry_total_cost
        nonlocal entry_buy_fee
        nonlocal entry_risk_amount
        nonlocal entry_risk_percent
        nonlocal highest_price
        nonlocal break_even_active
        nonlocal recovery_trade_open
        nonlocal winning_trades
        nonlocal losing_trades

        if not position_open:
            return

        sell_price = apply_sell_slippage(market_exit_price)

        gross_sell_value = quantity * sell_price

        sell_fee = calculate_fee(gross_sell_value)

        net_sell_value = gross_sell_value - sell_fee

        cash_balance += net_sell_value

        profit = net_sell_value - entry_total_cost

        if entry_total_cost > 0:
            profit_percent = (profit / entry_total_cost) * 100
        else:
            profit_percent = 0.0

        if profit > 0:
            winning_trades += 1
        else:
            losing_trades += 1

        trade_type = "RECOVERY" if recovery_trade_open else "NORMAL"

        trade_history.append(
            {
                "trade_number": len(trade_history) + 1,
                "trade_type": trade_type,
                "buy_time": entry_time,
                "sell_time": close_time,
                "buy_price": round(entry_price, 8),
                "sell_price": round(sell_price, 8),
                "quantity": round(quantity, 8),
                "notional": round(entry_notional, 8),
                "risk_amount": round(entry_risk_amount, 8),
                "risk_percent": round(entry_risk_percent, 4),
                "buy_fee": round(entry_buy_fee, 8),
                "sell_fee": round(sell_fee, 8),
                "profit": round(profit, 8),
                "profit_percent": round(profit_percent, 4),
                "exit_reason": exit_reason,
                "balance_after_trade": round(cash_balance, 8),
            }
        )

        print(
            f"{close_time} | "
            f"SELL @ {sell_price:.2f} | "
            f"{trade_type} | "
            f"{exit_reason} | "
            f"P&L €{profit:.2f}"
        )

        position_open = False
        quantity = 0.0
        entry_price = 0.0
        entry_time = None
        entry_notional = 0.0
        entry_total_cost = 0.0
        entry_buy_fee = 0.0
        entry_risk_amount = 0.0
        entry_risk_percent = 0.0
        highest_price = 0.0
        break_even_active = False
        recovery_trade_open = False

        risk_manager.register_trade_exit()

    # --------------------------------------------------------
    # MAIN LOOP
    # --------------------------------------------------------

    for index in range(len(data)):
        candle = data.iloc[index]
        candle_time = candle["open_time"]
        close_price = float(candle["close"])
        candle_high = float(candle["high"])
        candle_low = float(candle["low"])

        current_equity = calculate_equity(
            cash_balance=cash_balance,
            position_open=position_open,
            quantity=quantity,
            market_price=close_price,
        )

        risk_manager.update_period(
            candle_time=candle_time,
            current_equity=current_equity,
        )

        equity_history.append(current_equity)

        risk_manager.process_candle_cooldown()

        # ----------------------------------------------------
        # PROFIT EXTENSION TRAILING CHECK
        # ----------------------------------------------------

        if position_open and risk_manager.mode == TradingMode.PROFIT_LOCK:
            extension_exit = risk_manager.update_profit_extension(current_equity)

            if extension_exit is not None:
                close_position(
                    market_exit_price=close_price,
                    close_time=candle_time,
                    exit_reason=extension_exit,
                )
                continue

        # ----------------------------------------------------
        # ACCOUNT LIMIT CHECK
        # ----------------------------------------------------

        previous_mode = risk_manager.mode

        account_reason = risk_manager.evaluate_account_limits(
            current_equity=current_equity,
            position_open=position_open,
        )

        if previous_mode != risk_manager.mode:
            print(
                f"{candle_time} | "
                f"MODE: {risk_manager.mode.value} | "
                f"{risk_manager.mode_reason}"
            )

            if risk_manager.mode == TradingMode.PROFIT_LOCK:
                daily_profit_lock_hits += 1
            elif risk_manager.mode == TradingMode.RECOVERY_WATCH:
                daily_loss_hits += 1
            elif risk_manager.mode == TradingMode.WEEKLY_STOP:
                weekly_stop_hits += 1
            elif risk_manager.mode == TradingMode.HARD_LOCK:
                hard_lock_hits += 1

        # ----------------------------------------------------
        # FORCED ACCOUNT EXIT
        # ----------------------------------------------------

        if risk_manager.mode in {
            TradingMode.WEEKLY_STOP,
            TradingMode.HARD_LOCK,
            TradingMode.DAILY_STOP,
            TradingMode.MANUAL_STOP,
            TradingMode.RECOVERY_WATCH,
        }:
            if position_open:
                close_position(
                    market_exit_price=close_price,
                    close_time=candle_time,
                    exit_reason=risk_manager.mode_reason,
                )

            if risk_manager.mode == TradingMode.HARD_LOCK:
                break

        if risk_manager.mode == TradingMode.PROFIT_LOCK:
            continue

        # ----------------------------------------------------
        # RECOVERY WATCH
        # ----------------------------------------------------

        if risk_manager.mode == TradingMode.RECOVERY_WATCH:
            recovery_score = risk_manager.update_recovery_watch(candle)

            if recovery_score >= settings.recovery_score_required:
                recovery_setup_detections += 1

                print(
                    f"{candle_time} | "
                    f"RECOVERY SCORE "
                    f"{recovery_score:.0f}/100 | "
                    f"Confirmations "
                    f"{risk_manager.recovery_confirmation_count}/"
                    f"{settings.recovery_confirmation_candles}"
                )

            if not risk_manager.can_open_recovery_trade():
                continue

            signal = EmaRsiStrategyV2.generate_signal(candle)

            if signal != "BUY":
                continue

            buy_price = apply_buy_slippage(close_price)

            plan = risk_manager.calculate_position_plan(
                current_equity=current_equity,
                available_cash=cash_balance,
                entry_price=buy_price,
                recovery_trade=True,
            )

            buy_fee = calculate_fee(plan.notional)

            total_required = plan.notional + buy_fee

            if total_required > cash_balance:
                adjusted_notional = (
                    cash_balance
                    / (
                        1
                        + settings.trading_fee_percent
                        / 100
                    )
                )

                plan.quantity = adjusted_notional / buy_price
                plan.notional = adjusted_notional

                buy_fee = calculate_fee(adjusted_notional)

                total_required = adjusted_notional + buy_fee

            if plan.quantity <= 0:
                continue

            cash_balance -= total_required

            position_open = True
            quantity = plan.quantity
            entry_price = buy_price
            entry_time = candle_time
            highest_price = entry_price
            break_even_active = False

            entry_notional = plan.notional
            entry_buy_fee = buy_fee
            entry_total_cost = total_required
            entry_risk_amount = plan.risk_amount
            entry_risk_percent = plan.risk_percent_used

            recovery_trade_open = True
            recovery_trades += 1

            risk_manager.register_trade_entry(recovery_trade=True)

            print(
                f"{candle_time} | "
                f"RECOVERY BUY @ {entry_price:.2f} | "
                f"Notional €{entry_notional:.2f} | "
                f"Risk €{entry_risk_amount:.2f}"
            )

            continue

        # ----------------------------------------------------
        # OPEN POSITION EXIT
        # ----------------------------------------------------

        if position_open:
            highest_price = max(highest_price, candle_high)

            price_exit = risk_manager.check_position_exit(
                entry_price=entry_price,
                candle_low=candle_low,
                candle_high=candle_high,

            )

            atr = float(candle["ATR14"])

            if close_price >= entry_price * 1.01:
                break_even_active = True

            trailing_stop = highest_price - (atr * 2)

            if break_even_active and close_price < trailing_stop:
                close_position(
                    market_exit_price=close_price,
                    close_time=candle_time,
                    exit_reason="ATR TRAILING",
                )
                continue

            if price_exit is not None:
                exit_reason, exit_price = price_exit
                close_position(
                    market_exit_price=exit_price,
                    close_time=candle_time,
                    exit_reason=exit_reason,
                )
                continue

            strategy_signal = EmaRsiStrategyV2.generate_signal(candle)

            if strategy_signal == "SELL":
                close_position(
                    market_exit_price=close_price,
                    close_time=candle_time,
                    exit_reason="STRATEGY SELL",
                )
                continue

        # ----------------------------------------------------
        # NORMAL ENTRY
        # ----------------------------------------------------

        if position_open:
            continue

        if not risk_manager.can_open_normal_trade(candle):
            continue

        signal = EmaRsiStrategyV2.generate_signal(candle)

        if signal != "BUY":
            continue

        buy_price = apply_buy_slippage(close_price)

        current_equity = calculate_equity(
            cash_balance=cash_balance,
            position_open=False,
            quantity=0.0,
            market_price=close_price,
        )

        plan = risk_manager.calculate_position_plan(
            current_equity=current_equity,
            available_cash=cash_balance,
            entry_price=buy_price,
            recovery_trade=False,
        )

        buy_fee = calculate_fee(plan.notional)

        total_required = plan.notional + buy_fee

        if total_required > cash_balance:
            adjusted_notional = (
                cash_balance
                / (
                    1
                    + settings.trading_fee_percent
                    / 100
                )
            )

            plan.quantity = adjusted_notional / buy_price
            plan.notional = adjusted_notional

            buy_fee = calculate_fee(adjusted_notional)
            total_required = adjusted_notional + buy_fee

        if plan.quantity <= 0:
            continue

        cash_balance -= total_required

        position_open = True
        quantity = plan.quantity
        entry_price = buy_price
        entry_time = candle_time
        highest_price = entry_price
        break_even_active = False

        entry_notional = plan.notional
        entry_buy_fee = buy_fee
        entry_total_cost = total_required
        entry_risk_amount = plan.risk_amount
        entry_risk_percent = plan.risk_percent_used

        recovery_trade_open = False

        risk_manager.register_trade_entry(recovery_trade=False)

        print(
            f"{candle_time} | "
            f"BUY @ {entry_price:.2f} | "
            f"Notional €{entry_notional:.2f} | "
            f"Risk €{entry_risk_amount:.2f} | "
            f"Capital Used "
            f"{entry_notional / current_equity * 100:.2f}%"
        )

    # --------------------------------------------------------
    # CLOSE FINAL POSITION
    # --------------------------------------------------------

    final_candle = data.iloc[-1]

    final_price = float(final_candle["close"])
    final_time = final_candle["open_time"]

    if position_open:
        close_position(
            market_exit_price=final_price,
            close_time=final_time,
            exit_reason="END OF BACKTEST",
        )

    # --------------------------------------------------------
    # FINAL REPORT
    # --------------------------------------------------------

    final_balance = cash_balance
    total_profit = final_balance - settings.starting_capital
    roi_percent = (total_profit / settings.starting_capital) * 100
    completed_trades = len(trade_history)

    if completed_trades > 0:
        win_rate = (winning_trades / completed_trades) * 100
    else:
        win_rate = 0.0

    maximum_drawdown = calculate_max_drawdown(equity_history)

    status = risk_manager.status_report(final_balance)

    print()
    print("=" * 95)
    print("FA CRYPTO ENGINE — FINAL BACKTEST REPORT")
    print("=" * 95)

    print(
        f"Starting Capital          : "
        f"€{settings.starting_capital:.2f}"
    )
    print(
        f"Final Balance             : "
        f"€{final_balance:.2f}"
    )
    print(
        f"Total Profit/Loss         : "
        f"€{total_profit:.2f}"
    )
    print(
        f"ROI                       : "
        f"{roi_percent:.2f}%"
    )
    print(
        f"Completed Trades          : "
        f"{completed_trades}"
    )
    print(
        f"Winning Trades            : "
        f"{winning_trades}"
    )
    print(
        f"Losing Trades             : "
        f"{losing_trades}"
    )
    print(
        f"Win Rate                  : "
        f"{win_rate:.2f}%"
    )
    print(
        f"Maximum Drawdown          : "
        f"{maximum_drawdown:.2f}%"
    )
    print(
        f"Daily Profit Locks        : "
        f"{daily_profit_lock_hits}"
    )
    print(
        f"Daily Loss Circuits       : "
        f"{daily_loss_hits}"
    )
    print(
        f"Weekly Stops              : "
        f"{weekly_stop_hits}"
    )
    print(
        f"Hard Locks                : "
        f"{hard_lock_hits}"
    )
    print(
        f"Recovery Setups           : "
        f"{recovery_setup_detections}"
    )
    print(
        f"Recovery Trades           : "
        f"{recovery_trades}"
    )
    print(
        f"Final Engine Mode         : "
        f"{status['mode']}"
    )
    print(
        f"Final Mode Reason         : "
        f"{status['reason']}"
    )
    print("=" * 95)

    export_trade_history(
        trade_history,
        csv_file,
    )


# ============================================================
# PROGRAM ENTRY
# ============================================================

if __name__ == "__main__":

    try:

        run_backtest()

    except KeyboardInterrupt:

        print()
        print("Backtest stopped manually.")

    except Exception as error:

        print()
        print("=" * 95)
        print("BACKTEST ERROR")
        print("=" * 95)

        print(
            f"{type(error).__name__}: "
            f"{error}"
        )

        print("=" * 95)

        raise
    