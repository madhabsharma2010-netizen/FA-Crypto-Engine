from dataclasses import dataclass
from enum import Enum
from typing import Optional

from config.risk_settings import RiskSettings


class TradingMode(str, Enum):
    ACTIVE = "ACTIVE"
    PROFIT_LOCK = "PROFIT_LOCK"
    RECOVERY_WATCH = "RECOVERY_WATCH"
    DAILY_STOP = "DAILY_STOP"
    WEEKLY_STOP = "WEEKLY_STOP"
    HARD_LOCK = "HARD_LOCK"
    MANUAL_STOP = "MANUAL_STOP"


@dataclass
class PositionPlan:
    quantity: float
    notional: float
    risk_amount: float
    stop_distance: float
    capital_limit: float
    risk_percent_used: float


class RiskManager:
    """
    Central risk-control system for FA Crypto Engine.
    """

    def __init__(
        self,
        settings: RiskSettings,
        starting_equity: float,
    ) -> None:

        self.settings = settings

        self.starting_equity = float(starting_equity)

        self.current_day = None
        self.current_week = None

        self.day_start_equity = float(starting_equity)
        self.week_start_equity = float(starting_equity)

        self.mode = TradingMode.ACTIVE

        self.mode_reason = "ENGINE INITIALIZED"

        self.normal_trades_today = 0
        self.recovery_trades_today = 0

        self.recovery_confirmation_count = 0
        self.recovery_approved = False

        self.profit_lock_activated = False

        self.highest_equity_after_profit_lock = 0.0

        self.cooldown_candles_remaining = 0

        self.initialize_engine_state()

    # ========================================================
    # INITIALIZATION
    # ========================================================

    def initialize_engine_state(self) -> None:

        if self.settings.engine_control == "STOP":
            self.mode = TradingMode.MANUAL_STOP
            self.mode_reason = "ENGINE CONTROL SET TO STOP"
            return

        if (
            self.settings.apply_existing_loss_to_bot_lock
            and self.settings.existing_portfolio_drawdown_percent
            >= self.settings.total_drawdown_hard_lock_percent
        ):
            self.mode = TradingMode.HARD_LOCK

            self.mode_reason = (
                "EXISTING PORTFOLIO DRAWDOWN EXCEEDS "
                "TOTAL HARD-LOCK LIMIT"
            )

    # ========================================================
    # GENERAL CALCULATIONS
    # ========================================================

    @staticmethod
    def percentage_change(
        starting_value: float,
        current_value: float,
    ) -> float:

        if starting_value <= 0:
            return 0.0

        return (
            (current_value - starting_value)
            / starting_value
        ) * 100

    def daily_change(
        self,
        current_equity: float,
    ) -> float:

        return self.percentage_change(
            self.day_start_equity,
            current_equity,
        )

    def weekly_change(
        self,
        current_equity: float,
    ) -> float:

        return self.percentage_change(
            self.week_start_equity,
            current_equity,
        )

    def total_change(
        self,
        current_equity: float,
    ) -> float:

        return self.percentage_change(
            self.starting_equity,
            current_equity,
        )

    # ========================================================
    # DAY/WEEK RESET
    # ========================================================

    def update_period(
        self,
        candle_time,
        current_equity: float,
    ) -> None:

        candle_day = candle_time.date()

        iso_calendar = candle_time.isocalendar()

        candle_week = (
            int(iso_calendar.year),
            int(iso_calendar.week),
        )

        if candle_week != self.current_week:

            self.current_week = candle_week

            self.week_start_equity = float(
                current_equity
            )

            if self.mode == TradingMode.WEEKLY_STOP:
                self.mode = TradingMode.ACTIVE
                self.mode_reason = "NEW WEEK RESET"

        if candle_day != self.current_day:

            self.current_day = candle_day

            self.day_start_equity = float(
                current_equity
            )

            self.normal_trades_today = 0
            self.recovery_trades_today = 0

            self.recovery_confirmation_count = 0
            self.recovery_approved = False

            self.profit_lock_activated = False
            self.highest_equity_after_profit_lock = 0.0

            self.cooldown_candles_remaining = 0

            if self.mode in {
                TradingMode.DAILY_STOP,
                TradingMode.RECOVERY_WATCH,
                TradingMode.PROFIT_LOCK,
            }:
                self.mode = TradingMode.ACTIVE
                self.mode_reason = "NEW DAY RESET"

    # ========================================================
    # MASTER ACCOUNT LIMITS
    # ========================================================

    def evaluate_account_limits(
        self,
        current_equity: float,
        position_open: bool,
    ) -> Optional[str]:

        if self.mode == TradingMode.HARD_LOCK:
            return self.mode_reason

        if self.mode == TradingMode.MANUAL_STOP:
            return self.mode_reason

        total_change = self.total_change(
            current_equity
        )

        weekly_change = self.weekly_change(
            current_equity
        )

        daily_change = self.daily_change(
            current_equity
        )

        # ----------------------------------------------------
        # TOTAL HARD LOCK
        # ----------------------------------------------------

        if (
            total_change
            <= -self.settings.total_drawdown_hard_lock_percent
        ):
            self.mode = TradingMode.HARD_LOCK

            self.mode_reason = (
                "TOTAL DRAWDOWN HARD LOCK"
            )

            return self.mode_reason

        # ----------------------------------------------------
        # WEEKLY LOSS STOP
        # ----------------------------------------------------

        if (
            weekly_change
            <= -self.settings.weekly_loss_limit_percent
        ):
            self.mode = TradingMode.WEEKLY_STOP

            self.mode_reason = (
                "WEEKLY LOSS LIMIT REACHED"
            )

            return self.mode_reason

        # ----------------------------------------------------
        # DAILY LOSS CIRCUIT
        # ----------------------------------------------------

        if (
            daily_change
            <= -self.settings.daily_loss_limit_percent
        ):
            if self.settings.recovery_watch_enabled:

                self.mode = TradingMode.RECOVERY_WATCH

                self.mode_reason = (
                    "DAILY LOSS LIMIT REACHED — "
                    "RECOVERY WATCH ACTIVE"
                )

            else:
                self.mode = TradingMode.DAILY_STOP

                self.mode_reason = (
                    "DAILY LOSS LIMIT REACHED"
                )

            return self.mode_reason

        # ----------------------------------------------------
        # MAXIMUM DAILY PROFIT
        # ----------------------------------------------------

        if (
            daily_change
            >= self.settings.daily_profit_extension_percent
        ):
            self.mode = TradingMode.DAILY_STOP

            self.mode_reason = (
                "MAXIMUM DAILY PROFIT TARGET REACHED"
            )

            return self.mode_reason

        # ----------------------------------------------------
        # DAILY PROFIT LOCK
        # ----------------------------------------------------

        if (
            daily_change
            >= self.settings.daily_profit_lock_percent
            and not self.profit_lock_activated
        ):
            self.profit_lock_activated = True

            self.mode = TradingMode.PROFIT_LOCK

            self.mode_reason = (
                "DAILY PROFIT LOCK ACTIVATED"
            )

            self.highest_equity_after_profit_lock = (
                current_equity
            )

            # Position open hai to existing winner continue.
            # Position nahi hai to day finish.
            if not position_open:
                self.mode = TradingMode.DAILY_STOP

                self.mode_reason = (
                    "DAILY PROFIT SECURED — "
                    "NO OPEN POSITION"
                )

            return self.mode_reason

        return None

    # ========================================================
    # PROFIT EXTENSION AND TRAILING EQUITY
    # ========================================================

    def update_profit_extension(
        self,
        current_equity: float,
    ) -> Optional[str]:

        if self.mode != TradingMode.PROFIT_LOCK:
            return None

        if (
            current_equity
            > self.highest_equity_after_profit_lock
        ):
            self.highest_equity_after_profit_lock = (
                current_equity
            )

        trailing_floor = (
            self.highest_equity_after_profit_lock
            * (
                1
                - self.settings.trailing_stop_percent
                / 100
            )
        )

        if current_equity <= trailing_floor:

            self.mode = TradingMode.DAILY_STOP

            self.mode_reason = (
                "PROFIT EXTENSION TRAILING STOP"
            )

            return self.mode_reason

        daily_change = self.daily_change(
            current_equity
        )

        if (
            daily_change
            >= self.settings.daily_profit_extension_percent
        ):
            self.mode = TradingMode.DAILY_STOP

            self.mode_reason = (
                "DAILY PROFIT EXTENSION TARGET REACHED"
            )

            return self.mode_reason

        return None

    # ========================================================
    # MARKET REGIME
    # ========================================================

    def normal_market_is_tradeable(
        self,
        candle,
    ) -> bool:

        close_price = float(
            candle["close"]
        )

        ema20 = float(
            candle["EMA20"]
        )

        ema50 = float(
            candle["EMA50"]
        )

        ema200 = float(
            candle["EMA200"]
        )

        if (
            self.settings.require_price_above_ema200
            and close_price <= ema200
        ):
            return False

        if (
            self.settings.require_ema20_above_ema50
            and ema20 <= ema50
        ):
            return False

        return True

    # ========================================================
    # RECOVERY SCORE
    # ========================================================

    def calculate_recovery_score(
        self,
        candle,
    ) -> float:

        close_price = float(
            candle["close"]
        )

        ema20 = float(
            candle["EMA20"]
        )

        ema50 = float(
            candle["EMA50"]
        )

        ema200 = float(
            candle["EMA200"]
        )

        rsi = float(
            candle["RSI14"]
        )

        volume = float(
            candle["volume"]
        )

        volume_average = float(
            candle["VOLUME_AVG"]
        )

        score = 0.0

        # Price above long-term trend.
        if close_price > ema200:
            score += 25.0

        # Short-term bullish structure.
        if ema20 > ema50:
            score += 25.0

        # Price above fast EMA.
        if close_price > ema20:
            score += 15.0

        # Healthy RSI range.
        if (
            self.settings.recovery_minimum_rsi
            <= rsi
            <= self.settings.recovery_maximum_rsi
        ):
            score += 15.0

        # Volume confirmation.
        if volume > volume_average:
            score += 20.0

        return score

    def update_recovery_watch(
        self,
        candle,
    ) -> float:

        if self.mode != TradingMode.RECOVERY_WATCH:
            return 0.0

        score = self.calculate_recovery_score(
            candle
        )

        if (
            score
            >= self.settings.recovery_score_required
        ):
            self.recovery_confirmation_count += 1
        else:
            self.recovery_confirmation_count = 0
            self.recovery_approved = False

        setup_confirmed = (
            self.recovery_confirmation_count
            >= self.settings.recovery_confirmation_candles
        )

        if setup_confirmed:

            if (
                not self.settings.recovery_requires_manual_approval
            ):
                self.recovery_approved = True

            elif (
                self.settings.auto_approve_recovery_in_backtest
            ):
                self.recovery_approved = True

        return score

    def manually_approve_recovery(self) -> None:

        if self.mode != TradingMode.RECOVERY_WATCH:
            return

        self.recovery_approved = True

        self.mode_reason = (
            "RECOVERY TRADE MANUALLY APPROVED"
        )

    # ========================================================
    # ENTRY PERMISSION
    # ========================================================

    def can_open_normal_trade(
        self,
        candle,
    ) -> bool:

        if self.mode != TradingMode.ACTIVE:
            return False

        if (
            self.normal_trades_today
            >= self.settings.max_normal_trades_per_day
        ):
            return False

        if self.cooldown_candles_remaining > 0:
            return False

        return self.normal_market_is_tradeable(
            candle
        )

    def can_open_recovery_trade(
        self,
    ) -> bool:

        if self.mode != TradingMode.RECOVERY_WATCH:
            return False

        if not self.recovery_approved:
            return False

        if (
            self.recovery_trades_today
            >= self.settings.max_recovery_trades_per_day
        ):
            return False

        if self.cooldown_candles_remaining > 0:
            return False

        return True

    # ========================================================
    # POSITION SIZING
    # ========================================================

    def calculate_position_plan(
        self,
        current_equity: float,
        available_cash: float,
        entry_price: float,
        recovery_trade: bool = False,
    ) -> PositionPlan:

        if entry_price <= 0:
            return PositionPlan(
                quantity=0.0,
                notional=0.0,
                risk_amount=0.0,
                stop_distance=0.0,
                capital_limit=0.0,
                risk_percent_used=0.0,
            )

        if self.settings.compound_profits:
            sizing_equity = current_equity
        else:
            sizing_equity = min(
                current_equity,
                self.settings.fixed_trading_capital,
            )

        if recovery_trade:
            risk_percent = (
                self.settings.recovery_risk_per_trade_percent
            )
        else:
            risk_percent = (
                self.settings.risk_per_trade_percent
            )

        risk_amount = (
            sizing_equity
            * risk_percent
            / 100
        )

        stop_distance = (
            entry_price
            * self.settings.stop_loss_percent
            / 100
        )

        risk_based_quantity = (
            risk_amount
            / stop_distance
        )

        risk_based_notional = (
            risk_based_quantity
            * entry_price
        )

        capital_usage_limit = (
            sizing_equity
            * self.settings.max_capital_usage_percent
            / 100
            * self.settings.current_leverage
        )

        cash_limit = available_cash

        final_notional = min(
            risk_based_notional,
            capital_usage_limit,
            cash_limit,
        )

        final_quantity = (
            final_notional
            / entry_price
        )

        return PositionPlan(
            quantity=final_quantity,
            notional=final_notional,
            risk_amount=risk_amount,
            stop_distance=stop_distance,
            capital_limit=capital_usage_limit,
            risk_percent_used=risk_percent,
        )

    # ========================================================
    # TRADE EXIT
    # ========================================================

    def check_position_exit(
        self,
        entry_price: float,
        candle_low: float,
        candle_high: float,
    ) -> Optional[tuple[str, float]]:

        stop_price = (
            entry_price
            * (
                1
                - self.settings.stop_loss_percent
                / 100
            )
        )

        target_price = (
            entry_price
            * (
                1
                + self.settings.take_profit_percent
                / 100
            )
        )

        # Conservative assumption:
        # Same candle mein stop aur target dono hit hon,
        # to stop-loss first maana jayega.
        if candle_low <= stop_price:
            return "STOP LOSS", stop_price

        if candle_high >= target_price:
            return "TAKE PROFIT", target_price

        return None

    # ========================================================
    # TRADE COUNTERS
    # ========================================================

    def register_trade_entry(
        self,
        recovery_trade: bool,
    ) -> None:

        if recovery_trade:
            self.recovery_trades_today += 1
            self.recovery_approved = False
        else:
            self.normal_trades_today += 1

    def register_trade_exit(self) -> None:

        self.cooldown_candles_remaining = (
            self.settings.cooldown_candles_after_exit
        )

    def process_candle_cooldown(self) -> None:

        if self.cooldown_candles_remaining > 0:
            self.cooldown_candles_remaining -= 1

    # ========================================================
    # MANUAL CONTROL
    # ========================================================

    def manual_stop(self) -> None:

        self.mode = TradingMode.MANUAL_STOP
        self.mode_reason = "MANUAL STOP ACTIVATED"

    def manual_start(self) -> None:

        if self.mode == TradingMode.HARD_LOCK:
            raise RuntimeError(
                "Hard lock cannot be removed without risk review."
            )

        self.mode = TradingMode.ACTIVE
        self.mode_reason = "MANUAL START ACTIVATED"

    def clear_hard_lock_after_review(
        self,
        approved_new_baseline: float,
    ) -> None:

        if approved_new_baseline <= 0:
            raise ValueError(
                "Approved baseline must be above zero."
            )

        self.starting_equity = float(
            approved_new_baseline
        )

        self.day_start_equity = float(
            approved_new_baseline
        )

        self.week_start_equity = float(
            approved_new_baseline
        )

        self.mode = TradingMode.ACTIVE

        self.mode_reason = (
            "HARD LOCK CLEARED AFTER MANUAL REVIEW"
        )

    # ========================================================
    # STATUS
    # ========================================================

    def status_report(
        self,
        current_equity: float,
    ) -> dict:

        return {
            "mode": self.mode.value,
            "reason": self.mode_reason,
            "daily_change_percent": round(
                self.daily_change(current_equity),
                4,
            ),
            "weekly_change_percent": round(
                self.weekly_change(current_equity),
                4,
            ),
            "total_change_percent": round(
                self.total_change(current_equity),
                4,
            ),
            "normal_trades_today": (
                self.normal_trades_today
            ),
            "recovery_trades_today": (
                self.recovery_trades_today
            ),
            "recovery_confirmations": (
                self.recovery_confirmation_count
            ),
            "recovery_approved": (
                self.recovery_approved
            ),
            "cooldown_remaining": (
                self.cooldown_candles_remaining
            ),
        }