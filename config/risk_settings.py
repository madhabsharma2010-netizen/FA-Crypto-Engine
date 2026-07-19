from dataclasses import dataclass


@dataclass(frozen=True)
class RiskSettings:
    """
    FA Crypto Engine risk configuration.

    Percentage format:
        1.0 = 1%
        2.0 = 2%
    """

    # ========================================================
    # MASTER CONTROL
    # ========================================================

    # Valid values:
    # "START"
    # "STOP"
    engine_control: str = "START"

    allow_live_trading: bool = False
    allow_short_trading: bool = False

    # ========================================================
    # CAPITAL AND POSITION CONTROL
    # ========================================================

    starting_capital: float = 10000.0

    # False:
    # Position sizing original fixed capital ke basis par.
    #
    # True:
    # Profit/loss ke baad current equity ke basis par compounding.
    compound_profits: bool = False

    fixed_trading_capital: float = 10000.0

    # Maximum account risk if stop-loss hits.
    risk_per_trade_percent: float = 1.0

    # Total capital ka maximum portion jo ek position mein lagega.
    max_capital_usage_percent: float = 50.0

    stop_loss_percent: float = 2.0
    take_profit_percent: float = 5.0

    # ========================================================
    # DAILY PROFIT CONTROL
    # ========================================================

    # 3% hit hone ke baad new trades blocked.
    daily_profit_lock_percent: float = 3.0

    # Existing winning trade maximum 4% daily equity tak run kar sakti hai.
    daily_profit_extension_percent: float = 4.0

    enable_profit_extension: bool = True

    # Profit-lock ke baad existing winner ke liye trailing stop.
    trailing_stop_percent: float = 1.0

    # ========================================================
    # DAILY LOSS AND RECOVERY CONTROL
    # ========================================================

    daily_loss_limit_percent: float = 2.0

    recovery_watch_enabled: bool = True

    # Recovery setup quality score.
    recovery_score_required: float = 90.0

    # Setup itni consecutive candles tak valid hona chahiye.
    recovery_confirmation_candles: int = 3

    max_recovery_trades_per_day: int = 1

    # Live mode mein True hi rehna chahiye.
    recovery_requires_manual_approval: bool = True

    # Backtest mein manual button possible nahi hota.
    # True karne par backtest strong recovery setup ko
    # automatic approval maan kar test karega.
    auto_approve_recovery_in_backtest: bool = False

    # Recovery trade mein normal 1% ke badle reduced risk.
    recovery_risk_per_trade_percent: float = 0.50

    # ========================================================
    # WEEKLY AND TOTAL LOSS CONTROL
    # ========================================================

    weekly_loss_limit_percent: float = 5.0

    total_drawdown_hard_lock_percent: float = 10.0

    # Existing personal portfolio loss context.
    existing_portfolio_drawdown_percent: float = 35.0

    # Backtest ke liye False.
    # True karne par existing 35% loss ki wajah se bot immediately lock hoga.
    apply_existing_loss_to_bot_lock: bool = False

    # ========================================================
    # MARKET REGIME CONTROL
    # ========================================================

    require_price_above_ema200: bool = True
    require_ema20_above_ema50: bool = True

    recovery_minimum_rsi: float = 45.0
    recovery_maximum_rsi: float = 65.0

    require_volume_confirmation: bool = True
    volume_average_period: int = 20

    # ========================================================
    # TRADE FREQUENCY
    # ========================================================

    max_normal_trades_per_day: int = 3

    cooldown_candles_after_exit: int = 3

    # ========================================================
    # LEVERAGE
    # ========================================================

    max_allowed_leverage: float = 1.5

    # Current Spot engine:
    current_leverage: float = 1.0

    # ========================================================
    # COST ASSUMPTIONS
    # ========================================================

    trading_fee_percent: float = 0.10
    estimated_slippage_percent: float = 0.05

    # ========================================================
    # VALIDATION
    # ========================================================

    def validate(self) -> None:

        if self.engine_control not in {"START", "STOP"}:
            raise ValueError(
                "engine_control must be START or STOP."
            )

        if self.starting_capital <= 0:
            raise ValueError(
                "starting_capital must be above zero."
            )

        if self.fixed_trading_capital <= 0:
            raise ValueError(
                "fixed_trading_capital must be above zero."
            )

        if not 0 < self.risk_per_trade_percent <= 2:
            raise ValueError(
                "risk_per_trade_percent must be between 0 and 2."
            )

        if not 0 < self.recovery_risk_per_trade_percent <= 1:
            raise ValueError(
                "recovery_risk_per_trade_percent must be between 0 and 1."
            )

        if not 0 < self.max_capital_usage_percent <= 100:
            raise ValueError(
                "max_capital_usage_percent must be between 0 and 100."
            )

        if self.stop_loss_percent <= 0:
            raise ValueError(
                "stop_loss_percent must be above zero."
            )

        if self.take_profit_percent <= 0:
            raise ValueError(
                "take_profit_percent must be above zero."
            )

        if (
            self.daily_profit_extension_percent
            < self.daily_profit_lock_percent
        ):
            raise ValueError(
                "daily_profit_extension_percent cannot be below "
                "daily_profit_lock_percent."
            )

        if self.daily_loss_limit_percent <= 0:
            raise ValueError(
                "daily_loss_limit_percent must be above zero."
            )

        if self.weekly_loss_limit_percent <= 0:
            raise ValueError(
                "weekly_loss_limit_percent must be above zero."
            )

        if self.total_drawdown_hard_lock_percent <= 0:
            raise ValueError(
                "total_drawdown_hard_lock_percent must be above zero."
            )

        if self.max_normal_trades_per_day < 1:
            raise ValueError(
                "max_normal_trades_per_day must be at least 1."
            )

        if self.max_recovery_trades_per_day < 0:
            raise ValueError(
                "max_recovery_trades_per_day cannot be negative."
            )

        if not 0 <= self.recovery_score_required <= 100:
            raise ValueError(
                "recovery_score_required must be between 0 and 100."
            )

        if (
            self.recovery_minimum_rsi
            >= self.recovery_maximum_rsi
        ):
            raise ValueError(
                "recovery_minimum_rsi must be below recovery_maximum_rsi."
            )

        if self.current_leverage <= 0:
            raise ValueError(
                "current_leverage must be above zero."
            )

        if (
            self.current_leverage
            > self.max_allowed_leverage
        ):
            raise ValueError(
                "current_leverage exceeds max_allowed_leverage."
            )

        if self.allow_short_trading:
            raise ValueError(
                "Short trading is not available in the current Spot engine."
            )

        if self.trailing_stop_percent <= 0:
            raise ValueError(
                "trailing_stop_percent must be above zero."
            )

        if self.cooldown_candles_after_exit < 0:
            raise ValueError(
                "cooldown_candles_after_exit cannot be negative."
            )


RISK_SETTINGS = RiskSettings()

RISK_SETTINGS.validate()
