from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from core.surveillance_v3d import (
    AssetShockDecision,
    MarketState,
    MarketStateDecision,
    ShockLevel,
)


@dataclass(frozen=True)
class PullbackCandidate:
    symbol: str
    valid: bool
    score: float
    entry_price: float
    stop_price: float
    risk_percent: float
    projected_target: float
    estimated_reward_percent: float
    reward_risk: float
    failed_checks: tuple[str, ...]
    reasons: tuple[str, ...]


MAX_EXTENSION_BY_SYMBOL: dict[str, float] = {
    "BTCUSDT": 1.50,
    "ETHUSDT": 1.80,
    "SOLUSDT": 2.50,
    "XRPUSDT": 2.25,
    "LINKUSDT": 2.25,
    "DOGEUSDT": 3.00,
}


def evaluate_pullback_candidate(
    symbol: str,
    candle: pd.Series,
    previous: pd.Series,
    market_state: MarketStateDecision,
    asset_shock: AssetShockDecision,
) -> PullbackCandidate:
    reasons: list[str] = []
    failed: list[str] = []
    score = 0.0

    close_price = float(candle["close"])
    open_price = float(candle["open"])
    high_price = float(candle["high"])
    low_price = float(candle["low"])
    ema20 = float(candle["EMA20"])
    ema50 = float(candle["EMA50"])
    ema200 = float(candle["EMA200"])
    rsi = float(candle["RSI14"])
    adx = float(candle["ADX14"])
    previous_adx = float(previous["ADX14"])
    volume_ratio = float(candle["VOLUME_RATIO"])
    momentum_24h = float(candle["MOMENTUM_24H"])
    momentum_72h = float(candle["MOMENTUM_72H"])
    slope_12h = float(candle["EMA50_SLOPE_12H"])
    trend_stable = bool(candle["TREND_STABLE_4_OF_6"])
    recent_high = float(candle["RECENT_HIGH_72H"])
    recent_low = float(candle["RECENT_LOW_12H"])
    atr = float(candle["ATR14"])

    previous_close = float(previous["close"])
    previous_ema20 = float(previous["EMA20"])

    candle_range = max(
        high_price - low_price,
        1e-12,
    )
    close_location = (
        close_price - low_price
    ) / candle_range

    trend_stack = (
        close_price > ema200
        and ema20 > ema50 > ema200
    )

    if trend_stack:
        score += 18.0
        reasons.append("bullish EMA stack")
    else:
        failed.append("trend_stack")

    if trend_stable:
        score += 10.0
        reasons.append("trend stable 4-of-6h")
    else:
        failed.append("trend_stability")

    if slope_12h > 0:
        score += 8.0
    else:
        failed.append("ema50_slope")

    momentum_ok = (
        momentum_72h > 0
        and momentum_24h > -0.50
    )

    if momentum_ok:
        score += 10.0
    else:
        failed.append("momentum")

    touched_pullback_zone = (
        low_price
        <= ema20 + atr * 0.40
        and low_price
        >= ema50 - atr * 0.60
    )

    if touched_pullback_zone:
        score += 12.0
        reasons.append("controlled EMA pullback")
    else:
        failed.append("pullback_zone")

    bullish_reclaim = (
        close_price > ema20
        and close_price > open_price
        and close_location >= 0.60
        and (
            previous_close
            <= previous_ema20 * 1.005
            or low_price <= ema20
        )
    )

    if bullish_reclaim:
        score += 18.0
        reasons.append("bullish EMA20 reclaim")
    else:
        failed.append("reclaim")

    rsi_ok = 40.0 <= rsi <= 66.0

    if 44.0 <= rsi <= 62.0:
        score += 10.0
    elif rsi_ok:
        score += 5.0
    else:
        failed.append("rsi")

    adx_ok = (
        adx >= 18.0
        or adx > previous_adx
    )

    if adx_ok:
        score += 6.0
    else:
        failed.append("adx")

    volume_ok = volume_ratio >= 0.75

    if volume_ok:
        score += 4.0
    else:
        failed.append("volume")

    extension_percent = (
        (close_price - ema20)
        / ema20
        * 100
    )

    max_extension = (
        MAX_EXTENSION_BY_SYMBOL.get(
            symbol,
            2.25,
        )
    )

    not_extended = (
        extension_percent
        <= max_extension
    )

    if not_extended:
        score += 6.0
    else:
        failed.append("extension")

    # Use the nearest valid structural support, then protect
    # against stops that are either excessively tight or wide.
    nearest_support = max(
        recent_low,
        ema50,
    )

    support_stop = (
        nearest_support
        - atr * 0.35
    )

    atr_stop = (
        close_price
        - atr * 2.00
    )

    proposed_stop = max(
        support_stop,
        atr_stop,
    )

    minimum_distance_stop = (
        close_price
        - atr * 0.90
    )

    stop_price = min(
        proposed_stop,
        minimum_distance_stop,
    )

    stop_price = max(
        0.0,
        min(
            stop_price,
            close_price * 0.999,
        ),
    )

    risk_percent = (
        (close_price - stop_price)
        / close_price
        * 100
        if close_price > 0
        else 0.0
    )

    risk_ok = (
        0.35
        <= risk_percent
        <= 3.00
    )

    if not risk_ok:
        failed.append("stop_distance")

    # Trend systems should not treat the old 72h high as a hard
    # ceiling. Project continuation beyond it using ATR.
    projected_target = max(
        recent_high + atr * 0.50,
        close_price + atr * 3.00,
    )

    estimated_reward_percent = (
        (projected_target - close_price)
        / close_price
        * 100
    )

    reward_risk = (
        estimated_reward_percent
        / risk_percent
        if risk_percent > 0
        else 0.0
    )

    reward_ok = reward_risk >= 1.60

    if reward_ok:
        score += 8.0
        reasons.append(
            f"projected R:R {reward_risk:.2f}"
        )
    else:
        failed.append("reward_risk")

    candle_quality_ok = (
        close_location >= 0.60
        and close_price > open_price
    )

    if candle_quality_ok:
        score += 6.0
    else:
        failed.append("candle_quality")

    shock_safe = (
        asset_shock.level
        == ShockLevel.NORMAL
    )

    if not shock_safe:
        failed.append("asset_shock")

    market_allowed = False

    if market_state.state in {
        MarketState.STRONG_BULL,
        MarketState.BULL,
    }:
        market_allowed = (
            market_state.new_entries_allowed
        )

    elif (
        market_state.state
        == MarketState.NEUTRAL
        and symbol
        in {
            "BTCUSDT",
            "ETHUSDT",
        }
    ):
        market_allowed = (
            market_state.new_entries_allowed
        )

    if not market_allowed:
        failed.append("market_state")

    minimum_score = (
        78.0
        if market_state.state
        == MarketState.NEUTRAL
        else 70.0
    )

    score = min(score, 100.0)

    if score < minimum_score:
        failed.append("setup_score")

    valid = (
        len(failed) == 0
    )

    return PullbackCandidate(
        symbol=symbol,
        valid=valid,
        score=score,
        entry_price=close_price,
        stop_price=stop_price,
        risk_percent=risk_percent,
        projected_target=projected_target,
        estimated_reward_percent=(
            estimated_reward_percent
        ),
        reward_risk=reward_risk,
        failed_checks=tuple(failed),
        reasons=tuple(reasons),
    )
