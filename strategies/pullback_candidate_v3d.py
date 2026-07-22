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
    estimated_reward_percent: float
    reward_risk: float
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
    score = 0.0

    close_price = float(candle["close"])
    open_price = float(candle["open"])
    low_price = float(candle["low"])
    ema20 = float(candle["EMA20"])
    ema50 = float(candle["EMA50"])
    ema200 = float(candle["EMA200"])
    rsi = float(candle["RSI14"])
    adx = float(candle["ADX14"])
    volume_ratio = float(candle["VOLUME_RATIO"])
    momentum_24h = float(candle["MOMENTUM_24H"])
    momentum_72h = float(candle["MOMENTUM_72H"])
    slope_12h = float(candle["EMA50_SLOPE_12H"])
    trend_stable = bool(candle["TREND_STABLE_6"])
    recent_high = float(candle["RECENT_HIGH_72H"])
    recent_low = float(candle["RECENT_LOW_12H"])
    atr = float(candle["ATR14"])

    previous_close = float(previous["close"])
    previous_ema20 = float(previous["EMA20"])

    trend_stack = (
        close_price > ema200
        and ema20 > ema50 > ema200
    )

    if trend_stack:
        score += 22.0
        reasons.append("bullish EMA stack")
    if trend_stable:
        score += 12.0
        reasons.append("trend stable 6h")
    if slope_12h > 0:
        score += 10.0
    if momentum_24h > 0:
        score += 8.0
    if momentum_72h > 0:
        score += 6.0

    touched_ema20 = (
        low_price <= ema20 * 1.003
        and low_price >= ema50 * 0.985
    )
    reclaimed_ema20 = (
        close_price > ema20
        and (
            previous_close <= previous_ema20
            or low_price <= ema20
        )
        and close_price > open_price
    )

    if touched_ema20:
        score += 12.0
        reasons.append("healthy EMA20 pullback")
    if reclaimed_ema20:
        score += 16.0
        reasons.append("bullish EMA20 reclaim")

    if 42.0 <= rsi <= 62.0:
        score += 10.0
    elif 38.0 <= rsi <= 66.0:
        score += 4.0

    if adx >= 20.0:
        score += 5.0
    if volume_ratio >= 0.90:
        score += 5.0

    extension_percent = (
        (close_price - ema20)
        / ema20
        * 100
    )
    max_extension = MAX_EXTENSION_BY_SYMBOL.get(
        symbol,
        2.25,
    )
    not_extended = (
        extension_percent
        <= max_extension
    )

    if not_extended:
        score += 6.0
    else:
        reasons.append(
            f"extended {extension_percent:.2f}% above EMA20"
        )

    structure_stop = min(
        ema50 * 0.995,
        recent_low * 0.995,
    )
    atr_stop = (
        close_price
        - atr * 2.20
    )
    stop_price = max(
        0.0,
        min(
            structure_stop,
            atr_stop,
        ),
    )

    risk_percent = (
        (close_price - stop_price)
        / close_price
        * 100
        if close_price > 0
        else 0.0
    )

    estimated_reward_percent = (
        (recent_high - close_price)
        / close_price
        * 100
    )

    if estimated_reward_percent <= 0:
        estimated_reward_percent = (
            atr
            * 4.0
            / close_price
            * 100
        )

    reward_risk = (
        estimated_reward_percent
        / risk_percent
        if risk_percent > 0
        else 0.0
    )

    if reward_risk >= 2.0:
        score += 8.0
        reasons.append(
            f"estimated R:R {reward_risk:.2f}"
        )

    market_allowed = (
        market_state.new_entries_allowed
        and (
            market_state.altcoins_allowed
            or symbol
            in {
                "BTCUSDT",
                "ETHUSDT",
            }
        )
    )

    shock_safe = (
        asset_shock.level
        == ShockLevel.NORMAL
    )

    valid = all(
        (
            market_allowed,
            shock_safe,
            trend_stack,
            trend_stable,
            slope_12h > 0,
            momentum_24h > 0,
            touched_ema20,
            reclaimed_ema20,
            42.0 <= rsi <= 62.0,
            not_extended,
            reward_risk >= 2.0,
            score >= 72.0,
            market_state.state
            in {
                MarketState.STRONG_BULL,
                MarketState.BULL,
                MarketState.NEUTRAL,
            },
        )
    )

    return PullbackCandidate(
        symbol=symbol,
        valid=valid,
        score=min(score, 100.0),
        entry_price=close_price,
        stop_price=stop_price,
        risk_percent=risk_percent,
        estimated_reward_percent=estimated_reward_percent,
        reward_risk=reward_risk,
        reasons=tuple(reasons),
    )
