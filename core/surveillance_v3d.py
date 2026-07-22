from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

import pandas as pd


class ShockLevel(str, Enum):
    NORMAL = "NORMAL"
    WARNING = "WARNING"
    SHOCK = "SHOCK"
    SEVERE = "SEVERE"


class MarketState(str, Enum):
    STRONG_BULL = "STRONG_BULL"
    BULL = "BULL"
    NEUTRAL = "NEUTRAL"
    BEARISH_RECOVERY = "BEARISH_RECOVERY"
    BEAR = "BEAR"
    SHOCK = "SHOCK"


@dataclass(frozen=True)
class AssetShockProfile:
    warning_15m: float
    shock_15m: float
    severe_15m: float
    warning_60m: float
    shock_60m: float
    severe_60m: float


@dataclass(frozen=True)
class AssetShockDecision:
    symbol: str
    level: ShockLevel
    score: float
    freeze_new_entries: bool
    suggested_reduction_percent: float
    emergency_atr_multiple: float | None
    force_exit: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class MarketShockDecision:
    level: ShockLevel
    score: float
    freeze_all_entries: bool
    suggested_portfolio_reduction_percent: float
    force_cash_mode: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class MarketStateDecision:
    state: MarketState
    score: float
    btc_score: float
    eth_score: float
    breadth_score: float
    new_entries_allowed: bool
    altcoins_allowed: bool
    risk_multiplier: float
    reasons: tuple[str, ...]


SHOCK_PROFILES: dict[str, AssetShockProfile] = {
    "BTCUSDT": AssetShockProfile(-0.80, -1.50, -2.50, -1.50, -3.00, -5.00),
    "ETHUSDT": AssetShockProfile(-1.00, -2.00, -3.50, -2.00, -4.00, -6.50),
    "SOLUSDT": AssetShockProfile(-1.50, -3.00, -5.50, -3.00, -6.00, -9.00),
    "XRPUSDT": AssetShockProfile(-1.40, -2.80, -5.00, -2.80, -5.50, -8.50),
    "LINKUSDT": AssetShockProfile(-1.40, -2.80, -5.00, -2.80, -5.50, -8.50),
    "DOGEUSDT": AssetShockProfile(-2.00, -4.00, -7.00, -4.00, -8.00, -12.00),
}


def _profile_for(symbol: str) -> AssetShockProfile:
    return SHOCK_PROFILES.get(symbol, SHOCK_PROFILES["SOLUSDT"])


def evaluate_asset_shock(
    symbol: str,
    candle: pd.Series,
) -> AssetShockDecision:
    profile = _profile_for(symbol)
    reasons: list[str] = []
    score = 0.0

    return_15m = float(candle["RETURN_15M"])
    return_30m = float(candle["RETURN_30M"])
    return_60m = float(candle["RETURN_60M"])
    volume_ratio = float(candle["volume"]) / max(
        float(candle["VolumeSMA20"]), 1e-12
    )
    atr_expansion = float(candle["ATR_EXPANSION"])
    close_price = float(candle["close"])
    ema20 = float(candle["EMA20"])
    ema50 = float(candle["EMA50"])
    support_4h = float(candle["SUPPORT_4H"])
    relative_60m = float(candle["MARKET_RELATIVE_60M"])

    if return_15m <= profile.warning_15m:
        score += 18.0
        reasons.append(f"15m fall {return_15m:.2f}%")
    if return_15m <= profile.shock_15m:
        score += 22.0
    if return_15m <= profile.severe_15m:
        score += 30.0

    if return_60m <= profile.warning_60m:
        score += 15.0
        reasons.append(f"60m fall {return_60m:.2f}%")
    if return_60m <= profile.shock_60m:
        score += 20.0
    if return_60m <= profile.severe_60m:
        score += 25.0

    if return_30m <= profile.warning_15m * 1.35:
        score += 8.0
    if volume_ratio >= 1.80:
        score += 10.0
        reasons.append(f"sell-volume ratio {volume_ratio:.2f}")
    if volume_ratio >= 2.75:
        score += 10.0
    if atr_expansion >= 1.60:
        score += 8.0
        reasons.append(f"ATR expansion {atr_expansion:.2f}x")
    if atr_expansion >= 2.25:
        score += 10.0
    if close_price < ema20:
        score += 5.0
    if close_price < ema50:
        score += 8.0
        reasons.append("price below 15m EMA50")
    if close_price < support_4h:
        score += 15.0
        reasons.append("4h intraday support broken")
    if relative_60m <= -2.00:
        score += 8.0
        reasons.append(f"underperforming market {relative_60m:.2f}%")

    score = min(score, 100.0)
    severe_direct = (
        return_15m <= profile.severe_15m
        or return_60m <= profile.severe_60m
    )

    if severe_direct or score >= 75.0:
        level = ShockLevel.SEVERE
    elif score >= 50.0:
        level = ShockLevel.SHOCK
    elif score >= 25.0:
        level = ShockLevel.WARNING
    else:
        level = ShockLevel.NORMAL

    if level == ShockLevel.NORMAL:
        return AssetShockDecision(
            symbol, level, score, False, 0.0, None, False, tuple(reasons)
        )
    if level == ShockLevel.WARNING:
        return AssetShockDecision(
            symbol, level, score, True, 0.0, 2.00, False, tuple(reasons)
        )
    if level == ShockLevel.SHOCK:
        return AssetShockDecision(
            symbol, level, score, True, 35.0, 1.35, False, tuple(reasons)
        )
    return AssetShockDecision(
        symbol, level, score, True, 100.0, 0.90, True, tuple(reasons)
    )


def evaluate_market_shock(
    asset_decisions: Iterable[AssetShockDecision],
    median_return_15m: float,
    median_return_60m: float,
) -> MarketShockDecision:
    decisions = list(asset_decisions)
    by_symbol = {item.symbol: item for item in decisions}

    severe_count = sum(item.level == ShockLevel.SEVERE for item in decisions)
    shock_count = sum(
        item.level in {ShockLevel.SHOCK, ShockLevel.SEVERE}
        for item in decisions
    )
    warning_count = sum(
        item.level != ShockLevel.NORMAL
        for item in decisions
    )

    btc_level = by_symbol.get(
        "BTCUSDT",
        AssetShockDecision(
            "BTCUSDT", ShockLevel.NORMAL, 0.0, False, 0.0, None, False, ()
        ),
    ).level
    eth_level = by_symbol.get(
        "ETHUSDT",
        AssetShockDecision(
            "ETHUSDT", ShockLevel.NORMAL, 0.0, False, 0.0, None, False, ()
        ),
    ).level

    reasons: list[str] = []
    score = 0.0

    if btc_level == ShockLevel.WARNING:
        score += 15.0
        reasons.append("BTC warning")
    elif btc_level == ShockLevel.SHOCK:
        score += 30.0
        reasons.append("BTC shock")
    elif btc_level == ShockLevel.SEVERE:
        score += 50.0
        reasons.append("BTC severe shock")

    if eth_level == ShockLevel.WARNING:
        score += 8.0
    elif eth_level == ShockLevel.SHOCK:
        score += 18.0
        reasons.append("ETH shock")
    elif eth_level == ShockLevel.SEVERE:
        score += 30.0
        reasons.append("ETH severe shock")

    score += min(severe_count * 20.0, 40.0)
    score += min(shock_count * 10.0, 30.0)
    score += min(warning_count * 4.0, 20.0)

    if median_return_15m <= -1.25:
        score += 20.0
        reasons.append(f"market median 15m {median_return_15m:.2f}%")
    if median_return_60m <= -2.50:
        score += 25.0
        reasons.append(f"market median 60m {median_return_60m:.2f}%")

    score = min(score, 100.0)

    if severe_count >= 2 or score >= 75.0:
        return MarketShockDecision(
            ShockLevel.SEVERE, score, True, 100.0, True, tuple(reasons)
        )
    if shock_count >= 2 or score >= 50.0:
        return MarketShockDecision(
            ShockLevel.SHOCK, score, True, 50.0, False, tuple(reasons)
        )
    if warning_count >= 2 or score >= 25.0:
        return MarketShockDecision(
            ShockLevel.WARNING, score, True, 0.0, False, tuple(reasons)
        )
    return MarketShockDecision(
        ShockLevel.NORMAL, score, False, 0.0, False, tuple(reasons)
    )


def timeframe_trend_score(candle: pd.Series) -> float:
    score = 0.0
    close_price = float(candle["close"])
    ema20 = float(candle["EMA20"])
    ema50 = float(candle["EMA50"])
    ema200 = float(candle["EMA200"])
    rsi = float(candle["RSI14"])
    adx = float(candle["ADX14"])
    slope = float(candle["EMA50_SLOPE"])
    momentum = float(candle["MOMENTUM"])

    if close_price > ema200:
        score += 20.0
    if ema20 > ema50:
        score += 18.0
    if ema50 > ema200:
        score += 17.0
    if slope > 0:
        score += 15.0
    if momentum > 0:
        score += 10.0
    if 48.0 <= rsi <= 72.0:
        score += 10.0
    elif rsi > 72.0:
        score += 4.0
    if adx >= 20.0:
        score += 6.0
    if adx >= 28.0:
        score += 4.0

    return min(score, 100.0)


def evaluate_market_state(
    btc_1h: pd.Series,
    btc_2h: pd.Series,
    btc_4h: pd.Series,
    btc_1d: pd.Series,
    eth_1h: pd.Series,
    eth_2h: pd.Series,
    eth_4h: pd.Series,
    breadth_score: float,
    market_shock: ShockLevel,
) -> MarketStateDecision:
    if market_shock in {ShockLevel.SHOCK, ShockLevel.SEVERE}:
        return MarketStateDecision(
            MarketState.SHOCK,
            0.0,
            0.0,
            0.0,
            breadth_score,
            False,
            False,
            0.0,
            (f"15m market shock {market_shock.value}",),
        )

    btc_scores = {
        "1h": timeframe_trend_score(btc_1h),
        "2h": timeframe_trend_score(btc_2h),
        "4h": timeframe_trend_score(btc_4h),
        "1d": timeframe_trend_score(btc_1d),
    }
    eth_scores = {
        "1h": timeframe_trend_score(eth_1h),
        "2h": timeframe_trend_score(eth_2h),
        "4h": timeframe_trend_score(eth_4h),
    }

    btc_score = (
        btc_scores["1h"] * 0.20
        + btc_scores["2h"] * 0.25
        + btc_scores["4h"] * 0.35
        + btc_scores["1d"] * 0.20
    )
    eth_score = (
        eth_scores["1h"] * 0.20
        + eth_scores["2h"] * 0.30
        + eth_scores["4h"] * 0.50
    )
    combined = (
        btc_score * 0.65
        + eth_score * 0.15
        + breadth_score * 0.20
    )
    reasons = (
        f"BTC composite {btc_score:.1f}",
        f"ETH confirmation {eth_score:.1f}",
        f"breadth {breadth_score:.1f}",
    )

    if combined >= 80.0:
        return MarketStateDecision(
            MarketState.STRONG_BULL,
            combined,
            btc_score,
            eth_score,
            breadth_score,
            True,
            True,
            1.00,
            reasons,
        )
    if combined >= 65.0:
        return MarketStateDecision(
            MarketState.BULL,
            combined,
            btc_score,
            eth_score,
            breadth_score,
            True,
            True,
            0.75,
            reasons,
        )
    if combined >= 45.0:
        return MarketStateDecision(
            MarketState.NEUTRAL,
            combined,
            btc_score,
            eth_score,
            breadth_score,
            True,
            False,
            0.35,
            reasons,
        )

    fast_recovery = (
        btc_scores["1h"] >= 60.0
        and btc_scores["2h"] >= 55.0
        and btc_scores["4h"] < 50.0
    )
    if combined >= 30.0 and fast_recovery:
        return MarketStateDecision(
            MarketState.BEARISH_RECOVERY,
            combined,
            btc_score,
            eth_score,
            breadth_score,
            False,
            False,
            0.20,
            reasons,
        )

    return MarketStateDecision(
        MarketState.BEAR,
        combined,
        btc_score,
        eth_score,
        breadth_score,
        False,
        False,
        0.0,
        reasons,
    )
