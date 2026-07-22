from dataclasses import dataclass
from enum import Enum
import math
from typing import Any


class MarketRegime(str, Enum):
    SLEEP = "SLEEP"
    WATCH = "WATCH"
    ACTIVE = "ACTIVE"
    STRONG = "STRONG"


@dataclass(frozen=True)
class AssetRegimeProfile:
    minimum_adx: float
    minimum_volume_ratio: float
    minimum_ema_gap_percent: float
    minimum_atr_percent: float
    maximum_atr_percent: float
    strong_score_required: float = 75.0


@dataclass(frozen=True)
class RegimeDecision:
    symbol: str
    regime: MarketRegime
    score: float
    capital_usage_percent: float
    risk_percent: float
    reasons: tuple[str, ...]


DEFAULT_PROFILE = AssetRegimeProfile(
    minimum_adx=22.0,
    minimum_volume_ratio=1.00,
    minimum_ema_gap_percent=0.08,
    minimum_atr_percent=0.40,
    maximum_atr_percent=5.00,
)

ASSET_PROFILES: dict[str, AssetRegimeProfile] = {
    "BTCUSDT": AssetRegimeProfile(20.0, 1.00, 0.05, 0.25, 3.50),
    "ETHUSDT": AssetRegimeProfile(22.0, 1.00, 0.08, 0.35, 4.50),
    "SOLUSDT": AssetRegimeProfile(24.0, 1.05, 0.10, 0.50, 6.00),
    "XRPUSDT": AssetRegimeProfile(22.0, 1.05, 0.08, 0.45, 6.00),
    "LINKUSDT": AssetRegimeProfile(24.0, 1.08, 0.10, 0.45, 5.50),
    "DOGEUSDT": AssetRegimeProfile(
        27.0,
        1.15,
        0.12,
        0.60,
        6.50,
        strong_score_required=80.0,
    ),
}


def _finite_number(candle: Any, field: str) -> float | None:
    if field not in candle:
        return None

    try:
        value = float(candle[field])
    except (TypeError, ValueError):
        return None

    if not math.isfinite(value):
        return None

    return value


def _regime_from_score(
    score: float,
    profile: AssetRegimeProfile,
) -> MarketRegime:
    if score >= profile.strong_score_required:
        return MarketRegime.STRONG
    if score >= 60.0:
        return MarketRegime.ACTIVE
    if score >= 45.0:
        return MarketRegime.WATCH
    return MarketRegime.SLEEP


def _allocation_for_regime(
    regime: MarketRegime,
) -> tuple[float, float]:
    if regime == MarketRegime.STRONG:
        return 50.0, 0.75
    if regime == MarketRegime.ACTIVE:
        return 30.0, 0.50
    return 0.0, 0.0


def evaluate_regime(
    symbol: str,
    candle: Any,
) -> RegimeDecision:
    """
    Score one asset using current and historical-only indicators.

    Expected fields:
        close, EMA20, EMA50, EMA200, RSI14, ADX14,
        ATR14, volume, VolumeSMA20,
        MOMENTUM_24H, EMA50_SLOPE_12H
    """

    normalized_symbol = symbol.upper()
    profile = ASSET_PROFILES.get(
        normalized_symbol,
        DEFAULT_PROFILE,
    )

    required_fields = (
        "close",
        "EMA20",
        "EMA50",
        "EMA200",
        "RSI14",
        "ADX14",
        "ATR14",
        "volume",
        "VolumeSMA20",
        "MOMENTUM_24H",
        "EMA50_SLOPE_12H",
    )

    values: dict[str, float] = {}

    for field in required_fields:
        value = _finite_number(candle, field)

        if value is None:
            return RegimeDecision(
                symbol=normalized_symbol,
                regime=MarketRegime.SLEEP,
                score=0.0,
                capital_usage_percent=0.0,
                risk_percent=0.0,
                reasons=(f"Missing/invalid {field}",),
            )

        values[field] = value

    close = values["close"]
    ema20 = values["EMA20"]
    ema50 = values["EMA50"]
    ema200 = values["EMA200"]
    rsi = values["RSI14"]
    adx = values["ADX14"]
    atr = values["ATR14"]
    volume = values["volume"]
    volume_sma = values["VolumeSMA20"]
    momentum_24h = values["MOMENTUM_24H"]
    ema50_slope_12h = values["EMA50_SLOPE_12H"]

    if min(close, ema20, ema50, ema200, volume_sma) <= 0 or atr < 0:
        return RegimeDecision(
            symbol=normalized_symbol,
            regime=MarketRegime.SLEEP,
            score=0.0,
            capital_usage_percent=0.0,
            risk_percent=0.0,
            reasons=("Invalid non-positive market values",),
        )

    score = 0.0
    reasons: list[str] = []

    ema_gap_percent = ((ema20 - ema50) / ema50) * 100
    atr_percent = (atr / close) * 100
    volume_ratio = volume / volume_sma
    price_above_ema200_percent = (
        (close - ema200) / ema200
    ) * 100

    if close > ema200:
        score += 20.0
        reasons.append("Price above EMA200")
    else:
        score -= 20.0
        reasons.append("Price below EMA200")

    if ema20 > ema50:
        score += 15.0
        reasons.append("EMA20 above EMA50")
    else:
        score -= 15.0
        reasons.append("EMA20 below EMA50")

    if ema50_slope_12h > 0.15:
        score += 10.0
        reasons.append("EMA50 slope strongly positive")
    elif ema50_slope_12h > 0:
        score += 5.0
        reasons.append("EMA50 slope positive")
    else:
        score -= 5.0
        reasons.append("EMA50 slope non-positive")

    if ema_gap_percent >= profile.minimum_ema_gap_percent:
        score += 10.0
        reasons.append("EMA trend separation healthy")
    elif ema_gap_percent > 0:
        score += 4.0
        reasons.append("EMA trend separation weak")

    if adx >= profile.minimum_adx + 8:
        score += 15.0
        reasons.append("ADX strong")
    elif adx >= profile.minimum_adx:
        score += 10.0
        reasons.append("ADX qualified")
    elif adx >= 17:
        score += 3.0
        reasons.append("ADX developing")
    else:
        score -= 7.0
        reasons.append("ADX weak")

    if 50 <= rsi <= 68:
        score += 10.0
        reasons.append("RSI healthy bullish zone")
    elif 45 <= rsi < 50:
        score += 4.0
        reasons.append("RSI recovering")
    elif 68 < rsi <= 74:
        score += 3.0
        reasons.append("RSI strong but extended")
    elif rsi < 40:
        score -= 8.0
        reasons.append("RSI weak")
    elif rsi > 78:
        score -= 6.0
        reasons.append("RSI overheated")

    if volume_ratio >= profile.minimum_volume_ratio + 0.30:
        score += 10.0
        reasons.append("Volume expansion strong")
    elif volume_ratio >= profile.minimum_volume_ratio:
        score += 6.0
        reasons.append("Volume confirmed")
    elif volume_ratio >= 0.85:
        score += 1.0
        reasons.append("Volume neutral")
    else:
        score -= 4.0
        reasons.append("Volume weak")

    if (
        profile.minimum_atr_percent
        <= atr_percent
        <= profile.maximum_atr_percent
    ):
        score += 10.0
        reasons.append("ATR regime tradable")
    elif atr_percent > profile.maximum_atr_percent:
        score -= 8.0
        reasons.append("ATR regime too volatile")
    else:
        score -= 3.0
        reasons.append("ATR regime too quiet")

    if momentum_24h >= 3.0:
        score += 10.0
        reasons.append("24h momentum strong")
    elif momentum_24h >= 0.5:
        score += 6.0
        reasons.append("24h momentum positive")
    elif momentum_24h > -1.0:
        score += 1.0
        reasons.append("24h momentum neutral")
    else:
        score -= 7.0
        reasons.append("24h momentum negative")

    if 0 < price_above_ema200_percent <= 15:
        score += 5.0
        reasons.append("Price extension above EMA200 healthy")
    elif price_above_ema200_percent > 25:
        score -= 3.0
        reasons.append("Price highly extended above EMA200")

    score = max(0.0, min(100.0, score))
    regime = _regime_from_score(score, profile)
    capital_usage_percent, risk_percent = _allocation_for_regime(regime)

    return RegimeDecision(
        symbol=normalized_symbol,
        regime=regime,
        score=round(score, 2),
        capital_usage_percent=capital_usage_percent,
        risk_percent=risk_percent,
        reasons=tuple(reasons),
    )
