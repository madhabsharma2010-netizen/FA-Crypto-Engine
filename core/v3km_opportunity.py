from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from core.surveillance_v3d import ShockLevel


class OpportunityBand(str, Enum):
    CASH = "CASH"
    PROBE = "PROBE"
    BUILD = "BUILD"
    FULL_PACE = "FULL_PACE"


@dataclass(frozen=True)
class OpportunityDecision:
    score: float
    band: OpportunityBand
    target_fraction: float
    eligible_for_new_entry: bool

    market_score: float
    trend_15m_score: float
    trend_1h_score: float
    trend_2h_score: float
    trend_4h_score: float
    relative_strength_score: float
    positive_volume_score: float

    reasons: tuple[str, ...]


# Fixed research weights.
#
# These are declared before the historical audit and must not be silently
# tuned to make one particular window profitable.
WEIGHT_MARKET = 0.25
WEIGHT_TREND_15M = 0.20
WEIGHT_TREND_1H = 0.20
WEIGHT_TREND_2H = 0.15
WEIGHT_TREND_4H = 0.10
WEIGHT_RELATIVE_STRENGTH = 0.05
WEIGHT_POSITIVE_VOLUME = 0.05


def _clip(value: float, lower: float = 0.0, upper: float = 100.0) -> float:
    return max(lower, min(float(value), upper))


def relative_strength_component(relative_return_60m: float) -> float:
    """
    Convert 60-minute market-relative return into a 0-100 score.

    -2.50% or worse = 0
     0.00%          = 50
    +2.50% or better = 100
    """
    clipped = max(-2.50, min(float(relative_return_60m), 2.50))
    return ((clipped + 2.50) / 5.00) * 100.0


def positive_volume_component(
    return_15m: float,
    volume_ratio: float,
) -> float:
    """
    Reward above-normal volume only when the completed 15-minute
    candle itself closed positively.

    A high-volume falling candle is not treated as bullish confirmation.
    """
    if float(return_15m) <= 0.0:
        return 0.0

    clipped_ratio = max(0.75, min(float(volume_ratio), 1.75))
    return ((clipped_ratio - 0.75) / 1.00) * 100.0


def opportunity_band(score: float) -> OpportunityBand:
    value = _clip(score)

    if value >= 80.0:
        return OpportunityBand.FULL_PACE
    if value >= 65.0:
        return OpportunityBand.BUILD
    if value >= 45.0:
        return OpportunityBand.PROBE
    return OpportunityBand.CASH


def permitted_target_fraction(score: float) -> float:
    band = opportunity_band(score)

    if band == OpportunityBand.FULL_PACE:
        return 1.00
    if band == OpportunityBand.BUILD:
        return 0.75
    if band == OpportunityBand.PROBE:
        return 0.30
    return 0.00


def evaluate_opportunity(
    *,
    market_score: float,
    trend_15m_score: float,
    trend_1h_score: float,
    trend_2h_score: float,
    trend_4h_score: float,
    relative_return_60m: float,
    return_15m: float,
    volume_ratio: float,
    asset_shock: ShockLevel,
    market_shock: ShockLevel,
) -> OpportunityDecision:
    """
    Produce an uncalibrated confidence score.

    This function does not claim that a score of 70 means a 70%
    probability of profit.

    Shock status remains independent from opportunity strength.
    A high score can therefore remain visible for research while
    WARNING, SHOCK or SEVERE protection blocks actual entry.
    """
    market = _clip(market_score)
    trend_15m = _clip(trend_15m_score)
    trend_1h = _clip(trend_1h_score)
    trend_2h = _clip(trend_2h_score)
    trend_4h = _clip(trend_4h_score)

    relative_strength = relative_strength_component(
        relative_return_60m
    )
    positive_volume = positive_volume_component(
        return_15m,
        volume_ratio,
    )

    score = (
        market * WEIGHT_MARKET
        + trend_15m * WEIGHT_TREND_15M
        + trend_1h * WEIGHT_TREND_1H
        + trend_2h * WEIGHT_TREND_2H
        + trend_4h * WEIGHT_TREND_4H
        + relative_strength * WEIGHT_RELATIVE_STRENGTH
        + positive_volume * WEIGHT_POSITIVE_VOLUME
    )
    score = round(_clip(score), 4)

    band = opportunity_band(score)
    shock_normal = (
        asset_shock == ShockLevel.NORMAL
        and market_shock == ShockLevel.NORMAL
    )
    eligible = (
        shock_normal
        and band != OpportunityBand.CASH
    )

    reasons: list[str] = []

    if asset_shock != ShockLevel.NORMAL:
        reasons.append(
            f"asset shock blocks entry: {asset_shock.value}"
        )

    if market_shock != ShockLevel.NORMAL:
        reasons.append(
            f"market shock blocks entry: {market_shock.value}"
        )

    if band == OpportunityBand.CASH:
        reasons.append("opportunity score below 45")

    target_fraction = (
        permitted_target_fraction(score)
        if eligible
        else 0.0
    )

    return OpportunityDecision(
        score=score,
        band=band,
        target_fraction=target_fraction,
        eligible_for_new_entry=eligible,
        market_score=market,
        trend_15m_score=trend_15m,
        trend_1h_score=trend_1h,
        trend_2h_score=trend_2h,
        trend_4h_score=trend_4h,
        relative_strength_score=relative_strength,
        positive_volume_score=positive_volume,
        reasons=tuple(reasons),
    )


def run_self_test() -> None:
    strong = evaluate_opportunity(
        market_score=100.0,
        trend_15m_score=100.0,
        trend_1h_score=100.0,
        trend_2h_score=100.0,
        trend_4h_score=100.0,
        relative_return_60m=2.50,
        return_15m=1.00,
        volume_ratio=1.75,
        asset_shock=ShockLevel.NORMAL,
        market_shock=ShockLevel.NORMAL,
    )

    assert strong.score == 100.0
    assert strong.band == OpportunityBand.FULL_PACE
    assert strong.target_fraction == 1.00
    assert strong.eligible_for_new_entry

    blocked = evaluate_opportunity(
        market_score=100.0,
        trend_15m_score=100.0,
        trend_1h_score=100.0,
        trend_2h_score=100.0,
        trend_4h_score=100.0,
        relative_return_60m=2.50,
        return_15m=1.00,
        volume_ratio=1.75,
        asset_shock=ShockLevel.WARNING,
        market_shock=ShockLevel.NORMAL,
    )

    assert blocked.score == 100.0
    assert blocked.band == OpportunityBand.FULL_PACE
    assert blocked.target_fraction == 0.0
    assert not blocked.eligible_for_new_entry

    weak = evaluate_opportunity(
        market_score=0.0,
        trend_15m_score=0.0,
        trend_1h_score=0.0,
        trend_2h_score=0.0,
        trend_4h_score=0.0,
        relative_return_60m=-2.50,
        return_15m=-1.00,
        volume_ratio=3.00,
        asset_shock=ShockLevel.NORMAL,
        market_shock=ShockLevel.NORMAL,
    )

    assert weak.score == 0.0
    assert weak.band == OpportunityBand.CASH
    assert weak.target_fraction == 0.0
    assert not weak.eligible_for_new_entry

    assert opportunity_band(44.999) == OpportunityBand.CASH
    assert opportunity_band(45.0) == OpportunityBand.PROBE
    assert opportunity_band(65.0) == OpportunityBand.BUILD
    assert opportunity_band(80.0) == OpportunityBand.FULL_PACE

    weight_total = (
        WEIGHT_MARKET
        + WEIGHT_TREND_15M
        + WEIGHT_TREND_1H
        + WEIGHT_TREND_2H
        + WEIGHT_TREND_4H
        + WEIGHT_RELATIVE_STRENGTH
        + WEIGHT_POSITIVE_VOLUME
    )
    assert abs(weight_total - 1.0) < 1e-12

    print("V3KM opportunity self-test: PASS")
    print(f"Strong score: {strong.score:.2f}")
    print(f"Weak score: {weak.score:.2f}")
    print("Shock separation: PASS")
    print("Weight total: PASS")


if __name__ == "__main__":
    run_self_test()
