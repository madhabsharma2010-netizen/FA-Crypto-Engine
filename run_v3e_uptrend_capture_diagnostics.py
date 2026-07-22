from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from core.surveillance_v3d import (
    MarketState,
    ShockLevel,
)
from run_v3d3_portfolio_backtest import (
    BACKTEST_END,
    SYMBOLS,
    build_candidates,
    build_market_states,
    build_shock_tables,
    prepare_frames,
)


DETAIL_FILE = Path(
    "reports/v3e_uptrend_capture_detail.csv"
)

SUMMARY_FILE = Path(
    "reports/v3e_uptrend_capture_summary.csv"
)

BREAKOUT_FILE = Path(
    "reports/v3e_breakout_signal_detail.csv"
)

MISSED_FILE = Path(
    "reports/v3e_missed_uptrend_reasons.csv"
)


FORWARD_HOURS = 24
MAX_DETECTION_DELAY_HOURS = 6
MIN_EPISODE_SEPARATION_HOURS = 12


MIN_CLEAN_MOVE_PERCENT = {
    "BTCUSDT": 1.80,
    "ETHUSDT": 2.50,
    "SOLUSDT": 3.50,
    "XRPUSDT": 3.50,
    "LINKUSDT": 3.50,
    "DOGEUSDT": 4.50,
}


MAX_ADVERSE_PERCENT = {
    "BTCUSDT": 1.40,
    "ETHUSDT": 2.00,
    "SOLUSDT": 2.80,
    "XRPUSDT": 2.80,
    "LINKUSDT": 2.80,
    "DOGEUSDT": 3.50,
}


MAX_BREAKOUT_EXTENSION_PERCENT = {
    "BTCUSDT": 2.20,
    "ETHUSDT": 2.80,
    "SOLUSDT": 4.00,
    "XRPUSDT": 3.75,
    "LINKUSDT": 3.75,
    "DOGEUSDT": 5.00,
}


@dataclass(frozen=True)
class BreakoutSignal:
    symbol: str
    valid: bool
    score: float
    failed_checks: tuple[str, ...]
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class Opportunity:
    symbol: str
    start_time: pd.Timestamp
    target_time: pd.Timestamp
    entry_reference: float
    target_price: float
    move_percent: float
    maximum_adverse_percent: float
    market_state: str
    risk_allowed: bool


def percentage(
    numerator: int,
    denominator: int,
) -> float:
    return (
        numerator
        / denominator
        * 100.0
        if denominator
        else 0.0
    )


def frozen_risk_allows(
    symbol: str,
    state: MarketState,
) -> bool:
    if state in {
        MarketState.STRONG_BULL,
        MarketState.BULL,
    }:
        return True

    if (
        state == MarketState.NEUTRAL
        and symbol
        in {
            "BTCUSDT",
            "ETHUSDT",
        }
    ):
        return True

    return False


def evaluate_breakout_signal(
    symbol: str,
    frame: pd.DataFrame,
    decision_time: pd.Timestamp,
    market_state: MarketState,
    asset_shock_level: ShockLevel,
    relative_strength_percentile: float,
) -> BreakoutSignal:
    position = frame.index.get_loc(
        decision_time
    )

    if not isinstance(position, int) or position < 2:
        return BreakoutSignal(
            symbol=symbol,
            valid=False,
            score=0.0,
            failed_checks=("insufficient_history",),
            reasons=(),
        )

    candle = frame.iloc[position]
    previous = frame.iloc[position - 1]

    close_price = float(candle["close"])
    open_price = float(candle["open"])
    high_price = float(candle["high"])
    low_price = float(candle["low"])
    ema20 = float(candle["EMA20"])
    ema50 = float(candle["EMA50"])
    ema200 = float(candle["EMA200"])
    atr = float(candle["ATR14"])
    rsi = float(candle["RSI14"])
    adx = float(candle["ADX14"])
    previous_adx = float(previous["ADX14"])
    volume_ratio = float(candle["VOLUME_RATIO"])
    momentum_24h = float(candle["MOMENTUM_24H"])
    momentum_72h = float(candle["MOMENTUM_72H"])
    slope = float(candle["EMA50_SLOPE_12H"])

    prior_high_24h = float(
        frame["high"]
        .shift(1)
        .rolling(24)
        .max()
        .iloc[position]
    )

    prior_high_72h = float(
        frame["high"]
        .shift(1)
        .rolling(72)
        .max()
        .iloc[position]
    )

    atr_median_72h = float(
        frame["ATR14"]
        .shift(1)
        .rolling(72)
        .median()
        .iloc[position]
    )

    candle_range = max(
        high_price - low_price,
        1e-12,
    )

    close_location = (
        close_price - low_price
    ) / candle_range

    body_percent = (
        (close_price - open_price)
        / max(open_price, 1e-12)
        * 100.0
    )

    extension_percent = (
        (close_price - ema20)
        / max(ema20, 1e-12)
        * 100.0
    )

    failed: list[str] = []
    reasons: list[str] = []
    score = 0.0

    risk_allowed = frozen_risk_allows(
        symbol,
        market_state,
    )

    if risk_allowed:
        score += 12.0
    else:
        failed.append("market_state")

    shock_safe = (
        asset_shock_level
        == ShockLevel.NORMAL
    )

    if shock_safe:
        score += 8.0
    else:
        failed.append("asset_shock")

    trend_stack = (
        close_price > ema20 > ema50 > ema200
    )

    if trend_stack:
        score += 16.0
        reasons.append("bullish EMA stack")
    else:
        failed.append("trend_stack")

    trend_accelerating = (
        slope > 0
        and momentum_24h > 0
        and momentum_72h > 0
    )

    if trend_accelerating:
        score += 12.0
        reasons.append("positive multi-horizon momentum")
    else:
        failed.append("trend_acceleration")

    breakout_24h = (
        close_price
        >= prior_high_24h * 1.0005
    )

    near_72h_breakout = (
        close_price
        >= prior_high_72h * 0.9975
    )

    if breakout_24h:
        score += 18.0
        reasons.append("24h resistance breakout")
    else:
        failed.append("breakout")

    if near_72h_breakout:
        score += 5.0

    volume_confirmed = (
        volume_ratio >= 1.05
    )

    if volume_confirmed:
        score += 9.0
        reasons.append(
            f"volume ratio {volume_ratio:.2f}"
        )
    else:
        failed.append("volume")

    adx_confirmed = (
        adx >= 20.0
        or (
            adx >= 17.0
            and adx > previous_adx
        )
    )

    if adx_confirmed:
        score += 7.0
    else:
        failed.append("adx")

    candle_confirmed = (
        close_price > open_price
        and close_location >= 0.68
        and body_percent >= 0.10
    )

    if candle_confirmed:
        score += 8.0
        reasons.append("strong breakout close")
    else:
        failed.append("candle_quality")

    not_overextended = (
        extension_percent
        <= MAX_BREAKOUT_EXTENSION_PERCENT[
            symbol
        ]
    )

    if not_overextended:
        score += 5.0
    else:
        failed.append("extension")

    volatility_acceptable = (
        atr_median_72h > 0
        and atr
        <= atr_median_72h * 1.80
    )

    if volatility_acceptable:
        score += 4.0
    else:
        failed.append("volatility")

    relative_strength_ok = (
        relative_strength_percentile
        >= 50.0
    )

    if relative_strength_ok:
        score += 6.0
        reasons.append(
            f"relative strength pctl "
            f"{relative_strength_percentile:.1f}"
        )
    else:
        failed.append("relative_strength")

    rsi_ok = (
        52.0 <= rsi <= 76.0
    )

    if rsi_ok:
        score += 5.0
    else:
        failed.append("rsi")

    score = min(
        score,
        100.0,
    )

    mandatory_checks = {
        "market_state",
        "asset_shock",
        "trend_stack",
        "trend_acceleration",
        "breakout",
        "candle_quality",
        "extension",
    }

    valid = (
        score >= 72.0
        and not any(
            item in mandatory_checks
            for item in failed
        )
    )

    return BreakoutSignal(
        symbol=symbol,
        valid=valid,
        score=score,
        failed_checks=tuple(failed),
        reasons=tuple(reasons),
    )


def relative_strength_table(
    frames_1h: dict[str, pd.DataFrame],
    decision_time: pd.Timestamp,
) -> dict[str, float]:
    values = pd.Series(
        {
            symbol: float(
                frames_1h[
                    symbol
                ].loc[
                    decision_time,
                    "MOMENTUM_24H",
                ]
            )
            for symbol in SYMBOLS
        }
    )

    return (
        values
        .rank(
            pct=True,
            method="average",
        )
        .mul(100.0)
        .to_dict()
    )


def build_breakout_table(
    frames_1h: dict[str, pd.DataFrame],
    common_1h: pd.DatetimeIndex,
    market_states: dict[
        pd.Timestamp,
        Any,
    ],
    asset_shocks: dict[
        str,
        dict[
            pd.Timestamp,
            Any,
        ],
    ],
) -> tuple[
    dict[
        pd.Timestamp,
        dict[
            str,
            BreakoutSignal,
        ],
    ],
    pd.DataFrame,
]:
    shock_series = {}

    for symbol in SYMBOLS:
        shock_series[
            symbol
        ] = pd.Series(
            asset_shocks[
                symbol
            ]
        ).sort_index().reindex(
            common_1h,
            method="ffill",
        )

    result = {}
    rows = []

    for decision_time in common_1h:
        strength = relative_strength_table(
            frames_1h,
            decision_time,
        )

        signals = {}

        for symbol in SYMBOLS:
            signal = evaluate_breakout_signal(
                symbol=symbol,
                frame=frames_1h[symbol],
                decision_time=decision_time,
                market_state=(
                    market_states[
                        decision_time
                    ].state
                ),
                asset_shock_level=(
                    shock_series[
                        symbol
                    ].loc[
                        decision_time
                    ].level
                ),
                relative_strength_percentile=float(
                    strength[symbol]
                ),
            )

            signals[symbol] = signal

            rows.append(
                {
                    "decision_time": (
                        decision_time
                    ),
                    "symbol": symbol,
                    "valid": signal.valid,
                    "score": signal.score,
                    "failed_checks": "|".join(
                        signal.failed_checks
                    ),
                    "reasons": "|".join(
                        signal.reasons
                    ),
                }
            )

        result[decision_time] = signals

    return result, pd.DataFrame(rows)


def find_first_close_target(
    future: pd.DataFrame,
    target_price: float,
) -> pd.Timestamp | None:
    reached = future[
        future["close"] >= target_price
    ]

    if reached.empty:
        return None

    return pd.Timestamp(
        reached.index[0]
    )


def build_opportunities(
    frames_15m: dict[str, pd.DataFrame],
    frames_1h: dict[str, pd.DataFrame],
    common_1h: pd.DatetimeIndex,
    market_states: dict[
        pd.Timestamp,
        Any,
    ],
) -> list[Opportunity]:
    opportunities: list[
        Opportunity
    ] = []

    for symbol in SYMBOLS:
        blocked_until = pd.Timestamp.min
        one_hour = frames_1h[symbol]
        fifteen_minute = frames_15m[
            symbol
        ]

        for index in range(
            len(common_1h) - 1
        ):
            signal_time = (
                common_1h[index]
            )

            if signal_time < blocked_until:
                continue

            next_completion = (
                common_1h[index + 1]
            )

            entry_reference = float(
                one_hour.loc[
                    next_completion,
                    "open",
                ]
            )

            atr = float(
                one_hour.loc[
                    signal_time,
                    "ATR14",
                ]
            )

            atr_move_percent = (
                atr
                * 2.0
                / max(
                    entry_reference,
                    1e-12,
                )
                * 100.0
            )

            required_move_percent = max(
                MIN_CLEAN_MOVE_PERCENT[
                    symbol
                ],
                atr_move_percent,
            )

            target_price = (
                entry_reference
                * (
                    1.0
                    + required_move_percent
                    / 100.0
                )
            )

            end_time = min(
                signal_time
                + pd.Timedelta(
                    hours=FORWARD_HOURS
                ),
                BACKTEST_END,
            )

            future = fifteen_minute[
                (
                    fifteen_minute.index
                    > signal_time
                )
                & (
                    fifteen_minute.index
                    <= end_time
                )
            ]

            if future.empty:
                continue

            target_time = (
                find_first_close_target(
                    future,
                    target_price,
                )
            )

            if target_time is None:
                continue

            path_to_target = future[
                future.index
                <= target_time
            ]

            minimum_low = float(
                path_to_target["low"].min()
            )

            adverse_percent = max(
                0.0,
                (
                    entry_reference
                    - minimum_low
                )
                / max(
                    entry_reference,
                    1e-12,
                )
                * 100.0,
            )

            atr_adverse_limit = (
                atr
                * 1.25
                / max(
                    entry_reference,
                    1e-12,
                )
                * 100.0
            )

            adverse_limit = max(
                MAX_ADVERSE_PERCENT[
                    symbol
                ],
                atr_adverse_limit,
            )

            if adverse_percent > adverse_limit:
                continue

            maximum_close = float(
                future["close"].max()
            )

            move_percent = (
                maximum_close
                - entry_reference
            ) / max(
                entry_reference,
                1e-12,
            ) * 100.0

            state = market_states[
                signal_time
            ].state

            opportunities.append(
                Opportunity(
                    symbol=symbol,
                    start_time=signal_time,
                    target_time=target_time,
                    entry_reference=(
                        entry_reference
                    ),
                    target_price=target_price,
                    move_percent=move_percent,
                    maximum_adverse_percent=(
                        adverse_percent
                    ),
                    market_state=state.value,
                    risk_allowed=(
                        frozen_risk_allows(
                            symbol,
                            state,
                        )
                    ),
                )
            )

            blocked_until = max(
                target_time,
                signal_time
                + pd.Timedelta(
                    hours=(
                        MIN_EPISODE_SEPARATION_HOURS
                    )
                ),
            )

    return opportunities


def detection_for_opportunity(
    opportunity: Opportunity,
    candidates_by_time: dict[
        pd.Timestamp,
        list[Any],
    ],
    breakout_by_time: dict[
        pd.Timestamp,
        dict[
            str,
            BreakoutSignal,
        ],
    ],
    common_1h: pd.DatetimeIndex,
) -> dict[str, Any]:
    deadline = min(
        opportunity.target_time,
        opportunity.start_time
        + pd.Timedelta(
            hours=(
                MAX_DETECTION_DELAY_HOURS
            )
        ),
    )

    detection_times = common_1h[
        (
            common_1h
            >= opportunity.start_time
        )
        & (
            common_1h
            <= deadline
        )
    ]

    first_pullback = None
    first_breakout = None

    start_pullback_failed = ""
    start_breakout_failed = ""

    for decision_time in detection_times:
        pullback_candidates = {
            candidate.symbol: candidate
            for candidate in (
                candidates_by_time.get(
                    decision_time,
                    [],
                )
            )
        }

        pullback = (
            pullback_candidates.get(
                opportunity.symbol
            )
        )

        breakout = (
            breakout_by_time[
                decision_time
            ][
                opportunity.symbol
            ]
        )

        if (
            decision_time
            == opportunity.start_time
        ):
            if pullback is not None:
                start_pullback_failed = (
                    "|".join(
                        pullback.failed_checks
                    )
                )

            start_breakout_failed = (
                "|".join(
                    breakout.failed_checks
                )
            )

        if (
            first_pullback is None
            and pullback is not None
            and pullback.valid
        ):
            first_pullback = decision_time

        if (
            first_breakout is None
            and breakout.valid
        ):
            first_breakout = decision_time

    detected_times = [
        item
        for item in (
            first_pullback,
            first_breakout,
        )
        if item is not None
    ]

    first_detection = (
        min(detected_times)
        if detected_times
        else None
    )

    if (
        first_pullback is not None
        and first_breakout is not None
    ):
        route = (
            "PULLBACK"
            if first_pullback
            < first_breakout
            else "BREAKOUT"
            if first_breakout
            < first_pullback
            else "BOTH"
        )
    elif first_pullback is not None:
        route = "PULLBACK"
    elif first_breakout is not None:
        route = "BREAKOUT"
    else:
        route = "MISSED"

    delay_hours = (
        (
            first_detection
            - opportunity.start_time
        ).total_seconds()
        / 3600.0
        if first_detection is not None
        else None
    )

    return {
        "detected": (
            first_detection
            is not None
        ),
        "route": route,
        "first_detection_time": (
            first_detection
        ),
        "detection_delay_hours": (
            delay_hours
        ),
        "pullback_time": (
            first_pullback
        ),
        "breakout_time": (
            first_breakout
        ),
        "start_pullback_failed": (
            start_pullback_failed
        ),
        "start_breakout_failed": (
            start_breakout_failed
        ),
    }


def summarize(
    detail: pd.DataFrame,
    group_name: str,
    group_value: str,
) -> dict[str, Any]:
    allowed = detail[
        detail["risk_allowed"]
    ]

    detected = allowed[
        allowed["detected"]
    ]

    return {
        group_name: group_value,
        "opportunities": len(detail),
        "risk_allowed": len(allowed),
        "risk_blocked": int(
            (
                ~detail["risk_allowed"]
            ).sum()
        ),
        "detected": len(detected),
        "capture_percent": round(
            percentage(
                len(detected),
                len(allowed),
            ),
            4,
        ),
        "pullback_only": int(
            (
                detected["route"]
                == "PULLBACK"
            ).sum()
        ),
        "breakout_only": int(
            (
                detected["route"]
                == "BREAKOUT"
            ).sum()
        ),
        "both_same_time": int(
            (
                detected["route"]
                == "BOTH"
            ).sum()
        ),
        "missed": int(
            (
                allowed["detected"]
                == False
            ).sum()
        ),
        "median_detection_delay_hours": round(
            float(
                detected[
                    "detection_delay_hours"
                ].median()
            )
            if not detected.empty
            else 0.0,
            4,
        ),
        "median_move_percent": round(
            float(
                allowed[
                    "move_percent"
                ].median()
            )
            if not allowed.empty
            else 0.0,
            4,
        ),
        "median_adverse_percent": round(
            float(
                allowed[
                    "maximum_adverse_percent"
                ].median()
            )
            if not allowed.empty
            else 0.0,
            4,
        ),
    }


def main() -> None:
    print("=" * 132)
    print(
        "FA CRYPTO ENGINE — V3E UPTREND CAPTURE DIAGNOSTICS"
    )
    print("=" * 132)

    (
        frames_15m,
        frames_1h,
        common_15m,
        common_1h,
    ) = prepare_frames()

    print(
        "Building frozen surveillance and market-state tables..."
    )

    (
        asset_shocks,
        market_shocks,
    ) = build_shock_tables(
        frames_15m,
        common_15m,
    )

    market_states = build_market_states(
        frames_1h,
        common_1h,
        market_shocks,
    )

    candidates_by_time = build_candidates(
        frames_1h,
        common_1h,
        market_states,
        asset_shocks,
    )

    print(
        "Building independent breakout-route signals..."
    )

    (
        breakout_by_time,
        breakout_detail,
    ) = build_breakout_table(
        frames_1h,
        common_1h,
        market_states,
        asset_shocks,
    )

    print(
        "Labelling clean 24h uptrend opportunities..."
    )

    opportunities = build_opportunities(
        frames_15m,
        frames_1h,
        common_1h,
        market_states,
    )

    detail_rows = []

    for opportunity in opportunities:
        detection = (
            detection_for_opportunity(
                opportunity,
                candidates_by_time,
                breakout_by_time,
                common_1h,
            )
        )

        detail_rows.append(
            {
                "symbol": (
                    opportunity.symbol
                ),
                "start_time": (
                    opportunity.start_time
                ),
                "target_time": (
                    opportunity.target_time
                ),
                "market_state": (
                    opportunity.market_state
                ),
                "risk_allowed": (
                    opportunity.risk_allowed
                ),
                "entry_reference": (
                    opportunity.entry_reference
                ),
                "target_price": (
                    opportunity.target_price
                ),
                "move_percent": (
                    opportunity.move_percent
                ),
                "maximum_adverse_percent": (
                    opportunity
                    .maximum_adverse_percent
                ),
                **detection,
            }
        )

    detail = pd.DataFrame(
        detail_rows
    )

    DETAIL_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    detail.to_csv(
        DETAIL_FILE,
        index=False,
    )

    breakout_detail.to_csv(
        BREAKOUT_FILE,
        index=False,
    )

    summary_rows = [
        summarize(
            detail,
            "group",
            "OVERALL",
        )
    ]

    for symbol in SYMBOLS:
        summary_rows.append(
            summarize(
                detail[
                    detail["symbol"]
                    == symbol
                ],
                "group",
                symbol,
            )
        )

    summary = pd.DataFrame(
        summary_rows
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    missed = detail[
        detail["risk_allowed"]
        & ~detail["detected"]
    ]

    reason_counter = Counter()

    for value in missed[
        "start_pullback_failed"
    ].fillna(""):
        for reason in str(
            value
        ).split("|"):
            if reason:
                reason_counter[
                    f"pullback:{reason}"
                ] += 1

    for value in missed[
        "start_breakout_failed"
    ].fillna(""):
        for reason in str(
            value
        ).split("|"):
            if reason:
                reason_counter[
                    f"breakout:{reason}"
                ] += 1

    missed_rows = [
        {
            "reason": reason,
            "count": count,
        }
        for reason, count in (
            reason_counter
            .most_common()
        )
    ]

    pd.DataFrame(
        missed_rows
    ).to_csv(
        MISSED_FILE,
        index=False,
    )

    print()
    print("-" * 132)
    print(
        "UPTREND CAPTURE SUMMARY — FROZEN RISK ENGINE"
    )
    print("-" * 132)
    print(
        f"{'GROUP':<14} "
        f"{'OPPS':>7} "
        f"{'ALLOWED':>9} "
        f"{'BLOCKED':>9} "
        f"{'DETECTED':>10} "
        f"{'CAPTURE':>10} "
        f"{'PULLBACK':>10} "
        f"{'BREAKOUT':>10} "
        f"{'MISSED':>8} "
        f"{'MED DELAY':>11} "
        f"{'MED MOVE':>10}"
    )
    print("-" * 132)

    for row in summary.to_dict(
        orient="records"
    ):
        print(
            f"{row['group']:<14} "
            f"{int(row['opportunities']):>7} "
            f"{int(row['risk_allowed']):>9} "
            f"{int(row['risk_blocked']):>9} "
            f"{int(row['detected']):>10} "
            f"{float(row['capture_percent']):>9.2f}% "
            f"{int(row['pullback_only']):>10} "
            f"{int(row['breakout_only']):>10} "
            f"{int(row['missed']):>8} "
            f"{float(row['median_detection_delay_hours']):>10.2f}h "
            f"{float(row['median_move_percent']):>9.2f}%"
        )

    print("-" * 132)
    print(
        "TOP MISSED-UPTREND REASONS"
    )
    print("-" * 132)

    if missed_rows:
        for item in missed_rows[:15]:
            print(
                f"{item['reason']:<36}: "
                f"{item['count']}"
            )
    else:
        print(
            "No risk-allowed clean uptrends were missed."
        )

    latest_time = common_1h[-1]

    latest_pullbacks = {
        candidate.symbol: candidate
        for candidate in (
            candidates_by_time.get(
                latest_time,
                [],
            )
        )
    }

    print("-" * 132)
    print(
        "LATEST DUAL-ROUTE SIGNAL STATUS"
    )
    print("-" * 132)
    print(
        f"{'SYMBOL':<10} "
        f"{'PULLBACK':>12} "
        f"{'PB SCORE':>10} "
        f"{'BREAKOUT':>12} "
        f"{'BO SCORE':>10}"
    )
    print("-" * 132)

    for symbol in SYMBOLS:
        pullback = latest_pullbacks[
            symbol
        ]

        breakout = breakout_by_time[
            latest_time
        ][symbol]

        print(
            f"{symbol:<10} "
            f"{('VALID' if pullback.valid else 'WAIT'):>12} "
            f"{pullback.score:>10.2f} "
            f"{('VALID' if breakout.valid else 'WAIT'):>12} "
            f"{breakout.score:>10.2f}"
        )

    print("=" * 132)
    print(
        f"Opportunity detail : {DETAIL_FILE}"
    )
    print(
        f"Capture summary    : {SUMMARY_FILE}"
    )
    print(
        f"Breakout detail    : {BREAKOUT_FILE}"
    )
    print(
        f"Missed reasons     : {MISSED_FILE}"
    )
    print("=" * 132)


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        print()
        print(
            "V3E diagnostics stopped manually."
        )

    except Exception as error:
        print()
        print("=" * 132)
        print(
            "V3E DIAGNOSTIC ERROR"
        )
        print("=" * 132)
        print(
            f"{type(error).__name__}: "
            f"{error}"
        )
        print("=" * 132)
        raise
