from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from core.surveillance_v3d import (
    MarketState,
    ShockLevel,
)
from run_v3d3_portfolio_backtest import (
    SYMBOLS,
    build_candidates,
    build_market_states,
    build_shock_tables,
    prepare_frames,
)
from run_v3e_uptrend_capture_diagnostics import (
    MAX_DETECTION_DELAY_HOURS,
    build_breakout_table,
    build_opportunities,
    frozen_risk_allows,
)


DETAIL_FILE = Path(
    "reports/v3e2_signal_capture_detail.csv"
)

SUMMARY_FILE = Path(
    "reports/v3e2_signal_capture_summary.csv"
)

SIGNAL_FILE = Path(
    "reports/v3e2_signal_inventory.csv"
)


MAX_EXTENSION_PERCENT = {
    "BTCUSDT": 1.80,
    "ETHUSDT": 2.30,
    "SOLUSDT": 3.20,
    "XRPUSDT": 3.00,
    "LINKUSDT": 3.00,
    "DOGEUSDT": 4.00,
}


@dataclass(frozen=True)
class RouteSignal:
    symbol: str
    route: str
    valid: bool
    score: float
    failed_checks: tuple[str, ...]


def relative_strength_percentiles(
    frames_1h: dict[str, pd.DataFrame],
    decision_time: pd.Timestamp,
) -> dict[str, float]:
    momentum = pd.Series(
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
        momentum
        .rank(
            pct=True,
            method="average",
        )
        .mul(100.0)
        .to_dict()
    )


def risk_state_allowed(
    symbol: str,
    state: MarketState,
) -> bool:
    return frozen_risk_allows(
        symbol,
        state,
    )


def _ensure_route_rolling_features(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    Precompute route rolling features once per dataframe.

    Formulas, shifts and rolling windows are identical to
    the original route calculations. This is a performance
    optimization only.
    """

    if "_ROUTE_LOW_PREV_3" not in frame.columns:
        frame["_ROUTE_LOW_PREV_3"] = (
            frame["low"]
            .shift(1)
            .rolling(3)
            .min()
        )

    if "_ROUTE_LOW_SHIFT4_6" not in frame.columns:
        frame["_ROUTE_LOW_SHIFT4_6"] = (
            frame["low"]
            .shift(4)
            .rolling(6)
            .min()
        )

    if "_ROUTE_LOW_PREV_6" not in frame.columns:
        frame["_ROUTE_LOW_PREV_6"] = (
            frame["low"]
            .shift(1)
            .rolling(6)
            .min()
        )

    if "_ROUTE_HIGH_PREV_8" not in frame.columns:
        frame["_ROUTE_HIGH_PREV_8"] = (
            frame["high"]
            .shift(1)
            .rolling(8)
            .max()
        )

    if "_ROUTE_HIGH_PREV_12" not in frame.columns:
        frame["_ROUTE_HIGH_PREV_12"] = (
            frame["high"]
            .shift(1)
            .rolling(12)
            .max()
        )

    if "_ROUTE_HIGH_PREV_24" not in frame.columns:
        frame["_ROUTE_HIGH_PREV_24"] = (
            frame["high"]
            .shift(1)
            .rolling(24)
            .max()
        )

    if "_ROUTE_ATR_MEDIAN_PREV_48" not in frame.columns:
        frame["_ROUTE_ATR_MEDIAN_PREV_48"] = (
            frame["ATR14"]
            .shift(1)
            .rolling(48)
            .median()
        )

    return frame


def evaluate_ignition_signal(
    symbol: str,
    frame: pd.DataFrame,
    decision_time: pd.Timestamp,
    state: MarketState,
    shock_level: ShockLevel,
    relative_strength: float,
) -> RouteSignal:
    frame = _ensure_route_rolling_features(
        frame
    )

    position = frame.index.get_loc(
        decision_time
    )

    if (
        not isinstance(position, int)
        or position < 12
    ):
        return RouteSignal(
            symbol,
            "IGNITION",
            False,
            0.0,
            ("history",),
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

    previous_close = float(
        previous["close"]
    )
    previous_ema20 = float(
        previous["EMA20"]
    )

    ema20_slope_3h = (
        ema20
        / float(
            frame["EMA20"].iloc[
                position - 3
            ]
        )
        - 1.0
    ) * 100.0

    recent_low_3h = float(
        candle["_ROUTE_LOW_PREV_3"]
    )

    earlier_low_6h = float(
        candle["_ROUTE_LOW_SHIFT4_6"]
    )

    recent_high_8h = float(
        candle["_ROUTE_HIGH_PREV_8"]
    )

    candle_range = max(
        high_price - low_price,
        1e-12,
    )

    close_location = (
        close_price - low_price
    ) / candle_range

    extension_percent = (
        close_price - ema20
    ) / max(
        ema20,
        1e-12,
    ) * 100.0

    failed: list[str] = []
    score = 0.0

    if risk_state_allowed(
        symbol,
        state,
    ):
        score += 15.0
    else:
        failed.append("market_state")

    if shock_level == ShockLevel.NORMAL:
        score += 10.0
    else:
        failed.append("asset_shock")

    base_structure = (
        close_price > ema50
        and close_price > ema200
        and ema20 >= ema50 * 0.997
    )

    if base_structure:
        score += 15.0
    else:
        failed.append("base_structure")

    reclaim = (
        close_price > ema20
        and close_price > open_price
        and close_location >= 0.62
        and (
            previous_close
            <= previous_ema20 * 1.004
            or low_price <= ema20
        )
    )

    if reclaim:
        score += 18.0
    else:
        failed.append("reclaim")

    higher_low = (
        recent_low_3h
        >= earlier_low_6h
        * 0.997
    )

    if higher_low:
        score += 10.0
    else:
        failed.append("higher_low")

    acceleration = (
        ema20_slope_3h > 0
        and momentum_24h > -0.75
        and momentum_72h > -1.50
    )

    if acceleration:
        score += 10.0
    else:
        failed.append("acceleration")

    adx_ok = (
        adx >= 17.0
        or adx > previous_adx
    )

    if adx_ok:
        score += 6.0
    else:
        failed.append("adx")

    if volume_ratio >= 0.75:
        score += 5.0
    else:
        failed.append("volume")

    if 42.0 <= rsi <= 68.0:
        score += 5.0
    else:
        failed.append("rsi")

    if relative_strength >= 50.0:
        score += 4.0
    else:
        failed.append(
            "relative_strength"
        )

    early_break = (
        close_price
        >= recent_high_8h * 0.997
    )

    if early_break:
        score += 4.0

    not_extended = (
        extension_percent
        <= MAX_EXTENSION_PERCENT[
            symbol
        ]
    )

    if not_extended:
        score += 2.0
    else:
        failed.append("extension")

    score = min(score, 100.0)

    mandatory = {
        "market_state",
        "asset_shock",
        "base_structure",
        "reclaim",
        "higher_low",
        "acceleration",
        "extension",
    }

    valid = (
        score >= 68.0
        and not any(
            item in mandatory
            for item in failed
        )
    )

    return RouteSignal(
        symbol=symbol,
        route="IGNITION",
        valid=valid,
        score=score,
        failed_checks=tuple(failed),
    )


def evaluate_adaptive_breakout(
    symbol: str,
    frame: pd.DataFrame,
    decision_time: pd.Timestamp,
    state: MarketState,
    shock_level: ShockLevel,
    relative_strength: float,
) -> RouteSignal:
    frame = _ensure_route_rolling_features(
        frame
    )

    position = frame.index.get_loc(
        decision_time
    )

    if (
        not isinstance(position, int)
        or position < 24
    ):
        return RouteSignal(
            symbol,
            "ADAPTIVE_BREAKOUT",
            False,
            0.0,
            ("history",),
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
    previous_adx = float(
        previous["ADX14"]
    )
    volume_ratio = float(
        candle["VOLUME_RATIO"]
    )
    momentum_24h = float(
        candle["MOMENTUM_24H"]
    )
    momentum_72h = float(
        candle["MOMENTUM_72H"]
    )
    ema50_slope = float(
        candle["EMA50_SLOPE_12H"]
    )

    prior_high_8h = float(
        candle["_ROUTE_HIGH_PREV_8"]
    )

    prior_high_12h = float(
        candle["_ROUTE_HIGH_PREV_12"]
    )

    prior_high_24h = float(
        candle["_ROUTE_HIGH_PREV_24"]
    )

    atr_median_48h = float(
        candle["_ROUTE_ATR_MEDIAN_PREV_48"]
    )

    candle_range = max(
        high_price - low_price,
        1e-12,
    )

    close_location = (
        close_price - low_price
    ) / candle_range

    extension_percent = (
        close_price - ema20
    ) / max(
        ema20,
        1e-12,
    ) * 100.0

    failed: list[str] = []
    score = 0.0

    if risk_state_allowed(
        symbol,
        state,
    ):
        score += 15.0
    else:
        failed.append("market_state")

    if shock_level == ShockLevel.NORMAL:
        score += 10.0
    else:
        failed.append("asset_shock")

    trend_structure = (
        close_price > ema20
        and ema20 > ema50
        and close_price > ema200
        and ema50_slope > 0
    )

    if trend_structure:
        score += 15.0
    else:
        failed.append("trend_structure")

    broke_8h = (
        close_price
        >= prior_high_8h * 1.0003
    )

    broke_12h = (
        close_price
        >= prior_high_12h * 1.0001
    )

    broke_24h = (
        close_price
        >= prior_high_24h * 0.9995
    )

    breakout_ok = (
        broke_8h
        or broke_12h
    )

    if breakout_ok:
        score += 18.0
    else:
        failed.append("breakout")

    if broke_24h:
        score += 5.0

    momentum_ok = (
        momentum_24h > 0
        and momentum_72h > 0
    )

    if momentum_ok:
        score += 10.0
    else:
        failed.append("momentum")

    candle_ok = (
        close_price > open_price
        and close_location >= 0.60
    )

    if candle_ok:
        score += 10.0
    else:
        failed.append("candle_quality")

    adx_rising = (
        adx >= 18.0
        or adx > previous_adx
    )

    if adx_rising:
        score += 6.0
    else:
        failed.append("adx")

    volume_dynamic_ok = (
        volume_ratio >= 0.90
        or (
            volume_ratio >= 0.75
            and relative_strength
            >= 66.0
            and adx > previous_adx
        )
    )

    if volume_dynamic_ok:
        score += 6.0
    else:
        failed.append("volume")

    volatility_ok = (
        atr_median_48h > 0
        and atr
        <= atr_median_48h * 1.90
    )

    if volatility_ok:
        score += 4.0
    else:
        failed.append("volatility")

    if 50.0 <= rsi <= 78.0:
        score += 4.0
    else:
        failed.append("rsi")

    if relative_strength >= 50.0:
        score += 4.0
    else:
        failed.append(
            "relative_strength"
        )

    not_extended = (
        extension_percent
        <= MAX_EXTENSION_PERCENT[
            symbol
        ]
    )

    if not_extended:
        score += 2.0
    else:
        failed.append("extension")

    score = min(score, 100.0)

    mandatory = {
        "market_state",
        "asset_shock",
        "trend_structure",
        "breakout",
        "momentum",
        "candle_quality",
        "extension",
    }

    valid = (
        score >= 70.0
        and not any(
            item in mandatory
            for item in failed
        )
    )

    return RouteSignal(
        symbol=symbol,
        route="ADAPTIVE_BREAKOUT",
        valid=valid,
        score=score,
        failed_checks=tuple(failed),
    )


def build_new_routes(
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
            dict[
                str,
                RouteSignal,
            ],
        ],
    ],
    pd.DataFrame,
]:
    shock_series = {
        symbol: pd.Series(
            asset_shocks[symbol]
        )
        .sort_index()
        .reindex(
            common_1h,
            method="ffill",
        )
        for symbol in SYMBOLS
    }

    table = {}
    rows = []

    for decision_time in common_1h:
        strength = (
            relative_strength_percentiles(
                frames_1h,
                decision_time,
            )
        )

        table[decision_time] = {}

        for symbol in SYMBOLS:
            state = (
                market_states[
                    decision_time
                ].state
            )

            shock_level = (
                shock_series[
                    symbol
                ].loc[
                    decision_time
                ].level
            )

            ignition = (
                evaluate_ignition_signal(
                    symbol,
                    frames_1h[symbol],
                    decision_time,
                    state,
                    shock_level,
                    float(
                        strength[symbol]
                    ),
                )
            )

            breakout = (
                evaluate_adaptive_breakout(
                    symbol,
                    frames_1h[symbol],
                    decision_time,
                    state,
                    shock_level,
                    float(
                        strength[symbol]
                    ),
                )
            )

            table[decision_time][symbol] = {
                "IGNITION": ignition,
                "ADAPTIVE_BREAKOUT": (
                    breakout
                ),
            }

            for signal in (
                ignition,
                breakout,
            ):
                rows.append(
                    {
                        "decision_time": (
                            decision_time
                        ),
                        "symbol": symbol,
                        "route": (
                            signal.route
                        ),
                        "valid": signal.valid,
                        "score": signal.score,
                        "failed_checks": "|".join(
                            signal.failed_checks
                        ),
                    }
                )

    return table, pd.DataFrame(rows)


def first_route_detection(
    symbol: str,
    start_time: pd.Timestamp,
    target_time: pd.Timestamp,
    common_1h: pd.DatetimeIndex,
    current_candidates: dict[
        pd.Timestamp,
        list[Any],
    ],
    current_breakouts: dict[
        pd.Timestamp,
        dict[
            str,
            Any,
        ],
    ],
    new_routes: dict[
        pd.Timestamp,
        dict[
            str,
            dict[
                str,
                RouteSignal,
            ],
        ],
    ],
) -> dict[str, Any]:
    deadline = min(
        target_time,
        start_time
        + pd.Timedelta(
            hours=(
                MAX_DETECTION_DELAY_HOURS
            )
        ),
    )

    times = common_1h[
        (
            common_1h
            >= start_time
        )
        & (
            common_1h
            <= deadline
        )
    ]

    detections = {
        "CURRENT_PULLBACK": None,
        "CURRENT_BREAKOUT": None,
        "IGNITION": None,
        "ADAPTIVE_BREAKOUT": None,
    }

    for decision_time in times:
        pullback_by_symbol = {
            item.symbol: item
            for item in (
                current_candidates.get(
                    decision_time,
                    [],
                )
            )
        }

        pullback = (
            pullback_by_symbol.get(
                symbol
            )
        )

        if (
            detections[
                "CURRENT_PULLBACK"
            ]
            is None
            and pullback is not None
            and pullback.valid
        ):
            detections[
                "CURRENT_PULLBACK"
            ] = decision_time

        old_breakout = (
            current_breakouts[
                decision_time
            ][symbol]
        )

        if (
            detections[
                "CURRENT_BREAKOUT"
            ]
            is None
            and old_breakout.valid
        ):
            detections[
                "CURRENT_BREAKOUT"
            ] = decision_time

        for route in (
            "IGNITION",
            "ADAPTIVE_BREAKOUT",
        ):
            signal = (
                new_routes[
                    decision_time
                ][symbol][route]
            )

            if (
                detections[route]
                is None
                and signal.valid
            ):
                detections[
                    route
                ] = decision_time

    current_times = [
        value
        for key, value
        in detections.items()
        if key.startswith("CURRENT")
        and value is not None
    ]

    revised_times = [
        value
        for value in detections.values()
        if value is not None
    ]

    current_first = (
        min(current_times)
        if current_times
        else None
    )

    revised_first = (
        min(revised_times)
        if revised_times
        else None
    )

    revised_route = "MISSED"

    if revised_first is not None:
        revised_route = "+".join(
            key
            for key, value
            in detections.items()
            if value == revised_first
        )

    return {
        "current_detected": (
            current_first is not None
        ),
        "current_detection_time": (
            current_first
        ),
        "revised_detected": (
            revised_first is not None
        ),
        "revised_detection_time": (
            revised_first
        ),
        "revised_route": (
            revised_route
        ),
        "revised_delay_hours": (
            (
                revised_first
                - start_time
            ).total_seconds()
            / 3600.0
            if revised_first
            is not None
            else None
        ),
        **{
            f"{key.lower()}_time": value
            for key, value
            in detections.items()
        },
    }


def signal_precision(
    signal_inventory: pd.DataFrame,
    detail: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    opportunities = detail[
        detail["risk_allowed"]
    ]

    for (
        symbol,
        route,
    ), group in (
        signal_inventory[
            signal_inventory["valid"]
        ]
        .groupby(
            [
                "symbol",
                "route",
            ]
        )
    ):
        matched = 0

        symbol_opportunities = (
            opportunities[
                opportunities["symbol"]
                == symbol
            ]
        )

        for signal_time in group[
            "decision_time"
        ]:
            belongs = (
                (
                    symbol_opportunities[
                        "start_time"
                    ]
                    <= signal_time
                )
                & (
                    signal_time
                    <= symbol_opportunities[
                        "target_time"
                    ]
                )
            ).any()

            matched += int(belongs)

        total = len(group)

        rows.append(
            {
                "symbol": symbol,
                "route": route,
                "signals": total,
                "matched_signals": matched,
                "precision_percent": (
                    matched
                    / total
                    * 100.0
                    if total
                    else 0.0
                ),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    print("=" * 132)
    print(
        "FA CRYPTO ENGINE — V3E2 MULTI-ROUTE UPTREND CAPTURE CALIBRATION"
    )
    print("=" * 132)

    (
        frames_15m,
        frames_1h,
        common_15m,
        common_1h,
    ) = prepare_frames()

    print(
        "Building frozen V3D3 risk and surveillance tables..."
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

    current_candidates = build_candidates(
        frames_1h,
        common_1h,
        market_states,
        asset_shocks,
    )

    current_breakouts, _ = (
        build_breakout_table(
            frames_1h,
            common_1h,
            market_states,
            asset_shocks,
        )
    )

    print(
        "Building ignition and adaptive-breakout routes..."
    )

    new_routes, inventory = (
        build_new_routes(
            frames_1h,
            common_1h,
            market_states,
            asset_shocks,
        )
    )

    print(
        "Labelling the same clean uptrend opportunities..."
    )

    opportunities = build_opportunities(
        frames_15m,
        frames_1h,
        common_1h,
        market_states,
    )

    rows = []

    for item in opportunities:
        detection = first_route_detection(
            symbol=item.symbol,
            start_time=item.start_time,
            target_time=item.target_time,
            common_1h=common_1h,
            current_candidates=(
                current_candidates
            ),
            current_breakouts=(
                current_breakouts
            ),
            new_routes=new_routes,
        )

        rows.append(
            {
                "symbol": item.symbol,
                "start_time": (
                    item.start_time
                ),
                "target_time": (
                    item.target_time
                ),
                "market_state": (
                    item.market_state
                ),
                "risk_allowed": (
                    item.risk_allowed
                ),
                "move_percent": (
                    item.move_percent
                ),
                "maximum_adverse_percent": (
                    item
                    .maximum_adverse_percent
                ),
                **detection,
            }
        )

    detail = pd.DataFrame(rows)

    DETAIL_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    detail.to_csv(
        DETAIL_FILE,
        index=False,
    )

    inventory.to_csv(
        SIGNAL_FILE,
        index=False,
    )

    allowed = detail[
        detail["risk_allowed"]
    ]

    summary_rows = []

    groups = [
        ("OVERALL", allowed)
    ]

    for symbol in SYMBOLS:
        groups.append(
            (
                symbol,
                allowed[
                    allowed["symbol"]
                    == symbol
                ],
            )
        )

    for name, group in groups:
        current_count = int(
            group[
                "current_detected"
            ].sum()
        )

        revised_count = int(
            group[
                "revised_detected"
            ].sum()
        )

        summary_rows.append(
            {
                "group": name,
                "allowed_opportunities": (
                    len(group)
                ),
                "current_detected": (
                    current_count
                ),
                "current_capture_percent": (
                    current_count
                    / len(group)
                    * 100.0
                    if len(group)
                    else 0.0
                ),
                "revised_detected": (
                    revised_count
                ),
                "revised_capture_percent": (
                    revised_count
                    / len(group)
                    * 100.0
                    if len(group)
                    else 0.0
                ),
                "additional_captures": (
                    revised_count
                    - current_count
                ),
                "median_revised_delay_hours": (
                    float(
                        group[
                            "revised_delay_hours"
                        ].median()
                    )
                    if (
                        not group.empty
                        and group[
                            "revised_detected"
                        ].any()
                    )
                    else 0.0
                ),
            }
        )

    summary = pd.DataFrame(
        summary_rows
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    precision = signal_precision(
        inventory,
        detail,
    )

    print()
    print("-" * 132)
    print(
        "CAPTURE COMPARISON — RISK ENGINE UNCHANGED"
    )
    print("-" * 132)
    print(
        f"{'GROUP':<14} "
        f"{'ALLOWED':>9} "
        f"{'CURRENT':>10} "
        f"{'CURRENT %':>11} "
        f"{'REVISED':>10} "
        f"{'REVISED %':>11} "
        f"{'ADDED':>8} "
        f"{'MED DELAY':>11}"
    )
    print("-" * 132)

    for row in summary.to_dict(
        orient="records"
    ):
        print(
            f"{row['group']:<14} "
            f"{int(row['allowed_opportunities']):>9} "
            f"{int(row['current_detected']):>10} "
            f"{float(row['current_capture_percent']):>10.2f}% "
            f"{int(row['revised_detected']):>10} "
            f"{float(row['revised_capture_percent']):>10.2f}% "
            f"{int(row['additional_captures']):>8} "
            f"{float(row['median_revised_delay_hours']):>10.2f}h"
        )

    print("-" * 132)
    print(
        "NEW-ROUTE SIGNAL PRECISION"
    )
    print("-" * 132)
    print(
        f"{'SYMBOL':<10} "
        f"{'ROUTE':<20} "
        f"{'SIGNALS':>9} "
        f"{'MATCHED':>9} "
        f"{'PRECISION':>11}"
    )
    print("-" * 132)

    if precision.empty:
        print(
            "No revised signals were generated."
        )
    else:
        for row in precision.to_dict(
            orient="records"
        ):
            print(
                f"{row['symbol']:<10} "
                f"{row['route']:<20} "
                f"{int(row['signals']):>9} "
                f"{int(row['matched_signals']):>9} "
                f"{float(row['precision_percent']):>10.2f}%"
            )

    print("-" * 132)
    print(
        "REVISED FIRST-DETECTION ROUTES"
    )
    print("-" * 132)

    route_counts = (
        allowed[
            allowed["revised_detected"]
        ]["revised_route"]
        .value_counts()
    )

    if route_counts.empty:
        print(
            "No uptrend opportunities detected."
        )
    else:
        for route, count in (
            route_counts.items()
        ):
            print(
                f"{route:<54}: "
                f"{int(count)}"
            )

    print("=" * 132)
    print(
        f"Capture detail : {DETAIL_FILE}"
    )
    print(
        f"Capture summary: {SUMMARY_FILE}"
    )
    print(
        f"Signal inventory: {SIGNAL_FILE}"
    )
    print("=" * 132)


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        print()
        print(
            "V3E2 diagnostics stopped manually."
        )

    except Exception as error:
        print()
        print("=" * 132)
        print(
            "V3E2 DIAGNOSTIC ERROR"
        )
        print("=" * 132)
        print(
            f"{type(error).__name__}: "
            f"{error}"
        )
        print("=" * 132)
        raise
