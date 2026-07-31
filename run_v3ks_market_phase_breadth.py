from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import run_v3d_diagnostics as diagnostics

from run_v3km_opportunity_score_audit import (
    WINDOWS,
    configure_window,
    timeframe_trend_score,
)

from run_v3ko_slow_fast_interaction import (
    load_reports,
)

from run_v3kq_acceleration_exhaustion import (
    add_path_features,
)

from run_v3kr_price_overextension import (
    FEATURE_COLUMNS,
    add_diagnostic_states,
    build_raw_features,
)


OUTPUT_DIR = Path("reports")

ACTIVE_BANDS = (
    "BUILD",
    "FULL_PACE",
)

WINDOW_ORDER = {
    "2022": 0,
    "2023": 1,
    "2024": 2,
    "2025H1": 3,
}

BAND_ORDER = {
    "BUILD": 0,
    "FULL_PACE": 1,
}

PATH_ORDER = {
    "FRESH_ACCELERATION": 0,
    "HEALTHY_CONTINUATION": 1,
    "MIXED": 2,
    "LATE_EXHAUSTION": 3,
    "REVERSAL_RISK": 4,
    "INSUFFICIENT_HISTORY": 5,
}

LOCAL_FEATURE_COLUMNS = list(
    FEATURE_COLUMNS
)
MARKET_FEATURE_COLUMNS = [
    "breadth_above_ema20_percent",
    "breadth_above_ema50_percent",
    "breadth_above_ema200_percent",
    "breadth_full_alignment_percent",
    "positive_return_1h_breadth_percent",
    "positive_return_4h_breadth_percent",
    "market_median_return_1h_percent",
    "market_median_return_4h_percent",
    "market_return_dispersion_1h_percent",
    "market_return_dispersion_4h_percent",
    "btc_leadership_1h_percent",
    "btc_leadership_4h_percent",
    "eth_leadership_1h_percent",
    "eth_leadership_4h_percent",
    "btc_vs_eth_4h_percent",
    "btc_trend_15m_score",
    "btc_trend_1h_score",
    "btc_trend_4h_score",
    "btc_trend_mean_score",
    "eth_trend_15m_score",
    "eth_trend_1h_score",
    "eth_trend_4h_score",
    "eth_trend_mean_score",
    "breadth_ema20_delta_1h",
    "breadth_ema20_delta_4h",
    "breadth_ema50_delta_1h",
    "breadth_ema50_delta_4h",
    "breadth_ema200_delta_1h",
    "breadth_ema200_delta_4h",
    "market_score_delta_1h",
    "market_score_delta_4h",
    "market_return_1h_acceleration",
    "btc_leadership_delta_1h",
    "state_age_hours",
]

PHASE_BIN_COLUMNS = [
    "breadth_ema50_bin",
    "breadth_change_4h_bin",
    "market_return_4h_bin",
    "btc_leadership_4h_bin",
    "dispersion_4h_bin",
    "state_age_bin",
    "btc_trend_bin",
    "eth_trend_bin",
    "phase_alignment_state",
]


def positive_rate(
    values: pd.Series,
) -> float:
    if values.empty:
        return float("nan")

    return float(
        values.gt(0.0).mean()
        * 100.0
    )


def exact_return_percent(
    frame: pd.DataFrame,
    lag: int,
    expected_minutes: int,
) -> pd.Series:
    previous_close = (
        frame["close"]
        .shift(lag)
    )

    time_series = pd.Series(
        frame.index,
        index=frame.index,
    )

    previous_time = (
        time_series.shift(lag)
    )

    elapsed = (
        time_series
        - previous_time
    )

    result = (
        frame["close"]
        .div(previous_close)
        .sub(1.0)
        .mul(100.0)
    )

    return result.where(
        elapsed.eq(
            pd.Timedelta(
                minutes=expected_minutes
            )
        )
    )


def exact_series_delta(
    values: pd.Series,
    lag: int,
    expected_minutes: int,
) -> pd.Series:
    previous_value = (
        values.shift(lag)
    )

    time_series = pd.Series(
        values.index,
        index=values.index,
    )

    previous_time = (
        time_series.shift(lag)
    )

    elapsed = (
        time_series
        - previous_time
    )

    return (
        values
        .sub(previous_value)
        .where(
            elapsed.eq(
                pd.Timedelta(
                    minutes=expected_minutes
                )
            )
        )
    )


def trend_score_series(
    frame: pd.DataFrame,
) -> pd.Series:
    return frame.apply(
        lambda row: float(
            timeframe_trend_score(
                row
            )
        ),
        axis=1,
    )


def build_state_age(
    market: pd.DataFrame,
) -> pd.DataFrame:
    result = market.copy()

    time_series = pd.Series(
        result.index,
        index=result.index,
    )

    elapsed = (
        time_series
        - time_series.shift(1)
    )

    exact_next_bar = elapsed.eq(
        pd.Timedelta(
            minutes=15
        )
    )

    previous_state = (
        result["market_state"]
        .shift(1)
    )

    state_changed = (
        result["market_state"]
        .ne(previous_state)
        | ~exact_next_bar
    )

    if not state_changed.empty:
        state_changed.iloc[0] = True

    transition_group = (
        state_changed.cumsum()
    )

    result[
        "previous_market_state"
    ] = previous_state.where(
        exact_next_bar,
        "NONE",
    )

    result[
        "previous_market_state"
    ] = result[
        "previous_market_state"
    ].fillna("NONE")

    result[
        "market_state_changed"
    ] = state_changed

    result[
        "state_age_bars"
    ] = (
        result.groupby(
            transition_group,
            sort=False,
        )
        .cumcount()
    )

    result[
        "state_age_hours"
    ] = (
        result["state_age_bars"]
        * 0.25
    )

    result[
        "state_transition"
    ] = (
        result[
            "previous_market_state"
        ].astype(str)
        + "->"
        + result[
            "market_state"
        ].astype(str)
    )

    return result


def add_market_phase_states(
    data: pd.DataFrame,
) -> pd.DataFrame:
    result = data.copy()

    result["breadth_ema50_bin"] = pd.cut(
        result[
            "breadth_above_ema50_percent"
        ],
        bins=[
            -np.inf,
            25.0,
            50.0,
            75.0,
            np.inf,
        ],
        labels=[
            "VERY_NARROW",
            "NARROW",
            "BROAD",
            "VERY_BROAD",
        ],
        include_lowest=True,
        right=False,
    )

    result["breadth_change_4h_bin"] = pd.cut(
        result[
            "breadth_ema50_delta_4h"
        ],
        bins=[
            -np.inf,
            -16.0,
            -0.1,
            0.1,
            16.0,
            np.inf,
        ],
        labels=[
            "COLLAPSING",
            "WEAKENING",
            "FLAT",
            "IMPROVING",
            "SURGING",
        ],
        include_lowest=True,
        right=False,
    )

    result["market_return_4h_bin"] = pd.cut(
        result[
            "market_median_return_4h_percent"
        ],
        bins=[
            -np.inf,
            -2.0,
            -0.5,
            0.5,
            2.0,
            np.inf,
        ],
        labels=[
            "SHARP_FALL",
            "FALLING",
            "FLAT",
            "RISING",
            "SHARP_RISE",
        ],
        include_lowest=True,
        right=False,
    )

    result["btc_leadership_4h_bin"] = pd.cut(
        result[
            "btc_leadership_4h_percent"
        ],
        bins=[
            -np.inf,
            -2.0,
            -0.5,
            0.5,
            2.0,
            np.inf,
        ],
        labels=[
            "ALT_STRONGLY_LED",
            "ALT_LED",
            "BALANCED",
            "BTC_LED",
            "BTC_STRONGLY_LED",
        ],
        include_lowest=True,
        right=False,
    )

    result["dispersion_4h_bin"] = pd.cut(
        result[
            "market_return_dispersion_4h_percent"
        ],
        bins=[
            -np.inf,
            0.50,
            1.50,
            3.00,
            np.inf,
        ],
        labels=[
            "LOW",
            "NORMAL",
            "HIGH",
            "EXTREME",
        ],
        include_lowest=True,
        right=False,
    )

    result["state_age_bin"] = pd.cut(
        result[
            "state_age_hours"
        ],
        bins=[
            -np.inf,
            1.0,
            6.0,
            24.0,
            np.inf,
        ],
        labels=[
            "NEW_UNDER_1H",
            "EARLY_1_TO_6H",
            "ESTABLISHED_6_TO_24H",
            "MATURE_24H_PLUS",
        ],
        include_lowest=True,
        right=False,
    )

    result["btc_trend_bin"] = pd.cut(
        result[
            "btc_trend_mean_score"
        ],
        bins=[
            -np.inf,
            45.0,
            65.0,
            80.0,
            np.inf,
        ],
        labels=[
            "WEAK",
            "NEUTRAL",
            "BULL",
            "STRONG_BULL",
        ],
        include_lowest=True,
        right=False,
    )

    result["eth_trend_bin"] = pd.cut(
        result[
            "eth_trend_mean_score"
        ],
        bins=[
            -np.inf,
            45.0,
            65.0,
            80.0,
            np.inf,
        ],
        labels=[
            "WEAK",
            "NEUTRAL",
            "BULL",
            "STRONG_BULL",
        ],
        include_lowest=True,
        right=False,
    )

    broad_deterioration = (
        result[
            "breadth_ema50_delta_4h"
        ].le(-16.0)
        & result[
            "market_median_return_4h_percent"
        ].lt(0.0)
    )

    narrow_btc_led = (
        result[
            "btc_leadership_4h_percent"
        ].ge(1.0)
        & result[
            "breadth_above_ema50_percent"
        ].lt(50.0)
    )

    broad_risk_on = (
        result[
            "breadth_above_ema50_percent"
        ].ge(66.0)
        & result[
            "market_median_return_4h_percent"
        ].ge(0.50)
        & result[
            "btc_trend_mean_score"
        ].ge(65.0)
        & result[
            "eth_trend_mean_score"
        ].ge(65.0)
    )

    broad_recovery = (
        result[
            "breadth_ema50_delta_4h"
        ].ge(16.0)
        & result[
            "market_median_return_4h_percent"
        ].ge(0.0)
    )

    broad_risk_off = (
        result[
            "breadth_above_ema50_percent"
        ].le(33.34)
        & result[
            "market_median_return_4h_percent"
        ].le(-0.50)
    )

    result["phase_alignment_state"] = (
        np.select(
            [
                broad_deterioration,
                narrow_btc_led,
                broad_risk_on,
                broad_recovery,
                broad_risk_off,
            ],
            [
                "BROAD_DETERIORATION",
                "NARROW_BTC_LED",
                "BROAD_RISK_ON",
                "BROAD_RECOVERY",
                "BROAD_RISK_OFF",
            ],
            default="MIXED",
        )
    )

    return result


def build_window_snapshots(
    window: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    window_detail: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    print("")
    print(
        "Preparing V3KS market phase "
        f"for {window}: "
        f"{start} -> {end}"
    )

    configure_window(
        start,
        end,
    )

    frames_15m = {
        symbol: diagnostics.prepare_15m(
            symbol
        )
        for symbol in diagnostics.SYMBOLS
    }

    decision_times = pd.DatetimeIndex(
        sorted(
            window_detail[
                "decision_time"
            ].unique()
        )
    )

    if decision_times.has_duplicates:
        raise RuntimeError(
            f"DUPLICATE_DECISION_TIMES: "
            f"{window}"
        )

    local_parts: list[pd.DataFrame] = []
    integrity_rows: list[dict[str, object]] = []

    above_ema20: dict[str, pd.Series] = {}
    above_ema50: dict[str, pd.Series] = {}
    above_ema200: dict[str, pd.Series] = {}
    full_alignment: dict[str, pd.Series] = {}

    returns_1h: dict[str, pd.Series] = {}
    returns_4h: dict[str, pd.Series] = {}

    for symbol in diagnostics.SYMBOLS:
        frame = frames_15m[symbol]

        missing_times = (
            decision_times.difference(
                frame.index
            )
        )

        if len(missing_times) != 0:
            raise RuntimeError(
                f"MISSING_RAW_TIMES: "
                f"{window} {symbol} "
                f"{len(missing_times)}"
            )

        current = frame.reindex(
            decision_times
        )

        above_ema20[symbol] = (
            current["close"]
            .gt(current["EMA20"])
        )

        above_ema50[symbol] = (
            current["close"]
            .gt(current["EMA50"])
        )

        above_ema200[symbol] = (
            current["close"]
            .gt(current["EMA200"])
        )

        full_alignment[symbol] = (
            current["close"]
            .gt(current["EMA20"])
            & current["EMA20"]
            .gt(current["EMA50"])
            & current["EMA50"]
            .gt(current["EMA200"])
        )

        returns_1h[symbol] = (
            exact_return_percent(
                frame,
                lag=4,
                expected_minutes=60,
            )
            .reindex(decision_times)
        )

        returns_4h[symbol] = (
            exact_return_percent(
                frame,
                lag=16,
                expected_minutes=240,
            )
            .reindex(decision_times)
        )

        raw_features = (
            build_raw_features(
                frame
            )
            .reindex(
                decision_times
            )
        )

        selected_local = (
            raw_features.loc[
                :,
                LOCAL_FEATURE_COLUMNS,
            ]
            .copy()
        )

        missing_local = int(
            selected_local
            .isna()
            .any(axis=1)
            .sum()
        )

        selected_local.index.name = (
            "decision_time"
        )

        selected_local = (
            selected_local
            .reset_index()
        )

        selected_local["window"] = window
        selected_local["symbol"] = symbol

        selected_local = selected_local[
            [
                "window",
                "symbol",
                "decision_time",
                *LOCAL_FEATURE_COLUMNS,
            ]
        ]

        local_parts.append(
            selected_local
        )

        detail_symbol_rows = int(
            window_detail[
                "symbol"
            ]
            .eq(symbol)
            .sum()
        )

        integrity_rows.append(
            {
                "window": window,
                "symbol": symbol,
                "decision_times": int(
                    len(decision_times)
                ),
                "detail_symbol_rows": (
                    detail_symbol_rows
                ),
                "local_snapshot_rows": int(
                    len(selected_local)
                ),
                "missing_raw_times": int(
                    len(missing_times)
                ),
                "rows_with_missing_local_feature": (
                    missing_local
                ),
                "first_decision_time": (
                    decision_times.min()
                ),
                "last_decision_time": (
                    decision_times.max()
                ),
            }
        )

        print(
            f"  {symbol}: "
            f"{len(selected_local):,} "
            "local rows"
        )

    above_ema20_frame = pd.DataFrame(
        above_ema20
    ).astype(float)

    above_ema50_frame = pd.DataFrame(
        above_ema50
    ).astype(float)

    above_ema200_frame = pd.DataFrame(
        above_ema200
    ).astype(float)

    full_alignment_frame = pd.DataFrame(
        full_alignment
    ).astype(float)

    return_1h_frame = pd.DataFrame(
        returns_1h
    )

    return_4h_frame = pd.DataFrame(
        returns_4h
    )

    market = pd.DataFrame(
        index=decision_times
    )

    market.index.name = "decision_time"

    market[
        "breadth_above_ema20_percent"
    ] = (
        above_ema20_frame.mean(
            axis=1
        )
        * 100.0
    )

    market[
        "breadth_above_ema50_percent"
    ] = (
        above_ema50_frame.mean(
            axis=1
        )
        * 100.0
    )

    market[
        "breadth_above_ema200_percent"
    ] = (
        above_ema200_frame.mean(
            axis=1
        )
        * 100.0
    )

    market[
        "breadth_full_alignment_percent"
    ] = (
        full_alignment_frame.mean(
            axis=1
        )
        * 100.0
    )

    market[
        "positive_return_1h_breadth_percent"
    ] = (
        return_1h_frame.gt(0.0)
        .mean(axis=1)
        * 100.0
    )

    market[
        "positive_return_4h_breadth_percent"
    ] = (
        return_4h_frame.gt(0.0)
        .mean(axis=1)
        * 100.0
    )

    market[
        "market_median_return_1h_percent"
    ] = return_1h_frame.median(
        axis=1
    )

    market[
        "market_median_return_4h_percent"
    ] = return_4h_frame.median(
        axis=1
    )

    market[
        "market_return_dispersion_1h_percent"
    ] = return_1h_frame.std(
        axis=1,
        ddof=0,
    )

    market[
        "market_return_dispersion_4h_percent"
    ] = return_4h_frame.std(
        axis=1,
        ddof=0,
    )

    alt_symbols = [
        symbol
        for symbol in diagnostics.SYMBOLS
        if symbol != "BTCUSDT"
    ]

    non_btc_eth_symbols = [
        symbol
        for symbol in diagnostics.SYMBOLS
        if symbol not in (
            "BTCUSDT",
            "ETHUSDT",
        )
    ]

    market[
        "btc_leadership_1h_percent"
    ] = (
        return_1h_frame["BTCUSDT"]
        - return_1h_frame[
            alt_symbols
        ].median(axis=1)
    )

    market[
        "btc_leadership_4h_percent"
    ] = (
        return_4h_frame["BTCUSDT"]
        - return_4h_frame[
            alt_symbols
        ].median(axis=1)
    )

    market[
        "eth_leadership_1h_percent"
    ] = (
        return_1h_frame["ETHUSDT"]
        - return_1h_frame[
            non_btc_eth_symbols
        ].median(axis=1)
    )

    market[
        "eth_leadership_4h_percent"
    ] = (
        return_4h_frame["ETHUSDT"]
        - return_4h_frame[
            non_btc_eth_symbols
        ].median(axis=1)
    )

    market[
        "btc_vs_eth_4h_percent"
    ] = (
        return_4h_frame["BTCUSDT"]
        - return_4h_frame["ETHUSDT"]
    )

    btc_15m = (
        frames_15m["BTCUSDT"]
        .reindex(decision_times)
    )

    eth_15m = (
        frames_15m["ETHUSDT"]
        .reindex(decision_times)
    )

    btc_1h = diagnostics.align_frame(
        diagnostics.load_frame(
            "BTCUSDT",
            "1h",
        ),
        decision_times,
    )

    btc_4h = diagnostics.align_frame(
        diagnostics.load_frame(
            "BTCUSDT",
            "4h",
        ),
        decision_times,
    )

    eth_1h = diagnostics.align_frame(
        diagnostics.load_frame(
            "ETHUSDT",
            "1h",
        ),
        decision_times,
    )

    eth_4h = diagnostics.align_frame(
        diagnostics.load_frame(
            "ETHUSDT",
            "4h",
        ),
        decision_times,
    )

    market[
        "btc_trend_15m_score"
    ] = trend_score_series(
        btc_15m
    )

    market[
        "btc_trend_1h_score"
    ] = trend_score_series(
        btc_1h
    )

    market[
        "btc_trend_4h_score"
    ] = trend_score_series(
        btc_4h
    )

    market[
        "btc_trend_mean_score"
    ] = market[
        [
            "btc_trend_15m_score",
            "btc_trend_1h_score",
            "btc_trend_4h_score",
        ]
    ].mean(axis=1)

    market[
        "eth_trend_15m_score"
    ] = trend_score_series(
        eth_15m
    )

    market[
        "eth_trend_1h_score"
    ] = trend_score_series(
        eth_1h
    )

    market[
        "eth_trend_4h_score"
    ] = trend_score_series(
        eth_4h
    )

    market[
        "eth_trend_mean_score"
    ] = market[
        [
            "eth_trend_15m_score",
            "eth_trend_1h_score",
            "eth_trend_4h_score",
        ]
    ].mean(axis=1)

    state_conflicts = (
        window_detail.groupby(
            "decision_time",
            sort=False,
        )[
            [
                "market_state",
                "market_score",
            ]
        ]
        .nunique(
            dropna=False
        )
    )

    conflict_rows = int(
        state_conflicts.gt(1)
        .any(axis=1)
        .sum()
    )

    if conflict_rows != 0:
        raise RuntimeError(
            f"MARKET_STATE_CONFLICTS: "
            f"{window} {conflict_rows}"
        )

    state_table = (
        window_detail[
            [
                "decision_time",
                "market_state",
                "market_score",
            ]
        ]
        .drop_duplicates(
            subset=[
                "decision_time",
            ]
        )
        .set_index(
            "decision_time"
        )
        .sort_index()
        .reindex(
            decision_times
        )
    )

    if state_table.isna().any().any():
        raise RuntimeError(
            f"MISSING_MARKET_STATE_ROWS: "
            f"{window}"
        )

    market = market.join(
        state_table,
        how="left",
    )

    market = build_state_age(
        market
    )

    market[
        "breadth_ema20_delta_1h"
    ] = exact_series_delta(
        market[
            "breadth_above_ema20_percent"
        ],
        lag=4,
        expected_minutes=60,
    )

    market[
        "breadth_ema20_delta_4h"
    ] = exact_series_delta(
        market[
            "breadth_above_ema20_percent"
        ],
        lag=16,
        expected_minutes=240,
    )

    market[
        "breadth_ema50_delta_1h"
    ] = exact_series_delta(
        market[
            "breadth_above_ema50_percent"
        ],
        lag=4,
        expected_minutes=60,
    )

    market[
        "breadth_ema50_delta_4h"
    ] = exact_series_delta(
        market[
            "breadth_above_ema50_percent"
        ],
        lag=16,
        expected_minutes=240,
    )

    market[
        "breadth_ema200_delta_1h"
    ] = exact_series_delta(
        market[
            "breadth_above_ema200_percent"
        ],
        lag=4,
        expected_minutes=60,
    )

    market[
        "breadth_ema200_delta_4h"
    ] = exact_series_delta(
        market[
            "breadth_above_ema200_percent"
        ],
        lag=16,
        expected_minutes=240,
    )

    market[
        "market_score_delta_1h"
    ] = exact_series_delta(
        market["market_score"],
        lag=4,
        expected_minutes=60,
    )

    market[
        "market_score_delta_4h"
    ] = exact_series_delta(
        market["market_score"],
        lag=16,
        expected_minutes=240,
    )

    market[
        "market_return_1h_acceleration"
    ] = exact_series_delta(
        market[
            "market_median_return_1h_percent"
        ],
        lag=4,
        expected_minutes=60,
    )

    market[
        "btc_leadership_delta_1h"
    ] = exact_series_delta(
        market[
            "btc_leadership_1h_percent"
        ],
        lag=4,
        expected_minutes=60,
    )

    market = add_market_phase_states(
        market
    )

    market["window"] = window

    market = (
        market.reset_index()
    )

    market_duplicate_keys = int(
        market.duplicated(
            subset=[
                "window",
                "decision_time",
            ],
            keep=False,
        ).sum()
    )

    if market_duplicate_keys != 0:
        raise RuntimeError(
            f"MARKET_DUPLICATE_KEYS: "
            f"{window} "
            f"{market_duplicate_keys}"
        )

    local_snapshot = pd.concat(
        local_parts,
        ignore_index=True,
    )

    integrity = pd.DataFrame(
        integrity_rows
    )

    integrity[
        "market_snapshot_rows"
    ] = int(
        len(market)
    )

    integrity[
        "market_state_conflicts"
    ] = conflict_rows

    integrity[
        "market_rows_with_any_missing_feature"
    ] = int(
        market.loc[
            :,
            MARKET_FEATURE_COLUMNS,
        ]
        .isna()
        .any(axis=1)
        .sum()
    )

    print(
        "  Market decision rows: "
        f"{len(market):,}"
    )

    return (
        local_snapshot,
        market,
        integrity,
    )


def aggregate_outcomes(
    data: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    return (
        data.groupby(
            group_columns,
            observed=True,
            dropna=False,
            sort=False,
        )
        .agg(
            observations=(
                "net_return_24h_percent",
                "size",
            ),
            mean_score=(
                "score",
                "mean",
            ),
            mean_fast_score=(
                "fast_score",
                "mean",
            ),
            mean_breadth_ema20=(
                "breadth_above_ema20_percent",
                "mean",
            ),
            mean_breadth_ema50=(
                "breadth_above_ema50_percent",
                "mean",
            ),
            mean_breadth_ema200=(
                "breadth_above_ema200_percent",
                "mean",
            ),
            mean_breadth_alignment=(
                "breadth_full_alignment_percent",
                "mean",
            ),
            mean_breadth_ema50_delta_1h=(
                "breadth_ema50_delta_1h",
                "mean",
            ),
            mean_breadth_ema50_delta_4h=(
                "breadth_ema50_delta_4h",
                "mean",
            ),
            mean_market_return_1h=(
                "market_median_return_1h_percent",
                "mean",
            ),
            mean_market_return_4h=(
                "market_median_return_4h_percent",
                "mean",
            ),
            mean_dispersion_4h=(
                "market_return_dispersion_4h_percent",
                "mean",
            ),
            mean_btc_leadership_4h=(
                "btc_leadership_4h_percent",
                "mean",
            ),
            mean_eth_leadership_4h=(
                "eth_leadership_4h_percent",
                "mean",
            ),
            mean_btc_trend=(
                "btc_trend_mean_score",
                "mean",
            ),
            mean_eth_trend=(
                "eth_trend_mean_score",
                "mean",
            ),
            mean_market_score_delta_1h=(
                "market_score_delta_1h",
                "mean",
            ),
            mean_state_age_hours=(
                "state_age_hours",
                "mean",
            ),
            mean_ema20_distance_atr=(
                "ema20_distance_atr",
                "mean",
            ),
            mean_close_location=(
                "close_location",
                "mean",
            ),
            mean_upper_wick_ratio=(
                "upper_wick_ratio",
                "mean",
            ),
            mean_volume_ratio=(
                "volume_ratio_prior20",
                "mean",
            ),
            mean_atr_expansion=(
                "atr_expansion_prior48",
                "mean",
            ),
            mean_net_4h_percent=(
                "net_return_4h_percent",
                "mean",
            ),
            mean_net_12h_percent=(
                "net_return_12h_percent",
                "mean",
            ),
            mean_net_24h_percent=(
                "net_return_24h_percent",
                "mean",
            ),
            median_net_24h_percent=(
                "net_return_24h_percent",
                "median",
            ),
            positive_net_24h_rate_percent=(
                "net_return_24h_percent",
                positive_rate,
            ),
            loss_below_minus_1_rate_percent=(
                "net_return_24h_percent",
                lambda values: float(
                    values.le(-1.0)
                    .mean()
                    * 100.0
                ),
            ),
            mean_mfe_24h_percent=(
                "mfe_24h_percent",
                "mean",
            ),
            mean_mae_24h_percent=(
                "mae_24h_percent",
                "mean",
            ),
        )
        .reset_index()
    )


def build_phase_bin_summary(
    data: pd.DataFrame,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []

    for phase_column in (
        PHASE_BIN_COLUMNS
    ):
        source = data.dropna(
            subset=[
                phase_column,
            ]
        )

        summary = aggregate_outcomes(
            source,
            [
                "window",
                "band",
                "path_state",
                phase_column,
            ],
        )

        summary = summary.rename(
            columns={
                phase_column:
                "phase_bin",
            }
        )

        summary.insert(
            3,
            "phase_feature",
            phase_column,
        )

        parts.append(
            summary
        )

    return pd.concat(
        parts,
        ignore_index=True,
    )


def print_table(
    title: str,
    table: pd.DataFrame,
    columns: list[str],
) -> None:
    print("")
    print(title)

    if table.empty:
        print("NO_ROWS")
        return

    print(
        table[
            columns
        ].to_string(
            index=False,
            float_format=(
                lambda value:
                f"{value:.5f}"
            ),
        )
    )


def main() -> None:
    detail = load_reports()

    detail["window"] = (
        detail["window"]
        .astype(str)
    )

    detail["decision_time"] = (
        pd.to_datetime(
            detail["decision_time"]
        )
    )

    detail = add_path_features(
        detail
    )

    local_parts: list[pd.DataFrame] = []
    market_parts: list[pd.DataFrame] = []
    integrity_parts: list[pd.DataFrame] = []

    for (
        window,
        (
            start,
            end,
        ),
    ) in WINDOWS.items():
        window_detail = detail.loc[
            detail["window"].eq(
                window
            )
        ].copy()

        if window_detail.empty:
            raise RuntimeError(
                f"NO_DETAIL_ROWS: "
                f"{window}"
            )

        (
            local_snapshot,
            market_snapshot,
            integrity,
        ) = build_window_snapshots(
            window,
            start,
            end,
            window_detail,
        )

        local_parts.append(
            local_snapshot
        )

        market_parts.append(
            market_snapshot
        )

        integrity_parts.append(
            integrity
        )

    local_snapshot = pd.concat(
        local_parts,
        ignore_index=True,
    )

    market_snapshot = pd.concat(
        market_parts,
        ignore_index=True,
    )

    integrity = pd.concat(
        integrity_parts,
        ignore_index=True,
    )

    local_duplicate_keys = int(
        local_snapshot.duplicated(
            subset=[
                "window",
                "symbol",
                "decision_time",
            ],
            keep=False,
        ).sum()
    )

    market_duplicate_keys = int(
        market_snapshot.duplicated(
            subset=[
                "window",
                "decision_time",
            ],
            keep=False,
        ).sum()
    )

    if local_duplicate_keys != 0:
        raise RuntimeError(
            "LOCAL_DUPLICATE_KEYS: "
            f"{local_duplicate_keys}"
        )

    if market_duplicate_keys != 0:
        raise RuntimeError(
            "MARKET_DUPLICATE_KEYS: "
            f"{market_duplicate_keys}"
        )

    merged = detail.merge(
        local_snapshot,
        on=[
            "window",
            "symbol",
            "decision_time",
        ],
        how="left",
        validate="one_to_one",
    )

    merged = merged.merge(
        market_snapshot,
        on=[
            "window",
            "decision_time",
        ],
        how="left",
        validate="many_to_one",
    )

    if len(merged) != len(detail):
        raise RuntimeError(
            "ROW_COUNT_CHANGED_AFTER_MERGE"
        )

    fully_missing_local = int(
        merged.loc[
            :,
            LOCAL_FEATURE_COLUMNS,
        ]
        .isna()
        .all(axis=1)
        .sum()
    )

    fully_missing_market = int(
        merged.loc[
            :,
            MARKET_FEATURE_COLUMNS,
        ]
        .isna()
        .all(axis=1)
        .sum()
    )

    if fully_missing_local != 0:
        raise RuntimeError(
            "FULLY_MISSING_LOCAL_ROWS: "
            f"{fully_missing_local}"
        )

    if fully_missing_market != 0:
        raise RuntimeError(
            "FULLY_MISSING_MARKET_ROWS: "
            f"{fully_missing_market}"
        )

    merged = add_diagnostic_states(
        merged
    )

    active = merged.loc[
        merged["shock_normal"]
        & merged["band"].isin(
            ACTIVE_BANDS
        )
    ].copy()

    late = active.loc[
        active["path_state"].eq(
            "LATE_EXHAUSTION"
        )
    ].copy()

    fresh = active.loc[
        active["path_state"].eq(
            "FRESH_ACCELERATION"
        )
    ].copy()

    path_phase_summary = (
        aggregate_outcomes(
            active,
            [
                "window",
                "band",
                "path_state",
            ],
        )
    )

    phase_bin_summary = (
        build_phase_bin_summary(
            active
        )
    )

    late_phase_summary = (
        aggregate_outcomes(
            late,
            [
                "window",
                "band",
                "phase_alignment_state",
            ],
        )
    )

    fresh_phase_summary = (
        aggregate_outcomes(
            fresh,
            [
                "window",
                "band",
                "phase_alignment_state",
            ],
        )
    )

    transition_summary = (
        aggregate_outcomes(
            active.loc[
                active[
                    "path_state"
                ].isin(
                    [
                        "LATE_EXHAUSTION",
                        "FRESH_ACCELERATION",
                    ]
                )
            ],
            [
                "window",
                "band",
                "path_state",
                "state_transition",
                "state_age_bin",
            ],
        )
    )

    local_market_interactions = (
        aggregate_outcomes(
            late,
            [
                "window",
                "band",
                "phase_alignment_state",
                "extension_state",
                "candle_state",
            ],
        )
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    integrity.to_csv(
        OUTPUT_DIR
        / "v3ks_market_join_integrity.csv",
        index=False,
    )

    active.to_csv(
        OUTPUT_DIR
        / "v3ks_active_market_snapshot.csv",
        index=False,
    )

    path_phase_summary.to_csv(
        OUTPUT_DIR
        / "v3ks_path_phase_summary.csv",
        index=False,
    )

    phase_bin_summary.to_csv(
        OUTPUT_DIR
        / "v3ks_phase_bin_summary.csv",
        index=False,
    )

    late_phase_summary.to_csv(
        OUTPUT_DIR
        / "v3ks_late_exhaustion_phase_summary.csv",
        index=False,
    )

    fresh_phase_summary.to_csv(
        OUTPUT_DIR
        / "v3ks_fresh_acceleration_phase_summary.csv",
        index=False,
    )

    transition_summary.to_csv(
        OUTPUT_DIR
        / "v3ks_state_transition_summary.csv",
        index=False,
    )

    local_market_interactions.to_csv(
        OUTPUT_DIR
        / "v3ks_local_market_interactions.csv",
        index=False,
    )

    integrity["_window_order"] = (
        integrity["window"]
        .map(WINDOW_ORDER)
    )

    integrity = integrity.sort_values(
        [
            "_window_order",
            "symbol",
        ]
    )

    path_focus = (
        path_phase_summary.loc[
            path_phase_summary[
                "window"
            ].isin(
                [
                    "2024",
                    "2025H1",
                ]
            )
        ]
        .copy()
    )

    path_focus["_window_order"] = (
        path_focus["window"]
        .map(WINDOW_ORDER)
    )

    path_focus["_band_order"] = (
        path_focus["band"]
        .map(BAND_ORDER)
    )

    path_focus["_path_order"] = (
        path_focus["path_state"]
        .map(PATH_ORDER)
    )

    path_focus = path_focus.sort_values(
        [
            "_window_order",
            "_band_order",
            "_path_order",
        ]
    )

    late_phase_summary[
        "_window_order"
    ] = (
        late_phase_summary["window"]
        .map(WINDOW_ORDER)
    )

    late_phase_summary[
        "_band_order"
    ] = (
        late_phase_summary["band"]
        .map(BAND_ORDER)
    )

    late_phase_summary = (
        late_phase_summary.sort_values(
            [
                "_window_order",
                "_band_order",
                "mean_net_24h_percent",
            ]
        )
    )

    fresh_focus = (
        fresh_phase_summary.loc[
            fresh_phase_summary[
                "window"
            ].isin(
                [
                    "2024",
                    "2025H1",
                ]
            )
            & fresh_phase_summary[
                "observations"
            ].ge(20)
        ]
        .copy()
    )

    fresh_focus["_window_order"] = (
        fresh_focus["window"]
        .map(WINDOW_ORDER)
    )

    fresh_focus["_band_order"] = (
        fresh_focus["band"]
        .map(BAND_ORDER)
    )

    fresh_focus = fresh_focus.sort_values(
        [
            "_window_order",
            "_band_order",
            "mean_net_24h_percent",
        ]
    )

    transition_focus = (
        transition_summary.loc[
            transition_summary[
                "window"
            ].isin(
                [
                    "2024",
                    "2025H1",
                ]
            )
            & transition_summary[
                "observations"
            ].ge(20)
        ]
        .copy()
    )

    transition_focus[
        "_window_order"
    ] = (
        transition_focus["window"]
        .map(WINDOW_ORDER)
    )

    transition_focus[
        "_band_order"
    ] = (
        transition_focus["band"]
        .map(BAND_ORDER)
    )

    transition_focus = (
        transition_focus.sort_values(
            [
                "_window_order",
                "_band_order",
                "path_state",
                "mean_net_24h_percent",
            ]
        )
    )

    interaction_focus = (
        local_market_interactions.loc[
            local_market_interactions[
                "window"
            ].isin(
                [
                    "2024",
                    "2025H1",
                ]
            )
            & local_market_interactions[
                "observations"
            ].ge(20)
        ]
        .copy()
    )

    interaction_focus[
        "_window_order"
    ] = (
        interaction_focus["window"]
        .map(WINDOW_ORDER)
    )

    interaction_focus[
        "_band_order"
    ] = (
        interaction_focus["band"]
        .map(BAND_ORDER)
    )

    interaction_focus = (
        interaction_focus.sort_values(
            [
                "_window_order",
                "_band_order",
                "mean_net_24h_percent",
            ]
        )
    )

    bin_focus = (
        phase_bin_summary.loc[
            phase_bin_summary[
                "window"
            ].isin(
                [
                    "2024",
                    "2025H1",
                ]
            )
            & phase_bin_summary[
                "band"
            ].eq("FULL_PACE")
            & phase_bin_summary[
                "observations"
            ].ge(100)
        ]
        .copy()
    )

    bin_focus[
        "_window_order"
    ] = (
        bin_focus["window"]
        .map(WINDOW_ORDER)
    )

    bin_focus = bin_focus.sort_values(
        [
            "phase_feature",
            "_window_order",
            "path_state",
            "mean_net_24h_percent",
        ]
    )

    print("")
    print(
        "Loaded detail observations: "
        f"{len(detail):,}"
    )

    print(
        "Merged V3KS observations: "
        f"{len(merged):,}"
    )

    print(
        "Active shock-normal BUILD/FULL: "
        f"{len(active):,}"
    )

    print(
        "Late-exhaustion observations: "
        f"{len(late):,}"
    )

    print(
        "Fresh-acceleration observations: "
        f"{len(fresh):,}"
    )

    print(
        "Fully missing local joins: "
        f"{fully_missing_local}"
    )

    print(
        "Fully missing market joins: "
        f"{fully_missing_market}"
    )

    print("")
    print(
        "All market predictors use completed "
        "decision candles or earlier data."
    )

    print(
        "Higher timeframes are aligned by "
        "completed-candle forward fill."
    )

    print(
        "No future return, MFE, MAE or "
        "next-entry price is used as a predictor."
    )

    print_table(
        "========== V3KS JOIN INTEGRITY ==========",
        integrity,
        [
            "window",
            "symbol",
            "decision_times",
            "detail_symbol_rows",
            "local_snapshot_rows",
            "missing_raw_times",
            "rows_with_missing_local_feature",
            "market_snapshot_rows",
            "market_state_conflicts",
            "market_rows_with_any_missing_feature",
        ],
    )

    print_table(
        "========== MARKET PHASE BY PATH: 2024 VS 2025H1 ==========",
        path_focus,
        [
            "window",
            "band",
            "path_state",
            "observations",
            "mean_breadth_ema20",
            "mean_breadth_ema50",
            "mean_breadth_ema200",
            "mean_breadth_ema50_delta_1h",
            "mean_breadth_ema50_delta_4h",
            "mean_market_return_1h",
            "mean_market_return_4h",
            "mean_dispersion_4h",
            "mean_btc_leadership_4h",
            "mean_btc_trend",
            "mean_eth_trend",
            "mean_state_age_hours",
            "mean_net_4h_percent",
            "mean_net_12h_percent",
            "mean_net_24h_percent",
            "positive_net_24h_rate_percent",
        ],
    )

    print_table(
        "========== LATE EXHAUSTION BY MARKET PHASE ==========",
        late_phase_summary,
        [
            "window",
            "band",
            "phase_alignment_state",
            "observations",
            "mean_breadth_ema50",
            "mean_breadth_ema50_delta_4h",
            "mean_market_return_4h",
            "mean_dispersion_4h",
            "mean_btc_leadership_4h",
            "mean_btc_trend",
            "mean_eth_trend",
            "mean_state_age_hours",
            "mean_net_4h_percent",
            "mean_net_12h_percent",
            "mean_net_24h_percent",
            "positive_net_24h_rate_percent",
            "mean_mfe_24h_percent",
            "mean_mae_24h_percent",
        ],
    )

    print_table(
        "========== FRESH ACCELERATION BY MARKET PHASE: 2024 VS 2025H1 ==========",
        fresh_focus,
        [
            "window",
            "band",
            "phase_alignment_state",
            "observations",
            "mean_breadth_ema50",
            "mean_breadth_ema50_delta_4h",
            "mean_market_return_4h",
            "mean_dispersion_4h",
            "mean_btc_leadership_4h",
            "mean_btc_trend",
            "mean_eth_trend",
            "mean_net_4h_percent",
            "mean_net_12h_percent",
            "mean_net_24h_percent",
            "positive_net_24h_rate_percent",
        ],
    )

    print_table(
        "========== STATE TRANSITIONS: 2024 VS 2025H1 ==========",
        transition_focus,
        [
            "window",
            "band",
            "path_state",
            "state_transition",
            "state_age_bin",
            "observations",
            "mean_breadth_ema50",
            "mean_breadth_ema50_delta_4h",
            "mean_market_return_4h",
            "mean_btc_leadership_4h",
            "mean_net_4h_percent",
            "mean_net_12h_percent",
            "mean_net_24h_percent",
            "positive_net_24h_rate_percent",
        ],
    )

    print_table(
        "========== LOCAL + MARKET INTERACTIONS: 2024 VS 2025H1 ==========",
        interaction_focus,
        [
            "window",
            "band",
            "phase_alignment_state",
            "extension_state",
            "candle_state",
            "observations",
            "mean_breadth_ema50",
            "mean_breadth_ema50_delta_4h",
            "mean_market_return_4h",
            "mean_btc_leadership_4h",
            "mean_ema20_distance_atr",
            "mean_close_location",
            "mean_upper_wick_ratio",
            "mean_net_4h_percent",
            "mean_net_12h_percent",
            "mean_net_24h_percent",
            "positive_net_24h_rate_percent",
        ],
    )

    print_table(
        "========== FULL_PACE PHASE BINS: 2024 VS 2025H1 ==========",
        bin_focus,
        [
            "window",
            "path_state",
            "phase_feature",
            "phase_bin",
            "observations",
            "mean_breadth_ema50",
            "mean_breadth_ema50_delta_4h",
            "mean_market_return_4h",
            "mean_btc_leadership_4h",
            "mean_net_4h_percent",
            "mean_net_12h_percent",
            "mean_net_24h_percent",
            "positive_net_24h_rate_percent",
        ],
    )

    print("")
    print(
        "V3KS market phase reports saved."
    )

    print(
        "No opportunity weight, score band, "
        "risk rule or strategy logic changed."
    )


if __name__ == "__main__":
    main()


