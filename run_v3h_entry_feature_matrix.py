from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

DETAIL_PATH = REPORT_DIR / "v3h_failure_bucket_detail.csv"

NUMERIC_FEATURES = [
    "setup_score",
    "entry_reward_risk",
    "relative_strength_pct",
    "momentum_24h",
    "momentum_72h",
    "ema50_slope_12h",
    "rsi14",
    "adx14",
    "atr_percent",
    "volume_ratio",
    "ema20_distance_atr",
    "ema50_distance_atr",
    "ema200_distance_atr",
    "recent_high_72h_distance_atr",
    "candle_range_atr",
    "candle_body_atr",
    "upper_wick_atr",
    "lower_wick_atr",
    "close_location",
    "market_score",
    "btc_score",
    "eth_score",
    "breadth_score",
    "market_risk_multiplier",
]


def value_or_nan(row: pd.Series, name: str) -> float:
    if name not in row.index:
        return np.nan

    value = row[name]

    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def enum_value(obj, attribute: str):
    if obj is None:
        return None

    value = getattr(obj, attribute, None)

    if value is None:
        return None

    return getattr(value, "value", value)


def cliffs_delta(
    runner: pd.Series,
    failure: pd.Series,
) -> float:
    x = pd.to_numeric(
        runner,
        errors="coerce",
    ).dropna().to_numpy()

    y = pd.to_numeric(
        failure,
        errors="coerce",
    ).dropna().to_numpy()

    if len(x) == 0 or len(y) == 0:
        return np.nan

    greater = 0
    lower = 0

    for value in x:
        greater += int(
            np.sum(value > y)
        )
        lower += int(
            np.sum(value < y)
        )

    return (
        greater - lower
    ) / (
        len(x) * len(y)
    )


def classify_mfe(value: float) -> str:
    if value < 0.5:
        return "FALSE_<0.5R"

    if value >= 2.0:
        return "RUNNER_>=2R"

    return "MIDDLE_0.5-2R"


def extract_window() -> None:
    window = os.environ.get(
        "V3G4_WINDOW",
        "",
    ).strip()

    if not window:
        raise RuntimeError(
            "V3G4_WINDOW environment variable is required."
        )

    window_key = window.lower()

    # Import AFTER environment variables are set.
    # This reuses the exact V3G4 historical data,
    # surveillance and market-state pipeline.
    import run_v3g4_eth365_robustness as engine

    print()
    print("=" * 110)
    print(
        f"V3H ENTRY FEATURE EXTRACTION - {window}"
    )
    print("=" * 110)

    if not DETAIL_PATH.exists():
        raise FileNotFoundError(
            DETAIL_PATH
        )

    diagnostic = pd.read_csv(
        DETAIL_PATH
    )

    diagnostic["entry_time"] = pd.to_datetime(
        diagnostic["entry_time"]
    )

    diagnostic["exit_time"] = pd.to_datetime(
        diagnostic["exit_time"]
    )

    diagnostic["window_key"] = (
        diagnostic["window"]
        .astype(str)
        .str.lower()
    )

    diagnostic = diagnostic[
        diagnostic["window_key"]
        == window_key
    ].copy()

    if diagnostic.empty:
        raise RuntimeError(
            f"No V3H diagnostic trades found for {window}."
        )

    trade_path = (
        Path("logs/backtests")
        / (
            "v3g4_eth365_robustness_"
            f"{window_key}_baseline_trades.csv"
        )
    )

    if not trade_path.exists():
        raise FileNotFoundError(
            trade_path
        )

    trades = pd.read_csv(
        trade_path
    )

    trades = trades[
        trades["symbol"].isin(
            ["LINKUSDT", "SOLUSDT"]
        )
    ].copy()

    trades["entry_time"] = pd.to_datetime(
        trades["entry_time"]
    )

    trades["exit_time"] = pd.to_datetime(
        trades["exit_time"]
    )

    trade_metadata = trades[
        [
            "symbol",
            "entry_time",
            "exit_time",
            "setup_score",
            "entry_reward_risk",
            "account_risk_percent",
        ]
    ].copy()

    diagnostic = diagnostic.merge(
        trade_metadata,
        on=[
            "symbol",
            "entry_time",
            "exit_time",
        ],
        how="left",
        validate="one_to_one",
    )

    print(
        f"Trades to diagnose: {len(diagnostic)}"
    )

    print(
        "Building exact frozen historical frames..."
    )

    (
        frames_15m,
        frames_1h,
        common_15m,
        common_1h,
    ) = engine.prepare_frames()

    print(
        "Building frozen surveillance..."
    )

    (
        asset_shocks,
        market_shocks,
    ) = engine.build_shock_tables(
        frames_15m,
        common_15m,
    )

    print(
        "Building frozen market states..."
    )

    market_states = engine.build_market_states(
        frames_1h,
        common_1h,
        market_shocks,
    )

    rows = []
    missing = []

    for _, trade in diagnostic.iterrows():

        symbol = trade["symbol"]
        entry_time = trade["entry_time"]

        frame = frames_1h[symbol]

        if entry_time not in frame.index:
            missing.append(
                (
                    symbol,
                    entry_time,
                    "1H_FRAME",
                )
            )
            continue

        candle = frame.loc[
            entry_time
        ]

        market_state = market_states.get(
            entry_time
        )

        asset_state = (
            asset_shocks
            .get(symbol, {})
            .get(entry_time)
        )

        relative_strength = (
            engine.relative_strength_percentiles(
                frames_1h,
                entry_time,
            ).get(
                symbol,
                np.nan,
            )
        )

        close = value_or_nan(
            candle,
            "close",
        )

        open_price = value_or_nan(
            candle,
            "open",
        )

        high = value_or_nan(
            candle,
            "high",
        )

        low = value_or_nan(
            candle,
            "low",
        )

        atr = value_or_nan(
            candle,
            "ATR14",
        )

        ema20 = value_or_nan(
            candle,
            "EMA20",
        )

        ema50 = value_or_nan(
            candle,
            "EMA50",
        )

        ema200 = value_or_nan(
            candle,
            "EMA200",
        )

        recent_high = value_or_nan(
            candle,
            "RECENT_HIGH_72H",
        )

        candle_range = (
            high - low
        )

        body = (
            close - open_price
        )

        upper_wick = (
            high
            - max(
                open_price,
                close,
            )
        )

        lower_wick = (
            min(
                open_price,
                close,
            )
            - low
        )

        if atr > 0:
            ema20_distance_atr = (
                close - ema20
            ) / atr

            ema50_distance_atr = (
                close - ema50
            ) / atr

            ema200_distance_atr = (
                close - ema200
            ) / atr

            recent_high_distance_atr = (
                close - recent_high
            ) / atr

            candle_range_atr = (
                candle_range / atr
            )

            candle_body_atr = (
                body / atr
            )

            upper_wick_atr = (
                upper_wick / atr
            )

            lower_wick_atr = (
                lower_wick / atr
            )

        else:
            ema20_distance_atr = np.nan
            ema50_distance_atr = np.nan
            ema200_distance_atr = np.nan
            recent_high_distance_atr = np.nan
            candle_range_atr = np.nan
            candle_body_atr = np.nan
            upper_wick_atr = np.nan
            lower_wick_atr = np.nan

        if candle_range > 0:
            close_location = (
                close - low
            ) / candle_range
        else:
            close_location = np.nan

        atr_percent = (
            atr / close * 100.0
            if close > 0
            else np.nan
        )

        trend_stable_6 = bool(
            candle.get(
                "TREND_STABLE_6",
                False,
            )
        )

        trend_stable_4_of_6 = bool(
            candle.get(
                "TREND_STABLE_4_OF_6",
                False,
            )
        )

        rows.append(
            {
                "window": window_key,
                "symbol": symbol,
                "route": trade["route"],
                "entry_time": entry_time,
                "strict_mfe_r": float(
                    trade[
                        "strict_mfe_r"
                    ]
                ),
                "strict_mae_r": float(
                    trade[
                        "strict_mae_r"
                    ]
                ),
                "realized_r": float(
                    trade[
                        "realized_r"
                    ]
                ),
                "profit": float(
                    trade[
                        "profit_trade"
                    ]
                ),
                "mfe_class": classify_mfe(
                    float(
                        trade[
                            "strict_mfe_r"
                        ]
                    )
                ),
                "setup_score": float(
                    trade[
                        "setup_score"
                    ]
                ),
                "entry_reward_risk": float(
                    trade[
                        "entry_reward_risk"
                    ]
                ),
                "account_risk_percent": float(
                    trade[
                        "account_risk_percent"
                    ]
                ),
                "relative_strength_pct": float(
                    relative_strength
                ),
                "momentum_24h": value_or_nan(
                    candle,
                    "MOMENTUM_24H",
                ),
                "momentum_72h": value_or_nan(
                    candle,
                    "MOMENTUM_72H",
                ),
                "ema50_slope_12h": value_or_nan(
                    candle,
                    "EMA50_SLOPE_12H",
                ),
                "rsi14": value_or_nan(
                    candle,
                    "RSI14",
                ),
                "adx14": value_or_nan(
                    candle,
                    "ADX14",
                ),
                "atr_percent": atr_percent,
                "volume_ratio": value_or_nan(
                    candle,
                    "VOLUME_RATIO",
                ),
                "ema20_distance_atr": (
                    ema20_distance_atr
                ),
                "ema50_distance_atr": (
                    ema50_distance_atr
                ),
                "ema200_distance_atr": (
                    ema200_distance_atr
                ),
                "recent_high_72h_distance_atr": (
                    recent_high_distance_atr
                ),
                "candle_range_atr": (
                    candle_range_atr
                ),
                "candle_body_atr": (
                    candle_body_atr
                ),
                "upper_wick_atr": (
                    upper_wick_atr
                ),
                "lower_wick_atr": (
                    lower_wick_atr
                ),
                "close_location": (
                    close_location
                ),
                "trend_stable_6": int(
                    trend_stable_6
                ),
                "trend_stable_4_of_6": int(
                    trend_stable_4_of_6
                ),
                "market_state": (
                    enum_value(
                        market_state,
                        "state",
                    )
                ),
                "market_score": (
                    getattr(
                        market_state,
                        "score",
                        np.nan,
                    )
                    if market_state
                    else np.nan
                ),
                "btc_score": (
                    getattr(
                        market_state,
                        "btc_score",
                        np.nan,
                    )
                    if market_state
                    else np.nan
                ),
                "eth_score": (
                    getattr(
                        market_state,
                        "eth_score",
                        np.nan,
                    )
                    if market_state
                    else np.nan
                ),
                "breadth_score": (
                    getattr(
                        market_state,
                        "breadth_score",
                        np.nan,
                    )
                    if market_state
                    else np.nan
                ),
                "market_risk_multiplier": (
                    getattr(
                        market_state,
                        "risk_multiplier",
                        np.nan,
                    )
                    if market_state
                    else np.nan
                ),
                "asset_shock_level": (
                    enum_value(
                        asset_state,
                        "level",
                    )
                ),
                "asset_shock_score": (
                    getattr(
                        asset_state,
                        "score",
                        np.nan,
                    )
                    if asset_state
                    else np.nan
                ),
            }
        )

    result = pd.DataFrame(
        rows
    )

    output_path = (
        REPORT_DIR
        / (
            "v3h_entry_features_"
            f"{window_key}.csv"
        )
    )

    result.to_csv(
        output_path,
        index=False,
    )

    print()
    print(
        f"Extracted: {len(result)} / "
        f"{len(diagnostic)} trades"
    )

    print(
        f"Saved: {output_path}"
    )

    if missing:
        print()
        print(
            "WARNING - missing entry timestamps:"
        )

        for item in missing:
            print(item)


def summarize() -> None:
    paths = sorted(
        REPORT_DIR.glob(
            "v3h_entry_features_*.csv"
        )
    )

    if not paths:
        raise RuntimeError(
            "No V3H entry feature files found."
        )

    frames = [
        pd.read_csv(path)
        for path in paths
    ]

    data = pd.concat(
        frames,
        ignore_index=True,
    )

    # Defensive dedupe in case a file is rerun.
    data = (
        data
        .drop_duplicates(
            subset=[
                "window",
                "symbol",
                "entry_time",
            ],
            keep="last",
        )
        .reset_index(
            drop=True
        )
    )

    matrix_path = (
        REPORT_DIR
        / "v3h_entry_feature_matrix.csv"
    )

    data.to_csv(
        matrix_path,
        index=False,
    )

    print()
    print("=" * 120)
    print("V3H ENTRY FEATURE MATRIX - CLASS COUNTS")
    print("=" * 120)

    counts = (
        data
        .groupby(
            [
                "symbol",
                "window",
                "mfe_class",
            ]
        )
        .size()
        .rename("trades")
        .reset_index()
    )

    print(
        counts.to_string(
            index=False
        )
    )

    effect_rows = []

    for symbol, group in data.groupby(
        "symbol"
    ):
        false = group[
            group["mfe_class"]
            == "FALSE_<0.5R"
        ]

        runner = group[
            group["mfe_class"]
            == "RUNNER_>=2R"
        ]

        for feature in NUMERIC_FEATURES:

            if feature not in group.columns:
                continue

            false_values = pd.to_numeric(
                false[feature],
                errors="coerce",
            ).dropna()

            runner_values = pd.to_numeric(
                runner[feature],
                errors="coerce",
            ).dropna()

            if (
                len(false_values) == 0
                or len(runner_values) == 0
            ):
                continue

            false_median = float(
                false_values.median()
            )

            runner_median = float(
                runner_values.median()
            )

            median_difference = (
                runner_median
                - false_median
            )

            delta = cliffs_delta(
                runner_values,
                false_values,
            )

            direction = (
                "RUNNER_HIGHER"
                if median_difference > 0
                else (
                    "RUNNER_LOWER"
                    if median_difference < 0
                    else "SAME"
                )
            )

            windows_tested = 0
            same_direction_windows = 0

            for _, window_group in group.groupby(
                "window"
            ):
                window_false = pd.to_numeric(
                    window_group.loc[
                        window_group[
                            "mfe_class"
                        ]
                        == "FALSE_<0.5R",
                        feature,
                    ],
                    errors="coerce",
                ).dropna()

                window_runner = pd.to_numeric(
                    window_group.loc[
                        window_group[
                            "mfe_class"
                        ]
                        == "RUNNER_>=2R",
                        feature,
                    ],
                    errors="coerce",
                ).dropna()

                if (
                    len(window_false) == 0
                    or len(window_runner) == 0
                ):
                    continue

                windows_tested += 1

                window_difference = (
                    window_runner.median()
                    - window_false.median()
                )

                if (
                    median_difference > 0
                    and window_difference > 0
                ):
                    same_direction_windows += 1

                elif (
                    median_difference < 0
                    and window_difference < 0
                ):
                    same_direction_windows += 1

                elif (
                    median_difference == 0
                    and window_difference == 0
                ):
                    same_direction_windows += 1

            consistency = (
                same_direction_windows
                / windows_tested
                * 100.0
                if windows_tested
                else np.nan
            )

            effect_rows.append(
                {
                    "symbol": symbol,
                    "feature": feature,
                    "false_n": len(
                        false_values
                    ),
                    "runner_n": len(
                        runner_values
                    ),
                    "false_median": (
                        false_median
                    ),
                    "runner_median": (
                        runner_median
                    ),
                    "median_difference": (
                        median_difference
                    ),
                    "cliffs_delta": delta,
                    "abs_cliffs_delta": (
                        abs(delta)
                    ),
                    "direction": direction,
                    "windows_tested": (
                        windows_tested
                    ),
                    "same_direction_windows": (
                        same_direction_windows
                    ),
                    "direction_consistency_pct": (
                        consistency
                    ),
                }
            )

    effects = pd.DataFrame(
        effect_rows
    )

    effects = effects.sort_values(
        [
            "symbol",
            "abs_cliffs_delta",
            "direction_consistency_pct",
        ],
        ascending=[
            True,
            False,
            False,
        ],
    )

    effect_path = (
        REPORT_DIR
        / "v3h_entry_feature_effects.csv"
    )

    effects.to_csv(
        effect_path,
        index=False,
    )

    for symbol in [
        "LINKUSDT",
        "SOLUSDT",
    ]:
        print()
        print("=" * 120)
        print(
            f"{symbol} - FALSE BREAKOUT vs RUNNER"
        )
        print("=" * 120)

        table = effects[
            effects["symbol"]
            == symbol
        ].head(15)

        print(
            table[
                [
                    "feature",
                    "false_n",
                    "runner_n",
                    "false_median",
                    "runner_median",
                    "cliffs_delta",
                    "direction",
                    "windows_tested",
                    "same_direction_windows",
                    "direction_consistency_pct",
                ]
            ].to_string(
                index=False,
                float_format=lambda x: (
                    f"{x:.3f}"
                ),
            )
        )

    print()
    print("=" * 120)
    print("MARKET STATE DISTRIBUTION")
    print("=" * 120)

    state_table = (
        data
        .groupby(
            [
                "symbol",
                "mfe_class",
                "market_state",
            ],
            dropna=False,
        )
        .size()
        .rename("trades")
        .reset_index()
    )

    print(
        state_table.to_string(
            index=False
        )
    )

    print()
    print("=" * 120)
    print("TREND STABILITY")
    print("=" * 120)

    trend_table = (
        data
        .groupby(
            [
                "symbol",
                "mfe_class",
            ]
        )
        .agg(
            trades=(
                "entry_time",
                "size",
            ),
            trend_stable_6_pct=(
                "trend_stable_6",
                lambda x: (
                    float(
                        pd.to_numeric(
                            x,
                            errors="coerce",
                        ).mean()
                        * 100.0
                    )
                ),
            ),
            trend_stable_4_of_6_pct=(
                "trend_stable_4_of_6",
                lambda x: (
                    float(
                        pd.to_numeric(
                            x,
                            errors="coerce",
                        ).mean()
                        * 100.0
                    )
                ),
            ),
        )
        .reset_index()
    )

    print(
        trend_table.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.2f}"
            ),
        )
    )

    print()
    print(
        f"Saved: {matrix_path}"
    )
    print(
        f"Saved: {effect_path}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--summarize",
        action="store_true",
    )

    args = parser.parse_args()

    if args.summarize:
        summarize()
    else:
        extract_window()


if __name__ == "__main__":
    main()
