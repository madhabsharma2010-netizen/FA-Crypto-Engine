from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import run_v3d_diagnostics as diagnostics

from run_v3km_opportunity_score_audit import (
    WINDOWS,
    configure_window,
)

from run_v3ko_slow_fast_interaction import (
    load_reports,
)

from run_v3kq_acceleration_exhaustion import (
    add_path_features,
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

FEATURE_COLUMNS = (
    "ema20_distance_atr",
    "ema50_distance_atr",
    "ema200_distance_atr",
    "breakout_4h_distance_atr",
    "breakout_12h_distance_atr",
    "support_4h_distance_atr",
    "candle_range_atr",
    "signed_body_ratio",
    "body_ratio",
    "close_location",
    "upper_wick_ratio",
    "lower_wick_ratio",
    "volume_ratio_prior20",
    "atr_expansion_prior48",
    "return_15m_raw_percent",
    "return_60m_raw_percent",
)

BIN_SPECS = {
    "ema20_extension_bin": {
        "source": "ema20_distance_atr",
        "bins": [
            -np.inf,
            0.0,
            1.0,
            2.0,
            3.0,
            np.inf,
        ],
        "labels": [
            "BELOW_EMA20",
            "0_TO_1_ATR",
            "1_TO_2_ATR",
            "2_TO_3_ATR",
            "3_PLUS_ATR",
        ],
    },
    "ema50_extension_bin": {
        "source": "ema50_distance_atr",
        "bins": [
            -np.inf,
            0.0,
            1.0,
            2.0,
            4.0,
            np.inf,
        ],
        "labels": [
            "BELOW_EMA50",
            "0_TO_1_ATR",
            "1_TO_2_ATR",
            "2_TO_4_ATR",
            "4_PLUS_ATR",
        ],
    },
    "breakout_4h_bin": {
        "source": "breakout_4h_distance_atr",
        "bins": [
            -np.inf,
            -1.0,
            0.0,
            1.0,
            2.0,
            np.inf,
        ],
        "labels": [
            "BELOW_BY_1_PLUS_ATR",
            "BELOW_BY_0_TO_1_ATR",
            "FRESH_BREAKOUT_0_TO_1",
            "EXTENDED_BREAKOUT_1_TO_2",
            "EXTENDED_BREAKOUT_2_PLUS",
        ],
    },
    "breakout_12h_bin": {
        "source": "breakout_12h_distance_atr",
        "bins": [
            -np.inf,
            -1.0,
            0.0,
            1.0,
            2.0,
            np.inf,
        ],
        "labels": [
            "BELOW_BY_1_PLUS_ATR",
            "BELOW_BY_0_TO_1_ATR",
            "FRESH_BREAKOUT_0_TO_1",
            "EXTENDED_BREAKOUT_1_TO_2",
            "EXTENDED_BREAKOUT_2_PLUS",
        ],
    },
    "close_location_bin": {
        "source": "close_location",
        "bins": [
            -np.inf,
            0.20,
            0.40,
            0.60,
            0.80,
            np.inf,
        ],
        "labels": [
            "BOTTOM_20",
            "LOWER_20_TO_40",
            "MIDDLE_40_TO_60",
            "UPPER_60_TO_80",
            "TOP_20",
        ],
    },
    "upper_wick_bin": {
        "source": "upper_wick_ratio",
        "bins": [
            -np.inf,
            0.10,
            0.25,
            0.40,
            np.inf,
        ],
        "labels": [
            "VERY_SMALL",
            "SMALL",
            "MEDIUM",
            "LARGE",
        ],
    },
    "body_ratio_bin": {
        "source": "body_ratio",
        "bins": [
            -np.inf,
            0.20,
            0.40,
            0.60,
            np.inf,
        ],
        "labels": [
            "SMALL_BODY",
            "MEDIUM_BODY",
            "LARGE_BODY",
            "VERY_LARGE_BODY",
        ],
    },
    "range_atr_bin": {
        "source": "candle_range_atr",
        "bins": [
            -np.inf,
            0.50,
            1.00,
            1.50,
            2.00,
            np.inf,
        ],
        "labels": [
            "BELOW_HALF_ATR",
            "HALF_TO_1_ATR",
            "1_TO_1_5_ATR",
            "1_5_TO_2_ATR",
            "2_PLUS_ATR",
        ],
    },
    "volume_ratio_bin": {
        "source": "volume_ratio_prior20",
        "bins": [
            -np.inf,
            0.75,
            1.00,
            1.25,
            1.75,
            np.inf,
        ],
        "labels": [
            "LOW",
            "BELOW_NORMAL",
            "NORMAL_TO_MODERATE",
            "HIGH",
            "VERY_HIGH",
        ],
    },
    "atr_expansion_bin": {
        "source": "atr_expansion_prior48",
        "bins": [
            -np.inf,
            0.80,
            1.00,
            1.25,
            1.50,
            np.inf,
        ],
        "labels": [
            "COMPRESSED",
            "BELOW_NORMAL",
            "MODERATE_EXPANSION",
            "HIGH_EXPANSION",
            "EXTREME_EXPANSION",
        ],
    },
}


def safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    safe_denominator = denominator.where(
        denominator.abs().gt(1e-12)
    )

    result = (
        numerator
        / safe_denominator
    )

    return result.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )


def positive_rate(
    values: pd.Series,
) -> float:
    if values.empty:
        return float("nan")

    return float(
        values.gt(0.0).mean()
        * 100.0
    )


def safe_mean(
    values: pd.Series,
) -> float:
    if values.empty:
        return float("nan")

    return float(
        values.mean()
    )


def build_raw_features(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    data = frame.copy()

    candle_range = (
        data["high"]
        - data["low"]
    )

    signed_body = (
        data["close"]
        - data["open"]
    )

    absolute_body = (
        signed_body.abs()
    )

    upper_body_edge = pd.concat(
        [
            data["open"],
            data["close"],
        ],
        axis=1,
    ).max(axis=1)

    lower_body_edge = pd.concat(
        [
            data["open"],
            data["close"],
        ],
        axis=1,
    ).min(axis=1)

    upper_wick = (
        data["high"]
        - upper_body_edge
    ).clip(lower=0.0)

    lower_wick = (
        lower_body_edge
        - data["low"]
    ).clip(lower=0.0)

    prior_atr = (
        data["ATR14"]
        .shift(1)
    )

    prior_high_4h = (
        data["high"]
        .shift(1)
        .rolling(
            16,
            min_periods=16,
        )
        .max()
    )

    prior_low_4h = (
        data["low"]
        .shift(1)
        .rolling(
            16,
            min_periods=16,
        )
        .min()
    )

    prior_high_12h = (
        data["high"]
        .shift(1)
        .rolling(
            48,
            min_periods=48,
        )
        .max()
    )

    prior_volume_mean_20 = (
        data["volume"]
        .shift(1)
        .rolling(
            20,
            min_periods=20,
        )
        .mean()
    )

    atr_percent = (
        safe_divide(
            data["ATR14"],
            data["close"],
        )
        * 100.0
    )

    prior_atr_percent_median_48 = (
        atr_percent
        .shift(1)
        .rolling(
            48,
            min_periods=48,
        )
        .median()
    )

    features = pd.DataFrame(
        index=data.index
    )

    features[
        "decision_close"
    ] = data["close"]

    features[
        "prior_atr14"
    ] = prior_atr

    features[
        "ema20_distance_atr"
    ] = safe_divide(
        data["close"]
        - data["EMA20"],
        prior_atr,
    )

    features[
        "ema50_distance_atr"
    ] = safe_divide(
        data["close"]
        - data["EMA50"],
        prior_atr,
    )

    features[
        "ema200_distance_atr"
    ] = safe_divide(
        data["close"]
        - data["EMA200"],
        prior_atr,
    )

    features[
        "breakout_4h_distance_atr"
    ] = safe_divide(
        data["close"]
        - prior_high_4h,
        prior_atr,
    )

    features[
        "breakout_12h_distance_atr"
    ] = safe_divide(
        data["close"]
        - prior_high_12h,
        prior_atr,
    )

    features[
        "support_4h_distance_atr"
    ] = safe_divide(
        data["close"]
        - prior_low_4h,
        prior_atr,
    )

    features[
        "candle_range_atr"
    ] = safe_divide(
        candle_range,
        prior_atr,
    )

    features[
        "signed_body_ratio"
    ] = safe_divide(
        signed_body,
        candle_range,
    )

    features[
        "body_ratio"
    ] = safe_divide(
        absolute_body,
        candle_range,
    )

    features[
        "close_location"
    ] = safe_divide(
        data["close"]
        - data["low"],
        candle_range,
    )

    features[
        "upper_wick_ratio"
    ] = safe_divide(
        upper_wick,
        candle_range,
    )

    features[
        "lower_wick_ratio"
    ] = safe_divide(
        lower_wick,
        candle_range,
    )

    features[
        "volume_ratio_prior20"
    ] = safe_divide(
        data["volume"],
        prior_volume_mean_20,
    )

    features[
        "atr_expansion_prior48"
    ] = safe_divide(
        atr_percent,
        prior_atr_percent_median_48,
    )

    features[
        "return_15m_raw_percent"
    ] = (
        data["close"]
        .pct_change(1)
        * 100.0
    )

    features[
        "return_60m_raw_percent"
    ] = (
        data["close"]
        .pct_change(4)
        * 100.0
    )

    return features.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )


def load_raw_feature_snapshot(
    detail: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    feature_parts: list[pd.DataFrame] = []
    integrity_rows: list[dict[str, object]] = []

    for (
        window,
        (
            start,
            end,
        ),
    ) in WINDOWS.items():
        print("")
        print(
            "Preparing V3KR raw features "
            f"for {window}: "
            f"{start} -> {end}"
        )

        configure_window(
            start,
            end,
        )

        window_detail = detail.loc[
            detail["window"].eq(
                window
            )
        ]

        for symbol in diagnostics.SYMBOLS:
            symbol_detail = (
                window_detail.loc[
                    window_detail[
                        "symbol"
                    ].eq(symbol)
                ]
                .sort_values(
                    "decision_time"
                )
            )

            if symbol_detail.empty:
                raise RuntimeError(
                    f"NO_DETAIL_ROWS: "
                    f"{window} {symbol}"
                )

            raw_frame = (
                diagnostics.prepare_15m(
                    symbol
                )
            )

            raw_features = (
                build_raw_features(
                    raw_frame
                )
            )

            decision_times = pd.DatetimeIndex(
                symbol_detail[
                    "decision_time"
                ]
            )

            selected = raw_features.reindex(
                decision_times
            )

            completely_missing = int(
                selected.loc[:, list(FEATURE_COLUMNS)]
                .isna()
                .all(axis=1)
                .sum()
            )

            any_missing = int(
                selected.loc[:, list(FEATURE_COLUMNS)]
                .isna()
                .any(axis=1)
                .sum()
            )

            selected = (
                selected
                .reset_index()
                .rename(
                    columns={
                        "completion_time":
                        "decision_time",
                        "index":
                        "decision_time",
                    }
                )
            )

            selected["window"] = window
            selected["symbol"] = symbol

            selected = selected[
                [
                    "window",
                    "symbol",
                    "decision_time",
                    "decision_close",
                    "prior_atr14",
                    *FEATURE_COLUMNS,
                ]
            ]

            feature_parts.append(
                selected
            )

            integrity_rows.append(
                {
                    "window": window,
                    "symbol": symbol,
                    "detail_rows": int(
                        len(symbol_detail)
                    ),
                    "raw_frame_rows": int(
                        len(raw_frame)
                    ),
                    "selected_rows": int(
                        len(selected)
                    ),
                    "completely_missing_rows": (
                        completely_missing
                    ),
                    "any_missing_feature_rows": (
                        any_missing
                    ),
                    "first_detail_time": (
                        symbol_detail[
                            "decision_time"
                        ].min()
                    ),
                    "last_detail_time": (
                        symbol_detail[
                            "decision_time"
                        ].max()
                    ),
                    "first_raw_time": (
                        raw_frame.index.min()
                    ),
                    "last_raw_time": (
                        raw_frame.index.max()
                    ),
                }
            )

            print(
                f"  {symbol}: "
                f"{len(selected):,} joined rows, "
                f"{completely_missing} fully missing"
            )

    feature_snapshot = pd.concat(
        feature_parts,
        ignore_index=True,
    )

    integrity = pd.DataFrame(
        integrity_rows
    )

    duplicate_keys = int(
        feature_snapshot.duplicated(
            subset=[
                "window",
                "symbol",
                "decision_time",
            ],
            keep=False,
        ).sum()
    )

    if duplicate_keys != 0:
        raise RuntimeError(
            "RAW_FEATURE_DUPLICATE_KEYS: "
            f"{duplicate_keys}"
        )

    return (
        feature_snapshot,
        integrity,
    )


def add_diagnostic_states(
    data: pd.DataFrame,
) -> pd.DataFrame:
    result = data.copy()

    for (
        output_column,
        spec,
    ) in BIN_SPECS.items():
        result[output_column] = pd.cut(
            result[
                spec["source"]
            ],
            bins=spec["bins"],
            labels=spec["labels"],
            include_lowest=True,
            right=False,
        )

    breakout_distance = result[
        "breakout_4h_distance_atr"
    ]

    result["breakout_state"] = np.select(
        [
            breakout_distance.lt(-1.0),
            breakout_distance.lt(0.0),
            breakout_distance.lt(1.0),
            breakout_distance.lt(2.0),
        ],
        [
            "BELOW_HIGH_1_PLUS_ATR",
            "BELOW_HIGH_0_TO_1_ATR",
            "FRESH_BREAKOUT_0_TO_1_ATR",
            "EXTENDED_BREAKOUT_1_TO_2_ATR",
        ],
        default=(
            "EXTENDED_BREAKOUT_2_PLUS_ATR"
        ),
    )

    result.loc[
        breakout_distance.isna(),
        "breakout_state",
    ] = "MISSING"

    ema20_distance = result[
        "ema20_distance_atr"
    ]

    result["extension_state"] = np.select(
        [
            ema20_distance.lt(0.0),
            ema20_distance.lt(1.0),
            ema20_distance.lt(2.0),
            ema20_distance.lt(3.0),
        ],
        [
            "BELOW_EMA20",
            "NORMAL_0_TO_1_ATR",
            "ELEVATED_1_TO_2_ATR",
            "STRETCHED_2_TO_3_ATR",
        ],
        default="EXTREME_3_PLUS_ATR",
    )

    result.loc[
        ema20_distance.isna(),
        "extension_state",
    ] = "MISSING"

    close_location = result[
        "close_location"
    ]

    upper_wick = result[
        "upper_wick_ratio"
    ]

    signed_body = result[
        "signed_body_ratio"
    ]

    strong_close = (
        close_location.ge(0.80)
        & upper_wick.le(0.15)
        & signed_body.gt(0.0)
    )

    upper_rejection = (
        upper_wick.ge(0.35)
        & close_location.lt(0.65)
    )

    bearish_close = (
        signed_body.lt(0.0)
        & close_location.lt(0.40)
    )

    positive_close = (
        signed_body.gt(0.0)
        & close_location.ge(0.60)
    )

    result["candle_state"] = np.select(
        [
            strong_close,
            upper_rejection,
            bearish_close,
            positive_close,
        ],
        [
            "STRONG_CLOSE",
            "UPPER_REJECTION",
            "BEARISH_CLOSE",
            "POSITIVE_CLOSE",
        ],
        default="NEUTRAL",
    )

    missing_candle = (
        close_location.isna()
        | upper_wick.isna()
        | signed_body.isna()
    )

    result.loc[
        missing_candle,
        "candle_state",
    ] = "MISSING"

    volume_ratio = result[
        "volume_ratio_prior20"
    ]

    result["volume_state"] = np.select(
        [
            volume_ratio.lt(0.75),
            volume_ratio.lt(1.00),
            volume_ratio.lt(1.25),
            volume_ratio.lt(1.75),
        ],
        [
            "LOW",
            "BELOW_NORMAL",
            "NORMAL_TO_MODERATE",
            "HIGH",
        ],
        default="VERY_HIGH",
    )

    result.loc[
        volume_ratio.isna(),
        "volume_state",
    ] = "MISSING"

    atr_expansion = result[
        "atr_expansion_prior48"
    ]

    result["volatility_state"] = np.select(
        [
            atr_expansion.lt(0.80),
            atr_expansion.lt(1.00),
            atr_expansion.lt(1.25),
            atr_expansion.lt(1.50),
        ],
        [
            "COMPRESSED",
            "BELOW_NORMAL",
            "MODERATE_EXPANSION",
            "HIGH_EXPANSION",
        ],
        default="EXTREME_EXPANSION",
    )

    result.loc[
        atr_expansion.isna(),
        "volatility_state",
    ] = "MISSING"

    return result


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
            mean_ema20_distance_atr=(
                "ema20_distance_atr",
                "mean",
            ),
            mean_ema50_distance_atr=(
                "ema50_distance_atr",
                "mean",
            ),
            mean_breakout_4h_distance_atr=(
                "breakout_4h_distance_atr",
                "mean",
            ),
            mean_breakout_12h_distance_atr=(
                "breakout_12h_distance_atr",
                "mean",
            ),
            mean_candle_range_atr=(
                "candle_range_atr",
                "mean",
            ),
            mean_signed_body_ratio=(
                "signed_body_ratio",
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
            mean_volume_ratio_prior20=(
                "volume_ratio_prior20",
                "mean",
            ),
            mean_atr_expansion_prior48=(
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
                    values.le(-1.0).mean()
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


def build_feature_bin_summary(
    data: pd.DataFrame,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []

    for bin_column in BIN_SPECS:
        summary = aggregate_outcomes(
            data.dropna(
                subset=[
                    bin_column,
                ]
            ),
            [
                "window",
                "band",
                bin_column,
            ],
        )

        summary = summary.rename(
            columns={
                bin_column:
                "feature_bin",
            }
        )

        summary.insert(
            2,
            "feature",
            bin_column,
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

    raw_features, integrity = (
        load_raw_feature_snapshot(
            detail
        )
    )

    merged = detail.merge(
        raw_features,
        on=[
            "window",
            "symbol",
            "decision_time",
        ],
        how="left",
        validate="one_to_one",
    )

    if len(merged) != len(detail):
        raise RuntimeError(
            "ROW_COUNT_CHANGED_AFTER_MERGE"
        )

    fully_missing_rows = int(
        merged.loc[:, list(FEATURE_COLUMNS)]
        .isna()
        .all(axis=1)
        .sum()
    )

    if fully_missing_rows != 0:
        raise RuntimeError(
            "FULLY_MISSING_JOIN_ROWS: "
            f"{fully_missing_rows}"
        )

    close_mismatch = (
        merged[
            "decision_close"
        ]
        .sub(
            merged[
                "entry_open"
            ]
        )
        .abs()
    )

    merged[
        "decision_to_entry_gap_atr"
    ] = safe_divide(
        merged[
            "entry_open"
        ]
        - merged[
            "decision_close"
        ],
        merged[
            "prior_atr14"
        ],
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

    path_feature_summary = (
        aggregate_outcomes(
            active,
            [
                "window",
                "band",
                "path_state",
            ],
        )
    )

    feature_bin_summary = (
        build_feature_bin_summary(
            active
        )
    )

    late_feature_bin_summary = (
        build_feature_bin_summary(
            late
        )
    )

    late_symbol_summary = (
        aggregate_outcomes(
            late,
            [
                "window",
                "band",
                "symbol",
            ],
        )
    )

    late_feature_means = (
        aggregate_outcomes(
            late,
            [
                "window",
                "band",
            ],
        )
    )

    interaction_summary = (
        aggregate_outcomes(
            late,
            [
                "window",
                "band",
                "extension_state",
                "breakout_state",
                "candle_state",
            ],
        )
    )

    compact_interaction = (
        interaction_summary.loc[
            interaction_summary[
                "observations"
            ].ge(20)
        ]
        .copy()
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    integrity.to_csv(
        OUTPUT_DIR
        / "v3kr_feature_join_integrity.csv",
        index=False,
    )

    active.to_csv(
        OUTPUT_DIR
        / "v3kr_active_feature_snapshot.csv",
        index=False,
    )

    path_feature_summary.to_csv(
        OUTPUT_DIR
        / "v3kr_path_feature_summary.csv",
        index=False,
    )

    feature_bin_summary.to_csv(
        OUTPUT_DIR
        / "v3kr_feature_bin_summary.csv",
        index=False,
    )

    late_feature_bin_summary.to_csv(
        OUTPUT_DIR
        / "v3kr_late_exhaustion_feature_bins.csv",
        index=False,
    )

    late_symbol_summary.to_csv(
        OUTPUT_DIR
        / "v3kr_late_exhaustion_by_symbol.csv",
        index=False,
    )

    late_feature_means.to_csv(
        OUTPUT_DIR
        / "v3kr_late_exhaustion_feature_means.csv",
        index=False,
    )

    interaction_summary.to_csv(
        OUTPUT_DIR
        / "v3kr_late_exhaustion_interactions.csv",
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

    late_feature_means["_window_order"] = (
        late_feature_means["window"]
        .map(WINDOW_ORDER)
    )

    late_feature_means["_band_order"] = (
        late_feature_means["band"]
        .map(BAND_ORDER)
    )

    late_feature_means = (
        late_feature_means.sort_values(
            [
                "_window_order",
                "_band_order",
            ]
        )
    )

    path_focus = path_feature_summary.loc[
        path_feature_summary[
            "window"
        ].isin(
            [
                "2024",
                "2025H1",
            ]
        )
    ].copy()

    path_focus["_window_order"] = (
        path_focus["window"]
        .map(WINDOW_ORDER)
    )

    path_focus["_band_order"] = (
        path_focus["band"]
        .map(BAND_ORDER)
    )

    path_focus = path_focus.sort_values(
        [
            "_window_order",
            "_band_order",
            "path_state",
        ]
    )

    late_symbol_focus = (
        late_symbol_summary.loc[
            late_symbol_summary[
                "window"
            ].isin(
                [
                    "2024",
                    "2025H1",
                ]
            )
            & late_symbol_summary[
                "observations"
            ].ge(20)
        ]
        .copy()
    )

    late_symbol_focus["_window_order"] = (
        late_symbol_focus["window"]
        .map(WINDOW_ORDER)
    )

    late_symbol_focus["_band_order"] = (
        late_symbol_focus["band"]
        .map(BAND_ORDER)
    )

    late_symbol_focus = (
        late_symbol_focus.sort_values(
            [
                "_window_order",
                "_band_order",
                "mean_net_24h_percent",
            ]
        )
    )

    compact_interaction = (
        compact_interaction.loc[
            compact_interaction[
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

    compact_interaction[
        "_window_order"
    ] = (
        compact_interaction["window"]
        .map(WINDOW_ORDER)
    )

    compact_interaction[
        "_band_order"
    ] = (
        compact_interaction["band"]
        .map(BAND_ORDER)
    )

    compact_interaction = (
        compact_interaction.sort_values(
            [
                "_window_order",
                "_band_order",
                "mean_net_24h_percent",
            ]
        )
    )

    print("")
    print(
        "Loaded detail observations: "
        f"{len(detail):,}"
    )

    print(
        "Merged feature observations: "
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
        "Fully missing feature joins: "
        f"{fully_missing_rows}"
    )

    print(
        "Maximum decision-close to "
        "next-entry-open absolute gap: "
        f"{float(close_mismatch.max()):.10f}"
    )

    print(
        "Maximum absolute entry-gap ATR: "
        f"{float(
            merged[
                'decision_to_entry_gap_atr'
            ].abs().max()
        ):.10f}"
    )

    print("")
    print(
        "All raw predictors use the completed "
        "decision candle or earlier data."
    )

    print(
        "Prior ATR, prior volume baseline and "
        "prior breakout levels are shifted."
    )

    print(
        "No feature uses future return, MFE, "
        "MAE or next-entry price as a predictor."
    )

    print_table(
        "========== RAW FEATURE JOIN INTEGRITY ==========",
        integrity,
        [
            "window",
            "symbol",
            "detail_rows",
            "selected_rows",
            "completely_missing_rows",
            "any_missing_feature_rows",
            "first_detail_time",
            "last_detail_time",
        ],
    )

    print_table(
        "========== LATE EXHAUSTION FEATURE MEANS ==========",
        late_feature_means,
        [
            "window",
            "band",
            "observations",
            "mean_ema20_distance_atr",
            "mean_ema50_distance_atr",
            "mean_breakout_4h_distance_atr",
            "mean_breakout_12h_distance_atr",
            "mean_candle_range_atr",
            "mean_signed_body_ratio",
            "mean_close_location",
            "mean_upper_wick_ratio",
            "mean_volume_ratio_prior20",
            "mean_atr_expansion_prior48",
            "mean_net_4h_percent",
            "mean_net_12h_percent",
            "mean_net_24h_percent",
            "positive_net_24h_rate_percent",
            "mean_mfe_24h_percent",
            "mean_mae_24h_percent",
        ],
    )

    print_table(
        "========== PATH FEATURES: 2024 VS 2025H1 ==========",
        path_focus,
        [
            "window",
            "band",
            "path_state",
            "observations",
            "mean_ema20_distance_atr",
            "mean_ema50_distance_atr",
            "mean_breakout_4h_distance_atr",
            "mean_candle_range_atr",
            "mean_close_location",
            "mean_upper_wick_ratio",
            "mean_volume_ratio_prior20",
            "mean_atr_expansion_prior48",
            "mean_net_4h_percent",
            "mean_net_12h_percent",
            "mean_net_24h_percent",
            "positive_net_24h_rate_percent",
        ],
    )

    print_table(
        "========== LATE EXHAUSTION BY SYMBOL: 2024 VS 2025H1 ==========",
        late_symbol_focus,
        [
            "window",
            "band",
            "symbol",
            "observations",
            "mean_ema20_distance_atr",
            "mean_ema50_distance_atr",
            "mean_breakout_4h_distance_atr",
            "mean_close_location",
            "mean_upper_wick_ratio",
            "mean_volume_ratio_prior20",
            "mean_atr_expansion_prior48",
            "mean_net_4h_percent",
            "mean_net_12h_percent",
            "mean_net_24h_percent",
            "positive_net_24h_rate_percent",
        ],
    )

    print_table(
        "========== LATE EXHAUSTION INTERACTIONS: 2024 VS 2025H1 ==========",
        compact_interaction,
        [
            "window",
            "band",
            "extension_state",
            "breakout_state",
            "candle_state",
            "observations",
            "mean_ema20_distance_atr",
            "mean_breakout_4h_distance_atr",
            "mean_close_location",
            "mean_upper_wick_ratio",
            "mean_volume_ratio_prior20",
            "mean_atr_expansion_prior48",
            "mean_net_4h_percent",
            "mean_net_12h_percent",
            "mean_net_24h_percent",
            "positive_net_24h_rate_percent",
            "mean_mfe_24h_percent",
            "mean_mae_24h_percent",
        ],
    )

    print("")
    print(
        "V3KR causal price-location reports saved."
    )

    print(
        "No opportunity weight, score band, "
        "risk rule or strategy logic changed."
    )


if __name__ == "__main__":
    main()

