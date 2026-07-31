from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from run_v3ko_slow_fast_interaction import (
    FAST_COMPONENTS,
    FAST_WEIGHT,
    SLOW_COMPONENTS,
    SLOW_WEIGHT,
    load_reports,
    weighted_component_score,
)


OUTPUT_DIR = Path("reports")

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

ANCHOR_OFFSETS = (
    0,
    24,
    48,
    72,
)

BARS_PER_24H = 96

FAST_RISE_THRESHOLD = 5.0
FAST_FLAT_LIMIT = 1.0
FAST_REVERSAL_THRESHOLD = -5.0
RS_DANGER_THRESHOLD = -2.5
RS_REVERSAL_THRESHOLD = -5.0
FAST_SATURATION_THRESHOLD = 80.0


def safe_mean(
    values: pd.Series,
) -> float:
    if values.empty:
        return float("nan")

    return float(values.mean())


def positive_rate(
    values: pd.Series,
) -> float:
    if values.empty:
        return float("nan")

    return float(
        values.gt(0.0).mean()
        * 100.0
    )


def add_path_features(
    data: pd.DataFrame,
) -> pd.DataFrame:
    result = data.copy()

    result = result.sort_values(
        [
            "window",
            "symbol",
            "decision_time",
        ]
    ).reset_index(drop=True)

    result["fast_score"] = (
        weighted_component_score(
            result,
            FAST_COMPONENTS,
            FAST_WEIGHT,
        )
    )

    result["slow_score"] = (
        weighted_component_score(
            result,
            SLOW_COMPONENTS,
            SLOW_WEIGHT,
        )
    )

    result["reconstructed_score"] = (
        result["fast_score"]
        * FAST_WEIGHT
        + result["slow_score"]
        * SLOW_WEIGHT
    )

    result["score_reconstruction_error"] = (
        result["score"]
        - result["reconstructed_score"]
    ).abs()

    groups = result.groupby(
        [
            "window",
            "symbol",
        ],
        sort=False,
    )

    result["bar_number"] = (
        groups.cumcount()
    )

    previous_time_1 = (
        groups["decision_time"]
        .shift(1)
    )

    previous_time_4 = (
        groups["decision_time"]
        .shift(4)
    )

    previous_time_16 = (
        groups["decision_time"]
        .shift(16)
    )

    valid_15m = (
        result["decision_time"]
        - previous_time_1
    ).eq(
        pd.Timedelta(minutes=15)
    )

    valid_1h = (
        result["decision_time"]
        - previous_time_4
    ).eq(
        pd.Timedelta(minutes=60)
    )

    valid_4h = (
        result["decision_time"]
        - previous_time_16
    ).eq(
        pd.Timedelta(minutes=240)
    )

    def exact_delta(
        column: str,
        lag: int,
        valid_elapsed: pd.Series,
    ) -> pd.Series:
        previous_value = (
            groups[column]
            .shift(lag)
        )

        delta = (
            result[column]
            - previous_value
        )

        return delta.where(
            valid_elapsed
        )

    result["fast_delta_15m"] = (
        exact_delta(
            "fast_score",
            1,
            valid_15m,
        )
    )

    result["fast_delta_1h"] = (
        exact_delta(
            "fast_score",
            4,
            valid_1h,
        )
    )

    result["fast_delta_4h"] = (
        exact_delta(
            "fast_score",
            16,
            valid_4h,
        )
    )

    result["rs_delta_1h"] = (
        exact_delta(
            "relative_strength_score",
            4,
            valid_1h,
        )
    )

    result["rs_delta_4h"] = (
        exact_delta(
            "relative_strength_score",
            16,
            valid_4h,
        )
    )

    result["market_delta_1h"] = (
        exact_delta(
            "market_score",
            4,
            valid_1h,
        )
    )

    result["trend_15m_delta_1h"] = (
        exact_delta(
            "trend_15m_score",
            4,
            valid_1h,
        )
    )

    result["trend_1h_delta_4h"] = (
        exact_delta(
            "trend_1h_score",
            16,
            valid_4h,
        )
    )

    result["volume_delta_1h"] = (
        exact_delta(
            "positive_volume_score",
            4,
            valid_1h,
        )
    )

    required_history = (
        result[
            [
                "fast_delta_1h",
                "fast_delta_4h",
                "rs_delta_1h",
            ]
        ]
        .notna()
        .all(axis=1)
    )

    late_exhaustion = (
        required_history
        & result["fast_score"].ge(
            FAST_SATURATION_THRESHOLD
        )
        & result["fast_delta_4h"].ge(
            FAST_RISE_THRESHOLD
        )
        & result["fast_delta_1h"].le(
            FAST_FLAT_LIMIT
        )
        & result["rs_delta_1h"].le(
            RS_DANGER_THRESHOLD
        )
    )

    reversal_risk = (
        required_history
        & ~late_exhaustion
        & (
            result["fast_delta_1h"].le(
                FAST_REVERSAL_THRESHOLD
            )
            | result["rs_delta_1h"].le(
                RS_REVERSAL_THRESHOLD
            )
        )
    )

    fresh_acceleration = (
        required_history
        & ~reversal_risk
        & ~late_exhaustion
        & result["fast_delta_1h"].ge(
            FAST_RISE_THRESHOLD
        )
        & result["fast_delta_4h"].ge(
            FAST_RISE_THRESHOLD
        )
        & result["rs_delta_1h"].ge(
            0.0
        )
    )

    healthy_continuation = (
        required_history
        & ~reversal_risk
        & ~late_exhaustion
        & ~fresh_acceleration
        & result["fast_delta_4h"].ge(
            0.0
        )
        & result["fast_delta_1h"].ge(
            -1.0
        )
        & result["rs_delta_1h"].gt(
            RS_DANGER_THRESHOLD
        )
    )

    result["path_state"] = "MIXED"

    result.loc[
        ~required_history,
        "path_state",
    ] = "INSUFFICIENT_HISTORY"

    result.loc[
        reversal_risk,
        "path_state",
    ] = "REVERSAL_RISK"

    result.loc[
        late_exhaustion,
        "path_state",
    ] = "LATE_EXHAUSTION"

    result.loc[
        fresh_acceleration,
        "path_state",
    ] = "FRESH_ACCELERATION"

    result.loc[
        healthy_continuation,
        "path_state",
    ] = "HEALTHY_CONTINUATION"

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
            mean_slow_score=(
                "slow_score",
                "mean",
            ),
            mean_fast_delta_15m=(
                "fast_delta_15m",
                "mean",
            ),
            mean_fast_delta_1h=(
                "fast_delta_1h",
                "mean",
            ),
            mean_fast_delta_4h=(
                "fast_delta_4h",
                "mean",
            ),
            mean_rs_delta_1h=(
                "rs_delta_1h",
                "mean",
            ),
            mean_rs_delta_4h=(
                "rs_delta_4h",
                "mean",
            ),
            mean_market_delta_1h=(
                "market_delta_1h",
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


def build_candidate_impact(
    active: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for (
        window,
        band,
    ), group in active.groupby(
        [
            "window",
            "band",
        ],
        sort=False,
    ):
        candidate = group.loc[
            group["path_state"].eq(
                "LATE_EXHAUSTION"
            )
        ]

        kept = group.loc[
            ~group["path_state"].eq(
                "LATE_EXHAUSTION"
            )
        ]

        total_positive = int(
            group[
                "net_return_24h_percent"
            ]
            .gt(0.0)
            .sum()
        )

        kept_positive = int(
            kept[
                "net_return_24h_percent"
            ]
            .gt(0.0)
            .sum()
        )

        rows.append(
            {
                "window": window,
                "band": band,
                "baseline_observations": int(
                    len(group)
                ),
                "baseline_mean_net_24h_percent": (
                    safe_mean(
                        group[
                            "net_return_24h_percent"
                        ]
                    )
                ),
                "candidate_observations": int(
                    len(candidate)
                ),
                "candidate_rate_percent": (
                    float(
                        len(candidate)
                        / len(group)
                        * 100.0
                    )
                    if len(group) > 0
                    else float("nan")
                ),
                "candidate_mean_net_4h_percent": (
                    safe_mean(
                        candidate[
                            "net_return_4h_percent"
                        ]
                    )
                ),
                "candidate_mean_net_12h_percent": (
                    safe_mean(
                        candidate[
                            "net_return_12h_percent"
                        ]
                    )
                ),
                "candidate_mean_net_24h_percent": (
                    safe_mean(
                        candidate[
                            "net_return_24h_percent"
                        ]
                    )
                ),
                "candidate_positive_rate_percent": (
                    positive_rate(
                        candidate[
                            "net_return_24h_percent"
                        ]
                    )
                ),
                "candidate_mean_mfe_24h_percent": (
                    safe_mean(
                        candidate[
                            "mfe_24h_percent"
                        ]
                    )
                ),
                "candidate_mean_mae_24h_percent": (
                    safe_mean(
                        candidate[
                            "mae_24h_percent"
                        ]
                    )
                ),
                "kept_observations": int(
                    len(kept)
                ),
                "kept_mean_net_24h_percent": (
                    safe_mean(
                        kept[
                            "net_return_24h_percent"
                        ]
                    )
                ),
                "kept_positive_rate_percent": (
                    positive_rate(
                        kept[
                            "net_return_24h_percent"
                        ]
                    )
                ),
                "positive_observation_retention_percent": (
                    float(
                        kept_positive
                        / total_positive
                        * 100.0
                    )
                    if total_positive > 0
                    else float("nan")
                ),
            }
        )

    return pd.DataFrame(rows)


def build_non_overlap_anchor_check(
    active: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for anchor_offset in ANCHOR_OFFSETS:
        anchor_data = active.loc[
            active["bar_number"]
            .mod(BARS_PER_24H)
            .eq(anchor_offset)
        ]

        for (
            window,
            band,
        ), group in anchor_data.groupby(
            [
                "window",
                "band",
            ],
            sort=False,
        ):
            candidate = group.loc[
                group["path_state"].eq(
                    "LATE_EXHAUSTION"
                )
            ]

            kept = group.loc[
                ~group["path_state"].eq(
                    "LATE_EXHAUSTION"
                )
            ]

            rows.append(
                {
                    "anchor_offset": (
                        anchor_offset
                    ),
                    "window": window,
                    "band": band,
                    "baseline_observations": (
                        int(len(group))
                    ),
                    "baseline_mean_net_24h_percent": (
                        safe_mean(
                            group[
                                "net_return_24h_percent"
                            ]
                        )
                    ),
                    "candidate_observations": (
                        int(len(candidate))
                    ),
                    "candidate_mean_net_24h_percent": (
                        safe_mean(
                            candidate[
                                "net_return_24h_percent"
                            ]
                        )
                    ),
                    "candidate_positive_rate_percent": (
                        positive_rate(
                            candidate[
                                "net_return_24h_percent"
                            ]
                        )
                    ),
                    "kept_observations": (
                        int(len(kept))
                    ),
                    "kept_mean_net_24h_percent": (
                        safe_mean(
                            kept[
                                "net_return_24h_percent"
                            ]
                        )
                    ),
                    "candidate_minus_kept_24h_percent": (
                        safe_mean(
                            candidate[
                                "net_return_24h_percent"
                            ]
                        )
                        - safe_mean(
                            kept[
                                "net_return_24h_percent"
                            ]
                        )
                    ),
                }
            )

    return pd.DataFrame(rows)


def build_anchor_stability(
    anchor_check: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for (
        window,
        band,
    ), group in anchor_check.groupby(
        [
            "window",
            "band",
        ],
        sort=False,
    ):
        candidate_values = group[
            "candidate_mean_net_24h_percent"
        ].dropna()

        difference_values = group[
            "candidate_minus_kept_24h_percent"
        ].dropna()

        rows.append(
            {
                "window": window,
                "band": band,
                "valid_anchor_count": int(
                    len(candidate_values)
                ),
                "candidate_observations_total": int(
                    group[
                        "candidate_observations"
                    ].sum()
                ),
                "candidate_anchor_mean_percent": (
                    safe_mean(
                        candidate_values
                    )
                ),
                "candidate_anchor_min_percent": (
                    float(
                        candidate_values.min()
                    )
                    if not candidate_values.empty
                    else float("nan")
                ),
                "candidate_anchor_max_percent": (
                    float(
                        candidate_values.max()
                    )
                    if not candidate_values.empty
                    else float("nan")
                ),
                "negative_candidate_anchor_count": (
                    int(
                        candidate_values.lt(0.0)
                        .sum()
                    )
                ),
                "candidate_worse_than_kept_anchor_count": (
                    int(
                        difference_values.lt(0.0)
                        .sum()
                    )
                ),
                "mean_candidate_minus_kept_percent": (
                    safe_mean(
                        difference_values
                    )
                ),
            }
        )

    return pd.DataFrame(rows)


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
        table[columns].to_string(
            index=False,
            float_format=(
                lambda value: f"{value:.5f}"
            ),
        )
    )


def main() -> None:
    data = load_reports()

    data = add_path_features(
        data
    )

    maximum_error = float(
        data[
            "score_reconstruction_error"
        ].max()
    )

    shock_normal = data.loc[
        data["shock_normal"]
    ].copy()

    active = shock_normal.loc[
        shock_normal["band"].isin(
            [
                "BUILD",
                "FULL_PACE",
            ]
        )
    ].copy()

    path_summary = aggregate_outcomes(
        active,
        [
            "window",
            "band",
            "path_state",
        ],
    )

    candidate_impact = (
        build_candidate_impact(
            active
        )
    )

    anchor_check = (
        build_non_overlap_anchor_check(
            active
        )
    )

    anchor_stability = (
        build_anchor_stability(
            anchor_check
        )
    )

    late_only = active.loc[
        active["path_state"].eq(
            "LATE_EXHAUSTION"
        )
    ].copy()

    symbol_summary = aggregate_outcomes(
        late_only,
        [
            "window",
            "band",
            "symbol",
        ],
    )

    market_state_summary = (
        aggregate_outcomes(
            late_only,
            [
                "window",
                "band",
                "market_state",
            ],
        )
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    path_summary.to_csv(
        OUTPUT_DIR
        / "v3kq_path_state_summary.csv",
        index=False,
    )

    candidate_impact.to_csv(
        OUTPUT_DIR
        / "v3kq_late_exhaustion_impact.csv",
        index=False,
    )

    anchor_check.to_csv(
        OUTPUT_DIR
        / "v3kq_non_overlap_anchor_check.csv",
        index=False,
    )

    anchor_stability.to_csv(
        OUTPUT_DIR
        / "v3kq_anchor_stability.csv",
        index=False,
    )

    symbol_summary.to_csv(
        OUTPUT_DIR
        / "v3kq_late_exhaustion_by_symbol.csv",
        index=False,
    )

    market_state_summary.to_csv(
        OUTPUT_DIR
        / "v3kq_late_exhaustion_by_market_state.csv",
        index=False,
    )

    path_summary["_window_order"] = (
        path_summary["window"]
        .map(WINDOW_ORDER)
    )

    path_summary["_band_order"] = (
        path_summary["band"]
        .map(BAND_ORDER)
    )

    path_summary["_path_order"] = (
        path_summary["path_state"]
        .map(PATH_ORDER)
    )

    path_summary = (
        path_summary.sort_values(
            [
                "_window_order",
                "_band_order",
                "_path_order",
            ]
        )
    )

    candidate_impact["_window_order"] = (
        candidate_impact["window"]
        .map(WINDOW_ORDER)
    )

    candidate_impact["_band_order"] = (
        candidate_impact["band"]
        .map(BAND_ORDER)
    )

    candidate_impact = (
        candidate_impact.sort_values(
            [
                "_window_order",
                "_band_order",
            ]
        )
    )

    anchor_stability["_window_order"] = (
        anchor_stability["window"]
        .map(WINDOW_ORDER)
    )

    anchor_stability["_band_order"] = (
        anchor_stability["band"]
        .map(BAND_ORDER)
    )

    anchor_stability = (
        anchor_stability.sort_values(
            [
                "_window_order",
                "_band_order",
            ]
        )
    )

    symbol_focus = symbol_summary.loc[
        symbol_summary["window"].isin(
            [
                "2024",
                "2025H1",
            ]
        )
        & symbol_summary[
            "observations"
        ].ge(20)
    ].copy()

    symbol_focus["_window_order"] = (
        symbol_focus["window"]
        .map(WINDOW_ORDER)
    )

    symbol_focus["_band_order"] = (
        symbol_focus["band"]
        .map(BAND_ORDER)
    )

    symbol_focus = (
        symbol_focus.sort_values(
            [
                "_window_order",
                "_band_order",
                "mean_net_24h_percent",
            ]
        )
    )

    market_focus = (
        market_state_summary.loc[
            market_state_summary[
                "observations"
            ].ge(20)
        ]
        .copy()
    )

    market_focus["_window_order"] = (
        market_focus["window"]
        .map(WINDOW_ORDER)
    )

    market_focus["_band_order"] = (
        market_focus["band"]
        .map(BAND_ORDER)
    )

    market_focus = (
        market_focus.sort_values(
            [
                "_window_order",
                "_band_order",
                "mean_net_24h_percent",
            ]
        )
    )

    valid_1h_rate = float(
        data[
            "fast_delta_1h"
        ].notna().mean()
        * 100.0
    )

    valid_4h_rate = float(
        data[
            "fast_delta_4h"
        ].notna().mean()
        * 100.0
    )

    print(
        "Loaded total observations: "
        f"{len(data):,}"
    )

    print(
        "Shock-normal observations: "
        f"{len(shock_normal):,}"
    )

    print(
        "Active BUILD/FULL_PACE observations: "
        f"{len(active):,}"
    )

    print(
        "Maximum score reconstruction error: "
        f"{maximum_error:.8f}"
    )

    print(
        "Valid exact causal 1h delta coverage: "
        f"{valid_1h_rate:.4f}%"
    )

    print(
        "Valid exact causal 4h delta coverage: "
        f"{valid_4h_rate:.4f}%"
    )

    print("")
    print(
        "Thresholds are exploratory diagnostics, "
        "not strategy rules."
    )

    print_table(
        "========== CROSS-WINDOW PATH STATES ==========",
        path_summary,
        [
            "window",
            "band",
            "path_state",
            "observations",
            "mean_fast_score",
            "mean_slow_score",
            "mean_fast_delta_1h",
            "mean_fast_delta_4h",
            "mean_rs_delta_1h",
            "mean_net_4h_percent",
            "mean_net_12h_percent",
            "mean_net_24h_percent",
            "positive_net_24h_rate_percent",
            "mean_mfe_24h_percent",
            "mean_mae_24h_percent",
        ],
    )

    print_table(
        "========== LATE EXHAUSTION CANDIDATE IMPACT ==========",
        candidate_impact,
        [
            "window",
            "band",
            "baseline_observations",
            "baseline_mean_net_24h_percent",
            "candidate_observations",
            "candidate_rate_percent",
            "candidate_mean_net_4h_percent",
            "candidate_mean_net_12h_percent",
            "candidate_mean_net_24h_percent",
            "candidate_positive_rate_percent",
            "kept_mean_net_24h_percent",
            "positive_observation_retention_percent",
        ],
    )

    print_table(
        "========== NON-OVERLAPPING 24H ANCHOR STABILITY ==========",
        anchor_stability,
        [
            "window",
            "band",
            "valid_anchor_count",
            "candidate_observations_total",
            "candidate_anchor_mean_percent",
            "candidate_anchor_min_percent",
            "candidate_anchor_max_percent",
            "negative_candidate_anchor_count",
            "candidate_worse_than_kept_anchor_count",
            "mean_candidate_minus_kept_percent",
        ],
    )

    print_table(
        "========== LATE EXHAUSTION BY SYMBOL: 2024 AND 2025H1 ==========",
        symbol_focus,
        [
            "window",
            "band",
            "symbol",
            "observations",
            "mean_fast_score",
            "mean_slow_score",
            "mean_fast_delta_1h",
            "mean_fast_delta_4h",
            "mean_rs_delta_1h",
            "mean_net_4h_percent",
            "mean_net_12h_percent",
            "mean_net_24h_percent",
            "positive_net_24h_rate_percent",
            "mean_mfe_24h_percent",
            "mean_mae_24h_percent",
        ],
    )

    print_table(
        "========== LATE EXHAUSTION BY MARKET STATE ==========",
        market_focus,
        [
            "window",
            "band",
            "market_state",
            "observations",
            "mean_fast_score",
            "mean_slow_score",
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
        "V3KQ diagnostic reports saved."
    )

    print(
        "No opportunity weights, score bands, "
        "risk rules or strategy code changed."
    )


if __name__ == "__main__":
    main()

