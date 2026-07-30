from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from core.v3km_opportunity import (
    WEIGHT_MARKET,
    WEIGHT_POSITIVE_VOLUME,
    WEIGHT_RELATIVE_STRENGTH,
    WEIGHT_TREND_15M,
    WEIGHT_TREND_1H,
    WEIGHT_TREND_2H,
    WEIGHT_TREND_4H,
)


INPUT_FILES = {
    "2022": Path(
        "reports/v3km_opportunity_2022_detail.csv"
    ),
    "2023": Path(
        "reports/v3km_opportunity_2023_detail.csv"
    ),
    "2024": Path(
        "reports/v3km_opportunity_2024_detail.csv"
    ),
    "2025H1": Path(
        "reports/v3km_opportunity_2025h1_detail.csv"
    ),
}

OUTPUT_DIR = Path("reports")

BAND_ORDER = {
    "CASH": 0,
    "PROBE": 1,
    "BUILD": 2,
    "FULL_PACE": 3,
}

COMPONENT_WEIGHTS = {
    "market_score": WEIGHT_MARKET,
    "trend_15m_score": WEIGHT_TREND_15M,
    "trend_1h_score": WEIGHT_TREND_1H,
    "trend_2h_score": WEIGHT_TREND_2H,
    "trend_4h_score": WEIGHT_TREND_4H,
    "relative_strength_score": (
        WEIGHT_RELATIVE_STRENGTH
    ),
    "positive_volume_score": (
        WEIGHT_POSITIVE_VOLUME
    ),
}

NUMERIC_COLUMNS = (
    "score",
    "target_fraction",
    "market_score",
    "trend_15m_score",
    "trend_1h_score",
    "trend_2h_score",
    "trend_4h_score",
    "relative_strength_score",
    "positive_volume_score",
    "entry_open",
    "net_return_15m_percent",
    "net_return_1h_percent",
    "net_return_4h_percent",
    "net_return_12h_percent",
    "net_return_24h_percent",
    "mfe_24h_percent",
    "mae_24h_percent",
)


def parse_boolean(
    series: pd.Series,
) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(
            {
                "true": True,
                "false": False,
                "1": True,
                "0": False,
            }
        )
        .fillna(False)
        .astype(bool)
    )


def load_detail_reports() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for expected_window, path in INPUT_FILES.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Missing audit report: {path}"
            )

        frame = pd.read_csv(
            path,
            parse_dates=["decision_time"],
        )

        for column in NUMERIC_COLUMNS:
            frame[column] = pd.to_numeric(
                frame[column],
                errors="coerce",
            )

        frame["shock_normal"] = parse_boolean(
            frame["shock_normal"]
        )
        frame["eligible"] = parse_boolean(
            frame["eligible"]
        )

        reported_windows = set(
            frame["window"].astype(str).unique()
        )

        if reported_windows != {expected_window}:
            raise ValueError(
                f"Unexpected window labels in {path}: "
                f"{reported_windows}"
            )

        frame["window"] = (
            frame["window"]
            .astype(str)
        )

        frames.append(frame)

    combined = pd.concat(
        frames,
        ignore_index=True,
    )

    combined = combined.dropna(
        subset=[
            "decision_time",
            "symbol",
            "score",
            "band",
            "net_return_24h_percent",
            "mfe_24h_percent",
            "mae_24h_percent",
        ]
    )

    combined = combined.sort_values(
        [
            "window",
            "symbol",
            "decision_time",
        ]
    ).reset_index(drop=True)

    return combined


def add_sequence_features(
    data: pd.DataFrame,
) -> pd.DataFrame:
    result = data.copy()

    symbol_groups = result.groupby(
        ["window", "symbol"],
        sort=False,
    )

    result["previous_band"] = (
        symbol_groups["band"]
        .shift(1)
        .fillna("START")
    )

    result["previous_score"] = (
        symbol_groups["score"]
        .shift(1)
    )

    result["score_delta_15m"] = (
        result["score"]
        - result["previous_score"]
    )

    result["score_delta_1h"] = (
        result["score"]
        - symbol_groups["score"].shift(4)
    )

    result["transition"] = (
        result["previous_band"].astype(str)
        + "->"
        + result["band"].astype(str)
    )

    result["band_run_id"] = (
        symbol_groups["band"]
        .transform(
            lambda values: (
                values.ne(values.shift())
                .cumsum()
            )
        )
    )

    result["band_age_bars"] = (
        result.groupby(
            [
                "window",
                "symbol",
                "band_run_id",
            ],
            sort=False,
        )
        .cumcount()
        + 1
    )

    result["band_age_bucket"] = pd.cut(
        result["band_age_bars"],
        bins=[
            0,
            1,
            4,
            16,
            np.inf,
        ],
        labels=[
            "NEW_1_BAR",
            "EARLY_2_TO_4",
            "MATURE_5_TO_16",
            "EXTENDED_17_PLUS",
        ],
        include_lowest=True,
    )

    result["score_direction"] = pd.cut(
        result["score_delta_1h"],
        bins=[
            -np.inf,
            -5.0,
            -1.0,
            1.0,
            5.0,
            np.inf,
        ],
        labels=[
            "FALLING_FAST",
            "FALLING",
            "FLAT",
            "RISING",
            "RISING_FAST",
        ],
        include_lowest=True,
    )

    return result


def aggregate_outcomes(
    data: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    summary = (
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
            mean_score_delta_15m=(
                "score_delta_15m",
                "mean",
            ),
            mean_score_delta_1h=(
                "score_delta_1h",
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
                lambda values: float(
                    (values > 0.0).mean()
                    * 100.0
                ),
            ),
            loss_below_minus_1_rate_percent=(
                "net_return_24h_percent",
                lambda values: float(
                    (values <= -1.0).mean()
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

    return summary


def build_component_contrast(
    data: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for (
        window,
        band,
    ), group in data.groupby(
        ["window", "band"],
        sort=False,
    ):
        winners = group.loc[
            group["net_return_24h_percent"] > 0.0
        ]
        losers = group.loc[
            group["net_return_24h_percent"] <= 0.0
        ]

        for component in COMPONENT_WEIGHTS:
            q25 = float(
                group[component].quantile(0.25)
            )
            q75 = float(
                group[component].quantile(0.75)
            )

            bottom = group.loc[
                group[component] <= q25
            ]
            top = group.loc[
                group[component] >= q75
            ]

            winner_mean = (
                float(winners[component].mean())
                if not winners.empty
                else float("nan")
            )
            loser_mean = (
                float(losers[component].mean())
                if not losers.empty
                else float("nan")
            )

            bottom_return = (
                float(
                    bottom[
                        "net_return_24h_percent"
                    ].mean()
                )
                if not bottom.empty
                else float("nan")
            )
            top_return = (
                float(
                    top[
                        "net_return_24h_percent"
                    ].mean()
                )
                if not top.empty
                else float("nan")
            )

            rows.append(
                {
                    "window": window,
                    "band": band,
                    "component": component,
                    "weight": (
                        COMPONENT_WEIGHTS[
                            component
                        ]
                    ),
                    "observations": int(
                        len(group)
                    ),
                    "winner_component_mean": (
                        winner_mean
                    ),
                    "loser_component_mean": (
                        loser_mean
                    ),
                    "winner_minus_loser_component": (
                        winner_mean
                        - loser_mean
                    ),
                    "bottom_quartile_threshold": q25,
                    "top_quartile_threshold": q75,
                    "bottom_quartile_mean_net_24h_percent": (
                        bottom_return
                    ),
                    "top_quartile_mean_net_24h_percent": (
                        top_return
                    ),
                    "top_minus_bottom_net_24h_percent": (
                        top_return
                        - bottom_return
                    ),
                }
            )

    return pd.DataFrame(rows)


def scores_to_bands(
    score: pd.Series,
) -> pd.Series:
    values = np.select(
        [
            score >= 80.0,
            score >= 65.0,
            score >= 45.0,
        ],
        [
            "FULL_PACE",
            "BUILD",
            "PROBE",
        ],
        default="CASH",
    )

    return pd.Series(
        values,
        index=score.index,
        dtype="object",
    )


def bands_to_target_fraction(
    bands: pd.Series,
) -> pd.Series:
    return bands.map(
        {
            "CASH": 0.00,
            "PROBE": 0.30,
            "BUILD": 0.75,
            "FULL_PACE": 1.00,
        }
    ).astype(float)


def safe_mean(
    values: pd.Series,
) -> float:
    if values.empty:
        return float("nan")

    return float(values.mean())


def build_ablation_summary(
    data: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    variants: dict[str, pd.Series] = {
        "BASELINE": data["score"].copy()
    }

    for component, weight in (
        COMPONENT_WEIGHTS.items()
    ):
        ablated_score = (
            (
                data["score"]
                - (
                    data[component]
                    * weight
                )
            )
            / (1.0 - weight)
        ).clip(
            lower=0.0,
            upper=100.0,
        )

        variant_name = (
            "WITHOUT_"
            + component.upper()
        )

        variants[variant_name] = (
            ablated_score
        )

    for variant_name, variant_score in (
        variants.items()
    ):
        variant_band = scores_to_bands(
            variant_score
        )
        target_fraction = (
            bands_to_target_fraction(
                variant_band
            )
        )

        for window, group_index in (
            data.groupby(
                "window",
                sort=False,
            ).groups.items()
        ):
            index = pd.Index(group_index)

            window_return = data.loc[
                index,
                "net_return_24h_percent",
            ]
            window_band = variant_band.loc[
                index
            ]
            window_target = target_fraction.loc[
                index
            ]

            active_mask = window_band.ne(
                "CASH"
            )

            baseline_full_mask = (
                data.loc[index, "band"]
                .eq("FULL_PACE")
            )
            profitable_baseline_full_mask = (
                baseline_full_mask
                & window_return.gt(0.0)
            )

            retained_full_mask = (
                baseline_full_mask
                & window_band.eq(
                    "FULL_PACE"
                )
            )

            retained_profitable_full_mask = (
                profitable_baseline_full_mask
                & window_band.eq(
                    "FULL_PACE"
                )
            )

            baseline_full_count = int(
                baseline_full_mask.sum()
            )
            profitable_baseline_full_count = int(
                profitable_baseline_full_mask.sum()
            )

            rows.append(
                {
                    "variant": variant_name,
                    "window": window,
                    "observations": int(
                        len(index)
                    ),
                    "active_observations": int(
                        active_mask.sum()
                    ),
                    "active_rate_percent": float(
                        active_mask.mean()
                        * 100.0
                    ),
                    "diagnostic_exposure_weighted_net_24h_percent": float(
                        (
                            window_target
                            * window_return
                        ).mean()
                    ),
                    "active_mean_net_24h_percent": (
                        safe_mean(
                            window_return.loc[
                                active_mask
                            ]
                        )
                    ),
                    "active_positive_rate_percent": float(
                        (
                            window_return.loc[
                                active_mask
                            ]
                            > 0.0
                        ).mean()
                        * 100.0
                    )
                    if active_mask.any()
                    else float("nan"),
                    "probe_mean_net_24h_percent": (
                        safe_mean(
                            window_return.loc[
                                window_band.eq(
                                    "PROBE"
                                )
                            ]
                        )
                    ),
                    "build_mean_net_24h_percent": (
                        safe_mean(
                            window_return.loc[
                                window_band.eq(
                                    "BUILD"
                                )
                            ]
                        )
                    ),
                    "full_pace_mean_net_24h_percent": (
                        safe_mean(
                            window_return.loc[
                                window_band.eq(
                                    "FULL_PACE"
                                )
                            ]
                        )
                    ),
                    "original_full_pace_retention_percent": (
                        float(
                            retained_full_mask.sum()
                            / baseline_full_count
                            * 100.0
                        )
                        if baseline_full_count > 0
                        else float("nan")
                    ),
                    "original_profitable_full_pace_retention_percent": (
                        float(
                            retained_profitable_full_mask.sum()
                            / profitable_baseline_full_count
                            * 100.0
                        )
                        if profitable_baseline_full_count
                        > 0
                        else float("nan")
                    ),
                }
            )

    return pd.DataFrame(rows)


def print_compact_results(
    symbol_band: pd.DataFrame,
    transitions: pd.DataFrame,
    component_contrast: pd.DataFrame,
    ablation: pd.DataFrame,
) -> None:
    print("")
    print(
        "========== 2025H1 BUILD DAMAGE BY SYMBOL =========="
    )

    build_damage = (
        symbol_band.loc[
            (
                symbol_band["window"]
                == "2025H1"
            )
            & (
                symbol_band["band"]
                == "BUILD"
            )
        ]
        .sort_values(
            "mean_net_24h_percent"
        )
    )

    print(
        build_damage[
            [
                "symbol",
                "observations",
                "mean_net_24h_percent",
                "positive_net_24h_rate_percent",
                "mean_mfe_24h_percent",
                "mean_mae_24h_percent",
            ]
        ].to_string(
            index=False,
            float_format=(
                lambda value: f"{value:.4f}"
            ),
        )
    )

    print("")
    print(
        "========== 2025H1 TRANSITIONS INTO BUILD =========="
    )

    build_transitions = (
        transitions.loc[
            (
                transitions["window"]
                == "2025H1"
            )
            & (
                transitions["transition"]
                .astype(str)
                .str.endswith("->BUILD")
            )
            & (
                transitions["observations"]
                >= 100
            )
        ]
        .sort_values(
            "mean_net_24h_percent"
        )
    )

    print(
        build_transitions[
            [
                "transition",
                "observations",
                "mean_score_delta_15m",
                "mean_score_delta_1h",
                "mean_net_24h_percent",
                "positive_net_24h_rate_percent",
                "mean_mfe_24h_percent",
                "mean_mae_24h_percent",
            ]
        ].to_string(
            index=False,
            float_format=(
                lambda value: f"{value:.4f}"
            ),
        )
    )

    print("")
    print(
        "========== COMPONENT EFFECT: 2025H1 BUILD =========="
    )

    build_components = (
        component_contrast.loc[
            (
                component_contrast["window"]
                == "2025H1"
            )
            & (
                component_contrast["band"]
                == "BUILD"
            )
        ]
        .sort_values(
            "top_minus_bottom_net_24h_percent"
        )
    )

    print(
        build_components[
            [
                "component",
                "winner_minus_loser_component",
                "bottom_quartile_mean_net_24h_percent",
                "top_quartile_mean_net_24h_percent",
                "top_minus_bottom_net_24h_percent",
            ]
        ].to_string(
            index=False,
            float_format=(
                lambda value: f"{value:.4f}"
            ),
        )
    )

    print("")
    print(
        "========== COMPONENT EFFECT: 2024 FULL_PACE =========="
    )

    profitable_components = (
        component_contrast.loc[
            (
                component_contrast["window"]
                == "2024"
            )
            & (
                component_contrast["band"]
                == "FULL_PACE"
            )
        ]
        .sort_values(
            "top_minus_bottom_net_24h_percent",
            ascending=False,
        )
    )

    print(
        profitable_components[
            [
                "component",
                "winner_minus_loser_component",
                "bottom_quartile_mean_net_24h_percent",
                "top_quartile_mean_net_24h_percent",
                "top_minus_bottom_net_24h_percent",
            ]
        ].to_string(
            index=False,
            float_format=(
                lambda value: f"{value:.4f}"
            ),
        )
    )

    print("")
    print(
        "========== CONTROLLED ABLATION MAP =========="
    )
    print(
        "Diagnostic weighted value is not a portfolio return."
    )

    pivot = ablation.pivot(
        index="variant",
        columns="window",
        values=(
            "diagnostic_exposure_weighted_net_24h_percent"
        ),
    ).reset_index()

    ordered_columns = [
        "variant",
        "2022",
        "2023",
        "2024",
        "2025H1",
    ]

    for column in ordered_columns:
        if column not in pivot.columns:
            pivot[column] = np.nan

    pivot = pivot[ordered_columns]

    pivot["_order"] = np.where(
        pivot["variant"].eq("BASELINE"),
        0,
        1,
    )

    pivot = (
        pivot.sort_values(
            ["_order", "variant"]
        )
        .drop(columns=["_order"])
    )

    print(
        pivot.to_string(
            index=False,
            float_format=(
                lambda value: f"{value:.5f}"
            ),
        )
    )


def main() -> None:
    data = load_detail_reports()
    data = add_sequence_features(data)

    shock_normal = data.loc[
        data["shock_normal"]
    ].copy()

    symbol_band = aggregate_outcomes(
        shock_normal,
        [
            "window",
            "symbol",
            "band",
        ],
    )

    transitions = aggregate_outcomes(
        shock_normal.loc[
            shock_normal["previous_band"]
            != "START"
        ],
        [
            "window",
            "transition",
        ],
    )

    band_age = aggregate_outcomes(
        shock_normal,
        [
            "window",
            "band",
            "band_age_bucket",
        ],
    )

    score_direction = aggregate_outcomes(
        shock_normal.dropna(
            subset=["score_direction"]
        ),
        [
            "window",
            "band",
            "score_direction",
        ],
    )

    component_contrast = (
        build_component_contrast(
            shock_normal
        )
    )

    ablation = build_ablation_summary(
        shock_normal
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    symbol_band.to_csv(
        OUTPUT_DIR
        / "v3kn_symbol_band_weakness.csv",
        index=False,
    )

    transitions.to_csv(
        OUTPUT_DIR
        / "v3kn_transition_weakness.csv",
        index=False,
    )

    band_age.to_csv(
        OUTPUT_DIR
        / "v3kn_band_age_weakness.csv",
        index=False,
    )

    score_direction.to_csv(
        OUTPUT_DIR
        / "v3kn_score_direction_weakness.csv",
        index=False,
    )

    component_contrast.to_csv(
        OUTPUT_DIR
        / "v3kn_component_contrast.csv",
        index=False,
    )

    ablation.to_csv(
        OUTPUT_DIR
        / "v3kn_ablation_summary.csv",
        index=False,
    )

    print(
        f"Loaded observations: {len(data):,}"
    )
    print(
        "Shock-normal observations: "
        f"{len(shock_normal):,}"
    )

    print_compact_results(
        symbol_band,
        transitions,
        component_contrast,
        ablation,
    )

    print("")
    print(
        "V3KN weakness-map reports saved."
    )


if __name__ == "__main__":
    main()
