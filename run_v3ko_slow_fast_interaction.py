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

FAST_COMPONENTS = {
    "market_score": WEIGHT_MARKET,
    "trend_15m_score": WEIGHT_TREND_15M,
    "trend_1h_score": WEIGHT_TREND_1H,
    "relative_strength_score": (
        WEIGHT_RELATIVE_STRENGTH
    ),
    "positive_volume_score": (
        WEIGHT_POSITIVE_VOLUME
    ),
}

SLOW_COMPONENTS = {
    "trend_2h_score": WEIGHT_TREND_2H,
    "trend_4h_score": WEIGHT_TREND_4H,
}

FAST_WEIGHT = sum(
    FAST_COMPONENTS.values()
)

SLOW_WEIGHT = sum(
    SLOW_COMPONENTS.values()
)

BAND_LEVEL = {
    "CASH": 0,
    "PROBE": 1,
    "BUILD": 2,
    "FULL_PACE": 3,
}

BAND_ORDER = {
    "CASH": 0,
    "PROBE": 1,
    "BUILD": 2,
    "FULL_PACE": 3,
}

NUMERIC_COLUMNS = (
    "score",
    "market_score",
    "trend_15m_score",
    "trend_1h_score",
    "trend_2h_score",
    "trend_4h_score",
    "relative_strength_score",
    "positive_volume_score",
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


def score_to_band(
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
        (values > 0.0).mean()
        * 100.0
    )


def load_reports() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for expected_window, path in (
        INPUT_FILES.items()
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing report: {path}"
            )

        frame = pd.read_csv(
            path,
            parse_dates=["decision_time"],
        )

        frame["window"] = (
            frame["window"]
            .astype(str)
        )

        found_windows = set(
            frame["window"].unique()
        )

        if found_windows != {
            expected_window
        }:
            raise ValueError(
                f"Unexpected window in {path}: "
                f"{found_windows}"
            )

        for column in NUMERIC_COLUMNS:
            frame[column] = pd.to_numeric(
                frame[column],
                errors="coerce",
            )

        frame["shock_normal"] = (
            parse_boolean(
                frame["shock_normal"]
            )
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
            "band",
            "score",
            "net_return_24h_percent",
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


def weighted_component_score(
    data: pd.DataFrame,
    components: dict[str, float],
    total_weight: float,
) -> pd.Series:
    weighted_sum = pd.Series(
        0.0,
        index=data.index,
    )

    for component, weight in (
        components.items()
    ):
        weighted_sum = (
            weighted_sum
            + data[component] * weight
        )

    return (
        weighted_sum
        / total_weight
    ).clip(
        lower=0.0,
        upper=100.0,
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
            mean_slow_score=(
                "slow_score",
                "mean",
            ),
            mean_slow_fast_gap=(
                "slow_fast_gap",
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
            positive_net_24h_rate_percent=(
                "net_return_24h_percent",
                positive_rate,
            ),
            loss_below_minus_1_rate_percent=(
                "net_return_24h_percent",
                lambda values: float(
                    (
                        values <= -1.0
                    ).mean()
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


def build_gate_diagnostic(
    data: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for window, window_data in (
        data.groupby(
            "window",
            sort=False,
        )
    ):
        build = window_data.loc[
            window_data["band"]
            == "BUILD"
        ]

        blocked = build.loc[
            build["stale_build_candidate"]
        ]

        kept = build.loc[
            ~build["stale_build_candidate"]
        ]

        baseline_positive = int(
            (
                build[
                    "net_return_24h_percent"
                ]
                > 0.0
            ).sum()
        )

        kept_positive = int(
            (
                kept[
                    "net_return_24h_percent"
                ]
                > 0.0
            ).sum()
        )

        rows.append(
            {
                "window": window,
                "baseline_build_observations": (
                    int(len(build))
                ),
                "baseline_build_mean_net_24h_percent": (
                    safe_mean(
                        build[
                            "net_return_24h_percent"
                        ]
                    )
                ),
                "blocked_observations": (
                    int(len(blocked))
                ),
                "blocked_rate_percent": (
                    float(
                        len(blocked)
                        / len(build)
                        * 100.0
                    )
                    if len(build) > 0
                    else float("nan")
                ),
                "blocked_mean_net_24h_percent": (
                    safe_mean(
                        blocked[
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
                "kept_positive_rate_percent": (
                    positive_rate(
                        kept[
                            "net_return_24h_percent"
                        ]
                    )
                ),
                "positive_build_retention_percent": (
                    float(
                        kept_positive
                        / baseline_positive
                        * 100.0
                    )
                    if baseline_positive > 0
                    else float("nan")
                ),
            }
        )

    return pd.DataFrame(rows)


def build_full_pace_path_diagnostic(
    data: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for window, window_data in (
        data.groupby(
            "window",
            sort=False,
        )
    ):
        full_pace = window_data.loc[
            window_data["band"]
            == "FULL_PACE"
        ]

        recent_stale = full_pace.loc[
            full_pace[
                "recent_stale_build_1h"
            ]
        ]

        clean_path = full_pace.loc[
            ~full_pace[
                "recent_stale_build_1h"
            ]
        ]

        rows.append(
            {
                "window": window,
                "full_pace_observations": (
                    int(len(full_pace))
                ),
                "recent_stale_build_observations": (
                    int(len(recent_stale))
                ),
                "recent_stale_build_rate_percent": (
                    float(
                        len(recent_stale)
                        / len(full_pace)
                        * 100.0
                    )
                    if len(full_pace) > 0
                    else float("nan")
                ),
                "recent_stale_path_mean_net_24h_percent": (
                    safe_mean(
                        recent_stale[
                            "net_return_24h_percent"
                        ]
                    )
                ),
                "clean_path_mean_net_24h_percent": (
                    safe_mean(
                        clean_path[
                            "net_return_24h_percent"
                        ]
                    )
                ),
                "recent_stale_path_positive_rate_percent": (
                    positive_rate(
                        recent_stale[
                            "net_return_24h_percent"
                        ]
                    )
                ),
                "clean_path_positive_rate_percent": (
                    positive_rate(
                        clean_path[
                            "net_return_24h_percent"
                        ]
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
                lambda value: f"{value:.4f}"
            ),
        )
    )


def main() -> None:
    if not np.isclose(
        FAST_WEIGHT + SLOW_WEIGHT,
        1.0,
    ):
        raise ValueError(
            "Fast and slow weights do not "
            "sum to 1.0"
        )

    data = load_reports()

    data = data.loc[
        data["shock_normal"]
    ].copy()

    data["fast_score"] = (
        weighted_component_score(
            data,
            FAST_COMPONENTS,
            FAST_WEIGHT,
        )
    )

    data["slow_score"] = (
        weighted_component_score(
            data,
            SLOW_COMPONENTS,
            SLOW_WEIGHT,
        )
    )

    data["reconstructed_score"] = (
        data["fast_score"]
        * FAST_WEIGHT
        + data["slow_score"]
        * SLOW_WEIGHT
    )

    data["score_reconstruction_error"] = (
        data["score"]
        - data["reconstructed_score"]
    ).abs()

    data["fast_band"] = score_to_band(
        data["fast_score"]
    )

    data["slow_band"] = score_to_band(
        data["slow_score"]
    )

    data["fast_band_level"] = (
        data["fast_band"]
        .map(BAND_LEVEL)
        .astype(int)
    )

    data["slow_band_level"] = (
        data["slow_band"]
        .map(BAND_LEVEL)
        .astype(int)
    )

    data["slow_fast_gap"] = (
        data["slow_score"]
        - data["fast_score"]
    )

    data["alignment"] = np.select(
        [
            (
                data["slow_band_level"]
                > data["fast_band_level"]
            ),
            (
                data["fast_band_level"]
                > data["slow_band_level"]
            ),
        ],
        [
            "SLOW_LEADS",
            "FAST_LEADS",
        ],
        default="ALIGNED",
    )

    data["stale_build_candidate"] = (
        data["band"].eq("BUILD")
        & data["slow_band_level"].gt(
            data["fast_band_level"]
        )
    )

    grouped_stale = data.groupby(
        [
            "window",
            "symbol",
        ],
        sort=False,
    )["stale_build_candidate"]

    stale_lags = []

    for lag in range(1, 5):
        stale_lags.append(
            grouped_stale.shift(lag)
            .fillna(False)
            .astype(bool)
            .rename(
                f"stale_build_lag_{lag}"
            )
        )

    data["recent_stale_build_1h"] = (
        pd.concat(
            stale_lags,
            axis=1,
        )
        .any(axis=1)
    )

    pair_summary = aggregate_outcomes(
        data,
        [
            "window",
            "band",
            "fast_band",
            "slow_band",
        ],
    )

    alignment_summary = (
        aggregate_outcomes(
            data,
            [
                "window",
                "band",
                "alignment",
            ],
        )
    )

    gate_diagnostic = (
        build_gate_diagnostic(data)
    )

    path_diagnostic = (
        build_full_pace_path_diagnostic(
            data
        )
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    pair_summary.to_csv(
        OUTPUT_DIR
        / "v3ko_fast_slow_pair_summary.csv",
        index=False,
    )

    alignment_summary.to_csv(
        OUTPUT_DIR
        / "v3ko_alignment_summary.csv",
        index=False,
    )

    gate_diagnostic.to_csv(
        OUTPUT_DIR
        / "v3ko_build_gate_diagnostic.csv",
        index=False,
    )

    path_diagnostic.to_csv(
        OUTPUT_DIR
        / "v3ko_full_pace_path_diagnostic.csv",
        index=False,
    )

    build_2025 = pair_summary.loc[
        (
            pair_summary["window"]
            == "2025H1"
        )
        & (
            pair_summary["band"]
            == "BUILD"
        )
        & (
            pair_summary["observations"]
            >= 100
        )
    ].copy()

    build_2025["_fast_order"] = (
        build_2025["fast_band"]
        .map(BAND_ORDER)
    )

    build_2025["_slow_order"] = (
        build_2025["slow_band"]
        .map(BAND_ORDER)
    )

    build_2025 = build_2025.sort_values(
        [
            "_fast_order",
            "_slow_order",
        ]
    )

    full_2024 = pair_summary.loc[
        (
            pair_summary["window"]
            == "2024"
        )
        & (
            pair_summary["band"]
            == "FULL_PACE"
        )
        & (
            pair_summary["observations"]
            >= 100
        )
    ].copy()

    full_2024["_fast_order"] = (
        full_2024["fast_band"]
        .map(BAND_ORDER)
    )

    full_2024["_slow_order"] = (
        full_2024["slow_band"]
        .map(BAND_ORDER)
    )

    full_2024 = full_2024.sort_values(
        [
            "_fast_order",
            "_slow_order",
        ]
    )

    cross_window = alignment_summary.loc[
        alignment_summary["band"].isin(
            [
                "BUILD",
                "FULL_PACE",
            ]
        )
    ].copy()

    cross_window["_window_order"] = (
        cross_window["window"].map(
            {
                "2022": 0,
                "2023": 1,
                "2024": 2,
                "2025H1": 3,
            }
        )
    )

    cross_window["_band_order"] = (
        cross_window["band"].map(
            BAND_ORDER
        )
    )

    cross_window = (
        cross_window.sort_values(
            [
                "_window_order",
                "_band_order",
                "alignment",
            ]
        )
    )

    print(
        "Loaded shock-normal observations: "
        f"{len(data):,}"
    )

    print(
        "Maximum score reconstruction error: "
        f"{data['score_reconstruction_error'].max():.8f}"
    )

    print_table(
        "========== 2025H1 BUILD: FAST VS SLOW ==========",
        build_2025,
        [
            "fast_band",
            "slow_band",
            "observations",
            "mean_fast_score",
            "mean_slow_score",
            "mean_slow_fast_gap",
            "mean_net_24h_percent",
            "positive_net_24h_rate_percent",
            "mean_mfe_24h_percent",
            "mean_mae_24h_percent",
        ],
    )

    print_table(
        "========== 2024 FULL_PACE: FAST VS SLOW ==========",
        full_2024,
        [
            "fast_band",
            "slow_band",
            "observations",
            "mean_fast_score",
            "mean_slow_score",
            "mean_slow_fast_gap",
            "mean_net_24h_percent",
            "positive_net_24h_rate_percent",
            "mean_mfe_24h_percent",
            "mean_mae_24h_percent",
        ],
    )

    print_table(
        "========== CROSS-WINDOW ALIGNMENT ==========",
        cross_window,
        [
            "window",
            "band",
            "alignment",
            "observations",
            "mean_slow_fast_gap",
            "mean_net_24h_percent",
            "positive_net_24h_rate_percent",
            "mean_mfe_24h_percent",
            "mean_mae_24h_percent",
        ],
    )

    print_table(
        "========== BUILD GATE DIAGNOSTIC ==========",
        gate_diagnostic,
        [
            "window",
            "baseline_build_observations",
            "baseline_build_mean_net_24h_percent",
            "blocked_observations",
            "blocked_rate_percent",
            "blocked_mean_net_24h_percent",
            "kept_observations",
            "kept_mean_net_24h_percent",
            "kept_positive_rate_percent",
            "positive_build_retention_percent",
        ],
    )

    print_table(
        "========== FULL_PACE PATH PRESERVATION ==========",
        path_diagnostic,
        [
            "window",
            "full_pace_observations",
            "recent_stale_build_observations",
            "recent_stale_build_rate_percent",
            "recent_stale_path_mean_net_24h_percent",
            "clean_path_mean_net_24h_percent",
            "recent_stale_path_positive_rate_percent",
            "clean_path_positive_rate_percent",
        ],
    )

    print("")
    print(
        "V3KO diagnostic reports saved."
    )
    print(
        "Candidate gate is diagnostic only; "
        "no strategy rule was changed."
    )


if __name__ == "__main__":
    main()
