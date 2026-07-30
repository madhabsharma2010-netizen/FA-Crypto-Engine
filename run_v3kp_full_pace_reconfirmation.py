from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from run_v3ko_slow_fast_interaction import (
    BAND_LEVEL,
    FAST_COMPONENTS,
    FAST_WEIGHT,
    SLOW_COMPONENTS,
    SLOW_WEIGHT,
    load_reports,
    score_to_band,
    weighted_component_score,
)


OUTPUT_DIR = Path("reports")

WINDOW_ORDER = {
    "2022": 0,
    "2023": 1,
    "2024": 2,
    "2025H1": 3,
}

STATE_ORDER = {
    "CLEAN_PATH": 0,
    "PENDING_FAST": 1,
    "RECONFIRMED_FAST": 2,
}

RETURN_COLUMNS = {
    "4h": "net_return_4h_percent",
    "12h": "net_return_12h_percent",
    "24h": "net_return_24h_percent",
}


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


def add_interaction_features(
    data: pd.DataFrame,
) -> pd.DataFrame:
    result = data.copy()

    result["target_fraction"] = pd.to_numeric(
        result["target_fraction"],
        errors="coerce",
    )

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

    result["fast_band"] = score_to_band(
        result["fast_score"]
    )

    result["slow_band"] = score_to_band(
        result["slow_score"]
    )

    result["fast_band_level"] = (
        result["fast_band"]
        .map(BAND_LEVEL)
        .astype(int)
    )

    result["slow_band_level"] = (
        result["slow_band"]
        .map(BAND_LEVEL)
        .astype(int)
    )

    result["slow_fast_gap"] = (
        result["slow_score"]
        - result["fast_score"]
    )

    result["stale_build_candidate"] = (
        result["band"].eq("BUILD")
        & result["slow_band_level"].gt(
            result["fast_band_level"]
        )
    )

    groups = result.groupby(
        [
            "window",
            "symbol",
        ],
        sort=False,
    )

    recent_conditions: list[pd.Series] = []

    for lag in range(1, 5):
        lag_stale = (
            groups["stale_build_candidate"]
            .shift(lag)
            .fillna(False)
            .astype(bool)
        )

        lag_time = (
            groups["decision_time"]
            .shift(lag)
        )

        elapsed = (
            result["decision_time"]
            - lag_time
        )

        within_60_minutes = (
            elapsed.gt(
                pd.Timedelta(0)
            )
            & elapsed.le(
                pd.Timedelta(minutes=60)
            )
        )

        recent_conditions.append(
            lag_stale
            & within_60_minutes
        )

    result["recent_stale_build_60m"] = (
        pd.concat(
            recent_conditions,
            axis=1,
        )
        .any(axis=1)
    )

    result["full_pace_state"] = (
        "NOT_FULL_PACE"
    )

    full_pace = result["band"].eq(
        "FULL_PACE"
    )

    clean_path = (
        full_pace
        & ~result[
            "recent_stale_build_60m"
        ]
    )

    pending_fast = (
        full_pace
        & result[
            "recent_stale_build_60m"
        ]
        & result["fast_band"].ne(
            "FULL_PACE"
        )
    )

    reconfirmed_fast = (
        full_pace
        & result[
            "recent_stale_build_60m"
        ]
        & result["fast_band"].eq(
            "FULL_PACE"
        )
    )

    result.loc[
        clean_path,
        "full_pace_state",
    ] = "CLEAN_PATH"

    result.loc[
        pending_fast,
        "full_pace_state",
    ] = "PENDING_FAST"

    result.loc[
        reconfirmed_fast,
        "full_pace_state",
    ] = "RECONFIRMED_FAST"

    result["candidate_cap"] = (
        pending_fast
    )

    result["baseline_fraction"] = (
        result["target_fraction"]
    )

    result["candidate_fraction"] = (
        result["baseline_fraction"]
    )

    result.loc[
        result["candidate_cap"],
        "candidate_fraction",
    ] = 0.75

    return result


def build_state_summary(
    data: pd.DataFrame,
) -> pd.DataFrame:
    full_pace = data.loc[
        data["band"].eq("FULL_PACE")
    ].copy()

    return (
        full_pace.groupby(
            [
                "window",
                "full_pace_state",
            ],
            observed=True,
            sort=False,
        )
        .agg(
            observations=(
                "net_return_24h_percent",
                "size",
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


def build_cap_impact(
    data: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for window, window_data in (
        data.groupby(
            "window",
            sort=False,
        )
    ):
        active = window_data.loc[
            window_data[
                "baseline_fraction"
            ].gt(0.0)
        ].copy()

        full_pace = active.loc[
            active["band"].eq(
                "FULL_PACE"
            )
        ]

        capped = active.loc[
            active["candidate_cap"]
        ]

        positive_full = full_pace.loc[
            full_pace[
                "net_return_24h_percent"
            ].gt(0.0)
        ]

        negative_full = full_pace.loc[
            full_pace[
                "net_return_24h_percent"
            ].le(0.0)
        ]

        row: dict[str, object] = {
            "window": window,
            "active_observations": int(
                len(active)
            ),
            "full_pace_observations": int(
                len(full_pace)
            ),
            "capped_observations": int(
                len(capped)
            ),
            "capped_full_pace_rate_percent": (
                float(
                    len(capped)
                    / len(full_pace)
                    * 100.0
                )
                if len(full_pace) > 0
                else float("nan")
            ),
            "capped_positive_rate_percent": (
                positive_rate(
                    capped[
                        "net_return_24h_percent"
                    ]
                )
            ),
            "capped_mean_mfe_24h_percent": (
                safe_mean(
                    capped[
                        "mfe_24h_percent"
                    ]
                )
            ),
            "capped_mean_mae_24h_percent": (
                safe_mean(
                    capped[
                        "mae_24h_percent"
                    ]
                )
            ),
            "positive_full_pace_exposure_retention_percent": (
                float(
                    positive_full[
                        "candidate_fraction"
                    ].sum()
                    / positive_full[
                        "baseline_fraction"
                    ].sum()
                    * 100.0
                )
                if not positive_full.empty
                else float("nan")
            ),
            "negative_full_pace_exposure_retention_percent": (
                float(
                    negative_full[
                        "candidate_fraction"
                    ].sum()
                    / negative_full[
                        "baseline_fraction"
                    ].sum()
                    * 100.0
                )
                if not negative_full.empty
                else float("nan")
            ),
        }

        for label, return_column in (
            RETURN_COLUMNS.items()
        ):
            baseline_value = float(
                (
                    active[
                        "baseline_fraction"
                    ]
                    * active[return_column]
                ).mean()
            )

            candidate_value = float(
                (
                    active[
                        "candidate_fraction"
                    ]
                    * active[return_column]
                ).mean()
            )

            row[
                f"capped_mean_net_{label}_percent"
            ] = safe_mean(
                capped[return_column]
            )

            row[
                f"baseline_weighted_net_{label}_percent"
            ] = baseline_value

            row[
                f"candidate_weighted_net_{label}_percent"
            ] = candidate_value

            row[
                f"candidate_minus_baseline_{label}_percent"
            ] = (
                candidate_value
                - baseline_value
            )

        rows.append(row)

    return pd.DataFrame(rows)


def aggregate_pending(
    data: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    pending = data.loc[
        data["candidate_cap"]
    ].copy()

    return (
        pending.groupby(
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

    data = add_interaction_features(
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

    state_summary = (
        build_state_summary(
            shock_normal
        )
    )

    cap_impact = build_cap_impact(
        shock_normal
    )

    pending_symbol = aggregate_pending(
        shock_normal,
        [
            "window",
            "symbol",
        ],
    )

    pending_market_state = (
        aggregate_pending(
            shock_normal,
            [
                "window",
                "market_state",
            ],
        )
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    state_summary.to_csv(
        OUTPUT_DIR
        / "v3kp_reconfirmation_state_summary.csv",
        index=False,
    )

    cap_impact.to_csv(
        OUTPUT_DIR
        / "v3kp_cap_impact_summary.csv",
        index=False,
    )

    pending_symbol.to_csv(
        OUTPUT_DIR
        / "v3kp_pending_by_symbol.csv",
        index=False,
    )

    pending_market_state.to_csv(
        OUTPUT_DIR
        / "v3kp_pending_by_market_state.csv",
        index=False,
    )

    state_summary["_window_order"] = (
        state_summary["window"]
        .map(WINDOW_ORDER)
    )

    state_summary["_state_order"] = (
        state_summary[
            "full_pace_state"
        ].map(STATE_ORDER)
    )

    state_summary = (
        state_summary.sort_values(
            [
                "_window_order",
                "_state_order",
            ]
        )
    )

    cap_impact["_window_order"] = (
        cap_impact["window"]
        .map(WINDOW_ORDER)
    )

    cap_impact = cap_impact.sort_values(
        "_window_order"
    )

    symbol_focus = (
        pending_symbol.loc[
            pending_symbol["window"].isin(
                [
                    "2024",
                    "2025H1",
                ]
            )
            & pending_symbol[
                "observations"
            ].ge(50)
        ]
        .copy()
    )

    symbol_focus["_window_order"] = (
        symbol_focus["window"]
        .map(WINDOW_ORDER)
    )

    symbol_focus = (
        symbol_focus.sort_values(
            [
                "_window_order",
                "mean_net_24h_percent",
            ]
        )
    )

    market_focus = (
        pending_market_state.loc[
            pending_market_state[
                "observations"
            ].ge(50)
        ]
        .copy()
    )

    market_focus["_window_order"] = (
        market_focus["window"]
        .map(WINDOW_ORDER)
    )

    market_focus = (
        market_focus.sort_values(
            [
                "_window_order",
                "mean_net_24h_percent",
            ]
        )
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
        "Maximum score reconstruction error: "
        f"{maximum_error:.8f}"
    )

    print_table(
        "========== FULL_PACE RECONFIRMATION STATES ==========",
        state_summary,
        [
            "window",
            "full_pace_state",
            "observations",
            "mean_fast_score",
            "mean_slow_score",
            "mean_slow_fast_gap",
            "mean_net_4h_percent",
            "mean_net_12h_percent",
            "mean_net_24h_percent",
            "positive_net_24h_rate_percent",
            "mean_mfe_24h_percent",
            "mean_mae_24h_percent",
        ],
    )

    print_table(
        "========== TEMPORARY 75 PERCENT CAP IMPACT ==========",
        cap_impact,
        [
            "window",
            "full_pace_observations",
            "capped_observations",
            "capped_full_pace_rate_percent",
            "capped_mean_net_4h_percent",
            "capped_mean_net_12h_percent",
            "capped_mean_net_24h_percent",
            "baseline_weighted_net_24h_percent",
            "candidate_weighted_net_24h_percent",
            "candidate_minus_baseline_24h_percent",
            "positive_full_pace_exposure_retention_percent",
            "negative_full_pace_exposure_retention_percent",
        ],
    )

    print_table(
        "========== PENDING FAST BY SYMBOL: 2024 AND 2025H1 ==========",
        symbol_focus,
        [
            "window",
            "symbol",
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

    print_table(
        "========== PENDING FAST BY MARKET STATE ==========",
        market_focus,
        [
            "window",
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
        "V3KP reconfirmation reports saved."
    )

    print(
        "The 75 percent cap remains diagnostic; "
        "no strategy code was changed."
    )


if __name__ == "__main__":
    main()
