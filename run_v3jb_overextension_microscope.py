from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


REPORT_DIR = Path("reports")
SYMBOL = "LINKUSDT"

WINDOW_ORDER = [
    "2021",
    "2022",
    "2023",
    "2024",
    "2025h1",
]


def normalize_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series

    return (
        series.astype(str)
        .str.lower()
        .eq("true")
    )


def regime_from_rank(value: float) -> str:
    if pd.isna(value):
        return "UNKNOWN"

    if value <= (1.0 / 3.0):
        return "LOW"

    if value <= (2.0 / 3.0):
        return "MID"

    return "HIGH"


def classify_mfe(value: float) -> str:
    if value < 0.5:
        return "FALSE_<0.5R"

    if value >= 2.0:
        return "RUNNER_>=2R"

    return "MIDDLE_0.5-2R"


def enrich_window() -> None:
    window = os.environ.get(
        "V3G4_WINDOW",
        "",
    ).strip()

    if not window:
        raise RuntimeError(
            "V3G4_WINDOW is required."
        )

    window_key = window.lower()

    input_path = (
        REPORT_DIR
        / f"v3j_fresh_ignition_{window_key}_detail.csv"
    )

    if not input_path.exists():
        raise FileNotFoundError(
            input_path
        )

    detail = pd.read_csv(
        input_path
    )

    detail["decision_time"] = pd.to_datetime(
        detail["decision_time"]
    )

    for column in [
        "fresh_zone",
        "current_12h_selected",
        "fresh_12h_selected",
    ]:
        detail[column] = normalize_bool(
            detail[column]
        )

    # Import only after V3G4_WINDOW is fixed.
    import run_v3g4_eth365_robustness as engine

    print()
    print("=" * 110)
    print(
        f"V3J-B OVEREXTENSION MICROSCOPE - {window}"
    )
    print("=" * 110)
    print(
        "Diagnostic only: no strategy thresholds changed."
    )

    (
        _frames_15m,
        frames_1h,
        _common_15m,
        _common_1h,
    ) = engine.prepare_frames()

    link = frames_1h[
        SYMBOL
    ].copy()

    if "ADX14" not in link.columns:
        raise RuntimeError(
            "ADX14 missing from LINK 1h frame."
        )

    if (
        "EMA50_SLOPE_12H"
        not in link.columns
    ):
        link[
            "EMA50_SLOPE_12H"
        ] = (
            link["EMA50"]
            .pct_change(12)
            * 100.0
        )

    feature_table = link[
        [
            "ADX14",
            "EMA50_SLOPE_12H",
        ]
    ].copy()

    feature_table.index = pd.to_datetime(
        feature_table.index
    )

    feature_table = feature_table.rename(
        columns={
            "ADX14": "adx14",
            "EMA50_SLOPE_12H": (
                "ema50_slope_12h"
            ),
        }
    )

    detail = detail.merge(
        feature_table,
        left_on="decision_time",
        right_index=True,
        how="left",
        validate="many_to_one",
    )

    missing = detail[
        [
            "adx14",
            "ema50_slope_12h",
        ]
    ].isna().any(axis=1)

    if missing.any():
        raise RuntimeError(
            f"{int(missing.sum())} signals "
            "missing ADX/slope features."
        )

    # Within-window percentile ranks.
    # These are diagnostic ranks only.
    # No absolute strategy thresholds are fitted.
    detail["adx_rank"] = (
        detail["adx14"]
        .rank(
            method="average",
            pct=True,
        )
    )

    detail["slope_rank"] = (
        detail["ema50_slope_12h"]
        .rank(
            method="average",
            pct=True,
        )
    )

    # Simple pre-locked diagnostic composite.
    detail["hotness_rank"] = (
        (
            detail["adx_rank"]
            + detail["slope_rank"]
        )
        / 2.0
    )

    detail["adx_regime"] = (
        detail["adx_rank"]
        .map(regime_from_rank)
    )

    detail["slope_regime"] = (
        detail["slope_rank"]
        .map(regime_from_rank)
    )

    detail["hotness_regime"] = (
        detail["hotness_rank"]
        .map(regime_from_rank)
    )

    if "mfe_class" not in detail.columns:
        detail["mfe_class"] = (
            detail["mfe_r"]
            .map(classify_mfe)
        )

    # Same-selection comparison:
    # freshness among CURRENT_12H signals.
    detail[
        "current_fresh"
    ] = (
        detail["current_12h_selected"]
        & detail["fresh_zone"]
    )

    detail[
        "current_stale"
    ] = (
        detail["current_12h_selected"]
        & ~detail["fresh_zone"]
    )

    output_path = (
        REPORT_DIR
        / (
            "v3jb_overextension_"
            f"{window_key}_detail.csv"
        )
    )

    detail.to_csv(
        output_path,
        index=False,
    )

    print(
        f"Eligible signals : {len(detail)}"
    )
    print(
        "CURRENT_12H     : "
        f"{int(detail['current_12h_selected'].sum())}"
    )
    print(
        "Current fresh   : "
        f"{int(detail['current_fresh'].sum())}"
    )
    print(
        "Current stale   : "
        f"{int(detail['current_stale'].sum())}"
    )
    print(
        "FRESH_POLICY_12H: "
        f"{int(detail['fresh_12h_selected'].sum())}"
    )
    print()
    print(
        f"Saved: {output_path}"
    )


def metrics(
    frame: pd.DataFrame,
) -> dict[str, float | int]:
    if frame.empty:
        return {
            "signals": 0,
            "false_pct": np.nan,
            "runner_pct": np.nan,
            "median_mfe_r": np.nan,
            "median_mae_r": np.nan,
            "stopped_pct": np.nan,
        }

    return {
        "signals": len(frame),
        "false_pct": float(
            (
                frame["mfe_r"]
                < 0.5
            ).mean()
            * 100.0
        ),
        "runner_pct": float(
            (
                frame["mfe_r"]
                >= 2.0
            ).mean()
            * 100.0
        ),
        "median_mfe_r": float(
            frame["mfe_r"].median()
        ),
        "median_mae_r": float(
            frame["mae_r"].median()
        ),
        "stopped_pct": float(
            frame[
                "stopped_before_horizon"
            ].mean()
            * 100.0
        ),
    }


def summarize_all() -> None:
    paths = [
        REPORT_DIR
        / (
            "v3jb_overextension_"
            f"{window}_detail.csv"
        )
        for window in WINDOW_ORDER
    ]

    missing = [
        str(path)
        for path in paths
        if not path.exists()
    ]

    if missing:
        raise RuntimeError(
            "Missing V3J-B files:\n"
            + "\n".join(missing)
        )

    frames = []

    for path in paths:
        frame = pd.read_csv(
            path
        )

        frame["window"] = (
            frame["window"]
            .astype(str)
            .str.lower()
        )

        for column in [
            "fresh_zone",
            "current_12h_selected",
            "fresh_12h_selected",
            "current_fresh",
            "current_stale",
            "stopped_before_horizon",
        ]:
            frame[column] = normalize_bool(
                frame[column]
            )

        frames.append(
            frame
        )

    detail = pd.concat(
        frames,
        ignore_index=True,
    )

    cohorts = {
        "CURRENT_12H": (
            "current_12h_selected"
        ),
        "CURRENT_FRESH": (
            "current_fresh"
        ),
        "CURRENT_STALE": (
            "current_stale"
        ),
        "FRESH_POLICY_12H": (
            "fresh_12h_selected"
        ),
    }

    regime_columns = {
        "ADX": "adx_regime",
        "EMA50_SLOPE": (
            "slope_regime"
        ),
        "HOTNESS": (
            "hotness_regime"
        ),
    }

    summary_rows = []

    windows = (
        WINDOW_ORDER
        + ["ALL"]
    )

    for window in windows:
        if window == "ALL":
            window_frame = detail
        else:
            window_frame = detail[
                detail["window"]
                == window
            ]

        for cohort_name, flag in cohorts.items():
            cohort = window_frame[
                window_frame[flag]
            ]

            for feature_name, regime_column in (
                regime_columns.items()
            ):
                for regime in [
                    "LOW",
                    "MID",
                    "HIGH",
                ]:
                    cell = cohort[
                        cohort[
                            regime_column
                        ]
                        == regime
                    ]

                    summary_rows.append(
                        {
                            "window": window,
                            "cohort": cohort_name,
                            "feature": (
                                feature_name
                            ),
                            "regime": regime,
                            **metrics(cell),
                        }
                    )

    summary = pd.DataFrame(
        summary_rows
    )

    summary_path = (
        REPORT_DIR
        / "v3jb_overextension_regime_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    corr_rows = []

    for window in windows:
        if window == "ALL":
            window_frame = detail
        else:
            window_frame = detail[
                detail["window"]
                == window
            ]

        for cohort_name, flag in cohorts.items():
            cohort = window_frame[
                window_frame[flag]
            ].copy()

            for feature in [
                "adx_rank",
                "slope_rank",
                "hotness_rank",
            ]:
                if len(cohort) < 3:
                    correlation = np.nan
                else:
                    correlation = (
                        cohort[
                            [
                                feature,
                                "mfe_r",
                            ]
                        ]
                        .corr(
                            method="spearman"
                        )
                        .iloc[0, 1]
                    )

                corr_rows.append(
                    {
                        "window": window,
                        "cohort": cohort_name,
                        "feature": feature,
                        "signals": len(cohort),
                        "spearman_mfe": (
                            correlation
                        ),
                    }
                )

    correlations = pd.DataFrame(
        corr_rows
    )

    correlation_path = (
        REPORT_DIR
        / "v3jb_overextension_correlations.csv"
    )

    correlations.to_csv(
        correlation_path,
        index=False,
    )

    print()
    print("=" * 120)
    print(
        "V3J-B - HOTNESS REGIME SUMMARY"
    )
    print("=" * 120)

    focus = summary[
        (
            summary["feature"]
            == "HOTNESS"
        )
        & (
            summary["cohort"].isin(
                [
                    "CURRENT_FRESH",
                    "CURRENT_STALE",
                    "FRESH_POLICY_12H",
                ]
            )
        )
    ]

    print(
        focus.to_string(
            index=False
        )
    )

    print()
    print("=" * 120)
    print(
        "V3J-B - FRESH POLICY CORRELATIONS"
    )
    print("=" * 120)

    fresh_corr = correlations[
        correlations["cohort"]
        == "FRESH_POLICY_12H"
    ]

    print(
        fresh_corr.to_string(
            index=False
        )
    )

    print()
    print(
        f"Summary      : {summary_path}"
    )
    print(
        f"Correlations : {correlation_path}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--summarize",
        action="store_true",
    )

    args = parser.parse_args()

    if args.summarize:
        summarize_all()
    else:
        enrich_window()


if __name__ == "__main__":
    main()
