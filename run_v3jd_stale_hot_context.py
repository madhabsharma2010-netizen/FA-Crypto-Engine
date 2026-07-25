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

FEATURES = [
    "volume_ratio",
    "momentum_24h",
    "momentum_72h",
    "ema20_distance_atr",
    "ema50_distance_atr",
    "ema200_distance_atr",
]


def normalize_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series

    return (
        series.astype(str)
        .str.lower()
        .eq("true")
    )


def classify_mfe(value: float) -> str:
    if value < 0.5:
        return "FALSE_<0.5R"

    if value >= 2.0:
        return "RUNNER_>=2R"

    return "MIDDLE_0.5-2R"


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

    return float(
        (greater - lower)
        / (len(x) * len(y))
    )


def extract_window() -> None:
    window = os.environ.get(
        "V3G4_WINDOW",
        "",
    ).strip()

    if not window:
        raise RuntimeError(
            "V3G4_WINDOW is required."
        )

    window_key = window.lower()

    source = (
        REPORT_DIR
        / "v3jc_causal_stale_hot_detail.csv"
    )

    if not source.exists():
        raise FileNotFoundError(
            source
        )

    detail = pd.read_csv(
        source
    )

    detail["window"] = (
        detail["window"]
        .astype(str)
        .str.lower()
    )

    detail = detail[
        detail["window"]
        == window_key
    ].copy()

    if detail.empty:
        raise RuntimeError(
            f"No V3J-C detail for {window}."
        )

    detail["decision_time"] = pd.to_datetime(
        detail["decision_time"]
    )

    for column in [
        "fresh_zone",
        "current_12h_selected",
        "v3jc_stale_hot_reject",
        "reached_1r",
        "reached_2r",
        "stopped_before_horizon",
    ]:
        detail[column] = normalize_bool(
            detail[column]
        )

    # IMPORTANT:
    # These are the stale-hot signals that were
    # actually selected by the CURRENT 12h policy.
    detail[
        "baseline_stale_hot_removed"
    ] = (
        detail["current_12h_selected"]
        & detail["v3jc_stale_hot_reject"]
    )

    # Import after V3G4_WINDOW is fixed.
    import run_v3g4_eth365_robustness as engine

    print()
    print("=" * 112)
    print(
        f"V3J-D STALE-HOT CONTEXT MICROSCOPE - {window}"
    )
    print("=" * 112)
    print(
        "Diagnostic only: actual baseline-selected stale-hot signals."
    )
    print(
        "No strategy thresholds, risk, stops, exits or SOL changed."
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

    # Rebuild only causal entry-time features.
    if "MOMENTUM_24H" not in link.columns:
        link["MOMENTUM_24H"] = (
            link["close"]
            .pct_change(24)
            * 100.0
        )

    if "MOMENTUM_72H" not in link.columns:
        link["MOMENTUM_72H"] = (
            link["close"]
            .pct_change(72)
            * 100.0
        )

    if "VOLUME_RATIO" not in link.columns:
        link["VOLUME_RATIO"] = (
            link["volume"]
            / link["VolumeSMA20"]
        )

    link[
        "EMA20_DISTANCE_ATR"
    ] = (
        (
            link["close"]
            - link["EMA20"]
        )
        / link["ATR14"]
    )

    link[
        "EMA50_DISTANCE_ATR"
    ] = (
        (
            link["close"]
            - link["EMA50"]
        )
        / link["ATR14"]
    )

    link[
        "EMA200_DISTANCE_ATR"
    ] = (
        (
            link["close"]
            - link["EMA200"]
        )
        / link["ATR14"]
    )

    feature_table = link[
        [
            "VOLUME_RATIO",
            "MOMENTUM_24H",
            "MOMENTUM_72H",
            "EMA20_DISTANCE_ATR",
            "EMA50_DISTANCE_ATR",
            "EMA200_DISTANCE_ATR",
        ]
    ].copy()

    feature_table = feature_table.rename(
        columns={
            "VOLUME_RATIO": "volume_ratio",
            "MOMENTUM_24H": "momentum_24h",
            "MOMENTUM_72H": "momentum_72h",
            "EMA20_DISTANCE_ATR": (
                "ema20_distance_atr"
            ),
            "EMA50_DISTANCE_ATR": (
                "ema50_distance_atr"
            ),
            "EMA200_DISTANCE_ATR": (
                "ema200_distance_atr"
            ),
        }
    )

    feature_table.index = pd.to_datetime(
        feature_table.index
    )

    detail = detail.merge(
        feature_table,
        left_on="decision_time",
        right_index=True,
        how="left",
        validate="many_to_one",
    )

    missing = detail[
        FEATURES
    ].isna().any(axis=1)

    if missing.any():
        raise RuntimeError(
            f"{int(missing.sum())} signals "
            "missing entry-time features."
        )

    detail["outcome_class"] = (
        detail["mfe_r"]
        .map(classify_mfe)
    )

    output = (
        REPORT_DIR
        / (
            "v3jd_stale_hot_context_"
            f"{window_key}_detail.csv"
        )
    )

    detail.to_csv(
        output,
        index=False,
    )

    removed = detail[
        detail[
            "baseline_stale_hot_removed"
        ]
    ]

    print()
    print(
        f"Eligible LINK signals                : {len(detail)}"
    )
    print(
        "CURRENT_12H signals                 : "
        f"{int(detail['current_12h_selected'].sum())}"
    )
    print(
        "Baseline-selected stale-hot removed : "
        f"{len(removed)}"
    )

    if not removed.empty:
        print(
            "  False <0.5R : "
            f"{int((removed['mfe_r'] < 0.5).sum())}"
        )
        print(
            "  Middle      : "
            f"{int(((removed['mfe_r'] >= 0.5) & (removed['mfe_r'] < 2.0)).sum())}"
        )
        print(
            "  Runner >=2R : "
            f"{int((removed['mfe_r'] >= 2.0).sum())}"
        )

    print()
    print(
        f"Saved: {output}"
    )


def summarize_all() -> None:
    paths = [
        REPORT_DIR
        / (
            "v3jd_stale_hot_context_"
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
            "Missing V3J-D files:\n"
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

        frame[
            "baseline_stale_hot_removed"
        ] = normalize_bool(
            frame[
                "baseline_stale_hot_removed"
            ]
        )

        frames.append(
            frame
        )

    detail = pd.concat(
        frames,
        ignore_index=True,
    )

    removed = detail[
        detail[
            "baseline_stale_hot_removed"
        ]
    ].copy()

    if removed.empty:
        raise RuntimeError(
            "No baseline-selected stale-hot signals."
        )

    effect_rows = []

    for window in (
        WINDOW_ORDER
        + ["ALL"]
    ):
        if window == "ALL":
            frame = removed
        else:
            frame = removed[
                removed["window"]
                == window
            ]

        false = frame[
            frame["mfe_r"] < 0.5
        ]

        runner = frame[
            frame["mfe_r"] >= 2.0
        ]

        for feature in FEATURES:
            effect_rows.append(
                {
                    "window": window,
                    "feature": feature,
                    "signals": len(frame),
                    "false_n": len(false),
                    "runner_n": len(runner),
                    "false_median": (
                        float(
                            false[
                                feature
                            ].median()
                        )
                        if len(false)
                        else np.nan
                    ),
                    "runner_median": (
                        float(
                            runner[
                                feature
                            ].median()
                        )
                        if len(runner)
                        else np.nan
                    ),
                    "cliffs_delta_runner_minus_false": (
                        cliffs_delta(
                            runner[feature],
                            false[feature],
                        )
                    ),
                }
            )

    effects = pd.DataFrame(
        effect_rows
    )

    effects_path = (
        REPORT_DIR
        / "v3jd_stale_hot_feature_effects.csv"
    )

    effects.to_csv(
        effects_path,
        index=False,
    )

    state_rows = []

    for window in (
        WINDOW_ORDER
        + ["ALL"]
    ):
        if window == "ALL":
            frame = removed
        else:
            frame = removed[
                removed["window"]
                == window
            ]

        for state, group in frame.groupby(
            "market_state",
            dropna=False,
        ):
            state_rows.append(
                {
                    "window": window,
                    "market_state": state,
                    "signals": len(group),
                    "false_pct": float(
                        (
                            group["mfe_r"]
                            < 0.5
                        ).mean()
                        * 100.0
                    ),
                    "runner_pct": float(
                        (
                            group["mfe_r"]
                            >= 2.0
                        ).mean()
                        * 100.0
                    ),
                    "median_mfe_r": float(
                        group[
                            "mfe_r"
                        ].median()
                    ),
                }
            )

    states = pd.DataFrame(
        state_rows
    )

    states_path = (
        REPORT_DIR
        / "v3jd_stale_hot_market_states.csv"
    )

    states.to_csv(
        states_path,
        index=False,
    )

    print()
    print("=" * 122)
    print(
        "V3J-D - STALE-HOT FALSE vs RUNNER ENTRY FEATURES"
    )
    print("=" * 122)

    print(
        effects.to_string(
            index=False
        )
    )

    print()
    print("=" * 122)
    print(
        "V3J-D - MARKET STATE CONTEXT"
    )
    print("=" * 122)

    print(
        states.to_string(
            index=False
        )
    )

    print()
    print(
        f"Feature effects : {effects_path}"
    )
    print(
        f"Market states   : {states_path}"
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
        extract_window()


if __name__ == "__main__":
    main()
