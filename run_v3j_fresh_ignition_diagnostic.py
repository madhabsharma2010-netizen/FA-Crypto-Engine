from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

SYMBOL = "LINKUSDT"
ROUTE = "IGNITION"

# PRE-LOCKED V3J-A SETTINGS
HORIZON_HOURS = 36
COOLDOWN_HOURS = 12
CURRENT_BREAK_TOLERANCE = 0.997


def classify_mfe(mfe_r: float) -> str:
    if mfe_r < 0.5:
        return "FALSE_<0.5R"

    if mfe_r >= 2.0:
        return "RUNNER_>=2R"

    return "MIDDLE_0.5-2R"


def percentage(series: pd.Series) -> float:
    if len(series) == 0:
        return 0.0

    return float(
        series.astype(bool).mean() * 100.0
    )


def fresh_zone_flags(
    frame: pd.DataFrame,
    decision_time: pd.Timestamp,
) -> tuple[bool, bool, bool, bool]:
    """
    V3J-A changes ONLY freshness.

    Existing threshold:
        close >= prior_8h_high * 0.997

    Fresh version:
        current candle enters that same zone
        while previous candle was NOT already
        in its own equivalent zone.

    actual_cross is diagnostic only.
    It is NOT used by V3J-A selection.
    """

    position = frame.index.get_loc(
        decision_time
    )

    if (
        not isinstance(position, (int, np.integer))
        or position < 1
    ):
        return False, False, False, False

    current = frame.iloc[position]
    previous = frame.iloc[position - 1]

    current_level = float(
        current["_ROUTE_HIGH_PREV_8"]
    )

    previous_level = float(
        previous["_ROUTE_HIGH_PREV_8"]
    )

    close_price = float(
        current["close"]
    )

    previous_close = float(
        previous["close"]
    )

    if not all(
        np.isfinite(
            [
                current_level,
                previous_level,
                close_price,
                previous_close,
            ]
        )
    ):
        return False, False, False, False

    current_near_high = (
        close_price
        >= current_level
        * CURRENT_BREAK_TOLERANCE
    )

    previous_near_high = (
        previous_close
        >= previous_level
        * CURRENT_BREAK_TOLERANCE
    )

    fresh_zone = (
        current_near_high
        and not previous_near_high
    )

    # Diagnostic only — NOT V3J-A filter.
    actual_cross = (
        previous_close < current_level
        and close_price >= current_level
    )

    return (
        current_near_high,
        previous_near_high,
        fresh_zone,
        actual_cross,
    )


def measure_excursion(
    frame_15m: pd.DataFrame,
    decision_time: pd.Timestamp,
    entry_price: float,
    stop_price: float,
) -> dict[str, object]:
    """
    Fixed 36h signal-quality horizon.

    Signal is followed until:
      1. initial stop is first touched, OR
      2. 36 hours elapse.

    IMPORTANT:
    If stop is touched inside a 15m candle,
    that candle's high is NOT used.

    This avoids assuming whether high or low
    occurred first inside the candle.
    """

    risk_per_unit = (
        entry_price - stop_price
    )

    if risk_per_unit <= 0:
        raise ValueError(
            "Non-positive initial risk."
        )

    horizon_end = (
        decision_time
        + pd.Timedelta(
            hours=HORIZON_HOURS
        )
    )

    path = frame_15m[
        (frame_15m.index > decision_time)
        & (frame_15m.index <= horizon_end)
    ]

    max_high = entry_price
    min_low = entry_price
    bars = 0

    stopped = False
    stop_time = pd.NaT

    for timestamp, candle in path.iterrows():

        low_price = float(
            candle["low"]
        )

        # Conservative same-candle ordering:
        # stop candle contributes neither its
        # later high nor later low excursion.
        if low_price <= stop_price:
            stopped = True
            stop_time = timestamp
            break

        high_price = float(
            candle["high"]
        )

        max_high = max(
            max_high,
            high_price,
        )

        min_low = min(
            min_low,
            low_price,
        )

        bars += 1

    mfe_price = max(
        0.0,
        max_high - entry_price,
    )

    mae_price = max(
        0.0,
        entry_price - min_low,
    )

    mfe_r = (
        mfe_price / risk_per_unit
    )

    mae_r = (
        mae_price / risk_per_unit
    )

    return {
        "mfe_r": mfe_r,
        "mae_r": mae_r,
        "mfe_class": classify_mfe(
            mfe_r
        ),
        "reached_0_5r": (
            mfe_r >= 0.5
        ),
        "reached_1r": (
            mfe_r >= 1.0
        ),
        "reached_2r": (
            mfe_r >= 2.0
        ),
        "stopped_before_horizon": (
            stopped
        ),
        "stop_time": stop_time,
        "bars_before_stop_or_horizon": (
            bars
        ),
    }


def apply_cooldown(
    detail: pd.DataFrame,
    source_column: str,
    output_column: str,
) -> None:

    detail[output_column] = False

    last_selected = None

    for index, row in detail.sort_values(
        "decision_time"
    ).iterrows():

        if not bool(
            row[source_column]
        ):
            continue

        decision_time = pd.Timestamp(
            row["decision_time"]
        )

        if (
            last_selected is None
            or decision_time
            >= last_selected
            + pd.Timedelta(
                hours=COOLDOWN_HOURS
            )
        ):
            detail.at[
                index,
                output_column,
            ] = True

            last_selected = (
                decision_time
            )


def summarize_cohort(
    frame: pd.DataFrame,
    window: str,
    cohort: str,
) -> dict[str, object]:

    if frame.empty:
        return {
            "window": window,
            "cohort": cohort,
            "signals": 0,
            "false_pct": 0.0,
            "middle_pct": 0.0,
            "runner_pct": 0.0,
            "median_mfe_r": np.nan,
            "median_mae_r": np.nan,
            "reached_1r_pct": 0.0,
            "reached_2r_pct": 0.0,
            "stopped_pct": 0.0,
            "actual_cross_pct": 0.0,
        }

    return {
        "window": window,
        "cohort": cohort,
        "signals": len(frame),
        "false_pct": float(
            (
                frame["mfe_class"]
                == "FALSE_<0.5R"
            ).mean()
            * 100.0
        ),
        "middle_pct": float(
            (
                frame["mfe_class"]
                == "MIDDLE_0.5-2R"
            ).mean()
            * 100.0
        ),
        "runner_pct": float(
            (
                frame["mfe_class"]
                == "RUNNER_>=2R"
            ).mean()
            * 100.0
        ),
        "median_mfe_r": float(
            frame["mfe_r"].median()
        ),
        "median_mae_r": float(
            frame["mae_r"].median()
        ),
        "reached_1r_pct": percentage(
            frame["reached_1r"]
        ),
        "reached_2r_pct": percentage(
            frame["reached_2r"]
        ),
        "stopped_pct": percentage(
            frame[
                "stopped_before_horizon"
            ]
        ),
        "actual_cross_pct": percentage(
            frame["actual_cross"]
        ),
    }


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

    # Import AFTER V3G4_WINDOW is set.
    import run_v3g4_eth365_robustness as engine

    print()
    print("=" * 110)
    print(
        f"V3J-A FRESH IGNITION SIGNAL DIAGNOSTIC - {window}"
    )
    print("=" * 110)
    print(
        "Policy: preserve current 0.997 breakout-zone threshold; "
        "add freshness only."
    )
    print(
        f"Future quality horizon: {HORIZON_HOURS}h | "
        f"Route cooldown: {COOLDOWN_HOURS}h"
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

    market_states = (
        engine.build_market_states(
            frames_1h,
            common_1h,
            market_shocks,
        )
    )

    print(
        "Building unchanged route signals..."
    )

    (
        route_signals,
        _,
    ) = engine.build_new_routes(
        frames_1h,
        common_1h,
        market_states,
        asset_shocks,
    )

    link_1h = frames_1h[
        SYMBOL
    ]

    link_15m = frames_15m[
        SYMBOL
    ]

    rows = []

    total_route_valid = 0
    total_quality_eligible = 0

    for counter, decision_time in enumerate(
        common_1h,
        start=1,
    ):

        signal = (
            route_signals[
                decision_time
            ][SYMBOL][ROUTE]
        )

        if not signal.valid:
            continue

        total_route_valid += 1

        strength = (
            engine.relative_strength_percentiles(
                frames_1h,
                decision_time,
            )
        )

        relative_strength = float(
            strength[SYMBOL]
        )

        market_state = (
            market_states[
                decision_time
            ]
        )

        # Reuse engine's EXACT route converter.
        # This preserves current entry, stop,
        # R:R and route-quality rules.
        candidate = (
            engine._convert_route_signal(
                symbol=SYMBOL,
                route=ROUTE,
                signal=signal,
                frame=link_1h,
                decision_time=(
                    decision_time
                ),
                market_state=(
                    market_state
                ),
                relative_strength=(
                    relative_strength
                ),
            )
        )

        if not candidate.valid:
            continue

        total_quality_eligible += 1

        (
            current_near_high,
            previous_near_high,
            fresh_zone,
            actual_cross,
        ) = fresh_zone_flags(
            link_1h,
            decision_time,
        )

        excursion = measure_excursion(
            frame_15m=link_15m,
            decision_time=(
                decision_time
            ),
            entry_price=float(
                candidate.entry_price
            ),
            stop_price=float(
                candidate.stop_price
            ),
        )

        rows.append(
            {
                "window": window_key,
                "decision_time": (
                    decision_time
                ),
                "signal_score": float(
                    signal.score
                ),
                "relative_strength": (
                    relative_strength
                ),
                "market_state": getattr(
                    market_state.state,
                    "value",
                    str(
                        market_state.state
                    ),
                ),
                "entry_price": float(
                    candidate.entry_price
                ),
                "stop_price": float(
                    candidate.stop_price
                ),
                "risk_per_unit": float(
                    candidate.entry_price
                    - candidate.stop_price
                ),
                "reward_risk": float(
                    candidate.reward_risk
                ),
                "current_near_high": (
                    current_near_high
                ),
                "previous_near_high": (
                    previous_near_high
                ),
                "fresh_zone": (
                    fresh_zone
                ),
                "actual_cross": (
                    actual_cross
                ),
                **excursion,
            }
        )

    detail = pd.DataFrame(
        rows
    )

    if detail.empty:
        raise RuntimeError(
            f"No eligible LINK IGNITION signals for {window}."
        )

    # CURRENT production-like route cooldown.
    detail[
        "current_policy"
    ] = True

    apply_cooldown(
        detail,
        source_column="current_policy",
        output_column=(
            "current_12h_selected"
        ),
    )

    # V3J-A freshness first, then same cooldown.
    apply_cooldown(
        detail,
        source_column="fresh_zone",
        output_column=(
            "fresh_12h_selected"
        ),
    )

    detail_path = (
        REPORT_DIR
        / (
            "v3j_fresh_ignition_"
            f"{window_key}_detail.csv"
        )
    )

    detail.to_csv(
        detail_path,
        index=False,
    )

    cohorts = {
        "CURRENT_RAW": detail,
        "FRESH_RAW": detail[
            detail["fresh_zone"]
        ],
        "CURRENT_12H": detail[
            detail[
                "current_12h_selected"
            ]
        ],
        "FRESH_12H": detail[
            detail[
                "fresh_12h_selected"
            ]
        ],
    }

    summary = pd.DataFrame(
        [
            summarize_cohort(
                frame,
                window_key,
                name,
            )
            for name, frame
            in cohorts.items()
        ]
    )

    summary_path = (
        REPORT_DIR
        / (
            "v3j_fresh_ignition_"
            f"{window_key}_summary.csv"
        )
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    print()
    print(
        f"Route-valid IGNITION signals : {total_route_valid}"
    )
    print(
        f"Quality-eligible LINK signals: {total_quality_eligible}"
    )
    print()

    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print(
        f"Detail  : {detail_path}"
    )
    print(
        f"Summary : {summary_path}"
    )


def summarize_all() -> None:

    paths = sorted(
        REPORT_DIR.glob(
            "v3j_fresh_ignition_*_detail.csv"
        )
    )

    if not paths:
        raise RuntimeError(
            "No V3J detail files found."
        )

    frames = [
        pd.read_csv(path)
        for path in paths
    ]

    detail = pd.concat(
        frames,
        ignore_index=True,
    )

    # CSV inference mixes numeric years with "2025h1".
    # Normalize only for reporting/sorting.
    detail["window"] = (
        detail["window"].astype(str)
    )

    bool_columns = [
        "fresh_zone",
        "actual_cross",
        "reached_0_5r",
        "reached_1r",
        "reached_2r",
        "stopped_before_horizon",
        "current_12h_selected",
        "fresh_12h_selected",
    ]

    for column in bool_columns:
        detail[column] = (
            detail[column]
            .astype(str)
            .str.lower()
            .eq("true")
        )

    rows = []

    windows = list(
        sorted(
            detail["window"].unique()
        )
    )

    for window in windows + ["ALL"]:

        if window == "ALL":
            frame = detail
        else:
            frame = detail[
                detail["window"]
                == window
            ]

        cohorts = {
            "CURRENT_RAW": frame,
            "FRESH_RAW": frame[
                frame["fresh_zone"]
            ],
            "CURRENT_12H": frame[
                frame[
                    "current_12h_selected"
                ]
            ],
            "FRESH_12H": frame[
                frame[
                    "fresh_12h_selected"
                ]
            ],
        }

        for name, cohort in cohorts.items():
            rows.append(
                summarize_cohort(
                    cohort,
                    window,
                    name,
                )
            )

    summary = pd.DataFrame(
        rows
    )

    output = (
        REPORT_DIR
        / "v3j_fresh_ignition_all_summary.csv"
    )

    summary.to_csv(
        output,
        index=False,
    )

    print()
    print("=" * 120)
    print(
        "V3J-A FRESH IGNITION - CROSS-WINDOW SUMMARY"
    )
    print("=" * 120)

    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print(
        f"Saved: {output}"
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
