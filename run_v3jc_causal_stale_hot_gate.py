from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REPORT_DIR = Path("reports")

WINDOW_ORDER = [
    "2021",
    "2022",
    "2023",
    "2024",
    "2025h1",
]

COOLDOWN_HOURS = 12
HIGH_HOTNESS_CUTOFF = 2.0 / 3.0


def normalize_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series

    return (
        series.astype(str)
        .str.lower()
        .eq("true")
    )


def causal_percentile(
    history: list[float],
    current: float,
) -> float:
    """
    Empirical percentile using ONLY observations
    available up to and including current signal.

    No future-window information is used.
    """

    if not np.isfinite(current):
        return np.nan

    if not history:
        return 1.0

    prior = np.asarray(
        history,
        dtype=float,
    )

    less_or_equal = int(
        np.sum(prior <= current)
    )

    return float(
        (less_or_equal + 1)
        / (len(prior) + 1)
    )


def apply_cooldown(
    frame: pd.DataFrame,
    source_column: str,
    output_column: str,
) -> None:

    frame[output_column] = False

    last_selected = None

    for index, row in frame.sort_values(
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
            frame.at[
                index,
                output_column,
            ] = True

            last_selected = decision_time


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
            "reached_1r_pct": np.nan,
            "reached_2r_pct": np.nan,
            "stopped_pct": np.nan,
        }

    return {
        "signals": len(frame),
        "false_pct": float(
            (frame["mfe_r"] < 0.5).mean()
            * 100.0
        ),
        "runner_pct": float(
            (frame["mfe_r"] >= 2.0).mean()
            * 100.0
        ),
        "median_mfe_r": float(
            frame["mfe_r"].median()
        ),
        "median_mae_r": float(
            frame["mae_r"].median()
        ),
        "reached_1r_pct": float(
            frame["reached_1r"].mean()
            * 100.0
        ),
        "reached_2r_pct": float(
            frame["reached_2r"].mean()
            * 100.0
        ),
        "stopped_pct": float(
            frame[
                "stopped_before_horizon"
            ].mean()
            * 100.0
        ),
    }


def load_detail() -> pd.DataFrame:
    frames = []

    for window in WINDOW_ORDER:
        path = (
            REPORT_DIR
            / (
                "v3jb_overextension_"
                f"{window}_detail.csv"
            )
        )

        if not path.exists():
            raise FileNotFoundError(
                path
            )

        frame = pd.read_csv(
            path
        )

        frame["window"] = window

        frame["decision_time"] = (
            pd.to_datetime(
                frame["decision_time"]
            )
        )

        for column in [
            "fresh_zone",
            "current_12h_selected",
            "fresh_12h_selected",
            "reached_1r",
            "reached_2r",
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

    detail = detail.sort_values(
        "decision_time"
    ).reset_index(
        drop=True
    )

    return detail


def build_causal_gate(
    detail: pd.DataFrame,
) -> pd.DataFrame:

    adx_history: list[float] = []
    slope_history: list[float] = []

    adx_ranks = []
    slope_ranks = []
    hotness_ranks = []
    history_counts = []
    stale_hot_flags = []
    keep_flags = []

    for _, row in detail.iterrows():

        adx = float(
            row["adx14"]
        )

        slope = float(
            row["ema50_slope_12h"]
        )

        history_counts.append(
            len(adx_history)
        )

        adx_rank = causal_percentile(
            adx_history,
            adx,
        )

        slope_rank = causal_percentile(
            slope_history,
            slope,
        )

        hotness_rank = (
            adx_rank + slope_rank
        ) / 2.0

        fresh = bool(
            row["fresh_zone"]
        )

        stale_hot = (
            (not fresh)
            and hotness_rank
            > HIGH_HOTNESS_CUTOFF
        )

        keep = not stale_hot

        adx_ranks.append(
            adx_rank
        )

        slope_ranks.append(
            slope_rank
        )

        hotness_ranks.append(
            hotness_rank
        )

        stale_hot_flags.append(
            stale_hot
        )

        keep_flags.append(
            keep
        )

        # Append only AFTER current ranks are calculated.
        # Therefore future observations never affect current rank.
        adx_history.append(
            adx
        )

        slope_history.append(
            slope
        )

    detail[
        "causal_history_count"
    ] = history_counts

    detail[
        "causal_adx_rank"
    ] = adx_ranks

    detail[
        "causal_slope_rank"
    ] = slope_ranks

    detail[
        "causal_hotness_rank"
    ] = hotness_ranks

    detail[
        "v3jc_stale_hot_reject"
    ] = stale_hot_flags

    detail[
        "v3jc_gate_keep"
    ] = keep_flags

    # Cooldown resets inside each historical robustness window,
    # matching the existing window-level diagnostic structure.
    detail[
        "v3jc_12h_selected"
    ] = False

    for window in WINDOW_ORDER:
        mask = (
            detail["window"]
            == window
        )

        window_frame = (
            detail.loc[mask]
            .copy()
        )

        apply_cooldown(
            window_frame,
            source_column="v3jc_gate_keep",
            output_column="v3jc_12h_selected",
        )

        detail.loc[
            window_frame.index,
            "v3jc_12h_selected",
        ] = window_frame[
            "v3jc_12h_selected"
        ]

    return detail


def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--summary-only",
        action="store_true",
    )

    args = parser.parse_args()

    print()
    print("=" * 120)
    print(
        "V3J-C CAUSAL STALE-HOT GATE DIAGNOSTIC"
    )
    print("=" * 120)
    print(
        "Fresh = keep | "
        "Stale + causal hotness > 2/3 = reject | "
        "12h cooldown unchanged"
    )
    print(
        "Causal ranks use only current + prior "
        "eligible LINK signals."
    )
    print(
        "Risk, stops, exits, SOL, ranking and "
        "surveillance remain untouched."
    )

    detail = load_detail()

    detail = build_causal_gate(
        detail
    )

    detail_path = (
        REPORT_DIR
        / "v3jc_causal_stale_hot_detail.csv"
    )

    detail.to_csv(
        detail_path,
        index=False,
    )

    summary_rows = []

    for window in (
        WINDOW_ORDER
        + ["ALL"]
    ):

        if window == "ALL":
            frame = detail
        else:
            frame = detail[
                detail["window"]
                == window
            ]

        cohorts = {
            "CURRENT_12H": frame[
                frame[
                    "current_12h_selected"
                ]
            ],
            "V3JC_12H": frame[
                frame[
                    "v3jc_12h_selected"
                ]
            ],
            "REJECTED_STALE_HOT": frame[
                frame[
                    "v3jc_stale_hot_reject"
                ]
            ],
        }

        for cohort_name, cohort in (
            cohorts.items()
        ):
            summary_rows.append(
                {
                    "window": window,
                    "cohort": cohort_name,
                    **metrics(cohort),
                }
            )

    summary = pd.DataFrame(
        summary_rows
    )

    summary_path = (
        REPORT_DIR
        / "v3jc_causal_stale_hot_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    if not args.summary_only:
        print()
        print(
            "CAUSAL GATE COUNTS"
        )
        print("-" * 120)

        for window in WINDOW_ORDER:
            frame = detail[
                detail["window"]
                == window
            ]

            print(
                f"{window:<7} "
                f"eligible={len(frame):>3} | "
                f"stale-hot rejected="
                f"{int(frame['v3jc_stale_hot_reject'].sum()):>3} | "
                f"current12h="
                f"{int(frame['current_12h_selected'].sum()):>3} | "
                f"v3jc12h="
                f"{int(frame['v3jc_12h_selected'].sum()):>3}"
            )

    print()
    print("=" * 120)
    print(
        "CURRENT vs V3J-C SIGNAL QUALITY"
    )
    print("=" * 120)

    focus = summary[
        summary["cohort"].isin(
            [
                "CURRENT_12H",
                "V3JC_12H",
            ]
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
        "WHAT THE GATE REJECTED"
    )
    print("=" * 120)

    rejected = summary[
        summary["cohort"]
        == "REJECTED_STALE_HOT"
    ]

    print(
        rejected.to_string(
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


if __name__ == "__main__":
    main()
