from __future__ import annotations

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


def normalize_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series

    return (
        series.astype(str)
        .str.lower()
        .eq("true")
    )


def metrics(frame: pd.DataFrame) -> dict[str, float | int]:
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


def apply_cooldown(
    frame: pd.DataFrame,
    source_column: str,
    output_column: str,
) -> pd.DataFrame:

    frame = frame.sort_values(
        "decision_time"
    ).copy()

    frame[output_column] = False

    last_selected = None

    for index, row in frame.iterrows():

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

    return frame


def load_detail() -> pd.DataFrame:
    path = (
        REPORT_DIR
        / "v3je_market_volume_interaction_detail.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            path
        )

    detail = pd.read_csv(
        path
    )

    detail["window"] = (
        detail["window"]
        .astype(str)
        .str.lower()
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

    return detail


def build_v3jf_gate(
    detail: pd.DataFrame,
) -> pd.DataFrame:

    detail = detail.copy()

    # Pre-locked V3J-F rule:
    #
    # 1. Non stale-hot signals remain untouched.
    #
    # 2. Stale + causal HIGH-hotness signals survive
    #    ONLY when:
    #       market_state == STRONG_BULL
    #       AND causal volume regime is MID or HIGH.
    #
    # No new fitted thresholds are introduced.

    stale_hot = detail[
        "v3jc_stale_hot_reject"
    ]

    healthy_stale_hot = (
        stale_hot
        & detail[
            "market_state"
        ].eq("STRONG_BULL")
        & detail[
            "causal_volume_regime"
        ].isin(
            [
                "MID",
                "HIGH",
            ]
        )
    )

    detail[
        "v3jf_healthy_stale_hot"
    ] = healthy_stale_hot

    detail[
        "v3jf_gate_reject"
    ] = (
        stale_hot
        & ~healthy_stale_hot
    )

    detail[
        "v3jf_gate_keep"
    ] = ~detail[
        "v3jf_gate_reject"
    ]

    detail[
        "v3jf_12h_selected"
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

        window_frame = apply_cooldown(
            window_frame,
            source_column="v3jf_gate_keep",
            output_column="v3jf_12h_selected",
        )

        detail.loc[
            window_frame.index,
            "v3jf_12h_selected",
        ] = window_frame[
            "v3jf_12h_selected"
        ]

    # Signals that baseline CURRENT_12H actually used
    # but V3J-F's gate rejects.
    detail[
        "baseline_selected_rejected"
    ] = (
        detail["current_12h_selected"]
        & detail["v3jf_gate_reject"]
    )

    return detail


def build_summary(
    detail: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

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

            "V3JF_12H": frame[
                frame[
                    "v3jf_12h_selected"
                ]
            ],

            "BASELINE_SELECTED_REJECTED": frame[
                frame[
                    "baseline_selected_rejected"
                ]
            ],

            "HEALTHY_STALE_HOT": frame[
                frame[
                    "v3jf_healthy_stale_hot"
                ]
            ],
        }

        for cohort_name, cohort in cohorts.items():
            rows.append(
                {
                    "window": window,
                    "cohort": cohort_name,
                    **metrics(cohort),
                }
            )

    return pd.DataFrame(
        rows
    )


def main() -> None:

    print()
    print("=" * 124)
    print(
        "V3J-F CONTROLLED CAUSAL STALE-HOT CONTEXT ABLATION"
    )
    print("=" * 124)

    print(
        "Pre-locked rule:"
    )
    print(
        "Non stale-hot = KEEP"
    )
    print(
        "Stale-hot = KEEP only if "
        "STRONG_BULL + causal volume MID/HIGH"
    )
    print(
        "No threshold search. "
        "Risk, stops, exits, ranking and SOL untouched."
    )

    detail = load_detail()

    detail = build_v3jf_gate(
        detail
    )

    detail_path = (
        REPORT_DIR
        / "v3jf_controlled_ablation_detail.csv"
    )

    detail.to_csv(
        detail_path,
        index=False,
    )

    summary = build_summary(
        detail
    )

    summary_path = (
        REPORT_DIR
        / "v3jf_controlled_ablation_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    print()
    print(
        "V3J-F GATE COUNTS"
    )
    print("-" * 124)

    for window in WINDOW_ORDER:

        frame = detail[
            detail["window"]
            == window
        ]

        print(
            f"{window:<7} "
            f"eligible={len(frame):>3} | "
            f"stale_hot="
            f"{int(frame['v3jc_stale_hot_reject'].sum()):>3} | "
            f"healthy_stale_hot="
            f"{int(frame['v3jf_healthy_stale_hot'].sum()):>3} | "
            f"gate_reject="
            f"{int(frame['v3jf_gate_reject'].sum()):>3} | "
            f"current12h="
            f"{int(frame['current_12h_selected'].sum()):>3} | "
            f"v3jf12h="
            f"{int(frame['v3jf_12h_selected'].sum()):>3}"
        )

    print()
    print("=" * 124)
    print(
        "V3J-F - CURRENT vs CONTROLLED ABLATION"
    )
    print("=" * 124)

    comparison = summary[
        summary[
            "cohort"
        ].isin(
            [
                "CURRENT_12H",
                "V3JF_12H",
            ]
        )
    ]

    print(
        comparison.to_string(
            index=False
        )
    )

    print()
    print("=" * 124)
    print(
        "V3J-F - BASELINE SIGNALS THE GATE REMOVED"
    )
    print("=" * 124)

    rejected = summary[
        summary["cohort"]
        == "BASELINE_SELECTED_REJECTED"
    ]

    print(
        rejected.to_string(
            index=False
        )
    )

    print()
    print("=" * 124)
    print(
        "V3J-F - STALE-HOT SIGNALS THE RULE RESCUED"
    )
    print("=" * 124)

    rescued = summary[
        summary["cohort"]
        == "HEALTHY_STALE_HOT"
    ]

    print(
        rescued.to_string(
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
