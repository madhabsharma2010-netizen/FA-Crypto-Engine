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

LOW_CUTOFF = 1.0 / 3.0
HIGH_CUTOFF = 2.0 / 3.0


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

    if not np.isfinite(current):
        return np.nan

    if not history:
        return 1.0

    prior = np.asarray(
        history,
        dtype=float,
    )

    return float(
        (
            int(np.sum(prior <= current))
            + 1
        )
        / (
            len(prior)
            + 1
        )
    )


def volume_regime(rank: float) -> str:
    if not np.isfinite(rank):
        return "UNKNOWN"

    if rank <= LOW_CUTOFF:
        return "LOW"

    if rank <= HIGH_CUTOFF:
        return "MID"

    return "HIGH"


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
        "stopped_pct": float(
            frame[
                "stopped_before_horizon"
            ].mean()
            * 100.0
        ),
    }


def load_all() -> pd.DataFrame:
    frames = []

    for window in WINDOW_ORDER:
        path = (
            REPORT_DIR
            / (
                "v3jd_stale_hot_context_"
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

        frame["decision_time"] = pd.to_datetime(
            frame["decision_time"]
        )

        for column in [
            "current_12h_selected",
            "v3jc_stale_hot_reject",
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

    detail[
        "baseline_selected_stale_hot"
    ] = (
        detail["current_12h_selected"]
        & detail["v3jc_stale_hot_reject"]
    )

    return detail


def build_causal_volume(
    detail: pd.DataFrame,
) -> pd.DataFrame:

    history: list[float] = []

    ranks = []
    counts = []

    for _, row in detail.iterrows():
        value = float(
            row["volume_ratio"]
        )

        counts.append(
            len(history)
        )

        rank = causal_percentile(
            history,
            value,
        )

        ranks.append(
            rank
        )

        # Current signal enters history only
        # after its causal rank is calculated.
        history.append(
            value
        )

    detail[
        "causal_volume_history_count"
    ] = counts

    detail[
        "causal_volume_rank"
    ] = ranks

    detail[
        "causal_volume_regime"
    ] = detail[
        "causal_volume_rank"
    ].map(
        volume_regime
    )

    return detail


def summarize(
    detail: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    focus = detail[
        detail[
            "baseline_selected_stale_hot"
        ]
    ].copy()

    rows = []

    for window in (
        WINDOW_ORDER
        + ["ALL"]
    ):
        if window == "ALL":
            frame = focus
        else:
            frame = focus[
                focus["window"]
                == window
            ]

        for market_state in sorted(
            frame[
                "market_state"
            ].dropna().unique()
        ):
            state_frame = frame[
                frame["market_state"]
                == market_state
            ]

            for regime in [
                "LOW",
                "MID",
                "HIGH",
            ]:
                cell = state_frame[
                    state_frame[
                        "causal_volume_regime"
                    ]
                    == regime
                ]

                rows.append(
                    {
                        "window": window,
                        "market_state": (
                            market_state
                        ),
                        "volume_regime": regime,
                        **metrics(cell),
                    }
                )

    summary = pd.DataFrame(
        rows
    )

    corr_rows = []

    for window in (
        WINDOW_ORDER
        + ["ALL"]
    ):
        if window == "ALL":
            frame = focus
        else:
            frame = focus[
                focus["window"]
                == window
            ]

        for market_state in sorted(
            frame[
                "market_state"
            ].dropna().unique()
        ):
            state_frame = frame[
                frame["market_state"]
                == market_state
            ]

            if len(state_frame) >= 3:
                x_rank = (
                    state_frame[
                        "causal_volume_rank"
                    ].rank()
                )

                y_rank = (
                    state_frame[
                        "mfe_r"
                    ].rank()
                )

                correlation = (
                    x_rank.corr(
                        y_rank
                    )
                )
            else:
                correlation = np.nan

            corr_rows.append(
                {
                    "window": window,
                    "market_state": (
                        market_state
                    ),
                    "signals": len(
                        state_frame
                    ),
                    "spearman_volume_rank_mfe": (
                        correlation
                    ),
                }
            )

    correlations = pd.DataFrame(
        corr_rows
    )

    return (
        summary,
        correlations,
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--summary-only",
        action="store_true",
    )

    args = parser.parse_args()

    print()
    print("=" * 122)
    print(
        "V3J-E MARKET STATE x CAUSAL VOLUME PARTICIPATION"
    )
    print("=" * 122)
    print(
        "Diagnostic only: CURRENT_12H baseline-selected "
        "stale-hot LINK signals."
    )
    print(
        "Volume rank uses only prior eligible LINK signals."
    )
    print(
        "LOW/MID/HIGH boundaries remain fixed at historical "
        "tertiles; no threshold search."
    )
    print(
        "Risk, stops, exits, ranking and SOL remain untouched."
    )

    detail = load_all()

    detail = build_causal_volume(
        detail
    )

    detail_path = (
        REPORT_DIR
        / "v3je_market_volume_interaction_detail.csv"
    )

    detail.to_csv(
        detail_path,
        index=False,
    )

    (
        summary,
        correlations,
    ) = summarize(
        detail
    )

    summary_path = (
        REPORT_DIR
        / "v3je_market_volume_interaction_summary.csv"
    )

    correlations_path = (
        REPORT_DIR
        / "v3je_market_volume_correlations.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    correlations.to_csv(
        correlations_path,
        index=False,
    )

    focus = detail[
        detail[
            "baseline_selected_stale_hot"
        ]
    ]

    if not args.summary_only:
        print()
        print(
            "STALE-HOT SAMPLE COUNTS"
        )
        print("-" * 122)

        for window in WINDOW_ORDER:
            frame = focus[
                focus["window"]
                == window
            ]

            print(
                f"{window:<7} "
                f"signals={len(frame):>3} | "
                f"BULL="
                f"{int((frame['market_state'] == 'BULL').sum()):>3} | "
                f"STRONG_BULL="
                f"{int((frame['market_state'] == 'STRONG_BULL').sum()):>3}"
            )

    print()
    print("=" * 122)
    print(
        "V3J-E - MARKET STATE x CAUSAL VOLUME REGIME"
    )
    print("=" * 122)

    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print("=" * 122)
    print(
        "V3J-E - VOLUME RANK vs MFE CORRELATIONS"
    )
    print("=" * 122)

    print(
        correlations.to_string(
            index=False
        )
    )

    print()
    print(
        f"Detail       : {detail_path}"
    )
    print(
        f"Interaction  : {summary_path}"
    )
    print(
        f"Correlations : {correlations_path}"
    )


if __name__ == "__main__":
    main()
