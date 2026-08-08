from pathlib import Path
import numpy as np
import pandas as pd


REPORTS = Path("reports")

SOURCE = (
    REPORTS
    / "v3kx_early_path_outcome_join.csv"
)

EXPECTED_ROWS = 440

WINDOW_ORDER = [
    "2022",
    "2023",
    "2024",
    "2025H1",
]

CHECKPOINTS = [
    6,
    12,
    24,
    36,
]


if not SOURCE.exists():
    raise RuntimeError(
        f"SOURCE_MISSING: {SOURCE}"
    )


df = pd.read_csv(SOURCE)


if len(df) != EXPECTED_ROWS:
    raise RuntimeError(
        "ROW_COUNT_BAD: "
        f"expected={EXPECTED_ROWS} "
        f"actual={len(df)}"
    )


numeric_columns = [
    "checkpoint_hours",
    "exit_now_net_r",
    "mfe_gross_r",
    "mae_gross_r",
    "close_gross_r",
    "stop_progress_r",
    "stop_level_r",
    "distance_to_stop_r",
    "eventual_net_r",
]

for column in numeric_columns:

    df[column] = pd.to_numeric(
        df[column],
        errors="raise",
    )


df["checkpoint_hours"] = (
    df["checkpoint_hours"]
    .astype(int)
)


actual_checkpoints = sorted(
    df["checkpoint_hours"]
    .unique()
    .tolist()
)

if actual_checkpoints != CHECKPOINTS:
    raise RuntimeError(
        "CHECKPOINTS_BAD: "
        f"{actual_checkpoints}"
    )


df["eventual_win"] = (
    df["eventual_net_r"] > 0
)

df["eventual_loss"] = (
    df["eventual_net_r"] < 0
)

df["eventual_ge_2r"] = (
    df["eventual_net_r"] >= 2.0
)

df["eventual_ge_3r"] = (
    df["eventual_net_r"] >= 3.0
)

df["exit_benefit_r"] = (
    df["exit_now_net_r"]
    - df["eventual_net_r"]
)


def gate_mask(
    frame: pd.DataFrame,
    gate_name: str,
) -> pd.Series:

    current_nonpositive = (
        frame["exit_now_net_r"]
        <= 0.0
    )

    no_meaningful_mfe = (
        frame["mfe_gross_r"]
        < 0.5
    )

    no_stop_progress = (
        frame["stop_progress_r"]
        <= 0.0
    )

    if gate_name == "CURRENT_NONPOSITIVE":
        return current_nonpositive

    if gate_name == "NO_MEANINGFUL_MFE":
        return no_meaningful_mfe

    if gate_name == "NO_STOP_PROGRESS":
        return no_stop_progress

    if gate_name == "ALL_THREE":
        return (
            current_nonpositive
            & no_meaningful_mfe
            & no_stop_progress
        )

    raise RuntimeError(
        f"UNKNOWN_GATE: {gate_name}"
    )


GATES = [
    "CURRENT_NONPOSITIVE",
    "NO_MEANINGFUL_MFE",
    "NO_STOP_PROGRESS",
    "ALL_THREE",
]


def summarize(
    frame: pd.DataFrame,
    mask: pd.Series,
) -> dict:

    flagged = frame.loc[
        mask
    ].copy()

    kept = frame.loc[
        ~mask
    ].copy()

    checkpoint_winners = int(
        frame["eventual_win"].sum()
    )

    checkpoint_ge2 = int(
        frame["eventual_ge_2r"].sum()
    )

    checkpoint_ge3 = int(
        frame["eventual_ge_3r"].sum()
    )

    flagged_winners = int(
        flagged["eventual_win"].sum()
    )

    flagged_losses = int(
        flagged["eventual_loss"].sum()
    )

    flagged_ge2 = int(
        flagged["eventual_ge_2r"].sum()
    )

    flagged_ge3 = int(
        flagged["eventual_ge_3r"].sum()
    )

    n = len(flagged)

    return {
        "checkpoint_rows":
            int(len(frame)),

        "flagged_rows":
            int(n),

        "flag_rate_pct":
            (
                n
                / len(frame)
                * 100.0
                if len(frame)
                else 0.0
            ),

        "flagged_losses":
            flagged_losses,

        "flagged_winners":
            flagged_winners,

        "loss_precision_pct":
            (
                flagged_losses
                / n
                * 100.0
                if n
                else np.nan
            ),

        "winner_capture_pct":
            (
                flagged_winners
                / checkpoint_winners
                * 100.0
                if checkpoint_winners
                else 0.0
            ),

        "flagged_ge_2r":
            flagged_ge2,

        "ge_2r_capture_pct":
            (
                flagged_ge2
                / checkpoint_ge2
                * 100.0
                if checkpoint_ge2
                else 0.0
            ),

        "flagged_ge_3r":
            flagged_ge3,

        "ge_3r_capture_pct":
            (
                flagged_ge3
                / checkpoint_ge3
                * 100.0
                if checkpoint_ge3
                else 0.0
            ),

        "flagged_eventual_sum_r":
            float(
                flagged[
                    "eventual_net_r"
                ].sum()
            ),

        "flagged_eventual_mean_r":
            float(
                flagged[
                    "eventual_net_r"
                ].mean()
            )
            if n
            else np.nan,

        "flagged_eventual_median_r":
            float(
                flagged[
                    "eventual_net_r"
                ].median()
            )
            if n
            else np.nan,

        "flagged_exit_now_mean_r":
            float(
                flagged[
                    "exit_now_net_r"
                ].mean()
            )
            if n
            else np.nan,

        "mean_exit_benefit_r":
            float(
                flagged[
                    "exit_benefit_r"
                ].mean()
            )
            if n
            else np.nan,

        "median_exit_benefit_r":
            float(
                flagged[
                    "exit_benefit_r"
                ].median()
            )
            if n
            else np.nan,

        "exit_now_better_pct":
            float(
                (
                    flagged[
                        "exit_benefit_r"
                    ] > 0
                ).mean()
                * 100.0
            )
            if n
            else np.nan,

        "kept_eventual_sum_r":
            float(
                kept[
                    "eventual_net_r"
                ].sum()
            ),
    }


overall_rows = []
window_rows = []


for checkpoint in CHECKPOINTS:

    checkpoint_frame = df[
        df["checkpoint_hours"]
        == checkpoint
    ].copy()

    for gate_name in GATES:

        mask = gate_mask(
            checkpoint_frame,
            gate_name,
        )

        row = {
            "checkpoint_hours":
                checkpoint,

            "gate":
                gate_name,
        }

        row.update(
            summarize(
                checkpoint_frame,
                mask,
            )
        )

        overall_rows.append(
            row
        )


        for window in WINDOW_ORDER:

            window_frame = (
                checkpoint_frame[
                    checkpoint_frame[
                        "audit_window"
                    ] == window
                ]
                .copy()
            )

            if window_frame.empty:
                continue

            window_mask = gate_mask(
                window_frame,
                gate_name,
            )

            window_row = {
                "checkpoint_hours":
                    checkpoint,

                "gate":
                    gate_name,

                "audit_window":
                    window,
            }

            window_row.update(
                summarize(
                    window_frame,
                    window_mask,
                )
            )

            window_rows.append(
                window_row
            )


overall = pd.DataFrame(
    overall_rows
)

by_window = pd.DataFrame(
    window_rows
)


stability_rows = []


for (
    checkpoint,
    gate_name
), group in by_window.groupby(
    [
        "checkpoint_hours",
        "gate",
    ],
    sort=True,
):

    supported = group[
        group["flagged_rows"] > 0
    ].copy()

    stability_rows.append(
        {
            "checkpoint_hours":
                int(checkpoint),

            "gate":
                gate_name,

            "windows_with_flags":
                int(len(supported)),

            "minimum_window_flags":
                int(
                    supported[
                        "flagged_rows"
                    ].min()
                )
                if not supported.empty
                else 0,

            "negative_eventual_mean_windows":
                int(
                    (
                        supported[
                            "flagged_eventual_mean_r"
                        ] < 0
                    ).sum()
                ),

            "positive_mean_exit_benefit_windows":
                int(
                    (
                        supported[
                            "mean_exit_benefit_r"
                        ] > 0
                    ).sum()
                ),

            "positive_median_exit_benefit_windows":
                int(
                    (
                        supported[
                            "median_exit_benefit_r"
                        ] > 0
                    ).sum()
                ),

            "worst_window_mean_exit_benefit_r":
                float(
                    supported[
                        "mean_exit_benefit_r"
                    ].min()
                )
                if not supported.empty
                else np.nan,

            "worst_window_median_exit_benefit_r":
                float(
                    supported[
                        "median_exit_benefit_r"
                    ].min()
                )
                if not supported.empty
                else np.nan,

            "total_flagged_ge_2r":
                int(
                    supported[
                        "flagged_ge_2r"
                    ].sum()
                ),

            "total_flagged_ge_3r":
                int(
                    supported[
                        "flagged_ge_3r"
                    ].sum()
                ),
        }
    )


stability = pd.DataFrame(
    stability_rows
)


overall_path = (
    REPORTS
    / "v3kx_early_path_loss_separation_overall.csv"
)

window_path = (
    REPORTS
    / "v3kx_early_path_loss_separation_by_window.csv"
)

stability_path = (
    REPORTS
    / "v3kx_early_path_loss_separation_stability.csv"
)

summary_path = (
    REPORTS
    / "v3kx_early_path_loss_separation_summary.txt"
)


overall.to_csv(
    overall_path,
    index=False,
)

by_window.to_csv(
    window_path,
    index=False,
)

stability.to_csv(
    stability_path,
    index=False,
)


lines = []

lines.append(
    "=" * 150
)

lines.append(
    "V3KX CAUSAL EARLY-PATH LOSS-SEPARATION AUDIT"
)

lines.append(
    "=" * 150
)

lines.append(
    "Historical inspected windows only; not untouched out-of-sample."
)

lines.append(
    "No threshold search. No portfolio replay. No early exit executed."
)

lines.append(
    "Fixed gates: current net R <= 0; MFE < 0.5R; stop progress <= 0; conjunction of all three."
)

lines.append("")


for checkpoint in CHECKPOINTS:

    lines.append(
        "-" * 150
    )

    lines.append(
        f"CHECKPOINT {checkpoint}H"
    )

    lines.append(
        "-" * 150
    )

    view = overall[
        overall[
            "checkpoint_hours"
        ] == checkpoint
    ]

    columns = [
        "gate",
        "flagged_rows",
        "loss_precision_pct",
        "winner_capture_pct",
        "flagged_ge_2r",
        "flagged_ge_3r",
        "flagged_eventual_mean_r",
        "flagged_eventual_median_r",
        "mean_exit_benefit_r",
        "median_exit_benefit_r",
        "exit_now_better_pct",
    ]

    lines.append(
        view[
            columns
        ].to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.4f}"
            ),
        )
    )

    lines.append("")


lines.append(
    "-" * 150
)

lines.append(
    "CROSS-WINDOW STABILITY"
)

lines.append(
    "-" * 150
)

lines.append(
    stability.to_string(
        index=False,
        float_format=lambda x: (
            f"{x:.4f}"
        ),
    )
)

lines.append("")

lines.append(
    "MASTER OBJECTIVE CHECK"
)

lines.append(
    "1. Loss/drawdown improvement: NOT YET MEASURED; no portfolio replay."
)

lines.append(
    "2. Upside preservation: every gate reports captured eventual >=2R and >=3R winners."
)

lines.append(
    "3. Stability: each fixed gate is tested separately across 2022, 2023, 2024 and 2025H1."
)

lines.append(
    "4. EUR 200 weekly-average target: NOT YET MEASURED."
)

lines.append("")

lines.append(
    "No gate is approved by this script."
)

lines.append(
    "Next decision gate: nominate at most one candidate only if cross-window loss separation survives and large-winner damage is acceptable."
)

lines.append("")

lines.append(
    "V3KX causal loss-separation audit: PASS"
)


summary_path.write_text(
    "\n".join(lines) + "\n",
    encoding="utf-8",
)

print(
    "\n".join(lines)
)
