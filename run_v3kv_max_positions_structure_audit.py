from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


WINDOWS = [
    ("2022", "2022"),
    ("2023", "2023"),
    ("2024", "2024"),
    ("2025H1", "2025h1"),
]

REQUIRED_COLUMNS = {
    "disposition",
    "symbol",
    "candidate_id",
    "candidate_rank",
    "due_candidate_count",
    "entries_this_hour",
    "entry_time",
    "net_r",
}

REPORT_DIR = Path("reports")
EPSILON = 1e-12


def metrics(
    frame: pd.DataFrame,
) -> dict[str, object]:

    values = pd.to_numeric(
        frame["net_r"],
        errors="raise",
    )

    positive = (
        values[
            values > EPSILON
        ]
        .sort_values(
            ascending=False
        )
    )

    gross_positive = float(
        positive.sum()
    )

    top_five_share = (
        float(
            positive.head(5).sum()
        )
        / gross_positive
        * 100.0
        if gross_positive > 0
        else 0.0
    )

    return {
        "candidates":
            int(len(frame)),

        "wins":
            int(
                (
                    values > EPSILON
                ).sum()
            ),

        "losses":
            int(
                (
                    values < -EPSILON
                ).sum()
            ),

        "win_rate_pct":
            float(
                (
                    values > EPSILON
                ).mean()
                * 100.0
            ),

        "mean_r":
            float(
                values.mean()
            ),

        "median_r":
            float(
                values.median()
            ),

        "capped_3r_mean":
            float(
                values.clip(
                    lower=-3.0,
                    upper=3.0,
                ).mean()
            ),

        "ge_1r":
            int(
                (
                    values >= 1.0
                ).sum()
            ),

        "ge_2r":
            int(
                (
                    values >= 2.0
                ).sum()
            ),

        "le_minus_1r":
            int(
                (
                    values <= -1.0
                ).sum()
            ),

        "minimum_r":
            float(
                values.min()
            ),

        "maximum_r":
            float(
                values.max()
            ),

        "top5_positive_share_pct":
            top_five_share,
    }


frames: list[pd.DataFrame] = []

for audit_window, tag in WINDOWS:

    path = (
        REPORT_DIR
        / f"v3kv_shadow_outcomes_{tag}.csv"
    )

    if not path.exists():
        raise RuntimeError(
            f"MISSING_REPORT: {path}"
        )

    frame = pd.read_csv(path)

    missing = sorted(
        REQUIRED_COLUMNS
        - set(frame.columns)
    )

    if missing:
        raise RuntimeError(
            "MISSING_COLUMNS: "
            f"window={audit_window}, "
            f"columns={missing}"
        )

    frame.insert(
        0,
        "audit_window",
        audit_window,
    )

    frames.append(frame)


combined = pd.concat(
    frames,
    ignore_index=True,
    sort=False,
)

max_positions = combined[
    combined["disposition"]
    == "MAX_POSITIONS"
].copy()

if len(max_positions) != 226:
    raise RuntimeError(
        "MAX_POSITIONS_COUNT_MISMATCH: "
        f"expected=226, "
        f"actual={len(max_positions)}"
    )

if max_positions.duplicated(
    subset=[
        "audit_window",
        "candidate_id",
    ]
).any():
    raise RuntimeError(
        "DUPLICATE_MAX_POSITION_CANDIDATES"
    )


for column in (
    "candidate_rank",
    "due_candidate_count",
    "entries_this_hour",
    "net_r",
):

    max_positions[column] = pd.to_numeric(
        max_positions[column],
        errors="raise",
    )


policies = {
    "ALL_MAX_POSITIONS":
        pd.Series(
            True,
            index=max_positions.index,
        ),

    "RANK_1":
        (
            max_positions[
                "candidate_rank"
            ] == 1
        ),

    "RANK_GT_1_CONTROL":
        (
            max_positions[
                "candidate_rank"
            ] > 1
        ),

    "RANK_1_SINGLE_DUE":
        (
            (
                max_positions[
                    "candidate_rank"
                ] == 1
            )
            & (
                max_positions[
                    "due_candidate_count"
                ] == 1
            )
        ),

    "RANK_1_MULTI_DUE":
        (
            (
                max_positions[
                    "candidate_rank"
                ] == 1
            )
            & (
                max_positions[
                    "due_candidate_count"
                ] >= 2
            )
        ),

    "RANK_1_NO_ENTRY_THIS_HOUR":
        (
            (
                max_positions[
                    "candidate_rank"
                ] == 1
            )
            & (
                max_positions[
                    "entries_this_hour"
                ] == 0
            )
        ),

    "RANK_1_AFTER_ENTRY_THIS_HOUR":
        (
            (
                max_positions[
                    "candidate_rank"
                ] == 1
            )
            & (
                max_positions[
                    "entries_this_hour"
                ] >= 1
            )
        ),
}


policy_rows: list[
    dict[str, object]
] = []

window_rows: list[
    dict[str, object]
] = []


for policy_name, mask in policies.items():

    subset = max_positions[
        mask
    ].copy()

    if subset.empty:
        continue

    combined_metrics = metrics(
        subset
    )

    per_window: list[
        dict[str, object]
    ] = []

    for audit_window, _ in WINDOWS:

        window_subset = subset[
            subset["audit_window"]
            == audit_window
        ]

        if window_subset.empty:
            continue

        row = {
            "policy":
                policy_name,

            "audit_window":
                audit_window,

            **metrics(
                window_subset
            ),
        }

        per_window.append(
            row
        )

        window_rows.append(
            row
        )

    per_window_frame = pd.DataFrame(
        per_window
    )

    policy_rows.append(
        {
            "policy":
                policy_name,

            **combined_metrics,

            "windows_present":
                int(
                    len(
                        per_window_frame
                    )
                ),

            "positive_mean_windows":
                int(
                    (
                        per_window_frame[
                            "mean_r"
                        ] > EPSILON
                    ).sum()
                ),

            "positive_capped_windows":
                int(
                    (
                        per_window_frame[
                            "capped_3r_mean"
                        ] > EPSILON
                    ).sum()
                ),

            "minimum_window_mean_r":
                float(
                    per_window_frame[
                        "mean_r"
                    ].min()
                ),

            "minimum_window_capped_mean":
                float(
                    per_window_frame[
                        "capped_3r_mean"
                    ].min()
                ),

            "maximum_window_mean_r":
                float(
                    per_window_frame[
                        "mean_r"
                    ].max()
                ),
        }
    )


policy_summary = pd.DataFrame(
    policy_rows
)

window_summary = pd.DataFrame(
    window_rows
)


def grouped_table(
    columns: list[str],
) -> pd.DataFrame:

    output: list[
        dict[str, object]
    ] = []

    for keys, group in max_positions.groupby(
        columns,
        dropna=False,
        sort=True,
    ):

        if not isinstance(
            keys,
            tuple,
        ):
            keys = (
                keys,
            )

        row = {
            column: value
            for column, value in zip(
                columns,
                keys,
            )
        }

        row.update(
            metrics(group)
        )

        output.append(
            row
        )

    return pd.DataFrame(
        output
    )


rank_summary = grouped_table(
    [
        "candidate_rank",
    ]
)

window_rank_summary = grouped_table(
    [
        "audit_window",
        "candidate_rank",
    ]
)

symbol_summary = grouped_table(
    [
        "symbol",
    ]
)

window_symbol_summary = grouped_table(
    [
        "audit_window",
        "symbol",
    ]
)


policy_summary.to_csv(
    REPORT_DIR
    / "v3kv_max_positions_policy_summary.csv",
    index=False,
)

window_summary.to_csv(
    REPORT_DIR
    / "v3kv_max_positions_policy_by_window.csv",
    index=False,
)

rank_summary.to_csv(
    REPORT_DIR
    / "v3kv_max_positions_by_rank.csv",
    index=False,
)

window_rank_summary.to_csv(
    REPORT_DIR
    / "v3kv_max_positions_by_window_rank.csv",
    index=False,
)

symbol_summary.to_csv(
    REPORT_DIR
    / "v3kv_max_positions_by_symbol.csv",
    index=False,
)

window_symbol_summary.to_csv(
    REPORT_DIR
    / "v3kv_max_positions_by_window_symbol.csv",
    index=False,
)


display_columns = [
    "policy",
    "candidates",
    "win_rate_pct",
    "mean_r",
    "median_r",
    "capped_3r_mean",
    "ge_1r",
    "ge_2r",
    "le_minus_1r",
    "top5_positive_share_pct",
    "windows_present",
    "positive_mean_windows",
    "positive_capped_windows",
    "minimum_window_mean_r",
    "minimum_window_capped_mean",
]


lines: list[str] = []


def emit(
    value: str = "",
) -> None:

    print(value)
    lines.append(value)


emit("")
emit("=" * 122)
emit("V3KV MAX-POSITIONS STRUCTURE AUDIT")
emit("=" * 122)
emit(
    "Historical inspected windows only; "
    "not untouched out-of-sample."
)
emit(
    "All policy masks use information "
    "available before entry."
)
emit(
    "No third position, risk increase or "
    "portfolio-rule change was tested."
)
emit("")

emit(
    f"MAX_POSITIONS candidates: "
    f"{len(max_positions)}"
)

emit("")
emit("-" * 122)
emit("CAUSAL POLICY SUBGROUPS")
emit("-" * 122)

emit(
    policy_summary[
        display_columns
    ]
    .sort_values(
        [
            "positive_capped_windows",
            "capped_3r_mean",
            "candidates",
        ],
        ascending=[
            False,
            False,
            False,
        ],
    )
    .to_string(
        index=False,
        float_format=lambda value: (
            f"{value:.4f}"
        ),
    )
)

emit("")
emit("-" * 122)
emit("POLICY RESULTS BY WINDOW")
emit("-" * 122)

emit(
    window_summary[
        [
            "policy",
            "audit_window",
            "candidates",
            "win_rate_pct",
            "mean_r",
            "median_r",
            "capped_3r_mean",
            "ge_1r",
            "ge_2r",
            "le_minus_1r",
        ]
    ]
    .sort_values(
        [
            "policy",
            "audit_window",
        ]
    )
    .to_string(
        index=False,
        float_format=lambda value: (
            f"{value:.4f}"
        ),
    )
)

emit("")
emit("-" * 122)
emit("CANDIDATE-RANK CONTROL")
emit("-" * 122)

emit(
    rank_summary[
        [
            "candidate_rank",
            "candidates",
            "win_rate_pct",
            "mean_r",
            "median_r",
            "capped_3r_mean",
            "ge_1r",
            "ge_2r",
            "le_minus_1r",
            "top5_positive_share_pct",
        ]
    ]
    .to_string(
        index=False,
        float_format=lambda value: (
            f"{value:.4f}"
        ),
    )
)

emit("")
emit("-" * 122)
emit("MASTER OBJECTIVE CHECK")
emit("-" * 122)
emit(
    "1. Loss/drawdown: not yet measured; "
    "shared-portfolio replay has not occurred."
)
emit(
    "2. Upside preservation: rank-based "
    "positive-R opportunity distribution measured."
)
emit(
    "3. Stability: every policy is shown "
    "separately across inspected windows."
)
emit(
    "4. EUR 200 weekly-average target: "
    "cannot be inferred from overlapping shadows."
)
emit(
    "Next gate: only a robust rank-1 subgroup "
    "may proceed to max-two-position replacement replay."
)
emit("")
emit(
    "V3KV max-positions structure audit: PASS"
)

summary_path = (
    REPORT_DIR
    / "v3kv_max_positions_structure_audit_summary.txt"
)

summary_path.write_text(
    "\n".join(lines) + "\n",
    encoding="utf-8",
)
