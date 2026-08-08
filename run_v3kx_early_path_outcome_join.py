from pathlib import Path

import pandas as pd


REPORTS = Path("reports")

WINDOWS = [
    ("2022", "2022", 87),
    ("2023", "2023", 201),
    ("2024", "2024", 84),
    ("2025H1", "2025h1", 68),
]

EXPECTED_TOTAL_SNAPSHOTS = 440
EXPECTED_TOTAL_TRADES = 139

snapshot_frames = []
trade_frames = []


for window, tag, expected_snapshots in WINDOWS:

    snapshot_path = (
        REPORTS
        / f"v3kx_early_path_snapshots_{tag}.csv"
    )

    trade_path = (
        REPORTS
        / f"v3kx_early_path_{tag}_trades.csv"
    )

    if not snapshot_path.exists():
        raise RuntimeError(
            f"SNAPSHOT_FILE_MISSING: {snapshot_path}"
        )

    if not trade_path.exists():
        raise RuntimeError(
            f"TRADE_FILE_MISSING: {trade_path}"
        )

    snapshots = pd.read_csv(snapshot_path)
    trades = pd.read_csv(trade_path)

    if len(snapshots) != expected_snapshots:
        raise RuntimeError(
            f"SNAPSHOT_COUNT_BAD: {window} "
            f"expected={expected_snapshots} "
            f"actual={len(snapshots)}"
        )

    snapshots.insert(
        0,
        "audit_window",
        window,
    )

    trades.insert(
        0,
        "audit_window",
        window,
    )

    snapshot_frames.append(snapshots)
    trade_frames.append(trades)


snapshots = pd.concat(
    snapshot_frames,
    ignore_index=True,
    sort=False,
)

trades = pd.concat(
    trade_frames,
    ignore_index=True,
    sort=False,
)


if len(snapshots) != EXPECTED_TOTAL_SNAPSHOTS:
    raise RuntimeError(
        "TOTAL_SNAPSHOT_COUNT_BAD: "
        f"{len(snapshots)}"
    )

if len(trades) != EXPECTED_TOTAL_TRADES:
    raise RuntimeError(
        "TOTAL_TRADE_COUNT_BAD: "
        f"{len(trades)}"
    )


for frame in [snapshots, trades]:
    frame["entry_time"] = pd.to_datetime(
        frame["entry_time"],
        utc=True,
        errors="raise",
    )

snapshots["snapshot_time"] = pd.to_datetime(
    snapshots["snapshot_time"],
    utc=True,
    errors="raise",
)

trades["exit_time"] = pd.to_datetime(
    trades["exit_time"],
    utc=True,
    errors="raise",
)


trade_duplicates = int(
    trades.duplicated(
        subset=[
            "audit_window",
            "symbol",
            "entry_time",
        ]
    ).sum()
)

snapshot_duplicates = int(
    snapshots.duplicated(
        subset=["snapshot_key"]
    ).sum()
)

if trade_duplicates != 0:
    raise RuntimeError(
        f"TRADE_DUPLICATES: {trade_duplicates}"
    )

if snapshot_duplicates != 0:
    raise RuntimeError(
        f"SNAPSHOT_DUPLICATES: {snapshot_duplicates}"
    )


trade_lookup = trades[
    [
        "audit_window",
        "symbol",
        "entry_time",
        "exit_time",
        "exit_reason",
        "net_pnl_eur",
        "net_r",
        "holding_hours",
        "breakout_strength",
    ]
].rename(
    columns={
        "exit_time":
            "eventual_exit_time",

        "exit_reason":
            "eventual_exit_reason",

        "net_pnl_eur":
            "eventual_net_pnl_eur",

        "net_r":
            "eventual_net_r",

        "holding_hours":
            "eventual_holding_hours",

        "breakout_strength":
            "trade_breakout_strength",
    }
)


joined = snapshots.merge(
    trade_lookup,
    on=[
        "audit_window",
        "symbol",
        "entry_time",
    ],
    how="left",
    validate="many_to_one",
    indicator=True,
)


join_failures = int(
    (
        joined["_merge"] != "both"
    ).sum()
)

if join_failures != 0:
    raise RuntimeError(
        f"OUTCOME_JOIN_FAILURES: {join_failures}"
    )


after_exit = int(
    (
        joined["snapshot_time"]
        > joined["eventual_exit_time"]
    ).sum()
)

if after_exit != 0:
    raise RuntimeError(
        f"SNAPSHOT_AFTER_EXIT: {after_exit}"
    )


joined["eventual_win"] = (
    joined["eventual_net_r"] > 0
)

joined["eventual_loss"] = (
    joined["eventual_net_r"] < 0
)

joined["eventual_ge_1r"] = (
    joined["eventual_net_r"] >= 1.0
)

joined["eventual_ge_2r"] = (
    joined["eventual_net_r"] >= 2.0
)

joined["eventual_ge_3r"] = (
    joined["eventual_net_r"] >= 3.0
)

joined["remaining_r_vs_exit_now"] = (
    joined["eventual_net_r"]
    - joined["exit_now_net_r"]
)


checkpoint_counts = {
    int(k): int(v)
    for k, v in (
        joined["checkpoint_hours"]
        .astype(int)
        .value_counts()
        .sort_index()
        .items()
    )
}

expected_checkpoint_counts = {
    6: 128,
    12: 119,
    24: 106,
    36: 87,
}

if checkpoint_counts != expected_checkpoint_counts:
    raise RuntimeError(
        "CHECKPOINT_TOTALS_BAD: "
        f"{checkpoint_counts}"
    )


summary_rows = []

for checkpoint, frame in joined.groupby(
    "checkpoint_hours",
    sort=True,
):

    summary_rows.append(
        {
            "checkpoint_hours":
                int(checkpoint),

            "snapshot_rows":
                int(len(frame)),

            "eventual_winners":
                int(
                    frame["eventual_win"].sum()
                ),

            "eventual_losses":
                int(
                    frame["eventual_loss"].sum()
                ),

            "eventual_ge_1r":
                int(
                    frame["eventual_ge_1r"].sum()
                ),

            "eventual_ge_2r":
                int(
                    frame["eventual_ge_2r"].sum()
                ),

            "eventual_ge_3r":
                int(
                    frame["eventual_ge_3r"].sum()
                ),

            "mean_exit_now_net_r":
                float(
                    frame["exit_now_net_r"].mean()
                ),

            "median_exit_now_net_r":
                float(
                    frame["exit_now_net_r"].median()
                ),

            "mean_eventual_net_r":
                float(
                    frame["eventual_net_r"].mean()
                ),

            "median_eventual_net_r":
                float(
                    frame["eventual_net_r"].median()
                ),

            "mean_remaining_r":
                float(
                    frame[
                        "remaining_r_vs_exit_now"
                    ].mean()
                ),
        }
    )


summary = pd.DataFrame(
    summary_rows
)


joined_output = (
    REPORTS
    / "v3kx_early_path_outcome_join.csv"
)

summary_output = (
    REPORTS
    / "v3kx_early_path_outcome_join_summary.csv"
)

text_output = (
    REPORTS
    / "v3kx_early_path_outcome_join_summary.txt"
)


joined.drop(
    columns=["_merge"]
).to_csv(
    joined_output,
    index=False,
)

summary.to_csv(
    summary_output,
    index=False,
)


lines = [
    "=" * 112,
    "V3KX CAUSAL EARLY-PATH -> EVENTUAL OUTCOME JOIN",
    "=" * 112,
    "",
    "Historical inspected windows only; not untouched out-of-sample.",
    "Diagnostic join only. No exit rule, entry filter, portfolio change or risk change.",
    "",
    f"Baseline trades            : {len(trades)}",
    f"Snapshot rows              : {len(joined)}",
    f"Checkpoint counts          : {checkpoint_counts}",
    f"Trade duplicates           : {trade_duplicates}",
    f"Snapshot duplicates        : {snapshot_duplicates}",
    f"Outcome join failures      : {join_failures}",
    f"Snapshots after exit       : {after_exit}",
    "",
    "CHECKPOINT OUTCOME SUMMARY",
    summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    ),
    "",
    "MASTER OBJECTIVE CHECK",
    "1. Loss/drawdown improvement: NOT YET MEASURED; no portfolio replay.",
    "2. Upside preservation: eventual >=1R, >=2R and >=3R winners are now explicitly tagged.",
    "3. Stability: four historical windows are joined under one identical causal contract.",
    "4. EUR 200 weekly-average target: NOT YET MEASURED.",
    "",
    "Next gate: test pre-specified causal early-path features for loss separation across windows.",
    "",
    "V3KX early-path outcome join: PASS",
]


text_output.write_text(
    "\n".join(lines) + "\n",
    encoding="utf-8",
)

print(
    "\n".join(lines)
)
