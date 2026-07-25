from pathlib import Path
import pandas as pd
import numpy as np

detail = pd.read_csv(
    "reports/v3h_loss_excursion_detail.csv"
)

detail["entry_time"] = pd.to_datetime(
    detail["entry_time"]
)

detail["exit_time"] = pd.to_datetime(
    detail["exit_time"]
)

# ----------------------------------------------------------
# Load original trade records
# ----------------------------------------------------------

trade_frames = []

for path in sorted(
    Path("logs/backtests").glob(
        "v3g4_eth365_robustness_*_baseline_trades.csv"
    )
):
    window = (
        path.stem
        .replace("v3g4_eth365_robustness_", "")
        .replace("_baseline_trades", "")
    )

    df = pd.read_csv(path)

    df = df[
        df["symbol"].isin(
            ["LINKUSDT", "SOLUSDT"]
        )
    ].copy()

    if df.empty:
        continue

    df["window"] = window

    df["entry_time"] = pd.to_datetime(
        df["entry_time"]
    )

    df["exit_time"] = pd.to_datetime(
        df["exit_time"]
    )

    trade_frames.append(df)

trades = pd.concat(
    trade_frames,
    ignore_index=True,
)

needed = trades[
    [
        "window",
        "symbol",
        "entry_time",
        "exit_time",
        "entry_price",
        "initial_stop",
        "final_stop",
        "initial_quantity",
        "buy_fee",
        "sell_fees",
        "profit",
        "partial_profit_taken",
        "exit_reason",
    ]
].copy()

merged = detail.merge(
    needed,
    on=[
        "window",
        "symbol",
        "entry_time",
        "exit_time",
    ],
    suffixes=(
        "_diag",
        "_trade",
    ),
    validate="one_to_one",
)

merged["fees"] = (
    merged["buy_fee"].astype(float)
    + merged["sell_fees"].astype(float)
)

merged["gross_before_fees"] = (
    merged["profit_trade"].astype(float)
    + merged["fees"]
)

risk_per_unit = (
    merged["entry_price"].astype(float)
    - merged["initial_stop"].astype(float)
)

merged["final_stop_r"] = np.where(
    risk_per_unit > 0,
    (
        merged["final_stop"].astype(float)
        - merged["entry_price"].astype(float)
    ) / risk_per_unit,
    np.nan,
)

# ----------------------------------------------------------
# Pure diagnostic MFE buckets — NO strategy change
# ----------------------------------------------------------

merged["mfe_bucket"] = pd.cut(
    merged["strict_mfe_r"],
    bins=[
        -np.inf,
        0.5,
        1.0,
        2.0,
        np.inf,
    ],
    labels=[
        "<0.5R",
        "0.5-1R",
        "1-2R",
        ">=2R",
    ],
    right=False,
)

print()
print("=" * 110)
print("V3H FAILURE BUCKETS")
print("=" * 110)

bucket = (
    merged
    .groupby(
        [
            "symbol",
            "mfe_bucket",
        ],
        observed=True,
    )
    .agg(
        trades=("profit_trade", "size"),
        wins=("profit_trade", lambda x: int(
            (x > 0).sum()
        )),
        net_pnl=("profit_trade", "sum"),
        gross_before_fees=(
            "gross_before_fees",
            "sum",
        ),
        fees=("fees", "sum"),
        avg_realized_r=(
            "realized_r",
            "mean",
        ),
        median_mfe_r=(
            "strict_mfe_r",
            "median",
        ),
        median_mae_r=(
            "strict_mae_r",
            "median",
        ),
    )
    .reset_index()
)

bucket["win_rate"] = (
    bucket["wins"]
    / bucket["trades"]
    * 100
)

print(
    bucket.to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}",
    )
)


# ----------------------------------------------------------
# Trades that reached >=1R but still lost
# ----------------------------------------------------------

gave_back = merged[
    (merged["profit_trade"] <= 0)
    & (merged["strict_mfe_r"] >= 1.0)
].copy()

gave_back["stop_at_or_above_entry"] = (
    gave_back["final_stop"]
    >= gave_back["entry_price"]
)

gave_back["near_flat_loss"] = (
    gave_back["realized_r"]
    >= -0.25
)


print()
print("=" * 110)
print(">=1R FAVORABLE BUT FINISHED LOSING")
print("=" * 110)

summary = (
    gave_back
    .groupby("symbol")
    .agg(
        trades=("profit_trade", "size"),
        net_pnl=("profit_trade", "sum"),
        gross_before_fees=(
            "gross_before_fees",
            "sum",
        ),
        fees=("fees", "sum"),
        avg_realized_r=(
            "realized_r",
            "mean",
        ),
        median_mfe_r=(
            "strict_mfe_r",
            "median",
        ),
        stop_at_or_above_entry=(
            "stop_at_or_above_entry",
            "sum",
        ),
        near_flat_losses=(
            "near_flat_loss",
            "sum",
        ),
    )
    .reset_index()
)

print(
    summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}",
    )
)


print()
print("=" * 110)
print("DETAIL — >=1R LOSERS")
print("=" * 110)

print(
    gave_back[
        [
            "window",
            "symbol",
            "entry_time",
            "strict_mfe_r",
            "strict_mae_r",
            "realized_r",
            "profit_trade",
            "gross_before_fees",
            "fees",
            "final_stop_r",
            "stop_at_or_above_entry",
            "exit_reason_trade",
        ]
    ]
    .sort_values(
        [
            "symbol",
            "strict_mfe_r",
        ],
        ascending=[
            True,
            False,
        ],
    )
    .to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}",
    )
)


# ----------------------------------------------------------
# True immediate failures
# ----------------------------------------------------------

failures = merged[
    merged["strict_mfe_r"] < 0.5
]

print()
print("=" * 110)
print("TRUE EARLY FAILURES — NEVER REACHED 0.5R")
print("=" * 110)

early = (
    failures
    .groupby("symbol")
    .agg(
        trades=("profit_trade", "size"),
        net_pnl=("profit_trade", "sum"),
        gross_before_fees=(
            "gross_before_fees",
            "sum",
        ),
        fees=("fees", "sum"),
        median_mfe_r=(
            "strict_mfe_r",
            "median",
        ),
        median_mae_r=(
            "strict_mae_r",
            "median",
        ),
        avg_realized_r=(
            "realized_r",
            "mean",
        ),
    )
    .reset_index()
)

print(
    early.to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}",
    )
)

merged.to_csv(
    "reports/v3h_failure_bucket_detail.csv",
    index=False,
)

print()
print(
    "Saved: reports/v3h_failure_bucket_detail.csv"
)
