from pathlib import Path
import numpy as np
import pandas as pd

PATH = Path("reports/v3h_entry_feature_matrix.csv")

df = pd.read_csv(PATH)

FEATURES = [
    "adx14",
    "ema50_slope_12h",
    "momentum_72h",
    "ema50_distance_atr",
    "volume_ratio",
    "rsi14",
    "candle_range_atr",
]

df["is_false"] = (
    df["strict_mfe_r"] < 0.5
)

df["is_runner"] = (
    df["strict_mfe_r"] >= 2.0
)


def spearman_rank(x, y):
    table = pd.DataFrame({
        "x": pd.to_numeric(x, errors="coerce"),
        "y": pd.to_numeric(y, errors="coerce"),
    }).dropna()

    if (
        len(table) < 3
        or table["x"].nunique() < 2
        or table["y"].nunique() < 2
    ):
        return np.nan

    return table["x"].rank().corr(
        table["y"].rank()
    )


bucket_rows = []
corr_rows = []

for symbol, symbol_df in df.groupby("symbol"):

    for feature in FEATURES:

        for window, group in symbol_df.groupby("window"):

            values = pd.to_numeric(
                group[feature],
                errors="coerce",
            )

            valid = group.loc[
                values.notna()
            ].copy()

            if (
                len(valid) < 6
                or valid[feature].nunique() < 3
            ):
                continue

            valid["_pct_rank"] = (
                pd.to_numeric(
                    valid[feature],
                    errors="coerce",
                )
                .rank(
                    method="average",
                    pct=True,
                )
            )

            valid["_bucket"] = pd.cut(
                valid["_pct_rank"],
                bins=[
                    0.0,
                    1 / 3,
                    2 / 3,
                    1.0,
                ],
                labels=[
                    "LOW",
                    "MID",
                    "HIGH",
                ],
                include_lowest=True,
            )

            corr_rows.append({
                "symbol": symbol,
                "feature": feature,
                "window": window,
                "trades": len(valid),
                "spearman_mfe": spearman_rank(
                    valid[feature],
                    valid["strict_mfe_r"],
                ),
                "spearman_realized_r": spearman_rank(
                    valid[feature],
                    valid["realized_r"],
                ),
            })

            for bucket, part in valid.groupby(
                "_bucket",
                observed=True,
            ):
                bucket_rows.append({
                    "symbol": symbol,
                    "feature": feature,
                    "window": window,
                    "bucket": str(bucket),
                    "trades": len(part),
                    "false_rate_pct": (
                        part["is_false"].mean()
                        * 100
                    ),
                    "runner_rate_pct": (
                        part["is_runner"].mean()
                        * 100
                    ),
                    "median_mfe_r": (
                        part["strict_mfe_r"].median()
                    ),
                    "avg_realized_r": (
                        part["realized_r"].mean()
                    ),
                    "net_pnl": (
                        part["profit"].sum()
                    ),
                })


buckets = pd.DataFrame(bucket_rows)
corr = pd.DataFrame(corr_rows)

buckets.to_csv(
    "reports/v3h_feature_monotonicity_buckets.csv",
    index=False,
)

corr.to_csv(
    "reports/v3h_feature_monotonicity_correlations.csv",
    index=False,
)


print()
print("=" * 110)
print("V3H FEATURE MONOTONICITY - CORRELATION")
print("=" * 110)

summary = (
    corr
    .groupby(
        ["symbol", "feature"]
    )
    .agg(
        windows=("window", "size"),
        median_mfe_corr=(
            "spearman_mfe",
            "median",
        ),
        min_mfe_corr=(
            "spearman_mfe",
            "min",
        ),
        max_mfe_corr=(
            "spearman_mfe",
            "max",
        ),
        positive_windows=(
            "spearman_mfe",
            lambda x: int(
                (x > 0).sum()
            ),
        ),
        negative_windows=(
            "spearman_mfe",
            lambda x: int(
                (x < 0).sum()
            ),
        ),
    )
    .reset_index()
)

print(
    summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}",
    )
)


print()
print("=" * 110)
print("LOW vs HIGH FEATURE BUCKETS")
print("=" * 110)

for symbol in [
    "LINKUSDT",
    "SOLUSDT",
]:
    print()
    print(symbol)

    overall = (
        buckets[
            buckets["symbol"] == symbol
        ]
        .groupby(
            ["feature", "bucket"]
        )
        .agg(
            trades=("trades", "sum"),
            avg_false_rate=(
                "false_rate_pct",
                "mean",
            ),
            avg_runner_rate=(
                "runner_rate_pct",
                "mean",
            ),
            median_mfe=(
                "median_mfe_r",
                "median",
            ),
            net_pnl=(
                "net_pnl",
                "sum",
            ),
        )
        .reset_index()
    )

    print(
        overall.to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )


print()
print(
    "Saved: reports/v3h_feature_monotonicity_buckets.csv"
)
print(
    "Saved: reports/v3h_feature_monotonicity_correlations.csv"
)
