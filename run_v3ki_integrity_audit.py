from pathlib import Path

import pandas as pd


SPECS = {
    "2022": {
        "path": Path(
            "reports/v3ki_4h_donchian_2022.csv"
        ),
        "start": pd.Timestamp(
            "2022-01-01 00:00:00"
        ),
        "end": pd.Timestamp(
            "2023-01-01 00:00:00"
        ),
    },
    "2023": {
        "path": Path(
            "reports/v3ki_4h_donchian_2023.csv"
        ),
        "start": pd.Timestamp(
            "2023-01-01 00:00:00"
        ),
        "end": pd.Timestamp(
            "2024-01-01 00:00:00"
        ),
    },
    "2024": {
        "path": Path(
            "reports/v3ki_4h_donchian_2024.csv"
        ),
        "start": pd.Timestamp(
            "2024-01-01 00:00:00"
        ),
        "end": pd.Timestamp(
            "2025-01-01 00:00:00"
        ),
    },
    "2025H1": {
        "path": Path(
            "reports/v3ki_4h_donchian_2025h1.csv"
        ),
        "start": pd.Timestamp(
            "2025-01-01 00:00:00"
        ),
        "end": pd.Timestamp(
            "2025-08-01 00:00:00"
        ),
    },
}

OUTPUT = Path(
    "reports/"
    "v3ki_combined_trimmed_integrity_audit.csv"
)


frames = []


for window, spec in SPECS.items():
    path = spec["path"]

    if not path.exists():
        raise FileNotFoundError(
            f"Missing V3KI result: {path}"
        )

    frame = pd.read_csv(path)

    for column in [
        "signal_time",
        "entry_time",
        "exit_time",
    ]:
        frame[column] = pd.to_datetime(
            frame[column]
        )

    frame["source_window"] = window

    frame["outside_entry_window"] = (
        (
            frame["signal_time"]
            < spec["start"]
        )
        |
        (
            frame["signal_time"]
            >= spec["end"]
        )
    )

    frames.append(frame)


raw = pd.concat(
    frames,
    ignore_index=True,
)


print()
print("=" * 122)
print(
    "V3KI INTEGRITY AUDIT | "
    "WINDOW BOUNDARIES + DUPLICATES"
)
print("=" * 122)


for window, spec in SPECS.items():
    group = raw[
        raw["source_window"] == window
    ]

    outside = group[
        group["outside_entry_window"]
    ]

    end_marks = group[
        group["exit_reason"] == "END_MARK"
    ]

    print(
        f"{window:<8} | "
        f"Raw trades {len(group):>3} | "
        f"Signal range "
        f"{group['signal_time'].min()} -> "
        f"{group['signal_time'].max()} | "
        f"Outside {len(outside):>3} "
        f"({outside['net_r'].sum():>+7.2f}R) | "
        f"End marks {len(end_marks):>3}"
    )


duplicate_mask = raw.duplicated(
    subset=[
        "symbol",
        "signal_time",
    ],
    keep=False,
)

duplicates = raw[
    duplicate_mask
].sort_values(
    [
        "symbol",
        "signal_time",
        "source_window",
    ]
)


print("-" * 122)
print(
    f"Raw rows                     : "
    f"{len(raw)}"
)
print(
    f"Out-of-window rows           : "
    f"{int(raw['outside_entry_window'].sum())}"
)
print(
    f"Duplicate rows by signal     : "
    f"{len(duplicates)}"
)
print(
    f"Unique duplicated signals    : "
    f"{duplicates[
        ['symbol', 'signal_time']
    ].drop_duplicates().shape[0]}"
)


trimmed = raw[
    ~raw["outside_entry_window"]
].copy()

trimmed = trimmed.sort_values(
    [
        "signal_time",
        "symbol",
    ]
)

before_dedup = len(trimmed)

trimmed = trimmed.drop_duplicates(
    subset=[
        "symbol",
        "signal_time",
    ],
    keep="first",
)

removed_after_trim = (
    before_dedup
    - len(trimmed)
)


winners = trimmed[
    trimmed["net_r"] > 0
]

losers = trimmed[
    trimmed["net_r"] < 0
]

gross_positive_r = float(
    winners["net_r"].sum()
)

gross_negative_r = abs(
    float(
        losers["net_r"].sum()
    )
)

profit_factor = (
    gross_positive_r
    / gross_negative_r
    if gross_negative_r > 0
    else float("inf")
    if gross_positive_r > 0
    else 0.0
)


print("-" * 122)
print("TRIMMED + DE-DUPLICATED SIGNAL RESULT")
print("-" * 122)
print(
    f"Completed trades             : "
    f"{len(trimmed)}"
)
print(
    f"Duplicates removed after trim: "
    f"{removed_after_trim}"
)
print(
    f"Winners / Losers            : "
    f"{len(winners)} / {len(losers)}"
)
print(
    f"Total net R                  : "
    f"{trimmed['net_r'].sum():+.2f}R"
)
print(
    f"Average net R                : "
    f"{trimmed['net_r'].mean():+.3f}R"
)
print(
    f"Profit factor                : "
    f"{profit_factor:.2f}"
)


print("-" * 122)
print("TRIMMED RESULT BY WINDOW")
print("-" * 122)


for window in SPECS:
    group = trimmed[
        trimmed["source_window"]
        == window
    ]

    print(
        f"{window:<8} | "
        f"Trades {len(group):>3} | "
        f"Wins "
        f"{int((group['net_r'] > 0).sum()):>3} | "
        f"Net R "
        f"{group['net_r'].sum():>+8.2f}"
    )


print("-" * 122)
print("TRIMMED RESULT BY SYMBOL")
print("-" * 122)


symbol_summary = (
    trimmed.groupby("symbol")
    .agg(
        trades=("net_r", "size"),
        wins=(
            "net_r",
            lambda series: int(
                (series > 0).sum()
            ),
        ),
        net_r=("net_r", "sum"),
        average_r=("net_r", "mean"),
    )
    .reset_index()
    .sort_values(
        "net_r",
        ascending=False,
    )
)

print(
    symbol_summary.to_string(
        index=False,
        formatters={
            "net_r": (
                lambda value:
                f"{value:+.2f}"
            ),
            "average_r": (
                lambda value:
                f"{value:+.3f}"
            ),
        },
    )
)


# ------------------------------------------------------------
# PROFIT CONCENTRATION
# ------------------------------------------------------------

sorted_winners = (
    winners["net_r"]
    .sort_values(
        ascending=False
    )
)

best_trade = (
    float(sorted_winners.iloc[0])
    if len(sorted_winners)
    else 0.0
)

top_five_r = float(
    sorted_winners.head(5).sum()
)

best_trade_share = (
    best_trade
    / gross_positive_r
    * 100.0
    if gross_positive_r > 0
    else 0.0
)

top_five_share = (
    top_five_r
    / gross_positive_r
    * 100.0
    if gross_positive_r > 0
    else 0.0
)


print("-" * 122)
print("CONCENTRATION")
print("-" * 122)
print(
    f"Best trade                   : "
    f"{best_trade:+.2f}R"
)
print(
    f"Best-trade gross-win share   : "
    f"{best_trade_share:.2f}%"
)
print(
    f"Top-five gross-win share     : "
    f"{top_five_share:.2f}%"
)
print(
    f"Net R without best trade     : "
    f"{trimmed['net_r'].sum() - best_trade:+.2f}R"
)


# ------------------------------------------------------------
# RAW CONCURRENCY
# This does not enforce V3D3. It only measures how many
# positions the independent-symbol test held simultaneously.
# ------------------------------------------------------------

events = []

for _, trade in trimmed.iterrows():
    events.append(
        (
            trade["entry_time"],
            1,
        )
    )

    events.append(
        (
            trade["exit_time"],
            -1,
        )
    )

# Exit before entry when timestamps are identical.
events.sort(
    key=lambda event: (
        event[0],
        event[1],
    )
)

open_positions = 0
maximum_open_positions = 0

for _, change in events:
    open_positions += change

    maximum_open_positions = max(
        maximum_open_positions,
        open_positions,
    )


print("-" * 122)
print("PORTFOLIO-CAPACITY WARNING")
print("-" * 122)
print(
    f"Maximum simultaneous trades  : "
    f"{maximum_open_positions}"
)
print(
    "V3D3 production maximum      : 2"
)
print(
    "Meaning                      : "
    "raw summed R is signal-level, "
    "not final portfolio performance."
)


trimmed.to_csv(
    OUTPUT,
    index=False,
)


print("-" * 122)
print(
    f"Trimmed audit report         : "
    f"{OUTPUT}"
)
print("=" * 122)
