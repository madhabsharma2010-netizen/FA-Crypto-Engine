from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import run_v3kv_opportunity_observer as obs


FLOAT_TOLERANCE = 1e-9

DETAIL_OUTPUT = Path(
    "reports/"
    f"v3kv_shadow_outcomes_{obs.TAG}.csv"
)

SUMMARY_OUTPUT = Path(
    "reports/"
    f"v3kv_shadow_outcomes_{obs.TAG}_summary.csv"
)

PARITY_OUTPUT = Path(
    "reports/"
    f"v3kv_shadow_outcomes_{obs.TAG}_executed_parity.csv"
)


def timestamp(
    value: object,
) -> pd.Timestamp:
    return pd.Timestamp(value)


def time_key(
    value: object,
) -> int:
    return int(
        pd.Timestamp(value).value
    )


def candidate_key(
    symbol: str,
    signal_time: object,
) -> str:
    return (
        f"{symbol}|"
        f"{timestamp(signal_time).isoformat()}"
    )


due_rows = [
    dict(row)
    for row in obs.v3kv_observer_rows
    if row.get(
        "observer_stage"
    ) == "DUE"
]

if not due_rows:
    raise RuntimeError(
        "NO_DUE_OBSERVER_ROWS"
    )


due_by_time: dict[
    int,
    list[dict[str, object]],
] = defaultdict(list)

seen_candidate_ids: set[str] = set()


for row in due_rows:

    symbol = str(
        row["symbol"]
    )

    signal_time = timestamp(
        row["signal_time"]
    )

    event_time = timestamp(
        row["event_time"]
    )

    row["symbol"] = symbol
    row["signal_time"] = signal_time
    row["due_time"] = timestamp(
        row["due_time"]
    )
    row["event_time"] = event_time

    candidate_id = candidate_key(
        symbol,
        signal_time,
    )

    if candidate_id in seen_candidate_ids:
        raise RuntimeError(
            "DUPLICATE_DUE_CANDIDATE: "
            f"{candidate_id}"
        )

    seen_candidate_ids.add(
        candidate_id
    )

    row["candidate_id"] = (
        candidate_id
    )

    due_by_time[
        time_key(event_time)
    ].append(
        row
    )


equity_state = pd.DataFrame(
    obs.equity_rows
)

if equity_state.empty:
    raise RuntimeError(
        "OBSERVER_EQUITY_ROWS_EMPTY"
    )

required_equity_columns = {
    "time",
    "daily_entry_block",
    "weekly_entry_block",
    "hard_lock",
}

missing_equity_columns = sorted(
    required_equity_columns
    - set(equity_state.columns)
)

if missing_equity_columns:
    raise RuntimeError(
        "MISSING_EQUITY_STATE_COLUMNS: "
        f"{missing_equity_columns}"
    )


processed_times: list[
    pd.Timestamp
] = []

state_by_time: dict[
    int,
    dict[str, object],
] = {}

for raw_equity_row in obs.equity_rows:

    raw_event_time = timestamp(
        raw_equity_row["time"]
    )

    raw_event_key = time_key(
        raw_event_time
    )

    if raw_event_key in state_by_time:
        continue

    processed_times.append(
        raw_event_time
    )

    state_by_time[
        raw_event_key
    ] = dict(
        raw_equity_row
    )


risk_exit_reason: dict[
    int,
    str,
] = {}

previous_hard = False
previous_weekly = False
previous_daily = False


for event_time in processed_times:

    event_key = time_key(
        event_time
    )

    row = state_by_time[
        event_key
    ]

    current_hard = bool(
        row["hard_lock"]
    )

    current_weekly = bool(
        row["weekly_entry_block"]
    )

    current_daily = bool(
        row["daily_entry_block"]
    )

    reason = None

    if (
        current_hard
        and not previous_hard
    ):
        reason = (
            "HARD DRAWDOWN LOCK"
        )

    elif (
        current_weekly
        and not previous_weekly
    ):
        reason = (
            "WEEKLY LOSS LIMIT"
        )

    elif (
        current_daily
        and not previous_daily
    ):
        reason = (
            "DAILY LOSS LIMIT"
        )

    if reason is not None:
        risk_exit_reason[
            event_key
        ] = reason

    previous_hard = current_hard
    previous_weekly = current_weekly
    previous_daily = current_daily


shadow_positions: dict[
    str,
    dict[str, object],
] = {}

shadow_trades: list[
    dict[str, object]
] = []

last_prices: dict[
    str,
    float,
] = {}


def open_shadow(
    row: dict[str, object],
) -> None:

    candidate_id = str(
        row["candidate_id"]
    )

    if candidate_id in shadow_positions:
        raise RuntimeError(
            "SHADOW_ALREADY_OPEN: "
            f"{candidate_id}"
        )

    entry_price = float(
        row["entry_price"]
    )

    atr_4h = float(
        row["atr_4h"]
    )

    initial_stop = (
        entry_price
        - float(
            obs.INITIAL_STOP_ATR
        )
        * atr_4h
    )

    initial_risk_per_unit = (
        entry_price
        - initial_stop
    )

    if initial_risk_per_unit <= 0:
        raise RuntimeError(
            "INVALID_SHADOW_RISK: "
            f"{candidate_id}"
        )

    quantity = 1.0

    entry_notional = (
        quantity
        * entry_price
    )

    buy_fee = obs.calculate_fee(
        entry_notional
    )

    shadow_positions[
        candidate_id
    ] = {
        "metadata":
            dict(row),

        "candidate_id":
            candidate_id,

        "symbol":
            str(
                row["symbol"]
            ),

        "signal_time":
            timestamp(
                row["signal_time"]
            ),

        "entry_time":
            timestamp(
                row["event_time"]
            ),

        "entry_price":
            entry_price,

        "initial_stop":
            initial_stop,

        "stop_price":
            initial_stop,

        "initial_risk_per_unit":
            initial_risk_per_unit,

        "quantity":
            quantity,

        "entry_notional":
            entry_notional,

        "buy_fee":
            buy_fee,

        "entry_total_cost":
            entry_notional
            + buy_fee,

        "pending_due_time":
            None,

        "pending_reason":
            None,
    }


def close_shadow(
    candidate_id: str,
    raw_exit: float,
    exit_time: pd.Timestamp,
    exit_reason: str,
) -> None:

    position = shadow_positions.pop(
        candidate_id
    )

    execution_exit = obs.sell_price(
        float(raw_exit)
    )

    quantity = float(
        position["quantity"]
    )

    sell_notional = (
        quantity
        * execution_exit
    )

    sell_fee = obs.calculate_fee(
        sell_notional
    )

    cash_received = (
        sell_notional
        - sell_fee
    )

    net_pnl_eur = (
        cash_received
        - float(
            position[
                "entry_total_cost"
            ]
        )
    )

    initial_risk_eur = (
        quantity
        * float(
            position[
                "initial_risk_per_unit"
            ]
        )
    )

    net_r = (
        net_pnl_eur
        / initial_risk_eur
        if initial_risk_eur > 0
        else 0.0
    )

    holding_hours = (
        timestamp(exit_time)
        - timestamp(
            position["entry_time"]
        )
    ).total_seconds() / 3600.0

    record = dict(
        position["metadata"]
    )

    record.update(
        {
            "entry_time":
                position[
                    "entry_time"
                ],

            "exit_time":
                timestamp(
                    exit_time
                ),

            "entry_price":
                position[
                    "entry_price"
                ],

            "initial_stop":
                position[
                    "initial_stop"
                ],

            "final_stop":
                position[
                    "stop_price"
                ],

            "exit_price":
                execution_exit,

            "quantity":
                quantity,

            "entry_notional":
                position[
                    "entry_notional"
                ],

            "buy_fee":
                position[
                    "buy_fee"
                ],

            "sell_fee":
                sell_fee,

            "initial_risk_eur":
                initial_risk_eur,

            "exit_reason":
                exit_reason,

            "net_pnl_eur":
                net_pnl_eur,

            "net_r":
                net_r,

            "holding_hours":
                holding_hours,
        }
    )

    shadow_trades.append(
        record
    )


def close_all_shadow(
    event_time: pd.Timestamp,
    reason: str,
) -> None:

    for candidate_id in list(
        shadow_positions.keys()
    ):

        position = (
            shadow_positions[
                candidate_id
            ]
        )

        symbol = str(
            position["symbol"]
        )

        if symbol not in last_prices:
            raise RuntimeError(
                "MISSING_CLOSE_MARK: "
                f"candidate={candidate_id}, "
                f"time={event_time}"
            )

        close_shadow(
            candidate_id=candidate_id,
            raw_exit=float(
                last_prices[symbol]
            ),
            exit_time=event_time,
            exit_reason=reason,
        )


for event_time in processed_times:

    event_key = time_key(
        event_time
    )

    current_candles = {
        symbol: (
            obs.frames_1h[
                symbol
            ].loc[event_time]
        )
        for symbol in obs.SYMBOLS
        if event_time
        in obs.frames_1h[
            symbol
        ].index
    }

    if not current_candles:
        continue

    # Pending 4H exits execute first.
    for candidate_id in list(
        shadow_positions.keys()
    ):

        position = (
            shadow_positions[
                candidate_id
            ]
        )

        pending_due_time = (
            position[
                "pending_due_time"
            ]
        )

        symbol = str(
            position["symbol"]
        )

        if (
            pending_due_time is None
            or event_time
            < timestamp(
                pending_due_time
            )
            or symbol
            not in current_candles
        ):
            continue

        close_shadow(
            candidate_id=candidate_id,
            raw_exit=float(
                current_candles[
                    symbol
                ]["open"]
            ),
            exit_time=event_time,
            exit_reason=str(
                position[
                    "pending_reason"
                ]
            ),
        )

    # Every causal due candidate receives
    # an independent shadow entry.
    for row in due_by_time.get(
        event_key,
        [],
    ):
        open_shadow(
            row
        )

    # Intrabar active stop after entry.
    for candidate_id in list(
        shadow_positions.keys()
    ):

        position = (
            shadow_positions[
                candidate_id
            ]
        )

        symbol = str(
            position["symbol"]
        )

        if symbol not in current_candles:
            continue

        candle = current_candles[
            symbol
        ]

        raw_open = float(
            candle["open"]
        )

        low_price = float(
            candle["low"]
        )

        stop_price = float(
            position["stop_price"]
        )

        if low_price <= stop_price:

            raw_stop_fill = min(
                stop_price,
                raw_open,
            )

            close_shadow(
                candidate_id=candidate_id,
                raw_exit=raw_stop_fill,
                exit_time=event_time,
                exit_reason=(
                    "HARD_OR_TRAILING_STOP"
                ),
            )

    # Hourly closing marks.
    for symbol, candle in (
        current_candles.items()
    ):
        last_prices[symbol] = float(
            candle["close"]
        )

    # Canonical risk-event timing remains frozen.
    if event_key in risk_exit_reason:

        close_all_shadow(
            event_time=event_time,
            reason=risk_exit_reason[
                event_key
            ],
        )

    state = state_by_time[
        event_key
    ]

    current_hard_lock = bool(
        state["hard_lock"]
    )

    # Completed 4H stop update occurs last.
    if not current_hard_lock:

        for symbol in obs.SYMBOLS:

            if (
                event_time
                not in obs.bars_4h[
                    symbol
                ].index
            ):
                continue

            bar_4h = (
                obs.bars_4h[
                    symbol
                ].loc[event_time]
            )

            prior_low = bar_4h[
                "PRIOR_LOW_10"
            ]

            if pd.isna(
                prior_low
            ):
                continue

            candidate_stop = float(
                prior_low
            )

            matching_ids = [
                candidate_id
                for candidate_id, position
                in shadow_positions.items()
                if str(
                    position["symbol"]
                ) == symbol
            ]

            for candidate_id in matching_ids:

                position = (
                    shadow_positions[
                        candidate_id
                    ]
                )

                position[
                    "stop_price"
                ] = max(
                    float(
                        position[
                            "stop_price"
                        ]
                    ),
                    candidate_stop,
                )

                if (
                    float(
                        bar_4h["close"]
                    )
                    <= candidate_stop
                ):
                    position[
                        "pending_due_time"
                    ] = (
                        event_time
                        + pd.Timedelta(
                            hours=1
                        )
                    )

                    position[
                        "pending_reason"
                    ] = (
                        "4H_DONCHIAN_EXIT"
                    )


if shadow_positions:

    final_time = processed_times[-1]

    close_all_shadow(
        event_time=final_time,
        reason="END_MARK",
    )


shadow_detail = pd.DataFrame(
    shadow_trades
)

if len(shadow_detail) != len(
    due_rows
):
    raise RuntimeError(
        "SHADOW_COUNT_MISMATCH: "
        f"expected={len(due_rows)}, "
        f"actual={len(shadow_detail)}"
    )

if shadow_detail[
    "candidate_id"
].duplicated().any():
    raise RuntimeError(
        "DUPLICATE_SHADOW_OUTCOMES"
    )


canonical = pd.DataFrame(
    obs.trades
)

executed_shadow = shadow_detail.loc[
    shadow_detail[
        "disposition"
    ].eq("EXECUTED")
].copy()


for frame in (
    canonical,
    executed_shadow,
):
    for column in (
        "signal_time",
        "entry_time",
        "exit_time",
    ):
        frame[column] = pd.to_datetime(
            frame[column],
            utc=True,
            errors="raise",
        )


parity = executed_shadow.merge(
    canonical,
    on=[
        "symbol",
        "signal_time",
    ],
    how="outer",
    suffixes=(
        "_shadow",
        "_canonical",
    ),
    indicator=True,
    validate="one_to_one",
)


join_mismatches = int(
    parity["_merge"]
    .ne("both")
    .sum()
)

if join_mismatches != 0:
    raise RuntimeError(
        "EXECUTED_PARITY_JOIN_FAILED: "
        f"{join_mismatches}"
    )


for column in (
    "entry_time",
    "exit_time",
):
    parity[
        f"{column}_match"
    ] = (
        parity[
            f"{column}_shadow"
        ]
        .eq(
            parity[
                f"{column}_canonical"
            ]
        )
    )


parity[
    "exit_reason_match"
] = (
    parity[
        "exit_reason_shadow"
    ]
    .eq(
        parity[
            "exit_reason_canonical"
        ]
    )
)


float_columns = (
    "entry_price",
    "initial_stop",
    "final_stop",
    "exit_price",
    "net_r",
    "holding_hours",
)

for column in float_columns:

    parity[
        f"{column}_abs_diff"
    ] = (
        pd.to_numeric(
            parity[
                f"{column}_shadow"
            ],
            errors="raise",
        )
        - pd.to_numeric(
            parity[
                f"{column}_canonical"
            ],
            errors="raise",
        )
    ).abs()


time_mismatches = int(
    (
        ~parity[
            "entry_time_match"
        ]
        | ~parity[
            "exit_time_match"
        ]
    ).sum()
)

reason_mismatches = int(
    (
        ~parity[
            "exit_reason_match"
        ]
    ).sum()
)

max_float_difference = float(
    parity[
        [
            f"{column}_abs_diff"
            for column
            in float_columns
        ]
    ]
    .max()
    .max()
)


if time_mismatches != 0:
    raise RuntimeError(
        "SHADOW_TIME_PARITY_FAILED: "
        f"{time_mismatches}"
    )

if reason_mismatches != 0:
    raise RuntimeError(
        "SHADOW_REASON_PARITY_FAILED: "
        f"{reason_mismatches}"
    )

if (
    max_float_difference
    > FLOAT_TOLERANCE
):
    raise RuntimeError(
        "SHADOW_FLOAT_PARITY_FAILED: "
        f"max_diff="
        f"{max_float_difference}"
    )


shadow_summary = (
    shadow_detail
    .groupby(
        [
            "window",
            "disposition",
        ],
        dropna=False,
    )
    .agg(
        candidates=(
            "candidate_id",
            "size",
        ),

        winners=(
            "net_r",
            lambda values:
                int(
                    (
                        values > 0
                    ).sum()
                ),
        ),

        losers=(
            "net_r",
            lambda values:
                int(
                    (
                        values < 0
                    ).sum()
                ),
        ),

        total_net_r=(
            "net_r",
            "sum",
        ),

        mean_net_r=(
            "net_r",
            "mean",
        ),

        median_net_r=(
            "net_r",
            "median",
        ),

        winners_2r_plus=(
            "net_r",
            lambda values:
                int(
                    (
                        values >= 2.0
                    ).sum()
                ),
        ),

        losers_minus_1r_or_worse=(
            "net_r",
            lambda values:
                int(
                    (
                        values <= -1.0
                    ).sum()
                ),
        ),
    )
    .reset_index()
)


for output_path in (
    DETAIL_OUTPUT,
    SUMMARY_OUTPUT,
    PARITY_OUTPUT,
):
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


shadow_detail.to_csv(
    DETAIL_OUTPUT,
    index=False,
)

shadow_summary.to_csv(
    SUMMARY_OUTPUT,
    index=False,
)

parity.to_csv(
    PARITY_OUTPUT,
    index=False,
)


print("")
print(
    "========== V3KV SHADOW OUTCOMES =========="
)

print(
    f"Window:              {obs.WINDOW}"
)

print(
    f"Due candidates:      {len(due_rows)}"
)

print(
    f"Shadow outcomes:     {len(shadow_detail)}"
)

print(
    f"Executed parity set: {len(executed_shadow)}"
)

print(
    f"Time mismatches:     {time_mismatches}"
)

print(
    f"Reason mismatches:   {reason_mismatches}"
)

print(
    "Max numeric diff:   "
    f"{max_float_difference:.12f}"
)

print(
    f"Open shadows:        {len(shadow_positions)}"
)

print("")
print(
    "Executed exit parity: PASS"
)

print(
    f"Detail report:       {DETAIL_OUTPUT}"
)

print(
    f"Summary report:      {SUMMARY_OUTPUT}"
)

print(
    f"Parity report:       {PARITY_OUTPUT}"
)

print("")
print(
    "Shadow trades are independent diagnostics."
)

print(
    "They do not alter portfolio cash, ranking, "
    "position limits or frozen risk."
)
