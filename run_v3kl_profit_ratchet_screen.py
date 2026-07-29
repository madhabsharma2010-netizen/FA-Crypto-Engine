from __future__ import annotations

from pathlib import Path
import os

import pandas as pd

import run_v3ka_sol_ema_reload_engine as eng


# ============================================================================
# V3KL FIXED-ENTRY PROFIT-RATCHET SCREEN
#
# Uses the exact trades selected and sized by V3KJ.
# It changes only trade management.
#
# This is intentionally not yet a full portfolio rerun:
# earlier exits do not create additional entry capacity in this screen.
# ============================================================================


WINDOW = os.environ.get(
    "V3G4_WINDOW",
    "2022",
).upper()

TAG = WINDOW.lower()


SUPPORTED_WINDOWS = {
    "2022",
    "2023",
    "2024",
    "2025H1",
}


if WINDOW not in SUPPORTED_WINDOWS:
    raise ValueError(
        f"Unsupported V3KL window: {WINDOW}"
    )


INPUT_FILE = Path(
    f"reports/"
    f"v3kj_shared_portfolio_{TAG}_trades.csv"
)

DETAIL_OUTPUT = Path(
    f"reports/"
    f"v3kl_profit_ratchet_{TAG}_detail.csv"
)

SUMMARY_OUTPUT = Path(
    f"reports/"
    f"v3kl_profit_ratchet_{TAG}_summary.csv"
)


# Frozen V3KL overlay.
EARLY_PROOF_TRIGGER_R = 0.50
EARLY_PROOF_STOP_R = -0.10

FEE_BREAK_EVEN_TRIGGER_R = 1.00

PARTIAL_TRIGGER_R = 2.00
PARTIAL_PERCENT = 25.00

PEAK_PROFIT_RETENTION_PERCENT = 70.00

NO_PROGRESS_HOURS = 36.0
NO_PROGRESS_MINIMUM_MFE_R = 0.50


TRADING_FEE_PERCENT = float(
    eng.TRADING_FEE_PERCENT
)

SLIPPAGE_PERCENT = float(
    eng.SLIPPAGE_PERCENT
)

FEE_RATE = (
    TRADING_FEE_PERCENT
    / 100.0
)

SLIPPAGE_RATE = (
    SLIPPAGE_PERCENT
    / 100.0
)


def normalize_timestamp(
    value,
) -> pd.Timestamp:

    timestamp = pd.Timestamp(
        value
    )

    if timestamp.tzinfo is not None:
        timestamp = (
            timestamp
            .tz_convert(None)
        )

    return timestamp


def sell_execution_price(
    raw_price: float,
) -> float:

    return float(raw_price) * (
        1.0
        - SLIPPAGE_RATE
    )


def sell_fee(
    quantity: float,
    execution_price: float,
) -> float:

    return (
        float(quantity)
        * float(execution_price)
        * FEE_RATE
    )


def profit_factor(
    pnl_values: pd.Series,
) -> float:

    positive = float(
        pnl_values[
            pnl_values > 0
        ].sum()
    )

    negative = abs(
        float(
            pnl_values[
                pnl_values < 0
            ].sum()
        )
    )

    if negative > 0:
        return (
            positive
            / negative
        )

    if positive > 0:
        return float("inf")

    return 0.0


def complete_exit_from_raw_price(
    *,
    raw_price: float,
    remaining_quantity: float,
    remaining_entry_cost: float,
    realized_partial_pnl: float,
) -> tuple[
    float,
    float,
    float,
]:

    execution_price = (
        sell_execution_price(
            raw_price
        )
    )

    fee = sell_fee(
        remaining_quantity,
        execution_price,
    )

    net_proceeds = (
        remaining_quantity
        * execution_price
        - fee
    )

    total_trade_pnl = (
        realized_partial_pnl
        + net_proceeds
        - remaining_entry_cost
    )

    return (
        execution_price,
        fee,
        total_trade_pnl,
    )


def complete_exit_from_execution_price(
    *,
    execution_price: float,
    remaining_quantity: float,
    remaining_entry_cost: float,
    realized_partial_pnl: float,
) -> tuple[
    float,
    float,
    float,
]:

    fee = sell_fee(
        remaining_quantity,
        execution_price,
    )

    net_proceeds = (
        remaining_quantity
        * execution_price
        - fee
    )

    total_trade_pnl = (
        realized_partial_pnl
        + net_proceeds
        - remaining_entry_cost
    )

    return (
        execution_price,
        fee,
        total_trade_pnl,
    )


if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Missing V3KJ trades file: "
        f"{INPUT_FILE}"
    )


baseline = pd.read_csv(
    INPUT_FILE
)


if baseline.empty:
    raise RuntimeError(
        f"No V3KJ trades found in "
        f"{INPUT_FILE}"
    )


required_columns = {
    "symbol",
    "entry_time",
    "exit_time",
    "entry_price",
    "initial_stop",
    "quantity",
    "entry_notional",
    "buy_fee",
    "initial_risk_eur",
    "exit_price",
    "exit_reason",
    "net_pnl_eur",
    "net_r",
}


missing_columns = (
    required_columns
    - set(baseline.columns)
)


if missing_columns:
    raise RuntimeError(
        "V3KJ trade report is missing: "
        + ", ".join(
            sorted(
                missing_columns
            )
        )
    )


frames: dict[
    str,
    pd.DataFrame,
] = {}


for symbol in sorted(
    baseline["symbol"].unique()
):
    frame = (
        eng.prepare_1h(symbol)
        .copy()
        .sort_index()
    )

    frame.index = pd.to_datetime(
        frame.index
    )

    if frame.index.tz is not None:
        frame.index = (
            frame.index
            .tz_convert(None)
        )

    frame = frame[
        ~frame.index.duplicated(
            keep="last"
        )
    ]

    frames[symbol] = frame


results: list[dict] = []


for trade_number, row in (
    baseline.reset_index(
        drop=True
    ).iterrows()
):

    symbol = str(
        row["symbol"]
    )

    entry_time = normalize_timestamp(
        row["entry_time"]
    )

    baseline_exit_time = (
        normalize_timestamp(
            row["exit_time"]
        )
    )

    frame = frames[symbol]

    path = frame.loc[
        (
            frame.index
            >= entry_time
        )
        & (
            frame.index
            <= baseline_exit_time
        )
    ]


    if path.empty:
        raise RuntimeError(
            f"No hourly path for trade "
            f"{trade_number}: "
            f"{symbol} "
            f"{entry_time} -> "
            f"{baseline_exit_time}"
        )


    entry_price = float(
        row["entry_price"]
    )

    initial_stop = float(
        row["initial_stop"]
    )

    original_quantity = float(
        row["quantity"]
    )

    initial_risk_eur = float(
        row["initial_risk_eur"]
    )


    if original_quantity <= 0:
        raise RuntimeError(
            f"Invalid quantity for "
            f"trade {trade_number}"
        )


    initial_risk_per_unit = (
        entry_price
        - initial_stop
    )


    if initial_risk_per_unit <= 0:
        raise RuntimeError(
            f"Invalid initial risk for "
            f"trade {trade_number}"
        )


    original_entry_cost = (
        float(
            row["entry_notional"]
        )
        + float(
            row["buy_fee"]
        )
    )

    entry_cost_per_unit = (
        original_entry_cost
        / original_quantity
    )


    # Raw market trigger required to recover:
    # buy cost + sell slippage + sell fee.
    fee_adjusted_break_even_raw = (
        entry_cost_per_unit
        / (
            (
                1.0
                - SLIPPAGE_RATE
            )
            * (
                1.0
                - FEE_RATE
            )
        )
    )


    remaining_quantity = (
        original_quantity
    )

    remaining_entry_cost = (
        original_entry_cost
    )

    realized_partial_pnl = 0.0

    partial_taken = False
    partial_time = pd.NaT
    partial_execution_price = 0.0
    partial_fee_paid = 0.0
    partial_pnl = 0.0


    active_stop = initial_stop
    active_stop_mode = (
        "ORIGINAL_HARD_STOP"
    )

    maximum_favorable_r = 0.0

    final_execution_price = None
    final_sell_fee = 0.0
    overlay_net_pnl = None
    overlay_exit_time = None
    overlay_exit_reason = None


    for event_time, candle in (
        path.iterrows()
    ):

        event_time = normalize_timestamp(
            event_time
        )

        candle_open = float(
            candle["open"]
        )

        candle_high = float(
            candle["high"]
        )

        candle_low = float(
            candle["low"]
        )

        candle_close = float(
            candle["close"]
        )


        # ------------------------------------------------------------
        # Conservative intrabar ordering:
        # the stop that existed before this candle is checked first.
        # Any new stop based on this candle's high starts next hour.
        # ------------------------------------------------------------

        if candle_low <= active_stop:

            raw_stop_fill = min(
                active_stop,
                candle_open,
            )

            (
                final_execution_price,
                final_sell_fee,
                overlay_net_pnl,
            ) = complete_exit_from_raw_price(
                raw_price=raw_stop_fill,
                remaining_quantity=(
                    remaining_quantity
                ),
                remaining_entry_cost=(
                    remaining_entry_cost
                ),
                realized_partial_pnl=(
                    realized_partial_pnl
                ),
            )

            overlay_exit_time = (
                event_time
            )

            overlay_exit_reason = (
                active_stop_mode
            )

            break


        # At the original exit timestamp, preserve the baseline
        # Donchian/circuit-breaker execution as the fallback.
        if event_time >= baseline_exit_time:

            (
                final_execution_price,
                final_sell_fee,
                overlay_net_pnl,
            ) = (
                complete_exit_from_execution_price(
                    execution_price=float(
                        row["exit_price"]
                    ),
                    remaining_quantity=(
                        remaining_quantity
                    ),
                    remaining_entry_cost=(
                        remaining_entry_cost
                    ),
                    realized_partial_pnl=(
                        realized_partial_pnl
                    ),
                )
            )

            overlay_exit_time = (
                baseline_exit_time
            )

            overlay_exit_reason = (
                "BASELINE_"
                + str(
                    row["exit_reason"]
                )
            )

            break


        candle_peak_r = (
            (
                candle_high
                - entry_price
            )
            / initial_risk_per_unit
        )

        maximum_favorable_r = max(
            maximum_favorable_r,
            candle_peak_r,
        )


        # ------------------------------------------------------------
        # +2R: sell 25% once.
        # A resting profit order is assumed at the exact +2R trigger.
        # Stop-first ordering above prevents optimistic same-candle bias.
        # ------------------------------------------------------------

        if (
            not partial_taken
            and maximum_favorable_r
            >= PARTIAL_TRIGGER_R
        ):

            partial_quantity = (
                original_quantity
                * PARTIAL_PERCENT
                / 100.0
            )

            partial_quantity = min(
                partial_quantity,
                remaining_quantity,
            )

            partial_raw_price = (
                entry_price
                + PARTIAL_TRIGGER_R
                * initial_risk_per_unit
            )

            partial_execution_price = (
                sell_execution_price(
                    partial_raw_price
                )
            )

            partial_fee_paid = sell_fee(
                partial_quantity,
                partial_execution_price,
            )

            partial_net_proceeds = (
                partial_quantity
                * partial_execution_price
                - partial_fee_paid
            )

            partial_cost_basis = (
                entry_cost_per_unit
                * partial_quantity
            )

            partial_pnl = (
                partial_net_proceeds
                - partial_cost_basis
            )

            realized_partial_pnl += (
                partial_pnl
            )

            remaining_quantity -= (
                partial_quantity
            )

            remaining_entry_cost -= (
                partial_cost_basis
            )

            partial_taken = True
            partial_time = event_time


        holding_hours = (
            event_time
            - entry_time
        ).total_seconds() / 3600.0


        # ------------------------------------------------------------
        # 36-hour no-progress exit.
        # ------------------------------------------------------------

        if (
            holding_hours
            >= NO_PROGRESS_HOURS
            and maximum_favorable_r
            < NO_PROGRESS_MINIMUM_MFE_R
        ):

            (
                final_execution_price,
                final_sell_fee,
                overlay_net_pnl,
            ) = complete_exit_from_raw_price(
                raw_price=candle_close,
                remaining_quantity=(
                    remaining_quantity
                ),
                remaining_entry_cost=(
                    remaining_entry_cost
                ),
                realized_partial_pnl=(
                    realized_partial_pnl
                ),
            )

            overlay_exit_time = (
                event_time
            )

            overlay_exit_reason = (
                "V3KL_NO_PROGRESS_36H"
            )

            break


        # ------------------------------------------------------------
        # Stop ratchet for the next hourly candle.
        # Stop can only tighten; never widen.
        # ------------------------------------------------------------

        next_stop = active_stop
        next_stop_mode = (
            active_stop_mode
        )


        if (
            maximum_favorable_r
            >= EARLY_PROOF_TRIGGER_R
        ):

            early_proof_stop = (
                entry_price
                + EARLY_PROOF_STOP_R
                * initial_risk_per_unit
            )

            if early_proof_stop > next_stop:
                next_stop = (
                    early_proof_stop
                )

                next_stop_mode = (
                    "V3KL_SMALL_LOSS_STOP"
                )


        if (
            maximum_favorable_r
            >= FEE_BREAK_EVEN_TRIGGER_R
        ):

            if (
                fee_adjusted_break_even_raw
                > next_stop
            ):
                next_stop = (
                    fee_adjusted_break_even_raw
                )

                next_stop_mode = (
                    "V3KL_FEE_BREAK_EVEN_STOP"
                )


        if (
            maximum_favorable_r
            >= PARTIAL_TRIGGER_R
        ):

            protected_r = (
                maximum_favorable_r
                * PEAK_PROFIT_RETENTION_PERCENT
                / 100.0
            )

            ratchet_stop = (
                entry_price
                + protected_r
                * initial_risk_per_unit
            )

            if ratchet_stop > next_stop:
                next_stop = (
                    ratchet_stop
                )

                next_stop_mode = (
                    "V3KL_70_PERCENT_"
                    "PROFIT_RATCHET_STOP"
                )


        active_stop = max(
            active_stop,
            next_stop,
        )

        active_stop_mode = (
            next_stop_mode
        )


    if overlay_net_pnl is None:
        raise RuntimeError(
            f"Trade {trade_number} "
            f"did not produce an exit."
        )


    overlay_net_r = (
        overlay_net_pnl
        / initial_risk_eur
        if initial_risk_eur > 0
        else 0.0
    )

    overlay_holding_hours = (
        overlay_exit_time
        - entry_time
    ).total_seconds() / 3600.0


    results.append(
        {
            "window": WINDOW,
            "trade_number": (
                trade_number + 1
            ),
            "symbol": symbol,
            "entry_time": entry_time,
            "baseline_exit_time": (
                baseline_exit_time
            ),
            "overlay_exit_time": (
                overlay_exit_time
            ),
            "entry_price": entry_price,
            "initial_stop": initial_stop,
            "original_quantity": (
                original_quantity
            ),
            "remaining_quantity_at_exit": (
                remaining_quantity
            ),
            "maximum_favorable_r": (
                maximum_favorable_r
            ),
            "partial_taken": (
                partial_taken
            ),
            "partial_time": (
                partial_time
            ),
            "partial_execution_price": (
                partial_execution_price
            ),
            "partial_fee": (
                partial_fee_paid
            ),
            "partial_pnl_eur": (
                partial_pnl
            ),
            "final_execution_price": (
                final_execution_price
            ),
            "final_sell_fee": (
                final_sell_fee
            ),
            "baseline_exit_reason": (
                row["exit_reason"]
            ),
            "overlay_exit_reason": (
                overlay_exit_reason
            ),
            "baseline_holding_hours": (
                float(
                    row["holding_hours"]
                )
            ),
            "overlay_holding_hours": (
                overlay_holding_hours
            ),
            "baseline_net_pnl_eur": (
                float(
                    row["net_pnl_eur"]
                )
            ),
            "overlay_net_pnl_eur": (
                overlay_net_pnl
            ),
            "delta_net_pnl_eur": (
                overlay_net_pnl
                - float(
                    row["net_pnl_eur"]
                )
            ),
            "baseline_net_r": (
                float(
                    row["net_r"]
                )
            ),
            "overlay_net_r": (
                overlay_net_r
            ),
            "delta_net_r": (
                overlay_net_r
                - float(
                    row["net_r"]
                )
            ),
        }
    )


detail = pd.DataFrame(
    results
)


baseline_total_pnl = float(
    detail[
        "baseline_net_pnl_eur"
    ].sum()
)

overlay_total_pnl = float(
    detail[
        "overlay_net_pnl_eur"
    ].sum()
)

baseline_total_r = float(
    detail[
        "baseline_net_r"
    ].sum()
)

overlay_total_r = float(
    detail[
        "overlay_net_r"
    ].sum()
)


baseline_winners = int(
    (
        detail[
            "baseline_net_pnl_eur"
        ]
        > 0
    ).sum()
)

overlay_winners = int(
    (
        detail[
            "overlay_net_pnl_eur"
        ]
        > 0
    ).sum()
)


baseline_pf = profit_factor(
    detail[
        "baseline_net_pnl_eur"
    ]
)

overlay_pf = profit_factor(
    detail[
        "overlay_net_pnl_eur"
    ]
)


summary = pd.DataFrame(
    [
        {
            "window": WINDOW,
            "fixed_entry_screen": True,
            "trades": len(detail),
            "baseline_winners": (
                baseline_winners
            ),
            "overlay_winners": (
                overlay_winners
            ),
            "baseline_win_rate_percent": (
                baseline_winners
                / len(detail)
                * 100.0
            ),
            "overlay_win_rate_percent": (
                overlay_winners
                / len(detail)
                * 100.0
            ),
            "baseline_profit_factor": (
                baseline_pf
            ),
            "overlay_profit_factor": (
                overlay_pf
            ),
            "baseline_total_pnl_eur": (
                baseline_total_pnl
            ),
            "overlay_total_pnl_eur": (
                overlay_total_pnl
            ),
            "delta_total_pnl_eur": (
                overlay_total_pnl
                - baseline_total_pnl
            ),
            "baseline_total_r": (
                baseline_total_r
            ),
            "overlay_total_r": (
                overlay_total_r
            ),
            "delta_total_r": (
                overlay_total_r
                - baseline_total_r
            ),
            "partial_profit_trades": int(
                detail[
                    "partial_taken"
                ].sum()
            ),
            "no_progress_exits": int(
                (
                    detail[
                        "overlay_exit_reason"
                    ]
                    == "V3KL_NO_PROGRESS_36H"
                ).sum()
            ),
            "small_loss_stop_exits": int(
                (
                    detail[
                        "overlay_exit_reason"
                    ]
                    == "V3KL_SMALL_LOSS_STOP"
                ).sum()
            ),
            "fee_break_even_exits": int(
                (
                    detail[
                        "overlay_exit_reason"
                    ]
                    == "V3KL_FEE_BREAK_EVEN_STOP"
                ).sum()
            ),
            "profit_ratchet_exits": int(
                (
                    detail[
                        "overlay_exit_reason"
                    ]
                    == (
                        "V3KL_70_PERCENT_"
                        "PROFIT_RATCHET_STOP"
                    )
                ).sum()
            ),
        }
    ]
)


DETAIL_OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

detail.to_csv(
    DETAIL_OUTPUT,
    index=False,
)

summary.to_csv(
    SUMMARY_OUTPUT,
    index=False,
)


print()
print("=" * 104)

print(
    f"V3KL FIXED-ENTRY PROFIT RATCHET | "
    f"{WINDOW}"
)

print("=" * 104)

print(
    f"Trades                         : "
    f"{len(detail)}"
)

print(
    f"Baseline winners               : "
    f"{baseline_winners}"
)

print(
    f"V3KL winners                   : "
    f"{overlay_winners}"
)

print(
    f"Baseline total P&L             : "
    f"EUR {baseline_total_pnl:+,.2f}"
)

print(
    f"V3KL total P&L                 : "
    f"EUR {overlay_total_pnl:+,.2f}"
)

print(
    f"Difference                     : "
    f"EUR "
    f"{overlay_total_pnl - baseline_total_pnl:+,.2f}"
)

print(
    f"Baseline total R               : "
    f"{baseline_total_r:+.2f}R"
)

print(
    f"V3KL total R                   : "
    f"{overlay_total_r:+.2f}R"
)

print(
    f"Difference                     : "
    f"{overlay_total_r - baseline_total_r:+.2f}R"
)

print(
    f"Baseline profit factor         : "
    f"{baseline_pf:.2f}"
)

print(
    f"V3KL profit factor             : "
    f"{overlay_pf:.2f}"
)

print("-" * 104)

print(
    f"25% partial-profit trades      : "
    f"{int(detail['partial_taken'].sum())}"
)

print(
    f"36h no-progress exits          : "
    f"{int((detail['overlay_exit_reason'] == 'V3KL_NO_PROGRESS_36H').sum())}"
)

print(
    f"Small-loss stop exits          : "
    f"{int((detail['overlay_exit_reason'] == 'V3KL_SMALL_LOSS_STOP').sum())}"
)

print(
    f"Fee break-even exits           : "
    f"{int((detail['overlay_exit_reason'] == 'V3KL_FEE_BREAK_EVEN_STOP').sum())}"
)

print(
    f"70% profit-ratchet exits       : "
    f"{int((detail['overlay_exit_reason'] == 'V3KL_70_PERCENT_PROFIT_RATCHET_STOP').sum())}"
)

print("-" * 104)

print(
    "Important: entries and sizes are fixed "
    "to the original V3KJ trades."
)

print(
    "Earlier exits do not create replacement "
    "entries in this screening stage."
)

print(
    "New candle-high stop levels become active "
    "from the following 1h candle."
)

print("-" * 104)

print(
    f"Detail report                  : "
    f"{DETAIL_OUTPUT}"
)

print(
    f"Summary report                 : "
    f"{SUMMARY_OUTPUT}"
)

print("=" * 104)
