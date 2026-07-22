from __future__ import annotations

from pathlib import Path
import importlib
import time

import pandas as pd


POLICIES = [
    {
        "name": "BASELINE_BE_1R_PARTIAL_2R",
        "breakeven_trigger_r": 1.00,
        "partial_profit_r": 2.00,
        "partial_sell_percent": 25.0,
    },
    {
        "name": "DELAY_BE_1_5R_PARTIAL_2R",
        "breakeven_trigger_r": 1.50,
        "partial_profit_r": 2.00,
        "partial_sell_percent": 25.0,
    },
    {
        "name": "BANK_20_AT_1R_BE_1_5R",
        "breakeven_trigger_r": 1.50,
        "partial_profit_r": 1.00,
        "partial_sell_percent": 20.0,
    },
    {
        "name": "BANK_20_AT_1_5R_BE_1_5R",
        "breakeven_trigger_r": 1.50,
        "partial_profit_r": 1.50,
        "partial_sell_percent": 20.0,
    },
]

MULTI_TIMEFRAME_KEYS = [
    ("BTCUSDT", "2h"),
    ("BTCUSDT", "4h"),
    ("BTCUSDT", "1d"),
    ("ETHUSDT", "2h"),
    ("ETHUSDT", "4h"),
]


def retry_call(
    label: str,
    function,
    attempts: int = 5,
):
    last_error = None

    for attempt in range(
        1,
        attempts + 1,
    ):
        try:
            print(
                f"{label} "
                f"(attempt {attempt}/{attempts})..."
            )
            return function()

        except Exception as error:
            last_error = error

            if attempt >= attempts:
                break

            delay = attempt * 5

            print(
                f"Temporary connection error: "
                f"{type(error).__name__}: {error}"
            )
            print(
                f"Retrying after {delay} seconds..."
            )
            time.sleep(delay)

    raise last_error


def install_true_sol_only_gate(module) -> None:
    def sol_only_quality_allowed(
        symbol,
        route,
        signal,
        market_state,
        relative_strength,
    ):
        if not signal.valid:
            return False

        bullish_state = (
            market_state.state
            in {
                module.MarketState.STRONG_BULL,
                module.MarketState.BULL,
            }
        )

        return (
            symbol == "SOLUSDT"
            and route == "ADAPTIVE_BREAKOUT"
            and signal.score >= 99.0
            and relative_strength >= 66.0
            and bullish_state
        )

    module._route_quality_allowed = (
        sol_only_quality_allowed
    )

    module.IGNITION_ENABLED = set()
    module.BREAKOUT_ENABLED = {
        "SOLUSDT",
    }


def build_complete_offline_cache(module):
    original_prepare_frames = (
        module.prepare_frames
    )

    original_load_frame = (
        module.load_frame
    )

    cached_frames = retry_call(
        "Downloading 15M/1H historical data once",
        original_prepare_frames,
    )

    multi_timeframe_cache = {}

    for symbol, interval in MULTI_TIMEFRAME_KEYS:
        key = (
            symbol,
            interval,
        )

        multi_timeframe_cache[key] = retry_call(
            (
                "Downloading once: "
                f"{symbol} {interval}"
            ),
            lambda s=symbol, i=interval: (
                original_load_frame(
                    s,
                    i,
                ).copy()
            ),
        )

    def offline_load_frame(
        symbol: str,
        interval: str,
    ):
        key = (
            str(symbol).upper(),
            str(interval).lower(),
        )

        if key not in multi_timeframe_cache:
            raise RuntimeError(
                "Unexpected uncached timeframe request: "
                f"{key}. Matrix stopped to prevent "
                "silent internet access."
            )

        return multi_timeframe_cache[
            key
        ].copy()

    module.prepare_frames = (
        lambda: cached_frames
    )

    module.load_frame = (
        offline_load_frame
    )

    print(
        "COMPLETE OFFLINE CACHE READY."
    )
    print(
        "All four policies will run without further Binance requests."
    )

    return cached_frames, multi_timeframe_cache


def configure_policy(
    module,
    period: str,
    policy: dict,
) -> None:
    module.BREAKEVEN_TRIGGER_R = float(
        policy[
            "breakeven_trigger_r"
        ]
    )

    module.PARTIAL_PROFIT_R = float(
        policy[
            "partial_profit_r"
        ]
    )

    module.PARTIAL_SELL_PERCENT = float(
        policy[
            "partial_sell_percent"
        ]
    )

    slug = (
        policy["name"]
        .lower()
        .replace(".", "_")
    )

    prefix = (
        f"v3e9c_{period.lower()}_"
        f"{slug}"
    )

    module.TRADE_HISTORY_FILE = Path(
        f"logs/backtests/{prefix}_trades.csv"
    )

    module.EVENT_HISTORY_FILE = Path(
        f"logs/backtests/{prefix}_events.csv"
    )

    module.SUMMARY_FILE = Path(
        f"reports/{prefix}_summary.csv"
    )

    module.SYMBOL_SUMMARY_FILE = Path(
        f"reports/{prefix}_symbols.csv"
    )


def normalized_row(
    period: str,
    policy: dict,
    summary: dict,
) -> dict:
    return {
        "period": period,
        "policy": policy["name"],
        "breakeven_trigger_r": (
            policy[
                "breakeven_trigger_r"
            ]
        ),
        "partial_profit_r": (
            policy[
                "partial_profit_r"
            ]
        ),
        "partial_sell_percent": (
            policy[
                "partial_sell_percent"
            ]
        ),
        **summary,
    }


def print_comparison(
    result: pd.DataFrame,
    report_file: Path,
) -> None:
    print()
    print("=" * 138)
    print(
        "V3E9C OFFLINE SOL EXIT-POLICY COMPARISON"
    )
    print("=" * 138)
    print(
        f"{'POLICY':<34} "
        f"{'TRADES':>7} "
        f"{'WIN%':>8} "
        f"{'PF':>7} "
        f"{'P&L':>11} "
        f"{'ROI':>8} "
        f"{'DD':>8} "
        f"{'FEES':>10} "
        f"{'PARTIALS':>9} "
        f"{'HARD LOCK':>10}"
    )
    print("-" * 138)

    for row in result.to_dict(
        orient="records"
    ):
        print(
            f"{str(row['policy']):<34} "
            f"{int(row['completed_trades']):>7} "
            f"{float(row['win_rate']):>7.2f}% "
            f"{float(row['profit_factor']):>7.2f} "
            f"€{float(row['total_profit']):>9.2f} "
            f"{float(row['roi_percent']):>7.2f}% "
            f"{float(row['maximum_drawdown']):>7.2f}% "
            f"€{float(row['total_fees']):>8.2f} "
            f"{int(row['partial_profit_events']):>9} "
            f"{int(row['hard_lock_hits']):>10}"
        )

    print("=" * 138)
    print(
        f"Matrix report: {report_file}"
    )
    print("=" * 138)


def run_matrix(
    module_name: str,
    period: str,
    report_file: Path,
) -> None:
    module = importlib.import_module(
        module_name
    )

    install_true_sol_only_gate(
        module
    )

    build_complete_offline_cache(
        module
    )

    report_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_file = report_file.with_name(
        report_file.stem
        + "_checkpoint.csv"
    )

    rows = []

    if checkpoint_file.exists():
        existing = pd.read_csv(
            checkpoint_file
        )

        completed = set(
            existing.get(
                "policy",
                pd.Series(
                    dtype=str,
                ),
            ).astype(str)
        )

        rows = existing.to_dict(
            orient="records"
        )

        if completed:
            print(
                "Checkpoint found. Completed policies: "
                + ", ".join(
                    sorted(completed)
                )
            )

    else:
        completed = set()

    print("=" * 138)
    print(
        "FA CRYPTO ENGINE — V3E9C TRUE SOL-ONLY OFFLINE MATRIX"
    )
    print("=" * 138)
    print(
        f"Period: {period}"
    )
    print(
        "Only SOLUSDT Adaptive Breakout can enter."
    )
    print(
        "V3D3 surveillance, initial stop, sizing "
        "and portfolio risk controls remain frozen."
    )

    for index, policy in enumerate(
        POLICIES,
        start=1,
    ):
        if policy["name"] in completed:
            print(
                f"Skipping completed policy "
                f"{index}/4: {policy['name']}"
            )
            continue

        print()
        print("#" * 138)
        print(
            f"POLICY {index}/{len(POLICIES)}: "
            f"{policy['name']}"
        )
        print(
            f"Breakeven trigger: "
            f"{policy['breakeven_trigger_r']:.2f}R | "
            f"Partial: "
            f"{policy['partial_sell_percent']:.0f}% "
            f"at {policy['partial_profit_r']:.2f}R"
        )
        print("#" * 138)

        configure_policy(
            module,
            period,
            policy,
        )

        summary = module.run_backtest()

        rows.append(
            normalized_row(
                period,
                policy,
                summary,
            )
        )

        checkpoint = pd.DataFrame(
            rows
        )

        checkpoint.to_csv(
            checkpoint_file,
            index=False,
        )

        completed.add(
            policy["name"]
        )

    result = pd.DataFrame(
        rows
    )

    policy_order = {
        policy["name"]: index
        for index, policy in enumerate(
            POLICIES
        )
    }

    result[
        "_policy_order"
    ] = result[
        "policy"
    ].map(
        policy_order
    )

    result = (
        result.sort_values(
            "_policy_order"
        )
        .drop(
            columns=[
                "_policy_order",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    result.to_csv(
        report_file,
        index=False,
    )

    print_comparison(
        result,
        report_file,
    )


if __name__ == "__main__":
    try:
        run_matrix(
            module_name=(
                "run_v3e6_selective_validation"
            ),
            period="VALIDATION",
            report_file=Path(
                "reports/"
                "v3e9c_validation_exit_matrix.csv"
            ),
        )

    except KeyboardInterrupt:
        print()
        print(
            "V3E9C validation matrix stopped manually."
        )

    except Exception as error:
        print()
        print("=" * 138)
        print(
            "V3E9C VALIDATION MATRIX ERROR"
        )
        print("=" * 138)
        print(
            f"{type(error).__name__}: {error}"
        )
        print("=" * 138)
        raise
