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

    # The V3E6 candidate builder resolves this module-global function
    # at runtime. Replacing it truly blocks LINK and every non-SOL route.
    module._route_quality_allowed = (
        sol_only_quality_allowed
    )

    module.IGNITION_ENABLED = set()
    module.BREAKOUT_ENABLED = {
        "SOLUSDT",
    }


def prepare_frames_once(
    module,
    attempts: int = 4,
):
    original_prepare_frames = (
        module.prepare_frames
    )

    last_error = None

    for attempt in range(
        1,
        attempts + 1,
    ):
        try:
            print(
                f"Downloading historical data once "
                f"(attempt {attempt}/{attempts})..."
            )
            cached_frames = (
                original_prepare_frames()
            )
            print(
                "Historical data cached. "
                "All four policies will reuse it."
            )
            return cached_frames

        except Exception as error:
            last_error = error

            if attempt >= attempts:
                break

            delay = attempt * 5

            print(
                f"Temporary data connection error: "
                f"{type(error).__name__}: {error}"
            )
            print(
                f"Retrying after {delay} seconds..."
            )

            time.sleep(delay)

    raise last_error


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
        f"v3e9b_{period.lower()}_"
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

    cached_frames = prepare_frames_once(
        module
    )

    # Prevent four repeated Binance downloads.
    module.prepare_frames = (
        lambda: cached_frames
    )

    rows = []

    print("=" * 136)
    print(
        "FA CRYPTO ENGINE — V3E9B CORRECTED SOL EXIT MATRIX"
    )
    print("=" * 136)
    print(
        f"Period: {period}"
    )
    print(
        "TRUE SOL-only gate installed. "
        "LINK and all other entry routes are blocked."
    )
    print(
        "V3D3 surveillance, initial stop risk, sizing "
        "and portfolio loss limits remain frozen."
    )

    for index, policy in enumerate(
        POLICIES,
        start=1,
    ):
        print()
        print("#" * 136)
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
        print("#" * 136)

        configure_policy(
            module,
            period,
            policy,
        )

        summary = module.run_backtest()

        rows.append(
            {
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
        )

    result = pd.DataFrame(rows)

    report_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        report_file,
        index=False,
    )

    print()
    print("=" * 136)
    print(
        "V3E9B CORRECTED EXIT-POLICY COMPARISON"
    )
    print("=" * 136)
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
    print("-" * 136)

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

    print("=" * 136)
    print(
        f"Matrix report: {report_file}"
    )
    print("=" * 136)


if __name__ == "__main__":
    try:
        run_matrix(
            module_name=(
                "run_v3e6_selective_development"
            ),
            period="DEVELOPMENT",
            report_file=Path(
                "reports/"
                "v3e9b_development_exit_matrix.csv"
            ),
        )

    except KeyboardInterrupt:
        print()
        print(
            "V3E9B development matrix stopped manually."
        )

    except Exception as error:
        print()
        print("=" * 136)
        print(
            "V3E9B DEVELOPMENT MATRIX ERROR"
        )
        print("=" * 136)
        print(
            f"{type(error).__name__}: {error}"
        )
        print("=" * 136)
        raise
