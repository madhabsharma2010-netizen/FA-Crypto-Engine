from __future__ import annotations

from pathlib import Path
import importlib

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


def configure_module(
    module,
    period: str,
    policy: dict,
) -> None:
    # Profit-entry isolation only.
    # V3D3 surveillance, initial risk, loss limits, deployment,
    # position sizing and hard drawdown rules remain unchanged.
    module.IGNITION_ENABLED = set()
    module.BREAKOUT_ENABLED = {
        "SOLUSDT",
    }

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

    module.TRADE_HISTORY_FILE = Path(
        f"logs/backtests/"
        f"v3e9_{period.lower()}_"
        f"{slug}_trades.csv"
    )

    module.EVENT_HISTORY_FILE = Path(
        f"logs/backtests/"
        f"v3e9_{period.lower()}_"
        f"{slug}_events.csv"
    )

    module.SUMMARY_FILE = Path(
        f"reports/"
        f"v3e9_{period.lower()}_"
        f"{slug}_summary.csv"
    )

    module.SYMBOL_SUMMARY_FILE = Path(
        f"reports/"
        f"v3e9_{period.lower()}_"
        f"{slug}_symbols.csv"
    )


def run_matrix(
    module_name: str,
    period: str,
    report_file: Path,
) -> None:
    module = importlib.import_module(
        module_name
    )

    rows = []

    print("=" * 132)
    print(
        "FA CRYPTO ENGINE — V3E9 SOL EXIT-POLICY MATRIX"
    )
    print("=" * 132)
    print(
        f"Period: {period}"
    )
    print(
        "V3D3 surveillance, initial stop risk, sizing and loss limits are frozen."
    )
    print(
        "LINK disabled. Only SOL Adaptive Breakout is tested."
    )

    for index, policy in enumerate(
        POLICIES,
        start=1,
    ):
        print()
        print("#" * 132)
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
        print("#" * 132)

        configure_module(
            module,
            period,
            policy,
        )

        summary = module.run_backtest()

        rows.append(
            {
                "period": period,
                "policy": (
                    policy["name"]
                ),
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
    print("=" * 132)
    print(
        "V3E9 EXIT-POLICY COMPARISON"
    )
    print("=" * 132)
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
    print("-" * 132)

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

    print("=" * 132)
    print(
        f"Matrix report: {report_file}"
    )
    print("=" * 132)


if __name__ == "__main__":
    try:
        run_matrix(
            module_name=(
                "run_v3e6_selective_validation"
            ),
            period="VALIDATION",
            report_file=Path(
                "reports/"
                "v3e9_validation_exit_matrix.csv"
            ),
        )

    except KeyboardInterrupt:
        print()
        print(
            "V3E9 validation matrix stopped manually."
        )

    except Exception as error:
        print()
        print("=" * 132)
        print(
            "V3E9 VALIDATION MATRIX ERROR"
        )
        print("=" * 132)
        print(
            f"{type(error).__name__}: "
            f"{error}"
        )
        print("=" * 132)
        raise
