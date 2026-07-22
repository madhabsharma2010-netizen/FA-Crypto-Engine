from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd

from core.surveillance_v3d import (
    AssetShockDecision,
    MarketState,
    MarketStateDecision,
    ShockLevel,
)
from run_v3d_diagnostics import (
    BACKTEST_END,
    BACKTEST_START,
    SYMBOLS,
    common_index,
    prepare_1h,
)
from strategies.pullback_candidate_v3d2 import (
    evaluate_pullback_candidate,
)


MARKET_STATE_INPUT = Path(
    "reports/v3d_market_state_diagnostics.csv"
)

ASSET_SHOCK_INPUT = Path(
    "reports/v3d_asset_shock_diagnostics.csv"
)

DETAIL_FILE = Path(
    "reports/v3d2_pullback_candidates.csv"
)

SUMMARY_FILE = Path(
    "reports/v3d2_candidate_summary.csv"
)


def market_state_from_row(
    row: pd.Series,
) -> MarketStateDecision:
    return MarketStateDecision(
        state=MarketState(
            str(row["state"])
        ),
        score=float(row["score"]),
        btc_score=float(row["btc_score"]),
        eth_score=float(row["eth_score"]),
        breadth_score=float(
            row["breadth_score"]
        ),
        new_entries_allowed=bool(
            row["new_entries_allowed"]
        ),
        altcoins_allowed=bool(
            row["altcoins_allowed"]
        ),
        risk_multiplier=float(
            row["risk_multiplier"]
        ),
        reasons=(),
    )


def asset_shock_from_row(
    symbol: str,
    row: pd.Series,
) -> AssetShockDecision:
    multiple = row[
        "emergency_atr_multiple"
    ]

    return AssetShockDecision(
        symbol=symbol,
        level=ShockLevel(
            str(row["level"])
        ),
        score=float(row["score"]),
        freeze_new_entries=bool(
            row["freeze_new_entries"]
        ),
        suggested_reduction_percent=float(
            row[
                "suggested_reduction_percent"
            ]
        ),
        emergency_atr_multiple=(
            None
            if pd.isna(multiple)
            else float(multiple)
        ),
        force_exit=bool(
            row["force_exit"]
        ),
        reasons=(),
    )


def main() -> None:
    if not MARKET_STATE_INPUT.exists():
        raise FileNotFoundError(
            "Run run_v3d_diagnostics.py first."
        )

    if not ASSET_SHOCK_INPUT.exists():
        raise FileNotFoundError(
            "Run run_v3d_diagnostics.py first."
        )

    frames = {
        symbol: prepare_1h(symbol)
        for symbol in SYMBOLS
    }

    for frame in frames.values():
        trend_condition = (
            (frame["close"] > frame["EMA200"])
            & (frame["EMA20"] > frame["EMA50"])
            & (frame["EMA50"] > frame["EMA200"])
        )

        frame["TREND_STABLE_4_OF_6"] = (
            trend_condition
            .rolling(6)
            .sum()
            .ge(4)
        )

    decision_times = common_index(
        frames,
        BACKTEST_START,
        BACKTEST_END,
    )

    market_states = (
        pd.read_csv(
            MARKET_STATE_INPUT,
            parse_dates=["decision_time"],
        )
        .set_index("decision_time")
        .sort_index()
        .reindex(
            decision_times,
            method="ffill",
        )
    )

    shocks = pd.read_csv(
        ASSET_SHOCK_INPUT,
        parse_dates=["decision_time"],
    )

    shocks_by_symbol = {}

    for symbol in SYMBOLS:
        shocks_by_symbol[
            symbol
        ] = (
            shocks[
                shocks["symbol"]
                == symbol
            ]
            .set_index("decision_time")
            .sort_index()
            .reindex(
                decision_times,
                method="ffill",
            )
        )

    rows = []
    latest = []

    for position in range(
        1,
        len(decision_times),
    ):
        decision_time = (
            decision_times[position]
        )

        market_state = (
            market_state_from_row(
                market_states.loc[
                    decision_time
                ]
            )
        )

        current_candidates = []

        for symbol in SYMBOLS:
            candle = frames[
                symbol
            ].loc[decision_time]

            previous = frames[
                symbol
            ].loc[
                decision_times[
                    position - 1
                ]
            ]

            shock = asset_shock_from_row(
                symbol,
                shocks_by_symbol[
                    symbol
                ].loc[decision_time],
            )

            candidate = (
                evaluate_pullback_candidate(
                    symbol=symbol,
                    candle=candle,
                    previous=previous,
                    market_state=(
                        market_state
                    ),
                    asset_shock=shock,
                )
            )

            current_candidates.append(
                candidate
            )

            rows.append(
                {
                    "decision_time": (
                        decision_time
                    ),
                    "symbol": symbol,
                    "valid": candidate.valid,
                    "score": candidate.score,
                    "entry_price": (
                        candidate.entry_price
                    ),
                    "stop_price": (
                        candidate.stop_price
                    ),
                    "risk_percent": (
                        candidate.risk_percent
                    ),
                    "projected_target": (
                        candidate.projected_target
                    ),
                    "estimated_reward_percent": (
                        candidate
                        .estimated_reward_percent
                    ),
                    "reward_risk": (
                        candidate.reward_risk
                    ),
                    "market_state": (
                        market_state.state.value
                    ),
                    "asset_shock": (
                        shock.level.value
                    ),
                    "failed_checks": "|".join(
                        candidate.failed_checks
                    ),
                    "reasons": "|".join(
                        candidate.reasons
                    ),
                }
            )

        latest = sorted(
            current_candidates,
            key=lambda item: (
                item.valid,
                item.score,
                item.reward_risk,
            ),
            reverse=True,
        )

    result = pd.DataFrame(rows)

    DETAIL_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        DETAIL_FILE,
        index=False,
    )

    summary_rows = []

    print("=" * 122)
    print(
        "FA CRYPTO ENGINE — V3D2 PULLBACK CANDIDATE CALIBRATION"
    )
    print("=" * 122)
    print(
        f"{'SYMBOL':<10} "
        f"{'VALID':>10} "
        f"{'AVG RISK':>12} "
        f"{'AVG R:R':>12} "
        f"{'TOP REJECTION':>28}"
    )
    print("-" * 122)

    for symbol in SYMBOLS:
        symbol_rows = result[
            result["symbol"]
            == symbol
        ]

        valid_rows = symbol_rows[
            symbol_rows["valid"]
        ]

        rejection_counter = Counter()

        for value in symbol_rows[
            "failed_checks"
        ].fillna(""):
            for reason in str(
                value
            ).split("|"):
                if reason:
                    rejection_counter[
                        reason
                    ] += 1

        top_rejection = (
            rejection_counter
            .most_common(1)
        )

        top_text = (
            top_rejection[0][0]
            if top_rejection
            else "none"
        )

        average_risk = (
            float(
                valid_rows[
                    "risk_percent"
                ].mean()
            )
            if not valid_rows.empty
            else 0.0
        )

        average_rr = (
            float(
                valid_rows[
                    "reward_risk"
                ].mean()
            )
            if not valid_rows.empty
            else 0.0
        )

        summary_rows.append(
            {
                "symbol": symbol,
                "valid_candidates": len(
                    valid_rows
                ),
                "average_risk_percent": (
                    average_risk
                ),
                "average_reward_risk": (
                    average_rr
                ),
                "top_rejection": (
                    top_text
                ),
            }
        )

        print(
            f"{symbol:<10} "
            f"{len(valid_rows):>10} "
            f"{average_risk:>11.2f}% "
            f"{average_rr:>12.2f} "
            f"{top_text:>28}"
        )

    pd.DataFrame(
        summary_rows
    ).to_csv(
        SUMMARY_FILE,
        index=False,
    )

    print("-" * 122)
    print(
        "LATEST CANDIDATE RANKING"
    )
    print("-" * 122)

    for rank, candidate in enumerate(
        latest,
        start=1,
    ):
        status = (
            "VALID"
            if candidate.valid
            else "WAIT"
        )

        failed_text = (
            ",".join(
                candidate.failed_checks[:3]
            )
            if candidate.failed_checks
            else "-"
        )

        print(
            f"{rank}. "
            f"{candidate.symbol:<10} | "
            f"Score {candidate.score:>6.2f} | "
            f"Risk {candidate.risk_percent:>5.2f}% | "
            f"R:R {candidate.reward_risk:>5.2f} | "
            f"{status:<5} | "
            f"Failed: {failed_text}"
        )

    print("=" * 122)
    print(
        f"Detail saved to : {DETAIL_FILE}"
    )
    print(
        f"Summary saved to: {SUMMARY_FILE}"
    )
    print("=" * 122)


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        print()
        print(
            "V3D2 diagnostics stopped manually."
        )

    except Exception as error:
        print()
        print("=" * 122)
        print(
            "V3D2 DIAGNOSTIC ERROR"
        )
        print("=" * 122)
        print(
            f"{type(error).__name__}: "
            f"{error}"
        )
        print("=" * 122)
        raise
