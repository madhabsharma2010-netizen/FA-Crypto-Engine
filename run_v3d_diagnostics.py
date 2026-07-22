from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd
from binance.client import Client

from core.indicators import (
    calculate_adx,
    calculate_atr,
    calculate_ema,
    calculate_rsi,
)
from core.surveillance_v3d import (
    AssetShockDecision,
    MarketState,
    ShockLevel,
    evaluate_asset_shock,
    evaluate_market_shock,
    evaluate_market_state,
)
from market.historical_data import get_historical_candles
from strategies.pullback_candidate_v3d import (
    evaluate_pullback_candidate,
)


SYMBOLS = (
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "LINKUSDT",
    "DOGEUSDT",
)

BACKTEST_START = pd.Timestamp("2026-05-01 00:00:00")
BACKTEST_END = pd.Timestamp("2026-07-20 00:00:00")

ASSET_SHOCK_FILE = Path(
    "reports/v3d_asset_shock_diagnostics.csv"
)
MARKET_SHOCK_FILE = Path(
    "reports/v3d_market_shock_diagnostics.csv"
)
MARKET_STATE_FILE = Path(
    "reports/v3d_market_state_diagnostics.csv"
)
PULLBACK_FILE = Path(
    "reports/v3d_pullback_candidates.csv"
)
SUMMARY_FILE = Path(
    "reports/v3d_diagnostic_summary.csv"
)


INTERVAL_CONFIG = {
    "15m": {
        "binance": Client.KLINE_INTERVAL_15MINUTE,
        "delta": pd.Timedelta(minutes=15),
        "warmup_days": 12,
        "momentum_period": 16,
        "slope_period": 16,
    },
    "1h": {
        "binance": Client.KLINE_INTERVAL_1HOUR,
        "delta": pd.Timedelta(hours=1),
        "warmup_days": 22,
        "momentum_period": 24,
        "slope_period": 12,
    },
    "2h": {
        "binance": Client.KLINE_INTERVAL_2HOUR,
        "delta": pd.Timedelta(hours=2),
        "warmup_days": 42,
        "momentum_period": 12,
        "slope_period": 6,
    },
    "4h": {
        "binance": Client.KLINE_INTERVAL_4HOUR,
        "delta": pd.Timedelta(hours=4),
        "warmup_days": 75,
        "momentum_period": 6,
        "slope_period": 3,
    },
    "1d": {
        "binance": Client.KLINE_INTERVAL_1DAY,
        "delta": pd.Timedelta(days=1),
        "warmup_days": 320,
        "momentum_period": 7,
        "slope_period": 5,
    },
}


def load_frame(
    symbol: str,
    timeframe: str,
) -> pd.DataFrame:
    config = INTERVAL_CONFIG[timeframe]

    warmup_start = (
        BACKTEST_START
        - pd.Timedelta(
            days=int(config["warmup_days"])
        )
    )

    data = get_historical_candles(
        symbol=symbol,
        interval=config["binance"],
        limit=1000,
        start_time=warmup_start.strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        end_time=BACKTEST_END.strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        drop_incomplete=True,
    )

    if data.empty:
        raise ValueError(
            f"No data for {symbol} {timeframe}."
        )

    for column in (
        "open",
        "high",
        "low",
        "close",
        "volume",
    ):
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    data["EMA20"] = calculate_ema(data, 20)
    data["EMA50"] = calculate_ema(data, 50)
    data["EMA200"] = calculate_ema(data, 200)
    data["RSI14"] = calculate_rsi(data, 14)
    data["ADX14"] = calculate_adx(data, 14)
    data["ATR14"] = calculate_atr(data, 14)

    data["VolumeSMA20"] = (
        data["volume"]
        .rolling(20)
        .mean()
    )

    data["MOMENTUM"] = (
        data["close"]
        .pct_change(
            int(config["momentum_period"])
        )
        * 100
    )

    data["EMA50_SLOPE"] = (
        data["EMA50"]
        .pct_change(
            int(config["slope_period"])
        )
        * 100
    )

    data["completion_time"] = (
        pd.to_datetime(data["open_time"])
        + config["delta"]
    )

    return (
        data
        .dropna()
        .set_index("completion_time")
        .sort_index()
    )


def prepare_15m(symbol: str) -> pd.DataFrame:
    data = load_frame(symbol, "15m").copy()

    data["RETURN_15M"] = (
        data["close"].pct_change(1) * 100
    )
    data["RETURN_30M"] = (
        data["close"].pct_change(2) * 100
    )
    data["RETURN_60M"] = (
        data["close"].pct_change(4) * 100
    )
    data["ATR_PERCENT"] = (
        data["ATR14"]
        / data["close"]
        * 100
    )
    data["ATR_MEDIAN_48"] = (
        data["ATR_PERCENT"]
        .rolling(48)
        .median()
    )
    data["ATR_EXPANSION"] = (
        data["ATR_PERCENT"]
        / data["ATR_MEDIAN_48"]
    )
    data["SUPPORT_4H"] = (
        data["low"]
        .shift(1)
        .rolling(16)
        .min()
    )

    return data.dropna()


def prepare_1h(symbol: str) -> pd.DataFrame:
    data = load_frame(symbol, "1h").copy()

    data["MOMENTUM_24H"] = (
        data["close"].pct_change(24) * 100
    )
    data["MOMENTUM_72H"] = (
        data["close"].pct_change(72) * 100
    )
    data["EMA50_SLOPE_12H"] = (
        data["EMA50"].pct_change(12) * 100
    )
    data["VOLUME_RATIO"] = (
        data["volume"]
        / data["VolumeSMA20"]
    )

    trend_condition = (
        (data["close"] > data["EMA200"])
        & (data["EMA20"] > data["EMA50"])
        & (data["EMA50"] > data["EMA200"])
    )

    data["TREND_STABLE_6"] = (
        trend_condition
        .rolling(6)
        .sum()
        .eq(6)
    )
    data["RECENT_HIGH_72H"] = (
        data["high"]
        .shift(1)
        .rolling(72)
        .max()
    )
    data["RECENT_LOW_12H"] = (
        data["low"]
        .shift(1)
        .rolling(12)
        .min()
    )

    return data.dropna()


def common_index(
    frames: dict[str, pd.DataFrame],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DatetimeIndex:
    result = None

    for frame in frames.values():
        if result is None:
            result = frame.index
        else:
            result = result.intersection(
                frame.index
            )

    if result is None or len(result) == 0:
        raise ValueError(
            "No common timestamps found."
        )

    return result[
        (result >= start)
        & (result <= end)
    ].sort_values()


def align_frame(
    frame: pd.DataFrame,
    decision_times: pd.DatetimeIndex,
) -> pd.DataFrame:
    aligned = frame.reindex(
        decision_times,
        method="ffill",
    )

    if aligned.isna().all(axis=1).any():
        raise ValueError(
            "Higher-timeframe warmup is insufficient."
        )

    return aligned


def breadth_score_at(
    data_by_symbol: dict[str, pd.DataFrame],
    decision_time: pd.Timestamp,
) -> float:
    points = 0

    for symbol in SYMBOLS:
        candle = data_by_symbol[
            symbol
        ].loc[decision_time]

        points += int(
            float(candle["close"])
            > float(candle["EMA200"])
        )
        points += int(
            float(candle["EMA20"])
            > float(candle["EMA50"])
        )
        points += int(
            float(candle["MOMENTUM_24H"])
            > 0
        )

    return (
        points
        / (len(SYMBOLS) * 3)
        * 100
    )


def shock_from_row(
    symbol: str,
    row: pd.Series,
) -> AssetShockDecision:
    level = ShockLevel(
        str(row["level"])
    )

    return AssetShockDecision(
        symbol=symbol,
        level=level,
        score=float(row["score"]),
        freeze_new_entries=bool(
            row["freeze_new_entries"]
        ),
        suggested_reduction_percent=float(
            row["suggested_reduction_percent"]
        ),
        emergency_atr_multiple=(
            None
            if pd.isna(
                row["emergency_atr_multiple"]
            )
            else float(
                row["emergency_atr_multiple"]
            )
        ),
        force_exit=bool(
            row["force_exit"]
        ),
        reasons=(),
    )


def main() -> None:
    print("=" * 122)
    print(
        "FA CRYPTO ENGINE — V3D MULTI-TIMEFRAME SURVEILLANCE DIAGNOSTICS"
    )
    print("=" * 122)

    frames_15m = {
        symbol: prepare_15m(symbol)
        for symbol in SYMBOLS
    }
    frames_1h = {
        symbol: prepare_1h(symbol)
        for symbol in SYMBOLS
    }

    btc_2h = load_frame("BTCUSDT", "2h")
    btc_4h = load_frame("BTCUSDT", "4h")
    btc_1d = load_frame("BTCUSDT", "1d")
    eth_2h = load_frame("ETHUSDT", "2h")
    eth_4h = load_frame("ETHUSDT", "4h")

    common_15m = common_index(
        frames_15m,
        BACKTEST_START,
        BACKTEST_END,
    )

    market_return_60m = pd.DataFrame(
        {
            symbol: frames_15m[
                symbol
            ]["RETURN_60M"]
            for symbol in SYMBOLS
        }
    ).median(axis=1)

    for symbol in SYMBOLS:
        frames_15m[
            symbol
        ]["MARKET_RELATIVE_60M"] = (
            frames_15m[
                symbol
            ]["RETURN_60M"]
            - market_return_60m
        )

    asset_rows = []
    market_shock_rows = []
    latest_asset_decisions = {}
    latest_market_shock = None

    for decision_time in common_15m:
        decisions = []
        returns_15m = []
        returns_60m = []

        for symbol in SYMBOLS:
            candle = frames_15m[
                symbol
            ].loc[decision_time]

            decision = evaluate_asset_shock(
                symbol,
                candle,
            )

            decisions.append(decision)
            latest_asset_decisions[
                symbol
            ] = decision

            returns_15m.append(
                float(candle["RETURN_15M"])
            )
            returns_60m.append(
                float(candle["RETURN_60M"])
            )

            asset_rows.append(
                {
                    "decision_time": decision_time,
                    "symbol": symbol,
                    "level": decision.level.value,
                    "score": decision.score,
                    "freeze_new_entries": (
                        decision.freeze_new_entries
                    ),
                    "suggested_reduction_percent": (
                        decision.suggested_reduction_percent
                    ),
                    "emergency_atr_multiple": (
                        decision.emergency_atr_multiple
                    ),
                    "force_exit": decision.force_exit,
                    "reasons": " | ".join(
                        decision.reasons
                    ),
                }
            )

        market_decision = evaluate_market_shock(
            decisions,
            median_return_15m=float(
                pd.Series(returns_15m).median()
            ),
            median_return_60m=float(
                pd.Series(returns_60m).median()
            ),
        )

        latest_market_shock = (
            market_decision
        )

        market_shock_rows.append(
            {
                "decision_time": decision_time,
                "level": (
                    market_decision.level.value
                ),
                "score": market_decision.score,
                "freeze_all_entries": (
                    market_decision.freeze_all_entries
                ),
                "portfolio_reduction_percent": (
                    market_decision
                    .suggested_portfolio_reduction_percent
                ),
                "force_cash_mode": (
                    market_decision.force_cash_mode
                ),
                "reasons": " | ".join(
                    market_decision.reasons
                ),
            }
        )

    asset_result = pd.DataFrame(asset_rows)
    market_shock_result = pd.DataFrame(
        market_shock_rows
    )

    common_1h = common_index(
        frames_1h,
        BACKTEST_START,
        BACKTEST_END,
    )

    aligned_btc_2h = align_frame(
        btc_2h,
        common_1h,
    )
    aligned_btc_4h = align_frame(
        btc_4h,
        common_1h,
    )
    aligned_btc_1d = align_frame(
        btc_1d,
        common_1h,
    )
    aligned_eth_2h = align_frame(
        eth_2h,
        common_1h,
    )
    aligned_eth_4h = align_frame(
        eth_4h,
        common_1h,
    )

    market_shock_frame = (
        market_shock_result
        .set_index("decision_time")
        .sort_index()
    )
    aligned_market_shock = (
        market_shock_frame
        .reindex(
            common_1h,
            method="ffill",
        )
    )

    aligned_asset_shock = {}

    for symbol in SYMBOLS:
        aligned_asset_shock[
            symbol
        ] = (
            asset_result[
                asset_result["symbol"]
                == symbol
            ]
            .set_index("decision_time")
            .sort_index()
            .reindex(
                common_1h,
                method="ffill",
            )
        )

    state_rows = []
    candidate_rows = []
    latest_state = None
    latest_candidates = []

    previous_by_symbol = {
        symbol: None
        for symbol in SYMBOLS
    }

    for decision_time in common_1h:
        shock_level = ShockLevel(
            str(
                aligned_market_shock.loc[
                    decision_time,
                    "level",
                ]
            )
        )

        breadth = breadth_score_at(
            frames_1h,
            decision_time,
        )

        market_state = evaluate_market_state(
            btc_1h=frames_1h[
                "BTCUSDT"
            ].loc[decision_time],
            btc_2h=aligned_btc_2h.loc[
                decision_time
            ],
            btc_4h=aligned_btc_4h.loc[
                decision_time
            ],
            btc_1d=aligned_btc_1d.loc[
                decision_time
            ],
            eth_1h=frames_1h[
                "ETHUSDT"
            ].loc[decision_time],
            eth_2h=aligned_eth_2h.loc[
                decision_time
            ],
            eth_4h=aligned_eth_4h.loc[
                decision_time
            ],
            breadth_score=breadth,
            market_shock=shock_level,
        )

        latest_state = market_state

        state_rows.append(
            {
                "decision_time": decision_time,
                "state": (
                    market_state.state.value
                ),
                "score": market_state.score,
                "btc_score": (
                    market_state.btc_score
                ),
                "eth_score": (
                    market_state.eth_score
                ),
                "breadth_score": (
                    market_state.breadth_score
                ),
                "new_entries_allowed": (
                    market_state
                    .new_entries_allowed
                ),
                "altcoins_allowed": (
                    market_state
                    .altcoins_allowed
                ),
                "risk_multiplier": (
                    market_state
                    .risk_multiplier
                ),
                "reasons": " | ".join(
                    market_state.reasons
                ),
            }
        )

        current_candidates = []

        for symbol in SYMBOLS:
            candle = frames_1h[
                symbol
            ].loc[decision_time]
            previous = (
                previous_by_symbol[symbol]
            )

            if previous is None:
                previous_by_symbol[
                    symbol
                ] = candle
                continue

            shock = shock_from_row(
                symbol,
                aligned_asset_shock[
                    symbol
                ].loc[decision_time],
            )

            candidate = evaluate_pullback_candidate(
                symbol=symbol,
                candle=candle,
                previous=previous,
                market_state=market_state,
                asset_shock=shock,
            )

            current_candidates.append(
                candidate
            )

            candidate_rows.append(
                {
                    "decision_time": decision_time,
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
                    "reasons": " | ".join(
                        candidate.reasons
                    ),
                }
            )

            previous_by_symbol[
                symbol
            ] = candle

        latest_candidates = sorted(
            current_candidates,
            key=lambda item: (
                item.valid,
                item.score,
            ),
            reverse=True,
        )

    state_result = pd.DataFrame(
        state_rows
    )
    candidate_result = pd.DataFrame(
        candidate_rows
    )

    for file_path in (
        ASSET_SHOCK_FILE,
        MARKET_SHOCK_FILE,
        MARKET_STATE_FILE,
        PULLBACK_FILE,
        SUMMARY_FILE,
    ):
        file_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    asset_result.to_csv(
        ASSET_SHOCK_FILE,
        index=False,
    )
    market_shock_result.to_csv(
        MARKET_SHOCK_FILE,
        index=False,
    )
    state_result.to_csv(
        MARKET_STATE_FILE,
        index=False,
    )
    candidate_result.to_csv(
        PULLBACK_FILE,
        index=False,
    )

    summary_rows = []

    for symbol in SYMBOLS:
        symbol_shocks = asset_result[
            asset_result["symbol"]
            == symbol
        ]
        level_counts = Counter(
            symbol_shocks["level"]
        )

        valid_candidates = candidate_result[
            (
                candidate_result["symbol"]
                == symbol
            )
            & candidate_result["valid"]
        ]

        summary_rows.append(
            {
                "symbol": symbol,
                "15m_warning_count": (
                    level_counts[
                        ShockLevel.WARNING.value
                    ]
                ),
                "15m_shock_count": (
                    level_counts[
                        ShockLevel.SHOCK.value
                    ]
                ),
                "15m_severe_count": (
                    level_counts[
                        ShockLevel.SEVERE.value
                    ]
                ),
                "valid_pullback_candidates": (
                    len(valid_candidates)
                ),
            }
        )

    pd.DataFrame(
        summary_rows
    ).to_csv(
        SUMMARY_FILE,
        index=False,
    )

    print()
    print("-" * 122)
    print(
        "15M INDIVIDUAL-ASSET EMERGENCY SURVEILLANCE"
    )
    print("-" * 122)
    print(
        f"{'SYMBOL':<10} "
        f"{'WARNING':>10} "
        f"{'SHOCK':>10} "
        f"{'SEVERE':>10} "
        f"{'LATEST':>12} "
        f"{'ACTION':>26}"
    )
    print("-" * 122)

    for row in summary_rows:
        symbol = str(row["symbol"])
        latest = latest_asset_decisions[
            symbol
        ]

        if latest.force_exit:
            action = "EXIT POSITION"
        elif (
            latest
            .suggested_reduction_percent
            > 0
        ):
            action = (
                "REDUCE "
                f"{latest.suggested_reduction_percent:.0f}%"
            )
        elif latest.freeze_new_entries:
            action = "FREEZE + TIGHTEN"
        else:
            action = "NORMAL MANAGEMENT"

        print(
            f"{symbol:<10} "
            f"{int(row['15m_warning_count']):>10} "
            f"{int(row['15m_shock_count']):>10} "
            f"{int(row['15m_severe_count']):>10} "
            f"{latest.level.value:>12} "
            f"{action:>26}"
        )

    print("-" * 122)
    print(
        "15M MARKET-WIDE SHOCK SUMMARY"
    )
    print("-" * 122)

    market_counts = Counter(
        market_shock_result["level"]
    )

    for level in (
        ShockLevel.NORMAL,
        ShockLevel.WARNING,
        ShockLevel.SHOCK,
        ShockLevel.SEVERE,
    ):
        count = market_counts[level.value]
        percent = (
            count
            / len(market_shock_result)
            * 100
        )

        print(
            f"{level.value:<10}: "
            f"{count:>6} "
            f"({percent:>6.2f}%)"
        )

    if latest_market_shock is not None:
        print(
            "Latest market shock       : "
            f"{latest_market_shock.level.value}"
        )

    print("-" * 122)
    print(
        "MULTI-TIMEFRAME MARKET STATE"
    )
    print("-" * 122)

    state_counts = Counter(
        state_result["state"]
    )

    for state in MarketState:
        count = state_counts[state.value]
        percent = (
            count
            / len(state_result)
            * 100
        )

        print(
            f"{state.value:<20}: "
            f"{count:>6} "
            f"({percent:>6.2f}%)"
        )

    if latest_state is not None:
        print(
            "Latest market state       : "
            f"{latest_state.state.value}"
        )
        print(
            "Latest state score        : "
            f"{latest_state.score:.2f}"
        )
        print(
            "Latest risk multiplier    : "
            f"{latest_state.risk_multiplier:.2f}"
        )

    print("-" * 122)
    print(
        "1H PULLBACK / RECLAIM CANDIDATES"
    )
    print("-" * 122)
    print(
        f"{'SYMBOL':<10} "
        f"{'VALID SETUPS':>14}"
    )
    print("-" * 122)

    for row in summary_rows:
        print(
            f"{str(row['symbol']):<10} "
            f"{int(row['valid_pullback_candidates']):>14}"
        )

    print("-" * 122)
    print(
        "LATEST CANDIDATE RANKING"
    )
    print("-" * 122)

    for rank, candidate in enumerate(
        latest_candidates,
        start=1,
    ):
        status = (
            "VALID"
            if candidate.valid
            else "WAIT"
        )

        print(
            f"{rank}. "
            f"{candidate.symbol:<10} | "
            f"Score {candidate.score:>6.2f} | "
            f"R:R {candidate.reward_risk:>5.2f} | "
            f"Risk {candidate.risk_percent:>5.2f}% | "
            f"{status}"
        )

    print("=" * 122)
    print(
        f"Asset shock detail : {ASSET_SHOCK_FILE}"
    )
    print(
        f"Market shock detail: {MARKET_SHOCK_FILE}"
    )
    print(
        f"Market states      : {MARKET_STATE_FILE}"
    )
    print(
        f"Pullback candidates: {PULLBACK_FILE}"
    )
    print(
        f"Summary            : {SUMMARY_FILE}"
    )
    print("=" * 122)


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        print()
        print(
            "V3D diagnostics stopped manually."
        )

    except Exception as error:
        print()
        print("=" * 122)
        print(
            "V3D DIAGNOSTIC ERROR"
        )
        print("=" * 122)
        print(
            f"{type(error).__name__}: "
            f"{error}"
        )
        print("=" * 122)
        raise
