from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import run_v3d_diagnostics as diagnostics
from config.risk_settings import RISK_SETTINGS
from core.surveillance_v3d import (
    MarketStateDecision,
    ShockLevel,
    timeframe_trend_score,
)
from core.v3km_opportunity import evaluate_opportunity
from run_v3d3_portfolio_backtest import (
    build_market_states,
    build_shock_tables,
)


WINDOWS = {
    "2022": (
        pd.Timestamp("2022-01-01 00:00:00"),
        pd.Timestamp("2022-12-31 23:59:59"),
    ),
    "2023": (
        pd.Timestamp("2023-01-01 00:00:00"),
        pd.Timestamp("2023-12-31 23:59:59"),
    ),
    "2024": (
        pd.Timestamp("2024-01-01 00:00:00"),
        pd.Timestamp("2024-12-31 23:59:59"),
    ),
    "2025H1": (
        pd.Timestamp("2025-01-01 00:00:00"),
        pd.Timestamp("2025-06-30 23:59:59"),
    ),
}

HORIZON_BARS = {
    "15m": 1,
    "1h": 4,
    "4h": 16,
    "12h": 48,
    "24h": 96,
}

BAND_ORDER = {
    "CASH": 0,
    "PROBE": 1,
    "BUILD": 2,
    "FULL_PACE": 3,
}

REPORT_DIR = Path("reports")


def configure_window(
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> None:
    """
    The existing diagnostic loader reads these module globals at call time.
    Changing them here avoids modifying any frozen V3D source file.
    """
    diagnostics.BACKTEST_START = start
    diagnostics.BACKTEST_END = end


def prepare_frames(
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[
    dict[str, pd.DataFrame],
    dict[str, pd.DataFrame],
    dict[str, pd.DataFrame],
    dict[str, pd.DataFrame],
    pd.DatetimeIndex,
    pd.DatetimeIndex,
]:
    configure_window(start, end)

    frames_15m = {
        symbol: diagnostics.prepare_15m(symbol)
        for symbol in diagnostics.SYMBOLS
    }

    frames_1h = {
        symbol: diagnostics.prepare_1h(symbol)
        for symbol in diagnostics.SYMBOLS
    }

    frames_2h = {
        symbol: diagnostics.load_frame(symbol, "2h")
        for symbol in diagnostics.SYMBOLS
    }

    frames_4h = {
        symbol: diagnostics.load_frame(symbol, "4h")
        for symbol in diagnostics.SYMBOLS
    }

    common_15m = diagnostics.common_index(
        frames_15m,
        start,
        end,
    )

    common_1h = diagnostics.common_index(
        frames_1h,
        start,
        end,
    )

    market_return_60m = pd.DataFrame(
        {
            symbol: frames_15m[symbol]["RETURN_60M"]
            for symbol in diagnostics.SYMBOLS
        }
    ).median(axis=1)

    for symbol in diagnostics.SYMBOLS:
        frames_15m[symbol]["MARKET_RELATIVE_60M"] = (
            frames_15m[symbol]["RETURN_60M"]
            - market_return_60m
        )

    return (
        frames_15m,
        frames_1h,
        frames_2h,
        frames_4h,
        common_15m,
        common_1h,
    )


def net_long_return_percent(
    entry_open: float,
    exit_close: float,
) -> float:
    """
    Apply estimated slippage and fee on both entry and exit.

    Settings use percentage units:
    0.10 means 0.10%, not 10%.
    """
    fee_rate = (
        float(RISK_SETTINGS.trading_fee_percent)
        / 100.0
    )
    slippage_rate = (
        float(RISK_SETTINGS.estimated_slippage_percent)
        / 100.0
    )

    executed_entry = (
        float(entry_open)
        * (1.0 + slippage_rate)
    )
    total_entry_cost = (
        executed_entry
        * (1.0 + fee_rate)
    )

    executed_exit = (
        float(exit_close)
        * (1.0 - slippage_rate)
    )
    net_exit_proceeds = (
        executed_exit
        * (1.0 - fee_rate)
    )

    return (
        (net_exit_proceeds / total_entry_cost)
        - 1.0
    ) * 100.0


def build_market_state_series(
    frames_1h: dict[str, pd.DataFrame],
    common_1h: pd.DatetimeIndex,
    common_15m: pd.DatetimeIndex,
    market_shocks,
) -> pd.Series:
    market_states = build_market_states(
        frames_1h,
        common_1h,
        market_shocks,
    )

    hourly_series = pd.Series(
        market_states,
        dtype="object",
    ).sort_index()

    return hourly_series.reindex(
        common_15m,
        method="ffill",
    )


def summarize_groups(
    data: pd.DataFrame,
    group_columns: list[str],
    scope: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for keys, group in data.groupby(
        group_columns,
        sort=False,
        dropna=False,
    ):
        if not isinstance(keys, tuple):
            keys = (keys,)

        row = {
            column: value
            for column, value in zip(
                group_columns,
                keys,
            )
        }

        row.update(
            {
                "scope": scope,
                "observations": int(len(group)),
                "mean_score": float(group["score"].mean()),
                "median_score": float(group["score"].median()),
                "shock_normal_rate_percent": float(
                    group["shock_normal"].mean()
                    * 100.0
                ),
                "eligible_rate_percent": float(
                    group["eligible"].mean()
                    * 100.0
                ),
                "mean_net_15m_percent": float(
                    group["net_return_15m_percent"].mean()
                ),
                "mean_net_1h_percent": float(
                    group["net_return_1h_percent"].mean()
                ),
                "mean_net_4h_percent": float(
                    group["net_return_4h_percent"].mean()
                ),
                "mean_net_12h_percent": float(
                    group["net_return_12h_percent"].mean()
                ),
                "mean_net_24h_percent": float(
                    group["net_return_24h_percent"].mean()
                ),
                "median_net_24h_percent": float(
                    group["net_return_24h_percent"].median()
                ),
                "positive_net_24h_rate_percent": float(
                    (
                        group["net_return_24h_percent"]
                        > 0.0
                    ).mean()
                    * 100.0
                ),
                "mean_mfe_24h_percent": float(
                    group["mfe_24h_percent"].mean()
                ),
                "mean_mae_24h_percent": float(
                    group["mae_24h_percent"].mean()
                ),
            }
        )

        rows.append(row)

    return pd.DataFrame(rows)


def build_summaries(
    detail: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    overall_parts = [
        summarize_groups(
            detail,
            ["window", "band"],
            "ALL_OBSERVATIONS",
        ),
        summarize_groups(
            detail.loc[detail["shock_normal"]],
            ["window", "band"],
            "SHOCK_NORMAL_ONLY",
        ),
    ]

    symbol_parts = [
        summarize_groups(
            detail,
            ["window", "symbol", "band"],
            "ALL_OBSERVATIONS",
        ),
        summarize_groups(
            detail.loc[detail["shock_normal"]],
            ["window", "symbol", "band"],
            "SHOCK_NORMAL_ONLY",
        ),
    ]

    overall = pd.concat(
        overall_parts,
        ignore_index=True,
    )
    symbol = pd.concat(
        symbol_parts,
        ignore_index=True,
    )

    overall["band_order"] = overall["band"].map(
        BAND_ORDER
    )
    symbol["band_order"] = symbol["band"].map(
        BAND_ORDER
    )

    overall = overall.sort_values(
        ["scope", "band_order"]
    ).drop(columns=["band_order"])

    symbol = symbol.sort_values(
        ["scope", "symbol", "band_order"]
    ).drop(columns=["band_order"])

    return overall, symbol


def audit_window(
    window_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    start, end = WINDOWS[window_name]

    print(
        f"Preparing V3KM audit window "
        f"{window_name}: {start} -> {end}"
    )

    (
        frames_15m,
        frames_1h,
        frames_2h,
        frames_4h,
        common_15m,
        common_1h,
    ) = prepare_frames(start, end)

    asset_shocks, market_shocks = build_shock_tables(
        frames_15m,
        common_15m,
    )

    market_state_series = build_market_state_series(
        frames_1h,
        common_1h,
        common_15m,
        market_shocks,
    )

    aligned_1h = {
        symbol: diagnostics.align_frame(
            frames_1h[symbol],
            common_15m,
        )
        for symbol in diagnostics.SYMBOLS
    }

    aligned_2h = {
        symbol: diagnostics.align_frame(
            frames_2h[symbol],
            common_15m,
        )
        for symbol in diagnostics.SYMBOLS
    }

    aligned_4h = {
        symbol: diagnostics.align_frame(
            frames_4h[symbol],
            common_15m,
        )
        for symbol in diagnostics.SYMBOLS
    }

    source_positions = {
        symbol: frames_15m[symbol].index.get_indexer(
            common_15m
        )
        for symbol in diagnostics.SYMBOLS
    }

    records: list[dict[str, object]] = []
    maximum_horizon = max(HORIZON_BARS.values())

    for common_position, decision_time in enumerate(
        common_15m
    ):
        market_state = market_state_series.loc[
            decision_time
        ]

        if not isinstance(
            market_state,
            MarketStateDecision,
        ):
            continue

        market_shock = market_shocks[
            decision_time
        ].level

        for symbol in diagnostics.SYMBOLS:
            source_frame = frames_15m[symbol]
            source_position = int(
                source_positions[symbol][common_position]
            )

            if source_position < 0:
                continue

            if (
                source_position + maximum_horizon
                >= len(source_frame)
            ):
                continue

            expected_24h_time = (
                decision_time
                + pd.Timedelta(hours=24)
            )
            actual_24h_time = source_frame.index[
                source_position + maximum_horizon
            ]

            # Skip observations spanning missing candles.
            if actual_24h_time != expected_24h_time:
                continue

            candle_15m = source_frame.iloc[
                source_position
            ]
            candle_1h = aligned_1h[symbol].loc[
                decision_time
            ]
            candle_2h = aligned_2h[symbol].loc[
                decision_time
            ]
            candle_4h = aligned_4h[symbol].loc[
                decision_time
            ]

            asset_shock = asset_shocks[
                symbol
            ][decision_time].level

            volume_ratio = (
                float(candle_15m["volume"])
                / max(
                    float(candle_15m["VolumeSMA20"]),
                    1e-12,
                )
            )

            decision = evaluate_opportunity(
                market_score=float(
                    market_state.score
                ),
                trend_15m_score=timeframe_trend_score(
                    candle_15m
                ),
                trend_1h_score=timeframe_trend_score(
                    candle_1h
                ),
                trend_2h_score=timeframe_trend_score(
                    candle_2h
                ),
                trend_4h_score=timeframe_trend_score(
                    candle_4h
                ),
                relative_return_60m=float(
                    candle_15m[
                        "MARKET_RELATIVE_60M"
                    ]
                ),
                return_15m=float(
                    candle_15m["RETURN_15M"]
                ),
                volume_ratio=volume_ratio,
                asset_shock=asset_shock,
                market_shock=market_shock,
            )

            entry_row = source_frame.iloc[
                source_position + 1
            ]
            entry_open = float(entry_row["open"])

            horizon_results = {}

            for horizon_name, bars in (
                HORIZON_BARS.items()
            ):
                exit_close = float(
                    source_frame.iloc[
                        source_position + bars
                    ]["close"]
                )

                horizon_results[
                    f"net_return_{horizon_name}_percent"
                ] = net_long_return_percent(
                    entry_open,
                    exit_close,
                )

            path_24h = source_frame.iloc[
                source_position + 1:
                source_position + maximum_horizon + 1
            ]

            maximum_high = float(
                path_24h["high"].max()
            )
            minimum_low = float(
                path_24h["low"].min()
            )

            mfe_24h = (
                (maximum_high / entry_open)
                - 1.0
            ) * 100.0
            mae_24h = (
                (minimum_low / entry_open)
                - 1.0
            ) * 100.0

            shock_normal = (
                asset_shock == ShockLevel.NORMAL
                and market_shock == ShockLevel.NORMAL
            )

            records.append(
                {
                    "window": window_name,
                    "decision_time": decision_time,
                    "symbol": symbol,
                    "score": decision.score,
                    "band": decision.band.value,
                    "target_fraction": (
                        decision.target_fraction
                    ),
                    "eligible": (
                        decision.eligible_for_new_entry
                    ),
                    "shock_normal": shock_normal,
                    "asset_shock": asset_shock.value,
                    "market_shock": market_shock.value,
                    "market_state": (
                        market_state.state.value
                    ),
                    "market_score": (
                        decision.market_score
                    ),
                    "trend_15m_score": (
                        decision.trend_15m_score
                    ),
                    "trend_1h_score": (
                        decision.trend_1h_score
                    ),
                    "trend_2h_score": (
                        decision.trend_2h_score
                    ),
                    "trend_4h_score": (
                        decision.trend_4h_score
                    ),
                    "relative_strength_score": (
                        decision.relative_strength_score
                    ),
                    "positive_volume_score": (
                        decision.positive_volume_score
                    ),
                    "entry_open": entry_open,
                    **horizon_results,
                    "mfe_24h_percent": mfe_24h,
                    "mae_24h_percent": mae_24h,
                }
            )

    detail = pd.DataFrame(records)

    if detail.empty:
        raise RuntimeError(
            f"No audit observations produced for "
            f"{window_name}."
        )

    overall_summary, symbol_summary = (
        build_summaries(detail)
    )

    return detail, overall_summary, symbol_summary


def save_reports(
    window_name: str,
    detail: pd.DataFrame,
    overall_summary: pd.DataFrame,
    symbol_summary: pd.DataFrame,
) -> None:
    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    tag = window_name.lower()

    detail_path = REPORT_DIR / (
        f"v3km_opportunity_{tag}_detail.csv"
    )
    overall_path = REPORT_DIR / (
        f"v3km_opportunity_{tag}_band_summary.csv"
    )
    symbol_path = REPORT_DIR / (
        f"v3km_opportunity_{tag}_symbol_band_summary.csv"
    )

    detail.to_csv(
        detail_path,
        index=False,
    )
    overall_summary.to_csv(
        overall_path,
        index=False,
    )
    symbol_summary.to_csv(
        symbol_path,
        index=False,
    )

    print("")
    print(f"Detail report: {detail_path}")
    print(f"Band summary: {overall_path}")
    print(f"Symbol summary: {symbol_path}")


def print_compact_summary(
    summary: pd.DataFrame,
) -> None:
    compact = summary.loc[
        summary["scope"] == "SHOCK_NORMAL_ONLY",
        [
            "band",
            "observations",
            "mean_score",
            "mean_net_1h_percent",
            "mean_net_4h_percent",
            "mean_net_12h_percent",
            "mean_net_24h_percent",
            "positive_net_24h_rate_percent",
            "mean_mfe_24h_percent",
            "mean_mae_24h_percent",
        ],
    ]

    print("")
    print(
        "V3KM OPPORTUNITY AUDIT "
        "(SHOCK-NORMAL OBSERVATIONS)"
    )
    print(
        "Observations overlap and are not "
        "independent trades."
    )
    print(
        compact.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit V3KM Opportunity Score using "
            "causal completed-candle data."
        )
    )

    parser.add_argument(
        "--window",
        required=True,
        choices=tuple(WINDOWS),
        help=(
            "Historical window to audit: "
            "2022, 2023, 2024 or 2025H1."
        ),
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    detail, overall, symbol = audit_window(
        arguments.window
    )

    save_reports(
        arguments.window,
        detail,
        overall,
        symbol,
    )
    print_compact_summary(overall)


if __name__ == "__main__":
    main()
