from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import run_v3d_diagnostics as diagnostics


DEVELOPMENT_START = pd.Timestamp("2025-08-01 00:00:00")
DEVELOPMENT_END = pd.Timestamp("2026-04-30 00:00:00")

VALIDATION_START = pd.Timestamp("2026-05-01 00:00:00")
VALIDATION_END = pd.Timestamp("2026-07-20 00:00:00")

TRADE_FILES = {
    "DEVELOPMENT": Path(
        "logs/backtests/v3e6_development_trade_history.csv"
    ),
    "VALIDATION": Path(
        "logs/backtests/v3e6_validation_trade_history.csv"
    ),
}

REPORT_DIR = Path("reports")

DETAIL_FILE = REPORT_DIR / "v3e7_trade_environment_detail.csv"
MONTHLY_FILE = REPORT_DIR / "v3e7_monthly_stability.csv"
FILTER_FILE = REPORT_DIR / "v3e7_non_lookahead_filter_study.csv"
WIN_LOSS_FILE = REPORT_DIR / "v3e7_winner_loser_environment.csv"


def profit_factor(values: pd.Series) -> float:
    gross_profit = float(values[values > 0].sum())
    gross_loss = abs(float(values[values < 0].sum()))

    if gross_loss > 0:
        return gross_profit / gross_loss

    if gross_profit > 0:
        return float("inf")

    return 0.0


def summarize(group: pd.DataFrame) -> dict[str, Any]:
    trades = len(group)
    profits = group["profit"]
    winners = int((profits > 0).sum())

    net = float(profits.sum())
    fees = float(group["total_fees"].sum())

    return {
        "trades": trades,
        "winners": winners,
        "win_rate": (
            winners / trades * 100.0
            if trades
            else 0.0
        ),
        "net_profit": net,
        "profit_factor": profit_factor(profits),
        "expectancy": (
            net / trades
            if trades
            else 0.0
        ),
        "fees": fees,
        "pre_fee_profit": net + fees,
        "average_hold_hours": (
            float(group["hold_hours"].mean())
            if trades
            else 0.0
        ),
    }


def load_hourly_frame(
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    diagnostics.BACKTEST_START = start
    diagnostics.BACKTEST_END = end

    frame = diagnostics.prepare_1h(symbol).copy()
    frame = frame.sort_index()

    daily_close = (
        frame["close"]
        .resample("1D")
        .last()
    )

    daily = pd.DataFrame(
        {
            "daily_close": daily_close,
        }
    )

    daily["daily_ema20"] = (
        daily["daily_close"]
        .ewm(
            span=20,
            adjust=False,
        )
        .mean()
    )

    daily["daily_ema50"] = (
        daily["daily_close"]
        .ewm(
            span=50,
            adjust=False,
        )
        .mean()
    )

    daily["daily_momentum_20d"] = (
        daily["daily_close"]
        .pct_change(20)
        .mul(100.0)
    )

    daily["daily_ema20_slope_3d"] = (
        daily["daily_ema20"]
        .pct_change(3)
        .mul(100.0)
    )

    # Shift by one full day so every entry uses only completed prior-day data.
    daily = daily.shift(1)

    daily_hourly = daily.reindex(
        frame.index,
        method="ffill",
    )

    for column in daily_hourly.columns:
        frame[column] = daily_hourly[column]

    frame["daily_macro_bull"] = (
        (frame["daily_close"] > frame["daily_ema20"])
        & (frame["daily_ema20"] > frame["daily_ema50"])
        & (frame["daily_momentum_20d"] > 0)
        & (frame["daily_ema20_slope_3d"] > 0)
    )

    frame["daily_recovery"] = (
        (frame["daily_close"] > frame["daily_ema20"])
        & (frame["daily_momentum_20d"] > 0)
        & (frame["daily_ema20_slope_3d"] > 0)
    )

    frame["hourly_stack"] = (
        (frame["close"] > frame["EMA20"])
        & (frame["EMA20"] > frame["EMA50"])
        & (frame["EMA50"] > frame["EMA200"])
    )

    frame["ema20_extension_percent"] = (
        (frame["close"] - frame["EMA20"])
        / frame["EMA20"]
        * 100.0
    )

    frame["ema200_distance_percent"] = (
        (frame["close"] - frame["EMA200"])
        / frame["EMA200"]
        * 100.0
    )

    return frame


def load_period(
    period: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    trade_file = TRADE_FILES[period]

    if not trade_file.exists():
        raise FileNotFoundError(
            f"Trade history not found: {trade_file}"
        )

    trades = pd.read_csv(trade_file)

    if trades.empty:
        raise ValueError(
            f"{period} trade history is empty."
        )

    trades["entry_time"] = pd.to_datetime(
        trades["entry_time"],
        errors="coerce",
    )

    trades["exit_time"] = pd.to_datetime(
        trades["exit_time"],
        errors="coerce",
    )

    trades["profit"] = pd.to_numeric(
        trades["profit"],
        errors="coerce",
    ).fillna(0.0)

    trades["buy_fee"] = pd.to_numeric(
        trades.get("buy_fee", 0.0),
        errors="coerce",
    ).fillna(0.0)

    trades["sell_fees"] = pd.to_numeric(
        trades.get("sell_fees", 0.0),
        errors="coerce",
    ).fillna(0.0)

    trades["total_fees"] = (
        trades["buy_fee"]
        + trades["sell_fees"]
    )

    trades["hold_hours"] = (
        (
            trades["exit_time"]
            - trades["entry_time"]
        )
        .dt.total_seconds()
        .div(3600.0)
        .fillna(0.0)
    )

    trades["period"] = period
    trades["month"] = (
        trades["entry_time"]
        .dt.to_period("M")
        .astype(str)
    )

    frames = {}

    for symbol in sorted(
        trades["symbol"].unique()
    ):
        print(
            f"Loading {period} 1H environment: {symbol}"
        )
        frames[symbol] = load_hourly_frame(
            symbol,
            start,
            end,
        )

    rows = []

    feature_columns = [
        "close",
        "EMA20",
        "EMA50",
        "EMA200",
        "RSI14",
        "ADX14",
        "ATR14",
        "VOLUME_RATIO",
        "MOMENTUM_24H",
        "MOMENTUM_72H",
        "EMA50_SLOPE_12H",
        "daily_close",
        "daily_ema20",
        "daily_ema50",
        "daily_momentum_20d",
        "daily_ema20_slope_3d",
        "daily_macro_bull",
        "daily_recovery",
        "hourly_stack",
        "ema20_extension_percent",
        "ema200_distance_percent",
    ]

    for trade in trades.to_dict(
        orient="records"
    ):
        symbol = str(trade["symbol"])
        entry_time = pd.Timestamp(
            trade["entry_time"]
        )

        frame = frames[symbol]

        if entry_time not in frame.index:
            position = frame.index.get_indexer(
                [entry_time],
                method="ffill",
            )[0]

            if position < 0:
                continue

            feature_time = frame.index[position]
        else:
            feature_time = entry_time

        candle = frame.loc[feature_time]

        row = dict(trade)
        row["feature_time"] = feature_time

        for column in feature_columns:
            value = candle.get(
                column,
                np.nan,
            )

            if isinstance(
                value,
                (np.bool_, bool),
            ):
                row[column] = bool(value)
            else:
                row[column] = value

        row["momentum_confirmed"] = bool(
            float(row["MOMENTUM_24H"]) > 0
            and float(row["MOMENTUM_72H"]) > 0
        )

        row["trend_quality"] = bool(
            row["hourly_stack"]
            and float(row["ADX14"]) >= 20.0
            and float(row["VOLUME_RATIO"]) >= 0.90
        )

        row["strict_macro_trend"] = bool(
            row["daily_macro_bull"]
            and row["momentum_confirmed"]
            and float(row["ADX14"]) >= 20.0
        )

        row["winner"] = (
            float(row["profit"]) > 0
        )

        rows.append(row)

    return pd.DataFrame(rows)


def grouped_summary(
    frame: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    rows = []

    grouper = (
        group_columns[0]
        if len(group_columns) == 1
        else group_columns
    )

    for key, group in frame.groupby(
        grouper,
        dropna=False,
        sort=False,
    ):
        if not isinstance(key, tuple):
            key = (key,)

        row = {
            column: value
            for column, value
            in zip(
                group_columns,
                key,
            )
        }

        row.update(summarize(group))
        rows.append(row)

    return pd.DataFrame(rows)


def build_filter_study(
    detail: pd.DataFrame,
) -> pd.DataFrame:
    filters = {
        "BASE_ALL_TRADES": (
            pd.Series(
                True,
                index=detail.index,
            )
        ),
        "PRIOR_DAY_RECOVERY": (
            detail["daily_recovery"]
            .fillna(False)
        ),
        "PRIOR_DAY_MACRO_BULL": (
            detail["daily_macro_bull"]
            .fillna(False)
        ),
        "POSITIVE_24H_72H_MOMENTUM": (
            detail["momentum_confirmed"]
            .fillna(False)
        ),
        "HOURLY_TREND_QUALITY": (
            detail["trend_quality"]
            .fillna(False)
        ),
        "MACRO_BULL_PLUS_MOMENTUM": (
            detail["daily_macro_bull"]
            .fillna(False)
            & detail["momentum_confirmed"]
            .fillna(False)
        ),
        "STRICT_MACRO_TREND": (
            detail["strict_macro_trend"]
            .fillna(False)
        ),
        "ADX_25_PLUS": (
            detail["ADX14"] >= 25.0
        ),
        "VOLUME_1_PLUS": (
            detail["VOLUME_RATIO"] >= 1.0
        ),
        "NOT_OVEREXTENDED_2_PERCENT": (
            detail[
                "ema20_extension_percent"
            ] <= 2.0
        ),
    }

    rows = []

    for period in [
        "DEVELOPMENT",
        "VALIDATION",
        "COMBINED",
    ]:
        period_frame = (
            detail
            if period == "COMBINED"
            else detail[
                detail["period"] == period
            ]
        )

        for filter_name, mask in filters.items():
            period_mask = mask.reindex(
                period_frame.index
            ).fillna(False)

            filtered = period_frame[
                period_mask
            ]

            result = summarize(filtered)

            rows.append(
                {
                    "period": period,
                    "filter": filter_name,
                    **result,
                }
            )

    # Route-specific combinations.
    route_filters = [
        (
            "LINK_IGNITION_MACRO_BULL",
            (
                (detail["symbol"] == "LINKUSDT")
                & (detail["route"] == "IGNITION")
                & detail[
                    "daily_macro_bull"
                ].fillna(False)
            ),
        ),
        (
            "LINK_IGNITION_STRICT_MACRO",
            (
                (detail["symbol"] == "LINKUSDT")
                & (detail["route"] == "IGNITION")
                & detail[
                    "strict_macro_trend"
                ].fillna(False)
            ),
        ),
        (
            "SOL_BREAKOUT_MACRO_BULL",
            (
                (detail["symbol"] == "SOLUSDT")
                & (
                    detail["route"]
                    == "ADAPTIVE_BREAKOUT"
                )
                & detail[
                    "daily_macro_bull"
                ].fillna(False)
            ),
        ),
        (
            "SOL_BREAKOUT_STRICT_MACRO",
            (
                (detail["symbol"] == "SOLUSDT")
                & (
                    detail["route"]
                    == "ADAPTIVE_BREAKOUT"
                )
                & detail[
                    "strict_macro_trend"
                ].fillna(False)
            ),
        ),
    ]

    for period in [
        "DEVELOPMENT",
        "VALIDATION",
        "COMBINED",
    ]:
        period_mask = (
            pd.Series(
                True,
                index=detail.index,
            )
            if period == "COMBINED"
            else detail["period"].eq(period)
        )

        for name, route_mask in route_filters:
            filtered = detail[
                period_mask
                & route_mask
            ]

            rows.append(
                {
                    "period": period,
                    "filter": name,
                    **summarize(filtered),
                }
            )

    return pd.DataFrame(rows)


def print_summary_table(
    title: str,
    frame: pd.DataFrame,
    key_columns: list[str],
) -> None:
    print("-" * 160)
    print(title)
    print("-" * 160)

    if frame.empty:
        print("No data.")
        return

    for row in frame.to_dict(
        orient="records"
    ):
        keys = " | ".join(
            f"{column}={row[column]}"
            for column in key_columns
        )

        factor = float(
            row["profit_factor"]
        )

        factor_text = (
            "INF"
            if np.isinf(factor)
            else f"{factor:.2f}"
        )

        print(
            f"{keys:<75} | "
            f"Trades {int(row['trades']):>3} | "
            f"Win {float(row['win_rate']):>6.2f}% | "
            f"PF {factor_text:>5} | "
            f"P&L €{float(row['net_profit']):>8.2f} | "
            f"Fees €{float(row['fees']):>7.2f} | "
            f"Expect €{float(row['expectancy']):>7.2f}"
        )


def main() -> None:
    print("=" * 160)
    print(
        "FA CRYPTO ENGINE — V3E7 REGIME/ENVIRONMENT EDGE DIAGNOSTICS"
    )
    print("=" * 160)
    print(
        "Read-only analysis. V3D3 surveillance, risk and exits are not modified."
    )

    development = load_period(
        "DEVELOPMENT",
        DEVELOPMENT_START,
        DEVELOPMENT_END,
    )

    validation = load_period(
        "VALIDATION",
        VALIDATION_START,
        VALIDATION_END,
    )

    detail = pd.concat(
        [
            development,
            validation,
        ],
        ignore_index=True,
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    detail.to_csv(
        DETAIL_FILE,
        index=False,
    )

    monthly = grouped_summary(
        detail,
        [
            "period",
            "month",
            "route",
            "symbol",
        ],
    ).sort_values(
        [
            "period",
            "month",
            "route",
            "symbol",
        ]
    )

    monthly.to_csv(
        MONTHLY_FILE,
        index=False,
    )

    filter_study = build_filter_study(
        detail
    )

    filter_study.to_csv(
        FILTER_FILE,
        index=False,
    )

    environment_rows = []

    numeric_features = [
        "RSI14",
        "ADX14",
        "VOLUME_RATIO",
        "MOMENTUM_24H",
        "MOMENTUM_72H",
        "EMA50_SLOPE_12H",
        "ema20_extension_percent",
        "ema200_distance_percent",
        "daily_momentum_20d",
        "daily_ema20_slope_3d",
    ]

    for (
        period,
        symbol,
        route,
        winner,
    ), group in detail.groupby(
        [
            "period",
            "symbol",
            "route",
            "winner",
        ],
        dropna=False,
    ):
        row = {
            "period": period,
            "symbol": symbol,
            "route": route,
            "winner": winner,
            "trades": len(group),
        }

        for feature in numeric_features:
            row[
                f"median_{feature}"
            ] = float(
                pd.to_numeric(
                    group[feature],
                    errors="coerce",
                ).median()
            )

        row["macro_bull_percent"] = float(
            group[
                "daily_macro_bull"
            ].fillna(False).mean()
            * 100.0
        )

        row["strict_macro_percent"] = float(
            group[
                "strict_macro_trend"
            ].fillna(False).mean()
            * 100.0
        )

        environment_rows.append(row)

    winner_loser = pd.DataFrame(
        environment_rows
    )

    winner_loser.to_csv(
        WIN_LOSS_FILE,
        index=False,
    )

    period_summary = grouped_summary(
        detail,
        ["period"],
    )

    print_summary_table(
        "PERIOD SUMMARY",
        period_summary,
        ["period"],
    )

    print_summary_table(
        "MONTHLY ROUTE × ASSET STABILITY",
        monthly,
        [
            "period",
            "month",
            "route",
            "symbol",
        ],
    )

    important_filters = filter_study[
        filter_study["filter"].isin(
            [
                "BASE_ALL_TRADES",
                "PRIOR_DAY_RECOVERY",
                "PRIOR_DAY_MACRO_BULL",
                "POSITIVE_24H_72H_MOMENTUM",
                "MACRO_BULL_PLUS_MOMENTUM",
                "STRICT_MACRO_TREND",
                "LINK_IGNITION_MACRO_BULL",
                "LINK_IGNITION_STRICT_MACRO",
                "SOL_BREAKOUT_MACRO_BULL",
                "SOL_BREAKOUT_STRICT_MACRO",
            ]
        )
    ]

    print_summary_table(
        "NON-LOOKAHEAD FILTER STUDY",
        important_filters,
        [
            "period",
            "filter",
        ],
    )

    print("-" * 160)
    print(
        "WINNER / LOSER ENVIRONMENT MEDIANS"
    )
    print("-" * 160)

    for row in winner_loser.to_dict(
        orient="records"
    ):
        print(
            f"{row['period']:<12} | "
            f"{row['symbol']:<10} | "
            f"{row['route']:<20} | "
            f"{'WIN' if row['winner'] else 'LOSS':<4} | "
            f"N {int(row['trades']):>2} | "
            f"ADX {row['median_ADX14']:>6.2f} | "
            f"Vol {row['median_VOLUME_RATIO']:>5.2f} | "
            f"M24 {row['median_MOMENTUM_24H']:>7.2f}% | "
            f"M72 {row['median_MOMENTUM_72H']:>7.2f}% | "
            f"Daily20 {row['median_daily_momentum_20d']:>7.2f}% | "
            f"MacroBull {row['macro_bull_percent']:>6.2f}%"
        )

    print("=" * 160)
    print(
        f"Trade environment detail : {DETAIL_FILE}"
    )
    print(
        f"Monthly stability        : {MONTHLY_FILE}"
    )
    print(
        f"Filter study             : {FILTER_FILE}"
    )
    print(
        f"Winner/loser environment : {WIN_LOSS_FILE}"
    )
    print("=" * 160)


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        print()
        print(
            "V3E7 diagnostics stopped manually."
        )

    except Exception as error:
        print()
        print("=" * 160)
        print(
            "V3E7 DIAGNOSTIC ERROR"
        )
        print("=" * 160)
        print(
            f"{type(error).__name__}: {error}"
        )
        print("=" * 160)
        raise
