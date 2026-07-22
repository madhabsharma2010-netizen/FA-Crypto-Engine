from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PERIOD_CONFIG = {
    "development": {
        "candidate_audit": Path("reports/v3f1_development_candidate_execution_audit.csv"),
        "signal_events": Path("reports/v3f1_development_signal_events.csv"),
        "trade_log": Path("logs/backtests/v3e6_development_trade_history.csv"),
    },
    "validation": {
        "candidate_audit": Path("reports/v3f1_validation_candidate_execution_audit.csv"),
        "signal_events": Path("reports/v3f1_validation_signal_events.csv"),
        "trade_log": Path("logs/backtests/v3e6_validation_trade_history.csv"),
    },
}

ROUND_TRIP_COST_PERCENT = 0.30
MIN_GROUP_SIZE = 5

NUMERIC_FEATURES = (
    "relative_strength_percentile",
    "setup_score",
    "selection_score",
    "theoretical_rr_at_signal_close",
    "actual_rr_after_gap_and_slippage",
    "opening_gap_atr",
    "gross_projected_target_move_percent",
    "cost_as_percent_of_projected_target_move",
    "target_move_to_round_trip_cost_multiple",
    "signal_first_true_to_fill_hours",
)

CATEGORICAL_FEATURES = (
    "symbol",
    "route",
    "market_state",
)


def _read_csv(path: Path, required: bool = True) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise FileNotFoundError(
                f"Required file not found: {path}. Run V3F1 for the same period first."
            )
        return pd.DataFrame()
    return pd.read_csv(path)


def _parse_time(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        if column in result.columns:
            result[column] = pd.to_datetime(result[column], errors="coerce")
    return result


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y"})
    )


def _safe_numeric(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def _profit_factor(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    gross_profit = float(values[values > 0].sum())
    gross_loss = abs(float(values[values < 0].sum()))
    if gross_loss > 0:
        return gross_profit / gross_loss
    if gross_profit > 0:
        return float("inf")
    return 0.0


def _cohens_d(winners: pd.Series, losers: pd.Series) -> float:
    winners = pd.to_numeric(winners, errors="coerce").dropna()
    losers = pd.to_numeric(losers, errors="coerce").dropna()
    if len(winners) < 2 or len(losers) < 2:
        return 0.0

    variance_numerator = (
        (len(winners) - 1) * float(winners.var(ddof=1))
        + (len(losers) - 1) * float(losers.var(ddof=1))
    )
    denominator = len(winners) + len(losers) - 2
    if denominator <= 0:
        return 0.0

    pooled_std = math.sqrt(max(variance_numerator / denominator, 0.0))
    if pooled_std <= 0:
        return 0.0
    return (float(winners.mean()) - float(losers.mean())) / pooled_std


def _group_metrics(group: pd.DataFrame) -> dict[str, Any]:
    profits = pd.to_numeric(group["profit"], errors="coerce").dropna()
    winners = profits[profits > 0]
    losers = profits[profits < 0]
    total = len(profits)
    fees = 0.0
    for column in ("buy_fee", "sell_fees", "sell_fee"):
        if column in group.columns:
            fees += float(pd.to_numeric(group[column], errors="coerce").fillna(0.0).sum())

    return {
        "trades": int(total),
        "wins": int(len(winners)),
        "losses": int(len(losers)),
        "win_rate_percent": round(len(winners) / total * 100.0, 4) if total else 0.0,
        "net_pnl_eur": round(float(profits.sum()), 8),
        "gross_profit_eur": round(float(winners.sum()), 8),
        "gross_loss_eur": round(abs(float(losers.sum())), 8),
        "profit_factor": round(_profit_factor(profits), 8),
        "average_trade_eur": round(float(profits.mean()), 8) if total else 0.0,
        "average_win_eur": round(float(winners.mean()), 8) if len(winners) else 0.0,
        "average_loss_eur": round(abs(float(losers.mean())), 8) if len(losers) else 0.0,
        "explicit_fees_eur": round(fees, 8),
    }


def _prepare_dataset(period: str) -> pd.DataFrame:
    config = PERIOD_CONFIG[period]
    candidate = _read_csv(config["candidate_audit"])
    signals = _read_csv(config["signal_events"])
    trades = _read_csv(config["trade_log"])

    candidate = _parse_time(
        candidate,
        ("signal_first_true_time", "signal_evaluated_time", "order_fill_time"),
    )
    signals = _parse_time(
        signals,
        ("decision_time", "quality_episode_first_true_time"),
    )
    trades = _parse_time(trades, ("entry_time", "exit_time"))

    if "actual_trade_entered" not in candidate.columns:
        raise ValueError("V3F1 candidate audit is missing actual_trade_entered.")

    candidate["actual_trade_entered"] = _as_bool(candidate["actual_trade_entered"])
    executed = candidate[candidate["actual_trade_entered"]].copy()

    if executed.empty:
        raise ValueError("No executed candidates found in the V3F1 candidate audit.")

    signal_columns = [
        "decision_time",
        "symbol",
        "route",
        "market_state",
        "relative_strength_percentile",
        "setup_score",
        "selection_score",
        "candidate_emitted_after_route_cooldown",
    ]
    available_signal_columns = [column for column in signal_columns if column in signals.columns]
    emitted_signals = signals[available_signal_columns].copy()
    if "candidate_emitted_after_route_cooldown" in emitted_signals.columns:
        emitted_signals["candidate_emitted_after_route_cooldown"] = _as_bool(
            emitted_signals["candidate_emitted_after_route_cooldown"]
        )
        emitted_signals = emitted_signals[
            emitted_signals["candidate_emitted_after_route_cooldown"]
        ]

    emitted_signals = emitted_signals.drop_duplicates(
        subset=["decision_time", "symbol", "route"],
        keep="first",
    )

    dataset = executed.merge(
        emitted_signals,
        left_on=["signal_evaluated_time", "symbol", "route"],
        right_on=["decision_time", "symbol", "route"],
        how="left",
        suffixes=("", "_signal"),
    )

    trade_columns = [
        "entry_time",
        "exit_time",
        "symbol",
        "route",
        "profit",
        "exit_reason",
        "buy_fee",
        "sell_fees",
        "sell_fee",
        "partial_profit_taken",
    ]
    available_trade_columns = [column for column in trade_columns if column in trades.columns]
    trades_for_join = trades[available_trade_columns].copy()

    dataset = dataset.merge(
        trades_for_join,
        left_on=["signal_evaluated_time", "symbol", "route"],
        right_on=["entry_time", "symbol", "route"],
        how="left",
        validate="one_to_one",
    )

    if dataset["profit"].isna().any():
        missing = dataset[dataset["profit"].isna()][
            ["signal_evaluated_time", "symbol", "route"]
        ]
        raise ValueError(
            "Could not match all executed candidates to trade-log rows. "
            f"Missing matches: {len(missing)}"
        )

    dataset = _safe_numeric(
        dataset,
        NUMERIC_FEATURES
        + (
            "profit",
            "buy_fee",
            "sell_fees",
            "sell_fee",
        ),
    )

    projected_move = pd.to_numeric(
        dataset.get("gross_projected_target_move_percent", 0.0),
        errors="coerce",
    ).fillna(0.0)
    dataset["target_move_to_round_trip_cost_multiple"] = (
        projected_move / ROUND_TRIP_COST_PERCENT
    )

    dataset["winner"] = dataset["profit"] > 0
    dataset["loser"] = dataset["profit"] < 0
    dataset["holding_hours"] = (
        dataset["exit_time"] - dataset["entry_time"]
    ).dt.total_seconds() / 3600.0
    dataset["stop_type_exit"] = dataset["exit_reason"].astype(str).isin(
        {"STOP LOSS", "EMERGENCY STOP EXIT"}
    )
    dataset["month"] = dataset["entry_time"].dt.to_period("M").astype(str)

    return dataset


def _numeric_summary(dataset: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    winners = dataset[dataset["winner"]]
    losers = dataset[dataset["loser"]]

    for feature in NUMERIC_FEATURES:
        if feature not in dataset.columns:
            continue
        all_values = pd.to_numeric(dataset[feature], errors="coerce").dropna()
        winner_values = pd.to_numeric(winners[feature], errors="coerce").dropna()
        loser_values = pd.to_numeric(losers[feature], errors="coerce").dropna()
        if all_values.empty:
            continue

        rows.append(
            {
                "feature": feature,
                "available_trades": int(len(all_values)),
                "winner_n": int(len(winner_values)),
                "loser_n": int(len(loser_values)),
                "overall_mean": round(float(all_values.mean()), 8),
                "winner_mean": round(float(winner_values.mean()), 8) if len(winner_values) else 0.0,
                "loser_mean": round(float(loser_values.mean()), 8) if len(loser_values) else 0.0,
                "winner_median": round(float(winner_values.median()), 8) if len(winner_values) else 0.0,
                "loser_median": round(float(loser_values.median()), 8) if len(loser_values) else 0.0,
                "winner_minus_loser_mean": round(
                    float(winner_values.mean() - loser_values.mean()), 8
                )
                if len(winner_values) and len(loser_values)
                else 0.0,
                "cohens_d_winner_minus_loser": round(
                    _cohens_d(winner_values, loser_values), 8
                ),
                "discovery_only": True,
            }
        )

    result = pd.DataFrame(rows)
    if not result.empty:
        result["absolute_effect_size"] = result[
            "cohens_d_winner_minus_loser"
        ].abs()
        result = result.sort_values(
            ["absolute_effect_size", "available_trades"],
            ascending=[False, False],
        )
    return result


def _quartile_summary(dataset: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for feature in NUMERIC_FEATURES:
        if feature not in dataset.columns:
            continue
        usable = dataset.dropna(subset=[feature]).copy()
        if usable[feature].nunique() < 2 or len(usable) < 8:
            continue

        try:
            usable["feature_bucket"] = pd.qcut(
                usable[feature],
                q=min(4, usable[feature].nunique()),
                duplicates="drop",
            )
        except ValueError:
            continue

        for bucket, group in usable.groupby("feature_bucket", observed=True):
            metrics = _group_metrics(group)
            rows.append(
                {
                    "feature": feature,
                    "bucket": str(bucket),
                    "minimum_value": round(float(group[feature].min()), 8),
                    "maximum_value": round(float(group[feature].max()), 8),
                    **metrics,
                    "discovery_only": True,
                }
            )

    return pd.DataFrame(rows)


def _categorical_summary(dataset: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature in CATEGORICAL_FEATURES:
        if feature not in dataset.columns:
            continue
        for value, group in dataset.groupby(feature, dropna=False):
            rows.append(
                {
                    "feature": feature,
                    "value": str(value),
                    **_group_metrics(group),
                    "discovery_only": True,
                }
            )
    return pd.DataFrame(rows)


def _monthly_summary(dataset: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for month, group in dataset.groupby("month"):
        rows.append({"month": month, **_group_metrics(group)})
    return pd.DataFrame(rows).sort_values("month") if rows else pd.DataFrame()


def _median_split_discovery(dataset: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for feature in NUMERIC_FEATURES:
        if feature not in dataset.columns:
            continue
        usable = dataset.dropna(subset=[feature]).copy()
        if len(usable) < MIN_GROUP_SIZE * 2 or usable[feature].nunique() < 2:
            continue

        threshold = float(usable[feature].median())
        lower = usable[usable[feature] < threshold]
        upper = usable[usable[feature] >= threshold]
        if len(lower) < MIN_GROUP_SIZE or len(upper) < MIN_GROUP_SIZE:
            continue

        lower_metrics = _group_metrics(lower)
        upper_metrics = _group_metrics(upper)

        preferred_side = "UPPER" if upper_metrics["net_pnl_eur"] > lower_metrics["net_pnl_eur"] else "LOWER"
        preferred = upper_metrics if preferred_side == "UPPER" else lower_metrics
        rejected = lower_metrics if preferred_side == "UPPER" else upper_metrics

        preferred_months = upper if preferred_side == "UPPER" else lower
        profitable_months = 0
        observed_months = 0
        for _, month_group in preferred_months.groupby("month"):
            observed_months += 1
            profitable_months += int(float(month_group["profit"].sum()) > 0)

        rows.append(
            {
                "feature": feature,
                "median_threshold": round(threshold, 8),
                "preferred_side": preferred_side,
                "preferred_trades": preferred["trades"],
                "preferred_win_rate_percent": preferred["win_rate_percent"],
                "preferred_profit_factor": preferred["profit_factor"],
                "preferred_net_pnl_eur": preferred["net_pnl_eur"],
                "other_trades": rejected["trades"],
                "other_win_rate_percent": rejected["win_rate_percent"],
                "other_profit_factor": rejected["profit_factor"],
                "other_net_pnl_eur": rejected["net_pnl_eur"],
                "preferred_profitable_months": profitable_months,
                "preferred_observed_months": observed_months,
                "development_discovery_candidate": bool(
                    preferred["net_pnl_eur"] > 0
                    and preferred["profit_factor"] > 1.0
                    and rejected["profit_factor"] < preferred["profit_factor"]
                    and profitable_months >= 2
                ),
                "must_validate_out_of_sample": True,
            }
        )

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(
            [
                "development_discovery_candidate",
                "preferred_net_pnl_eur",
                "preferred_profit_factor",
            ],
            ascending=[False, False, False],
        )
    return result


def run_forensics(period: str) -> dict[str, Any]:
    print("=" * 144)
    print("FA CRYPTO ENGINE — V3F2 ENTRY-EDGE FEATURE FORENSICS")
    print("=" * 144)
    print(f"Period: {period.upper()}")
    print("Uses V3F1 reports and existing trade logs only; no Binance requests.")
    print("No strategy, route, exit or V3D3 risk threshold is changed.")

    dataset = _prepare_dataset(period)
    overall = _group_metrics(dataset)

    average_win = float(overall["average_win_eur"])
    average_loss = float(overall["average_loss_eur"])
    realized_ratio = average_win / average_loss if average_loss > 0 else 0.0
    break_even_win_rate = 100.0 / (1.0 + realized_ratio) if realized_ratio > 0 else 100.0

    numeric = _numeric_summary(dataset)
    quartiles = _quartile_summary(dataset)
    categorical = _categorical_summary(dataset)
    monthly = _monthly_summary(dataset)
    discovery = _median_split_discovery(dataset)

    low_headroom = int(
        (dataset["target_move_to_round_trip_cost_multiple"] < 4.0).sum()
    )
    stop_losers = dataset[dataset["loser"] & dataset["stop_type_exit"]]
    total_losers = dataset[dataset["loser"]]
    stop_loser_rate = (
        len(stop_losers) / len(total_losers) * 100.0
        if len(total_losers)
        else 0.0
    )

    discovery_count = (
        int(discovery["development_discovery_candidate"].sum())
        if not discovery.empty
        else 0
    )

    summary: dict[str, Any] = {
        "period": period,
        **overall,
        "realized_win_loss_ratio": round(realized_ratio, 8),
        "required_break_even_win_rate_percent": round(break_even_win_rate, 4),
        "actual_win_rate_gap_to_break_even_points": round(
            overall["win_rate_percent"] - break_even_win_rate, 4
        ),
        "round_trip_cost_percent": ROUND_TRIP_COST_PERCENT,
        "trades_with_projected_move_below_4x_cost": low_headroom,
        "trades_with_projected_move_below_4x_cost_percent": round(
            low_headroom / len(dataset) * 100.0, 4
        )
        if len(dataset)
        else 0.0,
        "losing_stop_type_exit_percent": round(stop_loser_rate, 4),
        "development_discovery_features_requiring_validation": discovery_count,
        "automatic_threshold_changes_made": False,
    }

    reports = Path("reports")
    reports.mkdir(parents=True, exist_ok=True)
    prefix = reports / f"v3f2_{period}"

    detail_path = Path(f"{prefix}_entry_feature_detail.csv")
    numeric_path = Path(f"{prefix}_numeric_winner_loser_summary.csv")
    quartile_path = Path(f"{prefix}_feature_quartile_buckets.csv")
    categorical_path = Path(f"{prefix}_categorical_summary.csv")
    monthly_path = Path(f"{prefix}_monthly_stability.csv")
    discovery_path = Path(f"{prefix}_discovery_shortlist.csv")
    summary_path = Path(f"{prefix}_summary.csv")

    dataset.to_csv(detail_path, index=False)
    numeric.to_csv(numeric_path, index=False)
    quartiles.to_csv(quartile_path, index=False)
    categorical.to_csv(categorical_path, index=False)
    monthly.to_csv(monthly_path, index=False)
    discovery.to_csv(discovery_path, index=False)
    pd.DataFrame([summary]).to_csv(summary_path, index=False)

    print()
    print("=" * 144)
    print("V3F2 ENTRY-EDGE RESULT")
    print("=" * 144)
    print(f"Matched executed trades                 : {len(dataset)}")
    print(f"Winning / losing                       : {overall['wins']} / {overall['losses']}")
    print(f"Actual win rate                        : {overall['win_rate_percent']:.2f}%")
    print(f"Average win / loss                     : €{average_win:.2f} / €{average_loss:.2f}")
    print(f"Realized win/loss ratio                : {realized_ratio:.2f}")
    print(f"Required break-even win rate           : {break_even_win_rate:.2f}%")
    print(f"Win-rate gap to break-even             : {summary['actual_win_rate_gap_to_break_even_points']:.2f} points")
    print(f"Profit factor / net P&L                : {overall['profit_factor']:.2f} / €{overall['net_pnl_eur']:.2f}")
    print(f"Explicit fees                          : €{overall['explicit_fees_eur']:.2f}")
    print(f"Losing stop-type exit rate             : {stop_loser_rate:.2f}%")
    print(f"Projected move below 4x cost hurdle    : {low_headroom}/{len(dataset)}")
    print(f"Discovery features needing validation  : {discovery_count}")

    if not discovery.empty:
        candidates = discovery[discovery["development_discovery_candidate"]].head(5)
        if not candidates.empty:
            print("-" * 144)
            print("DEVELOPMENT-ONLY DISCOVERY SHORTLIST — DO NOT DEPLOY WITHOUT VALIDATION")
            for _, row in candidates.iterrows():
                print(
                    f"{row['feature']:<48} | {row['preferred_side']:<5} median {row['median_threshold']:.4f} "
                    f"| PF {row['preferred_profit_factor']:.2f} | P&L €{row['preferred_net_pnl_eur']:.2f} "
                    f"| profitable months {int(row['preferred_profitable_months'])}/{int(row['preferred_observed_months'])}"
                )

    print("=" * 144)
    print(f"Trade feature detail : {detail_path}")
    print(f"Winner/loser summary : {numeric_path}")
    print(f"Quartile buckets     : {quartile_path}")
    print(f"Categorical summary  : {categorical_path}")
    print(f"Monthly stability    : {monthly_path}")
    print(f"Discovery shortlist  : {discovery_path}")
    print(f"Summary              : {summary_path}")
    print("=" * 144)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="V3F2 development-only entry feature forensics using V3F1 reports."
    )
    parser.add_argument(
        "--period",
        choices=sorted(PERIOD_CONFIG),
        default="development",
    )
    args = parser.parse_args()
    run_forensics(args.period)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nV3F2 forensics stopped manually.")
    except Exception as error:
        print()
        print("=" * 144)
        print("V3F2 ENTRY-EDGE FORENSICS ERROR")
        print("=" * 144)
        print(f"{type(error).__name__}: {error}")
        print("=" * 144)
        raise
