from __future__ import annotations

import argparse
import importlib
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


PERIOD_CONFIG = {
    "development": {
        "module": "run_v3e6_selective_development",
        "trade_file": Path("logs/backtests/v3e6_development_trade_history.csv"),
    },
    "validation": {
        "module": "run_v3e6_selective_validation",
        "trade_file": Path("logs/backtests/v3e6_validation_trade_history.csv"),
    },
}

HTF_REQUESTS = (
    ("BTCUSDT", "2h"),
    ("BTCUSDT", "4h"),
    ("BTCUSDT", "1d"),
    ("ETHUSDT", "2h"),
    ("ETHUSDT", "4h"),
)

DEFENSIVE_EXIT_REASONS = {
    "STOP LOSS",
    "ASSET SEVERE SHOCK",
    "MARKET SEVERE SHOCK",
    "DAILY LOSS LIMIT",
    "WEEKLY LOSS LIMIT",
    "HARD DRAWDOWN LOCK",
}

HIGH_BETA_ASSETS = {
    "SOLUSDT",
    "XRPUSDT",
    "LINKUSDT",
    "DOGEUSDT",
}


def _retry(label: str, function, attempts: int = 5):
    last_error: Exception | None = None

    for attempt in range(1, attempts + 1):
        try:
            print(f"{label} (attempt {attempt}/{attempts})...")
            return function()
        except Exception as error:  # network retries are intentionally broad
            last_error = error
            if attempt >= attempts:
                break
            pause_seconds = min(2 ** (attempt - 1), 12)
            print(
                f"Temporary error: {type(error).__name__}: {error}. "
                f"Retrying after {pause_seconds}s."
            )
            time.sleep(pause_seconds)

    assert last_error is not None
    raise last_error


def _prepare_complete_cache(module):
    frames_15m, frames_1h, common_15m, common_1h = _retry(
        "Downloading 15M/1H data once",
        module.prepare_frames,
    )

    original_load_frame = module.load_frame
    htf_cache: dict[tuple[str, str], pd.DataFrame] = {}

    for symbol, timeframe in HTF_REQUESTS:
        htf_cache[(symbol, timeframe)] = _retry(
            f"Downloading once: {symbol} {timeframe}",
            lambda s=symbol, tf=timeframe: original_load_frame(s, tf).copy(),
        )

    def offline_load_frame(symbol: str, timeframe: str) -> pd.DataFrame:
        key = (symbol, timeframe)
        if key not in htf_cache:
            raise RuntimeError(
                f"Unexpected offline frame request: {symbol} {timeframe}."
            )
        return htf_cache[key].copy()

    module.load_frame = offline_load_frame

    print("COMPLETE OFFLINE CACHE READY.")
    print("The remaining diagnostic will make no Binance requests.")

    return frames_15m, frames_1h, common_15m, common_1h


def _load_trades(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    trades = pd.read_csv(path)
    if trades.empty:
        return trades

    for column in ("entry_time", "exit_time"):
        if column in trades.columns:
            trades[column] = pd.to_datetime(trades[column], errors="coerce")

    if "profit" in trades.columns:
        trades["profit"] = pd.to_numeric(trades["profit"], errors="coerce")

    return trades.dropna(subset=["entry_time", "exit_time"]).copy()


def _as_text(values: Any) -> str:
    if values is None:
        return ""
    if isinstance(values, (tuple, list, set)):
        return "|".join(str(item) for item in values)
    return str(values)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def _active_trades_before(trades: pd.DataFrame, timestamp: pd.Timestamp) -> pd.DataFrame:
    if trades.empty:
        return trades
    return trades[
        (trades["entry_time"] < timestamp)
        & (trades["exit_time"] > timestamp)
    ]


def _cooldown_reason(
    trades: pd.DataFrame,
    symbol: str,
    timestamp: pd.Timestamp,
    normal_hours: int,
    defensive_hours: int,
) -> str | None:
    if trades.empty:
        return None

    previous = trades[
        (trades["symbol"] == symbol)
        & (trades["exit_time"] <= timestamp)
    ].sort_values("exit_time")

    if previous.empty:
        return None

    last_trade = previous.iloc[-1]
    reason = str(last_trade.get("exit_reason", ""))
    hours = defensive_hours if reason in DEFENSIVE_EXIT_REASONS else normal_hours
    cooldown_until = pd.Timestamp(last_trade["exit_time"]) + pd.Timedelta(hours=hours)

    if timestamp < cooldown_until:
        return f"POST_EXIT_COOLDOWN_UNTIL_{cooldown_until}"

    return None


def _trade_summary(trades: pd.DataFrame) -> dict[str, Any]:
    if trades.empty or "profit" not in trades.columns:
        return {
            "trade_log_found": False,
            "completed_trades": 0,
        }

    winners = trades[trades["profit"] > 0]
    losers = trades[trades["profit"] < 0]
    gross_profit = float(winners["profit"].sum())
    gross_loss = abs(float(losers["profit"].sum()))
    average_win = float(winners["profit"].mean()) if not winners.empty else 0.0
    average_loss = abs(float(losers["profit"].mean())) if not losers.empty else 0.0

    fees = 0.0
    for column in ("buy_fee", "sell_fees", "sell_fee"):
        if column in trades.columns:
            fees += float(pd.to_numeric(trades[column], errors="coerce").fillna(0).sum())

    losing_stop = 0
    winning_full_tp = 0
    if "exit_reason" in trades.columns:
        losing_stop = int(
            losers["exit_reason"].astype(str).isin(
                {"STOP LOSS", "EMERGENCY STOP EXIT"}
            ).sum()
        )
        winning_full_tp = int(
            winners["exit_reason"].astype(str).str.contains(
                "FULL TAKE PROFIT|FULL TP|TAKE PROFIT",
                case=False,
                regex=True,
            ).sum()
        )

    return {
        "trade_log_found": True,
        "completed_trades": int(len(trades)),
        "winning_trades": int(len(winners)),
        "losing_trades": int(len(losers)),
        "win_rate_percent": round(len(winners) / len(trades) * 100.0, 4) if len(trades) else 0.0,
        "gross_profit_eur": round(gross_profit, 8),
        "gross_loss_eur": round(gross_loss, 8),
        "average_win_eur": round(average_win, 8),
        "average_loss_eur": round(average_loss, 8),
        "realized_win_loss_ratio": round(average_win / average_loss, 8) if average_loss > 0 else 0.0,
        "profit_factor": round(gross_profit / gross_loss, 8) if gross_loss > 0 else 0.0,
        "losing_stop_exit_percent": round(losing_stop / len(losers) * 100.0, 4) if len(losers) else 0.0,
        "winning_full_take_profit_percent": round(winning_full_tp / len(winners) * 100.0, 4) if len(winners) else 0.0,
        "total_explicit_fees_eur": round(fees, 8),
        "fees_as_percent_of_gross_profit": round(fees / gross_profit * 100.0, 4) if gross_profit > 0 else 0.0,
    }


def run_diagnostic(period: str) -> dict[str, Any]:
    config = PERIOD_CONFIG[period]
    module = importlib.import_module(config["module"])

    print("=" * 140)
    print("FA CRYPTO ENGINE — V3F1 SIGNAL INTEGRITY + COST-HURDLE DIAGNOSTIC")
    print("=" * 140)
    print(f"Period: {period.upper()}")
    print(f"Source module: {config['module']}")
    print("No strategy thresholds or V3D3 risk limits are changed.")

    frames_15m, frames_1h, common_15m, common_1h = _prepare_complete_cache(module)

    print("Building 15M surveillance...")
    asset_shocks, market_shocks = module.build_shock_tables(
        frames_15m,
        common_15m,
    )

    print("Building non-lookahead market states...")
    market_states = module.build_market_states(
        frames_1h,
        common_1h,
        market_shocks,
    )

    print("Building raw route signals...")
    route_signals_by_time, route_inventory = module.build_new_routes(
        frames_1h,
        common_1h,
        market_states,
        asset_shocks,
    )

    # The pullback dictionary is required by the V3E6 function signature, but
    # broad pullbacks are disabled inside build_profit_candidates.
    empty_pullbacks = {timestamp: [] for timestamp in common_1h}

    print("Replaying the exact V3E6 route cooldown and best-route selection...")
    emitted_candidates_by_time = module.build_profit_candidates(
        pullbacks_by_time=empty_pullbacks,
        route_signals_by_time=route_signals_by_time,
        frames_1h=frames_1h,
        common_1h=common_1h,
        market_states=market_states,
    )

    emitted_map: dict[tuple[pd.Timestamp, str, str], Any] = {}
    selection_score_map: dict[tuple[pd.Timestamp, str, str], float] = {}
    rank_map: dict[tuple[pd.Timestamp, str, str], int] = {}

    for timestamp, candidates in emitted_candidates_by_time.items():
        ranked = module.rank_valid_candidates(candidates, frames_1h, timestamp)
        for rank, (candidate, selection_score) in enumerate(ranked, start=1):
            key = (pd.Timestamp(timestamp), candidate.symbol, candidate.route)
            emitted_map[key] = candidate
            selection_score_map[key] = float(selection_score)
            rank_map[key] = rank

    print("Tracing first-true episodes and stale re-triggers...")

    raw_state: dict[tuple[str, str], dict[str, Any]] = defaultdict(dict)
    quality_state: dict[tuple[str, str], dict[str, Any]] = defaultdict(dict)
    raw_episode_counter = Counter()
    quality_episode_counter = Counter()
    quality_emission_counter = Counter()

    signal_rows: list[dict[str, Any]] = []
    episode_accumulator: dict[tuple[str, str, int], dict[str, Any]] = {}

    for decision_time in common_1h:
        decision_time = pd.Timestamp(decision_time)
        strength = module.relative_strength_percentiles(frames_1h, decision_time)

        for symbol in module.SYMBOLS:
            for route in ("IGNITION", "ADAPTIVE_BREAKOUT"):
                key = (symbol, route)
                signal = route_signals_by_time[decision_time][symbol][route]
                converted = module._convert_route_signal(
                    symbol=symbol,
                    route=route,
                    signal=signal,
                    frame=frames_1h[symbol],
                    decision_time=decision_time,
                    market_state=market_states[decision_time],
                    relative_strength=float(strength[symbol]),
                )

                raw_valid = bool(signal.valid)
                quality_valid = bool(converted.valid)

                previous_raw = bool(raw_state[key].get("valid", False))
                if raw_valid and not previous_raw:
                    raw_episode_counter[key] += 1
                    raw_state[key] = {
                        "valid": True,
                        "episode_id": int(raw_episode_counter[key]),
                        "first_time": decision_time,
                        "occurrence": 1,
                    }
                elif raw_valid:
                    raw_state[key]["occurrence"] = int(raw_state[key].get("occurrence", 0)) + 1
                else:
                    raw_state[key] = {"valid": False}

                previous_quality = bool(quality_state[key].get("valid", False))
                if quality_valid and not previous_quality:
                    quality_episode_counter[key] += 1
                    quality_state[key] = {
                        "valid": True,
                        "episode_id": int(quality_episode_counter[key]),
                        "first_time": decision_time,
                        "occurrence": 1,
                    }
                    quality_emission_counter[(symbol, route, int(quality_episode_counter[key]))] = 0
                elif quality_valid:
                    quality_state[key]["occurrence"] = int(quality_state[key].get("occurrence", 0)) + 1
                else:
                    quality_state[key] = {"valid": False}

                emitted_key = (decision_time, symbol, route)
                emitted = emitted_key in emitted_map

                quality_episode_id = (
                    int(quality_state[key].get("episode_id", 0))
                    if quality_valid
                    else 0
                )
                quality_first_time = (
                    pd.Timestamp(quality_state[key].get("first_time"))
                    if quality_valid
                    else pd.NaT
                )

                stale_retrigger = False
                emission_number = 0
                if emitted and quality_episode_id > 0:
                    counter_key = (symbol, route, quality_episode_id)
                    quality_emission_counter[counter_key] += 1
                    emission_number = int(quality_emission_counter[counter_key])
                    stale_retrigger = emission_number > 1

                if raw_valid or quality_valid or emitted:
                    raw_first_time = (
                        pd.Timestamp(raw_state[key].get("first_time"))
                        if raw_valid
                        else pd.NaT
                    )
                    candidate = emitted_map.get(emitted_key, converted)
                    signal_rows.append(
                        {
                            "decision_time": decision_time,
                            "source_candle_open_time": decision_time - pd.Timedelta(hours=1),
                            "order_fill_time": decision_time,
                            "signal_to_fill_minutes": 0.0,
                            "symbol": symbol,
                            "route": route,
                            "market_state": market_states[decision_time].state.value,
                            "relative_strength_percentile": round(float(strength[symbol]), 4),
                            "raw_signal_valid": raw_valid,
                            "raw_score": round(float(signal.score), 4),
                            "raw_failed_checks": _as_text(signal.failed_checks),
                            "raw_episode_id": int(raw_state[key].get("episode_id", 0)) if raw_valid else 0,
                            "raw_episode_first_true_time": raw_first_time,
                            "raw_condition_age_hours": (
                                (decision_time - raw_first_time).total_seconds() / 3600.0
                                if raw_valid and not pd.isna(raw_first_time)
                                else 0.0
                            ),
                            "quality_valid": quality_valid,
                            "quality_failed_checks": _as_text(converted.failed_checks),
                            "quality_episode_id": quality_episode_id,
                            "quality_episode_first_true_time": quality_first_time,
                            "quality_condition_age_hours": (
                                (decision_time - quality_first_time).total_seconds() / 3600.0
                                if quality_valid and not pd.isna(quality_first_time)
                                else 0.0
                            ),
                            "candidate_emitted_after_route_cooldown": emitted,
                            "emission_number_in_same_quality_episode": emission_number,
                            "stale_same_episode_retrigger": stale_retrigger,
                            "setup_score": round(float(candidate.score), 4),
                            "theoretical_reward_risk": round(float(candidate.reward_risk), 4),
                            "selection_rank": rank_map.get(emitted_key, 0),
                            "selection_score": round(selection_score_map.get(emitted_key, 0.0), 4),
                        }
                    )

                if quality_valid:
                    episode_key = (symbol, route, quality_episode_id)
                    episode = episode_accumulator.setdefault(
                        episode_key,
                        {
                            "symbol": symbol,
                            "route": route,
                            "quality_episode_id": quality_episode_id,
                            "first_true_time": quality_first_time,
                            "last_true_time": decision_time,
                            "valid_candle_count": 0,
                            "emission_count": 0,
                            "stale_retrigger_count": 0,
                        },
                    )
                    episode["last_true_time"] = decision_time
                    episode["valid_candle_count"] += 1
                    if emitted:
                        episode["emission_count"] += 1
                    if stale_retrigger:
                        episode["stale_retrigger_count"] += 1

    episode_rows = []
    for episode in episode_accumulator.values():
        first_time = pd.Timestamp(episode["first_true_time"])
        last_time = pd.Timestamp(episode["last_true_time"])
        episode_rows.append(
            {
                **episode,
                "episode_span_hours": round(
                    (last_time - first_time).total_seconds() / 3600.0,
                    4,
                ),
                "persistent_beyond_route_cooldown": (
                    (last_time - first_time).total_seconds() / 3600.0
                    >= float(module.ROUTE_SIGNAL_COOLDOWN_HOURS)
                ),
            }
        )

    print("Auditing emitted candidates against next-open execution and trade logs...")
    trades = _load_trades(config["trade_file"])

    actual_entry_keys: set[tuple[pd.Timestamp, str, str]] = set()
    actual_entries_by_time: dict[pd.Timestamp, list[tuple[str, str]]] = defaultdict(list)
    if not trades.empty:
        for _, trade in trades.iterrows():
            trade_time = pd.Timestamp(trade["entry_time"])
            route = str(trade.get("route", ""))
            symbol = str(trade["symbol"])
            actual_entry_keys.add((trade_time, symbol, route))
            actual_entries_by_time[trade_time].append((symbol, route))

    next_open_map: dict[tuple[str, pd.Timestamp], tuple[pd.Timestamp, float]] = {}
    for symbol in module.SYMBOLS:
        frame = frames_1h[symbol]
        for index in range(len(common_1h) - 1):
            decision_time = pd.Timestamp(common_1h[index])
            next_completion = pd.Timestamp(common_1h[index + 1])
            # The next candle opens exactly at decision_time because the frame
            # index is candle completion_time.
            next_open_map[(symbol, decision_time)] = (
                decision_time,
                float(frame.loc[next_completion, "open"]),
            )

    candidate_rows: list[dict[str, Any]] = []
    reject_counts = Counter()

    fee_pct = float(module.TRADING_FEE_PERCENT)
    slippage_pct = float(module.SLIPPAGE_PERCENT)
    estimated_round_trip_cost_pct = 2.0 * (fee_pct + slippage_pct)

    for (decision_time, symbol, route), candidate in sorted(emitted_map.items()):
        fill_time, raw_open = next_open_map.get(
            (symbol, decision_time),
            (pd.NaT, float("nan")),
        )

        actual_entry = (decision_time, symbol, route) in actual_entry_keys
        rejection_reason = "EXECUTED" if actual_entry else ""
        slipped_entry = float("nan")
        actual_rr = 0.0
        gap_atr = 0.0

        signal_atr = float(frames_1h[symbol].loc[decision_time, "ATR14"])
        if math.isfinite(raw_open):
            slipped_entry = float(module.apply_buy_slippage(raw_open))
            gap_atr = (
                (slipped_entry - float(candidate.entry_price)) / signal_atr
                if signal_atr > 0
                else 0.0
            )
            risk_per_unit = slipped_entry - float(candidate.stop_price)
            projected_reward = float(candidate.projected_target) - slipped_entry
            actual_rr = projected_reward / risk_per_unit if risk_per_unit > 0 else 0.0

        if not actual_entry:
            market_state = market_states[decision_time]
            market_shock = market_shocks[decision_time]
            active = _active_trades_before(trades, decision_time)
            active_symbols = set(active["symbol"].astype(str)) if not active.empty else set()

            if pd.isna(fill_time):
                rejection_reason = "NO_NEXT_HOUR_OPEN"
            elif market_shock.freeze_all_entries:
                rejection_reason = "MARKET_SHOCK_ENTRY_FREEZE"
            elif not market_state.new_entries_allowed:
                rejection_reason = "MARKET_STATE_BLOCK"
            elif symbol in active_symbols:
                rejection_reason = "EXISTING_OPEN_POSITION"
            else:
                cooldown = _cooldown_reason(
                    trades,
                    symbol,
                    decision_time,
                    int(module.NORMAL_EXIT_COOLDOWN_HOURS),
                    int(module.DEFENSIVE_EXIT_COOLDOWN_HOURS),
                )
                if cooldown:
                    rejection_reason = cooldown
                elif len(active_symbols) >= int(module.MAX_OPEN_POSITIONS):
                    rejection_reason = "MAX_OPEN_POSITIONS"
                elif (
                    symbol in HIGH_BETA_ASSETS
                    and any(item in HIGH_BETA_ASSETS for item in active_symbols)
                ):
                    rejection_reason = "HIGH_BETA_CORRELATION_BUCKET"
                elif math.isfinite(slipped_entry) and (
                    slipped_entry
                    > float(candidate.entry_price) + signal_atr * 0.35
                ):
                    rejection_reason = "OPENING_GAP_ABOVE_0_35_ATR"
                elif actual_rr < 1.40:
                    rejection_reason = "POST_GAP_RR_BELOW_1_40"
                elif actual_entries_by_time.get(decision_time):
                    rejection_reason = "LOWER_RANK_THAN_EXECUTED_CANDIDATE"
                elif trades.empty:
                    rejection_reason = "TRADE_LOG_NOT_FOUND"
                else:
                    rejection_reason = "PORTFOLIO_OR_ACCOUNT_LIMIT_NOT_RECORDED"

        reject_counts[rejection_reason] += 1

        gross_target_move_pct = (
            (float(candidate.projected_target) - slipped_entry)
            / slipped_entry
            * 100.0
            if math.isfinite(slipped_entry) and slipped_entry > 0
            else 0.0
        )
        cost_share = (
            estimated_round_trip_cost_pct / gross_target_move_pct * 100.0
            if gross_target_move_pct > 0
            else 0.0
        )

        matching_signal = next(
            (
                row
                for row in signal_rows
                if row["decision_time"] == decision_time
                and row["symbol"] == symbol
                and row["route"] == route
                and row["candidate_emitted_after_route_cooldown"]
            ),
            None,
        )

        candidate_rows.append(
            {
                "signal_first_true_time": (
                    matching_signal["quality_episode_first_true_time"]
                    if matching_signal
                    else pd.NaT
                ),
                "signal_evaluated_time": decision_time,
                "order_fill_time": fill_time,
                "signal_first_true_to_fill_hours": (
                    (pd.Timestamp(fill_time) - pd.Timestamp(matching_signal["quality_episode_first_true_time"])).total_seconds() / 3600.0
                    if matching_signal
                    and not pd.isna(matching_signal["quality_episode_first_true_time"])
                    and not pd.isna(fill_time)
                    else 0.0
                ),
                "signal_evaluation_to_fill_minutes": 0.0 if not pd.isna(fill_time) else 0.0,
                "symbol": symbol,
                "route": route,
                "quality_episode_id": matching_signal["quality_episode_id"] if matching_signal else 0,
                "emission_number_in_same_episode": matching_signal["emission_number_in_same_quality_episode"] if matching_signal else 0,
                "stale_same_episode_retrigger": matching_signal["stale_same_episode_retrigger"] if matching_signal else False,
                "selection_rank": rank_map.get((decision_time, symbol, route), 0),
                "selection_score": round(selection_score_map.get((decision_time, symbol, route), 0.0), 4),
                "raw_signal_close": round(float(candidate.entry_price), 8),
                "raw_next_candle_open": round(raw_open, 8) if math.isfinite(raw_open) else "",
                "slipped_entry_price": round(slipped_entry, 8) if math.isfinite(slipped_entry) else "",
                "opening_gap_atr": round(gap_atr, 6),
                "theoretical_rr_at_signal_close": round(float(candidate.reward_risk), 6),
                "actual_rr_after_gap_and_slippage": round(actual_rr, 6),
                "estimated_round_trip_cost_percent": round(estimated_round_trip_cost_pct, 6),
                "gross_projected_target_move_percent": round(gross_target_move_pct, 6),
                "cost_as_percent_of_projected_target_move": round(cost_share, 4),
                "actual_trade_entered": actual_entry,
                "entry_or_reject_reason": rejection_reason,
            }
        )

    print("Calculating cross-symbol correlation exposure...")
    hourly_returns = pd.DataFrame(
        {
            symbol: frames_1h[symbol].reindex(common_1h)["close"].pct_change()
            for symbol in module.SYMBOLS
        }
    ).dropna(how="all")
    correlation_matrix = hourly_returns.corr()

    collision_rows = []
    emitted_by_time: dict[pd.Timestamp, list[str]] = defaultdict(list)
    for timestamp, symbol, route in emitted_map:
        emitted_by_time[timestamp].append(symbol)

    for timestamp, symbols in emitted_by_time.items():
        unique_symbols = sorted(set(symbols))
        if len(unique_symbols) < 2:
            continue

        window = hourly_returns.loc[:timestamp].tail(72)
        corr = window.corr()
        for left_index, left in enumerate(unique_symbols):
            for right in unique_symbols[left_index + 1 :]:
                value = _safe_float(corr.loc[left, right], 0.0)
                collision_rows.append(
                    {
                        "decision_time": timestamp,
                        "symbol_1": left,
                        "symbol_2": right,
                        "rolling_72h_correlation": round(value, 6),
                        "high_correlation_above_0_75": value >= 0.75,
                    }
                )

    signal_df = pd.DataFrame(signal_rows)
    episode_df = pd.DataFrame(episode_rows)
    candidate_df = pd.DataFrame(candidate_rows)
    collision_df = pd.DataFrame(collision_rows)

    stale_count = int(candidate_df["stale_same_episode_retrigger"].sum()) if not candidate_df.empty else 0
    emitted_count = int(len(candidate_df))
    executed_count = int(candidate_df["actual_trade_entered"].sum()) if not candidate_df.empty else 0
    persistent_episode_count = int(episode_df["persistent_beyond_route_cooldown"].sum()) if not episode_df.empty else 0
    multi_emission_episodes = int((episode_df["emission_count"] > 1).sum()) if not episode_df.empty else 0

    trade_metrics = _trade_summary(trades)

    summary: dict[str, Any] = {
        "period": period,
        "source_module": config["module"],
        "backtest_start": module.BACKTEST_START,
        "backtest_end": module.BACKTEST_END,
        "route_signal_cooldown_hours": float(module.ROUTE_SIGNAL_COOLDOWN_HOURS),
        "raw_valid_signal_rows": int(signal_df["raw_signal_valid"].sum()) if not signal_df.empty else 0,
        "quality_valid_signal_rows": int(signal_df["quality_valid"].sum()) if not signal_df.empty else 0,
        "quality_signal_episodes": int(len(episode_df)),
        "episodes_persistent_beyond_route_cooldown": persistent_episode_count,
        "episodes_with_multiple_emissions": multi_emission_episodes,
        "emitted_candidates": emitted_count,
        "stale_same_episode_retriggers": stale_count,
        "stale_retrigger_percent_of_candidates": round(stale_count / emitted_count * 100.0, 4) if emitted_count else 0.0,
        "executed_candidates_matched_in_trade_log": executed_count,
        "signal_evaluation_to_fill_minutes": 0.0,
        "estimated_round_trip_fee_plus_slippage_percent": round(estimated_round_trip_cost_pct, 6),
        "simultaneous_multi_symbol_candidate_hours": int(len(set(collision_df["decision_time"]))) if not collision_df.empty else 0,
        "high_correlation_candidate_pairs": int(collision_df["high_correlation_above_0_75"].sum()) if not collision_df.empty else 0,
        **trade_metrics,
    }

    reports = Path("reports")
    reports.mkdir(parents=True, exist_ok=True)
    prefix = reports / f"v3f1_{period}"

    signal_path = Path(f"{prefix}_signal_events.csv")
    episode_path = Path(f"{prefix}_signal_episodes.csv")
    candidate_path = Path(f"{prefix}_candidate_execution_audit.csv")
    collision_path = Path(f"{prefix}_candidate_correlation_collisions.csv")
    correlation_path = Path(f"{prefix}_hourly_return_correlation_matrix.csv")
    trade_summary_path = Path(f"{prefix}_trade_outcome_summary.csv")
    summary_path = Path(f"{prefix}_summary.csv")

    signal_df.to_csv(signal_path, index=False)
    episode_df.to_csv(episode_path, index=False)
    candidate_df.to_csv(candidate_path, index=False)
    collision_df.to_csv(collision_path, index=False)
    correlation_matrix.to_csv(correlation_path)
    pd.DataFrame([trade_metrics]).to_csv(trade_summary_path, index=False)
    pd.DataFrame([summary]).to_csv(summary_path, index=False)

    print()
    print("=" * 140)
    print("V3F1 SIGNAL INTEGRITY RESULT")
    print("=" * 140)
    print(f"Raw valid signal rows             : {summary['raw_valid_signal_rows']}")
    print(f"Quality-valid signal rows         : {summary['quality_valid_signal_rows']}")
    print(f"Quality signal episodes           : {summary['quality_signal_episodes']}")
    print(f"Persistent beyond 12h cooldown    : {persistent_episode_count}")
    print(f"Episodes with multiple emissions  : {multi_emission_episodes}")
    print(f"Emitted candidates                : {emitted_count}")
    print(f"Stale same-episode re-triggers     : {stale_count}")
    print(f"Matched executed entries          : {executed_count}")
    print(f"Signal evaluation -> fill delay   : 0 minutes (next candle opens at the completed-candle boundary)")
    print(f"Estimated round-trip cost hurdle  : {estimated_round_trip_cost_pct:.4f}%")
    print(f"Multi-symbol candidate hours      : {summary['simultaneous_multi_symbol_candidate_hours']}")
    print(f"High-correlation candidate pairs  : {summary['high_correlation_candidate_pairs']}")

    if trade_metrics.get("trade_log_found"):
        print("-" * 140)
        print(f"Completed trades                  : {trade_metrics.get('completed_trades', 0)}")
        print(f"Average win / loss                : €{trade_metrics.get('average_win_eur', 0):.2f} / €{trade_metrics.get('average_loss_eur', 0):.2f}")
        print(f"Realized win/loss ratio           : {trade_metrics.get('realized_win_loss_ratio', 0):.2f}")
        print(f"Losing stop-exit rate             : {trade_metrics.get('losing_stop_exit_percent', 0):.2f}%")
        print(f"Winning full-TP rate              : {trade_metrics.get('winning_full_take_profit_percent', 0):.2f}%")
        print(f"Explicit fees / gross profit      : {trade_metrics.get('fees_as_percent_of_gross_profit', 0):.2f}%")
    else:
        print("Trade log was not found; signal diagnostics still completed.")

    print("-" * 140)
    print("ENTRY / REJECT REASONS")
    for reason, count in reject_counts.most_common():
        print(f"{reason:<55}: {count:>6}")

    print("=" * 140)
    print(f"Signal events       : {signal_path}")
    print(f"Signal episodes     : {episode_path}")
    print(f"Candidate audit     : {candidate_path}")
    print(f"Correlation pairs   : {collision_path}")
    print(f"Correlation matrix  : {correlation_path}")
    print(f"Trade outcome audit : {trade_summary_path}")
    print(f"Summary             : {summary_path}")
    print("=" * 140)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="V3F1 signal timing, stale-trigger, execution and cost diagnostic."
    )
    parser.add_argument(
        "--period",
        choices=sorted(PERIOD_CONFIG),
        default="development",
        help="Chronological test period to diagnose.",
    )
    args = parser.parse_args()
    run_diagnostic(args.period)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nV3F1 diagnostic stopped manually.")
    except Exception as error:
        print()
        print("=" * 140)
        print("V3F1 SIGNAL INTEGRITY ERROR")
        print("=" * 140)
        print(f"{type(error).__name__}: {error}")
        print("=" * 140)
        raise
