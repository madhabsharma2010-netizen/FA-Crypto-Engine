from __future__ import annotations

import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_STATE_FILE = Path(
    "state/v3kj_risk_guard.json"
)

HARD_DRAWDOWN_LIMIT_PERCENT = 5.0


def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def atomic_write(
    path: Path,
    payload: dict[str, Any],
) -> None:
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    file_descriptor, temporary_name = (
        tempfile.mkstemp(
            prefix=f"{path.name}.",
            suffix=".tmp",
            dir=str(path.parent),
        )
    )

    temporary_path = Path(
        temporary_name
    )

    try:
        with os.fdopen(
            file_descriptor,
            "w",
            encoding="utf-8",
        ) as output:
            json.dump(
                payload,
                output,
                indent=2,
                sort_keys=True,
            )

            output.flush()
            os.fsync(
                output.fileno()
            )

        last_error = None

        for attempt in range(10):
            try:
                os.replace(
                    temporary_path,
                    path,
                )
                return

            except PermissionError as error:
                last_error = error

                time.sleep(
                    0.05
                    * (attempt + 1)
                )

        if last_error is not None:
            raise last_error

    finally:
        if temporary_path.exists():
            try:
                temporary_path.unlink()
            except OSError:
                pass


def load_or_create_state(
    state_file: Path,
    starting_capital: float,
) -> dict[str, Any]:

    if state_file.exists():
        with state_file.open(
            "r",
            encoding="utf-8",
        ) as source:
            state = json.load(source)

        required_fields = {
            "version",
            "peak_equity",
            "last_equity",
            "hard_lock",
            "hard_lock_reason",
            "hard_lock_time_utc",
        }

        missing = (
            required_fields
            - set(state)
        )

        if missing:
            raise RuntimeError(
                "Risk-state file is missing: "
                + ", ".join(
                    sorted(missing)
                )
            )

        return state

    capital = float(
        starting_capital
    )

    if capital <= 0:
        raise ValueError(
            "Starting capital must be positive."
        )

    state = {
        "version": 1,
        "created_time_utc": utc_now(),
        "updated_time_utc": utc_now(),
        "peak_equity": capital,
        "last_equity": capital,
        "last_drawdown_percent": 0.0,
        "hard_lock": False,
        "hard_lock_reason": None,
        "hard_lock_time_utc": None,
    }

    atomic_write(
        state_file,
        state,
    )

    return state


def update_equity_and_lock(
    state_file: Path,
    starting_capital: float,
    current_equity: float,
    drawdown_limit_percent: float = (
        HARD_DRAWDOWN_LIMIT_PERCENT
    ),
) -> dict[str, Any]:

    equity = float(
        current_equity
    )

    if equity < 0:
        raise ValueError(
            "Current equity cannot be negative."
        )

    state = load_or_create_state(
        state_file,
        starting_capital,
    )

    previous_peak = float(
        state["peak_equity"]
    )

    peak_equity = max(
        previous_peak,
        equity,
    )

    drawdown_percent = (
        (
            peak_equity
            - equity
        )
        / peak_equity
        * 100.0
        if peak_equity > 0
        else 0.0
    )

    state["peak_equity"] = (
        peak_equity
    )

    state["last_equity"] = equity

    state[
        "last_drawdown_percent"
    ] = drawdown_percent

    state["updated_time_utc"] = (
        utc_now()
    )

    if (
        not bool(state["hard_lock"])
        and drawdown_percent
        >= drawdown_limit_percent
    ):
        state["hard_lock"] = True

        state["hard_lock_reason"] = (
            f"Persistent hard drawdown lock: "
            f"{drawdown_percent:.4f}% >= "
            f"{drawdown_limit_percent:.4f}%"
        )

        state["hard_lock_time_utc"] = (
            utc_now()
        )

    atomic_write(
        state_file,
        state,
    )

    return state


def assert_trading_allowed(
    state_file: Path,
    starting_capital: float,
) -> None:

    state = load_or_create_state(
        state_file,
        starting_capital,
    )

    if bool(state["hard_lock"]):
        raise RuntimeError(
            "TRADING REFUSED: "
            + str(
                state["hard_lock_reason"]
            )
        )


def self_test() -> None:
    test_file = Path(
        "state/"
        "v3kj_risk_guard_selftest.json"
    )

    if test_file.exists():
        test_file.unlink()

    starting_capital = 10000.0

    update_equity_and_lock(
        test_file,
        starting_capital,
        10000.0,
    )

    peak_state = (
        update_equity_and_lock(
            test_file,
            starting_capital,
            11000.0,
        )
    )

    assert (
        peak_state["peak_equity"]
        == 11000.0
    )

    locked_state = (
        update_equity_and_lock(
            test_file,
            starting_capital,
            10450.0,
        )
    )

    assert (
        locked_state["hard_lock"]
        is True
    )

    reloaded_state = (
        load_or_create_state(
            test_file,
            starting_capital,
        )
    )

    assert (
        reloaded_state["hard_lock"]
        is True
    )

    assert (
        reloaded_state["peak_equity"]
        == 11000.0
    )

    refused = False

    try:
        assert_trading_allowed(
            test_file,
            starting_capital,
        )
    except RuntimeError:
        refused = True

    assert refused is True

    print()
    print("=" * 76)
    print(
        "V3KJ PERSISTENT RISK-GUARD SELF-TEST"
    )
    print("=" * 76)
    print(
        "Peak equity persisted       : PASS"
    )
    print(
        "5% hard lock triggered      : PASS"
    )
    print(
        "Lock survived reload        : PASS"
    )
    print(
        "Restart trading refused     : PASS"
    )
    print(
        "Automatic unlock available  : NO"
    )
    print("=" * 76)

    test_file.unlink()


if __name__ == "__main__":
    self_test()
