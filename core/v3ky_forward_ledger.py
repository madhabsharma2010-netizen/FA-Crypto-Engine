from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


CONFIG_PATH = Path(
    "config/v3ky_epoch_001.json"
)

LEDGER_PATH = Path(
    "state/v3ky/v3ky_001_forward_ledger.jsonl"
)

GENESIS_HASH = "GENESIS"


RECORD_TYPES = {
    "DECISION",
    "ENTRY",
    "EXIT",
    "EQUITY",
    "RISK_LOCK",
    "DATA_FAILURE",
    "WEEK_CLOSE",
    "HEARTBEAT",
}


PAYLOAD_FIELDS = [
    "symbol",
    "signal_state",
    "signal_timestamp",
    "pending_entry_state",
    "entry_decision",
    "entry_execution_timestamp",
    "entry_execution_price",
    "quantity",
    "initial_stop",
    "active_stop",
    "initial_risk_eur",
    "exit_decision",
    "exit_reason",
    "exit_execution_timestamp",
    "exit_execution_price",
    "fees_eur",
    "slippage_eur",
    "realized_pnl_eur",
    "realized_net_r",
    "cash_eur",
    "equity_eur",
    "open_positions",
    "open_risk_eur",
    "daily_lock_state",
    "weekly_lock_state",
    "hard_drawdown_lock_state",
    "data_health_state",
    "notes",
]


def parse_utc(value: str) -> datetime:
    text = str(value).strip()

    if text.endswith("Z"):
        text = (
            text[:-1]
            + "+00:00"
        )

    result = datetime.fromisoformat(
        text
    )

    if result.tzinfo is None:
        raise RuntimeError(
            f"TIMESTAMP_NOT_TIMEZONE_AWARE: {value}"
        )

    return result.astimezone(
        timezone.utc
    )


def utc_now_text() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        raise RuntimeError(
            f"CONFIG_MISSING: {CONFIG_PATH}"
        )

    config = json.loads(
        CONFIG_PATH.read_text(
            encoding="utf-8"
        )
    )

    if config["epoch_id"] != "V3KY-001":
        raise RuntimeError(
            "EPOCH_ID_MISMATCH"
        )

    if config["live_orders_authorized"]:
        raise RuntimeError(
            "LIVE_ORDER_FLAG_MUST_BE_FALSE"
        )

    if config["historical_backfill_allowed"]:
        raise RuntimeError(
            "BACKFILL_FLAG_MUST_BE_FALSE"
        )

    if config["retuning_allowed"]:
        raise RuntimeError(
            "RETUNING_FLAG_MUST_BE_FALSE"
        )

    return config


def config_sha256() -> str:
    return hashlib.sha256(
        CONFIG_PATH.read_bytes()
    ).hexdigest()


def record_hash(
    record_without_hash: dict,
) -> str:

    canonical = json.dumps(
        record_without_hash,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )

    return hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()


def initialize_ledger() -> None:
    config = load_config()

    LEDGER_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not LEDGER_PATH.exists():
        LEDGER_PATH.touch(
            exist_ok=False
        )

    count, last_hash = verify_ledger()

    print(
        "V3KY ledger initialization: PASS"
    )

    print(
        f"Epoch ID       : {config['epoch_id']}"
    )

    print(
        f"Config SHA256  : {config_sha256()}"
    )

    print(
        f"Ledger path    : {LEDGER_PATH}"
    )

    print(
        f"Ledger records : {count}"
    )

    print(
        f"Last hash      : {last_hash}"
    )


def verify_ledger() -> tuple[int, str]:
    config = load_config()

    if not LEDGER_PATH.exists():
        raise RuntimeError(
            f"LEDGER_MISSING: {LEDGER_PATH}"
        )

    expected_config_hash = (
        config_sha256()
    )

    forward_start = parse_utc(
        config["forward_start_utc"]
    )

    previous_hash = GENESIS_HASH
    expected_sequence = 1
    count = 0

    with LEDGER_PATH.open(
        "r",
        encoding="utf-8",
    ) as handle:

        for line_number, raw in enumerate(
            handle,
            start=1,
        ):
            raw = raw.strip()

            if not raw:
                continue

            record = json.loads(
                raw
            )

            if (
                record["sequence"]
                != expected_sequence
            ):
                raise RuntimeError(
                    "LEDGER_SEQUENCE_FAILURE: "
                    f"line={line_number}"
                )

            if (
                record["epoch_id"]
                != config["epoch_id"]
            ):
                raise RuntimeError(
                    "LEDGER_EPOCH_FAILURE: "
                    f"line={line_number}"
                )

            if (
                record["epoch_config_sha256"]
                != expected_config_hash
            ):
                raise RuntimeError(
                    "LEDGER_CONFIG_HASH_FAILURE: "
                    f"line={line_number}"
                )

            event_time = parse_utc(
                record[
                    "event_timestamp"
                ]
            )

            if event_time <= forward_start:
                raise RuntimeError(
                    "PRE_START_FORWARD_RECORD: "
                    f"line={line_number}"
                )

            if (
                record["previous_record_hash"]
                != previous_hash
            ):
                raise RuntimeError(
                    "LEDGER_CHAIN_FAILURE: "
                    f"line={line_number}"
                )

            claimed_hash = record[
                "record_hash"
            ]

            check_record = dict(
                record
            )

            check_record.pop(
                "record_hash"
            )

            calculated_hash = (
                record_hash(
                    check_record
                )
            )

            if claimed_hash != calculated_hash:
                raise RuntimeError(
                    "LEDGER_RECORD_HASH_FAILURE: "
                    f"line={line_number}"
                )

            previous_hash = (
                claimed_hash
            )

            expected_sequence += 1
            count += 1

    return count, previous_hash


def append_record(
    payload: dict,
) -> dict:

    config = load_config()

    count, previous_hash = (
        verify_ledger()
    )

    record_type = payload.get(
        "record_type"
    )

    if record_type not in RECORD_TYPES:
        raise RuntimeError(
            f"INVALID_RECORD_TYPE: {record_type}"
        )

    event_timestamp = payload.get(
        "event_timestamp"
    )

    if not event_timestamp:
        raise RuntimeError(
            "EVENT_TIMESTAMP_REQUIRED"
        )

    event_time = parse_utc(
        event_timestamp
    )

    forward_start = parse_utc(
        config["forward_start_utc"]
    )

    if event_time <= forward_start:
        raise RuntimeError(
            "FORWARD_RECORD_BEFORE_EPOCH_START"
        )

    record = {
        "sequence":
            count + 1,

        "recorded_at_utc":
            utc_now_text(),

        "epoch_id":
            config["epoch_id"],

        "epoch_config_sha256":
            config_sha256(),

        "record_type":
            record_type,

        "event_timestamp":
            event_timestamp,
    }

    for field in PAYLOAD_FIELDS:
        record[field] = payload.get(
            field
        )

    record[
        "previous_record_hash"
    ] = previous_hash

    record["record_hash"] = (
        record_hash(
            record
        )
    )

    with LEDGER_PATH.open(
        "a",
        encoding="utf-8",
    ) as handle:

        handle.write(
            json.dumps(
                record,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            )
            + "\n"
        )

    verify_ledger()

    return record


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--init",
        action="store_true",
    )

    parser.add_argument(
        "--verify",
        action="store_true",
    )

    args = parser.parse_args()

    if args.init:
        initialize_ledger()
        return

    if args.verify:
        count, last_hash = (
            verify_ledger()
        )

        print(
            "V3KY ledger verification: PASS"
        )

        print(
            f"Ledger records : {count}"
        )

        print(
            f"Last hash      : {last_hash}"
        )

        return

    parser.error(
        "Use --init or --verify"
    )


if __name__ == "__main__":
    main()
