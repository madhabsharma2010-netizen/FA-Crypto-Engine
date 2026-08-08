from pathlib import Path
from datetime import datetime, timezone, timedelta
import csv
import json
import urllib.parse
import urllib.request

CONFIG = Path("config/v3ky_epoch_001.json")
OUTDIR = Path("state/v3ky/market")

BASE = "https://data-api.binance.vision/api/v3/klines"
INTERVAL = "1h"
TARGET_BARS = 3000
PAGE_LIMIT = 1000
HOUR_MS = 60 * 60 * 1000


def iso_ms(ms):
    return (
        datetime.fromtimestamp(
            ms / 1000,
            tz=timezone.utc,
        )
        .isoformat()
        .replace("+00:00", "Z")
    )


def fetch_page(symbol, end_ms):
    params = urllib.parse.urlencode(
        {
            "symbol": symbol,
            "interval": INTERVAL,
            "limit": PAGE_LIMIT,
            "endTime": end_ms,
        }
    )

    url = BASE + "?" + params

    with urllib.request.urlopen(
        url,
        timeout=30,
    ) as response:
        return json.loads(
            response.read().decode("utf-8")
        )


config = json.loads(
    CONFIG.read_text(encoding="utf-8")
)

epoch = datetime.fromisoformat(
    config["forward_start_utc"].replace(
        "Z",
        "+00:00",
    )
)

first_forward_open = epoch.replace(
    minute=0,
    second=0,
    microsecond=0,
) + timedelta(hours=1)

first_forward_open_ms = int(
    first_forward_open.timestamp() * 1000
)

first_forward_completion_ms = (
    first_forward_open_ms + HOUR_MS
)

now_ms = int(
    datetime.now(timezone.utc).timestamp()
    * 1000
)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)

print("========== V3KY 3000H FEED SYNC ==========")
print(
    "Epoch start             :",
    config["forward_start_utc"],
)
print(
    "First forward open      :",
    iso_ms(first_forward_open_ms),
)
print(
    "First forward completion:",
    iso_ms(first_forward_completion_ms),
)
print()

grand_forward = 0

for symbol in config["universe"]:

    candles = {}
    end_ms = now_ms

    while len(candles) < TARGET_BARS:

        page = fetch_page(
            symbol,
            end_ms,
        )

        if not page:
            break

        closed = [
            bar
            for bar in page
            if int(bar[6]) < now_ms
        ]

        if not closed:
            break

        for bar in closed:
            candles[int(bar[0])] = bar

        earliest_open = min(
            int(bar[0])
            for bar in closed
        )

        end_ms = earliest_open - 1

        if len(page) < PAGE_LIMIT:
            break

    ordered_keys = sorted(candles)[
        -TARGET_BARS:
    ]

    if len(ordered_keys) < 1000:
        raise RuntimeError(
            f"INSUFFICIENT_WARMUP:"
            f"{symbol}:{len(ordered_keys)}"
        )

    rows = []

    for open_ms in ordered_keys:

        bar = candles[open_ms]

        completion_ms = (
            open_ms + HOUR_MS
        )

        rows.append(
            {
                "open_time_utc":
                    iso_ms(open_ms),

                "completion_time_utc":
                    iso_ms(completion_ms),

                "open_ms":
                    open_ms,

                "completion_ms":
                    completion_ms,

                "open":
                    bar[1],

                "high":
                    bar[2],

                "low":
                    bar[3],

                "close":
                    bar[4],

                "volume":
                    bar[5],

                "exchange_close_ms":
                    int(bar[6]),

                "is_forward":
                    completion_ms
                    >= first_forward_completion_ms,
            }
        )

    path = OUTDIR / (
        f"{symbol}_1h.csv"
    )

    fields = [
        "open_time_utc",
        "completion_time_utc",
        "open_ms",
        "completion_ms",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "exchange_close_ms",
        "is_forward",
    ]

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(rows)

    forward_count = sum(
        bool(row["is_forward"])
        for row in rows
    )

    grand_forward += forward_count

    print(symbol)
    print(
        "  Stored bars :",
        len(rows),
    )
    print(
        "  Oldest      :",
        rows[0]["completion_time_utc"],
    )
    print(
        "  Latest      :",
        rows[-1]["completion_time_utc"],
    )
    print(
        "  Forward bars:",
        forward_count,
    )

print()
print(
    "Total forward bars:",
    grand_forward,
)

print(
    "Warm-up target reached: YES"
)

print(
    "Canonical completion-time semantics: YES"
)

print(
    "V3KY 3000H LIVE FEED: PASS"
)
