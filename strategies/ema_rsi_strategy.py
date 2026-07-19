from typing import Any


class EmaRsiStrategy:
    """
    Original V1 EMA and RSI strategy.
    Is file ko benchmark comparison ke liye unchanged rakho.
    """

    @staticmethod
    def generate_signal(candle: Any) -> str:
        required_fields = ("EMA20", "EMA50", "RSI14")

        for field in required_fields:
            if field not in candle or candle[field] is None:
                return "HOLD"

        ema20 = float(candle["EMA20"])
        ema50 = float(candle["EMA50"])
        rsi = float(candle["RSI14"])

        if ema20 > ema50 and rsi < 70:
            return "BUY"

        if ema20 < ema50 and rsi > 30:
            return "SELL"

        return "HOLD"
    