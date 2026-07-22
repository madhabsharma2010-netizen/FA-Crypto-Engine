from typing import Any


class EmaRsiStrategyV2:
    """
    EMA + RSI + ADX + Volume strategy (V2A)
    """

    @staticmethod
    def generate_signal(candle: Any) -> str:

        required = (
            "close",
            "volume",
            "EMA20",
            "EMA50",
            "RSI14",
            "ADX14",
            "VolumeSMA20",
        )

        for field in required:
            if field not in candle:
                return "HOLD"

        close = float(candle["close"])
        volume = float(candle["volume"])
        ema20 = float(candle["EMA20"])
        ema50 = float(candle["EMA50"])
        rsi = float(candle["RSI14"])
        adx = float(candle["ADX14"])
        volume_sma = float(candle["VolumeSMA20"])

        if (
            ema20 > ema50
            and close > ema20
            and rsi < 70
            and adx >= 20
            and volume > volume_sma
        ):
            return "BUY"

        if ema20 < ema50 and rsi > 30:
            return "SELL"

        return "HOLD"
    