class EmaRsiStrategy:

    @staticmethod
    def generate_signal(candle):

        ema20 = candle["EMA20"]
        ema50 = candle["EMA50"]
        rsi = candle["RSI14"]

        if ema20 > ema50 and rsi < 70:
            return "BUY"

        elif ema20 < ema50 and rsi > 30:
            return "SELL"

        return "HOLD"
    