from strategies.ema_rsi_strategy import EmaRsiStrategy


class SignalEngine:

    @staticmethod
    def generate(candle):

        return EmaRsiStrategy.generate_signal(candle)