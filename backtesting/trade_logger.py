class TradeLogger:

    def __init__(self):

        self.trades = []

    def add_trade(self, buy_price, sell_price):

        profit = sell_price - buy_price

        self.trades.append(
            {
                "Buy": buy_price,
                "Sell": sell_price,
                "Profit": profit,
            }
        )

    def show(self):

        print("=" * 80)
        print("TRADE HISTORY")
        print("=" * 80)

        for trade in self.trades:
            print(trade)
            