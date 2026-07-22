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
        import csv

...

    def export_csv(self, filename="trade_history.csv"):

        with open(filename, "w", newline="") as file:

            writer = csv.DictWriter(
                file,
                fieldnames=["Buy", "Sell", "Profit"]
            )

            writer.writeheader()

            writer.writerows(self.trades)

        print(f"Trade history saved to {filename}")
        