class Portfolio:

    def __init__(self, starting_balance=10000):

        self.balance = float(starting_balance)

        self.position = False

        self.buy_price = 0.0

        self.sell_price = 0.0

        self.total_profit = 0.0

        self.total_trades = 0

    def buy(self, price):

        if self.position:
            return

        self.buy_price = price
        self.position = True
        self.total_trades += 1

        print(f"BUY  @ {price:.2f}")

    def sell(self, price):

        if not self.position:
            return

        self.sell_price = price

        profit = price - self.buy_price

        self.total_profit += profit

        self.balance += profit

        self.position = False

        print(f"SELL @ {price:.2f}")
        print(f"PROFIT : {profit:.2f}")

    def summary(self):

        print("=" * 80)
        print("PORTFOLIO SUMMARY")
        print("=" * 80)
        print(f"Balance : €{self.balance:.2f}")
        print(f"Profit  : €{self.total_profit:.2f}")
        print(f"Trades  : {self.total_trades}")
        print("=" * 80)
