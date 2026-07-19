class Performance:

    @staticmethod
    def report(portfolio):

        print("=" * 80)
        print("PERFORMANCE REPORT")
        print("=" * 80)

        print(f"Final Balance : €{portfolio.balance:.2f}")
        print(f"Total Profit  : €{portfolio.total_profit:.2f}")
        print(f"Total Trades  : {portfolio.total_trades}")

        print("=" * 80)
