import os
import time
from datetime import datetime

from binance.client import Client


SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "XRPUSDT",
    "SOLUSDT",
    "LINKUSDT",
    "POLUSDT",
    "DOGEUSDT",
    "PEPEUSDT",
     
]

REFRESH_SECONDS = 5

client = Client()


def clear_screen():
    """Terminal ka purana output clear karta hai."""
    os.system("cls" if os.name == "nt" else "clear")


def get_live_prices():
    """Binance se selected coins ki current prices fetch karta hai."""
    prices = {}

    for symbol in SYMBOLS:
        ticker = client.get_symbol_ticker(symbol=symbol)
        prices[symbol] = float(ticker["price"])

    return prices


def display_prices(prices):
    """Prices ko clean format mein terminal par display karta hai."""
    clear_screen()

    print("=" * 42)
    print(" FA CRYPTO ENGINE")
    print("=" * 42)
    print(f"Updated: {datetime.now():%d-%m-%Y %H:%M:%S}")
    print("-" * 42)

    for symbol, price in prices.items():
        coin = symbol.replace("USDT", "")
        print(f"{coin:<8} ${price:>15,.8f}")

    print("-" * 42)
    print(f"Refresh every {REFRESH_SECONDS} seconds")
    print("Press Ctrl + C to stop")
    print("=" * 42)


def main():
    while True:
        try:
            prices = get_live_prices()
            display_prices(prices)
            time.sleep(REFRESH_SECONDS)

        except KeyboardInterrupt:
            print("\nFA Crypto Engine stopped safely.")
            break

        except Exception as error:
            print(f"\nError: {error}")
            print("Retrying in 10 seconds...")
            time.sleep(10)


if __name__ == "__main__":    maifrom binanace.client import Client
client = Client()
btc = client.get_symbol_ticker(symbol="BTCUSDT")
print("=" * 50)
print("FA CRYPTO ENGINE PRO")
print("=" * 50)
print("BTC Price:", btc['price'])
print("=" * 50)
