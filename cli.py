from __future__ import annotations

import argparse

from market_oracle.engine import scan_market


def main() -> None:
    parser = argparse.ArgumentParser(description="MarketScope — probabilistyczny skaner rynku")
    parser.add_argument("symbols", nargs="+", help="np. SPY AAPL BTC-USD")
    parser.add_argument("--horizon", type=int, choices=(1, 5, 20), default=5)
    parser.add_argument("--years", type=int, default=8)
    args = parser.parse_args()
    result, errors = scan_market(args.symbols, args.horizon, args.years)
    if not result.empty:
        print(result.to_string(index=False))
    for symbol, error in errors.items():
        print(f"BŁĄD {symbol}: {error}")


if __name__ == "__main__":
    main()
