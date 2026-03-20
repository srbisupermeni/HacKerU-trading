#!/usr/bin/env python3
"""
HTTP-driven fake bot: uses `bot.execution.roostoo.Roostoo` client to interact with the
HTTP mock server. This exercises the real HTTP code paths in `bot/execution/roostoo.py`,
`ExecutionEngine`, and `Portfolio`.

Run: python mock/http_fake_bot.py --mock-url http://127.0.0.1:9000 --ticks 12 --seed 42
"""
from __future__ import annotations

import argparse
import random
import time
from typing import Dict

from pathlib import Path
from bot.api.roostoo import Roostoo
from bot.execution.execution_engine import ExecutionEngine
from bot.portfolio.portfolio import Portfolio


COINS = ["BTC", "ETH", "XRP"]


def random_walk_prices(rng: random.Random, prices: Dict[str, float]) -> Dict[str, float]:
    new_prices = {}
    for coin, p in prices.items():
        if coin == "BTC":
            drift = rng.uniform(-0.01, 0.01)
        elif coin == "ETH":
            drift = rng.uniform(-0.015, 0.015)
        else:
            drift = rng.uniform(-0.02, 0.02)
        new_prices[coin] = round(max(0.0001, p * (1.0 + drift)), 6)
    return new_prices


class SimpleRandomStrategy:
    def __init__(self, rng: random.Random):
        self.rng = rng

    def generate_signal(self, market_prices: Dict[str, float]):
        if self.rng.random() < 0.35:
            return None
        coin = self.rng.choice(COINS)
        side = self.rng.choice(["BUY", "SELL"])
        price = market_prices[coin]
        order_kind = self.rng.choices(["MARKET", "LIMIT"], weights=[0.4, 0.6], k=1)[0]
        if side == "BUY":
            qty = round(self.rng.uniform(0.01, 0.12), 4) if coin == "BTC" else round(self.rng.uniform(0.5, 5.0), 4)
        else:
            qty = round(self.rng.uniform(0.01, 0.08), 4) if coin == "BTC" else round(self.rng.uniform(0.2, 3.0), 4)

        limit_price = None
        if order_kind == "LIMIT":
            offset = self.rng.uniform(-0.015, 0.015)
            limit_price = round(price * (1 + offset), 4)

        return {
            "coin": coin,
            "side": side,
            "quantity": qty,
            "order_type": order_kind,
            "price": limit_price,
            "strategy_id": "http_fake_strategy",
        }


def print_portfolio_snapshot(portfolio: Portfolio):
    print("\n[Portfolio Snapshot]")
    snapshot = portfolio.get_pnl_snapshot()
    for coin in COINS:
        pos = portfolio.get_position(coin)
        print(f"  {coin}: free={pos['free']:.6f}, total={pos['total']:.6f}, avg={pos['avg_entry_price']:.4f}, uPnL={pos['unrealized_pnl']:.4f}")
    summary = snapshot.get("portfolio_summary", {})
    print(f"  SUMMARY: realized={summary.get('total_realized_pnl', 0.0):.4f}, unrealized={summary.get('total_unrealized_pnl', 0.0):.4f}, total={summary.get('total_pnl', 0.0):.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock-url", required=True)
    parser.add_argument("--ticks", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    # Provide a valid config file so Roostoo initialization succeeds, then override base_url
    cfg_path = Path(__file__).resolve().parent.parent / 'bot' / 'config' / 'roostoo.yaml'
    client = Roostoo(config_path=cfg_path)
    client.base_url = args.mock_url

    portfolio = Portfolio(execution_module=None)
    engine = ExecutionEngine(portfolio, roostoo_client=client)
    strategy = SimpleRandomStrategy(rng)

    market_prices = {"BTC": 65000.0, "ETH": 3200.0, "XRP": 0.55}
    portfolio.update_market_prices(market_prices)

    print("HTTP FAKE BOT START")
    for tick in range(1, args.ticks + 1):
        print(f"\n--- TICK {tick} ---")
        market_prices = random_walk_prices(rng, market_prices)
        portfolio.update_market_prices(market_prices)
        for c, p in market_prices.items():
            print(f"  {c}: {p}")

        signal = strategy.generate_signal(market_prices)
        if not signal:
            print("[Strategy] no signal")
        else:
            print(f"[Strategy] placing {signal['order_type']} {signal['side']} {signal['quantity']} {signal['coin']} @ {signal['price']}")
            res = engine.execute_order(
                coin=signal['coin'],
                side=signal['side'],
                quantity=signal['quantity'],
                price=signal['price'],
                order_type=signal['order_type'],
                strategy_id=signal['strategy_id']
            )
            print(f"[Execution Result] {res}")

        # query server and process responses
        q = client.query_order()
        print(f"[Exchange Query] {q}")
        processed = engine.process_query_response(q)
        print(f"[Engine Processed] {processed}")

        print_portfolio_snapshot(portfolio)
        time.sleep(0.2)

    print("HTTP FAKE BOT END")


if __name__ == '__main__':
    main()


