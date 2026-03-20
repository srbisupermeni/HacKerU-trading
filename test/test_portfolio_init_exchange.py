#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test: initialize Portfolio from real Roostoo.get_exchange_info()

This script will:
- instantiate Roostoo client (reads bot/config/roostoo.yaml)
- call get_exchange_info()
- create Portfolio and initialize it using the exchange info
- print summary of loaded data

Run: python test/test_portfolio_init_exchange.py
"""
import sys
import traceback
from pathlib import Path

# Ensure project root is importable when running the script directly
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bot.api.roostoo import Roostoo
from bot.portfolio.portfolio import Portfolio


def main():
    try:
        client = Roostoo()
        print("Calling get_exchange_info() on Roostoo client...")
        info = client.get_exchange_info()
        print("Raw response:")
        print(info)

        p = Portfolio(execution_module=None)
        ok = p.initialize_from_exchange_info(roostoo_client=client, exchange_info=info, fallback_balance=1000.0)
        print(f"initialize_from_exchange_info returned: {ok}")
        print(f"account_balance: {p.account_balance} {p.account_currency}")
        print(f"exchange_is_running: {p.exchange_is_running}")
        print(f"tradable pairs count: {len(p.tradable_pairs)}")
        print("sample pairs:")
        for i, (k, v) in enumerate(p.trade_pairs.items()):
            if i >= 5:
                break
            print(f"  {k}: {v}")

    except Exception as e:
        print("Exception during test:")
        traceback.print_exc()


if __name__ == '__main__':
    main()


