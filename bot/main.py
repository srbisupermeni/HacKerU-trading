"""Very small single-threaded bot runner.

Assumptions (your team agreed):
- Two strategies exist at HacKerU_bot/strategy/placeholder_strategy1.py
  and HacKerU_bot/strategy/placeholder_strategy2.py and expose `main()`.
- Each strategy's `main()` returns either None, a single raw order dict, or a list
  of raw order dicts. Raw order dict must include: 'coin', 'side', 'quantity'.

This runner keeps the loop explicit and tiny so it's trivial to inspect and
deploy to AWS.
"""

import logging
import sys
import time
from pathlib import Path
from typing import List


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from HacKerU_bot.api.roostoo import Roostoo
from HacKerU_bot.execution.execution_engine import ExecutionEngine
from HacKerU_bot.portfolio.portfolio import Portfolio


def setup_logging(level: str = 'INFO') -> None:
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO),
                        format="%(asctime)s - %(levelname)s - %(message)s")


def create_core_modules() -> tuple:
    """Instantiate Roostoo client, Portfolio and ExecutionEngine and wire them together."""
    client = Roostoo()
    portfolio = Portfolio(execution_module=None)
    engine = ExecutionEngine(portfolio, roostoo_client=client)
    portfolio.execution = engine
    return client, portfolio, engine


def load_strategies(engine) -> List:
    """Load placeholder strategies and pass the ExecutionEngine instance into each strategy.

    Each strategy receives the engine via its constructor and should call
    `self.engine.create_order(...)` when constructing orders.
    """
    from HacKerU_bot.strategy.placeholder_strategy1 import PlaceholderStrategy1
    from HacKerU_bot.strategy.placeholder_strategy2 import PlaceholderStrategy2

    # Provide the engine instance to each strategy so they can use self.engine
    strat1 = PlaceholderStrategy1(engine=engine)
    strat2 = PlaceholderStrategy2(engine=engine)
    strategies = [strat1, strat2]

    # strategies now receive engine via constructor and should call self.engine.create_order(...)
    return strategies


def run_bot_loop(client, portfolio, engine, strategies, loop_sleep_seconds: float = 60.0) -> None:
    """Main single-threaded loop. Keeps behavior identical but is more readable when modularized."""
    # in-memory order queue: mapping timestamp (int seconds) -> list of order dicts
    order_queue = {}

    # track when we last executed an order (epoch seconds); initialize to now
    last_execution_time_stamp = int(time.time())

    while True:
        start = time.time()

        # run strategies once per cycle; each strategy.main() MUST return either
        # None or a single, fully-formed order dict created via `engine.create_order()`.
        for idx, strat in enumerate(strategies, start=1):
            try:
                raw = strat.main()
                if not raw:
                    continue

                if not isinstance(raw, dict):
                    raise ValueError(f"strategy returned invalid order (must be dict or None): {raw}")

                ts = int(time.time())
                order_queue.setdefault(ts, []).append(raw)
            except Exception:
                logging.getLogger(__name__).exception("strategy execution failed")

        # Execution vs polling policy
        try:
            now = int(time.time())

            if order_queue:
                oldest_ts = sorted(order_queue.keys())[0]
                order_list = order_queue.get(oldest_ts, [])
                if order_list:
                    next_order = order_list.pop(0)
                    if not order_list:
                        order_queue.pop(oldest_ts, None)
                    try:
                        engine.execute_order(next_order)
                        logging.getLogger(__name__).info("executed queued order: %s", next_order.get('client_order_id'))
                    except Exception:
                        logging.getLogger(__name__).exception("failed to execute queued order")

            else:
                try:
                    # Get pending counts from both the client and our local engine snapshot
                    resp_pending = client.get_pending_count()
                    engine_pending = engine.get_pending_orders_snapshot()

                    # parse exchange pending total
                    resp_total = 0
                    if isinstance(resp_pending, dict):
                        try:
                            resp_total = int(resp_pending.get('TotalPending', 0))
                        except Exception:
                            resp_total = 0

                    # parse engine pending total
                    engine_total = 0
                    if isinstance(engine_pending, dict):
                        try:
                            engine_total = sum(len(v) for v in engine_pending.values())
                        except Exception:
                            engine_total = 0

                    # If both sides report zero pending orders, nothing to do
                    if resp_total == 0 and engine_total == 0:
                        has_pending = False
                    else:
                        # There's something to reconcile (mismatch or non-empty) — poll/process
                        has_pending = True
                except Exception:
                    logging.getLogger(__name__).exception("failed to get pending count from client")
                    has_pending = False

                if has_pending:
                    # Only run the heavier reconciliation polling at most once per minute
                    if now - last_execution_time_stamp >= 60:
                        try:
                            resp = client.query_order()
                            if resp:
                                engine.process_query_response(resp)
                        except Exception:
                            logging.getLogger(__name__).exception("failed to query/process orders")
                        last_execution_time_stamp = now

        except Exception:
            logging.getLogger(__name__).exception("execution/polling block failed")

        # refresh market prices
        try:
            portfolio.update_positions()
            portfolio.update_market_prices()
        except Exception:
            logging.getLogger(__name__).exception("failed to update market prices")

        # snapshot/log
        try:
            snap = portfolio.get_pnl_snapshot()
            logging.getLogger(__name__).info("snapshot: %s", snap)
        except Exception:
            logging.getLogger(__name__).exception("snapshot failed")

        elapsed = time.time() - start
        to_sleep = max(0.0, loop_sleep_seconds - elapsed)
        time.sleep(to_sleep)


def main() -> None:
    """Simplified entrypoint: no CLI args, more readable orchestration."""
    setup_logging('INFO')
    client, portfolio, engine = create_core_modules()
    strategies = load_strategies(engine)

    try:
        portfolio.initialize_from_exchange_info(roostoo_client=client)
    except Exception:
        logging.getLogger(__name__).warning("initialize_from_exchange_info failed; continuing")

    try:
        run_bot_loop(client, portfolio, engine, strategies)
    except KeyboardInterrupt:
        logging.getLogger(__name__).info("Shutting down (KeyboardInterrupt)")


if __name__ == "__main__":
    main()

