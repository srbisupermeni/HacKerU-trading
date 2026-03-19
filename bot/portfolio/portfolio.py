import threading
import time
from typing import Dict, Optional


class Portfolio:
    """
    Portfolio class manages all assets the bot trades.
    Responsibilities:
        - Track positions per coin
        - Store stop-loss / take-profit per coin
        - Prevent multiple strategies from trading the same coin simultaneously
        - Update positions automatically from Roostoo
        - Trigger stop-loss / take-profit via Execution module
    """

    def __init__(self, execution_module):
        """
        :param execution_module: reference to your Execution class instance
        """
        self.execution = execution_module

        self.account_balance: float = 1000.0

        # Current positions: {coin: {'free': float, 'locked': float}}
        self.positions: Dict[str, Dict[str, float]] = {}

        # Locks per coin to prevent multiple strategies from using same coin
        self.coin_locks: Dict[str, threading.Lock] = {}

        # Strategy-specific parameters: {coin: {'strategy_id': str, 'stop_loss': float, 'take_profit': float}}
        self.strategy_params: Dict[str, Dict] = {}

        # Internal thread for auto-monitoring stop-loss / take-profit
        self.monitoring_thread: Optional[threading.Thread] = None
        self.monitoring_active = False

    # ================= POSITION MANAGEMENT =================

    def update_positions(self, new_positions: Dict[str, Dict[str, float]] = None):
        self.positions = new_positions or self.positions

    def get_position(self, coin: str) -> Dict[str, float]:
        """
        Return current position for a given coin
        """
        return self.positions.get(coin, {"free": 0.0, "locked": 0.0})

    # ================= COIN LOCK MANAGEMENT =================

    def acquire_coin(self, coin: str, strategy_id: str) -> bool:
        """
        Attempt to lock coin for a strategy.
        Returns True if lock acquired, False otherwise.
        """
        # Ensure there's a Lock object for this coin
        if coin not in self.coin_locks:
            self.coin_locks[coin] = threading.Lock()

        lock = self.coin_locks[coin]

        # Try to acquire without blocking
        acquired = lock.acquire(blocking=False)
        if not acquired:
            return False

        # Record that this strategy currently holds the coin (owner info kept in strategy_params)
        existing = self.strategy_params.get(coin, {})
        existing['strategy_id'] = strategy_id
        # Keep stop_loss / take_profit keys present (may be set later)
        existing.setdefault('stop_loss', None)
        existing.setdefault('take_profit', None)
        self.strategy_params[coin] = existing

        return True

    def release_coin(self, coin: str):
        """
        Release the lock on a coin
        """
        lock = self.coin_locks.get(coin)
        if lock and lock.locked():
            try:
                lock.release()
            except RuntimeError:
                # already released or not owned by this thread; ignore
                return

        # Remove strategy ownership info but keep stop_loss/take_profit if present
        if coin in self.strategy_params:
            params = self.strategy_params.get(coin, {})
            # remove only the owner marker
            params.pop('strategy_id', None)

            # If both stop_loss and take_profit are missing or None, remove the entry entirely
            if params.get('stop_loss') is None and params.get('take_profit') is None:
                self.strategy_params.pop(coin, None)
            else:
                self.strategy_params[coin] = params

    # ================= STRATEGY PARAMETERS =================

    def set_strategy_params(self, coin: str, strategy_id: str, stop_loss: float, take_profit: float):
        """
        Register stop-loss and take-profit for a coin under a specific strategy
        """
        params = self.strategy_params.get(coin, {})
        owner = params.get('strategy_id')
        if owner and owner != strategy_id:
            # Do not overwrite params owned by another strategy
            raise PermissionError(f"Strategy {strategy_id} cannot set params for {coin} owned by {owner}")

        params['strategy_id'] = strategy_id
        params['stop_loss'] = stop_loss
        params['take_profit'] = take_profit
        self.strategy_params[coin] = params

    def get_strategy_params(self, coin: str) -> Dict:
        """
        Return strategy parameters (stop-loss, take-profit) for a coin
        """
        return self.strategy_params.get(coin, {})

    # ================= ORDER MONITORING =================

    def start_monitoring(self, interval: float = 2.0):
        """
        Start background thread that monitors stop-loss / take-profit triggers
        """
        if self.monitoring_thread is None or not self.monitoring_thread.is_alive():
            self.monitoring_active = True
            self.monitoring_thread = threading.Thread(target=self._monitor_loop, args=(interval,))
            self.monitoring_thread.start()

    def stop_monitoring(self):
        """
        Stop the background monitoring thread
        """
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join()

    def _monitor_loop(self, interval: float):
        """
        Internal loop to monitor positions and trigger Execution orders
        """
        while self.monitoring_active:
            # TODO: check each coin's current price vs stop-loss / take-profit
            # If triggered, call self.execution.place_order(...)
            try:
                # Minimal sleep to avoid busy loop; real implementation should fetch prices
                time.sleep(interval)
            except Exception:
                break

    # ================= ORDER UPDATES =================

    def register_order_execution(self, coin: str, strategy_id: str, side: str, qty: float, price: float):
        """
        Called by Execution after an order is filled
        Updates positions, releases locks if necessary
        """
        # Initialize position record if missing
        pos = self.positions.get(coin, {'free': 0.0, 'locked': 0.0})

        # Normalize side
        s = side.upper()
        if s == 'BUY':
            # Treat filled buy as available (free) balance increase
            pos['free'] = pos.get('free', 0.0) + float(qty)
        elif s == 'SELL':
            # For sell, deduct from free first, then locked
            remaining = float(qty)
            free = pos.get('free', 0.0)
            if free >= remaining:
                pos['free'] = free - remaining
            else:
                pos['free'] = 0.0
                locked = pos.get('locked', 0.0)
                pos['locked'] = max(0.0, locked - (remaining - free))
        else:
            # unknown side, do nothing
            pass

        self.positions[coin] = pos

        # If this strategy owned the coin lock, release it now (common pattern after order fill)
        params = self.strategy_params.get(coin, {})
        owner = params.get('strategy_id')
        if owner == strategy_id:
            # keep params (stop_loss/take_profit) but release lock
            try:
                self.release_coin(coin)
            except Exception:
                # ignore release errors
                pass

        # Optionally, notify execution module or log (execution module holds reference)
        return True

