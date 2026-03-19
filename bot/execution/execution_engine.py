import logging
from typing import Optional, Dict

from bot.execution.roostoo import Roostoo
from bot.portfolio.portfolio import Portfolio


class ExecutionEngine:
    """Simple execution engine that coordinates order placement with Portfolio locks.

	Usage:
		engine = ExecutionEngine(portfolio)
		res = engine.execute_order('BTC', 'BUY', 0.1, price=None, order_type=None, strategy_id='s1', stop_loss=30000, take_profit=40000)
	"""

    def __init__(self, portfolio: Portfolio, roostoo_client: Optional[Roostoo] = None):
        self.portfolio = portfolio
        self.client = roostoo_client or Roostoo()
        self.logger = logging.getLogger('ExecutionEngine')

    def execute_order(self, coin: str, side: str, quantity: float, price: Optional[float] = None,
                      order_type: Optional[str] = None, strategy_id: str = None, stop_loss: float = None,
                      take_profit: float = None) -> Dict:
        """Acquire lock, place order via Roostoo client, and register execution on success.

		Returns a dict with keys: success (bool), message (str), and extra fields on success/failure.
		"""

        # Acquire coin lock from portfolio
        acquired = False
        try:
            acquired = self.portfolio.acquire_coin(coin, strategy_id)
        except Exception as e:
            return {'success': False, 'error': 'acquire_error', 'message': str(e)}

        if not acquired:
            return {'success': False, 'error': 'coin_locked', 'message': f'Coin {coin} is locked by another strategy'}

        # Determine order_type default
        if order_type is None:
            order_type = 'LIMIT' if price is not None else 'MARKET'

        payload = {
            'pair_or_coin': coin,
            'side': side,
            'quantity': quantity,
            'price': price,
            'order_type': order_type,
            'strategy_id': strategy_id,
            'stop_loss': stop_loss,
            'take_profit': take_profit
        }

        # Place order via roostoo client
        try:
            result = self.client.place_order(coin, side, quantity, price=price, order_type=order_type)
        except Exception as e:
            # release lock on failure
            try:
                self.portfolio.release_coin(coin)
            except Exception:
                pass
            return {'success': False, 'error': 'client_exception', 'message': str(e)}

        if not result or not isinstance(result, dict) or not result.get('Success'):
            # release lock on failure
            try:
                self.portfolio.release_coin(coin)
            except Exception:
                pass
            return {'success': False, 'error': 'order_failed', 'message': 'Order placement failed', 'raw': result}

        # On success, extract useful fields and register
        order_id = None
        status = None
        filled_qty = 0.0

        # Try OrderDetail container (per API docs)
        if isinstance(result, dict):
            od = result.get('OrderDetail')
            if isinstance(od, dict):
                order_id = od.get('OrderID') or od.get('order_id')
                status = od.get('Status') or od.get('status')
                filled_qty = od.get('FilledQuantity') or od.get('FilledQty') or od.get('filled_qty') or 0

            # try top-level fallbacks
            if order_id is None:
                order_id = result.get('OrderID') or result.get('order_id')
            if status is None:
                status = result.get('Status') or result.get('status')

            # OrderMatched list fallback
            if not order_id and isinstance(result.get('OrderMatched'), list) and len(result.get('OrderMatched')) > 0:
                first = result.get('OrderMatched')[0]
                order_id = first.get('OrderID') or first.get('order_id')
                status = status or first.get('Status') or first.get('status')
                filled_qty = filled_qty or first.get('FilledQuantity') or first.get('FilledQty') or first.get(
                    'filled_qty') or 0

        try:
            filled_qty = float(filled_qty or 0.0)
        except Exception:
            filled_qty = 0.0

        # Post-processing depends on status
        try:
            st = (status or '').upper() if status else ''

            # If order was FILLED (taker or fully filled), register execution and set strategy params, then release lock
            if st == 'FILLED' or (filled_qty and float(filled_qty) > 0):
                self.portfolio.register_order_execution(coin, strategy_id, side, filled_qty or quantity, price or 0.0)
                # set strategy params (may raise if owner mismatch)
                self.portfolio.set_strategy_params(coin, strategy_id, stop_loss, take_profit)

                # release lock after execution
                try:
                    self.portfolio.release_coin(coin)
                except Exception:
                    pass

            # If order is pending (maker), do not register execution yet; keep the coin locked and set strategy params
            elif st == 'PENDING':
                # set params so monitoring can act when the order is filled
                self.portfolio.set_strategy_params(coin, strategy_id, stop_loss, take_profit)
            # keep the lock intentionally

            else:
                # Other statuses: attempt to set params and release lock to avoid deadlocks
                try:
                    self.portfolio.set_strategy_params(coin, strategy_id, stop_loss, take_profit)
                except Exception:
                    pass
                try:
                    self.portfolio.release_coin(coin)
                except Exception:
                    pass

        except Exception as e:
            # registration error: release lock and return failure
            try:
                self.portfolio.release_coin(coin)
            except Exception:
                pass
            return {'success': False, 'error': 'post_registration_failed', 'message': str(e), 'order_id': order_id,
                    'raw': result}

        return {'success': True, 'order_id': order_id, 'status': status, 'filled_qty': filled_qty, 'raw': result}
