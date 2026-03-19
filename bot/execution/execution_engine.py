import logging
from typing import Optional, Dict

from bot.execution.roostoo import Roostoo
from bot.portfolio.portfolio import Portfolio


class ExecutionEngine:
    """简单的执行引擎，负责在下单时与 Portfolio 的锁机制进行协调。

    用法：
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
        """获取锁，通过 Roostoo 客户端下单，成功后登记执行。

        返回值为字典，包含键: success (bool)、message (str)，以及成功/失败时的额外信息。
        """

        # 从 portfolio 获取币种锁
        acquired = False
        try:
            acquired = self.portfolio.acquire_coin(coin, strategy_id)
        except Exception as e:
            return {'success': False, 'error': 'acquire_error', 'message': str(e)}

        if not acquired:
            return {'success': False, 'error': 'coin_locked', 'message': f'Coin {coin} is locked by another strategy'}

        # 确定 order_type 的默认值
        if order_type is None:
            order_type = 'LIMIT' if price is not None else 'MARKET'

        # 通过 Roostoo 客户端下单
        try:
            result = self.client.place_order(coin, side, quantity, price=price, order_type=order_type)
        except Exception as e:
            # 发生异常时释放锁
            try:
                self.portfolio.release_coin(coin)
            except Exception:
                pass
            return {'success': False, 'error': 'client_exception', 'message': str(e)}

        if not result or not isinstance(result, dict) or not result.get('Success'):
            # 下单失败时释放锁
            try:
                self.portfolio.release_coin(coin)
            except Exception:
                pass
            return {'success': False, 'error': 'order_failed', 'message': 'Order placement failed', 'raw': result}

        # 成功时，提取有用字段并登记
        order_id = None
        status = None
        filled_qty = 0.0

        # 尝试从 OrderDetail 容器中读取（按 API 文档）
        if isinstance(result, dict):
            od = result.get('OrderDetail')
            if isinstance(od, dict):
                order_id = od.get('OrderID') or od.get('order_id')
                status = od.get('Status') or od.get('status')
                filled_qty = od.get('FilledQuantity') or od.get('FilledQty') or od.get('filled_qty') or 0

            # 再尝试顶层字段作为回退
            if order_id is None:
                order_id = result.get('OrderID') or result.get('order_id')
            if status is None:
                status = result.get('Status') or result.get('status')

            # OrderMatched 列表回退
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

        # 后处理步骤取决于订单状态
        try:
            st = (status or '').upper() if status else ''

            # 如果订单为 FILLED（吃单或完全成交），登记执行并设置策略参数，然后释放锁
            if st == 'FILLED' or (filled_qty and float(filled_qty) > 0):
                self.portfolio.register_order_execution(coin, strategy_id, side, filled_qty or quantity, price or 0.0)
                # 设置策略参数（若策略所有者不匹配可能抛出）
                self.portfolio.set_strategy_params(coin, strategy_id, stop_loss, take_profit)

                # 执行后释放锁
                try:
                    self.portfolio.release_coin(coin)
                except Exception:
                    pass

            # 如果订单为 PENDING（挂单），暂不登记执行；保持币种锁并设置策略参数
            elif st == 'PENDING':
                # 设置参数，便于监控在订单成交时采取动作
                self.portfolio.set_strategy_params(coin, strategy_id, stop_loss, take_profit)
            # 有意保留锁

            else:
                # 其他状态：尝试设置参数并释放锁以避免死锁
                try:
                    self.portfolio.set_strategy_params(coin, strategy_id, stop_loss, take_profit)
                except Exception:
                    pass
                try:
                    self.portfolio.release_coin(coin)
                except Exception:
                    pass

        except Exception as e:
            # 登记或后处理出错：释放锁并返回失败
            try:
                self.portfolio.release_coin(coin)
            except Exception:
                pass
            return {'success': False, 'error': 'post_registration_failed', 'message': str(e), 'order_id': order_id,
                    'raw': result}

        return {'success': True, 'order_id': order_id, 'status': status, 'filled_qty': filled_qty, 'raw': result}
