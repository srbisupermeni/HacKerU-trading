import logging
import threading
import time
from typing import Optional, Dict, List, Any

from bot.execution.roostoo import Roostoo
from bot.portfolio.portfolio import Portfolio


class ExecutionEngine:
    """
    执行引擎 - 负责下单和订单管理
    
    核心功能：
        1. 获取币种锁，防止重复下单
        2. 通过 Roostoo API 下单
        3. 跟踪待处理订单（部分成交、未完全成交的订单）
        4. 处理订单成交反馈并更新持仓
    
    订单生命周期：
        - FILLED: 完全成交，立即返回，锁释放
        - PENDING: 部分成交或未成交，进入待处理队列，锁保持
        - 部分成交: 保持锁，等待后续查询更新
    
    使用示例：
        engine = ExecutionEngine(portfolio, roostoo_client)
        result = engine.execute_order(
            coin='BTC',
            side='BUY',
            quantity=1.0,
            price=50000.0,
            strategy_id='my_strategy',
            stop_loss=45000.0,
            take_profit=60000.0
        )
        
        if result['success']:
            print(f"下单成功: {result['order_id']}")
            if result.get('remaining_qty', 0) > 0:
                print(f"部分成交，剩余 {result['remaining_qty']} 未成交")
    """

    def __init__(self, portfolio: Portfolio, roostoo_client: Optional[Roostoo] = None):
        self.portfolio = portfolio
        self.client = roostoo_client or Roostoo()
        self.logger = logging.getLogger('ExecutionEngine')
        # pending_orders_queue maps key ("{coin}:{strategy_id}") -> list of order dicts
        # order dict contains at least: order_id (str), side, quantity, price, created_at, raw
        self.pending_orders_queue: Dict[str, List[Dict[str, Any]]] = {}
        # index to find which queue key an order_id belongs to: order_id(str) -> queue_key
        self._orderid_index: Dict[str, str] = {}
        # lock protecting pending_orders_queue and _orderid_index
        self._pq_lock = threading.Lock()

    def execute_order(self, coin: str, side: str, quantity: float, price: Optional[float] = None,
                      order_type: Optional[str] = None, strategy_id: str = None, stop_loss: float = None,
                      take_profit: float = None) -> Dict:
        """
        下单接口（策略层主要调用此方法）
        
        此方法会：
        1. 自动获取币种锁（防止重复下单）
        2. 调用交易所 API 下单
        3. 根据成交情况处理（完全成交/部分成交/未成交）
        4. 返回下单结果
        
        参数：
            coin: 币种代码，例如 'BTC', 'ETH'
            side: 方向，'BUY' 或 'SELL'
            quantity: 数量
            price: 价格（可选）
                - 如果不提供 price，则自动使用 MARKET 订单
                - 如果提供 price，则使用 LIMIT 订单
            order_type: 订单类型（可选），'MARKET' 或 'LIMIT'
                - 不提供时自动判断（有价格 -> LIMIT，无价格 -> MARKET）
            strategy_id: 策略 ID（必须提供），用于锁定币种
            stop_loss: 止损价格（可选）
            take_profit: 止盈价格（可选）
        
        返回值：字典，包含：
            success (bool): 是否成功下单
            order_id: 交易所返回的订单 ID
            status: 订单状态 (FILLED/PENDING/等)
            filled_qty: 已成交数量
            remaining_qty: 未成交数量（重要！用于判断是否部分成交）
            queued (bool): 是否在待处理队列中
            error: 错误代码（失败时）
            message: 错误信息（失败时）
            raw: 交易所原始响应
        
        返回示例 - 完全成交：
            {
                'success': True,
                'order_id': '123456',
                'status': 'FILLED',
                'filled_qty': 1.0,
                'remaining_qty': 0.0,
                'raw': {...}
            }
        
        返回示例 - 部分成交：
            {
                'success': True,
                'order_id': '123457',
                'status': 'PENDING',
                'filled_qty': 0.5,
                'remaining_qty': 0.5,
                'queued': True,
                'raw': {...}
            }
        
        返回示例 - 失败：
            {
                'success': False,
                'error': 'coin_locked',
                'message': 'Coin BTC is locked by another strategy'
            }
        
        使用示例：
            result = engine.execute_order(
                coin='BTC',
                side='BUY',
                quantity=1.0,
                price=50000.0,
                strategy_id='strategy_1',
                stop_loss=45000.0,
                take_profit=60000.0
            )
            
            if result['success']:
                if result['remaining_qty'] <= 0:
                    print(f"订单完全成交: {result['filled_qty']}")
                else:
                    print(f"订单部分成交: {result['filled_qty']}/{result['filled_qty'] + result['remaining_qty']}")
            else:
                print(f"下单失败: {result['message']}")
        """

        # 从 portfolio 获取币种锁
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
                self.portfolio.release_coin(coin, strategy_id=strategy_id)
            except Exception:
                pass
            return {'success': False, 'error': 'client_exception', 'message': str(e)}

        if not result or not isinstance(result, dict) or not result.get('Success'):
            # 下单失败时释放锁
            try:
                self.portfolio.release_coin(coin, strategy_id=strategy_id)
            except Exception:
                pass
            return {'success': False, 'error': 'order_failed', 'message': 'Order placement failed', 'raw': result}

        # 成功时，提取有用字段并登记
        order_id = None
        status = None
        filled_qty = 0.0
        orig_qty = quantity
        exec_price = None
        commission_coin = None
        commission_value = 0.0
        commission_percent = None

        # 尝试从 OrderDetail 容器中读取（按 API 文档）
        if isinstance(result, dict):
            od = result.get('OrderDetail')
            if isinstance(od, dict):
                order_id = od.get('OrderID') or od.get('order_id')
                status = od.get('Status') or od.get('status')
                filled_qty = od.get('FilledQuantity') or od.get('FilledQty') or od.get('filled_qty') or 0
                exec_price = od.get('FilledAverPrice') or od.get('Price') or price
                # 提取原始数量以计算剩余量
                orig_qty = od.get('Quantity') or od.get('quantity') or quantity
                # 提取手续费信息（若有）
                commission_coin = od.get('CommissionCoin') or od.get('commission_coin')
                commission_value = od.get('CommissionChargeValue') if od.get('CommissionChargeValue') is not None else od.get('commission_charge_value')
                commission_percent = od.get('CommissionPercent') or od.get('commission_percent')
                # 规范化为数值
                try:
                    commission_value = float(commission_value) if commission_value is not None else 0.0
                except Exception:
                    commission_value = 0.0

            # OrderMatched 列表回退
            if not order_id and isinstance(result.get('OrderMatched'), list) and len(result.get('OrderMatched')) > 0:
                first = result.get('OrderMatched')[0]
                order_id = first.get('OrderID') or first.get('order_id')
                status = status or first.get('Status') or first.get('status')
                filled_qty = filled_qty or first.get('FilledQuantity') or first.get('FilledQty') or first.get(
                    'filled_qty') or 0
                orig_qty = first.get('Quantity') or first.get('quantity') or quantity
                if exec_price is None:
                    exec_price = first.get('FilledAverPrice') or first.get('Price') or price

        try:
            filled_qty = float(filled_qty or 0.0)
            orig_qty = float(orig_qty or quantity)
        except Exception:
            filled_qty = 0.0
            orig_qty = float(quantity)

        try:
            exec_price = float(exec_price if exec_price is not None else (price or 0.0))
        except Exception:
            exec_price = float(price or 0.0)

        # 计算剩余数量
        remaining_qty = max(0.0, orig_qty - filled_qty)

        # 后处理步骤取决于订单状态
        try:
            st = (status or '').upper() if status else ''

            # Helper: normalize order_id to str for indexing
            order_id_str = str(order_id) if order_id is not None else None

            # 完全成交的情况：status==FILLED 或 剩余量为 0
            # 在这种情况下，注册执行并释放锁
            if st == 'FILLED' or remaining_qty <= 0.001:  # 允许 0.001 的浮点误差
                self.portfolio.register_order_execution(coin, strategy_id, side, filled_qty or quantity,
                                                        exec_price,
                                                        fee_amount=(commission_value if 'commission_value' in locals() else 0.0),
                                                        fee_currency=(commission_coin if 'commission_coin' in locals() else None),
                                                        fee_percent=(commission_percent if 'commission_percent' in locals() else None))
                # 立即设置策略参数（若策略所有者不匹配可能抛出）
                try:
                    if stop_loss is not None or take_profit is not None:
                        self.portfolio.set_strategy_params(coin, strategy_id, stop_loss, take_profit)
                except Exception:
                    # Do not fail execution if setting params fails
                    self.logger.exception('Failed to set strategy params after filled order')

                # 注册执行会释放锁（portfolio.register_order_execution）
                # 这里额外尝试不会造成问题，因为 release_coin 检查所有权
                try:
                    self.portfolio.release_coin(coin, strategy_id=strategy_id)
                except Exception:
                    pass

                # If this order was previously indexed as pending (defensive), remove it
                if order_id_str:
                    try:
                        self._remove_pending_order_by_id(order_id_str)
                    except Exception:
                        pass

                return {'success': True, 'order_id': order_id, 'status': status, 'filled_qty': filled_qty,
                        'remaining_qty': 0.0, 'raw': result}

            # 部分成交的情况（有一些成交但还有剩余）
            # 注册已成交部分，但保持锁定并将订单加入待处理队列
            elif filled_qty > 0.0 and remaining_qty > 0.001:
                # 注册已成交部分（包含手续费信息，如果 API 提供）
                self.portfolio.register_order_execution(coin, strategy_id, side, filled_qty, exec_price,
                                                        fee_amount=(commission_value if 'commission_value' in locals() else 0.0),
                                                        fee_currency=(commission_coin if 'commission_coin' in locals() else None),
                                                        fee_percent=(commission_percent if 'commission_percent' in locals() else None))
                
                # 注意：register_order_execution 会检查所有权并释放锁
                # 但对于部分成交，我们想保持锁定！
                # 所以我们需要重新获取锁
                reacquired = self.portfolio.acquire_coin(coin, strategy_id)
                if not reacquired:
                    # 如果重新获取失败（被其他策略抢占），记录警告但继续
                    # 这是一个边界情况，可能不会经常发生
                    self.logger.warning(
                        f"Could not reacquire lock for {coin} after partial fill - another strategy may have taken it"
                    )
                
                # 设置策略参数
                if stop_loss is not None or take_profit is not None:
                    try:
                        self.portfolio.set_strategy_params(coin, strategy_id, stop_loss, take_profit)
                    except Exception:
                        self.logger.exception('Failed to set strategy params after partial fill')

                # 将订单加入队列以供后续轮询
                # 跟踪原始数量、已成交数量和剩余数量
                if order_id_str:
                    self._add_pending_order(coin, strategy_id, order_id_str, {
                        'order_id': order_id_str,
                        'side': side,
                        'original_quantity': orig_qty,
                        'filled_quantity': filled_qty,
                        'remaining_quantity': remaining_qty,
                        'price': price,
                        'commission_coin': commission_coin,
                        'commission_value': commission_value,
                        'commission_percent': commission_percent,
                        'created_at': int(time.time() * 1000),
                        'raw': result
                    })

                # 返回部分成交信息（保持锁）
                return {'success': True, 'order_id': order_id, 'status': status, 'filled_qty': filled_qty,
                        'remaining_qty': remaining_qty, 'queued': True, 'raw': result}

            # 如果订单为 PENDING（挂单 / MAKER），将订单加入 pending_orders_queue，保持币种锁并设置策略参数
            elif st == 'PENDING' or (isinstance(result.get('OrderDetail'), dict) and (
                    result.get('OrderDetail', {}).get('Role') == 'MAKER')):
                # 设置参数，便于监控在订单成交时采取动作
                if stop_loss is not None or take_profit is not None:
                    self.portfolio.set_strategy_params(coin, strategy_id, stop_loss, take_profit)

                # 将订单加入队列以供后续轮询
                # 跟踪原始数量和已成交数量
                if order_id_str:
                    self._add_pending_order(coin, strategy_id, order_id_str, {
                        'order_id': order_id_str,
                        'side': side,
                        'original_quantity': orig_qty,
                        'filled_quantity': filled_qty,
                        'remaining_quantity': remaining_qty,
                        'price': price,
                        'commission_coin': commission_coin,
                        'commission_value': commission_value,
                        'commission_percent': commission_percent,
                        'created_at': int(time.time() * 1000),
                        'raw': result
                    })

                # 返回已排队信息（保持锁）
                return {'success': True, 'order_id': order_id, 'status': status, 'queued': True, 'raw': result}

            else:
                # 其他状态（REJECTED/FAILED/UNKNOWN）：释放锁以避免死锁
                try:
                    if stop_loss is not None or take_profit is not None:
                        self.portfolio.set_strategy_params(coin, strategy_id, stop_loss, take_profit)
                except Exception:
                    pass
                try:
                    self.portfolio.release_coin(coin, strategy_id=strategy_id)
                except Exception:
                    pass

                return {'success': False, 'order_id': order_id, 'status': status, 'raw': result}

        except Exception as e:
            # 登记或后处理出错：释放锁并返回失败
            try:
                self.portfolio.release_coin(coin, strategy_id=strategy_id)
            except Exception:
                pass
            return {'success': False, 'error': 'post_registration_failed', 'message': str(e), 'order_id': order_id,
                    'raw': result}

    # ------------------------------
    # Pending orders queue helpers
    # ------------------------------
    def _queue_key(self, coin: str, strategy_id: str) -> str:
        return f"{coin}:{strategy_id}"

    def _add_pending_order(self, coin: str, strategy_id: str, order_id: str, meta: Dict[str, Any]):
        """Thread-safe add order to pending queue and index it by order_id."""
        key = self._queue_key(coin, strategy_id)
        with self._pq_lock:
            self.pending_orders_queue.setdefault(key, []).append(meta)
            self._orderid_index[str(order_id)] = key
        self.logger.info(f"Added pending order to queue: {key} -> {order_id}")

    def _remove_pending_order_by_id(self, order_id: str) -> bool:
        """Remove a pending order by order_id. Returns True if removed."""
        order_id = str(order_id)
        with self._pq_lock:
            key = self._orderid_index.get(order_id)
            if not key:
                return False
            lst = self.pending_orders_queue.get(key, [])
            new_lst = [o for o in lst if str(o.get('order_id')) != order_id]
            if new_lst:
                self.pending_orders_queue[key] = new_lst
            else:
                # remove empty queue
                self.pending_orders_queue.pop(key, None)
            self._orderid_index.pop(order_id, None)
        self.logger.info(f"Removed pending order from queue: {order_id}")
        return True

    def _update_pending_order_by_id(self, order_id: str, updates: Dict[str, Any]) -> bool:
        """Update metadata for a pending order by order_id. Returns True if updated.
        
        用于更新部分成交订单的元数据（已成交量、剩余量等）
        """
        order_id = str(order_id)
        with self._pq_lock:
            key = self._orderid_index.get(order_id)
            if not key:
                return False
            
            lst = self.pending_orders_queue.get(key, [])
            for order_meta in lst:
                if str(order_meta.get('order_id')) == order_id:
                    # 更新指定字段
                    order_meta.update(updates)
                    self.logger.info(f"Updated pending order {order_id}: {updates}")
                    return True
        
        return False

    def _get_pending_order_meta(self, order_id: str) -> Optional[Dict[str, Any]]:
        """线程安全地读取某个待处理订单的元数据副本（不移除）。

        返回订单元数据（字典）或 None 如果找不到。
        """
        order_id = str(order_id)
        with self._pq_lock:
            key = self._orderid_index.get(order_id)
            if not key:
                return None
            lst = self.pending_orders_queue.get(key, [])
            for order_meta in lst:
                if str(order_meta.get('order_id')) == order_id:
                    # 返回副本以避免外部修改内部结构
                    return dict(order_meta)
        return None

    def get_pending_orders_snapshot(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        获取待处理订单快照（只读）
        
        返回当前所有待处理订单（部分成交或未成交的订单）
        
        返回值：
            字典，格式为 {queue_key: [订单列表]}
            其中 queue_key = f"{coin}:{strategy_id}"
            
            每个订单包含：
                order_id: 订单 ID
                side: BUY 或 SELL
                original_quantity: 原始下单数量
                filled_quantity: 已成交数量
                remaining_quantity: 未成交数量
                price: 下单价格
                created_at: 创建时间戳
        
        使用示例：
            pending = engine.get_pending_orders_snapshot()
            for queue_key, orders in pending.items():
                coin, strategy = queue_key.split(':')
                for order in orders:
                    print(f"订单 {order['order_id']}: "
                          f"成交 {order['filled_quantity']}/{order['original_quantity']}")
        """
        with self._pq_lock:
            return {k: list(v) for k, v in self.pending_orders_queue.items()}

    def process_query_response(self, response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        处理订单查询响应（由后台轮询线程调用）
        
        此方法处理从交易所查询回来的订单状态，并根据成交情况更新内部状态。
        
        说明：
        - 如果订单被完全成交，自动注册执行并从待处理队列移除
        - 如果订单部分成交，更新待处理队列中的元数据，保持锁定
        - 如果订单被取消，释放锁并移除出队列
        
        参数：
            response: 交易所返回的查询响应，应包含 OrderMatched 或 OrderDetail 字段
        
        返回值：
            列表，每个元素是一个订单的处理结果
            
            每个结果包含：
                order_id: 订单 ID
                status: 订单状态
                processed: 是否成功处理
                type: 处理类型 ('full_fill'/'partial_fill'/'canceled'/'no_change')
                filled_qty: 当前已成交量
                remaining_qty: 当前未成交量
        
        返回示例 - 完全成交：
            [
                {
                    'order_id': '123456',
                    'status': 'FILLED',
                    'processed': True,
                    'type': 'full_fill',
                    'filled_qty': 1.0,
                    'remaining_qty': 0.0
                }
            ]
        
        返回示例 - 部分成交：
            [
                {
                    'order_id': '123457',
                    'status': 'PENDING',
                    'processed': True,
                    'type': 'partial_fill',
                    'filled_qty': 0.5,
                    'remaining_qty': 0.5
                }
            ]
        
        使用示例（后台轮询循环）：
            def poller_loop():
                while True:
                    try:
                        # 查询所有待处理订单
                        response = roostoo_client.query_order()
                        
                        # 处理响应
                        results = engine.process_query_response(response)
                        
                        # 记录处理结果
                        for result in results:
                            if result['type'] == 'full_fill':
                                logger.info(f"订单 {result['order_id']} 完全成交")
                            elif result['type'] == 'partial_fill':
                                logger.info(f"订单 {result['order_id']} 部分成交，"
                                          f"剩余 {result['remaining_qty']}")
                        
                        time.sleep(3)  # 每 3 秒查询一次
                    except Exception as e:
                        logger.error(f"轮询错误: {e}")
                        time.sleep(3)
        """
        processed = []
        if not response or not isinstance(response, dict):
            return processed

        entries: List[Dict[str, Any]] = []
        if isinstance(response.get('OrderMatched'), list):
            entries.extend(response.get('OrderMatched'))
        elif isinstance(response.get('OrderDetail'), dict):
            entries.append(response.get('OrderDetail'))

        for e in entries:
            oid = e.get('OrderID') or e.get('order_id')
            if oid is None:
                continue
            oid_s = str(oid)
            status = (e.get('Status') or e.get('status') or '').upper()
            filled = e.get('FilledQuantity') or e.get('FilledQty') or e.get('filled_qty') or 0
            orig_qty = e.get('Quantity') or e.get('quantity') or 0
            
            try:
                filled = float(filled or 0)
                orig_qty = float(orig_qty or 0)
            except Exception:
                filled = 0.0
                orig_qty = 0.0

            remaining = max(0.0, orig_qty - filled)

            # 检查这个订单是否在我们的待处理列表中
            with self._pq_lock:
                key = self._orderid_index.get(oid_s)

            if key:
                # key 格式是 coin:strategy_id
                try:
                    coin, strat = key.split(':', 1)
                except Exception:
                    coin, strat = key, None

                side = e.get('Side') or e.get('side') or 'BUY'
                price = e.get('FilledAverPrice') or e.get('Price') or 0.0

                # 读取当前我们记录的 pending 元数据（如果存在）
                pending_meta = self._get_pending_order_meta(oid_s)

                # 计算已成交的增量 (delta)
                prev_filled = 0.0
                if pending_meta:
                    prev_filled = float(pending_meta.get('filled_quantity', 0.0) or 0.0)

                newly_filled = max(0.0, filled - prev_filled)

                # 提取手续费信息（如果存在）
                commission_coin = e.get('CommissionCoin') or e.get('commission_coin') or None
                commission_value = e.get('CommissionChargeValue') or e.get('commissionchargevalue') or e.get('CommissionChargeValue'.lower()) or 0.0
                commission_percent = e.get('CommissionPercent') or e.get('commission_percent') or None
                try:
                    commission_value = float(commission_value or 0.0)
                except Exception:
                    commission_value = 0.0
                prev_commission = 0.0
                if pending_meta:
                    try:
                        prev_commission = float(pending_meta.get('commission_value', 0.0) or 0.0)
                    except Exception:
                        prev_commission = 0.0
                commission_delta = max(0.0, commission_value - prev_commission)

                # 情况 1：订单完全成交（remaining <= 0.001 或 status == FILLED）
                if remaining <= 0.001 or status == 'FILLED':
                    try:
                        # 如果有之前部分已登记，仅注册未登记的增量
                        if newly_filled > 0:
                            self.portfolio.register_order_execution(coin, strat, side, newly_filled, price,
                                                                    fee_amount=commission_delta,
                                                                    fee_currency=commission_coin,
                                                                    fee_percent=commission_percent)
                        # 如果没有增量但仍认为已完成（edge case），尝试注册整个已成交量以保证一致性
                        elif not pending_meta:
                            self.portfolio.register_order_execution(coin, strat, side, filled, price,
                                                                    fee_amount=commission_value,
                                                                    fee_currency=commission_coin,
                                                                    fee_percent=commission_percent)
                    except Exception:
                        self.logger.exception(f"Failed to register_order_execution for {oid_s}")

                    # 从待处理队列移除
                    try:
                        self._remove_pending_order_by_id(oid_s)
                    except Exception:
                        self.logger.exception(f"Failed to remove pending order {oid_s} after full fill")

                    processed.append({
                        'order_id': oid_s,
                        'status': status,
                        'processed': True,
                        'type': 'full_fill',
                        'filled_qty': filled,
                        'remaining_qty': 0.0
                    })

                # 情况 2：订单部分成交（0 < filled < orig_qty，且 remaining > 0.001）
                elif filled > 0.0 and remaining > 0.001:
                    try:
                        # 仅注册未登记的增量
                        if newly_filled > 0:
                            self.portfolio.register_order_execution(coin, strat, side, newly_filled, price,
                                                                    fee_amount=commission_delta,
                                                                    fee_currency=commission_coin,
                                                                    fee_percent=commission_percent)
                        # 更新待处理订单的元数据（累计已成交量）
                        self._update_pending_order_by_id(oid_s, {
                            'filled_quantity': filled,
                            'remaining_quantity': remaining,
                            'commission_value': commission_value,
                            'commission_coin': commission_coin,
                            'commission_percent': commission_percent,
                            'last_update': int(time.time() * 1000)
                        })
                    except Exception:
                        self.logger.exception(f"Failed to update/register pending order {oid_s} after partial fill")

                    processed.append({
                        'order_id': oid_s,
                        'status': status,
                        'processed': True,
                        'type': 'partial_fill',
                        'filled_qty': filled,
                        'remaining_qty': remaining
                    })

                # 情况 3：订单被取消或其他状态，暂不处理
                elif status == 'CANCELED':
                    # 可选：取消订单，释放锁
                    try:
                        self.portfolio.release_coin(coin, strategy_id=strat, force=False)
                    except Exception:
                        self.logger.exception(f"Failed to release lock for {coin} after cancel")
                    
                    try:
                        self._remove_pending_order_by_id(oid_s)
                    except Exception:
                        pass

                    processed.append({
                        'order_id': oid_s,
                        'status': status,
                        'processed': True,
                        'type': 'canceled'
                    })
                else:
                    # 订单状态未知或未成交
                    processed.append({
                        'order_id': oid_s,
                        'status': status,
                        'processed': False,
                        'type': 'no_change',
                        'filled_qty': filled,
                        'remaining_qty': remaining
                    })
            else:
                # 这个订单不在我们的待处理列表中
                processed.append({
                    'order_id': oid_s,
                    'status': status,
                    'processed': False,
                    'reason': 'order_not_tracked'
                })

        return processed
