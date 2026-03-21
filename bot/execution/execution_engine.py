import logging
import threading
import time
import uuid
from typing import Optional, Dict, List, Any

from bot.api.roostoo import Roostoo
from bot.portfolio.portfolio import Portfolio


class ExecutionEngine:
    """
    执行引擎（ExecutionEngine）——负责与交易所客户端交互、下单并管理未完成订单队列。

    核心职责：
    - 使用 `Roostoo` 客户端发起下单请求。
    - 解析交易所返回的 OrderDetail / OrderMatched，区分完全成交、部分成交与挂单情形。
    - 在完全成交时调用 `Portfolio.register_order_execution` 更新账本；在部分成交或挂单时，将订单加入本地待处理队列，供后台轮询更新。

    设计要点：
    - 对 Roostoo 的返回格式使用严格解析器 `_parse_order_obj`，遇到不合规响应会返回错误信息而非隐式忽略。
    - 待处理订单由 `pending_orders_queue` 管理，按 `"{coin}:{strategy_id}"` 分队列索引，并通过 `_orderid_index` 快速定位。
    - 该类仅处理下单与订单生命周期相关的逻辑；资金/持仓记账由 `Portfolio` 负责。
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

    # --- Exceptions / Parsers for strict Roostoo API schema ---
    class NonConformingRoostooResponse(ValueError):
        """Raised when a response does not match the canonical Roostoo API schema."""
        pass

    def _parse_order_obj(self, obj: Dict[str, Any]) -> Dict[str, Any]:
        """
        解析单个 Roostoo 订单对象（OrderDetail 或 OrderMatched 的元素），并返回规范化字典。

        要求字段（必须存在）：'OrderID', 'Status', 'Quantity', 'FilledQuantity'
        可选字段（存在时会尝试解析）：'FilledAverPrice', 'Price', 'CommissionChargeValue', 'CommissionCoin', 'CommissionPercent', 'Side'
        解析失败会抛出 NonConformingRoostooResponse。
        """
        if not isinstance(obj, dict):
            raise self.NonConformingRoostooResponse('Order object is not a dict')

        required = ['OrderID', 'Status', 'Quantity', 'FilledQuantity']
        missing = [k for k in required if k not in obj]
        if missing:
            raise self.NonConformingRoostooResponse(f"Missing keys in Roostoo order object: {missing}")

        try:
            order_id = str(obj['OrderID'])
            status = str(obj['Status']).upper() if obj['Status'] is not None else ''
            orig_qty = float(obj['Quantity'])
            filled_qty = float(obj.get('FilledQuantity') or 0.0)
        except Exception as e:
            raise self.NonConformingRoostooResponse(f"Failed to coerce required order fields: {e}")

        # execution price preference: FilledAverPrice then Price
        exec_price = None
        if 'FilledAverPrice' in obj and obj.get('FilledAverPrice') is not None:
            try:
                exec_price = float(obj.get('FilledAverPrice'))
            except Exception:
                exec_price = None
        elif 'Price' in obj and obj.get('Price') is not None:
            try:
                exec_price = float(obj.get('Price'))
            except Exception:
                exec_price = None

        # commission fields
        commission_value = 0.0
        if 'CommissionChargeValue' in obj and obj.get('CommissionChargeValue') is not None:
            try:
                commission_value = float(obj.get('CommissionChargeValue') or 0.0)
            except Exception:
                commission_value = 0.0

        commission_coin = obj.get('CommissionCoin') if 'CommissionCoin' in obj else None
        commission_percent = obj.get('CommissionPercent') if 'CommissionPercent' in obj else None
        side = obj.get('Side') if 'Side' in obj else None

        return {
            'order_id': order_id,
            'status': status,
            'original_quantity': orig_qty,
            'filled_quantity': filled_qty,
            'exec_price': exec_price,
            'commission_value': commission_value,
            'commission_coin': commission_coin,
            'commission_percent': commission_percent,
            'side': side,
            'raw': obj,
        }

    def create_order(self, coin: str, side: str, quantity: float, price: Optional[float] = None,
                      order_type: Optional[str] = None, strategy_id: str = None,
                      client_order_id: Optional[str] = None) -> Dict[str, Any]:
        """
        创建一个标准化的本地订单对象（便于策略层构造并传入 execute_order）。

        返回的字典包含：client_order_id, coin, side, quantity, price, order_type, strategy_id, created_at, status。
        该方法做基础参数校验与规范化，但不与交易所交互。
        """

        if not coin:
            raise ValueError('coin is required')
        if not side:
            raise ValueError('side is required')
        if quantity is None:
            raise ValueError('quantity is required')

        # normalize
        side_norm = str(side).upper()
        try:
            qty = float(quantity)
        except Exception:
            raise ValueError('quantity must be numeric')

        # decide default order_type similar to execute_order
        if order_type is None:
            order_type = 'LIMIT' if price is not None else 'MARKET'

        cid = client_order_id or f"cli_{int(time.time()*1000)}_{uuid.uuid4().hex[:8]}"

        order = {
            'client_order_id': cid,
            'coin': coin,
            'side': side_norm,
            'quantity': qty,
            'price': float(price) if price is not None else None,
            'order_type': order_type,
            'strategy_id': strategy_id,
            'created_at': int(time.time() * 1000),
            'status': 'CREATED'
        }

        return order

    def execute_order(self, order: Dict[str, Any]) -> Dict:
        """
        执行下单：将订单发送到交易所并根据返回结果做后续处理。

        行为说明：
        - 调用 `Roostoo.place_order` 提交订单。
        - 使用 `_parse_order_obj` 解析交易所返回的 OrderDetail/OrderMatched。解析失败会以错误形式返回。
        - 如果订单完全成交（状态为 FILLED 或剩余量接近 0），直接调用 `Portfolio.register_order_execution` 更新账本并返回成功。
        - 如果订单部分成交或为挂单（PENDING / MAKER），将订单加入 `pending_orders_queue`，返回 queued=True 以便后台轮询处理后续成交。

        返回值为字典，包含 success、order_id、status、filled_qty、remaining_qty、queued（若加入待处理队列）及原始响应 raw。
        错误时返回 success=False 并提供 error/message 字段。
        """

        # This interface strictly expects an order dict produced by create_order().
        if not isinstance(order, dict):
            raise ValueError('execute_order expects an order dict produced by create_order()')

        # Minimal guard to ensure order looks like it was created by create_order()
        required_keys = ['client_order_id', 'coin', 'side', 'quantity']
        missing = [k for k in required_keys if k not in order]
        if missing:
            raise ValueError(f'order dict missing required keys: {missing}')

        # Extract fields
        coin = order.get('coin')
        side = order.get('side')
        quantity = order.get('quantity')
        price = order.get('price') if 'price' in order else None
        order_type = order.get('order_type') if 'order_type' in order else None
        strategy_id = order.get('strategy_id') if 'strategy_id' in order else None

        # 策略层已自行管理币种所有权；执行层不再做应用级币种锁控制

        # 确定 order_type 的默认值
        if order_type is None:
            order_type = 'LIMIT' if price is not None else 'MARKET'

        # 通过 Roostoo 客户端下单
        try:
            result = self.client.place_order(coin, side, quantity, price=price, order_type=order_type)
        except Exception as e:
            return {'success': False, 'error': 'client_exception', 'message': str(e)}

        if not result or not isinstance(result, dict) or not result.get('Success'):
            return {'success': False, 'error': 'order_failed', 'message': 'Order placement failed', 'raw': result}

        # 成功时，提取有用字段并登记
        # Parse canonical OrderDetail / OrderMatched using strict parser
        try:
            parsed = None
            if isinstance(result, dict):
                od = result.get('OrderDetail')
                if isinstance(od, dict):
                    parsed = self._parse_order_obj(od)
                else:
                    om = result.get('OrderMatched')
                    if isinstance(om, list) and len(om) > 0 and isinstance(om[0], dict):
                        parsed = self._parse_order_obj(om[0])

            if parsed is None:
                return {'success': False, 'error': 'non_conforming_response', 'message': 'Response missing canonical OrderDetail or OrderMatched objects', 'raw': result}
        except self.NonConformingRoostooResponse as e:
            return {'success': False, 'error': 'non_conforming_response', 'message': str(e), 'raw': result}

        # Use parsed canonical fields
        order_id = parsed.get('order_id')
        status = parsed.get('status')
        filled_qty = parsed.get('filled_quantity', 0.0)
        orig_qty = parsed.get('original_quantity', quantity)
        exec_price = parsed.get('exec_price') if parsed.get('exec_price') is not None else price
        commission_coin = parsed.get('commission_coin')
        commission_value = parsed.get('commission_value', 0.0)
        commission_percent = parsed.get('commission_percent')

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
            # 在这种情况下，注册执行
            if st == 'FILLED' or remaining_qty <= 0.001:  # 允许 0.001 的浮点误差
                self.portfolio.register_order_execution(coin, strategy_id, side, filled_qty or quantity,
                                                        exec_price,
                                                        fee_amount=(commission_value if 'commission_value' in locals() else 0.0),
                                                        fee_currency=(commission_coin if 'commission_coin' in locals() else None),
                                                        fee_percent=(commission_percent if 'commission_percent' in locals() else None))

                # If this order was previously indexed as pending (defensive), remove it
                if order_id_str:
                    try:
                        self._remove_pending_order_by_id(order_id_str)
                    except Exception:
                        pass

                return {'success': True, 'order_id': order_id, 'status': status, 'filled_qty': filled_qty,
                        'remaining_qty': 0.0, 'raw': result}

            # 部分成交的情况（有一些成交但还有剩余）
            # 仅更新待处理队列，组合账本仅在完全成交时更新
            elif filled_qty > 0.0 and remaining_qty > 0.001:

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

                # 返回部分成交信息
                return {'success': True, 'order_id': order_id, 'status': status, 'filled_qty': filled_qty,
                        'remaining_qty': remaining_qty, 'queued': True, 'raw': result}

            # 如果订单为 PENDING（挂单 / MAKER），将订单加入 pending_orders_queue 并设置策略参数
            elif st == 'PENDING' or (parsed and parsed.get('raw', {}).get('Role') == 'MAKER'):
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

                # 返回已排队信息
                return {'success': True, 'order_id': order_id, 'status': status, 'queued': True, 'raw': result}

            else:
                return {'success': False, 'order_id': order_id, 'status': status, 'raw': result}

        except Exception as e:
            # 登记或后处理出错
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
        返回当前待处理订单的快照（只读）。

        结果格式：{ "{coin}:{strategy_id}": [order_meta, ...], ... }
        每个 order_meta 包含 order_id、side、original_quantity、filled_quantity、remaining_quantity、price、created_at、raw 等字段。
        """
        with self._pq_lock:
            return {k: list(v) for k, v in self.pending_orders_queue.items()}

    def process_query_response(self, response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        处理交易所的订单查询响应（用于后台轮询结果的汇总处理）。

        功能：对返回的 OrderMatched / OrderDetail 条目逐个解析，
        - 若订单已完全成交：调用 `Portfolio.register_order_execution`，并从 pending 队列移除；返回 type='full_fill'.
        - 若订单部分成交：更新 pending 队列中的已成交/剩余数量；返回 type='partial_fill'.
        - 若订单被取消：从队列移除并返回 type='canceled'.
        - 若该订单不在本地追踪集合中：返回 reason='order_not_tracked'.

        返回值：处理结果列表，每项包含 order_id、status、processed、type、filled_qty、remaining_qty 等字段。
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
            # Use strict parser for each entry
            try:
                parsed = self._parse_order_obj(e)
            except self.NonConformingRoostooResponse:
                # Skip non-conforming entries but record a processed entry indicating non-conformance
                oid = e.get('OrderID') or e.get('order_id')
                oid_s = str(oid) if oid is not None else 'unknown'
                processed.append({'order_id': oid_s, 'status': None, 'processed': False, 'reason': 'non_conforming_entry'})
                continue

            oid_s = str(parsed.get('order_id'))
            status = (parsed.get('status') or '').upper()
            filled = parsed.get('filled_quantity', 0.0)
            orig_qty = parsed.get('original_quantity', 0.0)
            remaining = max(0.0, (orig_qty or 0.0) - (filled or 0.0))

            # 检查这个订单是否在我们的待处理列表中
            with self._pq_lock:
                key = self._orderid_index.get(oid_s)

            if key:
                # key 格式是 coin:strategy_id
                try:
                    coin, strat = key.split(':', 1)
                except Exception:
                    coin, strat = key, None

                side = parsed.get('side') or 'BUY'
                price = parsed.get('exec_price') or 0.0

                # 读取当前我们记录的 pending 元数据（如果存在）
                pending_meta = self._get_pending_order_meta(oid_s)

                # 提取手续费信息（如果存在）
                commission_coin = parsed.get('commission_coin')
                commission_value = parsed.get('commission_value', 0.0)
                commission_percent = parsed.get('commission_percent')
                try:
                    commission_value = float(commission_value or 0.0)
                except Exception:
                    commission_value = 0.0

                # 情况 1：订单完全成交（remaining <= 0.001 或 status == FILLED）
                if remaining <= 0.001 or status == 'FILLED':
                    try:
                        booked_qty = filled
                        if booked_qty <= 0.0 and pending_meta:
                            booked_qty = float(pending_meta.get('original_quantity', 0.0) or 0.0)
                        if booked_qty <= 0.0:
                            booked_qty = float(orig_qty or 0.0)
                        # 仅在完全成交时更新组合账本
                        self.portfolio.register_order_execution(coin, strat, side, booked_qty, price,
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
                        # 仅更新待处理订单的元数据（累计已成交量）
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
