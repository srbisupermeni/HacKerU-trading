import threading
import time
from typing import Dict, Optional


class Portfolio:
    """
    Portfolio 类负责管理机器人交易的所有资产。
    职责：
        - 跟踪每个币种的持仓
        - 存储每个币种的止损 / 止盈参数
        - 防止多个策略同时交易同一币种（使用锁）
        - 可由 Roostoo 更新持仓信息
        - 通过 Execution 模块触发止损 / 止盈
    """

    def __init__(self, execution_module):
        """
        :param execution_module: 对应的 Execution 类实例引用
        """
        self.execution = execution_module

        self.account_balance: float = 1000.0

        # 当前持仓：{coin: {'free': float, 'locked': float}}
        self.positions: Dict[str, Dict[str, float]] = {}

        # 每个币种的锁，防止多个策略同时使用同一币种
        self.coin_locks: Dict[str, threading.Lock] = {}

        # 策略相关参数：{coin: {'strategy_id': str, 'stop_loss': float, 'take_profit': float}}
        self.strategy_params: Dict[str, Dict] = {}

        # 用于后台监控止损/止盈的线程
        self.monitoring_thread: Optional[threading.Thread] = None
        self.monitoring_active = False

    # ================= 持仓管理 =================

    def update_positions(self, new_positions: Dict[str, Dict[str, float]] = None):
        # 更新持仓（可以被外部同步调用）
        self.positions = new_positions or self.positions

    def get_position(self, coin: str) -> Dict[str, float]:
        """
        返回指定币种的当前持仓
        """
        return self.positions.get(coin, {"free": 0.0, "locked": 0.0})

    # ================= 币种锁管理 =================

    def acquire_coin(self, coin: str, strategy_id: str) -> bool:
        """
        尝试为某个策略锁定币种。
        成功返回 True，失败返回 False。
        """
        # 确保此币种存在 Lock 对象
        if coin not in self.coin_locks:
            self.coin_locks[coin] = threading.Lock()

        lock = self.coin_locks[coin]

        # 非阻塞地尝试获取锁
        acquired = lock.acquire(blocking=False)
        if not acquired:
            return False

        # 记录当前持有锁的策略（所有者信息保存在 strategy_params 中）
        existing = self.strategy_params.get(coin, {})
        existing['strategy_id'] = strategy_id
        # 确保 stop_loss / take_profit 键存在（可能稍后设置）
        existing.setdefault('stop_loss', None)
        existing.setdefault('take_profit', None)
        self.strategy_params[coin] = existing

        return True

    def release_coin(self, coin: str):
        """
        释放某个币种的锁
        """
        lock = self.coin_locks.get(coin)
        if lock and lock.locked():
            try:
                lock.release()
            except RuntimeError:
                # 已经被释放或当前线程不是持有者，忽略
                return

        # 移除策略的所有者标记，但保留 stop_loss/take_profit（如存在）
        if coin in self.strategy_params:
            params = self.strategy_params.get(coin, {})
            # 仅删除 owner 标记
            params.pop('strategy_id', None)

            # 如果 stop_loss 和 take_profit 都为空，则移除整个条目
            if params.get('stop_loss') is None and params.get('take_profit') is None:
                self.strategy_params.pop(coin, None)
            else:
                self.strategy_params[coin] = params

    # ================= 策略参数管理 =================

    def set_strategy_params(self, coin: str, strategy_id: str, stop_loss: float, take_profit: float):
        """
        为某个币种在指定策略下登记止损和止盈参数
        """
        params = self.strategy_params.get(coin, {})
        owner = params.get('strategy_id')
        if owner and owner != strategy_id:
            # 不允许覆盖被其他策略拥有的参数
            raise PermissionError(f"Strategy {strategy_id} cannot set params for {coin} owned by {owner}")

        params['strategy_id'] = strategy_id
        params['stop_loss'] = stop_loss
        params['take_profit'] = take_profit
        self.strategy_params[coin] = params

    def get_strategy_params(self, coin: str) -> Dict:
        """
        返回指定币种的策略参数（止损、止盈）
        """
        return self.strategy_params.get(coin, {})

    # ================= 订单监控 =================

    def start_monitoring(self, interval: float = 2.0):
        """
        启动后台线程，用于监控止损/止盈触发
        """
        if self.monitoring_thread is None or not self.monitoring_thread.is_alive():
            self.monitoring_active = True
            self.monitoring_thread = threading.Thread(target=self._monitor_loop, args=(interval,))
            self.monitoring_thread.start()

    def stop_monitoring(self):
        """
        停止后台监控线程
        """
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join()

    def _monitor_loop(self, interval: float):
        """
        内部循环：监控持仓并在触发止损/止盈时调用 Execution 下单
        """
        while self.monitoring_active:
            # TODO: 根据每个币种的当前价格与止损/止盈比较，触发相应的下单
            try:
                # 最小睡眠以避免忙等；实际实现应拉取价格数据并判断
                time.sleep(interval)
            except Exception:
                break

    # ================= 订单执行更新 =================

    def register_order_execution(self, coin: str, strategy_id: str, side: str, qty: float, price: float):
        """
        在订单被成交后由 Execution 调用。
        更新持仓，并在需要时释放锁。
        """
        # 如果缺少持仓记录则初始化
        pos = self.positions.get(coin, {'free': 0.0, 'locked': 0.0})

        # 标准化 side
        s = side.upper()
        if s == 'BUY':
            # 已成交的买单视为可用（free）余额增加
            pos['free'] = pos.get('free', 0.0) + float(qty)
        elif s == 'SELL':
            # 卖出优先从 free 扣除，其次从 locked 扣除
            remaining = float(qty)
            free = pos.get('free', 0.0)
            if free >= remaining:
                pos['free'] = free - remaining
            else:
                pos['free'] = 0.0
                locked = pos.get('locked', 0.0)
                pos['locked'] = max(0.0, locked - (remaining - free))
        else:
            # 未知方向则不处理
            pass

        self.positions[coin] = pos

        # 如果该策略曾拥有该币种锁，则在成交后释放锁（这是常见模式）
        params = self.strategy_params.get(coin, {})
        owner = params.get('strategy_id')
        if owner == strategy_id:
            # 保留 stop_loss/stop_profit 等参数，但释放锁
            try:
                self.release_coin(coin)
            except Exception:
                # 忽略释放时的错误
                pass

        # 可选：通知 execution 模块或记录日志（execution 持有引用）
        return True

