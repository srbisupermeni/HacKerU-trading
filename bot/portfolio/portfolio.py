import threading
import time
import logging
from typing import Dict, Optional


class Portfolio:
    """
    资产组合管理器 - 管理所有交易账户的资产状态
    
    核心功能：
        1. 管理持仓（每个币种的可用/锁定金额）
        2. 管理币种锁（防止同一币种被多个策略同时交易）
        3. 管理策略参数（每个币种的止损/止盈设置）
    
    锁机制说明（应用级别）：
        - 每个币种都有一个"锁"，防止多个策略同时下单
        - 锁是应用级别的，安全、高效、易于恢复
        - 线程安全：可以从任何线程释放锁
    
    使用示例：
        portfolio = Portfolio(execution_module=engine)
        if portfolio.acquire_coin('BTC', 'strategy_1'):
            # 成功获得 BTC 的锁，可以下单
            pass
        else:
            # BTC 被其他策略占用，请稍后重试
            pass
    """

    def __init__(self, execution_module):
        """
        :param execution_module: 对应的 Execution 类实例引用
        """
        self.execution = execution_module
        self.logger = logging.getLogger('Portfolio')

        self.account_balance: float = 1000.0

        # 当前持仓：{coin: {'free': float, 'locked': float}}
        self.positions: Dict[str, Dict[str, float]] = {}

        # 应用级锁管理：{coin: {'locked': bool, 'owner_strategy_id': str, 'acquired_at': timestamp}}
        self.coin_ownership: Dict[str, Dict] = {}
        # 保护 coin_ownership 字典访问的全局互斥锁
        self._ownership_lock = threading.Lock()

        # 策略相关参数：{coin: {'strategy_id': str, 'stop_loss': float, 'take_profit': float}}
        self.strategy_params: Dict[str, Dict] = {}

        # 用于后台监控止损/止盈的线程
        self.monitoring_thread: Optional[threading.Thread] = None
        self.monitoring_active = False

    # ================= 持仓管理 =================

    def update_positions(self, new_positions: Dict[str, Dict[str, float]] = None):
        """
        批量更新持仓信息（从交易所同步）
        
        参数：
            new_positions: 新的持仓字典 {coin: {'free': float, 'locked': float}, ...}
        
        示例：
            portfolio.update_positions({
                'BTC': {'free': 1.5, 'locked': 0.5},
                'ETH': {'free': 10.0, 'locked': 0.0}
            })
        """
        self.positions = new_positions or self.positions

    def get_position(self, coin: str) -> Dict[str, float]:
        """
        获取指定币种的当前持仓
        
        参数：
            coin: 币种代码，例如 'BTC'
        
        返回：
            {'free': 可用数量, 'locked': 锁定数量}
        
        示例：
            pos = portfolio.get_position('BTC')
            print(f"可用 BTC: {pos['free']}, 锁定: {pos['locked']}")
        """
        return self.positions.get(coin, {"free": 0.0, "locked": 0.0})

    # ================= 币种锁管理 =================

    def acquire_coin(self, coin: str, strategy_id: str) -> bool:
        """
        获取币种的锁（防止其他策略同时交易此币种）
        
        非阻塞式调用 - 立即返回结果
        
        参数：
            coin: 币种代码，例如 'BTC', 'ETH', 'XRP'
            strategy_id: 策略的唯一标识，例如 'ma_crossover_1', 'rsi_strategy'
        
        返回：
            True - 成功获取锁，可以下单
            False - 币种被其他策略占用，请稍后重试
        
        示例：
            if portfolio.acquire_coin('BTC', 'my_strategy'):
                print("成功获取 BTC 的锁")
                # 继续下单...
            else:
                print("BTC 被其他策略占用，请稍后重试")
        """
        with self._ownership_lock:
            ownership = self.coin_ownership.get(coin, {})
            
            # 如果币种已被锁定，返回 False（非阻塞）
            if ownership.get('locked', False):
                self.logger.debug(f"Coin {coin} is already locked by {ownership.get('owner_strategy_id')}")
                return False
            
            # 锁定币种，记录所有者和获取时间戳
            self.coin_ownership[coin] = {
                'locked': True,
                'owner_strategy_id': strategy_id,
                'acquired_at': time.time()
            }
            
            self.logger.debug(f"Acquired lock for coin {coin} by strategy {strategy_id}")
            return True

    def release_coin(self, coin: str, strategy_id: str = None, force: bool = False) -> bool:
        """
        释放币种的锁（完成下单后调用）
        
        参数：
            coin: 币种代码，例如 'BTC', 'ETH'
            strategy_id: 可选，建议传入你的策略 ID 以验证所有权
            force: 仅供系统级别使用，不建议策略层调用
        
        返回：
            True - 成功释放锁
            False - 释放失败（通常是所有权不匹配）
        
        示例：
            portfolio.release_coin('BTC', strategy_id='my_strategy')
        """
        with self._ownership_lock:
            ownership = self.coin_ownership.get(coin, {})
            
            # 如果币种未被锁定，无操作
            if not ownership.get('locked', False):
                self.logger.debug(f"Coin {coin} is not locked, nothing to release")
                return True
            
            owner = ownership.get('owner_strategy_id')
            
            # 检查所有权：如果指定了 strategy_id 且不匹配，且非强制模式，则拒绝
            if strategy_id is not None and owner != strategy_id and not force:
                self.logger.warning(
                    f"Cannot release coin {coin}: owned by {owner}, requested by {strategy_id}"
                )
                return False
            
            # 释放锁
            self.coin_ownership[coin] = {
                'locked': False,
                'owner_strategy_id': None,
                'acquired_at': None
            }
            
            self.logger.debug(f"Released lock for coin {coin} (was owned by {owner})")
            return True

    # ================= 策略参数管理 =================

    def set_strategy_params(self, coin: str, strategy_id: str, stop_loss: float, take_profit: float):
        """
        为币种设置止损和止盈价格（下单时设置）
        
        参数：
            coin: 币种代码，例如 'BTC'
            strategy_id: 你的策略 ID
            stop_loss: 止损价格（触发卖出的最低价格）
            take_profit: 止盈价格（触发卖出的最高价格）
        
        示例：
            portfolio.set_strategy_params('BTC', 'my_strategy', stop_loss=30000, take_profit=50000)
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
        获取币种的策略参数（止损、止盈价格）
        
        参数：
            coin: 币种代码
        
        返回：
            包含 'strategy_id', 'stop_loss', 'take_profit' 的字典
        
        示例：
            params = portfolio.get_strategy_params('BTC')
            print(f"止损价格: {params.get('stop_loss')}")
        """
        return self.strategy_params.get(coin, {})

    def get_coin_lock_status(self, coin: str) -> Dict:
        """
        获取某个币种的锁定状态。
        返回 {coin: ..., locked: bool, owner_strategy_id: str, acquired_at: float}
        """
        with self._ownership_lock:
            ownership = self.coin_ownership.get(coin, {})
            return {
                'coin': coin,
                'locked': ownership.get('locked', False),
                'owner_strategy_id': ownership.get('owner_strategy_id'),
                'acquired_at': ownership.get('acquired_at')
            }

    def force_release_coin(self, coin: str) -> bool:
        """
        强制释放币种锁（仅供管理员/恢复使用）。
        不检查所有权，直接解除锁定。
        """
        with self._ownership_lock:
            if coin in self.coin_ownership:
                owner = self.coin_ownership[coin].get('owner_strategy_id')
                self.coin_ownership[coin] = {
                    'locked': False,
                    'owner_strategy_id': None,
                    'acquired_at': None
                }
                self.logger.warning(
                    f"Force-released lock for coin {coin} (was owned by {owner}) - admin intervention"
                )
                return True
        return False

    def get_all_lock_status(self) -> Dict[str, Dict]:
        """获取所有币种的锁定状态快照。"""
        with self._ownership_lock:
            return {
                coin: {
                    'locked': info.get('locked', False),
                    'owner_strategy_id': info.get('owner_strategy_id'),
                    'acquired_at': info.get('acquired_at')
                }
                for coin, info in self.coin_ownership.items()
            }

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
        在订单被成交后由执行引擎调用（策略层通常不直接调用）。
        更新持仓，并在订单完全成交时自动释放锁。
        
        参数：
            coin: 币种代码
            strategy_id: 策略 ID
            side: 'BUY' 或 'SELL'
            qty: 成交数量
            price: 成交价格
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
        self.logger.info(f"Updated position for {coin}: {pos} (side={s}, qty={qty})")

        # 如果该策略拥有该币种锁，则释放锁
        # 这是为了在订单被成交时自动释放币种所有权
        ownership = self.coin_ownership.get(coin, {})
        if ownership.get('owner_strategy_id') == strategy_id:
            success = self.release_coin(coin, strategy_id=strategy_id)
            if success:
                self.logger.info(f"Released lock for coin {coin} after execution by {strategy_id}")
            else:
                self.logger.warning(f"Failed to release lock for coin {coin} (strategy_id mismatch)")
        
        return True

