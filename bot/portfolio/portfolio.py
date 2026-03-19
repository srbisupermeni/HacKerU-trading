import threading
import time
import logging
from typing import Dict, Optional, Any


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
        portfolio.initialize_from_exchange_info(roostoo_client=engine.client)
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

        # 账户余额：默认不硬编码，建议调用 initialize_from_exchange_info() 初始化
        self.account_currency: str = 'USD'
        self.account_balance: float = 0.0

        # 交易所信息缓存（由 initialize_from_exchange_info() 填充）
        self.exchange_info: Dict[str, Any] = {}
        self.exchange_is_running: bool = False
        self.initial_wallet: Dict[str, float] = {}
        self.trade_pairs: Dict[str, Dict[str, Any]] = {}
        self.tradable_pairs: Dict[str, Dict[str, Any]] = {}

        # ===== 持仓信息 =====
        # 当前持仓：{coin: {'free': float, 'locked': float}}
        self.positions: Dict[str, Dict[str, float]] = {}

        # ===== 成本基础追踪（新增）=====
        # {coin: {'total_quantity': float, 'total_cost': float, 'buy_transactions': [...], 'sell_transactions': [...]}}
        self.cost_basis: Dict[str, Dict] = {}

        # ===== 市场价格（用于计算未实现 PnL）=====
        # {coin: current_price}
        self.market_prices: Dict[str, float] = {}

        # ===== PnL 追踪（新增）=====
        # {coin: {'unrealized_pnl': float, 'realized_pnl': float, ...}}
        self.pnl_tracking: Dict[str, Dict] = {}

        # ===== 应用级锁管理 =====
        # {coin: {'locked': bool, 'owner_strategy_id': str, 'acquired_at': timestamp}}
        self.coin_ownership: Dict[str, Dict] = {}
        # 保护 coin_ownership 字典访问的全局互斥锁
        self._ownership_lock = threading.Lock()

        # ===== 策略参数 =====
        # {coin: {'strategy_id': str, 'stop_loss': float, 'take_profit': float}}
        self.strategy_params: Dict[str, Dict] = {}

        # 用于后台监控止损/止盈的线程
        self.monitoring_thread: Optional[threading.Thread] = None
        self.monitoring_active = False

    def initialize_from_exchange_info(self, roostoo_client=None, exchange_info: Dict[str, Any] = None,
                                      account_currency: str = 'USD', fallback_balance: float = 0.0,
                                      strict: bool = False) -> bool:
        """
        使用 Roostoo 的 exchangeInfo 初始化账户余额和交易对信息。

        参数：
            roostoo_client: Roostoo 客户端实例；如果未传入，会尝试使用 self.execution.client
            exchange_info: 已经获取好的 exchangeInfo 字典；如果传入则直接使用
            account_currency: 余额计价币种，默认 USD
            fallback_balance: 获取失败时的回退余额，默认 0.0
            strict: 是否严格模式；为 True 时，初始化失败会抛出异常

        返回：
            True - 初始化成功
            False - 初始化失败并使用回退值
        """
        try:
            if exchange_info is None:
                if roostoo_client is None and self.execution is not None:
                    roostoo_client = getattr(self.execution, 'client', None)

                if roostoo_client is None:
                    raise ValueError('Roostoo 客户端不存在，无法初始化 Portfolio')

                exchange_info = roostoo_client.get_exchange_info()

            if not isinstance(exchange_info, dict):
                raise ValueError('exchange_info 不是有效的字典')

            self.exchange_info = exchange_info
            self.exchange_is_running = bool(exchange_info.get('IsRunning', False))
            self.account_currency = account_currency

            # 初始化钱包余额
            initial_wallet = exchange_info.get('InitialWallet', {}) or {}
            self.initial_wallet = {}
            for currency, value in initial_wallet.items():
                try:
                    self.initial_wallet[currency] = float(value)
                except Exception:
                    self.logger.warning(f"无法解析初始钱包余额: {currency}={value}")

            balance = self.initial_wallet.get(account_currency)
            if balance is None:
                # 如果指定计价币种不存在，就选择第一个可解析的余额
                balance = next(iter(self.initial_wallet.values()), fallback_balance)

            try:
                self.account_balance = float(balance)
            except Exception:
                self.account_balance = float(fallback_balance)

            # 初始化交易对信息
            trade_pairs = exchange_info.get('TradePairs', {}) or {}
            self.trade_pairs = {}
            self.tradable_pairs = {}

            for pair_name, raw_info in trade_pairs.items():
                if not isinstance(raw_info, dict):
                    continue

                normalized = {
                    'pair': pair_name,
                    'coin': raw_info.get('Coin'),
                    'coin_full_name': raw_info.get('CoinFullName'),
                    'unit': raw_info.get('Unit'),
                    'unit_full_name': raw_info.get('UnitFullName'),
                    'can_trade': bool(raw_info.get('CanTrade', False)),
                }

                try:
                    normalized['price_precision'] = int(raw_info.get('PricePrecision', 0))
                except Exception:
                    normalized['price_precision'] = 0

                try:
                    normalized['amount_precision'] = int(raw_info.get('AmountPrecision', 0))
                except Exception:
                    normalized['amount_precision'] = 0

                try:
                    normalized['mini_order'] = float(raw_info.get('MiniOrder', 0.0))
                except Exception:
                    normalized['mini_order'] = 0.0

                self.trade_pairs[pair_name] = normalized
                if normalized['can_trade']:
                    self.tradable_pairs[pair_name] = normalized

            self.logger.info(
                f"Portfolio 已从 exchangeInfo 初始化：余额={self.account_balance} {self.account_currency}, "
                f"可交易交易对={len(self.tradable_pairs)}"
            )
            return True

        except Exception as e:
            self.logger.warning(f"初始化 Portfolio 失败，使用回退值：{e}")
            if strict:
                raise

            self.account_currency = account_currency
            self.account_balance = float(fallback_balance)
            self.exchange_info = {}
            self.exchange_is_running = False
            self.initial_wallet = {}
            self.trade_pairs = {}
            self.tradable_pairs = {}
            return False

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
        获取指定币种的详细持仓信息（包含成本和收益）
        
        参数：
            coin: 币种代码，例如 'BTC'
        
        返回：
            {
                'free': 可用数量,
                'locked': 锁定数量,
                'total': 总数量,
                'avg_entry_price': 平均买入价,
                'total_cost': 总成本,
                'current_price': 当前价格,
                'unrealized_pnl': 未实现收益,
                'unrealized_pnl_pct': 未实现收益率(%),
                'realized_pnl': 已实现收益,
                'total_pnl': 总收益
            }
        """
        # 获取持仓
        pos = self.positions.get(coin, {'free': 0.0, 'locked': 0.0})
        result = {
            'free': pos.get('free', 0.0),
            'locked': pos.get('locked', 0.0),
            'total': pos.get('free', 0.0) + pos.get('locked', 0.0)
        }
        
        # 获取成本信息
        cb = self.cost_basis.get(coin, {})
        total_qty = cb.get('total_quantity', 0.0)
        total_cost = cb.get('total_cost', 0.0)
        
        result['avg_entry_price'] = (total_cost / total_qty) if total_qty > 0 else 0.0
        result['total_cost'] = total_cost
        
        # 获取当前价格
        current_price = self.market_prices.get(coin, 0.0)
        result['current_price'] = current_price
        
        # 计算未实现 PnL
        if current_price > 0 and total_qty > 0:
            unrealized_value = current_price * total_qty
            unrealized_pnl = unrealized_value - total_cost
            unrealized_pnl_pct = (unrealized_pnl / total_cost * 100) if total_cost > 0 else 0.0
            result['unrealized_pnl'] = unrealized_pnl
            result['unrealized_pnl_pct'] = unrealized_pnl_pct
        else:
            result['unrealized_pnl'] = 0.0
            result['unrealized_pnl_pct'] = 0.0
        
        # 获取已实现 PnL
        pnl = self.pnl_tracking.get(coin, {})
        result['realized_pnl'] = pnl.get('realized_pnl', 0.0)
        
        # 总收益
        result['total_pnl'] = result['unrealized_pnl'] + result['realized_pnl']
        
        return result

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

    def register_order_execution(self, coin: str, strategy_id: str, side: str, qty: float, price: float,
                                 fee_amount: float = 0.0, fee_currency: str = None, fee_percent: float = None):
        """
        在订单被成交后由执行引擎调用，更新持仓、成本和 PnL
        
        参数：
            coin: 币种代码
            strategy_id: 策略 ID
            side: 'BUY' 或 'SELL'
            qty: 成交数量
            price: 成交价格（关键！用于成本计算）
        """
        # ===== 第一步：更新持仓数量 =====
        if coin not in self.positions:
            self.positions[coin] = {'free': 0.0, 'locked': 0.0}
        
        if side.upper() == 'BUY':
            self.positions[coin]['free'] += qty
        else:  # SELL
            self.positions[coin]['free'] -= qty
        
        # ===== 第二步：初始化成本追踪 =====
        if coin not in self.cost_basis:
            self.cost_basis[coin] = {
                'total_quantity': 0.0,
                'total_cost': 0.0,
                'buy_transactions': [],
                'sell_transactions': []
            }
        
        # ===== 第三步：处理 BUY 订单 =====
        if side.upper() == 'BUY':
            # 如果手续费以币种本身计费（fee_currency == coin），则收到的净数量为 qty - fee_amount
            net_qty = qty
            if fee_currency and fee_currency.upper() == coin.upper() and fee_amount:
                net_qty = max(0.0, qty - fee_amount)

            # 计算在报价货币（例如 USD）上的总成本
            # 如果手续费是以报价货币计价（如 USD），则将手续费加到总成本中
            if fee_currency and fee_currency.upper() != coin.upper() and fee_amount:
                total_cost = qty * price + fee_amount
            else:
                total_cost = qty * price

            # 更新成本和数量（使用净数量用于数量统计）
            self.cost_basis[coin]['total_quantity'] += net_qty
            self.cost_basis[coin]['total_cost'] += total_cost

            # 记录交易（包含手续费信息）
            self.cost_basis[coin]['buy_transactions'].append({
                'timestamp': time.time(),
                'quantity': qty,
                'net_quantity': net_qty,
                'price': price,
                'total_cost': total_cost,
                'strategy_id': strategy_id,
                'fee_amount': fee_amount,
                'fee_currency': fee_currency,
                'fee_percent': fee_percent
            })

            self.logger.info(f"[BUY] {coin}: +{net_qty} (gross {qty}) @ {price}, 总成本={self.cost_basis[coin]['total_cost']}")
        
        # ===== 第四步：处理 SELL 订单 =====
        else:  # SELL
            if coin not in self.pnl_tracking:
                self.pnl_tracking[coin] = {
                    'realized_pnl': 0.0,
                    'unrealized_pnl': 0.0,
                    'sell_transactions': []
                }

            # 当前平均成本（基于 total_cost / total_quantity）
            avg_cost = (self.cost_basis[coin]['total_cost'] / self.cost_basis[coin]['total_quantity']) \
                if self.cost_basis[coin]['total_quantity'] > 0 else 0.0

            # 防止卖出数量超过当前持仓：只按可用数量/锁定数量的合计进行结算
            free_qty = float(self.positions[coin].get('free', 0.0))
            locked_qty = float(self.positions[coin].get('locked', 0.0))
            available_qty = max(0.0, free_qty + locked_qty)
            effective_qty = min(float(qty), available_qty)
            if effective_qty < float(qty):
                self.logger.warning(
                    f"[SELL] {coin}: 请求卖出 {qty}，但可用数量只有 {available_qty}，已按 {effective_qty} 处理"
                )

            # 将手续费统一换算为报价货币（例如 USD）以便计算收益：
            fee_in_quote = 0.0
            if fee_amount and fee_currency:
                if fee_currency.upper() == coin.upper():
                    # 手续费以基础币计费，按成交价格换算为报价货币
                    fee_in_quote = fee_amount * price
                else:
                    # 假设非基础币即为报价币（如 USD），直接采用 fee_amount
                    fee_in_quote = fee_amount

            # 计算卖出收益：成交总额（price * qty）减去手续费（以报价计）再减去成本
            proceeds = price * effective_qty - fee_in_quote
            cost_for_qty = avg_cost * effective_qty
            profit = proceeds - cost_for_qty
            profit_pct = ((price - avg_cost) / avg_cost * 100) if avg_cost > 0 else 0.0

            # 更新成本基础（减少已卖出的数量和对应成本）
            self.cost_basis[coin]['total_quantity'] -= effective_qty
            self.cost_basis[coin]['total_cost'] -= (avg_cost * effective_qty)

            # 防止小负数
            if self.cost_basis[coin]['total_quantity'] < 0.001:
                self.cost_basis[coin]['total_quantity'] = 0.0
                self.cost_basis[coin]['total_cost'] = 0.0

            # 更新已实现 PnL
            self.pnl_tracking[coin]['realized_pnl'] += profit

            # 记录交易（包含手续费信息）
            self.pnl_tracking[coin]['sell_transactions'].append({
                'timestamp': time.time(),
                'quantity': effective_qty,
                'requested_quantity': qty,
                'sell_price': price,
                'avg_cost': avg_cost,
                'proceeds': proceeds,
                'fee_amount': fee_amount,
                'fee_currency': fee_currency,
                'profit': profit,
                'profit_pct': profit_pct,
                'strategy_id': strategy_id
            })

            self.logger.info(
                f"[SELL] {coin}: -{effective_qty} @ {price}, 成本={avg_cost:.2f}, 收益={profit:.2f} "
                f"({profit_pct:.2f}%) (fee {fee_amount} {fee_currency})"
            )
        
        # ===== 第五步：计算未实现 PnL =====
        if coin in self.market_prices:
            self._update_unrealized_pnl(coin)
        
        # ===== 第六步：自动释放锁 =====
        ownership = self.coin_ownership.get(coin, {})
        if ownership.get('owner_strategy_id') == strategy_id:
            self.release_coin(coin, strategy_id=strategy_id)
        
        return True
    
    def _update_unrealized_pnl(self, coin: str):
        """计算未实现 PnL（内部方法）"""
        if coin not in self.pnl_tracking:
            self.pnl_tracking[coin] = {
                'realized_pnl': 0.0,
                'unrealized_pnl': 0.0,
                'sell_transactions': []
            }

        # 如果该币种还没有成本基础，说明当前只是市场价格更新，直接置零即可
        if coin not in self.cost_basis:
            self.pnl_tracking[coin]['unrealized_pnl'] = 0.0
            return
        
        total_qty = self.cost_basis[coin]['total_quantity']
        total_cost = self.cost_basis[coin]['total_cost']
        current_price = self.market_prices.get(coin, 0.0)
        
        if total_qty > 0 and current_price > 0:
            unrealized_value = current_price * total_qty
            unrealized_pnl = unrealized_value - total_cost
            self.pnl_tracking[coin]['unrealized_pnl'] = unrealized_pnl
        else:
            self.pnl_tracking[coin]['unrealized_pnl'] = 0.0

    # ================= 市场价格和 PnL 查询 =================

    def update_market_prices(self, prices: Dict[str, float]):
        """
        更新市场价格（用于计算未实现 PnL）
        
        参数：
            prices: {'BTC': 55000.0, 'ETH': 2000.0, ...}
        
        示例：
            portfolio.update_market_prices({'BTC': 55000.0, 'ETH': 2000.0})
        """
        for coin, price in prices.items():
            self.market_prices[coin] = price
            self._update_unrealized_pnl(coin)
        
        self.logger.info(f"更新市场价格: {prices}")
    
    def get_pnl_snapshot(self) -> Dict[str, Any]:
        """
        获取所有币种的 PnL 汇总
        
        返回：
        {
            'BTC': {
                'total_quantity': 1.5,
                'avg_entry_price': 50000.0,
                'current_price': 55000.0,
                'unrealized_pnl': 7500.0,
                'realized_pnl': 1000.0,
                'total_pnl': 8500.0
            },
            ...
            'portfolio_summary': {
                'total_realized_pnl': ...,
                'total_unrealized_pnl': ...,
                'total_pnl': ...
            }
        }
        """
        snapshot = {}
        total_realized = 0.0
        total_unrealized = 0.0
        
        for coin in self.cost_basis.keys():
            cb = self.cost_basis[coin]
            pnl = self.pnl_tracking.get(coin, {})
            
            unrealized = pnl.get('unrealized_pnl', 0.0)
            realized = pnl.get('realized_pnl', 0.0)
            
            snapshot[coin] = {
                'total_quantity': cb['total_quantity'],
                'avg_entry_price': (cb['total_cost'] / cb['total_quantity']) if cb['total_quantity'] > 0 else 0.0,
                'current_price': self.market_prices.get(coin, 0.0),
                'unrealized_pnl': unrealized,
                'realized_pnl': realized,
                'total_pnl': unrealized + realized
            }
            
            total_unrealized += unrealized
            total_realized += realized
        
        snapshot['portfolio_summary'] = {
            'total_realized_pnl': total_realized,
            'total_unrealized_pnl': total_unrealized,
            'total_pnl': total_realized + total_unrealized
        }
        
        return snapshot
    
    def get_transaction_history(self, coin: str = None) -> Dict[str, Any]:
        """
        获取交易历史
        
        参数：
            coin: 可选，指定币种；如果不提供则返回所有币种
        
        返回：
        {
            'BTC': {
                'buy_transactions': [...],
                'sell_transactions': [...]
            },
            ...
        }
        """
        if coin:
            cb = self.cost_basis.get(coin, {})
            return {
                'buy_transactions': cb.get('buy_transactions', []),
                'sell_transactions': cb.get('sell_transactions', [])
            }
        else:
            result = {}
            for c in self.cost_basis.keys():
                cb = self.cost_basis[c]
                result[c] = {
                    'buy_transactions': cb.get('buy_transactions', []),
                    'sell_transactions': cb.get('sell_transactions', [])
                }
            return result
    
    def get_cost_basis(self, coin: str) -> Dict[str, Any]:
        """
        获取币种的成本基础信息
        
        参数：
            coin: 币种代码
        
        返回：
        {
            'total_quantity': 持仓数量,
            'total_cost': 总成本,
            'avg_entry_price': 平均买入价
        }
        """
        cb = self.cost_basis.get(coin, {})
        if not cb:
            return {'total_quantity': 0.0, 'total_cost': 0.0, 'avg_entry_price': 0.0}
        
        return {
            'total_quantity': cb.get('total_quantity', 0.0),
            'total_cost': cb.get('total_cost', 0.0),
            'avg_entry_price': (cb.get('total_cost', 0.0) / cb.get('total_quantity', 1.0)) 
                if cb.get('total_quantity', 0.0) > 0 else 0.0
        }
