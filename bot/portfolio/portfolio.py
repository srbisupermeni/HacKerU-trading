import logging
import time
from typing import Dict, Any

# Attempt to import Roostoo client for on-demand ticker fetching
try:
    from bot.api.roostoo import Roostoo
except Exception:
    Roostoo = None


class Portfolio:
    """
    资产组合管理器 - 管理所有交易账户的资产状态
    
    核心功能：
        1. 管理持仓（每个币种的可用/锁定金额）
        2. 管理策略参数（每个币种的止损/止盈设置）
        3. 管理成本基础与 PnL

    说明：
        - 组合层币种锁逻辑已停用，策略层自行控制币种分配
        - 保留 acquire/release 等方法仅用于兼容旧调用
    
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

            self.logger.info(
                f"[BUY] {coin}: +{net_qty} (gross {qty}) @ {price}, 总成本={self.cost_basis[coin]['total_cost']}")

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

    def update_market_prices(self):
        """
        自动从 Roostoo 拉取所有交易对的行情并更新当前持仓/成本基础中币种的价格，进而更新未实现 PnL。

        说明：方法不接受参数，会优先使用挂载在 self.execution.client 的 Roostoo 实例，
        如果不存在则尝试按需实例化 `bot.api.roostoo.Roostoo`。
        """
        # 尝试通过 Roostoo 客户端拉取所有交易对行情
        roostoo_client = None
        # 优先使用 execution 已有的 client（ExecutionEngine 通常会挂载 client）
        if getattr(self, 'execution', None) is not None:
            roostoo_client = getattr(self.execution, 'client', None)

        # 否则尝试按需实例化 Roostoo（如果可用）
        if roostoo_client is None and Roostoo is not None:
            try:
                roostoo_client = Roostoo()
            except Exception as e:
                self.logger.warning(f"无法实例化 Roostoo 客户端: {e}")

        if roostoo_client is None:
            self.logger.error("没有可用的 Roostoo 客户端，无法自动更新行情。")
            return False

        # 获取所有 ticker
        resp = roostoo_client.get_ticker()
        if not resp or not isinstance(resp, dict):
            self.logger.warning(f"从 Roostoo 获取 ticker 失败或返回格式不正确: {resp}")
            return False

        data = resp.get('Data') or resp.get('data') or {}

        # 构建需要更新的币种列表：优先为当前持仓与已知成本基础中的币种
        coins_to_update = set(self.positions.keys()) | set(self.cost_basis.keys()) | set(self.market_prices.keys())
        if not coins_to_update:
            # 如果当前没有持仓，则尝试从 trade_pairs 中挑选可交易的部分币种以更新其行情
            coins_to_update = {info.get('coin') for info in self.trade_pairs.values() if info.get('coin')}

        updated = {}
        for coin in coins_to_update:
            if not coin:
                continue

            # 常见 pair 命名形式: 'BTC/USD'
            pair_name = f"{coin}/USD"
            ticker = None

            # 直接尝试以 COIN/USD 查找
            if isinstance(data, dict) and pair_name in data:
                ticker = data.get(pair_name)
            else:
                # 尝试在 trade_pairs 中查找匹配的交易对名
                for p, info in self.trade_pairs.items():
                    if info.get('coin') and info.get('coin').upper() == coin.upper():
                        if p in data:
                            ticker = data.get(p)
                            pair_name = p
                            break

            # 如果还是没找到，则跳过
            if not ticker:
                self.logger.debug(f"未在行情数据中找到对应交易对: {coin} (tried {pair_name})")
                continue

            # 解析 last price 字段并更新
            last_price = None
            if isinstance(ticker, dict):
                # 常见字段名称: 'LastPrice'
                last_price = ticker.get('LastPrice') or ticker.get('lastPrice') or ticker.get('price')
            else:
                # 如果 ticker 不是 dict（不常见），尝试直接作为价格
                last_price = ticker

            try:
                price_val = float(last_price)
            except Exception:
                self.logger.warning(f"无法解析 {pair_name} 的价格: {last_price}")
                continue

            self.market_prices[coin] = price_val
            self._update_unrealized_pnl(coin)
            updated[coin] = price_val

        if updated:
            self.logger.info(f"自动更新市场价格: {updated}")
            return True
        else:
            self.logger.info("未找到可更新的币种行情。")
            return False

    def set_market_prices(self, prices: Dict[str, float]) -> bool:
        """
        手动设置市场价格（用于测试/模拟），并更新未实现 PnL。

        参数：
            prices: {'BTC': 55000.0, 'ETH': 2000.0, ...}
        """
        if not isinstance(prices, dict):
            self.logger.error("set_market_prices 需要一个 dict 参数")
            return False

        updated = {}
        for coin, price in prices.items():
            if not coin:
                continue
            try:
                price_val = float(price)
            except Exception:
                self.logger.warning(f"忽略无法解析的价格: {coin}={price}")
                continue

            self.market_prices[coin] = price_val
            self._update_unrealized_pnl(coin)
            updated[coin] = price_val

        if updated:
            self.logger.info(f"手动设置市场价格: {updated}")
            return True
        else:
            self.logger.info("没有有效的手动价格被设置。")
            return False

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
