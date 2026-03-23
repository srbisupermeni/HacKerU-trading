import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

# ==========================================
# 路径兼容处理
# ==========================================
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from bot.portfolio.portfolio import Portfolio
from bot.execution.execution_engine import ExecutionEngine


class ObiDynamicStrategy:
    """
    OBI 动态标的版：基于 BTC 活跃度在 ETH 和 TAO 之间切换的动量突破策略
    """

    def __init__(self, portfolio: Portfolio, execution: ExecutionEngine, strategy_id="obi_dynamic_01",
                 alloc_ratio=0.50):
        self.portfolio = portfolio
        self.execution = execution
        self.strategy_id = strategy_id
        self.alloc_ratio = alloc_ratio

        self.logger = logging.getLogger("OBI_Dynamic")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            ch = logging.StreamHandler()
            ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(ch)

        # 🏆 最新优化的参数字典
        self.HP_ETH = {
            'obi_slow_threshold': 0.197798,
            'obi_momentum_threshold': 0.016294,
            'vol_ratio_threshold': 1.7696,
            'ema_period': 45,
            'cooldown_bars': 6,
            'atr_sl_multiplier': 2.696711,
            'atr_tp_multiplier': 3.624748,
        }

        self.HP_TAO = {
            'obi_slow_threshold': 0.0457,
            'obi_momentum_threshold': 0.0231,
            'vol_ratio_threshold': 0.8988,
            'ema_period': 12,
            'cooldown_bars': 5,
            'atr_sl_multiplier': 2.910283,
            'atr_tp_multiplier': 3.494203,
        }

        # 实盘状态管理变量
        self.pos_qty = 0.0
        self.pos_entry = 0.0
        self.pos_sl = 0.0
        self.pos_tp = 0.0
        self.last_trade_index = -9999
        self.current_index = 0

        # 标的锁定状态
        self.focused_coin = None
        self.focused_hp = None
        # persistent timestamp to enforce minimum time between actual order executions
        self.last_execution_time = None

    def _call_with_rate_limit(self, fn, *args, **kwargs):
        """Call a function while enforcing a strict >=60 second spacing between calls.

        This replaces repeated ad-hoc checks and avoids using int(...) which truncates
        fractional seconds and can lead to calls earlier than 60s.
        """
        # compute how long to wait (respect fractional seconds)
        if self.last_execution_time is not None:
            elapsed = (datetime.now() - self.last_execution_time).total_seconds()
            to_wait = 60.0 - elapsed
            if to_wait > 0:
                # sleep the exact remaining fractional seconds
                time.sleep(to_wait)

        # perform the call and update last_execution_time
        res = fn(*args, **kwargs)
        self.last_execution_time = datetime.now()
        return res

    def is_btc_active(self, df_btc: pd.DataFrame) -> bool:
        """判断 BTC 是否活跃的 Regime Filter"""
        if len(df_btc) < 100:
            return False
        close = df_btc['close']
        ema20 = close.ewm(span=20, adjust=False).mean().iloc[-1]
        vol_short = df_btc['volume'].rolling(12).mean().iloc[-1]
        vol_long = df_btc['volume'].rolling(100).mean().iloc[-1]
        ret = (close.iloc[-1] / close.iloc[-20] - 1) * 100
        return sum([close.iloc[-1] > ema20, vol_short > vol_long * 0.85, ret > -5]) >= 2

    def compute_obi_indicators(self, df: pd.DataFrame, hp: dict):
        """核心指标计算（Pandas 矢量化近似形态法）"""
        if len(df) < 60:
            return None

        epsilon = 1e-8
        c = df['close']
        v = df['volume']
        h = df['high']
        l = df['low']
        o = df['open']

        net = ((c - o) / (h - l + epsilon)) * v
        obi_slow = net.rolling(20).mean() / (v.rolling(20).mean() + epsilon)
        obi_fast = net.rolling(5).mean() / (v.rolling(5).mean() + epsilon)
        obi_mom = obi_fast - obi_slow
        vol_ratio = v.rolling(5).mean() / (v.rolling(30).mean() + epsilon)
        ema = c.ewm(span=hp['ema_period'], adjust=False).mean()

        tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()

        latest = df.iloc[-1]
        print("checkpoint2")
        print(latest['close'], obi_slow.iloc[-1],obi_mom.iloc[-1],vol_ratio.iloc[-1],ema.iloc[-1],atr.iloc[-1])
        return {
            'timestamp': latest['open_time'],
            'price': float(latest['close']),
            'high': float(latest['high']),
            'low': float(latest['low']),
            'obi_slow': float(obi_slow.iloc[-1]),
            'obi_momentum': float(obi_mom.iloc[-1]),
            'vol_ratio': float(vol_ratio.iloc[-1]),
            'above_ema': latest['close'] > ema.iloc[-1],
            'ema_val': float(ema.iloc[-1]),
            'atr': float(atr.iloc[-1]),
            'obi_slow_prev': float(obi_slow.iloc[-2]) if len(obi_slow) > 1 else 0,
        }

    def on_tick(self, sim_data: dict, total_equity: float):
        """
        实盘专用流处理接口，由 main.py 的主循环喂入所有币种的最新 DataFrame 字典
        """
        try:
            self.current_index += 1
            df_btc = sim_data.get("BTC")
            if df_btc is None or df_btc.empty:
                return

            # ==========================================
            # 1. 决定监控标的 (Regime Switch)
            # ==========================================
            if self.pos_qty > 0 and self.focused_coin is not None:
                # 持仓中 → 专注同一币种
                target_coin = self.focused_coin
                hp = self.focused_hp
            else:
                # 空仓时 → 根据 BTC 活跃度判断
                btc_active = self.is_btc_active(df_btc)
                if btc_active:
                    target_coin = "ETH"
                    hp = self.HP_ETH
                else:
                    target_coin = "TAO"
                    hp = self.HP_TAO

                self.focused_coin = target_coin
                self.focused_hp = hp

            # ==========================================
            # 2. 拉取标的资料并计算指标
            # ==========================================
            df_target = sim_data.get(target_coin)
            if df_target is None or len(df_target) < 100:
                return

            ind = self.compute_obi_indicators(df_target, hp)
            print("checkpoint3\n")
            print(ind)
            print(df_target)
            if not ind or pd.isna(ind['atr']) or ind['atr'] <= 0:
                return

            curr_price = ind['price']

            # ==========================================
            # 3. 检查平仓 (Exit Logic)
            # ==========================================
            if self.pos_qty > 0:
                exit_triggered = False
                reason = ""
                exit_price = 0.0

                if ind['low'] <= self.pos_sl:
                    exit_triggered, reason, exit_price = True, "🛑 触发 ATR 物理止损", self.pos_sl
                elif ind['high'] >= self.pos_tp:
                    exit_triggered, reason, exit_price = True, "🎯 触发 ATR 物理止盈", self.pos_tp
                elif (curr_price < ind['ema_val']) and (ind['obi_slow'] < 0):
                    exit_triggered, reason, exit_price = True, "⚠️ 触发策略恶化平仓", curr_price

                if exit_triggered:
                    try:
                        order = self.portfolio.create_order(self.focused_coin, "SELL", self.pos_qty, order_type="MARKET",
                                                            strategy_id=self.strategy_id)
                        # enforce >=60s between actual execute_order calls
                        res = self._call_with_rate_limit(self.portfolio.execute_order, order)
                        if res.get('success'):
                            pnl = self.pos_qty * (exit_price - self.pos_entry)
                            try:
                                self.logger.info(f"✅ [{self.focused_coin} 已平仓] | {reason} | PnL ≈ ${pnl:+.2f}")
                            except Exception:
                                pass

                            # 清空状态
                            self.pos_qty = 0.0
                            self.pos_entry = 0.0
                            self.pos_sl = 0.0
                            self.pos_tp = 0.0
                            self.focused_coin = None
                            self.focused_hp = None
                    except Exception:
                        try:
                            self.logger.exception("Failed executing exit order")
                        except Exception:
                            pass

            # ==========================================
            # 4. 检查开仓 (Entry Logic)
            # ==========================================
            if self.pos_qty == 0:
                bars_since = self.current_index - self.last_trade_index

                if (ind['obi_slow'] > hp['obi_slow_threshold'] and
                        ind['obi_momentum'] > hp['obi_momentum_threshold'] and
                        ind['vol_ratio'] > hp['vol_ratio_threshold'] and
                        ind['above_ema'] and
                        bars_since >= hp['cooldown_bars']):

                    min_distance = curr_price * 0.005
                    sl_dist = max(ind['atr'] * hp['atr_sl_multiplier'], min_distance)
                    tp_dist = max(ind['atr'] * hp['atr_tp_multiplier'], min_distance * 1.5)

                    sl_price = round(curr_price - sl_dist, 4)
                    tp_price = round(curr_price + tp_dist, 4)

                    # 资金分配：目标额度为全局净值的 alloc_ratio (默认 50%)，且不超过现金余额
                    target_invest_amount = total_equity * self.alloc_ratio
                    invest_amount = min(target_invest_amount, self.portfolio.account_balance)

                    if invest_amount < 10.0:
                        try:
                            self.logger.warning("❌ 余额不足或被限制，无法下单")
                        except Exception:
                            pass
                        return

                    # 略低一点买入 (模拟 limit 下单以促进成交)
                    buy_price = curr_price * 0.999
                    quantity = round(invest_amount / buy_price, 4)

                    try:
                        order = self.portfolio.create_order(self.focused_coin, "BUY", quantity, price=buy_price,
                                                            order_type="LIMIT", strategy_id=self.strategy_id)
                        # enforce >=60s between actual execute_order calls
                        order_id = self._call_with_rate_limit(self.portfolio.execute_order, order)['order_id']
                        res = self._call_with_rate_limit(self.execution.client.query_order(pending_only=True))
                        if res:
                            response = self.execution.process_query_response(res)
                            for r in response:
                                if r.get('order_id') == order_id and r.get('status') != 'CANCELED' and not r.get('processed'):
                                    try:
                                        self.logger.info(f"🚀 全仓买入成功！ {self.focused_coin} @ {buy_price:.4f} × {quantity:.6f}")
                                        self.logger.info(f"   SL = {sl_price:.4f}   TP = {tp_price:.4f}")
                                    except Exception:
                                        pass

                                    self.pos_qty = quantity
                                    self.pos_entry = buy_price
                                    self.pos_sl = sl_price
                                    self.pos_tp = tp_price
                                    self.last_trade_index = self.current_index
                                    break
                    except Exception:
                        try:
                            self.logger.exception("Failed executing entry order / processing response")
                        except Exception:
                            pass
            # end of entry logic
        except Exception:
            try:
                self.logger.exception("Unhandled exception in ObiDynamicStrategy.on_tick")
            except Exception:
                pass
