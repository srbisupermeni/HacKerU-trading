import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# ==========================================
# 路径兼容处理
# ==========================================
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from database.Binance_Vision_fetcher import VisionFetcher
from bot.portfolio.portfolio import Portfolio
from bot.execution.execution_engine import ExecutionEngine
from bot.api.roostoo import Roostoo

class ObiDynamicStrategy:
    """
    OBI 动态标的版：基于 BTC 活跃度在 ETH 和 TAO 之间切换的动量突破策略
    """
    def __init__(self, portfolio: Portfolio, execution: ExecutionEngine, strategy_id="obi_dynamic_01", alloc_ratio=0.50):
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
            'obi_slow_threshold':     0.197798,
            'obi_momentum_threshold': 0.016294,
            'vol_ratio_threshold':    1.7696,
            'ema_period':             45,
            'cooldown_bars':          6,
            'atr_sl_multiplier':      2.696711,
            'atr_tp_multiplier':      3.624748,
        }

        self.HP_TAO = {
            'obi_slow_threshold':     0.0457,
            'obi_momentum_threshold': 0.0231,
            'vol_ratio_threshold':    0.8988,
            'ema_period':             12,
            'cooldown_bars':          5,
            'atr_sl_multiplier':      2.910283,
            'atr_tp_multiplier':      3.494203,
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

    def is_btc_active(self, df_btc: pd.DataFrame) -> bool:
        """判断 BTC 是否活跃的 Regime Filter"""
        if len(df_btc) < 100:
            return False
        close = df_btc['close']
        ema20 = close.ewm(span=20, adjust=False).mean().iloc[-1]
        vol_short = df_btc['volume'].rolling(12).mean().iloc[-1]
        vol_long  = df_btc['volume'].rolling(100).mean().iloc[-1]
        ret = (close.iloc[-1] / close.iloc[-20] - 1) * 100
        return sum([close.iloc[-1] > ema20, vol_short > vol_long * 0.85, ret > -5]) >= 2

    def compute_obi_indicators(self, df: pd.DataFrame, hp: dict) -> dict:
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
        obi_fast = net.rolling(5).mean()  / (v.rolling(5).mean()  + epsilon)
        obi_mom  = obi_fast - obi_slow
        vol_ratio = v.rolling(5).mean() / (v.rolling(30).mean() + epsilon)
        ema = c.ewm(span=hp['ema_period'], adjust=False).mean()

        tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()

        latest = df.iloc[-1]
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
            'obi_slow_prev': float(obi_slow.iloc[-2]) if len(obi_slow)>1 else 0,
        }

    def on_tick(self, sim_data: dict, total_equity: float):
        """
        实盘专用流处理接口，由 main.py 的主循环喂入所有币种的最新 DataFrame 字典
        """
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
                # self.logger.info("📈 BTC active → 监控 ETH") # 避免刷屏可注释
            else:
                target_coin = "TAO"
                hp = self.HP_TAO
                # self.logger.info("📉 BTC inactive → 监控 TAO") # 避免刷屏可注释
                
            self.focused_coin = target_coin
            self.focused_hp = hp

        # ==========================================
        # 2. 拉取标的资料并计算指标
        # ==========================================
        df_target = sim_data.get(target_coin)
        if df_target is None or len(df_target) < 100:
            return

        ind = self.compute_obi_indicators(df_target, hp)
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
                order = self.portfolio.create_order(self.focused_coin, "SELL", self.pos_qty, order_type="MARKET", strategy_id=self.strategy_id)
                res = self.portfolio.execute_order(order)
                if res.get('success'):
                    pnl = self.pos_qty * (exit_price - self.pos_entry)
                    self.logger.info(f"✅ [{self.focused_coin} 已平仓] | {reason} | PnL ≈ ${pnl:+.2f}")
                    
                    # 清空状态
                    self.pos_qty = 0.0
                    self.pos_entry = 0.0
                    self.pos_sl = 0.0
                    self.pos_tp = 0.0
                    self.focused_coin = None
                    self.focused_hp = None

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
                    self.logger.warning("❌ 余额不足或被限制，无法下单")
                    return
                
                # 略低一点买入 (模拟 limit 下单以促进成交)
                buy_price = curr_price * 0.999
                quantity = round(invest_amount / buy_price, 4)
                
                order = self.portfolio.create_order(self.focused_coin, "BUY", quantity, price=buy_price, order_type="LIMIT", strategy_id=self.strategy_id)
                res = self.portfolio.execute_order(order)
                
                if res.get('success'):
                    self.logger.info(f"🚀 全仓买入成功！ {self.focused_coin} @ {buy_price:.4f} × {quantity:.6f}")
                    self.logger.info(f"   SL = {sl_price:.4f}   TP = {tp_price:.4f}")
                    
                    self.pos_qty = quantity
                    self.pos_entry = buy_price
                    self.pos_sl = sl_price
                    self.pos_tp = tp_price
                    self.last_trade_index = self.current_index


# ==========================================
# 本地极速全景推演 (__main__)
# ==========================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🏆 OBI 动态标的版：拉取 Vision 数据进行本地回测推演")
    print("="*60)
    
    roostoo_client = Roostoo()
    portfolio = Portfolio(execution_module=None) 
    portfolio.account_balance = 25000.0          
    execution = ExecutionEngine(portfolio, roostoo_client)
    portfolio.execution = execution 
    
    strat = ObiDynamicStrategy(portfolio, execution, alloc_ratio=0.50)
    
    test_interval = "15m" 
    
    print(f"\n⏳ 正在通过 Vision 获取 BTC, ETH, TAO 过去 15 天的 {test_interval} 数据...")
    fetcher = VisionFetcher()
    end_date = datetime.today()
    start_date = end_date - timedelta(days=15)
    
    sim_data = {}
    for c in ["BTC", "ETH", "TAO"]:
        df = fetcher.fetch_klines_range(
            symbol=f"{c}USDT", interval=test_interval, 
            start_year=start_date.year, start_month=start_date.month, start_day=start_date.day,
            end_year=end_date.year, end_month=end_date.month, end_day=end_date.day, data_type="daily"
        )
        if df is not None and not df.empty:
            df['open_time'] = pd.to_datetime(df['open_time'])
            sim_data[c] = df.set_index('open_time', drop=False)
            
    if len(sim_data) < 3:
        print("⚠️ 数据获取失败，请检查网络或 Vision 源。")
        sys.exit()
        
    print(f"✅ 拉取成功！开始极速推演...")
    
    # 模拟主循环时间步推进
    base_idx = sim_data["BTC"].index
    start_idx = 120 # 预留指标预热窗口
    
    for i in range(start_idx, len(base_idx)):
        current_time = base_idx[i]
        
        # 截取该时间点之前的数据切片喂给策略
        current_sim_data = {}
        for c in ["BTC", "ETH", "TAO"]:
            current_sim_data[c] = sim_data[c].iloc[:i+1]
            
        # 这里为了回测简化，直接将可用余额作为 total_equity 传入
        # 实盘中 main.py 会计算真实的 total_equity 传入
        current_equity = portfolio.account_balance
        if strat.pos_qty > 0 and strat.focused_coin:
            current_equity += strat.pos_qty * current_sim_data[strat.focused_coin].iloc[-1]['close']
            
        strat.on_tick(current_sim_data, current_equity)
        
    roi = (portfolio.account_balance - 25000.0) / 25000.0 * 100
    print("\n" + "="*40)
    print(f"🏁 {test_interval} 级别 15天深度推演结束！")
    print(f"最终净值: ${portfolio.account_balance:.2f} (ROI: {roi:.2f}%)")
    print("="*40)