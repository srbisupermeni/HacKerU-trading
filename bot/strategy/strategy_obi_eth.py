import sys
import os
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# ==========================================
# 路径兼容处理
# ==========================================
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# 引入你的 VisionFetcher 用于半个月深度回测
from database.Binance_Vision_fetcher import VisionFetcher
from database.Binance_fetcher import BinanceDataFetcher
from bot.portfolio.portfolio import Portfolio
from bot.execution.execution_engine import ExecutionEngine
from bot.execution.roostoo import Roostoo

class ObiEthStrategy:
    """
    ETH 专属版：基于 OBI 的动量突破策略
    """
    def __init__(self, portfolio: Portfolio, execution: ExecutionEngine, 
                 symbol="ETHUSDT", coin="ETH", strategy_id="obi_eth_01", alloc_ratio=0.50):
        self.portfolio = portfolio
        self.execution = execution
        self.symbol = symbol          
        self.coin = coin              
        self.strategy_id = strategy_id 
        self.alloc_ratio = alloc_ratio 
        
        self.logger = logging.getLogger(f"OBI_{self.coin}")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            ch = logging.StreamHandler()
            ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(ch)

        # 🏆 严格遵循 Jesse 回测最佳参数 (ETH)
        self.obi_slow_threshold = 0.197798
        self.obi_momentum_threshold = 0.066294
        self.vol_ratio_threshold = 1.769604
        self.ema_period = 45
        self.cooldown_bars = 6
        self.atr_sl_multiplier = 2.696711
        self.atr_tp_multiplier = 3.624748
        
        self.last_trade_index = -9999
        self.current_index = 0

    def calculate_indicators(self, df: pd.DataFrame, return_full=False) -> pd.Series | pd.DataFrame:
        """
        利用 Pandas 矢量化计算策略指标。
        :param return_full: 若为 True，则返回包含所有指标的完整 DataFrame，用于极速回测。
        """
        df = df.copy()
        epsilon = 1e-8
        
        # 1. 🧨 核心降级：为了适配 Jesse 优化出的阈值，必须用回 K线形态近似法！
        # 无论实盘拿到了多么精准的 buy_volume，这里都强行采用回测时的估算逻辑
        df['net_buy_vol'] = ((df['close'] - df['open']) / (df['high'] - df['low'] + epsilon)) * df['volume']
        
        df['obi_fast'] = df['net_buy_vol'].rolling(5).mean() / (df['volume'].rolling(5).mean() + epsilon)
        df['obi_slow'] = df['net_buy_vol'].rolling(20).mean() / (df['volume'].rolling(20).mean() + epsilon)
        df['obi_momentum'] = df['obi_fast'] - df['obi_slow']
        
        # 2. 成交量比例
        vol_short = df['volume'].rolling(5).mean()
        vol_long = df['volume'].rolling(30).mean()
        df['vol_ratio'] = vol_short / (vol_long + epsilon)
        
        # 3. EMA
        df['ema'] = df['close'].ewm(span=self.ema_period, adjust=False).mean()
        df['above_ema'] = df['close'] > df['ema']
        
        # 4. ATR (简单移动平均方式)
        df['prev_close'] = df['close'].shift(1)
        tr1 = df['high'] - df['low']
        tr2 = (df['high'] - df['prev_close']).abs()
        tr3 = (df['low'] - df['prev_close']).abs()
        df['tr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr'] = df['tr'].rolling(14).mean()
        
        if return_full:
            return df
        return df.iloc[-1]

    def on_new_candle(self, recent_df: pd.DataFrame, current_price: float):
        """实盘接收新 K 线的入口"""
        self.current_index += 1
        
        if len(recent_df) < max(120, self.ema_period):
            return
            
        latest = self.calculate_indicators(recent_df)
        
        if pd.isna(latest['atr']) or latest['atr'] <= 0:
            return

        lock_status = self.portfolio.get_coin_lock_status(self.coin)
        
        if not lock_status['locked']:
            cooldown_ok = (self.current_index - self.last_trade_index) >= self.cooldown_bars
            
            if (latest['obi_slow'] > self.obi_slow_threshold and 
                latest['obi_momentum'] > self.obi_momentum_threshold and 
                latest['vol_ratio'] > self.vol_ratio_threshold and 
                latest['above_ema'] and cooldown_ok):
                
                self.logger.info(f"🎯 [{self.coin} OBI 突破确认] 触发做多！")
                
                min_distance = current_price * 0.005 
                sl_dist = max(latest['atr'] * self.atr_sl_multiplier, min_distance)
                tp_dist = max(latest['atr'] * self.atr_tp_multiplier, min_distance * 1.5)
                
                sl_price = round(current_price - sl_dist, 4)
                tp_price = round(current_price + tp_dist, 4)
                
                invest_amount = self.portfolio.account_balance * self.alloc_ratio
                quantity = round(invest_amount / current_price, 4)
                
                res = self.execution.execute_order(
                    coin=self.coin, side='BUY', quantity=quantity, 
                    price=current_price, order_type='LIMIT', 
                    strategy_id=self.strategy_id, stop_loss=sl_price, take_profit=tp_price
                )
                if res.get('success'):
                    self.last_trade_index = self.current_index
        else:
            if (current_price < latest['ema']) and (latest['obi_slow'] < 0):
                self.logger.warning(f"⚠️ [{self.coin} 清算警报] 跌破 EMA，建议平仓！")


# ==========================================
# 本地 15 天全景极速推演 (__main__)
# ==========================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🏆 ETH OBI 原生兼容版：拉取 Vision 近半个月数据进行推演")
    print("="*60)
    
    roostoo_client = Roostoo()
    portfolio = Portfolio(execution_module=None) 
    portfolio.account_balance = 10000.0          
    execution = ExecutionEngine(portfolio, roostoo_client)
    portfolio.execution = execution 
    
    coin = "ETH"
    symbol = "ETHUSDT"
    # 注意：务必将 interval 修改为你 Jesse 回测时使用的周期！(比如 "5m" 或 "15m")
    test_interval = "5m" 
    
    strat = ObiEthStrategy(portfolio, execution, symbol=symbol, coin=coin, alloc_ratio=0.50)
    
    # 使用 VisionFetcher 抓取近 15 天的数据
    print(f"\n⏳ 正在通过 Vision 获取 {symbol} 过去 15 天的 {test_interval} 数据...")
    fetcher = VisionFetcher()
    end_date = datetime.today()
    start_date = end_date - timedelta(days=15)
    
    raw_df = fetcher.fetch_klines_range(
        symbol=symbol, interval=test_interval, 
        start_year=start_date.year, start_month=start_date.month, start_day=start_date.day,
        end_year=end_date.year, end_month=end_date.month, end_day=end_date.day,
        data_type="daily" # 强行按日下载近15天
    )
        
    if raw_df is None or raw_df.empty:
        print("⚠️ 数据获取失败，请检查网络或 Vision 源。")
        sys.exit()
        
    print(f"✅ 拉取成功！共计 {len(raw_df)} 根 K线。开始全景极速推演...")
    
    # ⚡ 性能优化：一次性全量计算指标，保证 EMA 的收敛与 Jesse 完全一致！
    df_indicators = strat.calculate_indicators(raw_df, return_full=True)
    
    pos_qty, pos_entry, pos_sl, pos_tp = 0.0, 0.0, 0.0, 0.0
    trades_count = 0
    
    # 从 120 根 K 线之后开始遍历 (等待所有指标尤其是 EMA 和长周期 Vol 计算完毕)
    for i in range(120, len(df_indicators)):
        curr_row = df_indicators.iloc[i]
        curr_time = curr_row['open_time']
        curr_price = curr_row['close']
        curr_high = curr_row['high']
        curr_low = curr_row['low']
        
        strat.current_index = i
        
        # --- 平仓判定 ---
        if pos_qty > 0:
            exit_price, reason = None, ""
            
            if curr_low <= pos_sl:
                exit_price, reason = pos_sl, "🛑 触发 ATR 物理止损"
            elif curr_high >= pos_tp:
                exit_price, reason = pos_tp, "🎯 触发 ATR 物理止盈"
            elif (curr_price < curr_row['ema']) and (curr_row['obi_slow'] < 0):
                exit_price, reason = curr_price, "⚠️ 触发策略恶化平仓"
                
            if exit_price:
                revenue = pos_qty * exit_price * (1 - 0.001)
                pnl_pct = (exit_price - pos_entry) / pos_entry * 100
                portfolio.account_balance += revenue
                pos_qty, pos_entry = 0.0, 0.0
                trades_count += 1
                print(f"[{curr_time}] {reason} | 卖出: {exit_price:.2f} | 盈亏: {pnl_pct:.2f}% | 余额: ${portfolio.account_balance:.2f}")
                portfolio.force_release_coin(coin)
                
        # --- 开仓判定 ---
        if pos_qty == 0:
            cooldown_ok = (strat.current_index - strat.last_trade_index) >= strat.cooldown_bars
            
            if (not pd.isna(curr_row['atr']) and curr_row['atr'] > 0 and
                curr_row['obi_slow'] > strat.obi_slow_threshold and 
                curr_row['obi_momentum'] > strat.obi_momentum_threshold and 
                curr_row['vol_ratio'] > strat.vol_ratio_threshold and 
                curr_row['above_ema'] and cooldown_ok):
                
                min_distance = curr_price * 0.005
                pos_sl = curr_price - max(curr_row['atr'] * strat.atr_sl_multiplier, min_distance)
                pos_tp = curr_price + max(curr_row['atr'] * strat.atr_tp_multiplier, min_distance * 1.5)
                
                invest = portfolio.account_balance * strat.alloc_ratio
                portfolio.account_balance -= invest
                pos_qty = (invest * (1 - 0.001)) / curr_price
                pos_entry = curr_price
                
                strat.last_trade_index = strat.current_index
                portfolio.acquire_coin(coin, strat.strategy_id)
                
                print(f"[{curr_time}] 🟢 OBI 突破 | 资金: ${invest:.2f} | OBI Slow: {curr_row['obi_slow']:.3f} | 买入: {curr_price:.2f}")
    
    # 强制平仓清算
    if pos_qty > 0:
        final_price = df_indicators.iloc[-1]['close']
        portfolio.account_balance += pos_qty * final_price * (1 - 0.001)
            
    roi = (portfolio.account_balance - 10000.0) / 10000.0 * 100
    print("\n" + "="*40)
    print(f"🏁 {test_interval} 级别 15天深度推演结束！")
    print(f"总交易次数: {trades_count} | 最终净值: ${portfolio.account_balance:.2f} (ROI: {roi:.2f}%)")
    print("="*40)