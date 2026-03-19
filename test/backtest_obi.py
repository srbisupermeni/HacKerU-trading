import os
import sys
import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# 确保能引到你的 database 模块
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from database.Binance_Vision_fetcher import VisionFetcher

class OBIOptimizerAndBacktester:
    def __init__(self, df, initial_balance=10000.0):
        self.df = df.copy()
        self.initial_balance = initial_balance
        
        # 默认参数
        self.default_hp = {
            'obi_slow_threshold': 0.05,
            'obi_momentum_threshold': 0.02,
            'vol_ratio_threshold': 1.1,
            'ema_period': 55,
            'cooldown_bars': 5,
            'atr_sl_multiplier': 1.5,
            'atr_tp_multiplier': 3.0
        }
        
        # 待优化的参数空间定义
        self.param_space = {
            'obi_slow_threshold':     {'type': 'float', 'min': -0.15, 'max': 0.20},
            'obi_momentum_threshold': {'type': 'float', 'min': -0.10, 'max': 0.15},
            'vol_ratio_threshold':    {'type': 'float', 'min': 0.5,   'max': 1.8},
            'ema_period':             {'type': 'int',   'min': 5,     'max': 100},
            'cooldown_bars':          {'type': 'int',   'min': 1,     'max': 15},
            'atr_sl_multiplier':      {'type': 'float', 'min': 1.0,   'max': 3.0},
            'atr_tp_multiplier':      {'type': 'float', 'min': 1.5,   'max': 6.0}
        }

    def precalculate_fixed_indicators(self):
        """
        🚀 性能优化核心：预计算所有不受参数变化影响的指标
        这样在执行成百上千次参数寻优时，不需要重复计算这些复杂的滚动指标。
        """
        df = self.df
        epsilon = 1e-8
        
        # 1. 计算近似主动买盘
        df['net_buy_vol'] = ((df['close'] - df['open']) / (df['high'] - df['low'] + epsilon)) * df['volume']
        
        # 2. OBI 快慢线与动量 (窗口期是固定的 5 和 20)
        df['obi_fast'] = df['net_buy_vol'].rolling(5).mean() / (df['volume'].rolling(5).mean() + epsilon)
        df['obi_slow'] = df['net_buy_vol'].rolling(20).mean() / (df['volume'].rolling(20).mean() + epsilon)
        df['obi_momentum'] = df['obi_fast'] - df['obi_slow']
        
        # 3. 量能比 (vol_ratio)
        df['vol_short'] = df['volume'].rolling(5).mean()
        df['vol_long'] = df['volume'].rolling(30).mean()
        df['vol_ratio'] = df['vol_short'] / (df['vol_long'] + epsilon)
        
        # 4. 真实波幅 (ATR 14)
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift(1)).abs()
        low_close = (df['low'] - df['close'].shift(1)).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()

        # 清除 NaN 以对齐数据
        self.df = df.dropna().reset_index(drop=True)

    def run_backtest(self, hp=None, verbose=False):
        """核心回测引擎，接受动态参数组合 hp"""
        if hp is None:
            hp = self.default_hp
            
        # 动态计算受参数影响的指标：EMA
        self.df['ema_val'] = self.df['close'].ewm(span=hp['ema_period'], adjust=False).mean()

        balance = self.initial_balance
        position_qty = 0.0
        entry_price = 0.0
        sl_price = 0.0
        tp_price = 0.0
        last_trade_bar = -9999
        
        trades = [] 
        
        # 转换为 Numpy array 加速逐行迭代
        closes = self.df['close'].values
        opens = self.df['open'].values
        highs = self.df['high'].values
        lows = self.df['low'].values
        ema_vals = self.df['ema_val'].values
        obi_slows = self.df['obi_slow'].values
        obi_moms = self.df['obi_momentum'].values
        vol_ratios = self.df['vol_ratio'].values
        atrs = self.df['atr'].values
        open_times = self.df['open_time'].values
        
        for i in range(len(self.df)):
            curr_price = closes[i]
            curr_high = highs[i]
            curr_low = lows[i]
            curr_open = opens[i]
            
            # ==========================================
            # 1. 退出逻辑 (止损、止盈、策略平仓)
            # ==========================================
            if position_qty > 0:
                exit_price = None
                exit_reason = ""
                
                if curr_low <= sl_price:
                    exit_price = sl_price if curr_open > sl_price else curr_open
                    exit_reason = "Stop Loss"
                elif curr_high >= tp_price:
                    exit_price = tp_price if curr_open < tp_price else curr_open
                    exit_reason = "Take Profit"
                elif curr_price < ema_vals[i] and obi_slows[i] < 0:
                    exit_price = curr_price
                    exit_reason = "Liquidate"
                    
                if exit_price is not None:
                    balance += position_qty * exit_price
                    pnl_pct = (exit_price - entry_price) / entry_price
                    trades.append({
                        'exit_time': open_times[i],
                        'reason': exit_reason,
                        'pnl_%': pnl_pct * 100,
                        'balance': balance
                    })
                    position_qty = 0.0
                    continue 
                    
            # ==========================================
            # 2. 开仓逻辑 (should_long)
            # ==========================================
            if position_qty == 0:
                if (atrs[i] > 0 and 
                    obi_slows[i] > hp['obi_slow_threshold'] and 
                    obi_moms[i] > hp['obi_momentum_threshold'] and 
                    vol_ratios[i] > hp['vol_ratio_threshold'] and 
                    curr_price > ema_vals[i] and 
                    (i - last_trade_bar) >= hp['cooldown_bars']):
                    
                    # 半仓出击
                    invest_amount = balance * 0.50
                    entry_price = curr_price
                    position_qty = invest_amount / entry_price
                    balance -= invest_amount
                    last_trade_bar = i
                    
                    # 设定止盈止损线
                    min_distance = entry_price * 0.005 
                    sl_distance = max(atrs[i] * hp['atr_sl_multiplier'], min_distance)
                    tp_distance = max(atrs[i] * hp['atr_tp_multiplier'], min_distance * 1.5)
                    
                    sl_price = entry_price - sl_distance
                    tp_price = entry_price + tp_distance
                    
        # 结算最终净值
        total_equity = balance + (position_qty * closes[-1])
        roi = (total_equity - self.initial_balance) / self.initial_balance * 100
        
        # 如果需要打印详细报告
        if verbose:
            self._print_stats(trades, total_equity, roi)
            
        return total_equity, roi, trades

    def optimize(self, trials=500):
        """
        🚀 随机搜索优化算法 (Random Search)
        随机在指定的超参数空间内生成组合，找出能带来最高净值的参数。
        """
        print(f"\n" + "="*50)
        print(f"⚙️ 开始策略优化: 进行 {trials} 次参数组合测试")
        print("="*50)
        
        best_equity = -1
        best_params = None
        best_roi = -999
        
        for i in range(trials):
            # 1. 随机生成一组参数
            test_hp = {}
            for param_name, specs in self.param_space.items():
                if specs['type'] == 'int':
                    test_hp[param_name] = random.randint(specs['min'], specs['max'])
                else:
                    test_hp[param_name] = round(random.uniform(specs['min'], specs['max']), 4)
            
            # 2. 运行回测 (关闭 verbose 以加速)
            equity, roi, trades = self.run_backtest(hp=test_hp, verbose=False)
            
            # 3. 记录最优解 (这里以最终净值为优化目标。如果要稳健，可以加个条件如 len(trades) > 5)
            if equity > best_equity and len(trades) > 5:  # 至少交易5次，防止碰巧一次重仓全赢
                best_equity = equity
                best_roi = roi
                best_params = test_hp.copy()
                
            # 进度提示
            if (i + 1) % 100 == 0:
                print(f"⏳ 优化进度: {i+1}/{trials} | 当前最优 ROI: {best_roi:.2f}%")

        print("\n" + "🏆 优化完成！最佳参数组合: " + "🏆")
        for k, v in best_params.items():
            print(f"  - {k}: {v}")
            
        print("\n" + "🚀 使用最佳参数运行最终详细报告:")
        self.run_backtest(hp=best_params, verbose=True)
        return best_params

    def _print_stats(self, trades, total_equity, roi):
        win_trades = [t for t in trades if t['pnl_%'] > 0]
        loss_trades = [t for t in trades if t['pnl_%'] <= 0]
        win_rate = len(win_trades) / len(trades) * 100 if trades else 0.0
        
        print(f"💰 初始资金: ${self.initial_balance:.2f}")
        print(f"🏦 最终净值: ${total_equity:.2f} (ROI: {roi:.2f}%)")
        print(f"📊 交易总数: {len(trades)} 笔")
        print(f"🏆 胜率: {win_rate:.2f}% ({len(win_trades)} 胜 / {len(loss_trades)} 负)")
        
        if win_trades and loss_trades:
            avg_win = np.mean([t['pnl_%'] for t in win_trades])
            avg_loss = np.mean([t['pnl_%'] for t in loss_trades])
            print(f"📈 平均盈利: {avg_win:.2f}%")
            print(f"📉 平均亏损: {avg_loss:.2f}%")
            print(f"⚖️ 盈亏比: {abs(avg_win / avg_loss):.2f}")
        print("="*40)


if __name__ == "__main__":
    print("正在拉取过去 30 天的数据...")
    fetcher = VisionFetcher()
    end_date = datetime.today()
    start_date = end_date - timedelta(days=30)
    
    df_30d = fetcher.fetch_klines_range(
        symbol="ETHUSDT", interval="15m", 
        start_year=start_date.year, start_month=start_date.month, start_day=start_date.day,
        end_year=end_date.year, end_month=end_date.month, end_day=end_date.day,
        data_type="daily"
    )
    
    if df_30d is not None and not df_30d.empty:
        # 初始化回测器
        bt = OBIOptimizerAndBacktester(df_30d, initial_balance=10000.0)
        
        # 1. 预先计算出所有固定指标 (必须先运行)
        bt.precalculate_fixed_indicators()
        
        # 2. 运行参数优化，trials 参数决定搜索次数，你可以根据算力随意加大
        # (通常 500-1000 就能在这些参数空间里摸到一个极好的局部最优解)
        best_params = bt.optimize(trials=10000)
    else:
        print("⚠️ 数据拉取失败，请检查网络。")