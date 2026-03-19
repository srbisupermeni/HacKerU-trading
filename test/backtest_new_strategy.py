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

class TrailingStrategyOptimizer:
    def __init__(self, df, initial_balance=10000.0):
        self.df = df.copy()
        self.initial_balance = initial_balance
        
        # 默认参数 (完全对齐你的 __init__.py 默认值)
        self.default_hp = {
            'ema_period': 88,
            'roc_period': 10,
            'roc_threshold': 0.004926,
            'obi_slow_threshold': 0.1235,
            'obi_momentum_threshold': 0.215,
            'vol_intensity_threshold': 2.055,
            'atr_ratio_threshold': 0.869,
            'atr_sl_multiplier': 3.21,
            'trail_atr_multiplier': 2.33,
            'breakeven_atr_trigger': 2.35,
            'cooldown_bars': 78
        }
        
        # 待优化的参数空间定义
        self.param_space = {
            'ema_period':              {'type': 'int',   'min': 30,    'max': 100},
            'roc_period':              {'type': 'int',   'min': 6,     'max': 24},
            'roc_threshold':           {'type': 'float', 'min': 0.001, 'max': 0.008},
            'obi_slow_threshold':      {'type': 'float', 'min': 0.05,  'max': 0.25},
            'obi_momentum_threshold':  {'type': 'float', 'min': 0.05,  'max': 0.25},
            'vol_intensity_threshold': {'type': 'float', 'min': 1.2,   'max': 2.5},
            'atr_ratio_threshold':     {'type': 'float', 'min': 0.8,   'max': 1.8},
            'atr_sl_multiplier':       {'type': 'float', 'min': 1.5,   'max': 3.5},
            'trail_atr_multiplier':    {'type': 'float', 'min': 2.0,   'max': 5.0},
            'breakeven_atr_trigger':   {'type': 'float', 'min': 0.8,   'max': 2.5},
            'cooldown_bars':           {'type': 'int',   'min': 15,    'max': 80}
        }

    def precalculate_fixed_indicators(self):
        """
        🚀 预计算不受参数变化影响的指标，极大加速优化过程
        """
        df = self.df
        eps = 1e-8
        
        # 1. K线实体比例
        df['body_ratio'] = (df['close'] - df['open']).abs() / (df['open'] + eps)
        
        # 2. 订单流 (OBI) - 严格复现 _calc_obi
        df['imb'] = (df['close'] - df['open']) / (df['high'] - df['low'] + eps)
        df['obi_vol'] = df['imb'] * df['volume']
        
        df['obi_fast'] = df['obi_vol'].rolling(5).mean() / (df['volume'].rolling(5).mean() + eps)
        df['obi_slow'] = df['obi_vol'].rolling(20).mean() / (df['volume'].rolling(20).mean() + eps)
        df['obi_momentum'] = df['obi_fast'] - df['obi_slow']
        
        # 3. 量能层 (Vol Intensity)
        df['vol_short'] = df['volume'].rolling(5).mean()
        df['vol_long'] = df['volume'].rolling(24).mean()
        df['vol_intensity'] = df['vol_short'] / (df['vol_long'] + eps)
        
        # 4. 真实波幅 (ATR 14) 及 ATR_Ratio (14 / 60)
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift(1)).abs()
        low_close = (df['low'] - df['close'].shift(1)).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()
        df['atr_ratio'] = df['atr'] / (df['atr'].rolling(60).mean() + eps)

        # 丢弃 NaN 数据
        self.df = df.dropna().reset_index(drop=True)

    def run_backtest(self, hp=None, verbose=False):
        """核心回测引擎，包含手动移动追踪止损逻辑"""
        if hp is None:
            hp = self.default_hp
            
        eps = 1e-8
        # 动态计算受参数影响的指标：EMA 和 ROC
        self.df['ema_val'] = self.df['close'].ewm(span=hp['ema_period'], adjust=False).mean()
        self.df['roc'] = (self.df['close'] - self.df['close'].shift(hp['roc_period'])) / (self.df['close'].shift(hp['roc_period']) + eps)

        balance = self.initial_balance
        position_qty = 0.0
        entry_price = 0.0
        
        # 止损追踪状态变量
        sl_price = 0.0
        highest_price = 0.0
        trail_distance = 0.0
        
        last_trade_bar = -9999
        trades = [] 
        
        # 提取为 Numpy array 加速
        closes = self.df['close'].values
        opens = self.df['open'].values
        highs = self.df['high'].values
        lows = self.df['low'].values
        
        ema_vals = self.df['ema_val'].values
        rocs = self.df['roc'].values
        obi_slows = self.df['obi_slow'].values
        obi_moms = self.df['obi_momentum'].values
        body_ratios = self.df['body_ratio'].values
        vol_intensities = self.df['vol_intensity'].values
        atrs = self.df['atr'].values
        atr_ratios = self.df['atr_ratio'].values
        open_times = self.df['open_time'].values
        
        # 跳过开头的 NaN，确保 ROC 等有数据
        start_idx = hp['roc_period'] + 1
        
        for i in range(start_idx, len(self.df)):
            curr_price = closes[i]
            curr_high = highs[i]
            curr_low = lows[i]
            curr_open = opens[i]
            curr_atr = atrs[i]
            
            # ==========================================
            # 1. 持仓状态：移动追踪止损 & 主动平仓逻辑
            # ==========================================
            if position_qty > 0:
                exit_price = None
                exit_reason = ""
                
                # ── a. 更新追踪止损 ──
                if curr_high > highest_price:
                    highest_price = curr_high
                    
                new_trail_sl = highest_price - trail_distance
                
                # 保本激活检查：按当根K线的 ATR 计算触发价
                trigger_price = entry_price + curr_atr * hp['breakeven_atr_trigger']
                breakeven_floor = entry_price * 1.001 # 成本价 + 0.1% 缓冲
                
                if highest_price >= trigger_price:
                    new_trail_sl = max(new_trail_sl, breakeven_floor)
                    
                # 止损只上移，不下移
                sl_price = max(sl_price, new_trail_sl)
                
                # ── b. 检查是否触发止损 ──
                if curr_low <= sl_price:
                    # 如果向下跳空开盘，按开盘价走；否则按止损价走
                    exit_price = sl_price if curr_open > sl_price else curr_open
                    exit_reason = "Trailing SL"
                
                # ── c. 检查是否触发策略主动平仓 (Liquidate) ──
                elif (curr_price < ema_vals[i] and obi_slows[i] < -0.05) or (obi_slows[i] < -0.20):
                    exit_price = curr_price
                    exit_reason = "Liquidate"
                    
                # ── 执行平仓 ──
                if exit_price is not None:
                    balance += position_qty * exit_price
                    pnl_pct = (exit_price - entry_price) / entry_price
                    trades.append({
                        'exit_time': open_times[i],
                        'reason': exit_reason,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'pnl_%': pnl_pct * 100,
                        'balance': balance
                    })
                    position_qty = 0.0
                    continue 
                    
            # ==========================================
            # 2. 空仓状态：开仓逻辑 (should_long)
            # ==========================================
            if position_qty == 0:
                if (curr_atr > 0 and 
                    curr_price > ema_vals[i] and 
                    rocs[i] > hp['roc_threshold'] and
                    obi_slows[i] > hp['obi_slow_threshold'] and 
                    obi_moms[i] > hp['obi_momentum_threshold'] and 
                    body_ratios[i] > 0.0003 and
                    vol_intensities[i] > hp['vol_intensity_threshold'] and 
                    atr_ratios[i] > hp['atr_ratio_threshold'] and
                    (i - last_trade_bar) >= hp['cooldown_bars']):
                    
                    # 50%仓位出击 (对应 qty = self.balance * 0.50 / self.price)
                    invest_amount = balance * 0.50
                    entry_price = curr_price
                    position_qty = invest_amount / entry_price
                    balance -= invest_amount
                    last_trade_bar = i
                    
                    # ── 初始化追踪止损参数 (on_open_position) ──
                    trail_distance = max(curr_atr * hp['trail_atr_multiplier'], entry_price * 0.003)
                    sl_dist = max(curr_atr * hp['atr_sl_multiplier'], entry_price * 0.002)
                    
                    sl_price = entry_price - sl_dist
                    highest_price = entry_price
                    
        # 结算最终净值
        total_equity = balance + (position_qty * closes[-1])
        roi = (total_equity - self.initial_balance) / self.initial_balance * 100
        
        if verbose:
            self._print_stats(trades, total_equity, roi)
            
        return total_equity, roi, trades

    def optimize(self, trials=500):
        """随机搜索 (Random Search) 优化算法"""
        print(f"\n" + "="*50)
        print(f"⚙️ 开始 Trailing 策略优化: 进行 {trials} 次参数组合测试")
        print("="*50)
        
        best_equity = -1
        best_params = None
        best_roi = -999
        
        for i in range(trials):
            # 随机生成参数
            test_hp = {}
            for param_name, specs in self.param_space.items():
                if specs['type'] == 'int':
                    test_hp[param_name] = random.randint(specs['min'], specs['max'])
                else:
                    test_hp[param_name] = round(random.uniform(specs['min'], specs['max']), 6)
            
            # 运行并静默返回
            equity, roi, trades = self.run_backtest(hp=test_hp, verbose=False)
            
            # 设定合理的优化目标（至少交易 5 次过滤掉偶然性极高的巧合数据）
            if equity > best_equity and len(trades) >= 5: 
                best_equity = equity
                best_roi = roi
                best_params = test_hp.copy()
                
            if (i + 1) % 100 == 0:
                print(f"⏳ 优化进度: {i+1}/{trials} | 当前最优 ROI: {best_roi:.2f}%")

        if best_params:
            print("\n" + "🏆 优化完成！最佳参数组合: " + "🏆")
            for k, v in best_params.items():
                print(f"  - {k}: {v}")
                
            print("\n🚀 使用最佳参数运行最终详细报告:")
            self.run_backtest(hp=best_params, verbose=True)
            return best_params
        else:
            print("⚠️ 优化未能找到符合最低交易次数(>=5)的正收益参数组合，请尝试扩大参数空间或检查数据环境。")
            return None

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
        symbol="BTCUSDT", interval="1m", 
        start_year=start_date.year, start_month=start_date.month, start_day=start_date.day,
        end_year=end_date.year, end_month=end_date.month, end_day=end_date.day,
        data_type="daily"
    )
    
    if df_30d is not None and not df_30d.empty:
        bt = TrailingStrategyOptimizer(df_30d, initial_balance=10000.0)
        
        # 1. 预先计算出所有固定指标
        bt.precalculate_fixed_indicators()
        
        # 2. 运行参数优化，这里设定跑 10000 次组合测试
        best_params = bt.optimize(trials=10000)
    else:
        print("⚠️ 数据拉取失败，请检查网络。")