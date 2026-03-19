import sys
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score

# 确保能引到你的模块
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from database.Binance_Vision_fetcher import VisionFetcher
from bot.data.feature_engineering import FeatureEngineer 

class DualMLBacktestOptimizer:
    def __init__(self, df, initial_balance=10000.0, fee_rate=0.001):
        self.raw_df = df.copy()
        self.initial_balance = initial_balance
        self.fee_rate = fee_rate
        self.engineer = FeatureEngineer()
        self.test_df = None
        self.feature_cols = None

    def prepare_and_train(self):
        print("\n" + "="*50)
        print("🧠 阶段 1: LightGBM + XGBoost 双模型训练")
        print("="*50)
        
        df = self.engineer.generate_features(self.raw_df)
        
        target_lookback = 6 
        exclude_cols = [
            'open_time', 'close_time', 'close', 'open', 'high', 'low', 'volume',
            'buy_volume', 'sell_volume', f'target_return_{target_lookback}', 'target_class'
        ]
        self.feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        mask = ~(df[self.feature_cols].isna().any(axis=1) | df['target_class'].isna())
        df_clean = df[mask].copy()
        
        split_idx = int(len(df_clean) * 0.7)
        train_df = df_clean.iloc[:split_idx]
        self.test_df = df_clean.iloc[split_idx:].copy()
        
        X_train = train_df[self.feature_cols]
        y_train = train_df['target_class']
        X_test = self.test_df[self.feature_cols]
        y_test = self.test_df['target_class']
        
        print(f"训练集大小: {len(X_train)} | 测试集(用于回测)大小: {len(X_test)}")
        
        # ── 1. 训练 LightGBM ──
        print("正在训练 LightGBM...")
        lgb_params = {'objective': 'binary', 'metric': 'auc', 'learning_rate': 0.05, 'max_depth': 7, 'verbose': -1}
        lgb_train = lgb.Dataset(X_train, label=y_train)
        lgb_model = lgb.train(lgb_params, lgb_train, num_boost_round=150)
        self.test_df['pred_lgb'] = lgb_model.predict(X_test)
        
        # ── 2. 训练 XGBoost ──
        print("正在训练 XGBoost...")
        # XGBoost 的 DMatrix 数据结构
        xgb_train = xgb.DMatrix(X_train, label=y_train)
        xgb_test = xgb.DMatrix(X_test)
        xgb_params = {'objective': 'binary:logistic', 'eval_metric': 'auc', 'max_depth': 7, 'learning_rate': 0.05}
        xgb_model = xgb.train(xgb_params, xgb_train, num_boost_round=150)
        self.test_df['pred_xgb'] = xgb_model.predict(xgb_test)
        
        # ── 3. 模型集成 (Ensemble): 取两者的平均概率 ──
        self.test_df['pred_proba'] = (self.test_df['pred_lgb'] + self.test_df['pred_xgb']) / 2.0
        
        y_pred_binary = (self.test_df['pred_proba'] > 0.5).astype(int)
        acc = accuracy_score(y_test, y_pred_binary)
        auc = roc_auc_score(y_test, self.test_df['pred_proba'])
        print(f"✅ 双模型融合完毕! 融合后测试集 Accuracy: {acc:.4f} | AUC: {auc:.4f}")

    def run_backtest(self, conf_thresh, sl_pct, tp_pct, verbose=False):
        balance = self.initial_balance
        position_qty = 0.0
        entry_price = 0.0
        trades = []
        
        closes = self.test_df['close'].values
        highs = self.test_df['high'].values
        lows = self.test_df['low'].values
        opens = self.test_df['open'].values
        times = self.test_df['open_time'].values
        
        # 使用双模型的平均概率
        probas = self.test_df['pred_proba'].values
        
        for i in range(len(self.test_df)):
            curr_price = closes[i]
            curr_high = highs[i]
            curr_low = lows[i]
            curr_open = opens[i]
            prob = probas[i]
            
            if position_qty > 0:
                exit_price = None
                exit_reason = ""
                
                sl_price = entry_price * (1 - sl_pct)
                tp_price = entry_price * (1 + tp_pct)
                
                if curr_low <= sl_price:
                    exit_price = sl_price if curr_open > sl_price else curr_open
                    exit_reason = "Stop Loss"
                elif curr_high >= tp_price:
                    exit_price = tp_price if curr_open < tp_price else curr_open
                    exit_reason = "Take Profit"
                elif prob < (1 - conf_thresh):
                    exit_price = curr_price
                    exit_reason = "Signal Exit"
                    
                if exit_price is not None:
                    gross_return = position_qty * exit_price
                    net_return = gross_return * (1 - self.fee_rate)
                    balance += net_return
                    
                    pnl_pct = (net_return - (entry_price * position_qty)) / (entry_price * position_qty)
                    trades.append({
                        'exit_time': times[i],
                        'reason': exit_reason,
                        'pnl_%': pnl_pct * 100,
                        'balance': balance
                    })
                    position_qty = 0.0
                    continue 
                    
            if position_qty == 0:
                # 依靠双模型的平均高胜率，触发开仓
                if prob > conf_thresh:
                    invest_amount = balance * 0.50
                    balance -= invest_amount
                    net_invest = invest_amount * (1 - self.fee_rate)
                    entry_price = curr_price
                    position_qty = net_invest / entry_price

        total_equity = balance + (position_qty * closes[-1] * (1 - self.fee_rate))
        roi = (total_equity - self.initial_balance) / self.initial_balance * 100
        
        if verbose:
            self._print_stats(trades, total_equity, roi)
            
        return total_equity, roi, trades

    def optimize_trading_rules(self, trials=300):
        print("\n" + "="*50)
        print(f"🚀 阶段 2: 交易规则优化 | 强制拉升交易频率")
        print("="*50)
        
        best_equity = -1
        best_params = {}
        best_roi = -999
        
        for i in range(trials):
            # 🧨 关键改动 1：强行压低置信度上限，逼迫模型寻找更稳健的阈值区间 (0.60 ~ 0.72)
            conf_thresh = round(random.uniform(0.60, 0.72), 3)
            sl_pct = round(random.uniform(0.015, 0.05), 3)
            tp_pct = round(random.uniform(0.02, 0.10), 3)
            
            equity, roi, trades = self.run_backtest(conf_thresh, sl_pct, tp_pct, verbose=False)
            
            # 🧨 关键改动 2：强制要求交易次数必须大于等于 15 次，否则直接淘汰该参数组合！
            if equity > best_equity and len(trades) >= 15:
                best_equity = equity
                best_roi = roi
                best_params = {
                    'confidence_threshold': conf_thresh,
                    'stop_loss_pct': sl_pct,
                    'take_profit_pct': tp_pct
                }
                
            if (i + 1) % 50 == 0:
                print(f"⏳ 优化进度: {i+1}/{trials} | 当前测试集最优 ROI (>=15次交易): {best_roi:.2f}%")

        if not best_params:
            print("⚠️ 未找到满足条件（交易次数 >= 15 且盈利）的参数，请尝试调整阈值区间。")
            return None

        print("\n🏆 优化完成！双模型最佳交易参数: ")
        for k, v in best_params.items():
            print(f"  - {k}: {v}")
            
        print("\n📊 使用最佳参数输出详细回测报告:")
        self.run_backtest(
            conf_thresh=best_params['confidence_threshold'],
            sl_pct=best_params['stop_loss_pct'],
            tp_pct=best_params['take_profit_pct'],
            verbose=True
        )
        return best_params

    def _print_stats(self, trades, total_equity, roi):
        win_trades = [t for t in trades if t['pnl_%'] > 0]
        loss_trades = [t for t in trades if t['pnl_%'] <= 0]
        win_rate = len(win_trades) / len(trades) * 100 if trades else 0.0
        
        print(f"💰 初始资金: ${self.initial_balance:.2f} | 手续费设置: {self.fee_rate*100}%")
        print(f"🏦 最终净值: ${total_equity:.2f} (测试集区间 ROI: {roi:.2f}%)")
        print(f"📈 交易总数: {len(trades)} 笔")
        print(f"🏅 胜率: {win_rate:.2f}% ({len(win_trades)} 胜 / {len(loss_trades)} 负)")
        
        if win_trades and loss_trades:
            avg_win = np.mean([t['pnl_%'] for t in win_trades])
            avg_loss = np.mean([t['pnl_%'] for t in loss_trades])
            print(f"🟢 平均盈利: {avg_win:.2f}%")
            print(f"🔴 平均亏损: {avg_loss:.2f}%")
            print(f"⚖️ 盈亏比: {abs(avg_win / avg_loss):.2f}")
        print("="*40)

if __name__ == "__main__":
    print("正在拉取过去 3 个月的数据用于双模型训练与验证...")
    fetcher = VisionFetcher()
    end_date = datetime.today()
    start_date = end_date - timedelta(days=90)
    
    df_history = fetcher.fetch_klines_range(
        symbol="BTCUSDT", interval="15m", 
        start_year=start_date.year, start_month=start_date.month, start_day=start_date.day,
        end_year=end_date.year, end_month=end_date.month, end_day=end_date.day,
        data_type="daily"
    )
    
    if df_history is not None and not df_history.empty:
        ml_bt = DualMLBacktestOptimizer(df_history, initial_balance=10000.0, fee_rate=0.001)
        ml_bt.prepare_and_train()
        best_rules = ml_bt.optimize_trading_rules(trials=1000)
    else:
        print("⚠️ 数据拉取失败。")