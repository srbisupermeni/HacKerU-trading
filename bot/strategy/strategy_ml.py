import sys
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import lightgbm as lgb
import xgboost as xgb

# 确保能引到根目录的模块
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from database.Binance_Vision_fetcher import VisionFetcher
from database.Binance_fetcher import BinanceDataFetcher
from bot.data.feature_engineering import FeatureEngineer
from bot.portfolio.portfolio import Portfolio
from bot.execution.execution_engine import ExecutionEngine
from bot.execution.roostoo import Roostoo

class DualMLStrategy:
    """
    双模型机器学习量化策略 - 10天比赛高频专供版 (5m 级别)
    支持接入 Roostoo 实盘 API 及本地 Portfolio 资产管理
    """
    def __init__(self, portfolio: Portfolio, execution: ExecutionEngine, 
                 symbol="BTCUSDT", coin="BTC", strategy_id="ml_dual_01"):
        self.portfolio = portfolio
        self.execution = execution
        self.symbol = symbol
        self.coin = coin
        self.strategy_id = strategy_id
        
        self.logger = logging.getLogger("DualMLStrategy")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            ch = logging.StreamHandler()
            ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(ch)

        # 🏆 比赛专属优化参数 (基于 5m 级别刮头皮优化)
        self.conf_thresh = 0.663   # 极高胜率狙击门槛
        self.sl_pct = 0.019        # 1.9% 宽止损防插针
        self.tp_pct = 0.024        # 2.4% 大波段止盈
        
        self.engineer = FeatureEngineer()
        self.lgb_model = None
        self.xgb_model = None
        self.feature_cols = None
        self.is_trained = False

    def train_models(self, days_back=60):
        """拉取近期数据训练双模型 (锁定近期盘感)"""
        self.logger.info(f"🧠 开始拉取过去 {days_back} 天的 5m 数据用于实盘前训练...")
        fetcher = VisionFetcher()
        end_date = datetime.today()
        start_date = end_date - timedelta(days=days_back)
        
        df_history = fetcher.fetch_klines_range(
            symbol=self.symbol, interval="5m",  # 切换至 5m 级别
            start_year=start_date.year, start_month=start_date.month, start_day=start_date.day,
            end_year=end_date.year, end_month=end_date.month, end_day=end_date.day,
            data_type="daily" 
        )
        
        if df_history is None or df_history.empty:
            self.logger.error("数据拉取失败，无法训练模型！")
            return False

        self.logger.info("进行特征工程处理...")
        df = self.engineer.generate_features(df_history)
        
        target_lookback = 6 
        exclude_cols = [
            'open_time', 'close_time', 'close', 'open', 'high', 'low', 'volume',
            'buy_volume', 'sell_volume', f'target_return_{target_lookback}', 'target_class'
        ]
        self.feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        mask = ~(df[self.feature_cols].isna().any(axis=1) | df['target_class'].isna())
        df_clean = df[mask].copy()
        
        X_train = df_clean[self.feature_cols]
        y_train = df_clean['target_class']
        
        self.logger.info(f"训练集数据量: {len(X_train)}，开始训练 LightGBM 和 XGBoost...")
        
        lgb_params = {'objective': 'binary', 'metric': 'auc', 'learning_rate': 0.05, 'max_depth': 7, 'verbose': -1}
        lgb_train = lgb.Dataset(X_train, label=y_train)
        self.lgb_model = lgb.train(lgb_params, lgb_train, num_boost_round=150)
        
        xgb_train = xgb.DMatrix(X_train, label=y_train)
        xgb_params = {'objective': 'binary:logistic', 'eval_metric': 'auc', 'max_depth': 7, 'learning_rate': 0.05}
        self.xgb_model = xgb.train(xgb_params, xgb_train, num_boost_round=150)
        
        self.is_trained = True
        self.logger.info("✅ 模型训练完毕，实盘武装就绪！")
        return True

    def predict_signal(self, recent_df: pd.DataFrame) -> float:
        """根据最新的窗口数据预测上涨概率"""
        if not self.is_trained:
            self.logger.warning("模型尚未训练！")
            return 0.0
            
        features_df = self.engineer.generate_features(recent_df)
        if features_df.empty:
            return 0.0
            
        latest_features = features_df.iloc[-1:][self.feature_cols]
        if latest_features.isna().any(axis=1).iloc[0]:
            return 0.0 
            
        prob_lgb = self.lgb_model.predict(latest_features)[0]
        prob_xgb = self.xgb_model.predict(xgb.DMatrix(latest_features))[0]
        
        return float((prob_lgb + prob_xgb) / 2.0)

    def on_new_candle(self, recent_df: pd.DataFrame, current_price: float):
        """
        实盘执行逻辑：当新 5m K线到来时触发
        """
        prob = self.predict_signal(recent_df)
        lock_status = self.portfolio.get_coin_lock_status(self.coin)
        
        # 1. 空仓状态：寻找高确定性开仓机会
        if not lock_status['locked']:
            if prob > self.conf_thresh:
                self.logger.info(f"🎯 [狙击时刻] 置信度: {prob:.4f} > {self.conf_thresh}")
                
                # 满载复利：每次使用 85% 资金开仓
                invest_amount = self.portfolio.account_balance * 0.85
                quantity = round(invest_amount / current_price, 4)
                
                sl_price = round(current_price * (1 - self.sl_pct), 2)
                tp_price = round(current_price * (1 + self.tp_pct), 2)
                
                self.logger.info(f"🚀 发送买入订单: {quantity} {self.coin} | 止损: {sl_price}, 止盈: {tp_price}")
                
                res = self.execution.execute_order(
                    coin=self.coin, side='BUY', quantity=quantity, 
                    price=current_price, order_type='LIMIT', 
                    strategy_id=self.strategy_id, stop_loss=sl_price, take_profit=tp_price
                )
                
                if res.get('success'):
                    self.logger.info(f"✅ 订单提交成功，ID: {res.get('order_id')}")
                else:
                    self.logger.error(f"❌ 订单提交失败: {res.get('message')}")
                    
        # 2. 持仓状态：AI 智能盯盘，预防极端反转
        else:
            if prob < 0.20:
                self.logger.warning(f"⚠️ [AI 警报] 趋势极度恶化 (概率 {prob:.4f} < 0.20)，建议执行提前平仓！")
                # 这里可以接入你的 execution_engine 中的强平逻辑
                # 例如：self.execution.execute_order(..., side='SELL', ...)


# ==========================================
# 实盘推演模拟器 (模拟未来 10 天比赛真实环境)
# ==========================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🏆 双模型 ML 策略：10 天比赛前置推演模拟")
    print("="*60)
    
    roostoo_client = Roostoo()
    portfolio = Portfolio(execution_module=None) 
    portfolio.account_balance = 10000.0          
    execution = ExecutionEngine(portfolio, roostoo_client)
    portfolio.execution = execution 
    
    strategy = DualMLStrategy(portfolio, execution, symbol="BTCUSDT", coin="BTC")
    
    # 1. 训练近 60 天数据
    strategy.train_models(days_back=60)
    
    # 2. 拉取最近 10 天的 5m 级别数据，演习接下来的比赛
    print("\n⏳ 正在拉取近 10 天的数据进行实盘推演...")
    fetcher = VisionFetcher()
    end_date = datetime.today()
    start_date = end_date - timedelta(days=10)
    
    sim_df = fetcher.fetch_klines_range(
        symbol="BTCUSDT", interval="5m", 
        start_year=start_date.year, start_month=start_date.month, start_day=start_date.day,
        end_year=end_date.year, end_month=end_date.month, end_day=end_date.day,
        data_type="daily"
    )
    
    if sim_df is not None and not sim_df.empty:
        print(f"✅ 拉取成功！共 {len(sim_df)} 根 K 线，开始逐根 K 线推演...")
        
        balance = 10000.0
        position_qty = 0.0
        entry_price = 0.0
        trades_count = 0
        window_size = 50 
        
        for i in range(window_size, len(sim_df)):
            window_df = sim_df.iloc[i-window_size : i].copy()
            curr_row = window_df.iloc[-1]
            curr_price = curr_row['close']
            curr_high = curr_row['high']
            curr_low = curr_row['low']
            curr_time = curr_row['open_time']
            
            # --- 模拟 Portfolio 持仓管理 ---
            if position_qty > 0:
                sl_price = entry_price * (1 - strategy.sl_pct)
                tp_price = entry_price * (1 + strategy.tp_pct)
                prob = strategy.predict_signal(window_df)
                
                exit_price = None
                reason = ""
                
                if curr_low <= sl_price:
                    exit_price = sl_price
                    reason = "🛑 触发止损"
                elif curr_high >= tp_price:
                    exit_price = tp_price
                    reason = "🎯 触发大波段止盈"
                elif prob < 0.20:
                    exit_price = curr_price
                    reason = "⚠️ 触发 AI 提前平仓"
                    
                if exit_price:
                    revenue = position_qty * exit_price * (1 - 0.001)
                    pnl_pct = (exit_price - entry_price) / entry_price * 100
                    balance += revenue
                    position_qty = 0.0
                    trades_count += 1
                    print(f"[{curr_time}] {reason} | 卖出价: {exit_price:.2f} | 盈亏: {pnl_pct:.2f}% | 余额: ${balance:.2f}")
                    portfolio.force_release_coin("BTC")
                    
            # --- 模拟开仓 ---
            if position_qty == 0:
                prob = strategy.predict_signal(window_df)
                if prob > strategy.conf_thresh:
                    invest = balance * 0.85
                    balance -= invest
                    position_qty = (invest * (1 - 0.001)) / curr_price
                    entry_price = curr_price
                    print(f"[{curr_time}] 🟢 发起重仓买单 | 概率: {prob:.4f} | 买入价: {entry_price:.2f}")
                    portfolio.acquire_coin("BTC", strategy.strategy_id)
        
        final_equity = balance + (position_qty * curr_price * (1 - 0.001)) if position_qty > 0 else balance
        roi = (final_equity - 10000.0) / 10000.0 * 100
        print("\n" + "="*40)
        print("🏁 10天赛前推演结束！")
        print(f"总交易次数: {trades_count}")
        print(f"初始资金: $10000.00")
        print(f"最终净值: ${final_equity:.2f} (10天预期 ROI: {roi:.2f}%)")
        print("="*40)