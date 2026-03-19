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
    双模型机器学习量化策略 (LightGBM + XGBoost)
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

        # 固化优化后的最佳参数
        self.conf_thresh = 0.719
        self.sl_pct = 0.025
        self.tp_pct = 0.068
        
        self.engineer = FeatureEngineer()
        self.lgb_model = None
        self.xgb_model = None
        self.feature_cols = None
        self.is_trained = False

    def train_models(self, days_back=365):
        """拉取历史数据并训练双模型"""
        self.logger.info(f"🧠 开始拉取过去 {days_back} 天的数据用于训练...")
        fetcher = VisionFetcher()
        end_date = datetime.today()
        start_date = end_date - timedelta(days=days_back)
        
        # 为了高效，较长的数据使用 monthly 模式拉取，较新的可能需要 daily
        df_history = fetcher.fetch_klines_range(
            symbol=self.symbol, interval="15m", 
            start_year=start_date.year, start_month=start_date.month, start_day=start_date.day,
            end_year=end_date.year, end_month=end_date.month, end_day=end_date.day,
            data_type="daily" # 如果一年数据 daily 太慢，可优化为月度拼接
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
        
        # 训练 LightGBM
        lgb_params = {'objective': 'binary', 'metric': 'auc', 'learning_rate': 0.05, 'max_depth': 7, 'verbose': -1}
        lgb_train = lgb.Dataset(X_train, label=y_train)
        self.lgb_model = lgb.train(lgb_params, lgb_train, num_boost_round=150)
        
        # 训练 XGBoost
        xgb_train = xgb.DMatrix(X_train, label=y_train)
        xgb_params = {'objective': 'binary:logistic', 'eval_metric': 'auc', 'max_depth': 7, 'learning_rate': 0.05}
        self.xgb_model = xgb.train(xgb_params, xgb_train, num_boost_round=150)
        
        self.is_trained = True
        self.logger.info("✅ 模型训练完毕，准备就绪！")
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
            return 0.0 # 存在 NaN 放弃预测
            
        prob_lgb = self.lgb_model.predict(latest_features)[0]
        prob_xgb = self.xgb_model.predict(xgb.DMatrix(latest_features))[0]
        
        # 双模型集成：概率求均值
        return float((prob_lgb + prob_xgb) / 2.0)

    def on_new_candle(self, recent_df: pd.DataFrame, current_price: float):
        """
        实盘执行逻辑：当新 K 线到来时触发
        """
        prob = self.predict_signal(recent_df)
        
        # 检查当前是否已经持有该币种的仓位/锁
        lock_status = self.portfolio.get_coin_lock_status(self.coin)
        
        if not lock_status['locked']:
            # 依靠高置信度触发开仓
            if prob > self.conf_thresh:
                self.logger.info(f"🚀 捕获到高胜率交易信号! 置信度: {prob:.4f} > 阈值: {self.conf_thresh}")
                
                # 计算开仓数量 (假定使用账户余额的 50%)
                invest_amount = self.portfolio.account_balance * 0.50
                quantity = round(invest_amount / current_price, 4)
                
                sl_price = round(current_price * (1 - self.sl_pct), 2)
                tp_price = round(current_price * (1 + self.tp_pct), 2)
                
                self.logger.info(f"发送买入订单: {quantity} {self.coin} | 目标止损: {sl_price}, 止盈: {tp_price}")
                
                # 触发实盘/模拟盘执行
                res = self.execution.execute_order(
                    coin=self.coin,
                    side='BUY',
                    quantity=quantity,
                    price=current_price, # 模拟使用现价作为限价单
                    order_type='LIMIT',
                    strategy_id=self.strategy_id,
                    stop_loss=sl_price,
                    take_profit=tp_price
                )
                
                if res.get('success'):
                    self.logger.info(f"✅ 订单提交成功，ID: {res.get('order_id')}")
                else:
                    self.logger.error(f"❌ 订单提交失败: {res.get('message')}")
        else:
            # 当前持仓中，理论上需要监控止盈止损。实盘中这通常由 Portfolio 后台线程处理。
            pass


# ==========================================
# 步进式模拟器 (模拟未来交易)
# ==========================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🛸 双模型 ML 策略：实盘架构模拟测试")
    print("="*60)
    
    # 1. 初始化交易组件 (模拟环境不需要真实的 API keys，调用逻辑即可)
    roostoo_client = Roostoo()
    portfolio = Portfolio(execution_module=None) # 暂时置空以防死锁，后续组装
    portfolio.account_balance = 10000.0          # 模拟初始资金
    execution = ExecutionEngine(portfolio, roostoo_client)
    portfolio.execution = execution 
    
    strategy = DualMLStrategy(portfolio, execution, symbol="BTCUSDT", coin="BTC")
    
    # 2. 拉取近 1 年数据进行训练
    strategy.train_models(days_back=365)
    
    # 3. 拉取近 1 个月的数据用于步进式回测模拟
    print("\n⏳ 正在拉取近 1 个月的数据用于实盘行情推演...")
    fetcher = VisionFetcher()
    end_date = datetime.today()
    start_date = end_date - timedelta(days=30)
    
    sim_df = fetcher.fetch_klines_range(
        symbol="BTCUSDT", interval="15m", 
        start_year=start_date.year, start_month=start_date.month, start_day=start_date.day,
        end_year=end_date.year, end_month=end_date.month, end_day=end_date.day,
        data_type="daily"
    )
    
    if sim_df is not None and not sim_df.empty:
        print(f"✅ 拉取成功！共 {len(sim_df)} 根 K 线，开始步进式行情推演...")
        
        balance = 10000.0
        position_qty = 0.0
        entry_price = 0.0
        trades_count = 0
        window_size = 50 # 保证技术指标 (如 SMA_24) 计算拥有足够的预热期
        
        # 遍历模拟每一根新 K 线的产生
        for i in range(window_size, len(sim_df)):
            # 提取滑动窗口作为“当前”已知的历史数据
            window_df = sim_df.iloc[i-window_size : i].copy()
            curr_row = window_df.iloc[-1]
            curr_price = curr_row['close']
            curr_high = curr_row['high']
            curr_low = curr_row['low']
            curr_time = curr_row['open_time']
            
            # --- 模拟 Portfolio 和执行引擎的持仓管理与止盈止损 ---
            if position_qty > 0:
                sl_price = entry_price * (1 - strategy.sl_pct)
                tp_price = entry_price * (1 + strategy.tp_pct)
                
                exit_price = None
                reason = ""
                
                if curr_low <= sl_price:
                    exit_price = sl_price
                    reason = "🛑 触发止损"
                elif curr_high >= tp_price:
                    exit_price = tp_price
                    reason = "🎯 触发止盈"
                    
                if exit_price:
                    # 模拟平仓
                    revenue = position_qty * exit_price * (1 - 0.001)
                    pnl_pct = (exit_price - entry_price) / entry_price * 100
                    balance += revenue
                    position_qty = 0.0
                    trades_count += 1
                    print(f"[{curr_time}] {reason} | 卖出价: {exit_price:.2f} | PnL: {pnl_pct:.2f}% | 余额: ${balance:.2f}")
                    # 模拟释放锁
                    portfolio.force_release_coin("BTC")
                    
            # --- 模拟策略引擎寻找开仓信号 ---
            if position_qty == 0:
                prob = strategy.predict_signal(window_df)
                if prob > strategy.conf_thresh:
                    # 模拟触发开仓
                    invest = balance * 0.50
                    balance -= invest
                    position_qty = (invest * (1 - 0.001)) / curr_price
                    entry_price = curr_price
                    print(f"[{curr_time}] 🟢 发起买单 | 概率: {prob:.4f} | 买入价: {entry_price:.2f}")
                    # 模拟加锁
                    portfolio.acquire_coin("BTC", strategy.strategy_id)
        
        # 强制平仓清算最后的净值
        final_equity = balance + (position_qty * curr_price * (1 - 0.001))
        roi = (final_equity - 10000.0) / 10000.0 * 100
        print("\n" + "="*40)
        print("🏁 模拟运行结束！")
        print(f"总交易次数: {trades_count}")
        print(f"初始资金: $10000.00")
        print(f"最终净值: ${final_equity:.2f} (月度模拟 ROI: {roi:.2f}%)")
        print("="*40)