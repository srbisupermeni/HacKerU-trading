import sys
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# 导入机器学习核心库
import lightgbm as lgb
import xgboost as xgb

# ==========================================
# 路径兼容处理：确保在 AWS/Docker 容器内运行时，
# 能够正确识别项目根目录，避免 ModuleNotFoundError
# ==========================================
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# 导入项目内的自定义组件
from database.Binance_Vision_fetcher import VisionFetcher
from database.Binance_fetcher import BinanceDataFetcher
from bot.data.feature_engineering import FeatureEngineer
from bot.portfolio.portfolio import Portfolio
from bot.execution.execution_engine import ExecutionEngine
from bot.execution.roostoo import Roostoo

class DualMLStrategy:
    """
    多币种适配版：双模型机器学习量化策略 (5m 级别)
    
    【模块定位】
    本类是单个币种的“专属大脑”。如果在 main.py 中跑 3 个币，就会实例化 3 个本类的对象。
    它负责：
    1. 启动时拉取该币种专属的历史数据，训练 LightGBM 和 XGBoost。
    2. 实盘中每 5 分钟接收一次最新数据，输出上涨概率（胜率）。
    3. 根据胜率和当前持仓状态，向 ExecutionEngine 发送买入指令或强平警报。
    """
    def __init__(self, portfolio: Portfolio, execution: ExecutionEngine, 
                 symbol="BTCUSDT", coin="BTC", strategy_id="ml_dual_01", alloc_ratio=0.30):
        # 资产管家与执行引擎
        self.portfolio = portfolio
        self.execution = execution
        
        self.symbol = symbol          # 币安数据对，如 SOLUSDT
        self.coin = coin              # Roostoo下单币种，如 SOL
        self.strategy_id = strategy_id # 策略唯一标识符，用于 portfolio 记账
        
        # 🧨 多币种关键参数：资金分配率。0.30 代表本策略每次只动用账户总资金的 30%
        # 这样即使 3 个币同时出信号，也能保证有足够的余额下单。
        self.alloc_ratio = alloc_ratio 
        
        # 为不同币种配置独立的日志，方便在 AWS 上排查问题
        self.logger = logging.getLogger(f"DualML_{self.coin}")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            ch = logging.StreamHandler()
            ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(ch)

        # ==========================================
        # 🏆 比赛专属优化参数 (基于 5m 级别刮头皮优化得出)
        # ==========================================
        self.conf_thresh = 0.663   # 极高胜率狙击门槛，双模型认为上涨概率 > 66.3% 才开仓
        self.sl_pct = 0.019        # 1.9% 宽止损，防止 5m 级别的上下影线“插针”扫损
        self.tp_pct = 0.024        # 2.4% 波段止盈
        
        self.engineer = FeatureEngineer()
        self.lgb_model = None
        self.xgb_model = None
        self.feature_cols = None
        self.is_trained = False

    def train_models(self, days_back=60, train_end_days_ago=0):
        """
        【初始化训练阶段】
        增加了 train_end_days_ago 参数，用于在回测时隔离测试集，防止偷看未来数据。
        """
        self.logger.info(f"🧠 开始拉取 {self.symbol} 历史数据进行专模训练...")
        
        fetcher = VisionFetcher()
        # 🧨 核心修复：训练数据的结束时间，必须在测试数据的开始时间之前！
        end_date = datetime.today() - timedelta(days=train_end_days_ago)
        start_date = end_date - timedelta(days=days_back)
        
        df_history = fetcher.fetch_klines_range(
            symbol=self.symbol, interval="5m", 
            start_year=start_date.year, start_month=start_date.month, start_day=start_date.day,
            end_year=end_date.year, end_month=end_date.month, end_day=end_date.day,
            data_type="daily" 
        )
        
        if df_history is None or df_history.empty:
            self.logger.error(f"[{self.symbol}] 数据拉取失败，无法训练模型！")
            return False

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
        
        self.logger.info(f"[{self.symbol}] 训练集数据量: {len(X_train)}，开始训练...")
        lgb_params = {'objective': 'binary', 'metric': 'auc', 'learning_rate': 0.05, 'max_depth': 7, 'verbose': -1}
        lgb_train = lgb.Dataset(X_train, label=y_train)
        self.lgb_model = lgb.train(lgb_params, lgb_train, num_boost_round=150)
        
        xgb_train = xgb.DMatrix(X_train, label=y_train)
        xgb_params = {'objective': 'binary:logistic', 'eval_metric': 'auc', 'max_depth': 7, 'learning_rate': 0.05}
        self.xgb_model = xgb.train(xgb_params, xgb_train, num_boost_round=150)
        
        self.is_trained = True
        self.logger.info(f"✅ [{self.symbol}] 模型训练完毕！")
        return True

    def predict_signal(self, recent_df: pd.DataFrame) -> float:
        """
        【实时推理阶段】
        传入最新的 K 线窗口，提取最后一行特征，输出预测胜率。
        """
        if not self.is_trained:
            return 0.0
            
        features_df = self.engineer.generate_features(recent_df)
        if features_df.empty:
            return 0.0
            
        latest_features = features_df.iloc[-1:][self.feature_cols]
        if latest_features.isna().any(axis=1).iloc[0]:
            return 0.0 
            
        # 集成学习：LGB 和 XGB 概率求平均，提升稳定性
        prob_lgb = self.lgb_model.predict(latest_features)[0]
        prob_xgb = self.xgb_model.predict(xgb.DMatrix(latest_features))[0]
        return float((prob_lgb + prob_xgb) / 2.0)

    def on_new_candle(self, recent_df: pd.DataFrame, current_price: float):
        """
        【实盘决策核心】
        主程序 (main.py) 获取到新 K 线后调用此方法。执行推断并决定买入或强平。
        """
        prob = self.predict_signal(recent_df)
        lock_status = self.portfolio.get_coin_lock_status(self.coin)
        
        # 场景 A：空仓状态，寻找买点
        if not lock_status['locked']:
            if prob > self.conf_thresh:
                self.logger.info(f"🎯 [{self.coin} 狙击时刻] 胜率: {prob:.4f} > {self.conf_thresh}")
                
                # 分配指定比例的资金 (如 30%)
                invest_amount = self.portfolio.account_balance * self.alloc_ratio
                quantity = round(invest_amount / current_price, 4)
                
                sl_price = round(current_price * (1 - self.sl_pct), 4)
                tp_price = round(current_price * (1 + self.tp_pct), 4)
                
                self.logger.info(f"🚀 [{self.coin}] 发送买单: 数量 {quantity} | 止损: {sl_price}, 止盈: {tp_price}")
                
                res = self.execution.execute_order(
                    coin=self.coin, side='BUY', quantity=quantity, 
                    price=current_price, order_type='LIMIT', 
                    strategy_id=self.strategy_id, stop_loss=sl_price, take_profit=tp_price
                )
                
                if res.get('success'):
                    self.logger.info(f"✅ 订单提交成功，单号: {res.get('order_id')}")
                else:
                    self.logger.error(f"❌ 订单提交失败: {res.get('message')}")
                    
        # 场景 B：持仓状态，AI 风控盯盘
        else:
            # 如果上涨概率骤降到 20% 以下，说明跌的概率高达 80%，极度危险！
            if prob < 0.20:
                self.logger.warning(f"⚠️ [{self.coin} AI 警报] 趋势极度恶化 (胜率仅 {prob:.4f})，注意强平风险！")


# ==========================================
# 本地多币种并发推演测试区 (__main__)
# ==========================================
# ==========================================
# 本地“多币种选秀”批量推演测试区 (__main__)
# ==========================================
# ==========================================
# 本地“多币种选秀”批量推演测试区 (__main__)
# ==========================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🏆 严格样本外 (OOS) 多币种选秀大赛：寻找最强 3 币组合")
    print("="*60)
    
    CANDIDATES = [
        {"symbol": "ETHUSDT", "coin": "ETH"},
        {"symbol": "SOLUSDT", "coin": "SOL"},
        {"symbol": "DOGEUSDT", "coin": "DOGE"},
        {"symbol": "PEPEUSDT", "coin": "PEPE"},
        {"symbol": "WIFUSDT", "coin": "WIF"},
        {"symbol": "SUIUSDT", "coin": "SUI"},
        {"symbol": "APTUSDT", "coin": "APT"},
        {"symbol": "NEARUSDT", "coin": "NEAR"}
    ]
    
    leaderboard = []
    
    for item in CANDIDATES:
        symbol = item['symbol']
        coin = item['coin']
        print(f"\n" + "-"*40)
        print(f"🚀 正在盲测币种: {coin} (严格隔离未来数据)")
        print("-"*40)
        
        roostoo_client = Roostoo()
        portfolio = Portfolio(execution_module=None) 
        portfolio.account_balance = 10000.0          
        execution = ExecutionEngine(portfolio, roostoo_client)
        portfolio.execution = execution 
        
        strat_id = f"ml_dual_{coin.lower()}"
        strategy = DualMLStrategy(portfolio, execution, symbol=symbol, coin=coin, strategy_id=strat_id, alloc_ratio=0.85)
        
        # 🧨 核心修复：训练集只取到 10天前，严禁模型看到最近 10 天的答案！
        success = strategy.train_models(days_back=60, train_end_days_ago=10)
        if not success:
            print(f"⚠️ {coin} 模型训练失败，跳过。")
            continue
            
        # 测试集：最近的 10 天
        fetcher = VisionFetcher()
        end_date = datetime.today()
        start_date = end_date - timedelta(days=10)
        
        sim_df = fetcher.fetch_klines_range(
            symbol=symbol, interval="5m", 
            start_year=start_date.year, start_month=start_date.month, start_day=start_date.day,
            end_year=end_date.year, end_month=end_date.month, end_day=end_date.day,
            data_type="daily"
        )
        
        if sim_df is None or sim_df.empty:
            print(f"⚠️ {coin} 测试数据拉取失败，跳过。")
            continue
            
        balance = 10000.0
        position_qty = 0.0
        entry_price = 0.0
        trades_count = 0
        win_count = 0
        window_size = 50 
        
        for i in range(window_size, len(sim_df)):
            window_df = sim_df.iloc[i-window_size : i].copy()
            curr_row = window_df.iloc[-1]
            curr_price = curr_row['close']
            curr_high = curr_row['high']
            curr_low = curr_row['low']
            
            if position_qty > 0:
                sl_price = entry_price * (1 - strategy.sl_pct)
                tp_price = entry_price * (1 + strategy.tp_pct)
                prob = strategy.predict_signal(window_df)
                
                exit_price = None
                
                if curr_low <= sl_price:
                    exit_price = sl_price
                elif curr_high >= tp_price:
                    exit_price = tp_price
                elif prob < 0.20:
                    exit_price = curr_price
                    
                if exit_price:
                    revenue = position_qty * exit_price * (1 - 0.001)
                    if exit_price > entry_price:
                        win_count += 1
                    balance += revenue
                    position_qty = 0.0
                    trades_count += 1
                    portfolio.force_release_coin(coin)
                    
            if position_qty == 0:
                prob = strategy.predict_signal(window_df)
                if prob > strategy.conf_thresh:
                    invest = balance * strategy.alloc_ratio
                    balance -= invest
                    position_qty = (invest * (1 - 0.001)) / curr_price
                    entry_price = curr_price
                    portfolio.acquire_coin(coin, strategy.strategy_id)
        
        final_equity = balance + (position_qty * curr_price * (1 - 0.001)) if position_qty > 0 else balance
        roi = (final_equity - 10000.0) / 10000.0 * 100
        win_rate = (win_count / trades_count * 100) if trades_count > 0 else 0.0
        
        print(f"🏁 {coin} 盲测结束: ROI {roi:.2f}%, 交易次数 {trades_count}, 胜率 {win_rate:.1f}%")
        
        leaderboard.append({
            "Coin": coin,
            "ROI (%)": round(roi, 2),
            "Trades": trades_count,
            "Win Rate (%)": round(win_rate, 1)
        })

    print("\n" + "🏆"*20)
    print("      严谨样本外盲测 10天推演 排行榜 (按 ROI 排序)")
    print("🏆"*20)
    leaderboard_sorted = sorted(leaderboard, key=lambda x: x["ROI (%)"], reverse=True)
    df_leaderboard = pd.DataFrame(leaderboard_sorted)
    print(df_leaderboard.to_string(index=False))