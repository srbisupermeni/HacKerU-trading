"""
=============================================================================
Machine Learning Strategy (strategy4.py)
综合 LightGBM/XGBoost 机器学习模型的实盘交易策略
支持离线回测与在线实时推理
=============================================================================

【架构概览】
1. 数据准备阶段 (Data Preparation):
   - 从 Binance 获取历史 K 线数据 (e.g., 过去 7 天的 15 分钟线)
   - 通过 FeatureEngineer 生成 ML 特征集
   - 训练/验证 (70/30 split)

2. 模型训练阶段 (Training):
   - 使用 LightGBM 或 XGBoost 拟合历史数据
   - 评估模型性能 (精度、AUC、ROC)
   - 保存模型到本地

3. 实盘推理阶段 (Inference):
   - 实时获取最新 K 线数据
   - 生成实时特征
   - 使用模型预测下一个 6 周期内的价格涨跌概率
   - 基于置信度生成交易信号
   - 管理头寸、止损、止盈

【使用示例】
strategy = MLStrategy(
    symbol="BTCUSDT",
    interval="15m",
    model_type="lightgbm",  # or "xgboost"
    confidence_threshold=0.65,  # 信心阈值：預測概率 > 65% 才交易
    training_lookback=7  # 用过去 7 天数据训练
)

# 一次性训练
strategy.train(force_retrain=True)

# 持续运行推理
strategy.run_live_trading()
"""

import sys
from pathlib import Path
import time
import logging
import pickle
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# 添加项目根目录到 Python 搜索路径
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))
sys.path.append(str(root_dir / "database"))

from database.Binance_fetcher import BinanceDataFetcher
from bot.data.feature_engineering import FeatureEngineer

# ML 库
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings('ignore')

# ==========================================
# 日志配置
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    filename=str(root_dir / "bot" / "logs" / "strategy4.log"),
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MLStrategy:
    """
    机器学习交易策略核心类
    """
    def __init__(
        self,
        symbol="BTCUSDT",
        interval="15m",
        model_type="lightgbm",  # "lightgbm" or "xgboost"
        confidence_threshold=0.65,  # 预测置信度阈值 (0-1)
        training_lookback=7,  # 回溯天数用于训练
        target_lookback=6,  # 预测未来周期数
        max_positions=1,  # 最多持有头寸数
        stop_loss_pct=0.02,  # 止损百分比 (2%)
        take_profit_pct=0.05,  # 止盈百分比 (5%)
    ):
        self.symbol = symbol
        self.interval = interval
        self.model_type = model_type.lower()
        self.confidence_threshold = confidence_threshold
        self.training_lookback = training_lookback
        self.target_lookback = target_lookback  
        self.max_positions = max_positions
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        
        # 初始化数据获取器与特征工程师
        self.fetcher = BinanceDataFetcher()
        self.engineer = FeatureEngineer()
        
        # 模型与模型路径
        self.model = None
        self.model_path = root_dir / "bot" / "models" / f"model_{symbol}_{interval}_{model_type}.pkl"
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 特征列表（在训练时确定）
        self.feature_columns = None
        
        # 头寸追踪
        self.holding = False
        self.entry_price = None
        self.entry_time = None
        self.trade_log = []
        
        logger.info(f"MLStrategy initialized: {symbol} {interval} ({model_type})")
    
    def fetch_training_data(self, days_back=None):
        """
        获取用于训练的历史数据
        :param days_back: 向后回溯的天数（默认为 self.training_lookback）
        :return: DataFrame with raw OHLCV + buy/sell volumes
        """
        if days_back is None:
            days_back = self.training_lookback
        
        # 根据 interval 推断每天的 K 线数
        interval_to_count = {
            "1m": 1440,
            "5m": 288,
            "15m": 96,
            "1h": 24,
            "4h": 6,
            "1d": 1
        }
        limit = interval_to_count.get(self.interval, 100) * days_back
        limit = min(limit, 1000)  # Binance API 最多返回 1000 条
        
        logger.info(f"Fetching {limit} {self.interval} candles for training...")
        df = self.fetcher.fetch_recent_klines(
            symbol=self.symbol,
            interval=self.interval,
            limit=limit
        )
        
        if df is None or df.empty:
            logger.error("Failed to fetch training data")
            return None
        
        logger.info(f"Fetched {len(df)} candles for training")
        return df
    
    def prepare_training_data(self, raw_df):
        """
        数据预处理：
        1. 生成 ML 特征
        2. 删除 NaN 值
        3. 分离 X（特征）和 y（目标）
        :return: (X, y, feature_names)
        """
        if raw_df is None or raw_df.empty:
            return None, None, None
        
        # 生成特征
        features_df = self.engineer.generate_features(raw_df)
        
        if features_df.empty:
            logger.error("Feature generation produced empty DataFrame")
            return None, None, None
        
        # 提取特征列和目标列
        # 排除: open_time, close, open, high, low, volume, buy_volume, sell_volume, target_return_*, target_class
        exclude_cols = [
            'open_time', 'close_time', 'close', 'open', 'high', 'low', 'volume',
            'buy_volume', 'sell_volume', f'target_return_{self.target_lookback}', 'target_class'
        ]
        feature_cols = [col for col in features_df.columns if col not in exclude_cols]
        
        # 删除这一批数据中有 NaN 的行
        X = features_df[feature_cols].copy()
        y = features_df['target_class'].copy()
        
        # 删除行中存在 NaN 的记录
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]
        
        logger.info(f"Prepared {len(X)} samples for training with {len(feature_cols)} features")
        
        return X, y, feature_cols
    
    def train(self, force_retrain=False):
        """
        训练 ML 模型
        :param force_retrain: 是否强制重新训练（忽略已有模型）
        """
        # 如果模型已存在且不强制重训，直接加载
        if self.model_path.exists() and not force_retrain:
            logger.info(f"Loading existing model from {self.model_path}")
            self.load_model()
            return
        
        logger.info("Starting model training...")
        
        # 1. 获取历史数据
        raw_df = self.fetch_training_data()
        if raw_df is None:
            logger.error("Failed to fetch training data")
            return
        
        # 2. 准备数据
        X, y, feature_cols = self.prepare_training_data(raw_df)
        if X is None or X.empty:
            logger.error("Failed to prepare training data")
            return
        
        self.feature_columns = feature_cols
        
        # 3. 分割训练/验证集 (70/30)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        logger.info(f"Train set: {len(X_train)}, Test set: {len(X_test)}")
        logger.info(f"Class distribution: {y.value_counts().to_dict()}")
        
        # 4. 训练模型
        if self.model_type == "lightgbm":
            self.model = self._train_lightgbm(X_train, y_train, X_test, y_test)
        elif self.model_type == "xgboost":
            self.model = self._train_xgboost(X_train, y_train, X_test, y_test)
        else:
            logger.error(f"Unknown model type: {self.model_type}")
            return
        
        # 5. 评估模型
        self._evaluate_model(self.model, X_test, y_test)
        
        # 6. 保存模型
        self.save_model()
        logger.info(f"Model saved to {self.model_path}")
    
    def _train_lightgbm(self, X_train, y_train, X_test, y_test):
        """训练 LightGBM 模型"""
        logger.info("Training LightGBM model...")
        
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'learning_rate': 0.05,
            'max_depth': 7,
            'num_leaves': 31,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
        }
        
        train_data = lgb.Dataset(X_train, label=y_train, feature_names=self.feature_columns)
        test_data = lgb.Dataset(X_test, label=y_test, reference=train_data, feature_names=self.feature_columns)
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=200,
            valid_sets=[train_data, test_data],
            callbacks=[lgb.log_evaluation(period=50)],
        )
        
        return model
    
    def _train_xgboost(self, X_train, y_train, X_test, y_test):
        """训练 XGBoost 模型"""
        logger.info("Training XGBoost model...")
        
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'learning_rate': 0.05,
            'max_depth': 7,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'verbosity': 0,
        }
        
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=self.feature_columns)
        dtest = xgb.DMatrix(X_test, label=y_test, feature_names=self.feature_columns)
        
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=200,
            evals=[(dtrain, 'train'), (dtest, 'test')],
            verbose_eval=50,
        )
        
        return model
    
    def _evaluate_model(self, model, X_test, y_test):
        """评估模型性能"""
        y_pred_proba = self.predict_proba(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        logger.info(f"Model Performance:")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  AUC-ROC: {auc:.4f}")
        logger.info(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
        logger.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")
    
    def save_model(self):
        """保存模型到本地"""
        with open(self.model_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'feature_columns': self.feature_columns,
                'model_type': self.model_type,
            }, f)
        logger.info(f"Model saved to {self.model_path}")
    
    def load_model(self):
        """从本地加载模型"""
        if not self.model_path.exists():
            logger.warning(f"Model file not found: {self.model_path}")
            return False
        
        try:
            with open(self.model_path, 'rb') as f:
                data = pickle.load(f)
                self.model = data['model']
                self.feature_columns = data['feature_columns']
                logger.info(f"Model loaded from {self.model_path}")
                return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def predict_proba(self, X):
        """
        预测概率（上涨的概率）
        :param X: 特征 DataFrame
        :return: 上涨概率数组 [0, 1]
        """
        if self.model is None:
            return None
        
        if self.model_type == "lightgbm":
            return self.model.predict(X)
        elif self.model_type == "xgboost":
            dmatrix = xgb.DMatrix(X, feature_names=self.feature_columns)
            return self.model.predict(dmatrix)
    
    def generate_signal(self, prediction_proba):
        """
        基于预测概率生成交易信号
        :param prediction_proba: 价格上涨的概率 (0-1)
        :return: "Buy", "Sell", or None
        """
        if prediction_proba > self.confidence_threshold:
            return "Buy"
        elif prediction_proba < (1 - self.confidence_threshold):
            return "Sell"
        else:
            return None
    
    def run_inference_once(self):
        """
        单次推理：获取最新数据 -> 预测 -> 生成信号
        :return: (signal, confidence)
        """
        # 1. 获取最近数据（足够生成特征）
        recent_df = self.fetcher.fetch_recent_klines(
            symbol=self.symbol,
            interval=self.interval,
            limit=50  # 最少需要足够的历史数据用于特征计算
        )
        
        if recent_df is None or recent_df.empty:
            logger.warning("Failed to fetch recent data")
            return None, 0
        
        # 2. 生成特征
        features_df = self.engineer.generate_features(recent_df)
        
        if features_df.empty:
            logger.warning("Feature generation failed")
            return None, 0
        
        # 3. 取最后一行（最新的预测）
        latest_features = features_df[self.feature_columns].iloc[-1:]
        
        # 检查是否含有 NaN
        if latest_features.isna().any().any():
            logger.warning("Latest features contain NaN")
            return None, 0.5
        
        # 4. 预测
        pred_proba = self.predict_proba(latest_features)[0]
        
        # 5. 生成信号
        signal = self.generate_signal(pred_proba)
        
        logger.info(f"Inference: Probability={pred_proba:.4f}, Signal={signal}")
        
        return signal, pred_proba
    
    def manage_position(self, current_price, signal):
        """
        头寸管理逻辑：
        - 如果未持仓且信号为 Buy -> 开多
        - 如果持仓：检查止损/止盈
        - 如果信号为 Sell -> 平仓
        
        :param current_price: 当前价格
        :param signal: 交易信号 ("Buy", "Sell", None)
        :return: 执行的操作 ("Entry", "Exit", "StopLoss", None)
        """
        current_time = datetime.now()
        
        if not self.holding:
            # 未持仓逻辑
            if signal == "Buy":
                self.holding = True
                self.entry_price = current_price
                self.entry_time = current_time
                logger.info(f"ENTRY: Bought at {current_price} at {current_time}")
                self.trade_log.append({
                    'time': current_time,
                    'type': 'Entry',
                    'price': current_price,
                    'signal': signal
                })
                return "Entry"
        else:
            # 持仓逻辑
            price_change_pct = (current_price - self.entry_price) / self.entry_price
            
            # 检查止盈
            if price_change_pct > self.take_profit_pct:
                self.holding = False
                profit = (current_price - self.entry_price) * self.max_positions  # 假设头寸大小为 1
                logger.info(f"TAKE PROFIT: Sold at {current_price}, Profit: {profit}")
                self.trade_log.append({
                    'time': current_time,
                    'type': 'TakeProfit',
                    'price': current_price,
                    'pnl': profit
                })
                return "Exit"
            
            # 检查止损
            elif price_change_pct < -self.stop_loss_pct:
                self.holding = False
                loss = (current_price - self.entry_price) * self.max_positions
                logger.info(f"STOP LOSS: Sold at {current_price}, Loss: {loss}")
                self.trade_log.append({
                    'time': current_time,
                    'type': 'StopLoss',
                    'price': current_price,
                    'pnl': loss
                })
                return "StopLoss"
            
            # 检查信号反转
            elif signal == "Sell":
                self.holding = False
                pnl = (current_price - self.entry_price) * self.max_positions
                logger.info(f"SELL SIGNAL: Sold at {current_price}, PnL: {pnl}")
                self.trade_log.append({
                    'time': current_time,
                    'type': 'SignalExit',
                    'price': current_price,
                    'pnl': pnl
                })
                return "Exit"
        
        return None
    
    def run_live_trading(self, check_interval_sec=60):
        """
        实时交易主循环
        :param check_interval_sec: 检查间隔（秒）
        """
        logger.info("Starting live trading loop...")
        
        # 首先确保模型已加载或已训练
        if self.model is None:
            if not self.load_model():
                logger.info("No existing model, training new one...")
                self.train(force_retrain=True)
        
        cycle_count = 0
        try:
            while True:
                cycle_count += 1
                current_time = datetime.now()
                logger.info(f"==== Cycle {cycle_count} at {current_time} ====")
                
                try:
                    # 1. 推理获取信号
                    signal, confidence = self.run_inference_once()
                    
                    # 2. 获取当前价格（从最新 K 线的收盘价）
                    latest_df = self.fetcher.fetch_recent_klines(
                        symbol=self.symbol,
                        interval=self.interval,
                        limit=1
                    )
                    
                    if latest_df is not None and not latest_df.empty:
                        current_price = float(latest_df['close'].iloc[-1])
                        
                        # 3. 头寸管理
                        action = self.manage_position(current_price, signal)
                        
                        logger.info(f"Current Price: {current_price:.2f}, Signal: {signal}, Confidence: {confidence:.4f}, Action: {action}")
                    else:
                        logger.warning("Failed to fetch current price")
                    
                except Exception as e:
                    logger.error(f"Error in trading cycle: {e}", exc_info=True)
                
                # 等待下一个检查周期
                logger.info(f"Sleeping for {check_interval_sec} seconds...")
                time.sleep(check_interval_sec)
        
        except KeyboardInterrupt:
            logger.info("Live trading stopped by user")
            self.print_trade_summary()
    
    def print_trade_summary(self):
        """打印交易总结"""
        if not self.trade_log:
            logger.info("No trades executed")
            return
        
        total_pnl = sum([t.get('pnl', 0) for t in self.trade_log if 'pnl' in t])
        logger.info(f"=== Trade Summary ===")
        logger.info(f"Total Trades: {len(self.trade_log)}")
        logger.info(f"Total PnL: {total_pnl:.4f}")
        for trade in self.trade_log:
            logger.info(f"  {trade}")


# ==========================================
# 主程序入口
# ==========================================
if __name__ == "__main__":
    # 创建策略实例
    strategy = MLStrategy(
        symbol="BTCUSDT",
        interval="15m",
        model_type="lightgbm",  # 可以改为 "xgboost"
        confidence_threshold=0.65,  # 预测概率 > 65% 时才交易
        training_lookback=7,  # 用过去 7 天数据训练
        stop_loss_pct=0.02,  # 2% 止损
        take_profit_pct=0.05,  # 5% 止盈
    )
    
    # 选项 1: 训练模型（仅执行一次）
    print("\n=== Training Phase ===")
    strategy.train(force_retrain=False)  # 如果模型存在则加载，不存在则训练
    
    # 选项 2: 运行实时交易（持续运行）
    print("\n=== Live Trading Phase ===")
    strategy.run_live_trading(check_interval_sec=120)  # 每 2 分钟检查一次
