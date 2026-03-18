"""
=============================================================================
特征工程管道 (data/feature_engineering.py)
严格基于 Close (价格) 和 Volume (总成交量) 生成机器学习特征。
确保 Binance 历史数据和 CoinGecko 实时数据的特征空间完全一致。
=============================================================================
"""

import os
import sys
import pandas as pd
import numpy as np

# 将根目录加入系统路径，以便导入 database 下的 VisionFetcher
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(parent_dir)


if root_dir not in sys.path:
    sys.path.append(root_dir)

from database.Binance_Vision_fetcher import VisionFetcher

class FeatureEngineer:
    def __init__(self):
        pass

    def generate_features(self, df):
        """
        核心特征生成函数
        input: 包含 'close' 和 'volume' 列的 DataFrame (按时间正序排列)
        output: 包含衍生特征和预测目标的 DataFrame
        """
        # 确保数据没有空值，且按时间排序
        df = df.copy()
        if 'open_time' in df.columns:
            df = df.sort_values('open_time').reset_index(drop=True)

        # 1. 基础价格转换 (Price Transformations)
        # 对数收益率 (Log Return) - 消除绝对价格差异，使数据更平稳
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        # Momentum 
        for window in [3, 6, 12, 24]:
            df[f'roc_{window}'] = df['close'].pct_change(periods=window)

        
        # 2. 波动率特征 (Volatility Features)
        # 历史波动率 (基于对数收益率的标准差)
        for window in [6, 12, 24]:
            df[f'volatility_{window}'] = df['log_return'].rolling(window=window).std()

        # 3. 相对位置与均线特征 (Moving Averages & Oscillators)
        # 价格相对于 EMA 的偏离度 (Price to EMA ratio)
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df['price_to_ema12'] = df['close'] / ema_12 - 1
        
        # 传统 MACD (仅需 Close)
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # 简化版 RSI (仅依赖 Close 的涨跌)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi_14'] = 100 - (100 / (1 + rs))

        # 4. 成交量特征 (Volume Features)
        # 成交量变化率
        df['vol_change'] = df['volume'].pct_change()
        
        # 相对成交量强度 (Relative Volume) - 当前成交量与过去24个周期平均成交量的比值
        vol_sma_24 = df['volume'].rolling(window=24).mean()
        df['volume_intensity'] = df['volume'] / (vol_sma_24 + 1e-8) #加 1e-8 防止除以 0

        # 价格-成交量趋势 (Price-Volume Trend)
        # 当价格上涨且成交量放大时为正号，反之为负
        df['pvt'] = (df['log_return'] * df['volume_intensity']).rolling(window=6).sum()

        # 5. Statistical Features
        # 收益率的偏度 (Skewness) 和 峰度 (Kurtosis) 捕捉极端行情的尾部风险
        df['skew_12'] = df['log_return'].rolling(window=12).skew()
        df['kurt_12'] = df['log_return'].rolling(window=12).kurt()

        # 6. 生成预测目标 (Target Variable for ML)
        # 我们预测未来 N 个周期的对数收益率 (例如 N=6) 负号 shift 代表将未来的价格拉到现在同一行
        target_window = 6
        df[f'target_return_{target_window}'] = np.log(df['close'].shift(-target_window) / df['close'])

        # 二分类目标：未来 N 个周期是否上涨？(1 为上涨，0 为下跌)
        df['target_class'] = (df[f'target_return_{target_window}'] > 0).astype(int)
        
        # 将最后 target_window 行的 target 设为 NaN (因为未来数据未知)
        df.loc[df.index[-target_window:], [f'target_return_{target_window}', 'target_class']] = np.nan

        # 剔除由于 rolling 和 shift 产生的 NaN 值 (仅清理特征部分的 NaN)
        df = df.dropna(subset=[col for col in df.columns if 'target' not in col])

        return df

# ==========================================
# 独立运行测试代码
# ==========================================
if __name__ == "__main__":
    print("--- 正在通过 VisionFetcher 获取数据以测试特征工程 ---")
    fetcher = VisionFetcher()
    
    # 获取 2024年1月 的 BTCUSDT 15分钟线数据作为样本
    raw_df = fetcher.fetch_klines_from_vision(
        symbol="BTCUSDT", interval="15m", 
        year=2024, month=1, data_type="monthly"
    )
    
    if raw_df is not None:
        engineer = FeatureEngineer()
        features_df = engineer.generate_features(raw_df)
        
        print(f"\n raw_df   features: {len(raw_df)}, features rows: {len(features_df)}")
        print(list(features_df.columns))
        
        print("\nhead:")
        print(features_df[['open_time', 'close', 'volume', 'log_return', 'rsi_14', 'volume_intensity', 'target_class']].head())