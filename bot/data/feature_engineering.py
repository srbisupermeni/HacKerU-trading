"""
=============================================================================
特征工程管道 (data/feature_engineering.py)
严格基于 Close (价格) 和 Volume (成交量) 生成机器学习特征。
=============================================================================
【模块简介】
本模块负责将原始的 OHLCV (开高低收+成交量) 及主动买卖量 (Taker Volume) 数据，
转换为可以直接用于机器学习 (如 LightGBM, XGBoost, LSTM 等) 训练和实盘推理的高级特征集。
核心优势在于引入了基于订单流 (Order Flow) 和微观市场结构的不平衡度特征。

【外部导入与使用方法】
在你的回测脚本或实盘主循环中，按照以下步骤使用：

1. 导入模块:
    import sys
    from pathlib import Path
    # 确保根目录在 sys.path 中
    ROOT = Path(__file__).resolve().parent.parent # 根据你的执行文件位置调整
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))
    
    from data.feature_engineering import FeatureEngineer
    from database.Binance_fetcher import BinanceDataFetcher

2. 实例化并获取数据:
    fetcher = BinanceDataFetcher()
    # 必须包含: open, high, low, close, volume, buy_volume, sell_volume
    raw_df = fetcher.fetch_recent_klines(symbol="BTCUSDT", interval="15m", limit=100)

3. 生成特征:
    engineer = FeatureEngineer()
    features_df = engineer.generate_features(raw_df)
    
    # 此时 features_df 可直接切分为 X_train, y_train，或在实盘中取最后一行输入模型预测。

【特征数据字典与数学公式】
(注: 下标 t 代表当前周期，t-1 代表上一周期，t+n 代表未来第 n 个周期)

--- 1. 基础价格与形态特征 (Price & Candlestick) ---
* log_return: 对数收益率，消除绝对价格差异，使时间序列更平稳。
    公式: $\ln(Close_t / Close_{t-1})$
* body_size: K线实体比例，相对于开盘价的涨跌幅度。
    公式: $(Close_t - Open_t) / Open_t$
* upper_shadow: 上影线比例，反映上方抛压。
    公式: $(High_t - \max(Open_t, Close_t)) / Open_t$
* lower_shadow: 下影线比例，反映下方支撑。
    公式: $(\min(Open_t, Close_t) - Low_t) / Open_t$
* typical_price: 典型价格，常用于机构 VWAP 锚定。
    公式: $(High_t + Low_t + Close_t) / 3$

--- 2. 动量与趋势特征 (Momentum) ---
* roc_{window}: 变化率 (Rate of Change)，计算 3, 6, 12, 24 个周期的动量。
    公式: $(Close_t - Close_{t-window}) / Close_{t-window}$
* price_to_ema12: 价格偏离 12 周期指数移动平均线的比例 (乖离率)。
    公式: $(Close_t / EMA_{12}) - 1$

--- 3. 波动率特征 (Volatility) ---
* true_range: 归一化真实波幅，反映包含跳空缺口在内的真实市场波动。
    公式: $\max(High_t - Low_t, |High_t - Close_{t-1}|, |Low_t - Close_{t-1}|) / Close_{t-1}$
* atr_14: 14 周期平均真实波幅。
    公式: $\frac{1}{14}\sum_{i=0}^{13} true\_range_{t-i}$

--- 4. 订单流与微观结构特征 (Order Flow / Microstructure) ---
(注: $\epsilon$ 为极小值 1e-8，防止除以 0)
* buy_sell_ratio: 主动买卖比，衡量吃单资金的力量对比。
    公式: $BuyVolume_t / (SellVolume_t + \epsilon)$
* volume_imbalance: 订单流不平衡度，取值 [-1, 1]。1 为绝对买盘，-1 为绝对卖盘。
    公式: $(BuyVolume_t - SellVolume_t) / (Volume_t + \epsilon)$
* net_buy_vol_{window}: 窗口期内 (6, 12) 的累计净主动买入量 (CVD 变形)。
    公式: $\sum_{i=0}^{window-1} (BuyVolume_{t-i} - SellVolume_{t-i})$
* vol_imbalance_{window}: 窗口期内的订单流不平衡度平滑值。
    公式: $net\_buy\_vol\_window / (\sum_{i=0}^{window-1} Volume_{t-i} + \epsilon)$
* volume_intensity: 相对成交量强度，捕捉放量突变。
    公式: $Volume_t / (SMA(Volume, 24)_t + \epsilon)$

--- 5. 预测目标 (Target Variable) ---
* target_return_6: 未来 6 个周期的前瞻性对数收益率 (回归目标)。
    公式: $\ln(Close_{t+6} / Close_t)$
* target_class: 二分类预测目标 (分类目标)。未来 6 个周期价格是否高于当前。
    条件: 如果 $target\_return\_6 > 0$ 则为 1，否则为 0。

=============================================================================
"""


import sys
import pandas as pd
import numpy as np
from pathlib import Path

# 使用 pathlib 确保路径引用的鲁棒性，适配根目录的变动
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from database.Binance_Vision_fetcher import VisionFetcher
from database.Binance_fetcher import BinanceDataFetcher

class FeatureEngineer:
    def __init__(self):
        pass

    def generate_features(self, df):
        """
        核心特征生成函数
        输入: 包含完整K线和买卖量的 DataFrame (需包含 open, high, low, close, volume, buy_volume, sell_volume)
        输出: 包含衍生特征和预测目标的 DataFrame
        """
        df = df.copy()
        if 'open_time' in df.columns:
            df = df.sort_values('open_time').reset_index(drop=True)

        # 确保不会出现除以 0 的情况
        epsilon = 1e-8

        # ==========================================
        # 1. 基础价格与形态特征 (Price & Candlestick)
        # ==========================================
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        
        # K线实体与影线比例 (归一化到开盘价)
        df['body_size'] = (df['close'] - df['open']) / df['open']
        df['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['open']
        df['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['open']

        # 典型价格 (Typical Price) - 常用于机构的 VWAP 计算
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3

        # ==========================================
        # 2. 动量与趋势特征 (Momentum)
        # ==========================================
        for window in [3, 6, 12, 24]:
            df[f'roc_{window}'] = df['close'].pct_change(periods=window)
            
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        df['price_to_ema12'] = df['close'] / ema_12 - 1

        # ==========================================
        # 3. 波动率特征 (Volatility)
        # ==========================================
        # 真实波幅 (True Range) 及其移动平均 (ATR)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        df['true_range'] = np.max(ranges, axis=1) / df['close'].shift() # 归一化
        df['atr_14'] = df['true_range'].rolling(window=14).mean()

        # ==========================================
        # 4. 订单流与买卖量特征 (Order Flow / Microstructure) - 核心优势
        # ==========================================
        # 主动买单与卖单的比例 (Buy/Sell Ratio)
        df['buy_sell_ratio'] = df['buy_volume'] / (df['sell_volume'] + epsilon)
        
        # 订单流不平衡度 (Order Flow Imbalance) -> 取值范围 [-1, 1]
        # 接近 1 表示该周期全是主动买盘，-1 表示全是主动卖盘
        df['volume_imbalance'] = (df['buy_volume'] - df['sell_volume']) / (df['volume'] + epsilon)

        # 滚动累计量价背离 (Cumulative Volume Delta - CVD 变形)
        # 如果价格没怎么涨，但主动买盘极大，说明上方有隐形挂单压制；反之亦然。
        for window in [6, 12]:
            df[f'net_buy_vol_{window}'] = (df['buy_volume'] - df['sell_volume']).rolling(window=window).sum()
            df[f'vol_imbalance_{window}'] = df[f'net_buy_vol_{window}'] / (df['volume'].rolling(window=window).sum() + epsilon)

        # 成交量强度 (Volume Intensity)
        vol_sma_24 = df['volume'].rolling(window=24).mean()
        df['volume_intensity'] = df['volume'] / (vol_sma_24 + epsilon) 

        # 5. 生成预测目标 (Target Variable)
        # 预测未来 target_window 个周期的对数收益率
        target_window = 6 
        df[f'target_return_{target_window}'] = np.log(df['close'].shift(-target_window) / df['close'])

        # 二分类目标：未来是否上涨？(1 为上涨，0 为下跌)
        df['target_class'] = (df[f'target_return_{target_window}'] > 0).astype(int)
        
        # 将最后几行的 target 设为 NaN (因为未来数据未知)
        df.loc[df.index[-target_window:], [f'target_return_{target_window}', 'target_class']] = np.nan

        # 剔除由于 rolling 和 shift 产生的 NaN 值 (仅清理特征计算初期的 NaN) 保留最后 target_window 行用于实盘实时推断
        features_to_check = [col for col in df.columns if 'target' not in col]
        df = df.dropna(subset=features_to_check)

        return df

# ==========================================
# demo
# ==========================================
if __name__ == "__main__":
    print("Testing feature engineering...")
    
    # 使用 SDK 获取最新的 100 根 15分钟线
    fetcher = BinanceDataFetcher()
    raw_df = fetcher.fetch_recent_klines(symbol="BTCUSDT", interval="15m", limit=100)
    
    if raw_df is not None:
        engineer = FeatureEngineer()
        features_df = engineer.generate_features(raw_df)
        
        print(f"\n featured: {len(raw_df)}, features rows: {len(features_df)}")
        
        cols_to_show = ['open_time', 'close', 'volume_imbalance', 'buy_sell_ratio', 'vol_imbalance_6', 'target_class']
        print(features_df[cols_to_show].head())
        
        print(features_df[cols_to_show].tail(1))