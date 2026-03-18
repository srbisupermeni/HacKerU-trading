"""
=============================================================================
Binance 数据获取工具 (Binance_fetcher.py)
从 Binance 获取历史K线(OHLCV)数据，并自动保存为CSV文件，用于量化策略回测。

【依赖库安装】
请确保安装了 pandas  requests python-binance：


【辅助方法】show_data()
提供一个简单的工具方法，用于在控制台展示 DataFrame 的前几行和后几行数据，方便调试和验证数据是否正确获取。
"""


"""
=============================================================================
Binance 综合数据获取工具 (Data_fetcher.py)
结合了 Binance Vision (历史批量数据) 和 python-binance SDK (实时最新数据)。
适用于量化策略回测与实盘信号生成。

【功能模块】
获取最新/实时的热数据 (通过 Python SDK)
from Binance_fetcher import BinanceDataFetcher
fetcher = BinanceDataFetcher()
recent_df = fetcher.fetch_recent_klines(symbol="BTCUSDT", interval="1m", limit=10)  #获取最新的10条1分钟K线数据
fetcher.show_data(recent_df)

=============================================================================
"""

import requests
import numpy as np
import pandas as pd
import os
import zipfile
import io
from datetime import date, timedelta
from binance.client import Client 

class BinanceDataFetcher:
    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        current_path = base_dir

        while True:
            if os.path.exists(os.path.join(current_path, "requirements.txt")):
                root_dir = current_path
                break
            parent_path = os.path.dirname(current_path)
            if parent_path == current_path:
                root_dir = base_dir
                break
            current_path = parent_path

        self.raw_save_folder = os.path.join(root_dir, "database", "raw_data")
        self.processed_save_folder = os.path.join(root_dir, "database", "processed_data")
        
        # 初始化 Binance SDK Client (获取公开市场数据不需要 API Key)
        self.client = Client()

    def show_data(self, df, num_rows=5):
        if df is not None and not df.empty:
            print(f" {num_rows} of head:")
            print(df.head(num_rows))
            print(f"{num_rows} of tail:")
            print(df.tail(num_rows))
        else:
            print("empty df or None")

    def _add_technical_indicators(self, df):
        df['return_rate'] = df['close'].pct_change()
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        df['volatility_20'] = df['return_rate'].rolling(window=20).std()
        df['momentum_10'] = df['close'] - df['close'].shift(10)
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
        
        std_20 = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['sma_20'] + 2 * std_20
        df['bb_lower'] = df['sma_20'] - 2 * std_20
        
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        return df

    # ==========================================
    # 模块一：获取最新/实时的热数据 (通过 Python SDK)
    # ==========================================
    def fetch_recent_klines(self, symbol="BTCUSDT", interval="1m", limit=100):
        """
        获取当前时刻往前推的最新 K 线数据 (包含 buy_volume 和 sell_volume)
        :param limit: 获取的数据条数（最高 1000 条）
        """
        print(f"Fetching recent {limit} candles for {symbol} ({interval}) via SDK...")
        
        # 使用 SDK 获取数据
        raw_klines = self.client.get_klines(symbol=symbol.upper(), interval=interval, limit=limit)
        
        if not raw_klines:
            return None

        # SDK 返回的列表格式与 REST API 一致
        columns = [
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ]
        df = pd.DataFrame(raw_klines, columns=columns)

        # 统一时间格式（SDK 返回的是毫秒）
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        # UTC 转换为北京时间 (UTC+8)
        df['open_time'] = df['open_time'] + pd.Timedelta(hours=8)
        df['close_time'] = df['close_time'] + pd.Timedelta(hours=8)

        # 转换数值类型
        numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'taker_buy_base_asset_volume']
        for col in numeric_columns:
            df[col] = df[col].astype(float)

        # 计算买卖量
        df['buy_volume'] = df['taker_buy_base_asset_volume']
        df['sell_volume'] = df['volume'] - df['taker_buy_base_asset_volume']

        # 截取最终的 8 列
        df = df[['open_time', 'open', 'high', 'low', 'close', 'volume', 'buy_volume', 'sell_volume']]
        return df




if __name__ == "__main__":
    fetcher = BinanceDataFetcher()
    
    print("=== 测试获取当前时刻最新的 10 根 1 分钟线 (热数据) ===")
    recent_df = fetcher.fetch_recent_klines(symbol="BTCUSDT", interval="15m", limit=100)
    fetcher.show_data(recent_df, num_rows=3)

    
    print("\n=== 添加技术指标测试 ===")
    if recent_df is not None:
        recent_with_indicators = fetcher._add_technical_indicators(recent_df)
        print([col for col in recent_with_indicators.columns])