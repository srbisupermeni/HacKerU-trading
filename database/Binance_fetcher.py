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

import numpy as np
import pandas as pd
import os
import logging
from datetime import datetime
from binance.client import Client 


# Helper to obtain a logger but avoid raising OSError if the logging system's handlers
# are misconfigured or the filesystem is not writable. Falls back to a simple printer.
def _get_safe_logger(name):
    try:
        return logging.getLogger(name)
    except OSError:
        class _DummyLogger:
            def info(self, msg, *args, **kwargs):
                try:
                    print(str(msg))
                except Exception:
                    pass

            def warning(self, msg, *args, **kwargs):
                try:
                    print("WARNING:", str(msg))
                except Exception:
                    pass

            def exception(self, msg, *args, **kwargs):
                try:
                    print("EXCEPTION:", str(msg))
                except Exception:
                    pass

        return _DummyLogger()

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
        logger = _get_safe_logger(__name__)
        if df is not None and not df.empty:
            logger.info(f"{num_rows} of head:\n{df.head(num_rows)}")
            logger.info(f"{num_rows} of tail:\n{df.tail(num_rows)}")
        else:
            logger.info("empty df or None")

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
        :param limit: 获取的数据条数（最高 2000 条）
        """
        logger = _get_safe_logger(__name__)
        logger.info(f"Fetching recent {limit} candles for {symbol} ({interval}) via SDK...")

        try:
            # 使用 SDK 获取数据
            raw_klines = self.client.get_klines(symbol=symbol.upper(), interval=interval, limit=limit)
        except Exception as e:
            logger.exception("Exception while calling Binance SDK get_klines")
            return None

        if not raw_klines:
            logger.warning("No klines returned from SDK")
            return None

        # 支持 SDK 返回 list-of-lists (经典格式) 或 list-of-dicts (兼容其他客户端/REST库)
        columns = [
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ]

        try:
            first = raw_klines[0]
        except Exception:
            logger.exception("Unexpected klines format (not indexable)")
            return None

        try:
            if isinstance(first, dict):
                # map common camelCase keys to our expected snake_case
                df = pd.DataFrame(raw_klines)
                mapping = {
                    'openTime': 'open_time',
                    'open': 'open',
                    'high': 'high',
                    'low': 'low',
                    'close': 'close',
                    'volume': 'volume',
                    'closeTime': 'close_time',
                    'quoteAssetVolume': 'quote_asset_volume',
                    'numberOfTrades': 'number_of_trades',
                    'takerBuyBaseAssetVolume': 'taker_buy_base_asset_volume',
                    'takerBuyQuoteAssetVolume': 'taker_buy_quote_asset_volume'
                }
                df.rename(columns=mapping, inplace=True)
            else:
                df = pd.DataFrame(raw_klines, columns=columns)

            # 统一时间格式（SDK 返回的是毫秒）
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
            df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
            # UTC 转换为北京时间 (UTC+8)
            df['open_time'] = df['open_time'] + pd.Timedelta(hours=8)
            df['close_time'] = df['close_time'] + pd.Timedelta(hours=8)

            # 转换数值类型（更安全的转换方式）
            numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'taker_buy_base_asset_volume']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                else:
                    df[col] = np.nan

            # 计算买卖量（若缺失买量则置为 NaN）
            df['buy_volume'] = df.get('taker_buy_base_asset_volume')
            df['sell_volume'] = df['volume'] - df['buy_volume']

            # 截取最终的 8 列（确保列存在）
            desired = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'buy_volume', 'sell_volume']
            for c in desired:
                if c not in df.columns:
                    df[c] = np.nan

            df = df[desired]
            return df
        except Exception:
            logger.exception("Failed to parse klines into DataFrame")
            return None




if __name__ == "__main__":
    fetcher = BinanceDataFetcher()
    
    print("=== 测试获取当前时刻最新的 10 根 1 分钟线 (热数据) ===")
    recent_df = fetcher.fetch_recent_klines(symbol="BTCUSDT", interval="15m", limit=100)
    fetcher.show_data(recent_df, num_rows=3)

    
    print("\n=== 添加技术指标测试 ===")
    if recent_df is not None:
        recent_with_indicators = fetcher._add_technical_indicators(recent_df)
        print([col for col in recent_with_indicators.columns])