"""

Binance Vision 数据获取工具 (Vision_fetcher.py)
从 Binance Vision (data.binance.vision) 获取历史K线(OHLCV)数据，并自动保存为CSV文件。
不受地区限制，适用于获取大批量的历史月度数据，用于量化策略回测。

【依赖库安装】
请确保安装了 pandas 和 requests：
pip install pandas requests

【使用方法 1：直接运行】
直接运行此脚本，会自动在当前目录下（或往上寻找 requirements.txt 所在的根目录）
创建 `database/raw_data` 和 `database/processed_data` 文件夹，并下载测试数据。
raw data: ['open_time', 'open', 'high', 'low', 'close', 'volume'] 6columns
processed data: ['open_time', 'open', 'high', 'low', 'close', 'volume', 
'return_rate', 'log_return', 'volatility_20', 'momentum_10', 'sma_20', 'ema_20', 'bb_upper', 'bb_lower', 'rsi_14', 'macd', 'macd_signal'] 16columns

【使用方法 2：在其他代码中复用】
from Vision_fetcher import VisionFetcher

# 1. 初始化实例
fetcher = VisionFetcher()

# 2. 一键获取区间数据并保存 (例如获取 BTC 2024年1月到3月的 1小时线)
file_path = fetcher.get_and_save_range(
    symbol="BTCUSDT", interval="1h", 
    start_year=2024, start_month=1, 
    end_year=2024, end_month=3
)

# 3. 仅获取 Pandas DataFrame
df = fetcher.fetch_klines_range(
    symbol="ETHUSDT", interval="15m", 
    start_year=2024, start_month=1, end_year=2024, end_month=2
)
# 4. 获取单月数据
df = fetcher.fetch_klines_from_vision(
    symbol="ETHUSDT", interval="15m",
    year=2024, month=1
)

3,4 为raw_data, 包含6列: ['open_time', 'open', 'high', 'low', 'close', 'volume']

# 5. 添加技术指标   
df_with_indicators = fetcher._add_technical_indicators(df)
# 为processed_data, 包含16列: ['open_time', 'open', 'high', 'low', 'close', 'volume', 
# 'return_rate', 'log_return', 'volatility_20', 'momentum_  10', 'sma_20', 'ema_20', 'bb_upper', 'bb_lower', 'rsi_14', 'macd', 'macd_signal']

=============================================================================
"""

import requests
import numpy as np
import pandas as pd
import os
import zipfile
import io

class VisionFetcher:
    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        current_path = base_dir

        while True:
            # 检查当前目录是否包含根目录的标识文件（比如 requirements.txt）
            if os.path.exists(os.path.join(current_path, "requirements.txt")):
                root_dir = current_path
                break
            # 拿到上一级目录
            parent_path = os.path.dirname(current_path)
            # 如果已经到了系统的根目录依然没找到，就在当前脚本目录创建
            if parent_path == current_path:
                root_dir = base_dir
                break
            current_path = parent_path

        self.raw_save_folder = os.path.join(root_dir, "database", "raw_data")
        self.processed_save_folder = os.path.join(root_dir, "database", "processed_data")

    def show_data(self, df, num_rows=5):
        """辅助方法：展示 DataFrame 的前几行数据，方便调试和验证"""
        if df is not None and not df.empty:
            print(f" {num_rows} of head:")
            print(df.head(num_rows))
            print(f"{num_rows} of tail:")
            print(df.tail(num_rows))
        else:
            print("empty df or None")

    def _add_technical_indicators(self, df):
        """
        为原始 OHLCV 数据添加常用的量化交易特征
        """
        # 1. 收益率 (Return Rate)
        df['return_rate'] = df['close'].pct_change()
        
        # 2. 对数收益率 (Log Return) 
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        
        # 3. 波动率 (Volatility) - 过去20个周期的收益率标准差
        df['volatility_20'] = df['return_rate'].rolling(window=20).std()
        
        # 4. 动量 (Momentum) - 过去10个周期的绝对价格变化
        df['momentum_10'] = df['close'] - df['close'].shift(10)
        
        # 5. 简单移动平均线 (SMA) 和 指数移动平均线 (EMA)
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
        
        # 6. 布林带 (Bollinger Bands)
        std_20 = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['sma_20'] + 2 * std_20
        df['bb_lower'] = df['sma_20'] - 2 * std_20
        
        # 7. 相对强弱指数 (RSI_14)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # 8. MACD
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        return df

    def fetch_klines_from_vision(self, symbol="BTCUSDT", interval="1h", year=2025, month=1, data_type="monthly"):
        """
        从 data.binance.vision 下载历史K线数据（不受地区限制）
        """
        if data_type == "monthly":
            filename = f"{symbol.upper()}-{interval}-{year}-{month:02d}.zip"
            url = (f"https://data.binance.vision/data/spot/monthly/klines/"
                   f"{symbol.upper()}/{interval}/{filename}")
        else:
            raise ValueError("data_type 暂只支持 monthly")

        print(f"downloading from binance vision: {url}")
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"下载失败: {e}")
            return None

        # 解压 zip，读取 CSV
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            csv_filename = z.namelist()[0]  # zip 内只有一个 csv 文件
            with z.open(csv_filename) as f:
                df = pd.read_csv(f, header=None)

        # 和原来保持一致的列名
        df.columns = [
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ]

        # 时间处理（2025年后是微秒，之前是毫秒）
        if year >= 2025:
            df['open_time'] = pd.to_datetime(df['open_time'], unit='us')
            df['close_time'] = pd.to_datetime(df['close_time'], unit='us')
        else:
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
            df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')

        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            df[col] = df[col].astype(float)

        df = df[['open_time', 'open', 'high', 'low', 'close', 'volume']]
        print(f"成功获取 {len(df)} 条数据")
        return df

    def fetch_klines_range(self, symbol="BTCUSDT", interval="1h",
                           start_year=2025, start_month=1,
                           end_year=2025, end_month=3):
        """
        获取多个月份的数据并拼接
        """
        all_dfs = []
        
        year, month = start_year, start_month
        while (year, month) <= (end_year, end_month):
            df = self.fetch_klines_from_vision(symbol, interval, year, month)
            if df is not None:
                all_dfs.append(df)
            # 下一个月
            if month == 12:
                year += 1
                month = 1
            else:
                month += 1

        if not all_dfs:
            return None
        
        result = pd.concat(all_dfs, ignore_index=True)
        result = result.drop_duplicates(subset=['open_time']).sort_values('open_time')
        print(f"合并完成，共 {len(result)} 条数据")
        return result

    def get_and_save_range(self, symbol="BTCUSDT", interval="1h",
                           start_year=2025, start_month=1,
                           end_year=2025, end_month=3):
        """
        核心方法：获取多月拼接数据并自动保存为 CSV，同时生成带有技术指标的版本
        """
        df = self.fetch_klines_range(symbol, interval, start_year, start_month, end_year, end_month)
        
        if df is None or df.empty:
            print(f"failed to fetch data for {symbol}, skipping save.")
            return None

        # 确保 raw_data 文件夹存在
        os.makedirs(self.raw_save_folder, exist_ok=True)
        
        # 构建文件路径 (例如: raw_data/BTCUSDT_1h_202401_202403.csv)
        file_suffix = f"{start_year}{start_month:02d}_{end_year}{end_month:02d}"
        file_name = f"{symbol.upper()}_{interval}_{file_suffix}.csv"
        file_path = os.path.join(self.raw_save_folder, file_name)
        
        # 将 DataFrame 保存为原始 CSV 文件
        df.to_csv(file_path, index=False)
        print(f"successfully saved raw data: {file_path} ( {len(df)} rows )\n")

        # 额外功能：生成添加了技术指标的版本保存到 processed_data 文件夹
        df = self._add_technical_indicators(df)

        os.makedirs(self.processed_save_folder, exist_ok=True)
        processed_file_name = f"{symbol.upper()}_{interval}_{file_suffix}_with_indicators.csv"
        processed_file_path = os.path.join(self.processed_save_folder, processed_file_name)
        df.to_csv(processed_file_path, index=False)
        print(f"successfully saved processed data: {processed_file_path} ( {len(df)} rows )\n")

        return file_path


# ==========================================
# 独立运行测试代码 (当直接运行此脚本时执行)
# ==========================================
if __name__ == "__main__":
    fetcher = VisionFetcher()
    
    print("testing VisionFetcher\n")
    
    # 测试 1: 获取比特币 1小时级别 2024年1月-2024年2月 的数据
    print("Test 1: Fetching BTCUSDT 1h data (2024.01 - 2024.02)...")
    fetcher.get_and_save_range(
        symbol="BTCUSDT", interval="1h", 
        start_year=2024, start_month=1, 
        end_year=2024, end_month=2
    )
    
    # 测试 2: 检查文件夹内容
    print(f"\n'{fetcher.raw_save_folder}' as folder:")
    if os.path.exists(fetcher.raw_save_folder):
        for file in os.listdir(fetcher.raw_save_folder):
            print(f"  - {file}")
            
    print(f"\n'{fetcher.processed_save_folder}' as folder:")
    if os.path.exists(fetcher.processed_save_folder):
        for file in os.listdir(fetcher.processed_save_folder):
            print(f"  - {file}")

    # 测试 3: 展示获取的 DataFrame 数据
    print("\nTest 3: Showing top and bottom rows for fetched data...")
    df_btc = fetcher.fetch_klines_range(
        symbol="BTCUSDT", interval="1h", 
        start_year=2024, start_month=1, 
        end_year=2024, end_month=1
    )
    fetcher.show_data(df_btc, num_rows=3)
    print([col for col in df_btc.columns])

    df_btc_indicators = fetcher._add_technical_indicators(df_btc)
    fetcher.show_data(df_btc_indicators, num_rows=3)
    print([col for col in df_btc_indicators.columns])
