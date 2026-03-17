"""
=============================================================================
Binance 数据获取工具 (Binance_fetcher.py)
从 Binance 获取历史K线(OHLCV)数据，并自动保存为CSV文件，用于量化策略回测。

【依赖库安装】
请确保安装了 pandas 和 requests：

【使用方法 1：直接运行】
直接运行此脚本，会自动在当前目录下创建 `raw_data` 文件夹，并下载默认的 BTC 和 ETH 数据。raw_data使.ignore 文件已配置为忽略该文件夹中的内容，确保不会被版本控制系统跟踪。

【使用方法 2：在其他代码中复用】
from Binance_fetcher import BinanceFetcher，可以使用这个类

# 1. 初始化实例
fetcher = BinanceFetcher()

# 2. 一键获取并保存数据 (例如获取 BNB 过去 500 个 1小时级别 的K线)
# 这将在 raw_data 文件夹下生成 BNBUSDT_1h.csv, 
file_path = fetcher.get_and_save_data(symbol="BNBUSDT", interval="1h", limit=500)

# 3. 或者仅仅获取数据作为 Pandas DataFrame，不保存为文件 (直接喂给策略代码)
df = fetcher.fetch_klines_df(symbol="SOLUSDT", interval="15m", limit=100)
=============================================================================

【辅助方法】show_data()
提供一个简单的工具方法，用于在控制台展示 DataFrame 的前几行和后几行数据，方便调试和验证数据是否正确获取。
"""


import requests
import numpy as np
import pandas as pd
import os
from datetime import datetime
import zipfile
import io

class BinanceFetcher:
    def __init__(self):
        # 币安官方的公共 API 基础 URL
        self.base_url = "https://api3.binance.com"

        base_dir = os.path.dirname(os.path.abspath(__file__))
        current_path = base_dir

        while True:
            # 检查当前目录是否包含根目录的标识文件（比如 requirements.txt）
            if os.path.exists(os.path.join(current_path, "requirements.txt")):
                root_dir = current_path
                break
            # 拿到上一级目录
            parent_path = os.path.dirname(current_path)
            current_path = parent_path

        self.raw_save_folder = os.path.join(root_dir, "database", "raw_data")
        self.processed_save_folder = os.path.join(root_dir, "database", "processed_data")
    

    
    

    def _send_public_request(self, endpoint, params):
        """发送公共 GET 请求的底层逻辑"""
        url = f"{self.base_url}{endpoint}"
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"fail fetch from binance: {e}")
            if response is not None:
                print(f"server response: {response.text}")
            return None
        
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
            - 收益率 (Return Rate) : df['return_rate'] 
            - 对数收益率 (Log Return):  df['log_return']
            - 波动率 (Volatility): df['volatility_20']
            - 动量 (Momentum): df['momentum_10']
            - 简单移动平均线 (SMA) 和 指数移动平均线 (EMA): df['sma_20'], df['ema_20']
            - 布林带 (Bollinger Bands): df['bb_upper'], df['bb_lower']
            - 相对强弱指数 (RSI): df['rsi_14']
            - MACD: df['macd'], df['macd_signal']
        
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

    def fetch_klines_df(self, symbol="BTCUSDT", interval="1h", limit=1000):
        """
        获取K线数据并直接转换为格式化好的 Pandas DataFrame
        :param symbol: 交易对，如 "BTCUSDT", "ETHUSDT"
        :param interval: K线级别，如 "1m", "5m", "15m", "1h", "4h", "1d"
        :param limit: 获取的数据条数，最大 1000
        :return: pd.DataFrame 或 None
        """
        endpoint = "/api/v3/klines"
        params = {
            "symbol": symbol.upper(),
            "interval": interval,
            "limit": limit
        }
        
        print(f"fetching {symbol} ({interval}) data...")
        raw_data = self._send_public_request(endpoint, params)
        
        if not raw_data:
            return None

        # 币安 API 返回的是一个嵌套列表，将其映射为有意义的列名
        columns = [
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ]
        
        df = pd.DataFrame(raw_data, columns=columns)
        
        # cleaning
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            df[col] = df[col].astype(float)
            
        # 保留OHLCV
        df = df[['open_time', 'open', 'high', 'low', 'close', 'volume']]

        
        return df

    def get_and_save_data(self, symbol="BTCUSDT", interval="1h", limit=1000):
        """
        核心方法：获取数据并自动保存为 CSV 到 raw_data 文件夹
        """
        df = self.fetch_klines_df(symbol, interval, limit)
        
        if df is None or df.empty:
            print(f"failed to fetch data for {symbol}, skipping save.")
            return None

        # 确保 raw_data 文件夹存在
        os.makedirs(self.raw_save_folder, exist_ok=True)
        
        # 构建文件路径 (例如: raw_data/BTCUSDT_1h.csv)
        file_name = f"{symbol.upper()}_{interval}.csv"
        file_path = os.path.join(self.raw_save_folder, file_name)
        
        # 将 DataFrame 保存为 CSV 文件
        df.to_csv(file_path, index=False)
        print(f"successfully saved: {file_path} ( {len(df)} rows )\n")

        """
        额外功能：在保存原始数据的同时，也可以选择将添加了技术指标的版本保存到 processed_data 文件夹，方便后续直接使用。"""
        df=self._add_technical_indicators(df)

        os.makedirs(self.processed_save_folder, exist_ok=True)
        file_name = f"{symbol.upper()}_{interval}_with_indicators.csv"
        file_path = os.path.join(self.processed_save_folder, file_name)
        df.to_csv(file_path, index=False)
        print(f"successfully saved: {file_path} ( {len(df)} rows )\n")

        return file_path
    
    def fetch_klines_from_vision(self, symbol="BTCUSDT", interval="1h", year=2025, month=1, data_type="monthly"):
        # """
        # 从 data.binance.vision 下载历史K线数据（不受地区限制）
        # :param symbol: 交易对，如 "BTCUSDT"
        # :param interval: K线级别，如 "1h", "15m", "1d"
        # :param year: 年份
        # :param month: 月份（data_type="daily" 时需要额外传 day 参数）
        # :param data_type: "monthly" 或 "daily"
        # :return: pd.DataFrame 或 None
        # """
    

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
        import calendar
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


# ==========================================
# 独立运行测试代码 (当直接运行此脚本时执行)
# ==========================================
if __name__ == "__main__":
    fetcher = BinanceFetcher()
    
    print("testing fetcher\n")
    
    # 测试 1: 获取比特币 1小时级别 的最新 500 条数据
    fetcher.get_and_save_data(symbol="BTCUSDT", interval="1h", limit=500)
    
    # 测试 2: 获取以太坊 15分钟级别 的最新 1000 条数据
    fetcher.get_and_save_data(symbol="ETHUSDT", interval="15m", limit=1000)
    
    # 测试 3: 检查文件夹内容
    print(f" '{fetcher.raw_save_folder}' as folder:")
    for file in os.listdir(fetcher.raw_save_folder):
        print(f"  - {file}")
    for file in os.listdir(fetcher.processed_save_folder):
        print(f"  - {file}")

    df_eth = fetcher.fetch_klines_df(symbol="ETHUSDT", interval="15m", limit=100)
    fetcher.show_data(df_eth, num_rows=3)