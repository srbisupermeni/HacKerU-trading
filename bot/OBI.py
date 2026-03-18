import sys
from pathlib import Path

# 1. Get the current script's directory (/bot)
current_dir = Path(__file__).resolve().parent

# 2. Go UP one level to the parent folder (/Project_Parent)
parent_dir = current_dir.parent

# 3. Add the 'database' folder to Python's search path
database_dir = parent_dir / "database"
sys.path.append(str(database_dir))

# 4. Now import your file
from Binance_fetcher import BinanceDataFetcher
import numpy as np
import pandas as pd

import time

Holding = False

fetcher = BinanceDataFetcher()
recent_df = fetcher.fetch_recent_klines(symbol="BTCUSDT", interval="1m", limit=7)  #获取最新的10条1分钟K线数据

mean_Buy = recent_df['buy_volume'].mean()
mean_Sell = recent_df['sell_volume'].mean()

OBI = (mean_Buy - mean_Sell)/(mean_Buy + mean_Sell)



if OBI>0.75 & current_rsi<60 :
    print('Buy')
else:
    print('oh no')