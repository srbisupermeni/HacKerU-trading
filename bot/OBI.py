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
import logging

def calculate_rsi(df, period=14):
    # Calculate price changes based on the 'close' column
    delta = df['close'].diff()
    
    # Separate gains and losses
    gain = (delta.where(delta > 0, 0))
    loss = (-delta.where(delta < 0, 0))
    
    # Calculate Exponential Moving Averages (Wilder's Smoothing)
    # Using 'com=period - 1' is the standard way to match RSI formulas
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    
    # Calculate RS and RSI
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

logging.basicConfig(level=logging.INFO, filename='trading_bot.log', format='%(asctime)s - %(message)s')

logging.info("Bot started...")


Holding = False

while True:
    try:
        fetcher = BinanceDataFetcher()
        recent_df = fetcher.fetch_recent_klines(symbol="BTCUSDT", interval="1m", limit=7)  #获取最新的10条1分钟K线数据

        mean_Buy = recent_df['buy_volume'].mean()
        mean_Sell = recent_df['sell_volume'].mean()




        OBI = (mean_Buy - mean_Sell)/(mean_Buy + mean_Sell)

        recent_df['RSI'] = calculate_rsi(recent_df, period=7)

        current_rsi = recent_df['RSI'].iloc[-1]

        if  Holding:
            if (OBI>0.75) and (current_rsi<60) :
                    signal = 'Buy'
                    print(signal)
                    Holding = True
            else:
                print('sleep')

        else:
            if (OBI<-0.2) or (current_rsi>=70):
                signal = 'Sell'
                print(signal)
                print(OBI)
                print(current_rsi)
                Holding = False
            else:
                print('sleep')
        logging.info("Strategy cycle completed successfully.")
    except Exception as e:
            # If the bot hits an error (e.g., API timeout), log it and wait
        logging.error(f"Error occurred: {e}")
        time.sleep(10) # Wait a bit before retrying the loop
        continue

        # 4. SLEEP 2 MINUTES
        # Note: If your strategy takes 10 seconds to run, 
        # sleep for 110 seconds to keep the cycle close to 2 mins.
    time.sleep(120)