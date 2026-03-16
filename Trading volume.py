import requests
import pandas as pd
import time

def get_15m_cvd(symbol):
    # Binance returns up to 1000 trades per request
    # To get 15 minutes of data, we use aggTrades
    # startTime = now - 15 minutes
    start_time = int((time.time() - 15 * 60) * 1000)
    
    url = "https://api.binance.com/api/v3/aggTrades"
    params = {
        'symbol': symbol,
        'startTime': start_time,
        'limit': 1000  # Adjust if 15 mins has more than 1000 trades
    }
    
    response = requests.get(url, params=params)
    trades = response.json()
    
    df = pd.DataFrame(trades)
    
    # 'm' is isBuyerMaker (True = Sell, False = Buy)
    # If isBuyerMaker is True, it's a SELL trade
    # If isBuyerMaker is False, it's a BUY trade
    
    df['q'] = df['q'].astype(float) # Quantity
    
    buy_vol = df[df['m'] == False]['q'].sum()
    sell_vol = df[df['m'] == True]['q'].sum()
    
    return buy_vol, sell_vol

# --- RUN IT ---
buy, sell = get_15m_cvd("AVAXUSDT")
print(f"15m Buy Vol: {buy:.2f} | 15m Sell Vol: {sell:.2f}")
print(f"CVD (Net Flow): {(buy - sell):.2f}")