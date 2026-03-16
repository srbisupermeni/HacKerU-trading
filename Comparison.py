import requests
import pandas as pd
import time

def fetch_binance_klines(symbol, interval, start_time):
    base_url = "https://api.binance.com/api/v3/klines"
    start_ms = int(start_time.timestamp() * 1000)
    
    all_data = []
    current_start = start_ms
    
    print(f"Downloading {symbol}...")
    
    while True:
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': current_start,
            'limit': 1000
        }
        
        response = requests.get(base_url, params=params)
        
        # 1. CRITICAL FIX: Check if the request was successful
        if response.status_code != 200:
            print(f"Error: API returned {response.status_code}")
            print(response.json()) # This will show you exactly why it failed
            break
            
        data = response.json()
        
        # 2. FIX: Ensure data is a list and not empty
        if not isinstance(data, list) or len(data) == 0:
            print("Finished downloading or no more data.")
            break
            
        all_data.extend(data)
        
        # 3. Update pointer
        current_start = data[-1][0] + 60000
        
        # Be nice to the API
        time.sleep(0.2)
        
        # Stop if we hit the "current time" (the API returns < 1000 if it's the end)
        if len(data) < 1000:
            break
            
    # Convert to DataFrame
    df = pd.DataFrame(all_data, columns=[
        'Open_Time', 'Open', 'High', 'Low', 'Close', 'Volume', 
        'Close_Time', 'Quote_Asset_Volume', 'Number_of_Trades', 
        'Taker_Buy_Base_Vol', 'Taker_Buy_Quote_Vol', 'Ignore'
    ])
    
    # Clean up data types
    df['Close'] = pd.to_numeric(df['Close'])
    df['Open_Time'] = pd.to_datetime(df['Open_Time'], unit='ms')
    return df[['Open_Time', 'Close']].set_index('Open_Time')

# --- TEST ---
start_date = pd.Timestamp.utcnow() - pd.Timedelta(days=7)

# Ensure the symbol is correct. Binance uses uppercase without slash.
try:
    df_avax = fetch_binance_klines("UNIUSDT", "1m", start_date)
    print(df_avax.head())
    df_avax.to_csv("UNI_data.csv")
except Exception as e:
    print(f"Something went wrong: {e}")