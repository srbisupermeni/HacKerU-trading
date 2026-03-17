import requests
import os
import sys

root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root_dir)

from database.Binance_Vision_fetcher import VisionFetcher

def get_current_price(symbol):
    fetcher = VisionFetcher()
    
    
    response = requests.get(url, params=params)
    data = response.json()
    
    # Check if the API returned an error
    if 'price' in data:
        return float(data['price'])
    else:
        print(f"Error: {data}")
        return None

# --- USE IT ---
