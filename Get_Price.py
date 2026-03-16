import requests

def get_current_price(symbol):
    url = "https://api.binance.com/api/v3/ticker/price"
    params = {'symbol': symbol}
    
    response = requests.get(url, params=params)
    data = response.json()
    
    # Check if the API returned an error
    if 'price' in data:
        return float(data['price'])
    else:
        print(f"Error: {data}")
        return None

# --- USE IT ---
