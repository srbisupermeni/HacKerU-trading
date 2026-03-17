import requests
import time
import hmac
import hashlib

"""
Sample output from running this test script:
可用来暂时参考，稍后会重构为Class
--- Checking Server Time ---
{'ServerTime': 1773764301077}

--- Getting Exchange Info ---
Available Pairs: ['OPEN/USD', 'TRUMP/USD', 'TON/USD', 'S/USD', 'SOL/USD', 'OMNI/USD', 'CAKE/USD', 'ARB/USD', 'AVNT/USD', 'PAXG/USD', 'EDEN/USD', 'HEMI/USD', 'FET/USD', 'LINK/USD', 'FORM/USD', 'FLOKI/USD', 'BONK/USD', 'FIL/USD', 'BTC/USD', 'TAO/USD', 'UNI/USD', 'PEPE/USD', 'PUMP/USD', 'HBAR/USD', 'XRP/USD', 'AAVE/USD', 'WLFI/USD', 'EIGEN/USD', 'LINEA/USD', '1000CHEEMS/USD', 'BIO/USD', 'LISTA/USD', 'AVAX/USD', 'MIRA/USD', 'XLM/USD', 'SUI/USD', 'NEAR/USD', 'SEI/USD', 'PENGU/USD', 'ETH/USD', 'PENDLE/USD', 'PLUME/USD', 'WIF/USD', 'ICP/USD', 'BNB/USD', 'VIRTUAL/USD', 'APT/USD', 'SHIB/USD', 'POL/USD', 'ZEC/USD', 'DOGE/USD', 'CRV/USD', 'ASTER/USD', 'TRX/USD', 'BMT/USD', 'ZEN/USD', 'ONDO/USD', 'LTC/USD', 'STO/USD', 'SOMI/USD', 'WLD/USD', 'XPL/USD', 'CFX/USD', 'DOT/USD', 'TUT/USD', 'ADA/USD', 'ENA/USD']

--- Getting Market Ticker (BTC/USD) ---
{'MaxBid': 74137.86, 'MinAsk': 74137.87, 'LastPrice': 74137.87, 'Change': 0.0092, 'CoinTradeValue': 27340.12305, 'UnitTradeValue': 2034368085.2932057}

--- Getting Account Balance ---
{'Success': True, 'ErrMsg': '', 'SpotWallet': {'USD': {'Free': 50000, 'Lock': 0}}, 'MarginWallet': {}}

--- Checking Pending Orders ---
{'Success': False, 'ErrMsg': 'no pending order under this account', 'TotalPending': 0, 'OrderPairs': {}}
{'Success': True, 'ErrMsg': '', 'OrderDetail': {'Pair': 'BNB/USD', 'OrderID': 2761589, 'Status': 'FILLED', 'Role': 'TAKER', 'ServerTimeUsage': 0.004918678, 'CreateTimestamp': 1773764308244, 'FinishTimestamp': 1773764308249, 'Side': 'BUY', 'Type': 'MARKET', 'StopType': 'GTC', 'Price': 669.69, 'Quantity': 1, 'FilledQuantity': 1, 'FilledAverPrice': 669.69, 'CoinChange': 1, 'UnitChange': 669.69, 'CommissionCoin': 'USD', 'CommissionChargeValue': 0.66969, 'CommissionPercent': 0.001, 'OrderWalletType': 'SPOT', 'OrderSource': 'PUBLIC_API'}}
{'Success': True, 'ErrMsg': '', 'OrderDetail': {'Pair': 'BNB/USD', 'OrderID': 2761590, 'Status': 'FILLED', 'Role': 'TAKER', 'ServerTimeUsage': 0.003799002, 'CreateTimestamp': 1773764309610, 'FinishTimestamp': 1773764309614, 'Side': 'SELL', 'Type': 'MARKET', 'StopType': 'GTC', 'Price': 669.68, 'Quantity': 1, 'FilledQuantity': 1, 'FilledAverPrice': 669.68, 'CoinChange': 1, 'UnitChange': 669.68, 'CommissionCoin': 'USD', 'CommissionChargeValue': 0.66968, 'CommissionPercent': 0.001, 'OrderWalletType': 'SPOT', 'OrderSource': 'PUBLIC_API'}}
{'Success': True, 'ErrMsg': '', 'OrderMatched': [{'Pair': 'BNB/USD', 'OrderID': 2761590, 'Status': 'FILLED', 'Role': 'TAKER', 'ServerTimeUsage': 0.003799002, 'CreateTimestamp': 1773764309610, 'FinishTimestamp': 1773764309614, 'Side': 'SELL', 'Type': 'MARKET', 'StopType': 'GTC', 'Price': 669.68, 'Quantity': 1, 'FilledQuantity': 1, 'FilledAverPrice': 669.68, 'CoinChange': 1, 'UnitChange': 669.68, 'CommissionCoin': 'USD', 'CommissionChargeValue': 0.66968, 'CommissionPercent': 0.001, 'OrderWalletType': 'SPOT', 'OrderSource': 'PUBLIC_API'}, {'Pair': 'BNB/USD', 'OrderID': 2761589, 'Status': 'FILLED', 'Role': 'TAKER', 'ServerTimeUsage': 0.004918678, 'CreateTimestamp': 1773764308244, 'FinishTimestamp': 1773764308249, 'Side': 'BUY', 'Type': 'MARKET', 'StopType': 'GTC', 'Price': 669.69, 'Quantity': 1, 'FilledQuantity': 1, 'FilledAverPrice': 669.69, 'CoinChange': 1, 'UnitChange': 669.69, 'CommissionCoin': 'USD', 'CommissionChargeValue': 0.66969, 'CommissionPercent': 0.001, 'OrderWalletType': 'SPOT', 'OrderSource': 'PUBLIC_API'}]}"""

# --- API Configuration ---
BASE_URL = "https://mock-api.roostoo.com"
API_KEY = "XWox7FVBCwEtiO9AcWASMtAnw1iYP5xwb28E0QzL2KajzFO4HOxwuLO0k1uSIVII"      # Replace with your actual API key
SECRET_KEY = "IB8JB6lmGWZh9nTwh9Wo1u9MwIbjv5eK9R0QRPM5iQTdHR9exESHY8VHAdNWRdEY"  # Replace with your actual secret key


# ------------------------------
# Utility Functions
# ------------------------------

def _get_timestamp():
    """Return a 13-digit millisecond timestamp as string."""
    return str(int(time.time() * 1000))


def _get_signed_headers(payload: dict = {}):
    """
    Generate signed headers and totalParams for RCL_TopLevelCheck endpoints.
    """
    payload['timestamp'] = _get_timestamp()
    sorted_keys = sorted(payload.keys())
    total_params = "&".join(f"{k}={payload[k]}" for k in sorted_keys)

    signature = hmac.new(
        SECRET_KEY.encode('utf-8'),
        total_params.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()

    headers = {
        'RST-API-KEY': API_KEY,
        'MSG-SIGNATURE': signature
    }

    return headers, payload, total_params


# ------------------------------
# Public Endpoints
# ------------------------------

def check_server_time():
    """Check API server time."""
    url = f"{BASE_URL}/v3/serverTime"
    try:
        res = requests.get(url)
        res.raise_for_status()
        return res.json()
    except requests.exceptions.RequestException as e:
        print(f"Error checking server time: {e}")
        return None


def get_exchange_info():
    """Get exchange trading pairs and info."""
    url = f"{BASE_URL}/v3/exchangeInfo"
    try:
        res = requests.get(url)
        res.raise_for_status()
        return res.json()
    except requests.exceptions.RequestException as e:
        print(f"Error getting exchange info: {e}")
        return None


def get_ticker(pair=None):
    """Get ticker for one or all pairs."""
    url = f"{BASE_URL}/v3/ticker"
    params = {'timestamp': _get_timestamp()}
    if pair:
        params['pair'] = pair
    try:
        res = requests.get(url, params=params)
        res.raise_for_status()
        return res.json()
    except requests.exceptions.RequestException as e:
        print(f"Error getting ticker: {e}")
        return None


# ------------------------------
# Signed Endpoints
# ------------------------------

def get_balance():
    """Get wallet balances (RCL_TopLevelCheck)."""
    url = f"{BASE_URL}/v3/balance"
    headers, payload, _ = _get_signed_headers({})
    try:
        res = requests.get(url, headers=headers, params=payload)
        res.raise_for_status()
        return res.json()
    except requests.exceptions.RequestException as e:
        print(f"Error getting balance: {e}")
        print(f"Response text: {e.response.text if e.response else 'N/A'}")
        return None


def get_pending_count():
    """Get total pending order count."""
    url = f"{BASE_URL}/v3/pending_count"
    headers, payload, _ = _get_signed_headers({})
    try:
        res = requests.get(url, headers=headers, params=payload)
        res.raise_for_status()
        return res.json()
    except requests.exceptions.RequestException as e:
        print(f"Error getting pending count: {e}")
        print(f"Response text: {e.response.text if e.response else 'N/A'}")
        return None


def place_order(pair_or_coin, side, quantity, price=None, order_type=None):
    """
    Place a LIMIT or MARKET order.
    """
    url = f"{BASE_URL}/v3/place_order"
    pair = f"{pair_or_coin}/USD" if "/" not in pair_or_coin else pair_or_coin

    if order_type is None:
        order_type = "LIMIT" if price is not None else "MARKET"

    if order_type == 'LIMIT' and price is None:
        print("Error: LIMIT orders require 'price'.")
        return None

    payload = {
        'pair': pair,
        'side': side.upper(),
        'type': order_type.upper(),
        'quantity': str(quantity)
    }
    if order_type == 'LIMIT':
        payload['price'] = str(price)

    headers, _, total_params = _get_signed_headers(payload)
    headers['Content-Type'] = 'application/x-www-form-urlencoded'

    try:
        res = requests.post(url, headers=headers, data=total_params)
        res.raise_for_status()
        return res.json()
    except requests.exceptions.RequestException as e:
        print(f"Error placing order: {e}")
        print(f"Response text: {e.response.text if e.response else 'N/A'}")
        return None


def query_order(order_id=None, pair=None, pending_only=None):
    """Query order history or pending orders."""
    url = f"{BASE_URL}/v3/query_order"
    payload = {}
    if order_id:
        payload['order_id'] = str(order_id)
    elif pair:
        payload['pair'] = pair
        if pending_only is not None:
            payload['pending_only'] = 'TRUE' if pending_only else 'FALSE'

    headers, _, total_params = _get_signed_headers(payload)
    headers['Content-Type'] = 'application/x-www-form-urlencoded'

    try:
        res = requests.post(url, headers=headers, data=total_params)
        res.raise_for_status()
        return res.json()
    except requests.exceptions.RequestException as e:
        print(f"Error querying order: {e}")
        print(f"Response text: {e.response.text if e.response else 'N/A'}")
        return None


def cancel_order(order_id=None, pair=None):
    """Cancel specific or all pending orders."""
    url = f"{BASE_URL}/v3/cancel_order"
    payload = {}
    if order_id:
        payload['order_id'] = str(order_id)
    elif pair:
        payload['pair'] = pair

    headers, _, total_params = _get_signed_headers(payload)
    headers['Content-Type'] = 'application/x-www-form-urlencoded'

    try:
        res = requests.post(url, headers=headers, data=total_params)
        res.raise_for_status()
        return res.json()
    except requests.exceptions.RequestException as e:
        print(f"Error canceling order: {e}")
        print(f"Response text: {e.response.text if e.response else 'N/A'}")
        return None


# ------------------------------
# Quick Demo Section
# ------------------------------
if __name__ == "__main__":
    print("\n--- Checking Server Time ---")
    print(check_server_time())

    print("\n--- Getting Exchange Info ---")
    info = get_exchange_info()
    if info:
        print(f"Available Pairs: {list(info.get('TradePairs', {}).keys())}")

    print("\n--- Getting Market Ticker (BTC/USD) ---")
    ticker = get_ticker("BTC/USD")
    if ticker:
        print(ticker.get("Data", {}).get("BTC/USD", {}))

    print("\n--- Getting Account Balance ---")
    print(get_balance())

    print("\n--- Checking Pending Orders ---")
    print(get_pending_count())

    # Uncomment these to test trading actions:
    # print(place_order("BTC", "BUY", 0.01, price=95000))  # LIMIT
    print(place_order("BNB/USD", "BUY", 1))      
    print(place_order("BNB/USD", "SELL", 1))             # MARKET       
    print(query_order(pair="BNB/USD", pending_only=False))
    # print(cancel_order(pair="BNB/USD"))

