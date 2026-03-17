import requests
import time
import hmac
import hashlib
import yaml
import os
import sys
from pathlib import Path

"""
    Roostoo 实时数据 API 客户端类

    用于连接和操作 Roostoo 交易所 API，支持公共行情查询以及需 API 密钥签名的账户交易操作。
    通过读取 YAML 配置文件自动管理 API 密钥，支持多实例并发操作不同账户。

    【前置要求】
    1. 确保环境已安装 pyyaml: `pip install pyyaml`
    2. 准备好配置文件，默认路径为 `config/roostoo.yaml`。文件结构如下：
        roostoo:
          base_url: "https://mock-api.roostoo.com"
          api_key: "你的_API_KEY"
          secret_key: "你的_SECRET_KEY"

    【在其他文件中的使用示例】
    --------------------------------------------------
    from roostoo import Roostoo  

    # 1. 实例化客户端 (默认加载 config/roostoo.yaml)
    # 如果配置文件在其他地方，可传入参数: Roostoo(config_path="path/to/other.yaml")
    client = Roostoo()

    # 2. 调用公共接口 
    # 获取行情,realtime BTC/USD
    btc_ticker = client.get_ticker("BTC/USD")
    print("BTC 行情:", btc_ticker)
    return {'MaxBid': 74225.44, 'MinAsk': 74225.45, 'LastPrice': 74225.45, 'Change': 0.0051, 'CoinTradeValue': 27150.10254, 'UnitTradeValue': 2020685652.110588}

    # 3. 调用需签名的私有接口 (例如查询余额)
    my_balance = client.get_balance()
    print("当前余额:", my_balance)
    return: {'Success': True, 'ErrMsg': '', 'SpotWallet': {'BNB': {'Free': 0, 'Lock': 0}, 'USD': {'Free': 49998.65, 'Lock': 0}}, 'MarginWallet': {}}

    # 4. 
    # 下单操作 (市价买入 1 个 BNB)
    #市价订单
        response = client.place_order("BNB/USD", "BUY", 1)
    #限价订单
        response = client.place_order("BNB/USD", "BUY", 1, price=300, order_type="LIMIT")

    #获取订单状态
        order_status = client.query_order(order_id=response.get("OrderID"))

    #获取未结订单数量
        pending_count = client.get_pending_count()
    #取消订单
        cancel_response = client.cancel_order(order_id=response.get("OrderID"))

    --------------------------------------------------
    """
ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(ROOT))
class Roostoo:
    def __init__(self, config_path=ROOT/"bot/config/roostoo.yaml"):
        self._load_config(config_path)

    def _load_config(self, config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
                
            rst_config = config.get('roostoo', {})
            self.base_url = rst_config.get('base_url', "https://mock-api.roostoo.com")
            self.api_key = rst_config.get('api_key')
            self.secret_key = rst_config.get('secret_key')
            
            if not self.api_key or not self.secret_key:
                raise ValueError("API Key  Secret Key not found in yaml file")
                
        except FileNotFoundError:
            raise FileNotFoundError(f"yaml not found: {config_path}")

    # ------------------------------
    # 内部工具方法 (Utility Functions)
    # ------------------------------
    def _get_timestamp(self):
        """返回 13 位毫秒级时间戳"""
        return str(int(time.time() * 1000))

    def _get_signed_headers(self, payload=None):
        """
        生成带有签名的 headers 和用于 POST 请求的 totalParams
        """
        if payload is None:
            payload = {}
            
        payload['timestamp'] = self._get_timestamp()
        sorted_keys = sorted(payload.keys())
        total_params = "&".join(f"{k}={payload[k]}" for k in sorted_keys)

        signature = hmac.new(
            self.secret_key.encode('utf-8'),
            total_params.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

        headers = {
            'RST-API-KEY': self.api_key,
            'MSG-SIGNATURE': signature
        }

        return headers, payload, total_params

    # ------------------------------
    # 公共端点 (Public Endpoints)
    # ------------------------------
    def check_server_time(self):
        """检查 API 服务器时间"""
        url = f"{self.base_url}/v3/serverTime"
        try:
            res = requests.get(url)
            res.raise_for_status()
            return res.json()
        except requests.exceptions.RequestException as e:
            print(f"Error checking server time: {e}")
            return None

    def get_exchange_info(self):
        """获取交易所交易对和信息"""
        url = f"{self.base_url}/v3/exchangeInfo"
        try:
            res = requests.get(url)
            res.raise_for_status()
            return res.json()
        except requests.exceptions.RequestException as e:
            print(f"Error getting exchange info: {e}")
            return None

    def get_ticker(self, pair=None):
        """获取单个或所有交易对的行情"""
        url = f"{self.base_url}/v3/ticker"
        params = {'timestamp': self._get_timestamp()}
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
    # 需签名的端点 (Signed Endpoints)
    # ------------------------------
    def get_balance(self):
        """获取钱包余额 (RCL_TopLevelCheck)"""
        url = f"{self.base_url}/v3/balance"
        headers, payload, _ = self._get_signed_headers({})
        try:
            res = requests.get(url, headers=headers, params=payload)
            res.raise_for_status()
            return res.json()
        except requests.exceptions.RequestException as e:
            print(f"Error getting balance: {e}")
            print(f"Response text: {e.response.text if e.response else 'N/A'}")
            return None

    def get_pending_count(self):
        """获取未结订单总数"""
        url = f"{self.base_url}/v3/pending_count"
        headers, payload, _ = self._get_signed_headers({})
        try:
            res = requests.get(url, headers=headers, params=payload)
            res.raise_for_status()
            return res.json()
        except requests.exceptions.RequestException as e:
            print(f"Error getting pending count: {e}")
            print(f"Response text: {e.response.text if e.response else 'N/A'}")
            return None

    def place_order(self, pair_or_coin, side, quantity, price=None, order_type=None):
        """下达限价 (LIMIT) 或市价 (MARKET) 订单"""
        url = f"{self.base_url}/v3/place_order"
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

        headers, _, total_params = self._get_signed_headers(payload)
        headers['Content-Type'] = 'application/x-www-form-urlencoded'

        try:
            res = requests.post(url, headers=headers, data=total_params)
            res.raise_for_status()
            return res.json()
        except requests.exceptions.RequestException as e:
            print(f"Error placing order: {e}")
            print(f"Response text: {e.response.text if e.response else 'N/A'}")
            return None

    def query_order(self, order_id=None, pair=None, pending_only=None):
        """查询订单历史或未结订单"""
        url = f"{self.base_url}/v3/query_order"
        payload = {}
        if order_id:
            payload['order_id'] = str(order_id)
        elif pair:
            payload['pair'] = pair
            if pending_only is not None:
                payload['pending_only'] = 'TRUE' if pending_only else 'FALSE'

        headers, _, total_params = self._get_signed_headers(payload)
        headers['Content-Type'] = 'application/x-www-form-urlencoded'

        try:
            res = requests.post(url, headers=headers, data=total_params)
            res.raise_for_status()
            return res.json()
        except requests.exceptions.RequestException as e:
            print(f"Error querying order: {e}")
            print(f"Response text: {e.response.text if e.response else 'N/A'}")
            return None

    def cancel_order(self, order_id=None, pair=None):
        """取消特定订单或某交易对的所有未结订单"""
        url = f"{self.base_url}/v3/cancel_order"
        payload = {}
        if order_id:
            payload['order_id'] = str(order_id)
        elif pair:
            payload['pair'] = pair

        headers, _, total_params = self._get_signed_headers(payload)
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
# 实例调用演示 (Demo Section)
# ------------------------------
if __name__ == "__main__":
    # 初始化客户端，默认会去 "config/config.yaml" 找配置文件
    client = Roostoo()

    print("\n--- Checking Server Time ---")
    print(client.check_server_time())

    print("\n--- Getting Exchange Info ---")
    info = client.get_exchange_info()
    if info:
        print(f"Available Pairs: {list(info.get('TradePairs', {}).keys())}")

    print("\n--- Getting Market Ticker (BTC/USD) ---")
    ticker = client.get_ticker("BTC/USD")
    if ticker:
        print(ticker.get("Data", {}).get("BTC/USD", {}))

    print("\n--- Getting Account Balance ---")
    print(client.get_balance())

    # 测试下单
    # print(client.place_order("BNB/USD", "BUY", 1))