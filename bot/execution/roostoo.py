import requests
import time
import hmac
import hashlib
import yaml
import os
import sys
from pathlib import Path
import logging
from logging.handlers import TimedRotatingFileHandler
from collections import deque

"""
    Roostoo 实时数据 API 客户端类

    用于连接和操作 Roostoo 交易所 API，支持公共行情查询以及需 API 密钥签名的账户交易操作。
    通过读取 YAML 配置文件自动管理 API 密钥，支持多实例并发操作不同账户。

    !!!注意：每分钟请求频率限制为 30 次，请合理安排调用频率，避免被暂时封禁。

    【前置要求】
    1. 确保环境已安装 pyyaml: `pip install pyyaml`
    2. 准备好配置文件，默认路径为 `config/roostoo.yaml`。文件结构如下：
        roostoo:
          base_url: "https://mock-api.roostoo.com"
          api_key: "API_KEY"
          secret_key: "SECRET_KEY"

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
    MaxBid: 最高买价； MinAsk: 最低卖价; LastPrice: 最新成交价; Change: 24小时价格变动率; CoinTradeValue: 24小时成交量（以交易对基础货币计）；UnitTradeValue: 24小时成交额(USD)

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
    【日志系统】
    1. 日志文件位于 `logs/roostoo.log`，每天午夜自动滚动生成新文件，文件名格式为 `roostoo.log.YYYY-MM-DD`。
    2. 日志内容包括时间戳、日志级别和信息，既记录到文件也输出到控制台，方便调试和监控。
    """
ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(ROOT))

class Roostoo:
    def __init__(self, config_path=ROOT/"bot/config/roostoo.yaml"):
        self._setup_logger()

        self.request_times = deque(maxlen=30)
        self.last_order_time = 0.0

        self._load_config(config_path)
    
    def _setup_logger(self):
        """配置按天滚动的日志系统，并将日志强制存储到 bot/logs 文件夹中"""
       
        current_dir = Path(__file__).resolve().parent
        log_dir = current_dir.parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger("RoostooBot")
        self.logger.setLevel(logging.INFO)
        
        # 防止重复添加 Handler
        if not self.logger.handlers:
            # 拼接完整的日志文件绝对路径
            log_file = log_dir / "roostoo.log"
            
            # 使用 str(log_file) 传入绝对路径
            file_handler = TimedRotatingFileHandler(
                filename=str(log_file), 
                when="midnight", 
                interval=1, 
                encoding="utf-8"
            )
            file_handler.suffix = "%Y-%m-%d"
            
            console_handler = logging.StreamHandler()
            
            # 日志格式
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)
            
        self.logger.info(f"Roostoo API logs initialized in: {log_dir}")

    def _load_config(self, config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
                
            rst_config = config.get('roostoo', {})
            self.base_url = rst_config.get('base_url', "https://mock-api.roostoo.com")
            self.api_key = rst_config.get('api_key')
            self.secret_key = rst_config.get('secret_key')
            
            if not self.api_key or not self.secret_key:
                raise ValueError("API Key  Secret Key not found in yaml file")
                
        except FileNotFoundError:
            raise FileNotFoundError(f"yaml not found: {config_path}")
        
    #rate limits & request base
    def _enforce_global_rate_limit(self):
        """执行全局 30次/分钟 限制"""
        now = time.time()
        # 移除 60 秒之前的记录
        while self.request_times and now - self.request_times[0] > 60:
            self.request_times.popleft()
            
        if len(self.request_times) >= 30:
            wait_time = 60 - (now - self.request_times[0])
            if wait_time > 0:
                self.logger.warning(f"limits rate under 30/min, wait: {wait_time:.2f} seconds")
                time.sleep(wait_time)
                
        # 记录本次请求的时间戳
        self.request_times.append(time.time())

    def _make_request(self, method, url, headers=None, params=None, data=None):
        """统一的网络请求封装，自动处理全局限速和异常拦截"""
        self._enforce_global_rate_limit()
        
        try:
            if method.upper() == 'GET':
                res = requests.get(url, headers=headers, params=params)
            else:
                res = requests.post(url, headers=headers, data=data)
                
            res.raise_for_status()
            response_data = res.json()
            # 可以根据需要决定是否要打印每一次成功的请求
            self.logger.info(f"executed: [{method}] {url}") 
            return response_data
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"fail: [{method}] {url} - {e}")
            if e.response is not None:
                self.logger.error(f"server return: {e.response.text}")
            return None


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
        url = f"{self.base_url}/v3/serverTime"
        return self._make_request('GET', url)

    def get_exchange_info(self):
        url = f"{self.base_url}/v3/exchangeInfo"
        return self._make_request('GET', url)

    def get_ticker(self, pair=None):
        url = f"{self.base_url}/v3/ticker"
        params = {'timestamp': self._get_timestamp()}
        if pair:
            params['pair'] = pair
        return self._make_request('GET', url, params=params)

    # ------------------------------
    # 需签名的端点 (Signed Endpoints)
    # ------------------------------
    def get_balance(self):
        url = f"{self.base_url}/v3/balance"
        headers, payload, _ = self._get_signed_headers({})
        return self._make_request('GET', url, headers=headers, params=payload)

    def get_pending_count(self):
        url = f"{self.base_url}/v3/pending_count"
        headers, payload, _ = self._get_signed_headers({})
        return self._make_request('GET', url, headers=headers, params=payload)

    def place_order(self, pair_or_coin, side, quantity, price=None, order_type=None):
        """下达限价 (LIMIT) 或市价 (MARKET) 订单"""
        self._enforce_order_rate_limit() 
        
        url = f"{self.base_url}/v3/place_order"
        pair = f"{pair_or_coin}/USD" if "/" not in pair_or_coin else pair_or_coin

        if order_type is None:
            order_type = "LIMIT" if price is not None else "MARKET"

        if order_type == 'LIMIT' and price is None:
            self.logger.error("LIMIT order needs 'price' parameter, order placement terminated.")
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

        self.logger.info(f"Preparing to place order: {side.upper()} {quantity} {pair} @ {price if price else 'MARKET'}")
        result = self._make_request('POST', url, headers=headers, data=total_params)
        
        if result and result.get('Success'):
            self.logger.info(f"Order placed successfully! Details: {result}")
        else:
            self.logger.error(f"Failed to place order or received unexpected response: {result}")
            
        return result

    def query_order(self, order_id=None, pair=None, pending_only=None):
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

        return self._make_request('POST', url, headers=headers, data=total_params)

    def cancel_order(self, order_id=None, pair=None):
        url = f"{self.base_url}/v3/cancel_order"
        payload = {}
        if order_id:
            payload['order_id'] = str(order_id)
        elif pair:
            payload['pair'] = pair

        headers, _, total_params = self._get_signed_headers(payload)
        headers['Content-Type'] = 'application/x-www-form-urlencoded'
        
        self.logger.info(f"Trying to cancel order: OrderID={order_id}, Pair={pair}")
        return self._make_request('POST', url, headers=headers, data=total_params)


# ------------------------------
# Demo Section
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

    client.logger.info("demo execution")
    
    # 获取余额
    balance = client.get_balance()
    client.logger.info(f"balance: {balance}")

    # 获取行情
    ticker = client.get_ticker("BTC/USD")
    client.logger.info(f"BTC real time price: {ticker.get('Data', {}).get('BTC/USD', {}).get('LastPrice')}")
    
    # 如果你想测试挂单限制，可以解除下面代码的注释并运行：
    # client.place_order("BNB/USD", "BUY", 1) 
    # client.place_order("BNB/USD", "SELL", 1) # 这里会触发挂单限速等待近60秒