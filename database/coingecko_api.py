import requests
import time
from collections import deque

class CoinGeckoClient:
    """
    CoinGecko 免费版 API 
    获取今日（24小时内）的详细行情数据、盘口深度及日内走势。
    包含全局请求限速 (15次/分)，防止触发 API 封禁。
    """
    
    def __init__(self):
        self.base_url = "https://api.coingecko.com/api/v3"
        self.request_times = deque(maxlen=15)

    def _enforce_rate_limit(self):
        """执行全局限速 (15次/分钟)"""
        now = time.time()
        while self.request_times and now - self.request_times[0] > 60:
            self.request_times.popleft()
            
        if len(self.request_times) >= 15:
            wait_time = 60 - (now - self.request_times[0])
            if wait_time > 0:
                print(f"[limits] CoinGecko request rate limit reached, waiting {wait_time:.2f} seconds...")
                time.sleep(wait_time)
                
        self.request_times.append(time.time())

    def _make_request(self, endpoint, params=None):
        """统一的网络请求封装"""
        self._enforce_rate_limit()
        url = f"{self.base_url}{endpoint}"
        
        try:
            res = requests.get(url, params=params, timeout=10)
            res.raise_for_status()
            return res.json()
        except requests.exceptions.RequestException as e:
            print(f"[fail] GET {url} | params: {params} | error: {e}")
            if e.response is not None:
                print(f"[server return] {e.response.text}")
            return None

    # ------------------------------
    # 今日详细行情接口
    # ------------------------------
    
    def get_today_overview(self, coin_ids="bitcoin", vs_currencies="usd"):
        """
        1. 获取今日基础概况：当前价格、24小时交易量、24小时涨跌幅
        :param coin_ids: 币种名称全拼，如 'bitcoin', 'ethereum' (多个用逗号分隔)
        """
        endpoint = "/simple/price"
        params = {
            "ids": coin_ids,
            "vs_currencies": vs_currencies,
            "include_24hr_vol": "true",
            "include_24hr_change": "true"
        }
        return self._make_request(endpoint, params)

    def get_today_intraday_data(self, coin_id="bitcoin", vs_currency="usd"):
        """
        2. 获取今日（过去24小时）的日内时间序列数据
        返回约 288 个数据点（每 5 分钟更新一次），包含价格和总交易量。适合计算日内均线或动量。
        """
        endpoint = f"/coins/{coin_id}/market_chart"
        params = {
            "vs_currency": vs_currency,
            "days": "1"  # 固定为获取过去 1 天的数据
        }
        return self._make_request(endpoint, params)

    def get_market_depth(self, coin_id="bitcoin"):
        """
        3. 获取当前市场盘口深度 (±2% 深度资金量)
        通过各大交易所汇总的详细 Ticker，查看向上砸盘/向下砸盘需要多少美元资金。
        """
        endpoint = f"/coins/{coin_id}/tickers"
        return self._make_request(endpoint)


# ------------------------------
# 测试与调用示范
# ------------------------------
if __name__ == "__main__":
    cg = CoinGeckoClient()
    print("--- 正在获取 CoinGecko 今日详细行情 ---")
    
    # 1. 获取今日概况
    overview = cg.get_today_overview(coin_ids="bitcoin")
    print(f"\n[1. 今日概况] BTC数据: {overview}")
        
    # 2. 获取今日详细的 5分钟级 数据流
    intraday = cg.get_today_intraday_data(coin_id="bitcoin")
    if intraday and 'prices' in intraday:
        latest_price = intraday['prices'][-1]
        print(f"\n[2. 日内数据] 成功获取过去24小时 {len(intraday['prices'])} 个价格节点。最新节点: {latest_price}")
        
    # 3. 获取深度与抛压支撑 (以币安的数据为例)
    depth_data = cg.get_market_depth(coin_id="bitcoin")
    if depth_data and 'tickers' in depth_data:
        print("\n[3. 盘口深度] Binance BTC/USDT 最新数据:")
        for t in depth_data['tickers']:
            # 筛选币安交易所的 USDT 交易对
            if t['market']['identifier'] == 'binance' and t['target'] == 'USDT':
                print(f"  最新成交价: {t['last']}")
                print(f"  24h交易量: {t['volume']}")
                print(f"  向上 2% 抛压卖单厚度 (Ask): {t.get('cost_to_move_up_usd', 'N/A')} USD")
                print(f"  向下 2% 支撑买单厚度 (Bid): {t.get('cost_to_move_down_usd', 'N/A')} USD")
                break