import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from bot.api.roostoo import Roostoo
from bot.portfolio.portfolio import Portfolio
from bot.execution.execution_engine import ExecutionEngine
from database.Binance_Vision_fetcher import VisionFetcher

# 引入我们重构后的策略管理器
from bot.strategy.strategy_ml import DualMLLiveManager
from bot.strategy.strategy_obi_eth import ObiEthStrategy

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(ROOT / "bot" / "logs" / "main_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("MainControl")

# 初始化基础设施
roostoo_client = Roostoo()
portfolio = Portfolio(execution_module=None)
execution = ExecutionEngine(portfolio, roostoo_client)
portfolio.execution = execution
portfolio.initialize_from_exchange_info(roostoo_client)

# 初始化策略模块
ml_manager = DualMLLiveManager(portfolio, execution)
obi_strategy = ObiEthStrategy(portfolio, execution, symbol="ETHUSDT", coin="ETH")

# 数据抓取工具
fetcher = VisionFetcher()

def fetch_latest_market_data():
    """抓取最新 K 线数据供策略计算指标"""
    sim_data = {}
    end_dt = datetime.today()
    start_dt = end_dt - timedelta(days=5) # 获取过去5天的数据保证 EMA/SMA 有足够的长度

    # 获取 ML 策略涉及的币种 + OBI 的 ETH
    all_coins = [s["coin"] for s in ml_manager.strategies.values()] + ["ETH"]
    
    for coin in set(all_coins):
        symbol = f"{coin}USDT"
        df = fetcher.fetch_klines_range(
            symbol=symbol, interval="5m", 
            start_year=start_dt.year, start_month=start_dt.month, start_day=start_dt.day,
            end_year=end_dt.year, end_month=end_dt.month, end_day=end_dt.day,
            data_type="daily"
        )
        if df is not None and not df.empty:
            sim_data[coin] = df
            
    return sim_data

def main():
    logger.info("Starting live trading cycle...")
    
    while True:
        try:
            # 1. 后台队列订单同步与结算
            q = roostoo_client.query_order(pending_only=True)
            if q:
                processed = execution.process_query_response(q)
                if processed: logger.info(f"Processed pending orders: {processed}")

           # 2. 拉取全局最新数据 (5分钟级别)
            sim_data = fetch_latest_market_data()
            current_time = datetime.now()

            if sim_data:
                # --- 🟢 新增：计算全局总净值 (Total Equity) ---
                # 1. 提取所有币种的最新收盘价并更新到 Portfolio
                current_prices = {c: float(df.iloc[-1]["close"]) for c, df in sim_data.items()}
                portfolio.set_market_prices(current_prices)
                
                # 2. 现金 + 各币种持仓市值 = 真实总净值
                total_equity = portfolio.account_balance
                for coin, pos in portfolio.positions.items():
                    qty = pos.get('free', 0.0) + pos.get('locked', 0.0)
                    price = current_prices.get(coin, 0.0)
                    total_equity += qty * price
                # -----------------------------------------------

                # 3. 驱动 ML 截面策略矩阵 (传入 total_equity)
                ml_manager.on_tick(sim_data, current_time, total_equity)

                # 4. 驱动 OBI 动量突破策略 (传入 total_equity)
                eth_df = sim_data.get("ETH")
                if eth_df is not None:
                    obi_strategy.on_live_candle(eth_df, total_equity)
            else:
                logger.warning("Failed to fetch market data this cycle.")

            # 5. 输出快照并等待下一个 5 分钟轮询周期 
            # (实盘中若对接 WebSocket 可去轮询机制，这里沿用你的架构以轮询形式进行)
            snapshot = portfolio.get_pnl_snapshot()
            logger.info(f"Portfolio Total PnL Snapshot: {snapshot.get('portfolio_summary')}")
            
            time.sleep(300) # 休眠5分钟等待下一根 K 线

        except Exception as e:
            logger.error(f"Error in main loop: {e}", exc_info=True)
            time.sleep(10)  

if __name__ == "__main__":
    main()