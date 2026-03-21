import logging
import sys
import time
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from bot.api.roostoo import Roostoo
from bot.portfolio.portfolio import Portfolio
from bot.execution.execution_engine import ExecutionEngine

# 引入双轨数据抓取工具
from database.Binance_Vision_fetcher import VisionFetcher
from database.Binance_fetcher import BinanceDataFetcher

# 引入重构后的策略管理器
from bot.strategy.strategy_ml import DualMLLiveManager
from bot.strategy.strategy_obi_eth import ObiDynamicStrategy 

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(ROOT / "bot" / "logs" / "main_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("MainControl")

# ==========================================
# 基础设施初始化
# ==========================================
roostoo_client = Roostoo()
portfolio = Portfolio(execution_module=None)
execution = ExecutionEngine(portfolio, roostoo_client)
portfolio.execution = execution
portfolio.initialize_from_exchange_info(roostoo_client)

ml_manager = DualMLLiveManager(portfolio, execution)
obi_strategy = ObiDynamicStrategy(portfolio, execution)

vision_fetcher = VisionFetcher()
realtime_fetcher = BinanceDataFetcher()

# ==========================================
# 核心数据管道：冷热双轨内存池 (隔离多周期)
# ==========================================
# 建立两个独立的内存池供不同策略使用
global_market_data_5m = {}
global_market_data_15m = {}

def initialize_cold_data(coins, days, interval, target_dict):
    """【冷启动】按指定周期拉取数据填充目标内存池"""
    logger.info(f"🧊 开始冷启动 ({interval})：拉取过去 {days} 天历史数据...")
    end_dt = datetime.today()
    start_dt = end_dt - timedelta(days=days)
    
    for coin in set(coins):
        symbol = f"{coin}USDT"
        df = vision_fetcher.fetch_klines_range(
            symbol=symbol, interval=interval, 
            start_year=start_dt.year, start_month=start_dt.month, start_day=start_dt.day,
            end_year=end_dt.year, end_month=end_dt.month, end_day=end_dt.day,
            data_type="daily"
        )
        if df is not None and not df.empty:
            df['open_time'] = pd.to_datetime(df['open_time'])
            target_dict[coin] = df.sort_values('open_time')
            logger.info(f"✅ [{coin}-{interval}] 历史底表加载完成 ({len(df)} 根 K 线)")
        else:
            logger.error(f"❌ [{coin}-{interval}] 历史数据拉取失败！")

def update_realtime_data(coins, interval, target_dict):
    """【热更新】按指定周期拉取最新 K 线并与目标内存池拼接去重"""
    # 根据周期动态计算保留的 K 线数：确保内存不过载，5m保留65天，15m保留30天即可
    max_bars = (65 * 288) if interval == "5m" else (30 * 96)

    for coin in set(coins):
        symbol = f"{coin}USDT"
        recent_df = realtime_fetcher.fetch_recent_klines(symbol=symbol, interval=interval, limit=500)
        
        if recent_df is None or recent_df.empty:
            logger.warning(f"⚠️ [{coin}-{interval}] 实时数据拉取失败，本轮使用旧缓存")
            continue

        if coin not in target_dict or target_dict[coin].empty:
            target_dict[coin] = recent_df
        else:
            combined = pd.concat([target_dict[coin], recent_df], ignore_index=True)
            combined = combined.drop_duplicates(subset=['open_time'], keep='last').sort_values('open_time')
            target_dict[coin] = combined.iloc[-max_bars:].reset_index(drop=True)
            
    return target_dict

# ==========================================
# 实盘主循环
# ==========================================
def main():
    logger.info("Starting live trading initialization...")
    
    # 明确各策略所需的币种
    ml_coins = list(ml_manager.strategies.keys()) + ["BTC"] # ML 必须有 BTC 算 Regime
    obi_coins = ["BTC", "ETH", "TAO"]                       # OBI 监控标的
    
    # 1. 分周期盘前冷启动
    initialize_cold_data(ml_coins, days=65, interval="5m", target_dict=global_market_data_5m)
    initialize_cold_data(obi_coins, days=30, interval="15m", target_dict=global_market_data_15m)
    
    logger.info("🚀 跨周期实盘交易引擎正式启动，进入监听轮询...")
    
    while True:
        try:
            # 2. 队列订单同步与结算
            q = roostoo_client.query_order(pending_only=True)
            if q:
                processed = execution.process_query_response(q)
                if processed: logger.info(f"Processed pending orders: {processed}")

            current_time = datetime.now()

            # 3. 双路并行更新热数据
            sim_data_5m = update_realtime_data(ml_coins, interval="5m", target_dict=global_market_data_5m)
            sim_data_15m = update_realtime_data(obi_coins, interval="15m", target_dict=global_market_data_15m)

            if sim_data_5m and sim_data_15m:
                # 4. 计算全局总净值 (用最新、最密集的 5m 价格去估值更加准确)
                current_prices = {c: float(df.iloc[-1]["close"]) for c, df in sim_data_5m.items() if not df.empty}
                for c, df in sim_data_15m.items():
                    if c not in current_prices and not df.empty:
                        current_prices[c] = float(df.iloc[-1]["close"])
                portfolio.set_market_prices(current_prices)
                
                total_equity = portfolio.account_balance
                for coin, pos in portfolio.positions.items():
                    qty = pos.get('free', 0.0) + pos.get('locked', 0.0)
                    total_equity += qty * current_prices.get(coin, 0.0)

                # 5. 分别将 5m 和 15m 数据注入对应策略
                ml_manager.on_tick(sim_data_5m, current_time, total_equity)
                obi_strategy.on_tick(sim_data_15m, total_equity)
                
            else:
                logger.warning("Failed to fetch market data this cycle.")

            # 6. 输出快照并等待
            snapshot = portfolio.get_pnl_snapshot()
            logger.info(f"Portfolio Total PnL Snapshot: {snapshot.get('portfolio_summary')}")
            
            # 由于 ML 最低颗粒度是 5m，所以主循环设为 5 分钟睡眠完全合适。
            # (15m 级别的 OBI 在中间那两次 5m 轮询中，因为 K 线还没收盘，它的内部控制流会自然过滤)
            time.sleep(300) 

        except Exception as e:
            logger.error(f"Error in main loop: {e}", exc_info=True)
            time.sleep(10)  

if __name__ == "__main__":
    main()