import sys
import time
import logging
from pathlib import Path
from datetime import datetime

# 确保能引到 bot 目录
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# 导入你的核心组件
from bot.execution.roostoo import Roostoo
from bot.portfolio.portfolio import Portfolio
from bot.execution.execution_engine import ExecutionEngine
from bot.strategy.strategy_ml import DualMLStrategy
from database.Binance_fetcher import BinanceDataFetcher

# 配置主程序日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(ROOT / "bot" / "logs" / "main_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("MainControl")

def main():
    logger.info("🚀 启动商赛专供高频量化机器人 (5m 级别) ...")
    
    # 1. 初始化交易组件 (配置好你的 config/roostoo.yaml)
    logger.info("初始化 Roostoo 客户端与资产管理器...")
    roostoo_client = Roostoo()  
    portfolio = Portfolio(execution_module=None) 
    
    # 初始化资金 (可以从 Roostoo 接口拉取真实余额，这里为了安全先赋初值)
    # 比赛如果发了初始资金，可以通过 roostoo_client.get_balance() 动态获取
    portfolio.account_balance = 10000.0  
    
    # 绑定执行引擎
    execution = ExecutionEngine(portfolio, roostoo_client)
    portfolio.execution = execution 
    
    # 2. 实例化策略
    strategy = DualMLStrategy(portfolio, execution, symbol="BTCUSDT", coin="BTC")
    
    # 3. 启动前训练模型 (拉取近 60 天数据寻找最新盘感)
    # 注意：这步比较耗时，只在机器人启动时执行一次
    success = strategy.train_models(days_back=60)
    if not success:
        logger.error("🛑 模型训练失败，程序退出！请检查网络或 Binance Vision 数据源。")
        return
        
    logger.info("✅ 机器人武装完毕，进入实时盯盘模式！")
    
    # 4. 实时数据获取器 (用于每 5 分钟获取最新 K 线)
    fetcher = BinanceDataFetcher()
    
    # 5. 主循环 (Event Loop)
    while True:
        try:
            # 等待到下一个 5 分钟的整点 (例如 10:05:00, 10:10:00)
            # 实盘中稍微多等 2-3 秒，确保交易所的 K 线刚好收盘并生成
            current_time = datetime.now()
            seconds_to_next_5m = 300 - (current_time.minute % 5) * 60 - current_time.second
            
            # 如果马上就到整点了，稍微顺延一点，避免拿到没收盘的旧线
            if seconds_to_next_5m < 5: 
                seconds_to_next_5m += 300
                
            logger.info(f"💤 正在等待下一根 5m K 线收盘，休眠 {seconds_to_next_5m} 秒...")
            time.sleep(seconds_to_next_5m + 3) # 多加 3 秒作为网络缓冲
            
            # --- 以下是每 5 分钟触发一次的核心逻辑 ---
            logger.info("📡 正在获取最新 5m K 线数据...")
            # 抓取最新的 50 根 K 线，足够计算各种技术指标 (SMA_20, 动量等)
            recent_df = fetcher.fetch_recent_klines(symbol="BTCUSDT", interval="5m", limit=50)
            
            if recent_df is not None and not recent_df.empty:
                # 获取刚刚收盘的那根线的最新价格
                current_price = recent_df.iloc[-1]['close']
                logger.info(f"📊 当前 BTCUSDT 现价: {current_price}")
                
                # 喂给策略，让模型进行预测并决定是否开枪
                strategy.on_new_candle(recent_df, current_price)
            else:
                logger.warning("⚠️ 未能获取到最新行情数据，跳过本轮检测。")
                
        except Exception as e:
            logger.exception(f"❌ 主循环发生未捕获异常: {e}")
            logger.info("等待 30 秒后重试...")
            time.sleep(30)

if __name__ == "__main__":
    main()
