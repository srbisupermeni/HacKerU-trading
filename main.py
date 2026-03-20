import logging
import sys
import time
from pathlib import Path
import random

# 确保能引到 bot 目录
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# 导入你的核心组件
from bot.api.roostoo import Roostoo
from bot.portfolio.portfolio import Portfolio
from bot.execution.execution_engine import ExecutionEngine

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

roostoo_client = Roostoo()
portfolio = Portfolio(execution_module=None)
execution = ExecutionEngine(portfolio, roostoo_client)
portfolio.execution = execution
portfolio.initialize_from_exchange_info(roostoo_client)


def main():
    logger.info("Starting main loop...")
    while True:
        try:

            # 1. 查询订单状态
            q = roostoo_client.query_order()
            logger.info(f"Queried order status: {q}")

            # 2. 处理查询结果
            processed = execution.process_query_response(q)
            logger.info(f"Processed query response: {processed}")

            # 3. 输出当前投资组合快照
            snapshot = portfolio.get_pnl_snapshot()
            logger.info(f"Portfolio snapshot: {snapshot}")

            # 4. 等待下一轮
            time.sleep(1)
        except Exception as e:
            logger.error(f"Error in main loop: {e}", exc_info=True)
            time.sleep(1)  # 遇到错误时稍微等待一下再重试


if __name__ == "__main__":
    while True:
        main()
