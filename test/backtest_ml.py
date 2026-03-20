import sys
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# 导入机器学习核心库
import lightgbm as lgb
import xgboost as xgb

# ==========================================
# 路径兼容处理：确保在 AWS/Docker 容器内运行时，
# 能够正确识别项目根目录，避免 ModuleNotFoundError
# ==========================================
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# 导入项目内的自定义组件
from database.Binance_Vision_fetcher import VisionFetcher
from database.Binance_fetcher import BinanceDataFetcher
from bot.data.feature_engineering import FeatureEngineer
from bot.portfolio.portfolio import Portfolio
from bot.execution.execution_engine import ExecutionEngine
from bot.execution.roostoo import Roostoo

class DualMLStrategy:
    """
    双模型机器学习量化策略 - 10天比赛高频专供版 (5m 级别)
    
    【模块定位】
    本类是整个交易机器人的“大脑”。它负责：
    1. 启动时拉取历史数据，训练 LightGBM 和 XGBoost 两个模型。
    2. 实盘中每 5 分钟接收一次最新数据，输出上涨概率（胜率）。
    3. 根据胜率和当前持仓状态，向 ExecutionEngine 发送买入、止损、止盈或提前平仓指令。
    """
    def __init__(self, portfolio: Portfolio, execution: ExecutionEngine, 
                 symbol="BTCUSDT", coin="BTC", strategy_id="ml_dual_01"):
        # 资产管家（负责查余额、加锁防重复下单）
        self.portfolio = portfolio
        # 执行引擎（负责调 Roostoo API 发送真实订单）
        self.execution = execution
        
        self.symbol = symbol          # 交易对，如 BTCUSDT
        self.coin = coin              # 目标资产，如 BTC
        self.strategy_id = strategy_id # 策略唯一标识符，用于 portfolio 记账
        
        # 配置日志，方便在 AWS CloudWatch 或本地查看运行状态
        self.logger = logging.getLogger("DualMLStrategy")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            ch = logging.StreamHandler()
            ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(ch)

        # ==========================================
        # 🏆 比赛专属优化参数 (基于 5m 级别刮头皮优化得出)
        # ==========================================
        # conf_thresh: 极高胜率狙击门槛。只有当双模型认为上涨概率 > 66.3% 时才开枪
        self.conf_thresh = 0.663   
        # sl_pct: 1.9% 宽止损。5m级别噪音大，留够空间防止被庄家“上下影线”意外扫损
        self.sl_pct = 0.019        
        # tp_pct: 2.4% 大波段止盈。在微观 5m 寻找入场点，吃宏观的波段利润
        self.tp_pct = 0.024        
        
        # 特征工程引擎：负责把 OHLCV 转化成 RSI、MACD、订单流不平衡等几十个特征
        self.engineer = FeatureEngineer()
        self.lgb_model = None
        self.xgb_model = None
        self.feature_cols = None
        self.is_trained = False # 标记模型是否已经就绪

    def train_models(self, days_back=60):
        """
        【初始化训练阶段】
        实盘程序启动时必须先运行此方法。它会拉取最近 60 天的数据重新训练模型，
        确保模型掌握的是“当下市场”的最新盘感（避免使用半年前的失效规律）。
        """
        self.logger.info(f"🧠 开始拉取过去 {days_back} 天的 5m 数据用于实盘前训练...")
        
        # 使用 VisionFetcher 批量拉取历史数据（通常是从币安官方数据源下载 zip 包并解压）
        fetcher = VisionFetcher()
        end_date = datetime.today()
        start_date = end_date - timedelta(days=days_back)
        
        # ⚠️ 注意：这里必须是 5m 级别，与实盘获取的实时数据级别严格对齐
        df_history = fetcher.fetch_klines_range(
            symbol=self.symbol, interval="5m", 
            start_year=start_date.year, start_month=start_date.month, start_day=start_date.day,
            end_year=end_date.year, end_month=end_date.month, end_day=end_date.day,
            data_type="daily" 
        )
        
        if df_history is None or df_history.empty:
            self.logger.error("数据拉取失败，无法训练模型！请检查网络或 Binance API 连通性。")
            return False

        self.logger.info("进行特征工程处理...")
        # 将原始的 K 线数据转化为机器学习所需的特征矩阵
        df = self.engineer.generate_features(df_history)
        
        # target_lookback = 6 意味着模型试图预测未来 6 根 5m K线 (即 30 分钟内) 的涨跌
        target_lookback = 6 
        
        # 剔除不能作为特征的列（如时间、原始价格、未来标签等），防止数据穿越 (Data Leakage)
        exclude_cols = [
            'open_time', 'close_time', 'close', 'open', 'high', 'low', 'volume',
            'buy_volume', 'sell_volume', f'target_return_{target_lookback}', 'target_class'
        ]
        self.feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # 清洗数据，剔除含有 NaN 的行
        mask = ~(df[self.feature_cols].isna().any(axis=1) | df['target_class'].isna())
        df_clean = df[mask].copy()
        
        X_train = df_clean[self.feature_cols]
        y_train = df_clean['target_class']
        
        self.logger.info(f"训练集数据量: {len(X_train)}，开始训练 LightGBM 和 XGBoost...")
        
        # 训练 LightGBM 模型 (擅长处理非线性且训练速度极快)
        lgb_params = {'objective': 'binary', 'metric': 'auc', 'learning_rate': 0.05, 'max_depth': 7, 'verbose': -1}
        lgb_train = lgb.Dataset(X_train, label=y_train)
        self.lgb_model = lgb.train(lgb_params, lgb_train, num_boost_round=150)
        
        # 训练 XGBoost 模型 (鲁棒性极强，防过拟合)
        xgb_train = xgb.DMatrix(X_train, label=y_train)
        xgb_params = {'objective': 'binary:logistic', 'eval_metric': 'auc', 'max_depth': 7, 'learning_rate': 0.05}
        self.xgb_model = xgb.train(xgb_params, xgb_train, num_boost_round=150)
        
        self.is_trained = True
        self.logger.info("✅ 模型训练完毕，实盘武装就绪！")
        return True

    def predict_signal(self, recent_df: pd.DataFrame) -> float:
        """
        【实时推理阶段】
        传入主程序最新拉取的 DataFrame（需包含足够的历史窗口，例如 50 根 K 线以计算 SMA_20 等指标），
        输出最新的上涨概率 (0.0 ~ 1.0)。
        """
        if not self.is_trained:
            self.logger.warning("模型尚未训练！")
            return 0.0
            
        # 生成最新的一批特征
        features_df = self.engineer.generate_features(recent_df)
        if features_df.empty:
            return 0.0
            
        # 只取最新的一行特征用于预测当前时刻
        latest_features = features_df.iloc[-1:][self.feature_cols]
        
        # 如果最新特征中存在 NaN（比如数据刚启动，均线还没预热好），则放弃预测
        if latest_features.isna().any(axis=1).iloc[0]:
            return 0.0 
            
        # 让两个模型分别预测，取平均值 (Ensemble 集成学习，提高稳定性)
        prob_lgb = self.lgb_model.predict(latest_features)[0]
        prob_xgb = self.xgb_model.predict(xgb.DMatrix(latest_features))[0]
        
        return float((prob_lgb + prob_xgb) / 2.0)

    def on_new_candle(self, recent_df: pd.DataFrame, current_price: float):
        """
        【实盘决策核心】
        主程序 (main.py) 每 5 分钟获取到新 K 线后，必须调用此方法！
        它会计算概率 -> 检查持仓 -> 决定是否下单或强平。
        """
        # 1. 计算最新胜率
        prob = self.predict_signal(recent_df)
        
        # 2. 检查我们当前是否已经持有该币种 (通过 Portfolio 的锁机制判断)
        lock_status = self.portfolio.get_coin_lock_status(self.coin)
        
        # ==================================
        # 场景 A：当前空仓，寻找狙击机会
        # ==================================
        if not lock_status['locked']:
            # 如果胜率突破阈值 (0.663)，果断开枪
            if prob > self.conf_thresh:
                self.logger.info(f"🎯 [狙击时刻] 发现极高胜率机会! 胜率: {prob:.4f} > 阈值: {self.conf_thresh}")
                
                # 满载复利：比赛专用，每次梭哈可用资金的 85%
                invest_amount = self.portfolio.account_balance * 0.85
                quantity = round(invest_amount / current_price, 4) # 计算买入数量，保留4位小数
                
                # 提前计算好止损和止盈的绝对价格
                sl_price = round(current_price * (1 - self.sl_pct), 2)
                tp_price = round(current_price * (1 + self.tp_pct), 2)
                
                self.logger.info(f"🚀 发送买入订单: {quantity} {self.coin} | 现价: {current_price} | 止损: {sl_price}, 止盈: {tp_price}")
                
                # 调动执行引擎下单。
                # 注意：这里模拟使用的是 LIMIT 单（以现价挂单），如果 Roostoo 支持 MARKET 市价单，实盘中改用市价单成交率更高
                res = self.execution.execute_order(
                    coin=self.coin, side='BUY', quantity=quantity, 
                    price=current_price, order_type='LIMIT', 
                    strategy_id=self.strategy_id, stop_loss=sl_price, take_profit=tp_price
                )
                
                if res.get('success'):
                    self.logger.info(f"✅ 订单提交成功，平台单号: {res.get('order_id')}")
                    # 💡 注意：执行引擎在下单成功后，会自动调用 portfolio.acquire_coin 锁住该币种，
                    # 这样下一个 5 分钟到来时，就不会重复触发买入。
                else:
                    self.logger.error(f"❌ 订单提交失败，API 返回: {res.get('message')}")
                    
        # ==================================
        # 场景 B：当前持仓中，进行 AI 智能风控
        # ==================================
        else:
            # 日常的止损止盈通常由 ExecutionEngine 或 Portfolio 后台轮询处理，
            # 但这里我们加入了独家的“AI 防御机制”：
            # 如果持仓期间，模型发现市场风向骤变，上涨概率跌破 20% (说明跌的概率高达 80%)，
            # 不要死等 1.9% 的物理止损，立刻拉响警报，准备割肉离场！
            if prob < 0.20:
                self.logger.warning(f"⚠️ [AI 警报] 趋势极度恶化 (胜率仅 {prob:.4f} < 0.20)，建议立刻执行强平！")
                
                # 队友注意：如果要在实盘中启用 AI 强平功能，请解除下面代码的注释，
                # 并确保你的 portfolio 记录了当前持仓的 quantity。
                """
                # 获取当前持仓量 (假设 portfolio 里存了)
                # current_qty = self.portfolio.get_position_quantity(self.coin) 
                
                # 强平指令：
                # self.execution.execute_order(
                #     coin=self.coin, side='SELL', quantity=current_qty, 
                #     price=current_price, order_type='MARKET', strategy_id=self.strategy_id
                # )
                
                # 平仓后别忘了释放锁，让资金可以投入下一次交易：
                # self.portfolio.force_release_coin(self.coin)
                """


# ==========================================
# 本地回测/推演模拟器
# 【说明】这部分代码在 AWS 实盘运行 main.py 时不会被执行。
# 它是留给我们在本地跑 python ml_strategy.py 看看策略“模拟未来10天”表现的测试台。
# ==========================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🏆 双模型 ML 策略：10 天比赛前置推演模拟")
    print("="*60)
    
    # 初始化虚拟组件 (不需要真实 API Keys)
    roostoo_client = Roostoo()
    portfolio = Portfolio(execution_module=None) 
    portfolio.account_balance = 10000.0          
    execution = ExecutionEngine(portfolio, roostoo_client)
    portfolio.execution = execution 
    
    strategy = DualMLStrategy(portfolio, execution, symbol="BTCUSDT", coin="BTC")
    
    # 1. 训练近 60 天数据
    strategy.train_models(days_back=60)
    
    # 2. 拉取最近 15 天的 5m 级别数据进行演习
    print("\n⏳ 正在拉取近 15 天的数据进行实盘推演...")
    fetcher = VisionFetcher()
    end_date = datetime.today()
    start_date = end_date - timedelta(days=15)
    
    sim_df = fetcher.fetch_klines_range(
        symbol="BTCUSDT", interval="5m", 
        start_year=start_date.year, start_month=start_date.month, start_day=start_date.day,
        end_year=end_date.year, end_month=end_date.month, end_day=end_date.day,
        data_type="daily"
    )
    
    if sim_df is not None and not sim_df.empty:
        print(f"✅ 拉取成功！共 {len(sim_df)} 根 K 线，开始逐根 K 线推演...")
        
        balance = 10000.0
        position_qty = 0.0
        entry_price = 0.0
        trades_count = 0
        window_size = 50 # 保证给特征工程留出 50 根 K 线的计算余量
        
        # 模拟时间流逝，一根一根喂 K 线
        for i in range(window_size, len(sim_df)):
            window_df = sim_df.iloc[i-window_size : i].copy()
            curr_row = window_df.iloc[-1]
            curr_price = curr_row['close']
            curr_high = curr_row['high']
            curr_low = curr_row['low']
            curr_time = curr_row['open_time']
            
            # 模拟：如果有持仓，先判断是否触碰了物理止损止盈，或者触发了 AI 强平
            if position_qty > 0:
                sl_price = entry_price * (1 - strategy.sl_pct)
                tp_price = entry_price * (1 + strategy.tp_pct)
                prob = strategy.predict_signal(window_df)
                
                exit_price = None
                reason = ""
                
                if curr_low <= sl_price:
                    exit_price = sl_price
                    reason = "🛑 触发物理止损"
                elif curr_high >= tp_price:
                    exit_price = tp_price
                    reason = "🎯 触发大波段止盈"
                elif prob < 0.20:
                    exit_price = curr_price
                    reason = "⚠️ 触发 AI 提前强平"
                    
                if exit_price:
                    # 模拟平仓结算
                    revenue = position_qty * exit_price * (1 - 0.001) # 扣除千分之一手续费
                    pnl_pct = (exit_price - entry_price) / entry_price * 100
                    balance += revenue
                    position_qty = 0.0
                    trades_count += 1
                    print(f"[{curr_time}] {reason} | 卖出价: {exit_price:.2f} | 盈亏: {pnl_pct:.2f}% | 余额: ${balance:.2f}")
                    # 强制释放锁，准备下一轮开枪
                    portfolio.force_release_coin("BTC")
                    
            # 模拟：如果空仓，交给策略判断是否需要狙击
            if position_qty == 0:
                prob = strategy.predict_signal(window_df)
                if prob > strategy.conf_thresh:
                    invest = balance * 0.85 # 梭哈 85% 资金
                    balance -= invest
                    position_qty = (invest * (1 - 0.001)) / curr_price
                    entry_price = curr_price
                    print(f"[{curr_time}] 🟢 发起重仓买单 | 概率: {prob:.4f} | 买入价: {entry_price:.2f}")
                    # 加上锁
                    portfolio.acquire_coin("BTC", strategy.strategy_id)
        
        # 结束后清算最后的一笔单子
        final_equity = balance + (position_qty * curr_price * (1 - 0.001)) if position_qty > 0 else balance
        roi = (final_equity - 10000.0) / 10000.0 * 100
        print("\n" + "="*40)
        print("🏁 10天赛前推演结束！")
        print(f"总交易次数: {trades_count}")
        print(f"初始资金: $10000.00")
        print(f"最终净值: ${final_equity:.2f} (10天预期 ROI: {roi:.2f}%)")
        print("="*40)