import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

# 导入你的模块
from database.Binance_fetcher import BinanceDataFetcher
from bot.data.feature_engineering import FeatureEngineer
from bot.strategy.rl_env import CryptoSpotEnv

def run_backtest():
    print("1. 正在获取最近 15 天的未见数据 (Out-of-Sample) 用于回测...")
    fetcher = BinanceDataFetcher()
    # 15天 * 24小时 * 4 (15分钟线) = 1440 根 K 线，拉取 1500 根
    raw_df = fetcher.fetch_recent_klines(symbol="BTCUSDT", interval="15m", limit=1500)
    
    if raw_df is None or raw_df.empty:
        print("错误：未能获取回测数据。")
        return

    print("2. 执行特征工程...")
    engineer = FeatureEngineer()
    features_df = engineer.generate_features(raw_df)
    features_df.dropna(inplace=True)
    features_df.reset_index(drop=True, inplace=True)
    
    # 保存对齐后的收盘价用于后续画图对比
    # 注意：特征工程截掉了前几个用于计算移动平均的行，所以要对齐
    test_prices = features_df['close'].values 

    print("3. 初始化回测环境...")
    # 这里 fee_rate 必须和实盘（或比赛平台）完全一致
    env = CryptoSpotEnv(df=features_df, window_size=12, initial_balance=50000.0, fee_rate=0.0005)

    print("4. 加载最优 PPO 模型...")
    model_path = "./models/best_model/best_model"
    if not os.path.exists(f"{model_path}.zip"):
        print(f"找不到模型文件 {model_path}.zip，请确认路径是否正确。")
        # 降级使用最终模型
        model_path = "./models/ppo_crypto_final"
        
    model = PPO.load(model_path)

    print("\n=== 开始逐 K 线回测 ===")
    obs, info = env.reset()
    done = False
    
    # 用于记录回测数据的列表
    history_net_worth = []
    history_actions = []
    
    while not done:
        # 【极其重要】deterministic=True 意味着在推理时不加入随机探索噪音
        # 让模型输出它认为绝对最优的动作
        action, _states = model.predict(obs, deterministic=True)
        
        obs, reward, done, truncated, info = env.step(action)
        
        history_net_worth.append(info['net_worth'])
        action_val = int(action.item()) if hasattr(action, "item") else int(action)
        history_actions.append(action_val * 0.5)  # 记录实际的仓位比例

    # ==========================================
    # 5. 回测结果统计与可视化
    # ==========================================
    final_net_worth = info['net_worth']
    total_return = (final_net_worth - 50000) / 50000 * 100
    
    # 简单的买入持有策略 (Buy & Hold) 收益率对比
    bh_return = (test_prices[-1] - test_prices[0]) / test_prices[0] * 100
    
    print("\n=== 回测统计报告 ===")
    print(f"回测周期: 约 {len(features_df)} 根 15m K线 (约 {len(features_df)/96:.1f} 天)")
    print(f"期初资金: $50,000.00")
    print(f"期末资金: ${final_net_worth:,.2f}")
    print(f"策略总收益: {total_return:.2f}%")
    print(f"同期大盘 (持币不动) 收益: {bh_return:.2f}%")
    
    # 检查模型是否变成了“死鱼” (全天候输出固定的动作，比如永远空仓)
    unique_actions = np.unique(np.round(history_actions, 2))
    print(f"\n模型动作分布特征: {unique_actions[:5]} ... (如果只有一个值 0.0，说明模型怕交手续费，直接摆烂空仓了)")

    # 绘制资金曲线对比图
    print("\n正在生成回测图表...")
    fig, ax1 = plt.subplots(figsize=(12, 6))

    color = 'tab:red'
    ax1.set_xlabel('Steps (15m Candles)')
    ax1.set_ylabel('Strategy Net Worth ($)', color=color)
    ax1.plot(history_net_worth, color=color, label='PPO Strategy')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('BTC Price ($)', color=color)  
    ax2.plot(test_prices[env.window_size:], color=color, alpha=0.3, label='BTC Price (B&H)')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout() 
    plt.title("PPO Strategy Backtest vs Buy & Hold")
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    run_backtest()