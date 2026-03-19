import os
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback

# 导入你写好的模块
from database.Binance_fetcher import BinanceDataFetcher
from data.feature_engineering import FeatureEngineer
from rl_env import CryptoSpotEnv

def prepare_data(symbol="BTCUSDT", interval="15m", limit=10000):
    """拉取数据并划分为训练集和验证集"""
    print(f"1. 正在从 Binance 获取 {limit} 根 {interval} K线数据...")
    fetcher = BinanceDataFetcher()
    # 为了让模型有足够的数据学习，我们拉取 10000 根线 (约 100 天的 15 分钟数据)
    raw_df = fetcher.fetch_recent_klines(symbol=symbol, interval=interval, limit=limit)
    
    print("2. 执行特征工程...")
    engineer = FeatureEngineer()
    features_df = engineer.generate_features(raw_df)
    
    # 剔除可能包含 NaN 的行 (通常在滚动窗口初期产生)
    features_df.dropna(inplace=True)
    features_df.reset_index(drop=True, inplace=True)
    
    # 按照时间序列 80% 训练，20% 验证
    split_idx = int(len(features_df) * 0.8)
    train_df = features_df.iloc[:split_idx]
    eval_df = features_df.iloc[split_idx:]
    
    print(f"数据准备完毕: 训练集 {len(train_df)} 行, 验证集 {len(eval_df)} 行.")
    return train_df, eval_df

def main():
    # 创建保存模型和日志的目录
    os.makedirs("./models", exist_ok=True)
    os.makedirs("./tensorboard_logs", exist_ok=True)

    # 1. 准备数据
    train_df, eval_df = prepare_data(symbol="BTCUSDT", interval="15m", limit=10000)

    # 2. 创建训练环境和验证环境
    # Stable-Baselines3 要求环境必须用 VecEnv 包装
    train_env = DummyVecEnv([lambda: CryptoSpotEnv(df=train_df, window_size=12)])
    eval_env = DummyVecEnv([lambda: CryptoSpotEnv(df=eval_df, window_size=12)])

    # 3. 配置 EvalCallback (在训练时定期评估，保存最优模型)
    eval_callback = EvalCallback(
        eval_env, 
        best_model_save_path='./models/best_model',
        log_path='./models/eval_logs', 
        eval_freq=5000,          # 每训练 5000 步评估一次
        deterministic=True,      # 评估时使用确定性动作 (不带探索噪音)
        render=False
    )

    # 4. 初始化 PPO 模型
    print("\n3. 初始化 PPO 模型...")
    # 这里我们调整了几个超参数来适应量化金融的非平稳噪声特性
    model = PPO(
        "MlpPolicy",               # 多层感知机策略网络
        train_env, 
        learning_rate=3e-4,        # 较小的学习率，防止在噪声中剧烈震荡
        n_steps=2048,              # 每次更新前收集的步数
        batch_size=64,             # 批次大小
        n_epochs=10,               # 每次收集数据后优化网络的轮数
        gamma=0.99,                # 折扣因子 (更看重长期收益)
        ent_coef=0.01,             # 熵系数 (稍微大一点，鼓励模型多尝试不同的仓位，避免陷入全空仓的局部最优)
        clip_range=0.2,            # PPO 核心裁剪比例
        tensorboard_log="./tensorboard_logs/",
        verbose=1
    )

    # 5. 开始训练
    print("\n4. 开始训练 (这可能需要几分钟到十几分钟的时间)...")
    TOTAL_TIMESTEPS = 100_000  # 训练总步数，可以根据电脑算力调整
    
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS, 
        callback=eval_callback,
        tb_log_name="PPO_BTC_15m"
    )

    # 6. 保存最终模型 (即使它可能不是验证集上最好的)
    model.save("./models/ppo_crypto_final")
    print("\n训练结束！最优模型已保存在 ./models/best_model/best_model.zip")

if __name__ == "__main__":
    main()