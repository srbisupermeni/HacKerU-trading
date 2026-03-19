import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class CryptoSpotEnv(gym.Env):
    """
    针对现货交易优化的加密货币环境 (仅做多/平仓)
    """
    def __init__(self, df, window_size=12, initial_balance=50000.0, fee_rate=0.0005):
        super(CryptoSpotEnv, self).__init__()
        
        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.fee_rate = fee_rate 
        
        cols_to_drop = ['open_time', 'open', 'high', 'low', 'close', 'target_return_6', 'target_class']
        cols_to_drop = [c for c in cols_to_drop if c in self.df.columns]
        
        self.features = self.df.drop(columns=cols_to_drop).values
        self.prices = self.df['close'].values
        
        # 动作空间限制为 [0.0, 1.0]，代表现货仓位比例
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        
        self.obs_shape = (self.window_size, self.features.shape[1])
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.obs_shape, dtype=np.float32)
        #内部持仓，多为正
        self.current_step = 0
        self.cash_balance = 0.0    # 现金 (如 USDT)
        self.crypto_position = 0.0 # 持币数量 (如 BTC)
        self.net_worth = 0.0       # 总净值 (现金 + 币值)
        self.max_net_worth = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.cash_balance = self.initial_balance
        self.crypto_position = 0.0
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        
        return self._get_observation(), {}

    def _get_observation(self):
        obs = self.features[self.current_step - self.window_size : self.current_step]
        return np.array(obs, dtype=np.float32)

    def step(self, action):
        current_price = self.prices[self.current_step]
        
        # 限制动作在 0 到 1 之间 (0=全现金, 1=全仓币)
        target_weight = np.clip(action[0], 0.0, 1.0)
        
        # 计算目标持币价值和数量
        target_crypto_value = self.net_worth * target_weight
        target_crypto_qty = target_crypto_value / current_price
        
        # 计算需要交易的数量 (买入为正，卖出为负)
        trade_qty = target_crypto_qty - self.crypto_position
        
        # 扣除现货交易手续费
        trade_cost = np.abs(trade_qty) * current_price * self.fee_rate
        
        # 更新账户状态 (现货逻辑：动用现金买币，或卖币换回现金)
        self.crypto_position = target_crypto_qty
        self.cash_balance = self.net_worth - target_crypto_value - trade_cost
        
        # 时间步前进，获取新价格
        self.current_step += 1
        next_price = self.prices[self.current_step]
        
        # 计算新的总净值 (最新现金 + 最新持币市值)
        old_net_worth = self.net_worth
        self.net_worth = self.cash_balance + (self.crypto_position * next_price)
        self.max_net_worth = max(self.max_net_worth, self.net_worth)
        
        # 奖励计算 (单步收益率)
        step_return = (self.net_worth - old_net_worth) / old_net_worth
        reward = step_return
        
        # 回撤惩罚与终止条件
        drawdown = (self.max_net_worth - self.net_worth) / self.max_net_worth
        if drawdown > 0.15: 
            reward -= 1.0 
            done = True
        else:
            done = self.current_step >= len(self.df) - 1
            
        obs = self._get_observation()
        info = {'net_worth': self.net_worth, 'cash': self.cash_balance, 'crypto_qty': self.crypto_position, 'price': next_price}
        
        return obs, reward, done, False, info
    
if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # 确保能正常导入你项目里的其他模块
    ROOT = Path(__file__).resolve().parent.parent.parent 
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))
        
    try:
        from database.Binance_fetcher import BinanceDataFetcher
        from bot.data.feature_engineering import FeatureEngineer
    except ImportError as e:
        print(f"导入依赖失败，请检查路径: {e}")
        sys.exit(1)

    print("1. 初始化数据获取器并拉取最新数据...")
    fetcher = BinanceDataFetcher()
    # 拉取 500 根 15 分钟线用于测试
    raw_df = fetcher.fetch_recent_klines(symbol="BTCUSDT", interval="15m", limit=500)
    
    if raw_df is None or raw_df.empty:
        print("错误：未能获取到原始数据。")
        sys.exit(1)

    print(f"2. 开始特征工程... (原始数据行数: {len(raw_df)})")
    engineer = FeatureEngineer()
    features_df = engineer.generate_features(raw_df)
    print(f"   特征生成完毕。 (可用数据行数: {len(features_df)})")

    print("\n3. 初始化现货交易环境...")
    # 实例化我们刚写的现货环境
    env = CryptoSpotEnv(df=features_df, window_size=12, initial_balance=50000.0, fee_rate=0.0005)
    
    obs, info = env.reset()
    done = False
    step_count = 0

    print("4. 开始随机动作测试 (Random Action Agent)...")
    while not done:
        # 随机生成一个 [0.0, 1.0] 之间的动作 (现货仓位比例)
        action = env.action_space.sample() 
        
        obs, reward, done, truncated, info = env.step(action)
        step_count += 1
        
        # 每 50 步打印一次状态，防止刷屏
        if step_count % 50 == 0:
            print(f"Step: {step_count:3d} | 动作(仓位): {action[0]:.2f} | 净值: ${info['net_worth']:,.2f} | 现金: ${info['cash']:,.2f} | 币价: ${info['price']:.2f}")

    print("\n=== 环境测试完成 ===")
    print(f"总运行步数: {step_count}")
    print(f"期初净值: $50,000.00")
    print(f"期末净值: ${info['net_worth']:,.2f}")
    
    net_return = (info['net_worth'] - 50000) / 50000 * 100
    print(f"随机策略总收益率: {net_return:.2f}% (如果亏损严重是正常的，因为在随机交手续费)")