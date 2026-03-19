import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd


class CryptoSpotEnv(gym.Env):
    """
    针对现货交易优化的加密货币环境 (仅做多/平仓)
    修复版本:
      1. 修复 done 变量作用域 Bug
      2. 动作空间扩展为 11 档 (0%~100%，每10%一档)
      3. 奖励函数改为风险调整收益，防止模型退化为"永远满仓"
      4. 修复 FOMO 惩罚阈值与动作空间不匹配的问题
    """

    # ── 动作档位：11 个离散值对应 0%, 10%, 20%, ..., 100% 仓位 ──
    N_ACTIONS = 11

    def __init__(self, df, window_size=12, initial_balance=50000.0, fee_rate=0.001):
        super(CryptoSpotEnv, self).__init__()

        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.fee_rate = fee_rate

        cols_to_drop = ['open_time', 'open', 'high', 'low', 'close',
                        'target_return_6', 'target_class']
        cols_to_drop = [c for c in cols_to_drop if c in self.df.columns]

        self.features = self.df.drop(columns=cols_to_drop).values
        self.prices = self.df['close'].values

        # ── 修复 Bug 2：动作空间改为 11 档，覆盖 0%~100% 每 10% 一档 ──
        self.action_space = spaces.Discrete(self.N_ACTIONS)

        self.obs_shape = (self.window_size, self.features.shape[1])
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=self.obs_shape, dtype=np.float32
        )

        # 内部状态
        self.current_step = 0
        self.cash_balance = 0.0
        self.crypto_position = 0.0
        self.net_worth = 0.0
        self.max_net_worth = 0.0

        # 用于计算滚动 Sharpe 的收益率缓存
        self._recent_returns = []

    # ──────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.cash_balance = self.initial_balance
        self.crypto_position = 0.0
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self._recent_returns = []
        return self._get_observation(), {}

    # ──────────────────────────────────────────────
    def _get_observation(self):
        obs = self.features[self.current_step - self.window_size: self.current_step]
        return np.array(obs, dtype=np.float32)

    # ──────────────────────────────────────────────
    def step(self, action):
        current_price = self.prices[self.current_step]

        # 统一提取整数动作
        action_val = int(action.item()) if hasattr(action, "item") else int(action)

        # ── 修复 Bug 2：target_weight 范围 [0, 1.0]，步进 0.1 ──
        target_weight = action_val / (self.N_ACTIONS - 1)   # 0.0 ~ 1.0

        # 计算目标持币价值和数量
        target_crypto_value = self.net_worth * target_weight
        target_crypto_qty = target_crypto_value / current_price

        # 计算需要交易的数量（买入为正，卖出为负）
        trade_qty = target_crypto_qty - self.crypto_position

        # 扣除现货交易手续费
        trade_cost = abs(trade_qty) * current_price * self.fee_rate

        # 更新账户状态
        self.crypto_position = target_crypto_qty
        self.cash_balance = self.net_worth - target_crypto_value - trade_cost

        # 时间步前进，获取新价格
        self.current_step += 1
        next_price = self.prices[self.current_step]

        # 计算新的总净值
        old_net_worth = self.net_worth
        self.net_worth = self.cash_balance + (self.crypto_position * next_price)
        self.max_net_worth = max(self.max_net_worth, self.net_worth)

        # ── 奖励函数：超额收益 + 决策力激励 ─────────────────────────
        step_return = (self.net_worth - old_net_worth) / old_net_worth
        market_return = (next_price - current_price) / current_price
        current_position_ratio = (self.crypto_position * current_price) / max(self.net_worth, 1e-8)

        # 缓存最近 N 步收益率，用于计算波动率
        self._recent_returns.append(step_return)
        if len(self._recent_returns) > 48:
            self._recent_returns.pop(0)

        vol = np.std(self._recent_returns) + 1e-8 if len(self._recent_returns) >= 4 else 1e-4

        # 1. 核心奖励：相对大盘的超额收益 / 波动率
        #    持有 X% 仓位时，"应得"的市场收益是 market_return * X%
        #    超额部分才是模型真正贡献的 alpha
        benchmark_return = market_return * current_position_ratio
        excess_return = step_return - benchmark_return
        reward = excess_return / vol

        # 2. 决策力激励：鼓励模型做出明确的仓位决策（靠近 0 或 1）
        #    惩罚永远待在 0.5 的"和稀泥"行为
        #    abs(0.5 - 0.5) = 0  →  无奖励
        #    abs(1.0 - 0.5) = 0.5 → 奖励 0.05
        #    abs(0.0 - 0.5) = 0.5 → 奖励 0.05
        decisiveness_bonus = abs(target_weight - 0.5) * 0.1
        reward += decisiveness_bonus

        # 3. FOMO / 止损 惩罚（阈值与11档动作空间对齐）
        #    市场涨超 0.1% 但仓位低于 15% → 错失惩罚
        if market_return > 0.001 and current_position_ratio < 0.15:
            reward -= market_return * 0.3
        #    市场跌超 0.1% 但仓位高于 85% → 止损惩罚
        if market_return < -0.001 and current_position_ratio > 0.85:
            reward += market_return * 0.3   # market_return 为负，等于减分

        # ── 修复 Bug 1：done 变量在所有分支前初始化 ──────────────
        done = False

        # 回撤惩罚与终止条件
        drawdown = (self.max_net_worth - self.net_worth) / self.max_net_worth
        if drawdown > 0.15:
            reward -= 1.0
            done = True
        else:
            done = self.current_step >= len(self.df) - 1

        obs = self._get_observation()
        info = {
            'net_worth': self.net_worth,
            'cash': self.cash_balance,
            'crypto_qty': self.crypto_position,
            'price': next_price,
            'position_ratio': current_position_ratio,
            'drawdown': drawdown,
        }

        return obs, reward, done, False, info


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    from pathlib import Path

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
    raw_df = fetcher.fetch_recent_klines(symbol="BTCUSDT", interval="15m", limit=500)

    if raw_df is None or raw_df.empty:
        print("错误：未能获取到原始数据。")
        sys.exit(1)

    print(f"2. 开始特征工程... (原始数据行数: {len(raw_df)})")
    engineer = FeatureEngineer()
    features_df = engineer.generate_features(raw_df)
    print(f"   特征生成完毕。 (可用数据行数: {len(features_df)})")

    print("\n3. 初始化现货交易环境...")
    env = CryptoSpotEnv(df=features_df, window_size=12, initial_balance=50000.0, fee_rate=0.0005)

    obs, info = env.reset()
    done = False
    step_count = 0

    print("4. 开始随机动作测试 (Random Action Agent)...")
    while not done:
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        step_count += 1

        if step_count % 50 == 0:
            print(
                f"Step: {step_count:3d} | 动作: {action} ({action/(env.N_ACTIONS-1)*100:.0f}%) "
                f"| 净值: ${info['net_worth']:,.2f} | 仓位: {info['position_ratio']:.1%} "
                f"| 回撤: {info['drawdown']:.2%}"
            )

    print("\n=== 环境测试完成 ===")
    print(f"总运行步数: {step_count}")
    print(f"期初净值: $50,000.00")
    print(f"期末净值: ${info['net_worth']:,.2f}")
    net_return = (info['net_worth'] - 50000) / 50000 * 100
    print(f"随机策略总收益率: {net_return:.2f}%")