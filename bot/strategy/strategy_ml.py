import pickle
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
import logging

import numpy as np
import pandas as pd
try:
    import lightgbm as lgb
except Exception:
    lgb = None
    # delay heavy import error until training is attempted

try:
    import xgboost as xgb
except Exception:
    xgb = None

# ============================================================
# 路径与外部模块导入
# ============================================================
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from database.Binance_Vision_fetcher import VisionFetcher
from bot.data.feature_engineering import FeatureEngineer
from bot.portfolio.portfolio import Portfolio
from bot.execution.execution_engine import ExecutionEngine

# ============================================================
# 全局参数区 (商赛实战配置)
# ============================================================
INITIAL_CAPITAL = 10_000.0
BACKTEST_DAYS = 20
TRAIN_DAYS = 60
TOTAL_FETCH_DAYS = BACKTEST_DAYS + TRAIN_DAYS + 2
BAR_INTERVAL = "5m"
FEE_RATE = 0.001
SLIPPAGE = 0.0005
MAX_POSITIONS = 3
MAX_CAPITAL_USAGE = 0.50
MIN_ORDER_USD = 10.0
BTC_TREND_LOOKBACK_BARS = 288
BTC_TREND_SHIFT_BARS = 12
MIN_ATR_PCT = 0.0015

COIN_GROUPS = {
    "meme": ["PEPE", "WIF"],
    "layer1": ["SOL", "APT", "SUI"],
    "ai": ["FET"],
    "btc": ["BTC"]
}

GROUP_LIMITS = {
    "meme": 1,
    "layer1": 2,
    "ai": 1,
    "btc": 0  # 彻底禁封 BTC
}

# 保留备用或兜底的板块 Edge Floor
EDGE_FLOOR = {
    "meme": 0.032,
    "layer1": 0.017,
    "ai": 0.019,
    "btc": 0.015,
}

# 🟢 改动 3：按币定阈值，实施最高精度的 Edge 拦截
EDGE_FLOOR_COIN = {
    "PEPE": 0.034,
    "WIF": 0.040,
    "SOL": 0.017,
    "SUI": 0.018,
    "APT": 0.019,
    "FET": 0.019,
    "BTC": 0.015
}


def get_group(coin_name, groups=None):
    if groups is None:
        groups = COIN_GROUPS
    for group, coins in groups.items():
        if coin_name in coins: return group
    return "other"


CACHE_DIR = ROOT / "data_cache"
CACHE_DIR.mkdir(exist_ok=True)


# ============================================================
# 数据工具函数
# ============================================================
def fetch_with_cache(symbol: str, interval: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    cache_key = f"{symbol}_{interval}_{start_date:%Y%m%d%H}_{end_date:%Y%m%d%H}"
    cache_file = CACHE_DIR / f"{cache_key}.pkl"
    if cache_file.exists():
        try:
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        except:
            pass
    fetcher = VisionFetcher()
    df = fetcher.fetch_klines_range(symbol=symbol, interval=interval,
                                    start_year=start_date.year, start_month=start_date.month, start_day=start_date.day,
                                    end_year=end_date.year, end_month=end_date.month, end_day=end_date.day,
                                    data_type="daily")
    if df is not None and not df.empty:
        with open(cache_file, "wb") as f: pickle.dump(df, f)
    return df


def normalize_kline_df(df: pd.DataFrame):
    if df is None or df.empty: return None
    out = df.copy()
    out["open_time"] = pd.to_datetime(out["open_time"])
    out = out.sort_values("open_time").drop_duplicates(subset=["open_time"])
    numeric_cols = ["open", "high", "low", "close", "volume", "buy_volume", "sell_volume"]
    for col in numeric_cols:
        if col in out.columns: out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.set_index("open_time", drop=False)
    out.index.name = "timestamp_idx"
    return out


def align_market_data_flexible(sim_data: dict) -> dict:
    if "BTC" not in sim_data: return {}
    btc_norm = normalize_kline_df(sim_data["BTC"])
    main_index = btc_norm.index
    aligned = {"BTC": btc_norm.reset_index(drop=True)}
    for coin, df in sim_data.items():
        if coin == "BTC": continue
        df_norm = normalize_kline_df(df)
        if df_norm is not None:
            aligned[coin] = df_norm.reindex(main_index).reset_index(drop=True)
    return aligned


def smart_quantity(invest_usd: float, price: float) -> float:
    if price <= 0: return 0.0
    qty = invest_usd / price
    if price < 0.0001: return round(qty, 0)
    if price < 0.01: return round(qty, 2)
    if price < 1: return round(qty, 4)
    return round(qty, 6)


def smart_price(price: float) -> float:
    if price < 0.0001: return round(price, 8)
    if price < 0.01: return round(price, 6)
    return round(price, 4)


# ============================================================
# 策略类：交易指标加权 + 严格数据隔离
# ============================================================
class DualMLStrategy:
    def __init__(self, portfolio: Portfolio, execution: ExecutionEngine,
                 symbol: str = "BTCUSDT", coin: str = "BTC", strategy_id: str = "ml_dual_01"):
        self.portfolio = portfolio
        self.execution = execution
        self.symbol = symbol
        self.coin = coin
        self.strategy_id = strategy_id

        coin_group = get_group(coin)

        # 🟢 改动 2：针对 WIF 单独收紧，精细化配置 Score 门槛
        if coin == "WIF":
            self.conf_thresh = 1.75
        elif coin == "PEPE":
            self.conf_thresh = 1.50
        elif coin_group == "meme":
            self.conf_thresh = 1.55
        else:
            self.conf_thresh = 1.00

        self.engineer = FeatureEngineer()
        self.lgb_model, self.xgb_model = None, None
        self.feature_cols = None
        self.is_trained = False
        self.weight_lgb, self.weight_xgb = 0.5, 0.5
        self.latest_features = {}
        self.val_prob_mean, self.val_prob_std = 0.5, 0.1
        self.latest_raw_prob, self.latest_score = 0.0, 0.0

    def _get_trading_strength(self, y_true, y_prob, returns):
        try:
            threshold = np.percentile(y_prob, 90)
            top_mask = y_prob >= threshold
            if not np.any(top_mask): return 1e-6
            avg_ret = np.mean(returns[top_mask])
            return max(avg_ret, 1e-6)
        except:
            return 1e-6

    def train_models_from_df(self, df_history: pd.DataFrame) -> bool:
        df_feat = self.engineer.generate_features(df_history)
        if df_feat is None or df_feat.empty: return False

        target_col = 'target_return_12'
        exclude_cols = ["open_time", "close_time", "close", "open", "high", "low", "volume",
                        "buy_volume", "sell_volume", target_col, "target_class", "sma_20", "rsi_14", "atr_14"]

        self.feature_cols = [c for c in df_feat.columns if c not in exclude_cols]
        mask = ~(df_feat[self.feature_cols].isna().any(axis=1) | df_feat["target_class"].isna())
        df_clean = df_feat.loc[mask].copy()

        split_idx = int(len(df_clean) * 0.8)
        purge_bars = 12
        if split_idx + purge_bars >= len(df_clean): return False

        X_tr, y_tr = df_clean[self.feature_cols].iloc[:split_idx], df_clean["target_class"].iloc[:split_idx]
        X_val, y_val = df_clean[self.feature_cols].iloc[split_idx + purge_bars:], df_clean["target_class"].iloc[
            split_idx + purge_bars:]
        val_returns = df_clean[target_col].iloc[split_idx + purge_bars:].values

        lgb_params = {"objective": "binary", "metric": "auc", "learning_rate": 0.03, "max_depth": 6, "lambda_l2": 3.0,
                      "verbose": -1}
        self.lgb_model = lgb.train(lgb_params, lgb.Dataset(X_tr, label=y_tr), num_boost_round=400,
                                   valid_sets=[lgb.Dataset(X_val, label=y_val)],
                                   callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])

        xgb_params = {"objective": "binary:logistic", "eval_metric": "auc", "max_depth": 5, "eta": 0.03, "lambda": 3.0}
        self.xgb_model = xgb.train(xgb_params, xgb.DMatrix(X_tr, label=y_tr), num_boost_round=400,
                                   evals=[(xgb.DMatrix(X_val, label=y_val), "v")], early_stopping_rounds=50,
                                   verbose_eval=False)

        p_lgb = self.lgb_model.predict(X_val, num_iteration=self.lgb_model.best_iteration)
        p_xgb = self.xgb_model.predict(xgb.DMatrix(X_val), iteration_range=(0, self.xgb_model.best_iteration + 1))

        s_lgb = self._get_trading_strength(y_val, p_lgb, val_returns)
        s_xgb = self._get_trading_strength(y_val, p_xgb, val_returns)
        self.weight_lgb, self.weight_xgb = s_lgb / (s_lgb + s_xgb), s_xgb / (s_lgb + s_xgb)

        ensemble_v = p_lgb * self.weight_lgb + p_xgb * self.weight_xgb
        self.val_prob_mean, self.val_prob_std = float(np.mean(ensemble_v)), float(np.std(ensemble_v) + 1e-9)
        self.is_trained = True
        return True

    def predict_signal(self, recent_df: pd.DataFrame) -> float:
        if not self.is_trained or recent_df.empty: return 0.0
        feat_df = self.engineer.generate_features(recent_df)
        if feat_df is None or feat_df.empty: return 0.0
        self.latest_features = feat_df.iloc[-1].to_dict()
        X = feat_df.iloc[-1:][self.feature_cols]
        if X.isna().any(axis=1).iloc[0]: return 0.0
        p_l = self.lgb_model.predict(X, num_iteration=self.lgb_model.best_iteration)[0]
        p_x = self.xgb_model.predict(xgb.DMatrix(X), iteration_range=(0, self.xgb_model.best_iteration + 1))[0]
        self.latest_raw_prob = float(p_l * self.weight_lgb + p_x * self.weight_xgb)
        self.latest_score = (self.latest_raw_prob - self.val_prob_mean) / self.val_prob_std
        return self.latest_score


# ============================================================
# 🟢 新增：实盘管理器 (Live Manager) 供 main.py 调用
# ============================================================
class DualMLLiveManager:
    """
    封装了原先散落在 __main__ 中的截面风控、仓位管理和止盈止损逻辑。
    用于在实盘主循环中每根 K 线调用一次 on_tick()。
    """

    def __init__(self, portfolio, execution, target_coins=None):
        self.portfolio = portfolio
        self.execution = execution

        if target_coins is None:
            target_coins = [
                {"symbol": "BTCUSDT", "coin": "BTC"}, {"symbol": "SOLUSDT", "coin": "SOL"},
                {"symbol": "PEPEUSDT", "coin": "PEPE"}, {"symbol": "WIFUSDT", "coin": "WIF"},
                {"symbol": "SUIUSDT", "coin": "SUI"}, {"symbol": "APTUSDT", "coin": "APT"},
                {"symbol": "FETUSDT", "coin": "FET"}
            ]

        self.strategies = {item["coin"]: DualMLStrategy(portfolio, execution, item["symbol"], item["coin"]) for item in
                           target_coins}

        # 🟢 修复3：增加 invested_cash 字段用于准确计算包含手续费的 PnL
        self.positions_state = {
            c: {"qty": 0.0, "entry_price": 0.0, "entry_bar": 0, "sl_pct": 0.0, "tp_pct": 0.0, "invested_cash": 0.0} for
            c in self.strategies}
        self.cooldowns = {c: 0 for c in self.strategies}
        self.consecutive_losses = {c: 0 for c in self.strategies}

        self.last_train_time = None
        self.current_bar_index = 0
        self.last_candle_time = None
        # persistent timestamp to enforce minimum time between actual order executions
        self.last_execution_time = None
        # safe logger for the live manager
        try:
            self.logger = logging.getLogger("DualMLLiveManager")
            self.logger.setLevel(logging.INFO)
            if not self.logger.handlers:
                ch = logging.StreamHandler()
                ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
                self.logger.addHandler(ch)
        except Exception:
            # fallback to simple print-based logger
            class _Dl:
                def info(self, *a, **k):
                    try: print(*a)
                    except: pass
                def warning(self, *a, **k):
                    try: print('WARNING:', *a)
                    except: pass
                def exception(self, *a, **k):
                    try: print('EXCEPTION:', *a)
                    except: pass
            self.logger = _Dl()

    def on_tick(self, sim_data: dict, current_time: datetime, total_equity: float):
        # use self.last_execution_time (persistent across ticks) to enforce throttling
        """主程序每获取到最新 K 线数据时调用此方法"""
        # wrap the whole on_tick in a single try/except to prevent exceptions from bubbling up
        try:
            base_df = sim_data.get("BTC")
            if base_df is None or base_df.empty:
                return

            latest_time = base_df.iloc[-1]["open_time"]
            if self.last_candle_time != latest_time:
                self.current_bar_index += 1
                self.last_candle_time = latest_time
            else:
                return

            # 1. 模型重训触发机制
            if self.last_train_time is None or (current_time - self.last_train_time).total_seconds() >= 86400:
                try:
                    self.logger.info(f"🔄 [{current_time}] 触发机制：正在融合最新盘感重训模型...")
                    for coin in self.strategies:
                        try:
                            train_window_bars = TRAIN_DAYS * 288
                            df_train = sim_data[coin].iloc[-(train_window_bars + 1): -1]
                            self.strategies[coin].train_models_from_df(df_train)
                        except Exception:
                            self.logger.exception(f"Training failed for {coin}")
                    self.last_train_time = current_time
                    self.logger.info("✅ 模型矩阵升级完毕，恢复交易巡航。")
                except Exception:
                    self.logger.exception("Failed during training cycle")

            btc_sma_now = base_df.iloc[-288:]["close"].mean()
            btc_sma_prev = base_df.iloc[-300:-12]["close"].mean()
            btc_multiplier = 1.0 if btc_sma_now > btc_sma_prev else 0.5

            predictions = []

            for coin, strat in self.strategies.items():
                try:
                    df = sim_data.get(coin)
                    if df is None or df.empty or pd.isna(df.iloc[-1]["open"]):
                        continue

                    curr_row = df.iloc[-1]
                    score = strat.predict_signal(df.iloc[-300:].copy())
                    feat = strat.latest_features
                    pos = self.positions_state[coin]
                    coin_group = get_group(coin)
                except Exception:
                    self.logger.exception(f"Error preparing data/prediction for {coin}")
                    continue

                # --- 平仓逻辑 ---
                if pos["qty"] > 0:
                    sl_p = pos["entry_price"] * (1 - pos["sl_pct"])
                    tp_p = pos["entry_price"] * (1 + pos["tp_pct"])
                    exit_price, reason = None, ""
                    bars_held = self.current_bar_index - pos["entry_bar"]

                    if curr_row["open"] <= sl_p:
                        exit_price, reason = curr_row["open"] * (1 - SLIPPAGE), "🛑 跳空止损"
                    elif curr_row["open"] >= tp_p:
                        exit_price, reason = curr_row["open"] * (1 - SLIPPAGE), "🎯 跳空止盈"
                    elif curr_row["low"] <= sl_p:
                        exit_price, reason = sl_p * (1 - SLIPPAGE), "🛑 常规止损"
                    elif curr_row["high"] >= tp_p:
                        exit_price, reason = tp_p * (1 - SLIPPAGE), "🎯 常规止盈"
                    elif bars_held >= 6 and score < -0.5:
                        exit_price, reason = curr_row["open"] * (1 - SLIPPAGE), "⚠️ AI动量衰竭强平"
                    elif bars_held >= 12:
                        exit_price, reason = curr_row["open"] * (1 - SLIPPAGE), "⏰ 严格12根超时平仓"

                    if exit_price:
                        try:
                            order = self.portfolio.create_order(coin, "SELL", pos["qty"], order_type="MARKET",
                                                                strategy_id=strat.strategy_id)
                            if self.last_execution_time is None:
                                res = self.portfolio.execute_order(order)
                                self.last_execution_time = datetime.now()
                            else:
                                duration = (datetime.now() - self.last_execution_time).total_seconds()
                                if duration >= 60:
                                    res = self.portfolio.execute_order(order)
                                    self.last_execution_time = datetime.now()
                                else:
                                    time.sleep(max(0, int(60 - duration)))
                                    res = self.portfolio.execute_order(order)
                                    self.last_execution_time = datetime.now()

                            if res.get("success"):
                                rev = pos["qty"] * exit_price * (1 - FEE_RATE)
                                pnl = rev - pos["invested_cash"]

                                if pnl > 0:
                                    self.consecutive_losses[coin] = 0
                                    self.cooldowns[coin] = self.current_bar_index + 6
                                else:
                                    self.consecutive_losses[coin] += 1
                                    if self.consecutive_losses[coin] >= 2:
                                        self.cooldowns[coin] = self.current_bar_index + (36 if coin_group == "meme" else 24)
                                        self.consecutive_losses[coin] = 0
                                    else:
                                        self.cooldowns[coin] = self.current_bar_index + (12 if coin_group == "meme" else 6)

                                self.positions_state[coin] = {"qty": 0.0, "entry_price": 0.0, "entry_bar": 0, "sl_pct": 0.0,
                                                              "tp_pct": 0.0, "invested_cash": 0.0}
                                try:
                                    self.logger.info(f"[{current_time}] 卖出 [{coin}] {reason} | 净赚: ${pnl:.2f}")
                                except Exception:
                                    pass
                        except Exception:
                            self.logger.exception(f"Failed to execute exit order for {coin}")

                # --- 开仓漏斗 ---
                else:
                    if self.current_bar_index < self.cooldowns.get(coin, 0):
                        continue
                    if coin == "BTC":
                        continue

                    if coin_group == "meme":
                        if btc_sma_now <= btc_sma_prev:
                            continue
                        if float(feat.get("volume_intensity", 0.0)) < 1.15:
                            continue

                    if score <= strat.conf_thresh:
                        continue
                    if feat.get("rsi_14", 50) >= 68:
                        continue

                    sma_p = (not pd.isna(feat.get("sma_20"))) and (df.iloc[-2]["close"] > feat.get("sma_20"))
                    if not sma_p:
                        continue

                    atr_p = float(feat.get("atr_14", 0.0))
                    if atr_p < MIN_ATR_PCT:
                        continue

                    tp_pct = float(np.clip(atr_p * 2.8, 0.015, 0.040))
                    sl_pct = float(np.clip(atr_p * 2.0, 0.012, 0.028))
                    expected_edge = (score * tp_pct) - (2 * FEE_RATE + 2 * SLIPPAGE) - (sl_pct / max(score, 0.1))

                    floor_threshold = EDGE_FLOOR_COIN.get(coin, EDGE_FLOOR.get(coin_group, 0.015))
                    if expected_edge <= floor_threshold:
                        continue

                    predictions.append({
                        "coin": coin, "score": score, "price": curr_row["open"],
                        "tp_pct": tp_pct, "sl_pct": sl_pct, "expected_edge": expected_edge
                    })

            # 执行资金分配与买入
            active_cnt = sum(1 for p in self.positions_state.values() if p["qty"] > 0)
            if active_cnt < MAX_POSITIONS and predictions:
                predictions.sort(key=lambda x: x["expected_edge"], reverse=True)
                group_counts = Counter(get_group(c) for c, p in self.positions_state.items() if p["qty"] > 0)

                target_exposure = total_equity * MAX_CAPITAL_USAGE
                current_gross = sum(
                    p["qty"] * float(sim_data[c].iloc[-1]["open"]) for c, p in self.positions_state.items() if p["qty"] > 0)
                remaining_budget = target_exposure - current_gross

                for target in predictions:
                    if active_cnt >= MAX_POSITIONS or remaining_budget < MIN_ORDER_USD:
                        break
                    coin, g = target["coin"], get_group(target["coin"])
                    if group_counts[g] >= GROUP_LIMITS.get(g, 1):
                        continue

                    invest = min((remaining_budget / (MAX_POSITIONS - active_cnt)) * btc_multiplier, total_equity * 0.15)
                    invest = min(invest, self.portfolio.account_balance)

                    if invest < MIN_ORDER_USD:
                        continue

                    entry_p = float(target["price"] * (1 + SLIPPAGE))
                    qty = float(smart_quantity(invest / (1 + FEE_RATE), entry_p))

                    try:
                        order = self.portfolio.create_order(coin, "BUY", qty, order_type="MARKET",
                                                            strategy_id=self.strategies[coin].strategy_id)
                        if self.last_execution_time is None:
                            res = self.portfolio.execute_order(order)
                            self.last_execution_time = datetime.now()
                        else:
                            duration = (datetime.now() - self.last_execution_time).total_seconds()
                            if duration >= 60:
                                res = self.portfolio.execute_order(order)
                                self.last_execution_time = datetime.now()
                            else:
                                time.sleep(max(0, int(60 - duration)))
                                res = self.portfolio.execute_order(order)
                                self.last_execution_time = datetime.now()

                        if res.get("success"):
                            actual_cost = qty * entry_p * (1 + FEE_RATE)
                            self.positions_state[coin] = {
                                "qty": qty, "entry_price": entry_p, "entry_bar": self.current_bar_index,
                                "sl_pct": target["sl_pct"], "tp_pct": target["tp_pct"],
                                "invested_cash": actual_cost
                            }
                            group_counts[g] += 1
                            active_cnt += 1
                            remaining_budget -= actual_cost
                            try:
                                self.logger.info(f"[{current_time}] 🟢 买入 [{coin}] | Edge: {target['expected_edge']:.4f}")
                            except Exception:
                                pass
                    except Exception:
                        self.logger.exception(f"Failed to execute buy order for {coin}")
        except Exception:
            try:
                self.logger.exception("Unhandled exception in DualMLLiveManager.on_tick")
            except Exception:
                pass
