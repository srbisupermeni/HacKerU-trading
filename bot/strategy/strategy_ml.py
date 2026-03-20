import sys
import logging
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from collections import Counter

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import roc_auc_score

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
from bot.execution.roostoo import Roostoo

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
    "meme": ["PEPE", "DOGE", "WIF"],
    "layer1": ["SOL", "APT", "SUI", "NEAR"],
    "ai": ["FET"],
    "btc": ["BTC"]
}

# 🟢 修复 4：Meme 板块严格降杠杆，最多只允许持有一个，拒绝相关性灾难！
GROUP_LIMITS = {
    "meme": 1,
    "layer1": 2,
    "ai": 1,
    "btc": 1
}

def get_group(coin_name, groups=COIN_GROUPS):
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
            with open(cache_file, "rb") as f: return pickle.load(f)
        except: pass
    fetcher = VisionFetcher()
    df = fetcher.fetch_klines_range(symbol=symbol, interval=interval,
        start_year=start_date.year, start_month=start_date.month, start_day=start_date.day,
        end_year=end_date.year, end_month=end_date.month, end_day=end_date.day, data_type="daily")
    if df is not None and not df.empty:
        with open(cache_file, "wb") as f: pickle.dump(df, f)
    return df

def normalize_kline_df(df: pd.DataFrame) -> pd.DataFrame:
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
        
        # 差分阈值：只接受极高确定性的异动
        coin_group = get_group(coin)
        if coin_group == "meme":
            self.conf_thresh = 1.20   
        else:
            self.conf_thresh = 0.85   
            
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
        except: return 1e-6

    def train_models_from_df(self, df_history: pd.DataFrame) -> bool:
        df_feat = self.engineer.generate_features(df_history)
        if df_feat is None or df_feat.empty: return False
        
        target_col = 'target_return_12'
        exclude_cols = ["open_time","close_time","close","open","high","low","volume",
                        "buy_volume","sell_volume",target_col,"target_class","sma_20","rsi_14","atr_14"]
        
        self.feature_cols = [c for c in df_feat.columns if c not in exclude_cols]
        mask = ~(df_feat[self.feature_cols].isna().any(axis=1) | df_feat["target_class"].isna())
        df_clean = df_feat.loc[mask].copy()
        
        split_idx = int(len(df_clean) * 0.8)
        purge_bars = 12 
        if split_idx + purge_bars >= len(df_clean): return False

        X_tr, y_tr = df_clean[self.feature_cols].iloc[:split_idx], df_clean["target_class"].iloc[:split_idx]
        X_val, y_val = df_clean[self.feature_cols].iloc[split_idx + purge_bars:], df_clean["target_class"].iloc[split_idx + purge_bars:]
        val_returns = df_clean[target_col].iloc[split_idx + purge_bars:].values

        lgb_params = {"objective": "binary", "metric": "auc", "learning_rate": 0.03, "max_depth": 6, "lambda_l2": 3.0, "verbose": -1}
        self.lgb_model = lgb.train(lgb_params, lgb.Dataset(X_tr, label=y_tr), num_boost_round=400,
                                   valid_sets=[lgb.Dataset(X_val, label=y_val)],
                                   callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])

        xgb_params = {"objective": "binary:logistic", "eval_metric": "auc", "max_depth": 5, "eta": 0.03, "lambda": 3.0}
        self.xgb_model = xgb.train(xgb_params, xgb.DMatrix(X_tr, label=y_tr), num_boost_round=400,
                                   evals=[(xgb.DMatrix(X_val, label=y_val), "v")], early_stopping_rounds=50, verbose_eval=False)

        p_lgb = self.lgb_model.predict(X_val, num_iteration=self.lgb_model.best_iteration)
        p_xgb = self.xgb_model.predict(xgb.DMatrix(X_val), iteration_range=(0, self.xgb_model.best_iteration + 1))

        s_lgb = self._get_trading_strength(y_val, p_lgb, val_returns)
        s_xgb = self._get_trading_strength(y_val, p_xgb, val_returns)
        self.weight_lgb, self.weight_xgb = s_lgb/(s_lgb+s_xgb), s_xgb/(s_lgb+s_xgb)

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
        p_x = self.xgb_model.predict(xgb.DMatrix(X), iteration_range=(0, self.xgb_model.best_iteration+1))[0]
        self.latest_raw_prob = float(p_l * self.weight_lgb + p_x * self.weight_xgb)
        self.latest_score = (self.latest_raw_prob - self.val_prob_mean) / self.val_prob_std
        return self.latest_score


# ============================================================
# 主程序：大师版回测循环
# ============================================================
if __name__ == "__main__":
    print("\n" + "=" * 80 + "\n🏆 极高胜率狙击版：动态超时期 + 动量衰竭止损\n" + "=" * 80)
    roostoo_client = Roostoo()
    portfolio = Portfolio(None); portfolio.account_balance = INITIAL_CAPITAL
    execution = ExecutionEngine(portfolio, roostoo_client); portfolio.execution = execution

    TARGET_COINS = [{"symbol": "BTCUSDT", "coin": "BTC"},{"symbol": "SOLUSDT", "coin": "SOL"},{"symbol": "DOGEUSDT", "coin": "DOGE"},
                    {"symbol": "PEPEUSDT", "coin": "PEPE"},{"symbol": "WIFUSDT", "coin": "WIF"},{"symbol": "SUIUSDT", "coin": "SUI"},
                    {"symbol": "APTUSDT", "coin": "APT"},{"symbol": "NEARUSDT", "coin": "NEAR"},{"symbol": "FETUSDT", "coin": "FET"}]

    strategies = {item["coin"]: DualMLStrategy(portfolio, execution, item["symbol"], item["coin"]) for item in TARGET_COINS}

    print(f"⏳ 并发拉取 {TOTAL_FETCH_DAYS} 天数据...")
    end_dt = datetime.today()
    start_dt = end_dt - timedelta(days=TOTAL_FETCH_DAYS)
    sim_data_raw = {}
    with ThreadPoolExecutor(max_workers=9) as exc:
        futures = {exc.submit(fetch_with_cache, s["symbol"], "5m", start_dt, end_dt): s["coin"] for s in TARGET_COINS}
        for f in futures:
            coin, df = futures[f], f.result()
            if df is not None: sim_data_raw[coin] = df

    sim_data = align_market_data_flexible(sim_data_raw)
    base_df = sim_data["BTC"]
    test_bars, train_window_bars = BACKTEST_DAYS * 288, TRAIN_DAYS * 288
    start_idx = len(base_df) - test_bars

    positions = {c: {"qty": 0.0, "entry_price": 0.0, "entry_bar": 0, "sl_pct": 0.0, "tp_pct": 0.0, "invested_cash": 0.0, "entry_score": 0.0} for c in strategies}
    trades_count, win_count, last_train_time = 0, 0, None
    trade_pnls = []

    for i in range(start_idx, len(base_df)):
        curr_time = base_df.iloc[i]["open_time"]
        
        # 每日滚动重训
        if last_train_time is None or (curr_time - last_train_time).total_seconds() >= 86400:
            print(f"\n🔄 [{curr_time}] 触发机制：正在融合最新盘感重训模型...")
            with ThreadPoolExecutor(max_workers=4) as exec:
                exec.map(lambda c: strategies[c].train_models_from_df(sim_data[c].iloc[i-train_window_bars:i]), strategies)
            last_train_time = curr_time
            print(f"✅ 模型矩阵升级完毕，恢复交易巡航。\n")

        btc_sma_now = base_df.iloc[i-288:i]["close"].mean()
        btc_sma_prev = base_df.iloc[i-300:i-12]["close"].mean()
        btc_multiplier = 1.0 if btc_sma_now > btc_sma_prev else 0.5

        predictions = []
        for coin, strat in strategies.items():
            df = sim_data[coin]
            if pd.isna(df.iloc[i]["open"]): continue 
            score = strat.predict_signal(df.iloc[i-300:i].copy())
            feat = strat.latest_features
            
            pos = positions[coin]
            coin_group = get_group(coin)
            
            # 平仓逻辑
            if pos["qty"] > 0:
                curr_row = df.iloc[i]
                sl_p, tp_p = pos["entry_price"]*(1-pos["sl_pct"]), pos["entry_price"]*(1+pos["tp_pct"])
                exit_price, reason = None, ""
                
                bars_held = i - pos["entry_bar"]
                entry_score = pos.get("entry_score", 0.0)
                
                # 🟢 修复 1：严格匹配 target_return_12 的模型视野
                if coin_group == "meme": max_bars = 12
                elif coin_group in ["layer1", "ai"]: max_bars = 18
                else: max_bars = 24
                
                if curr_row["open"] <= sl_p: exit_price, reason = curr_row["open"]*(1-SLIPPAGE), "🛑 跳空止损"
                elif curr_row["open"] >= tp_p: exit_price, reason = curr_row["open"]*(1-SLIPPAGE), "🎯 跳空止盈"
                elif curr_row["low"] <= sl_p: exit_price, reason = sl_p*(1-SLIPPAGE), "🛑 常规止损"
                elif curr_row["high"] >= tp_p: exit_price, reason = tp_p*(1-SLIPPAGE), "🎯 常规止盈"
                
                # 🟢 修复 2：真正的 Trailing Stop (动量衰减追踪)，注意使用 max()！
                elif bars_held >= 3 and score < max(-1.2, entry_score - 1.5): 
                    exit_price, reason = curr_row["open"]*(1-SLIPPAGE), "⚠️ AI动量衰竭强平"
                elif bars_held >= max_bars: 
                    exit_price, reason = curr_row["open"]*(1-SLIPPAGE), "⏰ 信号超时强平"
                
                if exit_price:
                    rev = pos["qty"] * exit_price * (1-FEE_RATE)
                    pnl = rev - pos["invested_cash"]
                    trade_pnls.append(pnl)
                    portfolio.account_balance += rev
                    trades_count += 1
                    if pnl > 0: win_count += 1
                    print(f"[{curr_time}] 卖出 [{coin}] {reason} | 价: {smart_price(exit_price)} | 赚: ${pnl:.2f} | 余额: ${portfolio.account_balance:.2f}")
                    positions[coin] = {"qty":0.0, "entry_price":0.0, "entry_bar":0, "sl_pct":0.0, "tp_pct":0.0, "invested_cash":0.0, "entry_score":0.0}
            else:
                atr_p = float(feat.get("atr_14", 0.0))
                sma_p = (not pd.isna(feat.get("sma_20"))) and (df.iloc[i-1]["close"] > feat.get("sma_20"))
                # 🟢 修复 8：移除 raw_prob > 0.5 的限制，全权交由 Z-Score 做纯粹的截面排序
                if score > strat.conf_thresh and feat.get("rsi_14", 50) < 68 and sma_p and atr_p >= MIN_ATR_PCT:
                    predictions.append({"coin":coin, "score":score, "price":df.iloc[i]["open"], "atr":atr_p})

        active_cnt = sum(1 for p in positions.values() if p["qty"] > 0)
        if active_cnt < MAX_POSITIONS and predictions:
            predictions.sort(key=lambda x: x["score"], reverse=True)
            group_counts = Counter(get_group(c, COIN_GROUPS) for c, p in positions.items() if p["qty"] > 0)
            
            current_gross_exposure = sum(p["qty"] * float(sim_data[c].iloc[i]["open"]) for c, p in positions.items() if p["qty"] > 0 and not pd.isna(sim_data[c].iloc[i]["open"]))
            total_equity = portfolio.account_balance + current_gross_exposure
            
            # 🟢 修复 5：单笔上限防爆体系 (绝不允许一单吃掉 15% 以上本金)
            per_trade_cap = total_equity * 0.15 
            
            target_exposure = total_equity * MAX_CAPITAL_USAGE
            remaining_budget = target_exposure - current_gross_exposure
            
            for target in predictions:
                if active_cnt >= MAX_POSITIONS or remaining_budget < MIN_ORDER_USD: break
                coin, g = target["coin"], get_group(target["coin"], COIN_GROUPS)
                if group_counts[g] >= GROUP_LIMITS.get(g, 1): continue
                
                # 计算基础可投资额，并施加 15% 的总权益硬上限
                base_invest = float((remaining_budget / (MAX_POSITIONS - active_cnt)) * btc_multiplier)
                invest = min(base_invest, per_trade_cap)
                
                if invest < MIN_ORDER_USD or portfolio.account_balance < invest: continue
                
                entry_p = float(target["price"] * (1 + SLIPPAGE))
                target_notional = invest / (1 + FEE_RATE)
                qty = float(smart_quantity(target_notional, entry_p))
                actual_cost = qty * entry_p * (1 + FEE_RATE)
                
                portfolio.account_balance -= actual_cost
                remaining_budget -= (qty * entry_p)
                
                positions[coin] = {"qty":qty, "entry_price":entry_p, "entry_bar":i, 
                                   "sl_pct":float(np.clip(target["atr"]*2, 0.012, 0.028)), 
                                   "tp_pct":float(np.clip(target["atr"]*3.5, 0.018, 0.045)),
                                   "invested_cash": actual_cost,
                                   "entry_score": float(target["score"])} 
                group_counts[g] += 1; active_cnt += 1
                print(f"[{curr_time}] 🟢 买入 [{coin}] ({g}) | Z: {target['score']:.2f} | 投入: ${actual_cost:.2f} | 价: {smart_price(entry_p)}")

    print("\n" + "-"*40 + "\n🏁 回测主循环结束，执行尾盘强制清算...\n" + "-"*40)
    for coin, pos in positions.items():
        if pos["qty"] > 0:
            last_price = float(sim_data[coin].iloc[-1]["close"])
            final_exit_price = last_price * (1 - SLIPPAGE)
            revenue = pos["qty"] * final_exit_price * (1 - FEE_RATE)
            pnl = revenue - pos["invested_cash"]
            trade_pnls.append(pnl)
            if pnl > 0: win_count += 1
            portfolio.account_balance += revenue
            trades_count += 1
            print(f"[END] 强制平仓 [{coin}] | 价: {smart_price(final_exit_price)} | 赚: ${pnl:.2f} | 余额: ${portfolio.account_balance:.2f}")

    roi = (portfolio.account_balance - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    avg_pnl = float(np.mean(trade_pnls)) if trade_pnls else 0.0
    
    print(f"\n" + "="*70)
    print(f"🏁 终极盲测战报 (MTM 结算版)")
    print(f"初始本金: ${INITIAL_CAPITAL:.2f}")
    print(f"总交易次数: {trades_count} 次")
    print(f"胜率: {win_count/trades_count:.1%}" if trades_count else "胜率: 0.0%")
    print(f"单笔平均 PnL: ${avg_pnl:.2f}")
    print(f"最终净值: ${portfolio.account_balance:.2f}")
    print(f"💰 综合 ROI: {roi:.2f}% (最大资金使用率 50%)")
    print("="*70)