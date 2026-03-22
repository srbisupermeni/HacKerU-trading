"""
=============================================================================
Binance 数据获取工具 (Binance_fetcher.py)
结合 python-binance SDK 与 REST 直连获取最新 K 线数据。
适用于量化策略回测与实盘信号生成。

修复说明：
1. 不在 __init__ 中直接初始化 Client，避免 AWS 上启动时因默认 ping 导致崩溃
2. SDK 初始化改为懒加载，关闭默认 ping，并设置 timeout
3. SDK 请求失败时，自动回退到 REST 直连
4. 日志输出做防护，避免 handler 的 I/O 异常反过来导致程序中断
=============================================================================
"""

import logging
import os
from typing import Optional

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from binance.client import Client


def _get_safe_logger(name):
    """Return a logger wrapper that never raises due to logging handler errors."""
    lg = logging.getLogger(name)

    class _SafeLogger:
        def __init__(self, underlying):
            self._underlying = underlying

        def info(self, msg, *args, **kwargs):
            try:
                self._underlying.info(msg, *args, **kwargs)
            except Exception:
                try:
                    print(str(msg))
                except Exception:
                    pass

        def warning(self, msg, *args, **kwargs):
            try:
                self._underlying.warning(msg, *args, **kwargs)
            except Exception:
                try:
                    print("WARNING:", str(msg))
                except Exception:
                    pass

        def exception(self, msg, *args, **kwargs):
            try:
                self._underlying.exception(msg, *args, **kwargs)
            except Exception:
                try:
                    print("EXCEPTION:", str(msg))
                except Exception:
                    pass

    return _SafeLogger(lg)


def _wrap_provided_logger(logger):
    class _ProvidedSafeLogger:
        def __init__(self, underlying):
            self._underlying = underlying

        def info(self, msg, *args, **kwargs):
            try:
                self._underlying.info(msg, *args, **kwargs)
            except Exception:
                try:
                    print(str(msg))
                except Exception:
                    pass

        def warning(self, msg, *args, **kwargs):
            try:
                self._underlying.warning(msg, *args, **kwargs)
            except Exception:
                try:
                    print("WARNING:", str(msg))
                except Exception:
                    pass

        def exception(self, msg, *args, **kwargs):
            try:
                self._underlying.exception(msg, *args, **kwargs)
            except Exception:
                try:
                    print("EXCEPTION:", str(msg))
                except Exception:
                    pass

    return _ProvidedSafeLogger(logger)


class BinanceDataFetcher:
    def __init__(self, logger=None):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        current_path = base_dir

        while True:
            if os.path.exists(os.path.join(current_path, "requirements.txt")):
                root_dir = current_path
                break
            parent_path = os.path.dirname(current_path)
            if parent_path == current_path:
                root_dir = base_dir
                break
            current_path = parent_path

        self.raw_save_folder = os.path.join(root_dir, "database", "raw_data")
        self.processed_save_folder = os.path.join(root_dir, "database", "processed_data")

        if logger is None:
            self.logger = _get_safe_logger(__name__)
        else:
            try:
                self.logger = _wrap_provided_logger(logger)
            except Exception:
                self.logger = _get_safe_logger(__name__)

        # 不在 __init__ 中直接联网初始化，避免 AWS 上启动即失败
        self.client: Optional[Client] = None

        # REST 备用通道：即使 SDK 初始化失败，也能直接拉公开市场数据
        self.base_url = "https://api.binance.com"
        self.session = requests.Session()
        retry = Retry(
            total=3,
            connect=3,
            read=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=frozenset(["GET"]),
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def _safe_close_client(self):
        if self.client is None:
            return
        try:
            self.client.close_connection()
        except Exception:
            pass
        finally:
            self.client = None

    def _ensure_client(self):
        """Lazily create Binance SDK client without default ping."""
        if self.client is not None:
            return True

        try:
            # 注意：python-binance 用 requests_params 传 timeout，不是 timeout=10
            self.client = Client(
                requests_params={"timeout": 10},
                ping=False,
            )
            return True
        except Exception:
            self.logger.exception("Failed to initialize Binance SDK Client")
            self.client = None
            return False

    def show_data(self, df, num_rows=5):
        logger = self.logger
        if df is not None and not df.empty:
            logger.info(f"{num_rows} of head:\n{df.head(num_rows)}")
            logger.info(f"{num_rows} of tail:\n{df.tail(num_rows)}")
        else:
            logger.info("empty df or None")

    def _add_technical_indicators(self, df):
        logger = self.logger
        try:
            df['return_rate'] = df['close'].pct_change()
            df['log_return'] = np.log(df['close'] / df['close'].shift(1))
            df['volatility_20'] = df['return_rate'].rolling(window=20).std()
            df['momentum_10'] = df['close'] - df['close'].shift(10)
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()

            std_20 = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['sma_20'] + 2 * std_20
            df['bb_lower'] = df['sma_20'] - 2 * std_20

            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi_14'] = 100 - (100 / (1 + rs))

            ema_12 = df['close'].ewm(span=12, adjust=False).mean()
            ema_26 = df['close'].ewm(span=26, adjust=False).mean()
            df['macd'] = ema_12 - ema_26
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()

            return df
        except Exception:
            try:
                logger.exception("Failed to compute technical indicators; returning original df")
            except Exception:
                try:
                    print("Failed to compute technical indicators; returning original df")
                except Exception:
                    pass
            return df

    def _fetch_recent_klines_via_sdk(self, symbol="BTCUSDT", interval="1m", limit=100):
        if not self._ensure_client():
            return None

        try:
            return self.client.get_klines(
                symbol=symbol.upper(),
                interval=interval,
                limit=limit,
                requests_params={"timeout": 10},
            )
        except Exception:
            self.logger.exception("Exception while calling Binance SDK get_klines")
            self._safe_close_client()
            return None

    def _fetch_recent_klines_via_rest(self, symbol="BTCUSDT", interval="1m", limit=100):
        url = f"{self.base_url}/api/v3/klines"
        params = {
            "symbol": symbol.upper(),
            "interval": interval,
            "limit": limit,
        }

        try:
            resp = self.session.get(url, params=params, timeout=(5, 15))
            resp.raise_for_status()
            return resp.json()
        except Exception:
            self.logger.exception("Exception while calling Binance REST /api/v3/klines")
            return None

    def _parse_klines_to_df(self, raw_klines):
        logger = self.logger
        if not raw_klines:
            logger.warning("No klines returned")
            return None

        columns = [
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ]

        try:
            first = raw_klines[0]
        except Exception:
            logger.exception("Unexpected klines format (not indexable)")
            return None

        try:
            if isinstance(first, dict):
                df = pd.DataFrame(raw_klines)
                mapping = {
                    'openTime': 'open_time',
                    'open': 'open',
                    'high': 'high',
                    'low': 'low',
                    'close': 'close',
                    'volume': 'volume',
                    'closeTime': 'close_time',
                    'quoteAssetVolume': 'quote_asset_volume',
                    'numberOfTrades': 'number_of_trades',
                    'takerBuyBaseAssetVolume': 'taker_buy_base_asset_volume',
                    'takerBuyQuoteAssetVolume': 'taker_buy_quote_asset_volume'
                }
                df.rename(columns=mapping, inplace=True)
            else:
                df = pd.DataFrame(raw_klines, columns=columns)

            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
            df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')

            # 保持你原来的时区习惯：UTC -> UTC+8
            df['open_time'] = df['open_time'] + pd.Timedelta(hours=8)
            df['close_time'] = df['close_time'] + pd.Timedelta(hours=8)

            numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'taker_buy_base_asset_volume']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                else:
                    df[col] = np.nan

            df['buy_volume'] = df.get('taker_buy_base_asset_volume')
            df['sell_volume'] = df['volume'] - df['buy_volume']

            desired = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'buy_volume', 'sell_volume']
            for c in desired:
                if c not in df.columns:
                    df[c] = np.nan

            return df[desired]
        except Exception:
            logger.exception("Failed to parse klines into DataFrame")
            return None

    # ==========================================
    # 模块一：获取最新/实时的热数据
    # 优先使用 SDK，失败后自动回退到 REST
    # ==========================================
    def fetch_recent_klines(self, symbol="BTCUSDT", interval="1m", limit=100):
        logger = self.logger
        logger.info(f"Fetching recent {limit} candles for {symbol} ({interval})...")

        raw_klines = self._fetch_recent_klines_via_sdk(
            symbol=symbol,
            interval=interval,
            limit=limit,
        )

        if raw_klines is None:
            logger.warning("SDK failed, fallback to REST")
            raw_klines = self._fetch_recent_klines_via_rest(
                symbol=symbol,
                interval=interval,
                limit=limit,
            )

        return self._parse_klines_to_df(raw_klines)
